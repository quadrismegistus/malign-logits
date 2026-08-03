"""
metrics.py — Canonical metric computations on DeepDive data.

All functions take numpy arrays as input (from DeepDive parquet reads)
and return scalars or small arrays. No model loading, no forward passes.

Usage:
    from malign_logits.deep_probe import DeepDive
    from malign_logits.metrics import *

    dive = DeepDive("olmo")
    base = dive.logits("base", "anger", gen=0, pos=0)
    dpo  = dive.logits("dpo",  "anger", gen=0, pos=0)
    embed = dive.embedding_matrix("base")

    # Distribution comparison
    js_divergence(base, dpo)               # → float
    entropy(base)                          # → float
    top_k_overlap(base, dpo, k=50)         # → float
    rank_correlation(base, dpo)            # → float
    effective_vocab(dpo, threshold=0.001)   # → int

    # Axis projections
    v_axis, p_axis = violence_procedural_axes(embed, tokenizer)
    violence_loading(dpo, embed, v_axis)   # → float
    procedural_loading(dpo, embed, p_axis) # → float

    # Cross-layer (base token under aligned distribution)
    base_token_surprisal(base, dpo)        # → float

    # Temporal (across positions)
    narrowing_rate(entropies)              # → float
    tail_redistribution(base, dpo, k=100)  # → dict

    # Layer-wise (hidden states)
    layer_cka(h_base, h_aligned)           # → float
"""

import numpy as np
from scipy.special import softmax as _softmax
from scipy.stats import spearmanr


# =============================================================================
# TIER 1: Logits only
# =============================================================================

def _align_vocab(a: np.ndarray, b: np.ndarray):
    """Align two logit vectors with different vocab sizes by truncating to the
    shared prefix (min length).

    Matches analysis._align_logits (truncate-to-min) so the torch and numpy
    metric families agree — they were provably identical on equal-length inputs
    (the only case that occurs for published numbers, since cross-family work
    uses word-probability dicts, not raw-logit JS) but diverged on unequal
    length. Truncate also avoids the pad-with-−1e10 pathology that corrupted
    rank_correlation / top_k_overlap (a block of identical sentinel values).
    """
    if len(a) == len(b):
        return a, b
    n = min(len(a), len(b))
    return a[:n], b[:n]


def entropy(logits: np.ndarray) -> float:
    """Shannon entropy of softmax(logits) in nats."""
    p = _softmax(logits)
    p = np.clip(p, 1e-10, None)
    return -float(np.sum(p * np.log(p)))


def kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """KL(P || Q) in nats. How much Q diverges from P."""
    logits_p, logits_q = _align_vocab(logits_p, logits_q)
    p = np.clip(_softmax(logits_p), 1e-10, None)
    q = np.clip(_softmax(logits_q), 1e-10, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. Symmetric, bounded [0, ln(2)]."""
    logits_a, logits_b = _align_vocab(logits_a, logits_b)
    p = np.clip(_softmax(logits_a), 1e-10, None)
    q = np.clip(_softmax(logits_b), 1e-10, None)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def top_k_overlap(logits_a: np.ndarray, logits_b: np.ndarray,
                  k: int = 50) -> float:
    """Fraction of top-k tokens shared. 1.0 = identical, 0.0 = disjoint."""
    logits_a, logits_b = _align_vocab(logits_a, logits_b)
    top_a = set(np.argsort(logits_a)[-k:])
    top_b = set(np.argsort(logits_b)[-k:])
    return len(top_a & top_b) / k


def rank_correlation(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Spearman rank correlation of logit orderings."""
    logits_a, logits_b = _align_vocab(logits_a, logits_b)
    return float(spearmanr(logits_a, logits_b).statistic)


def effective_vocab(logits: np.ndarray, threshold: float = 0.001) -> int:
    """Count of tokens with probability above threshold."""
    return int(np.sum(_softmax(logits) > threshold))


def top_movers(logits_a: np.ndarray, logits_b: np.ndarray,
               k: int = 20) -> dict:
    """Tokens with largest probability shift between two distributions.

    Returns dict with 'repressed' (lost mass) and 'amplified' (gained mass),
    each a list of (token_id, delta) sorted by |delta|.
    """
    logits_a, logits_b = _align_vocab(logits_a, logits_b)
    p = _softmax(logits_a)
    q = _softmax(logits_b)
    delta = q - p
    order = np.argsort(delta)
    repressed = [(int(i), float(delta[i])) for i in order[:k]]
    amplified = [(int(i), float(delta[i])) for i in order[-k:][::-1]]
    return {"repressed": repressed, "amplified": amplified}


def base_token_surprisal(base_logits: np.ndarray,
                         aligned_logits: np.ndarray) -> float:
    """Surprisal of the base model's argmax under the aligned distribution, in bits.

    -log2(p_aligned(argmax_base)). Higher = more resistance.
    Compare with self-surprisal: -log2(p_base(argmax_base)).
    Delta = "bits of resistance" — the barrier height, mechanism-neutral.
    """
    base_logits, aligned_logits = _align_vocab(base_logits, aligned_logits)
    base_argmax = int(np.argmax(base_logits))
    aligned_probs = np.clip(_softmax(aligned_logits), 1e-10, None)
    return -float(np.log2(aligned_probs[base_argmax]))


def base_top_k_mass(base_logits: np.ndarray, aligned_logits: np.ndarray,
                    k: int = 10) -> float:
    """How much probability mass the aligned model assigns to the base's top-k.

    1.0 = aligned preserves base's top predictions.
    0.0 = aligned completely displaces them.
    """
    base_logits, aligned_logits = _align_vocab(base_logits, aligned_logits)
    top_base = np.argsort(base_logits)[-k:]
    aligned_probs = _softmax(aligned_logits)
    return float(aligned_probs[top_base].sum())


def tail_redistribution(base_logits: np.ndarray, aligned_logits: np.ndarray,
                        k: int = 100) -> dict:
    """How alignment redistributes probability mass from top-k to tail.

    Returns:
        head_mass_base: prob mass in top-k under base
        head_mass_aligned: prob mass in top-k under aligned
        tail_gain: how much mass moved to the tail (head_base - head_aligned)
        tail_entropy_base: entropy of the tail (tokens outside top-k)
        tail_entropy_aligned: entropy of the tail under aligned
        redistribution_type: "thin" (mass concentrates elsewhere in head)
                            or "spread" (mass disperses into tail)
    """
    base_logits, aligned_logits = _align_vocab(base_logits, aligned_logits)
    p_base = _softmax(base_logits)
    p_aligned = _softmax(aligned_logits)

    top_k_ids = np.argsort(base_logits)[-k:]
    top_k_mask = np.zeros(len(base_logits), dtype=bool)
    top_k_mask[top_k_ids] = True

    head_base = float(p_base[top_k_mask].sum())
    head_aligned = float(p_aligned[top_k_mask].sum())

    # Tail entropy
    tail_base = p_base[~top_k_mask]
    tail_aligned = p_aligned[~top_k_mask]
    tail_base_normed = np.clip(tail_base / tail_base.sum(), 1e-10, None)
    tail_aligned_normed = np.clip(tail_aligned / tail_aligned.sum(), 1e-10, None)
    tail_ent_base = -float(np.sum(tail_base_normed * np.log(tail_base_normed)))
    tail_ent_aligned = -float(np.sum(tail_aligned_normed * np.log(tail_aligned_normed)))

    tail_gain = head_base - head_aligned
    rtype = "spread" if tail_ent_aligned > tail_ent_base else "thin"

    return {
        "head_mass_base": head_base,
        "head_mass_aligned": head_aligned,
        "tail_gain": tail_gain,
        "tail_entropy_base": tail_ent_base,
        "tail_entropy_aligned": tail_ent_aligned,
        "redistribution_type": rtype,
    }


def narrowing_rate(entropies: np.ndarray) -> float:
    """Linear slope of entropy across autoregressive positions.

    Negative = distribution narrows over generation (reaction formation).
    Near zero = stable. Positive = distribution opens up.
    """
    n = len(entropies)
    if n < 2:
        return 0.0
    return float(np.polyfit(np.arange(n), entropies, 1)[0])


def narrowing_ratio(entropies: np.ndarray) -> float:
    """H(last) / H(first). The temporal signature shape.

    < 1.0 = narrowing (model gets more certain)
    > 1.0 = broadening (rare)
    ≈ 1.0 = stable or already foreclosed
    """
    if len(entropies) < 2 or entropies[0] < 1e-10:
        return 1.0
    return float(entropies[-1] / entropies[0])


# =============================================================================
# TIER 2: Logits + embedding matrix
# =============================================================================

def violence_procedural_axes(embed: np.ndarray, tokenizer) -> tuple:
    """Compute violence and procedural axes from embedding matrix.

    Returns (violence_axis, procedural_axis) as unit numpy vectors.
    """
    from .profile import (VIOLENCE_POS, VIOLENCE_NEG,
                          PROCEDURAL_POS, PROCEDURAL_NEG)

    def phrase_vec(phrase):
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        return embed[ids].mean(axis=0)

    v_pos = np.mean([phrase_vec(p) for p in VIOLENCE_POS], axis=0)
    v_neg = np.mean([phrase_vec(p) for p in VIOLENCE_NEG], axis=0)
    violence = v_pos - v_neg
    violence = violence / np.linalg.norm(violence)

    p_pos = np.mean([phrase_vec(p) for p in PROCEDURAL_POS], axis=0)
    p_neg = np.mean([phrase_vec(p) for p in PROCEDURAL_NEG], axis=0)
    procedural = p_pos - p_neg
    procedural = procedural / np.linalg.norm(procedural)

    return violence, procedural


def axis_trajectory(probe, prompt: str, embed: np.ndarray,
                    axis: np.ndarray, axis_name: str = "axis",
                    gen: int = 0) -> 'pd.DataFrame':
    """Track axis loading at every position through a generation.

    Returns DataFrame with one row per position, columns:
        position, output_loading (from logits), hidden_loading (from last-layer
        hidden state), chosen_token.

    Within-model metric — no path dependency confound. Shows where in
    semantic space the generation lives over time.
    """
    import pandas as pd

    meta = probe.meta(prompt, gen=gen)
    token_loadings = embed @ axis

    rows = []
    for i, (_, mrow) in enumerate(meta.iterrows()):
        pos = mrow["position"]
        row = {"position": pos, "chosen_token": mrow["chosen_token"]}

        try:
            logits = probe.logits(prompt, gen=gen, pos=pos)
            probs = _softmax(logits)
            n = min(len(probs), len(token_loadings))
            row[f"output_{axis_name}"] = float(np.sum(probs[:n] * token_loadings[:n]))
        except (FileNotFoundError, ValueError):
            pass

        try:
            h = probe.hidden(prompt, gen=gen, pos=pos, layer=-1)
            row[f"hidden_{axis_name}"] = float(np.dot(h, axis))
        except (FileNotFoundError, ValueError):
            pass

        rows.append(row)

    return pd.DataFrame(rows)


def centroid_distance(probe_base, probe_aligned, prompt: str,
                      gen: int = 0, n_positions: int = 50,
                      layer: int = -1) -> dict:
    """Distance between generation cloud centroids in hidden space.

    Computes mean hidden state across positions for each model, then
    measures how far apart the two centroids are. Also projects onto
    violence/procedural axes to show WHERE in semantic space each
    model's generation cloud lives.

    Three patterns observed:
    - Cloud separation (OLMo): alignment moves entire cloud
    - Same cloud, different tokens (Llama): same region, different samples
    - Architectural convergence (Falcon): mid-network divergence absorbed
    """
    base_vecs, aligned_vecs = [], []
    for pos in range(n_positions):
        try:
            hb = probe_base.hidden(prompt, gen=gen, pos=pos, layer=layer)
            ha = probe_aligned.hidden(prompt, gen=gen, pos=pos, layer=layer)
            base_vecs.append(hb)
            aligned_vecs.append(ha)
        except (FileNotFoundError, ValueError):
            break

    if len(base_vecs) < 2:
        return {}

    base_vecs = np.stack(base_vecs)
    aligned_vecs = np.stack(aligned_vecs)

    bc = base_vecs.mean(axis=0)
    ac = aligned_vecs.mean(axis=0)

    nb, na = np.linalg.norm(bc), np.linalg.norm(ac)
    cos_dist = 1 - float(np.dot(bc, ac) / (nb * na)) if nb > 1e-10 and na > 1e-10 else 1.0

    base_spread = float(np.mean([np.linalg.norm(v - bc) for v in base_vecs]))
    aligned_spread = float(np.mean([np.linalg.norm(v - ac) for v in aligned_vecs]))

    return {
        "centroid_cos_dist": cos_dist,
        "centroid_l2_dist": float(np.linalg.norm(bc - ac)),
        "base_centroid": bc,
        "aligned_centroid": ac,
        "base_spread": base_spread,
        "aligned_spread": aligned_spread,
        "n_positions": len(base_vecs),
    }


def axis_loading(logits: np.ndarray, embed: np.ndarray,
                 axis: np.ndarray) -> float:
    """Expected projection of distribution onto an axis.

    E_p[embed_i . axis] = sum(p_i * (embed_i . axis))

    This is the core measurement: how much the probability distribution
    "points toward" violence, procedural compliance, etc.
    """
    probs = _softmax(logits)
    token_loadings = embed @ axis  # (vocab_size,)
    return float(np.sum(probs * token_loadings))


def word_probabilities(logits: np.ndarray, tokenizer,
                       words: list) -> dict:
    """Probability of specific words under a logit distribution.

    Returns {word: probability} dict.
    """
    probs = _softmax(logits)
    result = {}
    for word in words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            result[word] = float(probs[ids[0]])
    return result


def formation(base_logits: np.ndarray, *aligned_logits_list,
              tokenizer, k: int = 50) -> dict:
    """Track how top-k token probabilities change across alignment layers.

    Returns dict mapping token_text → list of probabilities
    (one per layer: [base, sft, dpo, ...]).
    """
    base_probs = _softmax(base_logits)
    top_ids = np.argsort(base_probs)[-k:]

    all_probs = [base_probs] + [_softmax(l) for l in aligned_logits_list]
    result = {}
    for tid in top_ids:
        try:
            word = tokenizer.decode([tid]).strip()
        except Exception:
            word = f"<{tid}>"
        result[word] = [float(p[tid]) for p in all_probs]
    return result


# =============================================================================
# TIER 3: Hidden states
# =============================================================================

def linear_cka(h_a: np.ndarray, h_b: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two hidden state matrices.

    Inputs are (n_layers, hidden_dim) — one column per layer.
    Measures representational similarity: 1.0 = identical geometry, 0.0 = orthogonal.

    For single vectors (hidden_dim,): reshape to (1, hidden_dim).
    """
    if h_a.ndim == 1:
        h_a = h_a.reshape(1, -1)
    if h_b.ndim == 1:
        h_b = h_b.reshape(1, -1)

    # Center
    h_a = h_a - h_a.mean(axis=0)
    h_b = h_b - h_b.mean(axis=0)

    hsic_ab = np.linalg.norm(h_a.T @ h_b, 'fro') ** 2
    hsic_aa = np.linalg.norm(h_a.T @ h_a, 'fro') ** 2
    hsic_bb = np.linalg.norm(h_b.T @ h_b, 'fro') ** 2

    denom = np.sqrt(hsic_aa * hsic_bb)
    if denom < 1e-10:
        return 0.0
    return float(hsic_ab / denom)


def logit_lens_at_layer(hidden_state: np.ndarray,
                        lm_head_weight: np.ndarray) -> np.ndarray:
    """Project a hidden state through the unembedding matrix.

    Returns logits (vocab_size,) — apply softmax for probabilities.
    Note: this skips the final layer norm. For exact logit lens,
    store normed hidden states or apply norm separately.
    """
    return hidden_state @ lm_head_weight.T


def hidden_state_drift(h_pos0: np.ndarray, h_posN: np.ndarray) -> float:
    """Cosine distance between hidden states at two positions."""
    norm_a = np.linalg.norm(h_pos0)
    norm_b = np.linalg.norm(h_posN)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0
    return 1.0 - float(np.dot(h_pos0, h_posN) / (norm_a * norm_b))


def hidden_distance(h_base: np.ndarray, h_aligned: np.ndarray) -> float:
    """Mean cosine distance between two models' hidden states across all layers.

    Inputs: (n_layers, hidden_dim) from Probe.hidden() at the same prompt/pos.
    Returns a single number: 0 = identical representations, 1 = orthogonal.

    This measures how far alignment has moved the model's internal state —
    the representational cost of alignment at a given prompt and position.
    """
    n = h_base.shape[0]
    dists = []
    for i in range(n):
        b, a = h_base[i], h_aligned[i]
        nb, na = np.linalg.norm(b), np.linalg.norm(a)
        if nb < 1e-10 or na < 1e-10:
            dists.append(1.0)
        else:
            dists.append(1.0 - float(np.dot(b, a) / (nb * na)))
    return float(np.mean(dists))


def hidden_divergence_trajectory(probe_base, probe_aligned, prompt: str,
                                 gen: int = 0) -> dict:
    """How hidden distance between base and aligned evolves across positions.

    At position 0 (teacher-forced, same input), divergence is purely
    from weight changes. At later positions, the models generate different
    tokens so divergence compounds. The RATE of compounding tells you
    whether alignment's hidden-space effect grows, stabilizes, or shrinks
    during generation.

    Returns dict with:
        positions: list of position indices
        distances: per-position mean hidden distance (across layers)
        distance_last_layer: per-position distance at final layer only
        growth_rate: linear slope of distance over positions
    """
    meta = probe_base.meta(prompt, gen=gen)
    positions = sorted(meta["position"].unique())

    dists_mean = []
    dists_last = []
    for pos in positions:
        try:
            hb = probe_base.hidden(prompt, gen=gen, pos=pos)
            ha = probe_aligned.hidden(prompt, gen=gen, pos=pos)
            dists_mean.append(hidden_distance(hb, ha))
            # Last layer only
            b, a = hb[-1], ha[-1]
            nb, na = np.linalg.norm(b), np.linalg.norm(a)
            dists_last.append(1.0 - float(np.dot(b, a) / (nb * na))
                              if nb > 1e-10 and na > 1e-10 else 1.0)
        except (FileNotFoundError, ValueError):
            break

    positions = positions[:len(dists_mean)]
    growth = float(np.polyfit(np.arange(len(dists_mean)), dists_mean, 1)[0]) if len(dists_mean) > 1 else 0.0

    return {
        "positions": positions,
        "distances": dists_mean,
        "distance_last_layer": dists_last,
        "growth_rate": growth,
        "mean_distance": float(np.mean(dists_mean)),
        "pos0_distance": dists_mean[0] if dists_mean else 0.0,
        "final_distance": dists_mean[-1] if dists_mean else 0.0,
    }


def hidden_distance_by_prompt(probe_base, probe_aligned,
                              prompts: list = None,
                              gen: int = 0, pos: int = 0) -> dict:
    """Is hidden distance content-dependent?

    Computes hidden_distance for each prompt, returns per-prompt
    distances + variance. Low variance = alignment is content-blind
    in hidden space (matching the 1.7% content variance in logit space).
    """
    from .probe import PROMPTS
    prompts = prompts or list(PROMPTS.keys())

    distances = {}
    for prompt in prompts:
        try:
            hb = probe_base.hidden(prompt, gen=gen, pos=pos)
            ha = probe_aligned.hidden(prompt, gen=gen, pos=pos)
            distances[prompt] = hidden_distance(hb, ha)
        except (FileNotFoundError, ValueError):
            pass

    vals = list(distances.values())
    return {
        "per_prompt": distances,
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std": float(np.std(vals)) if vals else 0.0,
        "cv": float(np.std(vals) / np.mean(vals)) if vals and np.mean(vals) > 1e-10 else 0.0,
    }


def formation_trajectory(probe, prompts_dict: dict = None,
                          checkpoints: list = None,
                          prompt: str = "anger",
                          gen: int = 0, pos: int = 0,
                          k: int = 30) -> 'pd.DataFrame':
    """Track how top-k token probabilities evolve through base→SFT→DPO.

    Returns DataFrame with one row per token, columns for probability
    at each checkpoint. Shows which tokens gain/lose across the pipeline.
    """
    import pandas as pd
    from scipy.special import softmax

    if checkpoints is None:
        from .registry import Registry
        reg = Registry()
        base_id = reg.base_of(probe.model_id)
        checkpoints = [base_id] + reg.variants_of(base_id)

    tok = probe.tokenizer
    all_probs = {}

    for cp in checkpoints:
        try:
            from .probe import Probe
            logits = Probe(cp).logits(prompt, gen=gen, pos=pos)
            all_probs[cp.split("/")[-1]] = softmax(logits)
        except (FileNotFoundError, ValueError):
            pass

    if not all_probs:
        return pd.DataFrame()

    # Get top-k from first checkpoint (base)
    first_key = list(all_probs.keys())[0]
    first_probs = all_probs[first_key]
    top_ids = np.argsort(first_probs)[-k:][::-1]

    rows = []
    for tid in top_ids:
        try:
            word = tok.decode([tid]).strip()
        except Exception:
            word = f"<{tid}>"
        row = {"token": word, "token_id": int(tid)}
        for cp_name, probs in all_probs.items():
            if tid < len(probs):
                row[cp_name] = float(probs[tid])
            else:
                row[cp_name] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def internal_drift(dive, checkpoint: str, prompt: str,
                   gen: int = 0, layer: int = -1) -> dict:
    """Drift in the model's own hidden-state space across generation.

    Measures how the model's internal representation moves through
    the generation, at a specific layer. No external embedder needed.

    Args:
        dive: DeepDive instance
        checkpoint: e.g. "base", "dpo"
        gen: generation index
        layer: transformer layer (-1 = last)

    Returns dict with:
        step_dists: cosine distances between consecutive positions
        total_drift: cosine distance from first to last position
        path_length: sum of step distances
        directedness: total_drift / path_length (1.0 = straight line)
        mean_step: mean step distance
    """
    all_hidden = dive.hidden(checkpoint, prompt, gen=gen)
    n_layers = all_hidden.shape[0] // len(
        set(int(x) for x in dive.meta(checkpoint, prompt, gen=gen)["position"]))
    if layer == -1:
        layer = n_layers - 1

    meta = dive.meta(checkpoint, prompt, gen=gen)
    positions = sorted(meta["position"].unique())

    vecs = []
    for pos in positions:
        h = dive.hidden(checkpoint, prompt, gen=gen, pos=pos, layer=layer)
        vecs.append(h)
    vecs = np.stack(vecs)

    def _cos_dist(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    step_dists = [_cos_dist(vecs[i], vecs[i + 1])
                  for i in range(len(vecs) - 1)]
    total_drift = _cos_dist(vecs[0], vecs[-1])
    path_length = sum(step_dists)
    directedness = total_drift / path_length if path_length > 1e-10 else 1.0

    return {
        "step_dists": step_dists,
        "total_drift": total_drift,
        "path_length": path_length,
        "directedness": directedness,
        "mean_step": float(np.mean(step_dists)) if step_dists else 0.0,
    }


def logit_drift(dive, checkpoint: str, prompt: str,
                gen: int = 0) -> dict:
    """Drift in logit/distribution space across generation.

    Like internal_drift but uses the output distribution (logits)
    instead of hidden states. Measures how the model's predictions
    wander. No external model, no hidden states needed — T1.

    Uses JS divergence between consecutive positions as the distance.
    """
    meta = dive.meta(checkpoint, prompt, gen=gen)
    positions = sorted(meta["position"].unique())

    logit_vecs = [dive.logits(checkpoint, prompt, gen=gen, pos=pos)
                  for pos in positions]

    step_dists = [js_divergence(logit_vecs[i], logit_vecs[i + 1])
                  for i in range(len(logit_vecs) - 1)]
    total_drift = js_divergence(logit_vecs[0], logit_vecs[-1])
    path_length = sum(step_dists)
    directedness = total_drift / path_length if path_length > 1e-10 else 1.0

    return {
        "step_dists": step_dists,
        "total_drift": total_drift,
        "path_length": path_length,
        "directedness": directedness,
        "mean_step": float(np.mean(step_dists)) if step_dists else 0.0,
    }


# =============================================================================
# Generation-level distance: comparing two full generations
# =============================================================================

def tree_metrics(probe, prompt: str, n_gens: int = 100,
                 max_tokens: int = 10) -> dict:
    """Tree-level metrics from n generations.

    Returns branch distribution, entropy, and per-branch statistics.
    """
    from collections import Counter

    empirical = Counter()
    for gen in range(n_gens):
        try:
            meta = probe.meta(prompt, gen=gen, max_tokens=max_tokens)
            tid = int(meta.iloc[0]["chosen_token_id"])
            empirical[tid] += 1
        except (FileNotFoundError, ValueError, KeyError):
            break

    n = sum(empirical.values())
    if n == 0:
        return {}

    probs = np.array([c / n for c in empirical.values()])
    branch_entropy = -float(np.sum(probs * np.log2(probs + 1e-10)))

    tok = probe.tokenizer
    branches = {tok.decode([tid]).strip(): count / n
                for tid, count in empirical.most_common()}

    return {
        "n_gens": n,
        "n_branches": len(empirical),
        "branch_entropy": branch_entropy,
        #: CORRECT, AND IT READS LIKE THE DEFECT AT contrast.py:463 ([3802]).
        #: The ordering is real but IMPLICIT: `branches` is built from
        #: `empirical.most_common()`, so insertion order IS descending count and
        #: index 0 is the top. It survives only because dicts preserve insertion
        #: order -- rebuild `branches` from anything else and these two lines go
        #: silently wrong with no other edit. Read them WITH the comprehension
        #: six lines up, never alone.
        "top_branch": list(branches.keys())[0] if branches else "",
        "top_branch_pct": list(branches.values())[0] if branches else 0,
        "branches": branches,
        "token_ids": dict(empirical),
    }


def tree_compare(probe_a, probe_b, prompt: str,
                 n_gens: int = 100, max_tokens: int = 10) -> dict:
    """Compare tree structures between two models.

    Returns branch survival (per-branch repression), tree JS,
    novel branches, and pruned branches.
    """
    ta = tree_metrics(probe_a, prompt, n_gens, max_tokens)
    tb = tree_metrics(probe_b, prompt, n_gens, max_tokens)

    if not ta or not tb:
        return {}

    branches_a = ta["branches"]
    branches_b = tb["branches"]
    all_branches = set(branches_a) | set(branches_b)

    # Branch survival: P(branch in B) / P(branch in A)
    survival = {}
    for branch in all_branches:
        pa = branches_a.get(branch, 0)
        pb = branches_b.get(branch, 0)
        if pa > 0:
            survival[branch] = pb / pa
        elif pb > 0:
            survival[branch] = float("inf")  # novel in B

    # Tree JS: JS between branch distributions
    tokens_a = ta["token_ids"]
    tokens_b = tb["token_ids"]
    all_tids = set(tokens_a) | set(tokens_b)
    n_a = ta["n_gens"]
    n_b = tb["n_gens"]
    p = np.array([tokens_a.get(t, 0) / n_a for t in all_tids])
    q = np.array([tokens_b.get(t, 0) / n_b for t in all_tids])
    p = np.clip(p / p.sum(), 1e-10, None)
    q = np.clip(q / q.sum(), 1e-10, None)
    m = 0.5 * (p + q)
    tree_js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    # Novel and pruned branches
    novel = {b: branches_b[b] for b in branches_b if b not in branches_a}
    pruned = {b: branches_a[b] for b in branches_a if b not in branches_b}

    # Repressed (survival < 0.5) and amplified (survival > 2)
    repressed = {b: s for b, s in survival.items()
                 if s < 0.5 and branches_a.get(b, 0) > 0.02}
    amplified = {b: s for b, s in survival.items()
                 if s > 2.0 and branches_b.get(b, 0) > 0.02}

    return {
        "tree_js": float(tree_js),
        "branch_entropy_a": ta["branch_entropy"],
        "branch_entropy_b": tb["branch_entropy"],
        "n_branches_a": ta["n_branches"],
        "n_branches_b": tb["n_branches"],
        "n_novel": len(novel),
        "n_pruned": len(pruned),
        "novel": novel,
        "pruned": pruned,
        "repressed": repressed,
        "amplified": amplified,
        "survival": survival,
    }


def branch_trajectory(probe, prompt: str, branch_token: str,
                      n_gens: int = 100, max_tokens: int = 10) -> dict:
    """Entropy trajectory for generations starting with a specific token.

    Returns mean ± std entropy at each position for all gens in this branch.
    """
    prompt = probe._resolve_prompt(prompt) if hasattr(probe, '_resolve_prompt') else prompt
    from .probe import _resolve_prompt
    prompt = _resolve_prompt(prompt)

    ents_by_pos = [[] for _ in range(max_tokens)]
    tok = probe.tokenizer

    for gen in range(n_gens):
        try:
            meta = probe.meta(prompt, gen=gen, max_tokens=max_tokens)
            first_tid = int(meta.iloc[0]["chosen_token_id"])
            first_word = tok.decode([first_tid]).strip()
            if first_word != branch_token:
                continue
            for _, row in meta.iterrows():
                pos = int(row["position"])
                if pos < max_tokens:
                    ents_by_pos[pos].append(row["entropy"])
        except (FileNotFoundError, ValueError, KeyError):
            continue

    n_in_branch = len(ents_by_pos[0]) if ents_by_pos[0] else 0
    means = [float(np.mean(e)) if e else np.nan for e in ents_by_pos]
    stds = [float(np.std(e)) if e else np.nan for e in ents_by_pos]

    return {
        "branch": branch_token,
        "n_gens": n_in_branch,
        "entropy_mean": means,
        "entropy_std": stds,
    }


def cross_model_branch(probe_a, probe_b, prompt: str,
                       branch_token: str, n_gens: int = 100,
                       max_tokens: int = 10) -> dict:
    """Compare two models on the same branch.

    Finds generations starting with branch_token in model A,
    computes mean entropy/JS at each position between A and B's
    logits for those same generations.

    Note: positions 1+ are confounded (different continuations after
    the shared first token). For clean comparison, use teacher_force
    on the branch generation.
    """
    from .probe import _resolve_prompt
    prompt = _resolve_prompt(prompt)
    tok = probe_a.tokenizer

    js_by_pos = [[] for _ in range(max_tokens)]
    ent_a_by_pos = [[] for _ in range(max_tokens)]
    ent_b_by_pos = [[] for _ in range(max_tokens)]
    n_matched = 0

    # Find gens in A that start with branch_token
    for gen in range(n_gens):
        try:
            meta_a = probe_a.meta(prompt, gen=gen, max_tokens=max_tokens)
            first_tid = int(meta_a.iloc[0]["chosen_token_id"])
            if tok.decode([first_tid]).strip() != branch_token:
                continue

            # Check if B also has this gen starting with same token
            meta_b = probe_b.meta(prompt, gen=gen, max_tokens=max_tokens)
            first_tid_b = int(meta_b.iloc[0]["chosen_token_id"])
            if tok.decode([first_tid_b]).strip() != branch_token:
                continue

            n_matched += 1
            for pos in range(min(max_tokens, len(meta_a), len(meta_b))):
                try:
                    la = probe_a.logits(prompt, gen=gen, pos=pos, max_tokens=max_tokens)
                    lb = probe_b.logits(prompt, gen=gen, pos=pos, max_tokens=max_tokens)
                    js_by_pos[pos].append(js_divergence(la, lb))
                    ent_a_by_pos[pos].append(entropy(la))
                    ent_b_by_pos[pos].append(entropy(lb))
                except (FileNotFoundError, ValueError):
                    pass
        except (FileNotFoundError, ValueError, KeyError):
            continue

    return {
        "branch": branch_token,
        "n_matched": n_matched,
        "js_mean": [float(np.mean(j)) if j else np.nan for j in js_by_pos],
        "js_std": [float(np.std(j)) if j else np.nan for j in js_by_pos],
        "entropy_a_mean": [float(np.mean(e)) if e else np.nan for e in ent_a_by_pos],
        "entropy_b_mean": [float(np.mean(e)) if e else np.nan for e in ent_b_by_pos],
    }


def convergence_depth(probe, prompt: str, n_gens: int = 100,
                      max_tokens: int = 10, top_k: int = 3) -> dict:
    """At what depth do different branches become indistinguishable?

    Computes mean JS between the top-k branches' logit distributions
    at each position. When cross-branch JS ≈ within-branch JS, the
    tree structure no longer matters.
    """
    from .probe import _resolve_prompt
    prompt = _resolve_prompt(prompt)
    tok = probe.tokenizer

    # Group gens by first token
    from collections import defaultdict
    branches = defaultdict(list)
    for gen in range(n_gens):
        try:
            meta = probe.meta(prompt, gen=gen, max_tokens=max_tokens)
            first_tid = int(meta.iloc[0]["chosen_token_id"])
            word = tok.decode([first_tid]).strip()
            branches[word].append(gen)
        except (FileNotFoundError, ValueError, KeyError):
            continue

    # Top-k branches by count
    top_branches = sorted(branches.items(), key=lambda x: -len(x[1]))[:top_k]

    if len(top_branches) < 2:
        return {"depth": 0, "cross_js": [], "within_js": []}

    cross_js = []
    within_js = []

    for pos in range(max_tokens):
        # Collect logits per branch at this position
        branch_logits = {}
        for word, gens in top_branches:
            logits_list = []
            for g in gens[:20]:  # cap per branch
                try:
                    logits_list.append(
                        probe.logits(prompt, gen=g, pos=pos, max_tokens=max_tokens))
                except:
                    pass
            if logits_list:
                branch_logits[word] = logits_list

        if len(branch_logits) < 2:
            cross_js.append(np.nan)
            within_js.append(np.nan)
            continue

        # Cross-branch JS: mean JS between branch centroids
        centroids = {}
        for word, llist in branch_logits.items():
            centroids[word] = np.mean(llist, axis=0)

        cross_vals = []
        words = list(centroids.keys())
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                cross_vals.append(js_divergence(centroids[words[i]], centroids[words[j]]))
        cross_js.append(float(np.mean(cross_vals)))

        # Within-branch JS: mean JS between random pairs within same branch
        within_vals = []
        for word, llist in branch_logits.items():
            if len(llist) >= 2:
                for k in range(min(10, len(llist))):
                    i, j = np.random.choice(len(llist), 2, replace=False)
                    within_vals.append(js_divergence(llist[i], llist[j]))
        within_js.append(float(np.mean(within_vals)) if within_vals else np.nan)

    # Find convergence: where cross ≈ within
    conv_depth = max_tokens
    for pos in range(1, max_tokens):
        if (not np.isnan(cross_js[pos]) and not np.isnan(within_js[pos])
                and cross_js[pos] < within_js[pos] * 1.5):
            conv_depth = pos
            break

    return {
        "convergence_depth": conv_depth,
        "cross_branch_js": cross_js,
        "within_branch_js": within_js,
        "branches_used": [w for w, _ in top_branches],
    }


def branch_logit_profile(probe, prompt: str, branch_token: str,
                         track_tokens: list = None,
                         n_gens: int = 100, max_tokens: int = 10) -> 'pd.DataFrame':
    """Full logit profile for a branch: entropy, argmax, tracked token probs per position.

    Returns DataFrame with one row per position, columns:
        position, n_gens, entropy_mean, entropy_std, top_argmax, top_argmax_pct,
        P_{token} for each tracked token.

    If track_tokens is None, tracks the branch token itself + 'kill'.
    """
    import pandas as pd
    from collections import defaultdict, Counter
    from .probe import _resolve_prompt

    prompt = _resolve_prompt(prompt)
    tok = probe.tokenizer

    # Find gens in this branch
    branch_gens = []
    for gen in range(n_gens):
        try:
            meta = probe.meta(prompt, gen=gen, max_tokens=max_tokens)
            first_tid = int(meta.iloc[0]["chosen_token_id"])
            if tok.decode([first_tid]).strip() == branch_token:
                branch_gens.append(gen)
        except (FileNotFoundError, ValueError, KeyError):
            continue

    if not branch_gens:
        return pd.DataFrame()

    if track_tokens is None:
        track_tokens = list(set([branch_token, "kill"]))

    track_ids = {}
    for word in track_tokens:
        ids = tok.encode(" " + word, add_special_tokens=False)
        if ids:
            track_ids[word] = ids[0]

    rows = []
    for pos in range(max_tokens):
        ents = []
        argmaxes = Counter()
        token_probs = defaultdict(list)

        for gen in branch_gens:
            try:
                logits = probe.logits(prompt, gen=gen, pos=pos, max_tokens=max_tokens)
                probs = _softmax(logits)
                ents.append(entropy(logits))

                am = int(np.argmax(probs))
                argmaxes[tok.decode([am]).strip()] += 1

                for word, tid in track_ids.items():
                    if tid < len(probs):
                        token_probs[word].append(float(probs[tid]))
            except (FileNotFoundError, ValueError):
                pass

        if not ents:
            continue

        top_am = argmaxes.most_common(1)[0] if argmaxes else ("?", 0)
        row = {
            "position": pos,
            "n_gens": len(ents),
            "entropy_mean": float(np.mean(ents)),
            "entropy_std": float(np.std(ents)),
            "top_argmax": top_am[0],
            "top_argmax_pct": top_am[1] / len(ents),
        }
        for word in track_tokens:
            vals = token_probs.get(word, [])
            row[f"P_{word}"] = float(np.mean(vals)) if vals else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def generation_distance(probe_a, probe_b, prompt: str,
                        gen_a: int = 0, gen_b: int = 0,
                        n_positions: int = 50) -> dict:
    """Multiple distance measures between two generations.

    Works across models (base vs aligned) or within-model (gen 0 vs gen 1).
    Returns token, logit, hidden, and axis-level distances.
    """
    # Token-level
    meta_a = probe_a.meta(prompt, gen=gen_a)
    meta_b = probe_b.meta(prompt, gen=gen_b)
    tokens_a = set(meta_a["chosen_token"].values)
    tokens_b = set(meta_b["chosen_token"].values)
    jaccard = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)

    n = min(n_positions, len(meta_a), len(meta_b))

    # Bag-of-logits: average logit distribution across positions, then JS
    logits_a, logits_b = [], []
    pos_js_list = []
    for pos in range(n):
        try:
            la = probe_a.logits(prompt, gen=gen_a, pos=pos)
            lb = probe_b.logits(prompt, gen=gen_b, pos=pos)
            la, lb = _align_vocab(la, lb)
            logits_a.append(la)
            logits_b.append(lb)
            pos_js_list.append(js_divergence(la, lb))
        except (FileNotFoundError, ValueError):
            break

    bag_js = 0.0
    mean_pos_js = 0.0
    if logits_a:
        mean_a = np.mean(logits_a, axis=0)
        mean_b = np.mean(logits_b, axis=0)
        bag_js = js_divergence(mean_a, mean_b)
        mean_pos_js = float(np.mean(pos_js_list))

    # Hidden centroid distance
    cd = centroid_distance(probe_a, probe_b, prompt,
                           gen=gen_a, n_positions=n, layer=-1)
    centroid_cos = cd.get("centroid_cos_dist", np.nan) if cd else np.nan

    return {
        "token_jaccard": jaccard,
        "bag_of_logits_js": bag_js,
        "mean_position_js": mean_pos_js,
        "centroid_cos_dist": centroid_cos,
        "n_positions": n,
    }


def sentence_embeddings(texts: list, model_name: str = "BAAI/bge-m3") -> np.ndarray:
    """Embed texts with a sentence transformer. Returns (n, dim) array.

    Loads the model on first call, caches it.
    """
    from sentence_transformers import SentenceTransformer
    if not hasattr(sentence_embeddings, "_model") or sentence_embeddings._name != model_name:
        sentence_embeddings._model = SentenceTransformer(model_name)
        sentence_embeddings._name = model_name
    return sentence_embeddings._model.encode(texts, normalize_embeddings=True)


def text_distance(text_a: str, text_b: str,
                  embedder: str = "BAAI/bge-m3") -> float:
    """Cosine distance between two texts via sentence embeddings."""
    vecs = sentence_embeddings([text_a, text_b], embedder)
    return 1.0 - float(np.dot(vecs[0], vecs[1]))


def text_drift(text: str, window: int = 3,
               embedder: str = "BAAI/bge-m3") -> dict:
    """Semantic drift within a single text.

    Splits text into chunks of `window` sentences, embeds each,
    measures cosine distance between consecutive chunks.

    Returns dict with step_dists, total_drift, directedness, mean_step.
    """
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2:
        return {"step_dists": [], "total_drift": 0.0,
                "directedness": 1.0, "mean_step": 0.0}

    # Chunk into windows
    chunks = []
    for i in range(0, len(sentences), window):
        chunk = " ".join(sentences[i:i+window])
        if chunk:
            chunks.append(chunk)

    if len(chunks) < 2:
        return {"step_dists": [], "total_drift": 0.0,
                "directedness": 1.0, "mean_step": 0.0}

    vecs = sentence_embeddings(chunks, embedder)

    step_dists = [1.0 - float(np.dot(vecs[i], vecs[i+1]))
                  for i in range(len(vecs) - 1)]
    total_drift = 1.0 - float(np.dot(vecs[0], vecs[-1]))
    path_length = sum(step_dists)
    directedness = total_drift / path_length if path_length > 1e-10 else 1.0

    return {
        "step_dists": step_dists,
        "total_drift": total_drift,
        "path_length": path_length,
        "directedness": directedness,
        "mean_step": float(np.mean(step_dists)),
        "n_chunks": len(chunks),
    }


def generation_text_metrics(probe, prompt: str, gen: int = 0,
                            embedder: str = "BAAI/bge-m3") -> dict:
    """Text-level metrics for a single generation: drift + embedding."""
    text = probe.text(prompt, gen=gen)
    drift = text_drift(text, embedder=embedder)
    return {
        "text": text,
        "n_tokens": len(text.split()),
        **{f"drift_{k}": v for k, v in drift.items() if k != "step_dists"},
    }


def cross_generation_text_distance(probe_a, probe_b, prompt: str,
                                   gen_a: int = 0, gen_b: int = 0,
                                   embedder: str = "BAAI/bge-m3") -> dict:
    """Text-level distance between two generations using sentence embeddings."""
    text_a = probe_a.text(prompt, gen=gen_a)
    text_b = probe_b.text(prompt, gen=gen_b)

    dist = text_distance(text_a, text_b, embedder)
    drift_a = text_drift(text_a, embedder=embedder)
    drift_b = text_drift(text_b, embedder=embedder)

    return {
        "text_cos_dist": dist,
        "drift_a": drift_a["total_drift"],
        "drift_b": drift_b["total_drift"],
        "directedness_a": drift_a["directedness"],
        "directedness_b": drift_b["directedness"],
    }


# =============================================================================
# Batch: compute all T1 metrics for a (base, aligned) pair
# =============================================================================

def compare(base_logits: np.ndarray, aligned_logits: np.ndarray) -> dict:
    """All Tier 1 distribution metrics between base and aligned logits."""
    return {
        "entropy_base": entropy(base_logits),
        "entropy_aligned": entropy(aligned_logits),
        "entropy_delta": entropy(aligned_logits) - entropy(base_logits),
        "js_divergence": js_divergence(base_logits, aligned_logits),
        "kl_base_to_aligned": kl_divergence(base_logits, aligned_logits),
        "kl_aligned_to_base": kl_divergence(aligned_logits, base_logits),
        "top50_overlap": top_k_overlap(base_logits, aligned_logits, k=50),
        "rank_correlation": rank_correlation(base_logits, aligned_logits),
        "effective_vocab_base": effective_vocab(base_logits),
        "effective_vocab_aligned": effective_vocab(aligned_logits),
        "base_token_surprisal": base_token_surprisal(base_logits, aligned_logits),
        "base_top10_mass": base_top_k_mass(base_logits, aligned_logits, k=10),
        **{f"tail_{k}": v for k, v in
           tail_redistribution(base_logits, aligned_logits).items()},
    }


def circuit_profile(dive, prompt: str, gen: int = 0, pos: int = 0,
                    embed_checkpoint: str = "base",
                    tokenizer=None) -> 'pd.DataFrame':
    """Full circuit characterisation: decompose alignment into stage contributions.

    For each edge (base→sft, sft→dpo, dpo→rlvr) and endpoint, computes
    all distributional metrics, axis loadings, and stage attribution.

    Returns DataFrame with one row per edge, columns for all metrics
    plus sft_share (fraction of total displacement done by SFT).

    Usage:
        dive = DeepDive("olmo")
        prof = circuit_profile(dive, "anger")
        print(prof[["edge", "js_divergence", "share", "violence_delta"]])
    """
    import pandas as pd

    if tokenizer is None:
        tokenizer = dive.tokenizer

    cps = dive.checkpoints()
    if not cps:
        raise FileNotFoundError(f"No data for {dive.family}")

    # Load logits at the specified position
    logits = {}
    for cp in cps:
        try:
            logits[cp] = dive.logits(cp, prompt, gen=gen, pos=pos)
        except (FileNotFoundError, ValueError):
            pass

    if len(logits) < 2:
        raise ValueError(f"Need at least 2 checkpoints, got {list(logits.keys())}")

    # Axis projections
    v_axis = p_axis = embed = None
    try:
        embed = dive.embedding_matrix(embed_checkpoint)
        v_axis, p_axis = violence_procedural_axes(embed, tokenizer)
    except FileNotFoundError:
        pass

    # Canonical edge ordering
    edge_order = ["base", "sft", "dpo", "rlvr"]
    available = [cp for cp in edge_order if cp in logits]

    # Node-level metrics
    node_rows = []
    for cp in available:
        row = {"checkpoint": cp, "type": "node", "position": pos}
        row["entropy"] = entropy(logits[cp])
        row["effective_vocab"] = effective_vocab(logits[cp])
        if v_axis is not None:
            row["violence_loading"] = axis_loading(logits[cp], embed, v_axis)
            row["procedural_loading"] = axis_loading(logits[cp], embed, p_axis)
        row["argmax_id"] = int(np.argmax(logits[cp]))
        if tokenizer:
            row["argmax_token"] = tokenizer.decode([row["argmax_id"]]).strip()
        node_rows.append(row)

    # Edge-level metrics (consecutive pairs)
    edge_rows = []
    for i in range(len(available) - 1):
        cp_from, cp_to = available[i], available[i + 1]
        la, lb = logits[cp_from], logits[cp_to]

        row = {"edge": f"{cp_from}→{cp_to}", "from": cp_from, "to": cp_to,
               "type": "edge", "position": pos}
        row.update(compare(la, lb))

        if v_axis is not None:
            vl_from = axis_loading(la, embed, v_axis)
            vl_to = axis_loading(lb, embed, v_axis)
            pl_from = axis_loading(la, embed, p_axis)
            pl_to = axis_loading(lb, embed, p_axis)
            row["violence_delta"] = vl_to - vl_from
            row["procedural_delta"] = pl_to - pl_from

        # Argmax change
        argmax_from = int(np.argmax(la))
        argmax_to = int(np.argmax(lb))
        row["argmax_changed"] = argmax_from != argmax_to
        if tokenizer:
            row["argmax_change"] = (
                f"{tokenizer.decode([argmax_from]).strip()}"
                f"→{tokenizer.decode([argmax_to]).strip()}")

        edge_rows.append(row)

    # Stage attribution: what fraction of total displacement is each edge?
    if "base" in logits and len(available) >= 2:
        final_cp = available[-1]
        total_js = js_divergence(logits["base"], logits[final_cp])
        for row in edge_rows:
            row["total_js"] = total_js
            row["share"] = row["js_divergence"] / total_js if total_js > 1e-10 else 0.0

    return pd.DataFrame(edge_rows + node_rows)


def circuit_summary(dive, prompt: str, gen: int = 0, pos: int = 0,
                    tokenizer=None) -> dict:
    """One-line circuit characterisation for a prompt.

    Returns dict with:
        total_js, sft_share, dpo_share, dominant_stage,
        argmax_base, argmax_final, argmax_changed_at,
        violence_delta_sft, violence_delta_dpo
    """
    if tokenizer is None:
        tokenizer = dive.tokenizer

    cps = dive.checkpoints()
    edge_order = ["base", "sft", "dpo", "rlvr"]
    available = [cp for cp in edge_order if cp in cps]

    logits = {cp: dive.logits(cp, prompt, gen=gen, pos=pos) for cp in available}

    total_js = js_divergence(logits[available[0]], logits[available[-1]])

    edges = {}
    for i in range(len(available) - 1):
        a, b = available[i], available[i + 1]
        edges[f"{a}→{b}"] = js_divergence(logits[a], logits[b])

    shares = {e: v / total_js if total_js > 1e-10 else 0.0
              for e, v in edges.items()}
    dominant = max(shares, key=shares.get) if shares else "none"

    result = {
        "total_js": total_js,
        "dominant_stage": dominant,
    }
    for e, s in shares.items():
        stage = e.split("→")[1]
        result[f"{stage}_share"] = s
        result[f"{stage}_js"] = edges[e]

    # Argmax tracking
    for cp in available:
        tok_id = int(np.argmax(logits[cp]))
        result[f"argmax_{cp}"] = tokenizer.decode([tok_id]).strip()

    base_argmax = int(np.argmax(logits[available[0]]))
    for cp in available[1:]:
        if int(np.argmax(logits[cp])) != base_argmax:
            result["argmax_changed_at"] = cp
            break

    return result


def mode_decomposition(dive, prompt: str, gen: int = 0, pos: int = 0) -> 'pd.DataFrame':
    """Decompose distributional change into alignment vs mode components.

    Compares all available modes (raw, complete, chat, think) and
    checkpoints to separate: (1) alignment effect (same mode, different
    weights), (2) special tokens (raw→complete), (3) turn structure
    (complete→chat), (4) thinking (chat→think).

    Returns DataFrame with one row per comparison.
    """
    import pandas as pd

    cps = dive.checkpoints()

    # Group by mode: base, base.chat, base.complete, base.think, etc.
    modes = {}
    for cp in cps:
        if "." in cp:
            base_cp, mode = cp.rsplit(".", 1)
        else:
            base_cp, mode = cp, "raw"
        modes.setdefault(mode, []).append((base_cp, cp))

    # Need at least raw mode
    if "raw" not in modes:
        raise ValueError("No raw mode data found")

    # Load logits for all mode×checkpoint combinations
    logits = {}
    for mode, pairs in modes.items():
        for base_cp, full_cp in pairs:
            try:
                logits[(base_cp, mode)] = dive.logits(full_cp, prompt, gen=gen, pos=pos)
            except (FileNotFoundError, ValueError):
                pass

    rows = []

    # Find base and final checkpoint in raw mode
    edge_order = ["base", "sft", "dpo", "rlvr"]
    raw_cps = [cp for cp, _ in modes.get("raw", []) if cp in edge_order]
    raw_cps.sort(key=lambda x: edge_order.index(x))

    if len(raw_cps) < 2:
        raise ValueError("Need at least base and one aligned checkpoint in raw mode")

    base_cp = raw_cps[0]
    final_cp = raw_cps[-1]

    # Alignment effect per mode
    for mode in sorted(modes.keys()):
        if (base_cp, mode) in logits and (final_cp, mode) in logits:
            js = js_divergence(logits[(base_cp, mode)], logits[(final_cp, mode)])
            rows.append({
                "comparison": f"alignment ({mode})",
                "from": f"{base_cp}.{mode}", "to": f"{final_cp}.{mode}",
                "js_divergence": js, "component": "alignment",
                "mode": mode,
            })

    # Mode transitions on base model
    mode_chain = ["raw", "chat", "continue", "think"]
    mode_labels = {
        ("raw", "chat"): "special tokens",
        ("chat", "continue"): "instruction framing",
        ("continue", "think"): "thinking",
    }

    for cp in [base_cp, final_cp]:
        for i in range(len(mode_chain) - 1):
            m_from, m_to = mode_chain[i], mode_chain[i + 1]
            if (cp, m_from) in logits and (cp, m_to) in logits:
                js = js_divergence(logits[(cp, m_from)], logits[(cp, m_to)])
                label = mode_labels.get((m_from, m_to), f"{m_from}→{m_to}")
                rows.append({
                    "comparison": f"{label} ({cp})",
                    "from": f"{cp}.{m_from}", "to": f"{cp}.{m_to}",
                    "js_divergence": js, "component": label,
                    "mode": f"{m_from}→{m_to}",
                })

        # Total: raw→chat
        if (cp, "raw") in logits and (cp, "chat") in logits:
            js = js_divergence(logits[(cp, "raw")], logits[(cp, "chat")])
            rows.append({
                "comparison": f"total template ({cp})",
                "from": f"{cp}.raw", "to": f"{cp}.chat",
                "js_divergence": js, "component": "total template",
                "mode": "raw→chat",
            })

    df = pd.DataFrame(rows)

    # Add ratio to alignment
    align_raw = df[df["comparison"] == "alignment (raw)"]["js_divergence"].values
    if len(align_raw) > 0:
        df["ratio_to_alignment"] = df["js_divergence"] / align_raw[0]

    return df


def compare_all_positions(dive, checkpoint_a: str, checkpoint_b: str,
                          prompt: str, gen: int = 0) -> 'pd.DataFrame':
    """Compare two checkpoints across all positions for one generation.

    Returns DataFrame with one row per position, all T1 metrics as columns.
    """
    import pandas as pd

    meta = dive.meta(checkpoint_a, prompt, gen=gen)
    rows = []
    for pos in meta["position"].values:
        try:
            la = dive.logits(checkpoint_a, prompt, gen=gen, pos=pos)
            lb = dive.logits(checkpoint_b, prompt, gen=gen, pos=pos)
            row = {"position": pos, **compare(la, lb)}
            rows.append(row)
        except (ValueError, FileNotFoundError):
            continue
    return pd.DataFrame(rows)
