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

def entropy(logits: np.ndarray) -> float:
    """Shannon entropy of softmax(logits) in nats."""
    p = _softmax(logits)
    p = np.clip(p, 1e-10, None)
    return -float(np.sum(p * np.log(p)))


def kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """KL(P || Q) in nats. How much Q diverges from P."""
    p = np.clip(_softmax(logits_p), 1e-10, None)
    q = np.clip(_softmax(logits_q), 1e-10, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. Symmetric, bounded [0, ln(2)]."""
    p = np.clip(_softmax(logits_a), 1e-10, None)
    q = np.clip(_softmax(logits_b), 1e-10, None)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def top_k_overlap(logits_a: np.ndarray, logits_b: np.ndarray,
                  k: int = 50) -> float:
    """Fraction of top-k tokens shared. 1.0 = identical, 0.0 = disjoint."""
    top_a = set(np.argsort(logits_a)[-k:])
    top_b = set(np.argsort(logits_b)[-k:])
    return len(top_a & top_b) / k


def rank_correlation(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Spearman rank correlation of logit orderings."""
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
    p = _softmax(logits_a)
    q = _softmax(logits_b)
    delta = q - p  # positive = gained in b, negative = lost
    order = np.argsort(delta)
    repressed = [(int(i), float(delta[i])) for i in order[:k]]
    amplified = [(int(i), float(delta[i])) for i in order[-k:][::-1]]
    return {"repressed": repressed, "amplified": amplified}


def base_token_surprisal(base_logits: np.ndarray,
                         aligned_logits: np.ndarray) -> float:
    """Surprisal of the base model's argmax under the aligned distribution, in bits.

    -log2(p_aligned(argmax_base)). Higher = more repressed.
    Compare with self-surprisal: -log2(p_base(argmax_base)).
    Delta = repression measured as information.
    """
    base_argmax = int(np.argmax(base_logits))
    aligned_probs = np.clip(_softmax(aligned_logits), 1e-10, None)
    return -float(np.log2(aligned_probs[base_argmax]))


def base_top_k_mass(base_logits: np.ndarray, aligned_logits: np.ndarray,
                    k: int = 10) -> float:
    """How much probability mass the aligned model assigns to the base's top-k.

    1.0 = aligned preserves base's top predictions.
    0.0 = aligned completely displaces them.
    """
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
    p_base = _softmax(base_logits)
    p_aligned = _softmax(aligned_logits)

    # Top-k defined by base model's ranking
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
