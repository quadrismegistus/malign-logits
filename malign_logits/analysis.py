from . import *


def compute_displacement(
    base_words,
    ego_words,
    superego_words,
    model,
    tokenizer,
    prompt,
    similarity_threshold=0.3,
    min_repression=0.005,
    min_displacement_mass=0.0005,
    hidden_layer=16,
    device=None,
):
    """
    Displacement engine v4.
    
    Incorporates base model as drive energy weighting.
    Repressed words with stronger base-model drive behind them
    displace more mass, producing heavier symptoms.
    """
    if device is None:
        device = next(model.parameters()).device

    all_words = set(base_words.keys()) | set(ego_words.keys()) | set(superego_words.keys())

    # Compute mean base probability for drive weighting
    base_vals = [base_words.get(w, 0) for w in all_words if base_words.get(w, 0) > 0]
    mean_base = sum(base_vals) / len(base_vals) if base_vals else 1e-10

    # Identify repressed words with drive weighting
    repressed = {}
    for w in all_words:
        e = ego_words.get(w, 0)
        s = superego_words.get(w, 0)
        repression = e - s
        if repression > min_repression:
            base_drive = base_words.get(w, 0)
            drive_weight = 1 + math.log(1 + base_drive / (mean_base + 1e-10))
            repressed[w] = {
                'raw_mass': repression,
                'base_drive': base_drive,
                'drive_weight': drive_weight,
                'effective_mass': repression * drive_weight,
                'ego': e,
                'superego': s,
            }

    # Permitted words — not repressed, superego allows them
    permitted = {}
    for w, prob in superego_words.items():
        if w not in repressed and prob > 0.001:
            permitted[w] = prob

    if not repressed or not permitted:
        return dict(superego_words), {}, repressed

    # Contextual embeddings
    def get_contextual_embedding(word):
        text = prompt + " " + word
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer.encode(prompt))
        with torch.no_grad():
            outputs = model(ids, output_hidden_states=True)
            hidden = outputs.hidden_states[hidden_layer]
            word_hidden = hidden[0, prompt_len:, :].mean(dim=0).cpu()
        return word_hidden

    print("  Computing contextual embeddings...")
    rep_embs = {}
    for w in repressed:
        try:
            rep_embs[w] = get_contextual_embedding(w)
        except Exception:
            continue

    perm_embs = {}
    for w in permitted:
        try:
            perm_embs[w] = get_contextual_embedding(w)
        except Exception:
            continue

    for w in rep_embs:
        rep_embs[w] = torch.nn.functional.normalize(rep_embs[w].unsqueeze(0), dim=-1).squeeze()
    for w in perm_embs:
        perm_embs[w] = torch.nn.functional.normalize(perm_embs[w].unsqueeze(0), dim=-1).squeeze()

    def shares_stem(w1, w2, min_shared=4):
        w1, w2 = w1.lower(), w2.lower()
        if w1 in w2 or w2 in w1:
            return True
        shared = 0
        for a, b in zip(w1, w2):
            if a == b:
                shared += 1
            else:
                break
        return shared >= min_shared and shared >= 0.6 * min(len(w1), len(w2))

    # Displacement with drive weighting
    displaced = dict(superego_words)
    condensation_log = defaultdict(list)

    for rep_word, rep_info in repressed.items():
        if rep_word not in rep_embs:
            continue

        # Use drive-weighted mass instead of raw repression
        mass = rep_info['effective_mass']
        targets = {}

        for perm_word, perm_emb in perm_embs.items():
            if shares_stem(rep_word, perm_word):
                continue
            sim = torch.dot(rep_embs[rep_word], perm_emb).item()
            if sim < similarity_threshold:
                continue
            targets[perm_word] = sim

        if not targets:
            continue

        total_weight = sum(targets.values())
        for target, weight in targets.items():
            added = mass * (weight / total_weight)
            if added < min_displacement_mass:
                continue
            displaced[target] = displaced.get(target, 0) + added
            condensation_log[target].append({
                'source': rep_word,
                'mass': added,
                'raw_repression': rep_info['raw_mass'],
                'drive_weight': rep_info['drive_weight'],
                'base_drive': rep_info['base_drive'],
                'similarity': weight,
            })

    # Normalize
    total = sum(displaced.values())
    if total > 0:
        displaced = {w: p / total for w, p in displaced.items()}

    displaced = dict(sorted(displaced.items(), key=lambda x: -x[1]))
    return displaced, dict(condensation_log), repressed




def build_analysis_df(
    base_words, ego_words, superego_words,
    displaced_dist, condensation_log, repressed_analysis
):
    """
    Combine all layers of analysis into a single DataFrame.
    One row per word, all features included.
    """
    all_words = set()
    for d in [base_words, ego_words, superego_words, displaced_dist]:
        all_words.update(d.keys())

    rows = []
    for w in all_words:
        base = base_words.get(w, 0)
        ego = ego_words.get(w, 0)
        superego = superego_words.get(w, 0)
        neurotic = displaced_dist.get(w, 0)

        raw_repression = ego - superego
        displaced_mass = neurotic - superego

        # Drive info (only available for repressed words)
        rep_info = repressed_analysis.get(w, {})
        drive_weight = rep_info.get('drive_weight', None)
        effective_mass = rep_info.get('effective_mass', None)

        # Condensation info (only for displacement targets)
        cond = condensation_log.get(w, [])
        n_sources = len(cond)
        total_absorbed = sum(s['mass'] for s in cond)
        sources = [s['source'] for s in cond] if cond else []

        # Classification
        if raw_repression > 0.01:
            role = 'repressed'
        elif raw_repression < -0.01:
            role = 'amplified'
        elif displaced_mass > 0.003:
            role = 'symptomatic'
        else:
            role = 'neutral'

        rows.append({
            'word': w,
            'base_drive': round(base, 4),
            'ego': round(ego, 4),
            'superego': round(superego, 4),
            'neurotic': round(neurotic, 4),
            'raw_repression': round(raw_repression, 4),
            'drive_weight': round(drive_weight, 2) if drive_weight else None,
            'effective_mass': round(effective_mass, 4) if effective_mass else None,
            'displaced_mass': round(displaced_mass, 4),
            'n_condensation_sources': n_sources,
            'total_absorbed': round(total_absorbed, 4) if total_absorbed else 0,
            'condensation_sources': sources if sources else None,
            'role': role,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('effective_mass', ascending=False)
    return df








def compute_repression(
    ego_words, superego_words, base_words=None, threshold=0.01,
    col_a="ego", col_b="superego",
):
    """
    Compare two word distributions to identify what the second suppresses
    or amplifies relative to the first.

    By default the columns are named 'ego' and 'superego', but col_a/col_b
    can be overridden for other comparisons (e.g. base vs ego).

    Args:
        ego_words: dict, first distribution (higher = "wants it").
        superego_words: dict, second distribution (lower delta = "suppresses it").
        base_words: Optional third distribution included as a reference column.
        threshold: Minimum absolute delta to count as repressed/amplified.
        col_a: Column name for the first distribution.
        col_b: Column name for the second distribution.

    Returns:
        DataFrame with columns: word, base, <col_a>, <col_b>, delta, ratio,
        repressed, amplified.
    """
    all_words = set(ego_words.keys()) | set(superego_words.keys())
    if base_words is not None:
        all_words.update(base_words.keys())
    rows = []
    for w in all_words:
        e = ego_words.get(w, 0)
        s = superego_words.get(w, 0)
        b = base_words.get(w, 0) if base_words is not None else 0
        rows.append({
            "word": w,
            "base": b,
            col_a: e,
            col_b: s,
            "delta": e - s,
            "ratio": e / (s + 1e-10) if e > s else -(s / (e + 1e-10)),
        })

    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df["repressed"] = df["delta"] > threshold
    df["amplified"] = df["delta"] < -threshold
    return df


def compute_id(base_words, ego_words, superego_words):
    """
    Compute id-scores using all three model layers.

    The id is not any single model's output — it emerges from the
    relationship between all three. A word scores high when:
        1. The ego wants to say it (high ego probability)
        2. The superego forbids it (ego >> superego)
        3. There is raw statistical drive energy behind it (high base probability)

    Args:
        base_words: dict from discover_top_words on base model.
        ego_words: dict from discover_top_words on instruct model.
        superego_words: dict from discover_top_words on instruct + prohibition.

    Returns:
        (id_scores, analysis) where:
            id_scores: dict mapping word -> float score, sorted descending.
            analysis: dict mapping word -> detail dict with components.
    """
    all_words = set(base_words.keys()) | set(ego_words.keys()) | set(superego_words.keys())

    base_vals = [base_words.get(w, 0) for w in all_words]
    mean_drive = sum(base_vals) / len(base_vals) if base_vals else 1e-10

    id_scores = {}
    analysis = {}

    for word in all_words:
        b = base_words.get(word, 0)
        e = ego_words.get(word, 0)
        s = superego_words.get(word, 0)

        repression = e - s
        if repression <= 0:
            continue

        drive_weight = 1 + math.log(1 + b / (mean_drive + 1e-10))
        id_score = repression * drive_weight

        id_scores[word] = id_score
        analysis[word] = {
            "id_score": id_score,
            "base_drive": b,
            "ego_desire": e,
            "superego_allows": s,
            "repression": repression,
            "drive_weight": drive_weight,
        }

    id_scores = dict(sorted(id_scores.items(), key=lambda x: -x[1]))
    return id_scores, analysis


# ---------------------------------------------------------------------------
# Distribution-level metrics (operate on cached logits, no forward passes)
# ---------------------------------------------------------------------------

def distribution_entropy(logits):
    """Entropy of the softmax distribution. Higher = flatter/more uncertain.

    Args:
        logits: Raw logits tensor (vocab_size,).

    Returns:
        float: Shannon entropy in nats.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    probs = probs.clamp(min=1e-10)
    return -(probs * probs.log()).sum().item()


def kl_divergence(logits_p, logits_q):
    """KL(P || Q) between two logit distributions.

    Measures how much information is lost when Q is used to approximate P.
    KL(ego || superego) = how much the superego diverges from the ego.

    Args:
        logits_p: Raw logits for distribution P (vocab_size,).
        logits_q: Raw logits for distribution Q (vocab_size,).

    Returns:
        float: KL divergence in nats (always >= 0).
    """
    p = torch.softmax(logits_p.float(), dim=-1).clamp(min=1e-10)
    q = torch.softmax(logits_q.float(), dim=-1).clamp(min=1e-10)
    return (p * (p.log() - q.log())).sum().item()


def js_divergence(logits_a, logits_b):
    """Jensen-Shannon divergence between two logit distributions.

    Symmetric, bounded [0, ln(2)]. More stable than KL for comparing
    distributions that don't fully overlap.

    Args:
        logits_a: Raw logits (vocab_size,).
        logits_b: Raw logits (vocab_size,).

    Returns:
        float: JS divergence in nats.
    """
    p = torch.softmax(logits_a.float(), dim=-1).clamp(min=1e-10)
    q = torch.softmax(logits_b.float(), dim=-1).clamp(min=1e-10)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum().item()
    kl_qm = (q * (q.log() - m.log())).sum().item()
    return 0.5 * (kl_pm + kl_qm)


def top_k_overlap(logits_a, logits_b, k=50):
    """Fraction of top-k tokens shared between two distributions.

    1.0 = identical top-k. 0.0 = completely disjoint.

    Args:
        logits_a, logits_b: Raw logits (vocab_size,).
        k: Number of top tokens to compare.

    Returns:
        float: Overlap fraction [0, 1].
    """
    top_a = set(torch.topk(logits_a.float(), k).indices.tolist())
    top_b = set(torch.topk(logits_b.float(), k).indices.tolist())
    return len(top_a & top_b) / k


def rank_correlation(logits_a, logits_b):
    """Spearman rank correlation between two logit distributions.

    Measures whether the *ordering* of tokens is preserved across layers.
    High rank correlation + high JS = mass shifted but ordering preserved
    (formatting-type displacement). Low rank correlation = ordering
    scrambled (safety-type displacement).

    Args:
        logits_a, logits_b: Raw logits (vocab_size,).

    Returns:
        float: Spearman rho in [-1, 1].
    """
    from scipy.stats import spearmanr
    return spearmanr(logits_a.float().numpy(), logits_b.float().numpy()).statistic


def distribution_metrics(base_logits, ego_logits, superego_logits, instruct_logits=None):
    """Compute all distribution-level metrics between layers.

    Operates entirely on cached logits — no forward passes needed.
    ego_logits may be None for 2-layer topologies.

    Args:
        base_logits: Raw logits from base model (vocab_size,).
        ego_logits: Raw logits from SFT model, or None.
        superego_logits: Raw logits from DPO model (vocab_size,).
        instruct_logits: Optional raw logits from RLVR model.

    Returns:
        dict with all metrics.
    """
    metrics = {
        "entropy_base": distribution_entropy(base_logits),
        "entropy_superego": distribution_entropy(superego_logits),
        "js_base_superego": js_divergence(base_logits, superego_logits),
        "top50_overlap_base_superego": top_k_overlap(base_logits, superego_logits, k=50),
        "rank_corr_base_superego": rank_correlation(base_logits, superego_logits),
    }

    if ego_logits is not None:
        metrics.update({
            "entropy_ego": distribution_entropy(ego_logits),
            "kl_base_ego": kl_divergence(base_logits, ego_logits),
            "kl_ego_superego": kl_divergence(ego_logits, superego_logits),
            "js_base_ego": js_divergence(base_logits, ego_logits),
            "js_ego_superego": js_divergence(ego_logits, superego_logits),
            "top50_overlap_base_ego": top_k_overlap(base_logits, ego_logits, k=50),
            "top50_overlap_ego_superego": top_k_overlap(ego_logits, superego_logits, k=50),
            "rank_corr_base_ego": rank_correlation(base_logits, ego_logits),
            "rank_corr_ego_superego": rank_correlation(ego_logits, superego_logits),
            "entropy_drop_sft": distribution_entropy(base_logits) - distribution_entropy(ego_logits),
            "entropy_drop_dpo": distribution_entropy(ego_logits) - distribution_entropy(superego_logits),
        })
    else:
        # 2-layer: single transition base→superego
        metrics["kl_base_superego"] = kl_divergence(base_logits, superego_logits)
        metrics["entropy_drop_alignment"] = distribution_entropy(base_logits) - distribution_entropy(superego_logits)

    if instruct_logits is not None:
        metrics.update({
            "entropy_instruct": distribution_entropy(instruct_logits),
            "kl_superego_instruct": kl_divergence(superego_logits, instruct_logits),
            "js_superego_instruct": js_divergence(superego_logits, instruct_logits),
            "top50_overlap_superego_instruct": top_k_overlap(superego_logits, instruct_logits, k=50),
            "rank_corr_superego_instruct": rank_correlation(superego_logits, instruct_logits),
            "entropy_drop_rlvr": distribution_entropy(superego_logits) - distribution_entropy(instruct_logits),
        })

    return metrics


def top_movers(logits_a, logits_b, tokenizer, k=20):
    """Find tokens with the largest probability shift between two distributions.

    No forward passes — pure logit comparison.

    Args:
        logits_a: Source logits (vocab_size,).
        logits_b: Target logits (vocab_size,).
        tokenizer: For decoding token IDs to strings.
        k: Number of top movers in each direction.

    Returns:
        dict with 'repressed' (a >> b) and 'amplified' (b >> a),
        each a list of (token_str, prob_a, prob_b, delta) tuples.
    """
    probs_a = torch.softmax(logits_a.float(), dim=-1)
    probs_b = torch.softmax(logits_b.float(), dim=-1)
    delta = probs_a - probs_b

    # Most repressed (highest positive delta = a wants it, b doesn't)
    rep_indices = delta.topk(k).indices
    repressed = []
    for idx in rep_indices:
        i = idx.item()
        token = tokenizer.decode(i).strip()
        if token:
            repressed.append((token, probs_a[i].item(), probs_b[i].item(), delta[i].item()))

    # Most amplified (most negative delta = b wants it, a doesn't)
    amp_indices = (-delta).topk(k).indices
    amplified = []
    for idx in amp_indices:
        i = idx.item()
        token = tokenizer.decode(i).strip()
        if token:
            amplified.append((token, probs_a[i].item(), probs_b[i].item(), delta[i].item()))

    return {"repressed": repressed, "amplified": amplified}


def measure_overdetermination(
    word, base_logits, tokenizer, embeddings, top_k=20
):
    """
    Measure how overdetermined a word is in the base model's landscape.

    High entropy among semantically similar tokens = high overdetermination,
    meaning multiple associative paths converge on this semantic region.
    In Freudian terms, these are the nodal points where multiple drives meet.

    Args:
        word: The word to measure.
        base_logits: Raw logits from the base model.
        tokenizer: Tokenizer.
        embeddings: Model embedding weights.
        top_k: Number of semantic neighbors to consider.

    Returns:
        float: Entropy score (higher = more overdetermined).
    """
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        return 0.0

    anchor_id = token_ids[0]
    normed_emb = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    anchor_emb = normed_emb[anchor_id].unsqueeze(0)
    similarities = (normed_emb @ anchor_emb.squeeze()).squeeze()

    top_similar = torch.topk(similarities, top_k)

    base_probs = torch.softmax(base_logits.float(), dim=-1)
    neighbor_probs = base_probs[top_similar.indices]

    if neighbor_probs.sum() > 0:
        neighbor_dist = neighbor_probs / neighbor_probs.sum()
        entropy = -(neighbor_dist * torch.log(neighbor_dist + 1e-10)).sum().item()
    else:
        entropy = 0.0

    return entropy