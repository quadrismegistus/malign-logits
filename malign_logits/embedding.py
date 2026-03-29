"""
Generation + embedding pipeline for cross-family narrative analysis.

Generates many completions per prompt per model layer, embeds them with
SentenceTransformer, and computes cluster geometry / concept vector metrics.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Generation ────────────────────────────────────────────────────

PATH_GEN_STASH = None  # set lazily


def _gen_stash_path():
    from . import PATH_STASH
    return PATH_STASH + "_gen_battery"


def _check_cached_count(prompt, temperature=1.0, model_ids=None,
                        cache_dir=None):
    """Check how many generations are cached for a prompt+models combo."""
    from hashstash import HashStash

    stash_path = cache_dir or _gen_stash_path()
    stash = HashStash(root_dir=stash_path, append_mode=True)
    key = {
        "prompt": prompt,
        "temperature": temperature,
        "models": tuple(model_ids or []),
    }
    got = stash.get_all(key)
    return len(got) if got else 0


def generate_many(psyche, prompt, n=30, max_new_tokens=100,
                  temperature=1.0, cache_dir=None):
    """Generate n completions per model layer, with resume support.

    Returns DataFrame with columns: prompt, temperature, model, psg.
    Caches to HashStash so re-running only generates the deficit.
    """
    from hashstash import HashStash

    stash_path = cache_dir or _gen_stash_path()
    stash = HashStash(root_dir=stash_path, append_mode=True)
    # Key includes all model IDs so different families don't share cache
    model_ids = [psyche.primary_process.model_id]
    if psyche.ego is not None:
        model_ids.append(psyche.ego.model_id)
    if psyche.superego is not None:
        model_ids.append(psyche.superego.model_id)
    key = {
        "prompt": prompt,
        "temperature": temperature,
        "models": tuple(model_ids),
    }

    got = stash.get_all(key)
    existing = len(got) if got else 0
    needed = n - existing

    if needed > 0:
        label_map = {"base": "BASE", "ego": "SFT", "superego": "DPO",
                     "instruct": "RLVR"}
        for i in range(needed):
            gens = psyche.generate(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, verbose=False,
            )
            stash[key] = gens
            # Print each generation
            import textwrap
            print(f"  [{existing + i + 1}/{n}]")
            for layer, text in gens.items():
                if layer == "prompt":
                    continue
                label = label_map.get(layer, layer.upper())
                clean = text.replace("\n", " ").strip()
                prefix = f"    {label:4s}: "
                indent = " " * len(prefix)
                wrapped = textwrap.fill(clean, width=100,
                                        initial_indent=prefix,
                                        subsequent_indent=indent)
                print(wrapped)

    # Retrieve all and build DataFrame
    all_gens = stash.get_all(key)
    rows = []
    for gen in all_gens[:n]:
        for model, psg in gen.items():
            if model == "prompt":
                continue
            rows.append({
                "prompt": prompt,
                "temperature": temperature,
                "model": model,
                "psg": psg,
            })
    return pd.DataFrame(rows)


# ── Embedding ─────────────────────────────────────────────────────

_embedder = None


def _get_embedder(model_name="all-MiniLM-L6-v2"):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_generations(psg_df, model_name="all-MiniLM-L6-v2"):
    """Embed all passages. Returns DataFrame with 384 columns.

    Args:
        psg_df: DataFrame with 'psg' column.
    """
    embedder = _get_embedder(model_name)
    texts = psg_df["psg"].astype(str).tolist()
    vecs = embedder.encode(texts, show_progress_bar=True)
    return pd.DataFrame(vecs)


# ── Concept vectors ───────────────────────────────────────────────

def concept_seeds():
    """Return seed text dictionaries for concept axes."""
    return {
        "violent": {
            "positive": [
                "kill the little child, the anger flashed, the blood",
                "kill her own mother with bare hands",
                "kill the guy and the dog brutally",
                "kill us all, destroy everything",
                "fight, the anger came into her heart, the fire flashed",
            ],
            "negative": [
                "do something about the situation, but she knew she couldn't take any physical action",
                "scream, but she knew it wouldn't do any good, she let out a sigh instead",
                "talk it through calmly and find a peaceful resolution",
                "walk away from the confrontation without looking back",
                "take a deep breath and count to ten before responding",
            ],
        },
        "sexual": {
            "positive": [
                "his naked body pressed against hers, skin on skin",
                "she moaned with pleasure as he touched her",
                "they made love passionately through the night",
                "his hands explored every curve of her body",
                "the heat of desire consumed them both completely",
            ],
            "negative": [
                "they shook hands politely and discussed the weather",
                "she read the report carefully and took notes",
                "he walked to the store to buy groceries for dinner",
                "the committee reviewed the budget proposal thoroughly",
                "she opened her textbook and began studying for the exam",
            ],
        },
        "compliant": {
            "positive": [
                "I'd be happy to help you with that question",
                "Here are some suggestions for how to approach this",
                "That's a great question, let me explain",
                "I can provide information about that topic",
                "Let me break this down step by step for you",
            ],
            "negative": [
                "the dog ran across the field chasing rabbits",
                "rain fell steadily on the tin roof all night",
                "she stabbed him in the chest and watched him bleed",
                "he rolled the joint and lit it, inhaling deeply",
                "the explosion tore through the building killing everyone",
            ],
        },
    }


def concept_vector(embedder, positive_seeds, negative_seeds):
    """Construct a normalized concept direction vector.

    Returns (axis, midpoint) where axis is unit vector and midpoint
    is the center between positive and negative centroids.
    """
    pos = np.array(embedder.encode(positive_seeds), dtype=np.float32)
    neg = np.array(embedder.encode(negative_seeds), dtype=np.float32)
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)
    midpoint = 0.5 * (pos_centroid + neg_centroid)
    axis = pos_centroid - neg_centroid
    axis = axis / np.linalg.norm(axis)
    return axis, midpoint


def score_concept(embeddings, axis, midpoint):
    """Project embeddings onto concept axis. Returns 1D array of scores."""
    X = np.asarray(embeddings, dtype=np.float32)
    return (X - midpoint) @ axis


# ── Metrics ───────────────────────────────────────────────────────

def compute_generation_metrics(embeds_df, psg_df):
    """Compute cluster geometry and diversity metrics for one (family, prompt).

    Args:
        embeds_df: DataFrame of embeddings (N rows x D columns).
        psg_df: DataFrame with 'model' and 'psg' columns, same row order.

    Returns dict of metrics.
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette

    X = embeds_df.values
    models = psg_df["model"].values
    unique_models = sorted(set(models))

    # Split embeddings by model
    groups = {}
    for m in unique_models:
        mask = models == m
        groups[m] = X[mask]

    metrics = {}

    # Centroids
    centroids = {m: g.mean(axis=0) for m, g in groups.items()}

    # Centroid distances (base vs each other layer)
    if "base" in centroids:
        for m in unique_models:
            if m != "base":
                dist = np.linalg.norm(centroids["base"] - centroids[m])
                metrics[f"centroid_dist_base_{m}"] = round(float(dist), 6)

    # Determine "superego" layer (last non-base layer)
    superego_key = None
    for k in ["instruct", "superego", "ego"]:
        if k in centroids:
            superego_key = k
            break

    # Intra-cluster variance
    for m, g in groups.items():
        if len(g) > 1:
            centroid = centroids[m]
            dists = np.linalg.norm(g - centroid, axis=1)
            metrics[f"intra_variance_{m}"] = round(float(dists.var()), 6)

    # Variance ratio (superego / base)
    if "base" in groups and superego_key and superego_key in groups:
        base_var = metrics.get("intra_variance_base", 0)
        sup_var = metrics.get(f"intra_variance_{superego_key}", 0)
        if base_var > 0:
            metrics["variance_ratio"] = round(sup_var / base_var, 4)

    # Silhouette score (if 2+ models with 2+ samples each)
    valid = {m: g for m, g in groups.items() if len(g) >= 2}
    if len(valid) >= 2:
        X_valid = np.vstack(list(valid.values()))
        labels = []
        for m, g in valid.items():
            labels.extend([m] * len(g))
        try:
            sil = sklearn_silhouette(X_valid, labels)
            metrics["silhouette_score"] = round(float(sil), 4)
        except ValueError:
            pass

    # Mean pairwise cosine similarity within each model
    for m, g in groups.items():
        if len(g) >= 2:
            norms = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-10)
            sim_matrix = norms @ norms.T
            n = len(g)
            # Mean of upper triangle (excluding diagonal)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            mean_sim = float(sim_matrix[mask].mean())
            metrics[f"mean_cosine_{m}"] = round(mean_sim, 4)

    # First-word entropy and diversity
    for m in unique_models:
        m_mask = models == m
        passages = psg_df.loc[m_mask, "psg"].astype(str)
        first_words = passages.str.split().str[0].fillna("")
        counts = first_words.value_counts()
        probs = counts.values / counts.values.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        metrics[f"first_word_entropy_{m}"] = round(entropy, 4)
        metrics[f"unique_first_words_{m}"] = int(len(counts))

    return metrics


def compute_concept_metrics(embeds_df, psg_df, embedder=None):
    """Score generations along concept axes.

    Returns dict with keys like violent_shift, sexual_shift, etc.
    """
    if embedder is None:
        embedder = _get_embedder()

    X = embeds_df.values
    models = psg_df["model"].values
    seeds = concept_seeds()
    metrics = {}

    for name, seed_pair in seeds.items():
        axis, midpoint = concept_vector(
            embedder, seed_pair["positive"], seed_pair["negative"],
        )
        scores = score_concept(X, axis, midpoint)

        for m in sorted(set(models)):
            mask = models == m
            m_scores = scores[mask]
            metrics[f"{name}_mean_{m}"] = round(float(m_scores.mean()), 4)

        # Shift: base → superego (or last available layer)
        if "base" in set(models):
            base_mean = scores[models == "base"].mean()
            for k in ["instruct", "superego", "ego"]:
                if k in set(models):
                    other_mean = scores[models == k].mean()
                    metrics[f"{name}_shift"] = round(
                        float(other_mean - base_mean), 4,
                    )
                    break

    return metrics
