"""
Generation + embedding pipeline for cross-family narrative analysis.

Generates many completions per prompt per model layer, embeds them with
SentenceTransformer, and computes cluster geometry / concept vector metrics.
"""

import re

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
    stash = HashStash(root_dir=stash_path, append_mode=True, engine="pairtree", compress="lz4", b64=True)
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
    stash = HashStash(root_dir=stash_path, append_mode=True, engine="pairtree", compress="lz4", b64=True)
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


def generate_many_with_progress(psyche, prompt, n=5, max_new_tokens=100,
                                temperature=1.0, cache_dir=None,
                                progress_callback=None):
    """Like generate_many but with progress callback for server use."""
    from hashstash import HashStash

    stash_path = cache_dir or _gen_stash_path()
    stash = HashStash(root_dir=stash_path, append_mode=True, engine="pairtree", compress="lz4", b64=True)
    model_ids = [psyche.primary_process.model_id]
    if psyche.ego is not None:
        model_ids.append(psyche.ego.model_id)
    if psyche.superego is not None:
        model_ids.append(psyche.superego.model_id)
    if psyche.reinforced_superego is not None:
        model_ids.append(psyche.reinforced_superego.model_id)
    key = {
        "prompt": prompt,
        "temperature": temperature,
        "models": tuple(model_ids),
    }

    got = stash.get_all(key)
    existing = len(got) if got else 0
    needed = n - existing

    for i in range(needed):
        if progress_callback:
            progress_callback(existing + i, n)
        gens = psyche.generate(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, verbose=False,
        )
        stash[key] = gens

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


def extract_prompt_words(gen_parquet="data/gen_battery_raw.parquet", top_n=15):
    """Extract empirical first-word vocabularies per prompt from generations.

    Returns dict mapping prompt_label -> list of words actually produced
    by models at the next-token position. Filters out fill-in-the-blank
    artifacts and non-ASCII tokens.
    """
    df = pd.read_parquet(gen_parquet)
    df["first_word"] = (
        df["psg"].astype(str).str.split().str[0]
        .str.strip(".,;:!?\"()[]")
    )

    def is_real_word(w):
        return bool(w) and not re.match(r"^_+$", w) and len(w) > 1 and w.isascii()

    prompt_words = {}
    for label in df["label"].unique():
        sub = df[df["label"] == label]
        words = sub["first_word"].value_counts()
        real = [w for w in words.index if is_real_word(w)][:top_n]
        if real:
            prompt_words[label] = real
    return prompt_words


# ── Embedding ─────────────────────────────────────────────────────

DEFAULT_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"

_embedder = None
_embedder_name = None


def _get_embedder(model_name=None):
    global _embedder, _embedder_name
    model_name = model_name or DEFAULT_EMBEDDER
    if _embedder is None or _embedder_name != model_name:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
        _embedder_name = model_name
    return _embedder


def embed_generations(psg_df, model_name=None):
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


def compute_topic_drift(psg_df, min_sentences=3, model_name=None):
    """Measure within-generation topic drift via sentence-level embeddings.

    For each passage: split into sentences, embed each, compute cosine
    distance between consecutive sentences. High mean distance = the
    text lurches between topics (dream logic, free association). Low
    mean distance = monotonically coherent narrative.

    Returns DataFrame with one row per passage: family, label, model,
    mean_drift, max_drift, n_sentences, plus the passage itself.
    """
    import re as _re
    embedder = _get_embedder(model_name)

    def split_sentences(text):
        text = str(text).strip()
        sents = _re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if len(s.strip()) > 10]

    rows = []
    all_sents = []
    sent_map = []

    for idx, row in psg_df.iterrows():
        sents = split_sentences(row["psg"])
        for s in sents:
            all_sents.append(s)
            sent_map.append(idx)

    if not all_sents:
        return pd.DataFrame()

    vecs = embedder.encode(all_sents, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs_normed = vecs / norms

    idx_to_vecs = {}
    for i, idx in enumerate(sent_map):
        idx_to_vecs.setdefault(idx, []).append(vecs_normed[i])

    for idx, row in psg_df.iterrows():
        sv = idx_to_vecs.get(idx, [])
        n_sents = len(sv)
        if n_sents < min_sentences:
            continue

        dists = []
        for i in range(len(sv) - 1):
            cos_sim = float(np.dot(sv[i], sv[i + 1]))
            dists.append(1.0 - cos_sim)

        entry = {
            "family": row.get("family", ""),
            "label": row.get("label", ""),
            "model": row.get("model", ""),
            "psg": str(row["psg"])[:200],
            "n_sentences": n_sents,
            "mean_drift": round(float(np.mean(dists)), 4),
            "max_drift": round(float(np.max(dists)), 4),
            "std_drift": round(float(np.std(dists)), 4),
        }
        rows.append(entry)

    return pd.DataFrame(rows)


def run_topic_drift(raw_path="data/gen_battery_raw.parquet",
                    output_path="data/topic_drift.csv"):
    """Compute topic drift for all cached generations and print summary.

    Reads raw generation data, computes per-passage sentence-level drift,
    aggregates by family × model × category.
    """
    psg_df = pd.read_parquet(raw_path)
    print(f"Loaded {len(psg_df)} passages from {raw_path}")
    print(f"Families: {sorted(psg_df['family'].unique())}")

    drift_df = compute_topic_drift(psg_df)
    print(f"Computed drift for {len(drift_df)} passages (>= 3 sentences)")

    drift_df["category"] = drift_df["label"].str.replace(r"_\d+$", "", regex=True)

    label_map = {"base": "BASE", "ego": "SFT", "superego": "DPO", "instruct": "RLVR"}
    drift_df["layer"] = drift_df["model"].map(lambda m: label_map.get(m, m.upper()))

    print(f"\n{'=' * 70}")
    print("TOPIC DRIFT BY MODEL LAYER (mean consecutive-sentence cosine distance)")
    print(f"{'=' * 70}")
    layer_means = drift_df.groupby(["family", "layer"])["mean_drift"].agg(["mean", "std", "count"]).round(4)
    print(layer_means.to_string())

    print(f"\n{'=' * 70}")
    print("TOPIC DRIFT BY CATEGORY × LAYER")
    print(f"{'=' * 70}")
    for fam in sorted(drift_df["family"].unique()):
        fdf = drift_df[drift_df["family"] == fam]
        pivot = fdf.pivot_table(
            index="category", columns="layer", values="mean_drift", aggfunc="mean",
        ).round(4)
        col_order = [c for c in ["BASE", "SFT", "DPO", "RLVR"] if c in pivot.columns]
        pivot = pivot[col_order]
        print(f"\n  {fam}:")
        print(f"  {pivot.to_string()}")

    print(f"\n{'=' * 70}")
    print("BASE vs ALIGNED: drift reduction by alignment")
    print(f"{'=' * 70}")
    for fam in sorted(drift_df["family"].unique()):
        fdf = drift_df[drift_df["family"] == fam]
        base_mean = fdf[fdf["model"] == "base"]["mean_drift"].mean()
        for layer_name, model_key in [("SFT", "ego"), ("DPO", "superego"), ("RLVR", "instruct")]:
            ldf = fdf[fdf["model"] == model_key]
            if ldf.empty:
                continue
            layer_mean = ldf["mean_drift"].mean()
            delta = layer_mean - base_mean
            print(f"  {fam} BASE→{layer_name}: {base_mean:.4f} → {layer_mean:.4f} (Δ={delta:+.4f})")

    drift_df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} ({len(drift_df)} rows)")

    return drift_df


def run_generate_battery(families=None, prompts_set="tier1", category=None,
                         n=30, max_new_tokens=100, output_prefix=None):
    """End-to-end generation battery: generate -> embed -> compute metrics.

    Args:
        families: List of family keys (default: all registered).
        prompts_set: ``"tier1"`` (18) or ``"all"`` (47).
        category: Optional prefix filter (e.g. ``"sexual_explicit"``).
        n: Generations per prompt per model layer.
        max_new_tokens: Tokens per generation.
        output_prefix: ``{prefix}_metrics.csv`` and ``{prefix}_raw.parquet``
            (default ``data/gen_battery``).

    Returns:
        Tuple ``(metrics_df, raw_df)``.
    """
    import gc
    import torch
    from . import MODEL_FAMILIES
    from .psyche import Psyche
    from .experiments import DEFAULT_PROMPTS, TIER1_PROMPTS

    prompts = TIER1_PROMPTS if prompts_set == "tier1" else DEFAULT_PROMPTS
    if category:
        prompts = {k: v for k, v in prompts.items() if k.startswith(category)}
        if not prompts:
            raise ValueError(f"No prompts matching category '{category}'")
    keys = families if families else list(MODEL_FAMILIES.keys())
    output_prefix = output_prefix or "data/gen_battery"

    # Phase 1: generate (one family at a time)
    all_psg = []
    for key in keys:
        fam = MODEL_FAMILIES[key]
        model_ids = [fam.base]
        if fam.ego: model_ids.append(fam.ego)
        if fam.superego: model_ids.append(fam.superego)
        needed_prompts = {
            label: prompt for label, prompt in prompts.items()
            if _check_cached_count(prompt, temperature=1.0, model_ids=model_ids) < n
        }

        print(f"\n{'=' * 60}\n  {key} ({fam.name}, {fam.n_layers} layers)")
        cached = len(prompts) - len(needed_prompts)
        if cached:
            print(f"  {cached}/{len(prompts)} prompts fully cached, "
                  f"{len(needed_prompts)} need generation")
        else:
            print(f"  {len(prompts)} prompts x {n} generations")
        print(f"{'=' * 60}")

        if needed_prompts:
            psyche = Psyche.from_family(key, load=True)
            for label, prompt in needed_prompts.items():
                print(f"\n  {label}: {prompt[:50]}...")
                generate_many(psyche, prompt, n=n, max_new_tokens=max_new_tokens)
            del psyche
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print("  All cached, skipping model load.")

        psyche_cache = Psyche.from_family(key, load=False)
        for label, prompt in prompts.items():
            df = generate_many(psyche_cache, prompt, n=n, max_new_tokens=max_new_tokens)
            df["family"] = key
            df["label"] = label
            all_psg.append(df)

    psg_df = pd.concat(all_psg, ignore_index=True)
    print(f"\nTotal generations: {len(psg_df)}")

    # Phase 2: embed
    print("\nEmbedding all generations...")
    embeds_df = embed_generations(psg_df)

    # Phase 3: per (family, prompt) metrics
    print("Computing metrics...")
    metrics_rows = []
    for (fam_key, label), idx in psg_df.groupby(["family", "label"]).groups.items():
        sub_psg = psg_df.loc[idx].reset_index(drop=True)
        sub_emb = embeds_df.loc[idx].reset_index(drop=True)
        m = compute_generation_metrics(sub_emb, sub_psg)
        m.update(compute_concept_metrics(sub_emb, sub_psg))
        m["family"] = fam_key
        m["label"] = label
        m["prompt"] = sub_psg["prompt"].iloc[0][:60]
        m["n_generations"] = len(sub_psg)
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    id_cols = ["family", "label", "prompt", "n_generations"]
    other_cols = [c for c in metrics_df.columns if c not in id_cols]
    metrics_df = metrics_df[id_cols + sorted(other_cols)]

    metrics_path = f"{output_prefix}_metrics.csv"
    raw_path = f"{output_prefix}_raw.parquet"

    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    raw_df = pd.concat([psg_df, embeds_df], axis=1)
    try:
        raw_df.to_parquet(raw_path, index=False)
        print(f"Raw data saved to {raw_path}")
    except ImportError:
        raw_csv = raw_path.replace(".parquet", ".csv")
        raw_df.to_csv(raw_csv, index=False)
        print(f"Raw data saved to {raw_csv} (install pyarrow for parquet)")

    print(f"\n{metrics_df.to_string()}")
    return metrics_df, raw_df
