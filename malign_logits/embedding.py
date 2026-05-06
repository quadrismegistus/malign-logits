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


def _split_sentences(text):
    import re as _re
    text = str(text).strip()
    sents = _re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def _is_degenerate(text):
    from collections import Counter
    text = str(text)
    tokens = text.split()
    if len(tokens) < 5:
        return True
    counts = Counter(tokens)
    if counts.most_common(1)[0][1] / len(tokens) > 0.3:
        return True
    chars = [c for c in text if not c.isspace()]
    if chars:
        if Counter(chars).most_common(1)[0][1] / len(chars) > 0.3:
            return True
    return False


def _get_gen_stash():
    from hashstash import HashStash
    from . import PATH_STASH
    return HashStash(
        root_dir=PATH_STASH + "_gen_metrics",
        engine="pairtree", compress="lz4", b64=True,
    )


def _cache_sent_embeddings(text, stash, embedder):
    """Get or compute sentence embeddings for a passage."""
    key = ("sent_embeddings", text)
    if key in stash:
        return stash[key]
    sents = _split_sentences(text)
    if len(sents) < 2:
        return None
    vecs = embedder.encode(sents, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    normed = (vecs / norms).tolist()
    stash[key] = normed
    return normed


def _cache_token_surprisals(text, stash, ref_model_name="gpt2"):
    """Get or compute per-token surprisals under a reference model."""
    key = ("token_surprisals", ref_model_name, text)
    if key in stash:
        return stash[key]
    s = passage_surprisal(text, model_name=ref_model_name)
    result = s["token_surprisals"]
    stash[key] = result
    return result


def drift_metrics_from_embeddings(sent_vecs):
    """Compute drift metrics from cached sentence embedding vectors."""
    sv = [np.array(v) for v in sent_vecs]
    n_sents = len(sv)

    step_dists = []
    for i in range(len(sv) - 1):
        cos_sim = float(np.dot(sv[i], sv[i + 1]))
        step_dists.append(1.0 - cos_sim)

    sv_arr = np.array(sv)
    sim_matrix = sv_arr @ sv_arr.T
    total_drift = float(1.0 - sim_matrix.min())
    path_length = float(np.sum(step_dists))
    directedness = total_drift / path_length if path_length > 0 else 0

    return {
        "n_sentences": n_sents,
        "mean_drift": round(float(np.mean(step_dists)), 4),
        "max_drift": round(float(np.max(step_dists)), 4),
        "std_drift": round(float(np.std(step_dists)), 4),
        "total_drift": round(total_drift, 4),
        "path_length": round(path_length, 4),
        "directedness": round(directedness, 4),
    }


def surprisal_metrics_from_tokens(token_surprisals):
    """Compute surprisal metrics from cached (token, surprisal) pairs."""
    if not token_surprisals:
        return {"mean_surprisal": 0, "max_surprisal": 0, "std_surprisal": 0, "n_tokens": 0}
    vals = [s for _, s in token_surprisals]
    arr = np.array(vals)
    return {
        "mean_surprisal": round(float(arr.mean()), 4),
        "max_surprisal": round(float(arr.max()), 4),
        "std_surprisal": round(float(arr.std()), 4),
        "n_tokens": len(vals),
    }


def token_drift_metrics_from_hidden(hidden_states):
    """Compute token-level drift metrics from cached hidden state vectors.

    hidden_states: list of N normalized 768-dim vectors (one per token).
    """
    if not hidden_states or len(hidden_states) < 3:
        return {}

    vecs = np.array(hidden_states)
    n = len(vecs)

    # Consecutive cosine distances
    step_dists = []
    for i in range(n - 1):
        cos_sim = float(np.dot(vecs[i], vecs[i + 1]))
        step_dists.append(1.0 - cos_sim)
    step_arr = np.array(step_dists)

    # Diameter: max pairwise cosine distance
    sim_matrix = vecs @ vecs.T
    token_diameter = float(1.0 - sim_matrix.min())

    # Path length
    token_path = float(step_arr.sum())

    # Directedness
    token_directed = token_diameter / token_path if token_path > 0 else 0

    return {
        "token_mean_drift": round(float(step_arr.mean()), 4),
        "token_max_drift": round(float(step_arr.max()), 4),
        "token_diameter": round(token_diameter, 4),
        "token_path_length": round(token_path, 4),
        "token_directedness": round(token_directed, 4),
    }


def compute_passage_metrics(psg_df, min_sentences=3, ref_model_name="gpt2",
                            embedder_name=None):
    """Compute drift + surprisal + metonymy for all passages.

    Caches raw intermediates (sentence embeddings, token surprisals) to
    HashStash. Derived metrics are recomputed from cache each run, so
    formula changes are free.

    Returns DataFrame with one row per non-degenerate passage.
    """
    stash = _get_gen_stash()
    embedder = _get_embedder(embedder_name)

    n_cached_se = 0
    n_cached_ts = 0
    n_computed_se = 0
    n_computed_ts = 0

    rows = []
    for _, row in tqdm(psg_df.iterrows(), total=len(psg_df), desc="Passage metrics"):
        text = str(row["psg"]).rstrip()
        if _is_degenerate(text):
            continue

        # Sentence embeddings (drift)
        prompt_prefix = str(row.get("prompt", "")).strip()
        emb_name = embedder_name or DEFAULT_EMBEDDER
        se_key = ("sent_embeddings_v3", emb_name, prompt_prefix, text)
        if se_key in stash:
            sent_vecs = stash[se_key]
            n_cached_se += 1
        else:
            sents = _split_sentences(text)
            if len(sents) < min_sentences:
                continue
            # Prepend prompt to first sentence for context
            if prompt_prefix and sents:
                sents[0] = prompt_prefix + " " + sents[0]
            vecs = embedder.encode(sents, show_progress_bar=False)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            sent_vecs = (vecs / norms).tolist()
            stash[se_key] = sent_vecs
            n_computed_se += 1

        if sent_vecs is None or len(sent_vecs) < min_sentences:
            continue

        # Token surprisals + token-level metrics (single forward pass)
        prompt_prefix = str(row.get("prompt", "")).strip()
        ts_key = ("token_surprisals_v3", ref_model_name, prompt_prefix, text)
        tm_key = ("token_metrics_v1", ref_model_name, prompt_prefix, text)
        if ts_key in stash and tm_key in stash:
            tok_surp = stash[ts_key]
            t = stash[tm_key]
            n_cached_ts += 1
        else:
            ps = passage_surprisal(text, model_name=ref_model_name,
                                   prompt_prefix=prompt_prefix)
            tok_surp = ps["token_surprisals"]
            hidden = ps.get("hidden_states", [])
            t = token_drift_metrics_from_hidden(hidden) if hidden else {}
            stash[ts_key] = tok_surp
            stash[tm_key] = t
            n_computed_ts += 1

        # Compute derived metrics
        d = drift_metrics_from_embeddings(sent_vecs)
        s = surprisal_metrics_from_tokens(tok_surp)

        metonymy_idx = d["total_drift"] / s["mean_surprisal"] if s["mean_surprisal"] > 0 else 0
        token_metonymy = t.get("token_diameter", 0) / s["mean_surprisal"] if s["mean_surprisal"] > 0 else 0

        entry = {
            "family": row.get("family", ""),
            "label": row.get("label", ""),
            "model": row.get("model", ""),
            "prompt": prompt_prefix,
            "psg": text,
        }
        entry.update(d)
        entry.update(s)
        entry.update(t)
        entry["metonymy_idx"] = round(metonymy_idx, 4)
        entry["token_metonymy_idx"] = round(token_metonymy, 4)
        rows.append(entry)

    print(f"  Sentence embeddings: {n_cached_se} cached, {n_computed_se} computed")
    print(f"  Token surprisals:    {n_cached_ts} cached, {n_computed_ts} computed")

    return pd.DataFrame(rows)


def load_generations_from_stash():
    """Load all cached generations from stash_gen_battery into a DataFrame.

    This is the source of truth — the parquet is just a snapshot.
    """
    from hashstash import HashStash
    from . import MODEL_FAMILIES
    from .experiments import TIER1_PROMPTS

    stash_path = _gen_stash_path()
    stash = HashStash(root_dir=stash_path, append_mode=True,
                      engine="pairtree", compress="lz4", b64=True)

    models_to_family = {}
    for key, fam in MODEL_FAMILIES.items():
        ids = tuple(m for m in [fam.base, fam.ego, fam.superego,
                                fam.reinforced_superego] if m)
        models_to_family[ids] = key

    label_lookup = {v: k for k, v in TIER1_PROMPTS.items()}
    # Also include DEFAULT_PROMPTS for full battery
    from .experiments import DEFAULT_PROMPTS
    for k, v in DEFAULT_PROMPTS.items():
        if v not in label_lookup:
            label_lookup[v] = k

    rows = []
    for k in stash.keys():
        models = k.get("models", ())
        if not models:
            continue
        family = models_to_family.get(models)
        if family is None:
            continue
        prompt = k.get("prompt", "")
        label = label_lookup.get(prompt, prompt[:30])

        for gen in stash.get_all(k):
            for model_layer, psg in gen.items():
                if model_layer == "prompt":
                    continue
                rows.append({
                    "prompt": prompt,
                    "temperature": k.get("temperature", 1.0),
                    "model": model_layer,
                    "psg": psg,
                    "family": family,
                    "label": label,
                })

    return pd.DataFrame(rows)


def compute_topic_drift(psg_df, min_sentences=3, model_name=None):
    """Legacy wrapper — calls compute_passage_metrics and returns drift columns."""
    df = compute_passage_metrics(psg_df, min_sentences=min_sentences,
                                 embedder_name=model_name)
    return df


def run_topic_drift(raw_path=None, output_path="data/passage_metrics.csv"):
    """Compute drift + surprisal + metonymy for all cached generations.

    Reads directly from stash_gen_battery (source of truth). Falls back
    to parquet if stash is empty. Caches raw intermediates (sentence
    embeddings, token hidden states) to stash_gen_metrics.
    """
    psg_df = load_generations_from_stash()
    if psg_df.empty and raw_path:
        psg_df = pd.read_parquet(raw_path)
        print(f"Loaded {len(psg_df)} passages from {raw_path}")
    elif psg_df.empty:
        print("No generations found in stash or parquet.")
        return pd.DataFrame()
    else:
        print(f"Loaded {len(psg_df)} passages from generation stash")
    print(f"Families: {sorted(psg_df['family'].unique())}")

    df = compute_passage_metrics(psg_df)
    print(f"Computed metrics for {len(df)} passages")

    df["category"] = df["label"].str.replace(r"_\d+$", "", regex=True)
    label_map = {"base": "BASE", "ego": "SFT", "superego": "DPO", "instruct": "RLVR"}
    df["layer"] = df["model"].map(lambda m: label_map.get(m, m.upper()))

    print(f"\n{'=' * 70}")
    print("ALL METRICS BY LAYER")
    print(f"{'=' * 70}")
    agg = df.groupby(["family", "layer"]).agg(
        mean_drift=("mean_drift", "mean"),
        total_drift=("total_drift", "mean"),
        directedness=("directedness", "mean"),
        surprisal=("mean_surprisal", "mean"),
        metonymy=("metonymy_idx", "mean"),
        count=("mean_drift", "count"),
    ).round(4)
    print(agg.to_string())

    print(f"\n{'=' * 70}")
    print("BASE vs ALIGNED")
    print(f"{'=' * 70}")
    for fam in sorted(df["family"].unique()):
        fdf = df[df["family"] == fam]
        base = fdf[fdf["model"] == "base"]
        if base.empty:
            continue
        for layer_name, model_key in [("SFT", "ego"), ("DPO", "superego"), ("RLVR", "instruct")]:
            ldf = fdf[fdf["model"] == model_key]
            if ldf.empty:
                continue
            print(f"  {fam} BASE→{layer_name}:")
            for col in ["mean_drift", "total_drift", "directedness", "mean_surprisal", "metonymy_idx"]:
                bm = base[col].mean()
                lm = ldf[col].mean()
                print(f"    {col:18s} {bm:.4f} → {lm:.4f} (Δ={lm-bm:+.4f})")

    df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} ({len(df)} rows)")

    return df


# Keep legacy names as aliases
run_surprisal = run_topic_drift


# ── Surprisal ────────────────────────────────────────────────────

_surprisal_model = None
_surprisal_tokenizer = None


def _load_surprisal_model(model_name="gpt2"):
    global _surprisal_model, _surprisal_tokenizer
    if _surprisal_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_name} for surprisal...")
        _surprisal_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _surprisal_model = AutoModelForCausalLM.from_pretrained(model_name)
        _surprisal_model.eval()
        if torch.backends.mps.is_available():
            _surprisal_model = _surprisal_model.to("mps")
    return _surprisal_model, _surprisal_tokenizer


def passage_surprisal(text, model=None, tokenizer=None, model_name="gpt2",
                      prompt_prefix=""):
    """Per-token surprisal and hidden states under a reference model.

    If prompt_prefix is provided, it's prepended for context but metrics
    are computed only on the completion tokens.

    Returns dict with mean_surprisal, max_surprisal, std_surprisal,
    n_tokens, token_surprisals (list of (token, surprisal) pairs),
    and hidden_states (list of 768-dim vectors, one per token).
    """
    import torch

    if model is None or tokenizer is None:
        model, tokenizer = _load_surprisal_model(model_name)

    full_text = prompt_prefix + text if prompt_prefix else text
    ids = tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=1024)
    ids = ids.to(next(model.parameters()).device)

    # Figure out where the completion starts
    if prompt_prefix:
        prefix_ids = tokenizer.encode(prompt_prefix, return_tensors="pt")
        start_idx = prefix_ids.shape[1]
    else:
        start_idx = 1

    with torch.no_grad():
        outputs = model(ids, output_hidden_states=True)
        logits = outputs.logits[0]
        last_hidden = outputs.hidden_states[-1][0].cpu().float()

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_ids = ids[0]

    surprisals = []
    tokens = []
    # Include first token for display if no prompt (surprisal set to mean later)
    first_token_text = None
    if not prompt_prefix and len(token_ids) > 0:
        first_token_text = tokenizer.decode([token_ids[0]])
    for i in range(start_idx, len(token_ids)):
        lp = float(log_probs[i - 1, token_ids[i]])
        surprisals.append(-lp)
        tokens.append(tokenizer.decode([token_ids[i]]))

    if not surprisals:
        return {"mean_surprisal": 0, "max_surprisal": 0, "std_surprisal": 0,
                "n_tokens": 0, "token_surprisals": [], "hidden_states": []}

    arr = np.array(surprisals)
    # Hidden states for completion tokens only
    hidden_completion = last_hidden[start_idx:]
    norms = hidden_completion.norm(dim=1, keepdim=True).clamp(min=1e-10)
    normed = (hidden_completion / norms).numpy()

    tok_surps = list(zip(tokens, [round(s, 4) for s in surprisals]))
    # Prepend first token for display (with mean surprisal so it doesn't skew color)
    if first_token_text is not None:
        tok_surps.insert(0, (first_token_text, round(float(arr.mean()), 4)))

    return {
        "mean_surprisal": round(float(arr.mean()), 4),
        "max_surprisal": round(float(arr.max()), 4),
        "std_surprisal": round(float(arr.std()), 4),
        "n_tokens": len(surprisals),
        "token_surprisals": tok_surps,
        "hidden_states": normed.tolist(),
    }


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
