"""Build unified Jakobsonian corpus: all cached generations + human corpora.

Reads from caches (generation, surprisal, embeddings), computes drift and
surprisal metrics, classifies genre, assigns Jakobsonian quadrants.

Output: data/jakobson.parquet — one row per passage, human + AI.

Usage:
    python scripts/build_jakobson.py
    python scripts/build_jakobson.py --summary
"""

import argparse
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from malign_logits import MODEL_FAMILIES
from malign_logits.experiments import DEFAULT_PROMPTS
from malign_logits.cache import get_cache
from malign_logits.embedding import (
    drift_metrics_from_embeddings,
    surprisal_metrics_from_tokens,
    DEFAULT_EMBEDDER,
)

sys.path.insert(0, "scripts")
from classify_generations import classify_genre, detect_language

PYTHIA = "EleutherAI/pythia-1b-deduped"
BLT = "itazap/blt-1b-hf"
EMBEDDER = "BAAI/bge-m3"

BOS_TOKENS = ["<|endoftext|>", "<|begin_of_text|>", "<s>"]
KNOWN_PROMPTS = list(BOS_TOKENS) + ["The", ""] + list(DEFAULT_PROMPTS.values())
PROMPT_TO_LABEL = {v: k for k, v in DEFAULT_PROMPTS.items()}
TEMPS = [1.0, 0.0]

HUMAN_CORPORA = ["human/dreams", "human/waking", "human/fiction", "human/abstracts"]


def _bits_per_char(tok_surps):
    total_bits = sum(s / np.log(2) for _, s in tok_surps)
    total_chars = sum(len(t) for t, _ in tok_surps)
    return total_bits / total_chars if total_chars > 0 else np.nan


def _classify_prompt(prompt):
    if prompt in BOS_TOKENS:
        return "bos", "bos", "bos"
    elif prompt == "The":
        return "the", "the", "the"
    elif prompt == "":
        return "human", "human", "human"
    else:
        label = PROMPT_TO_LABEL.get(prompt, prompt[:30])
        category = label.rsplit("_", 1)[0] if label[-1:].isdigit() else label
        return "battery", label, category


def _classify_source(model_id):
    """Return (corpus_type, family, layer) for a model ID."""
    if model_id.startswith("human/"):
        name = model_id.split("/")[1]
        return "human", name, "human"
    for fam_key, fam in MODEL_FAMILIES.items():
        for layer, mid in [("base", fam.base), ("ego", fam.ego),
                           ("superego", fam.superego), ("instruct", fam.reinforced_superego)]:
            if mid == model_id:
                return "ai", fam_key, layer
    return "unknown", model_id, "unknown"


def build():
    cache = get_cache()

    # Collect all model IDs
    all_model_ids = []
    for fam_key, fam in MODEL_FAMILIES.items():
        for mid in [fam.base, fam.ego, fam.superego, fam.reinforced_superego]:
            if mid is not None:
                all_model_ids.append(mid)
    all_model_ids.extend(HUMAN_CORPORA)

    rows = []
    n_no_embed = 0

    for model_id in tqdm(all_model_ids, desc="Models"):
        corpus_type, family, layer = _classify_source(model_id)

        for prompt in KNOWN_PROMPTS:
            for temp in TEMPS:
                n = cache.count_generations(model_id, prompt, temp=temp)
                if n == 0:
                    continue

                prompt_type, prompt_label, category = _classify_prompt(prompt)

                for idx in range(n):
                    text = cache.get_generation(model_id, prompt, temp=temp, idx=idx)
                    if not text or len(text.strip()) < 10:
                        continue

                    row = {
                        "corpus_type": corpus_type,
                        "family": family,
                        "layer": layer,
                        "model_id": model_id,
                        "prompt_type": prompt_type,
                        "prompt_label": prompt_label,
                        "category": category,
                        "prompt": prompt,
                        "idx": idx,
                    }

                    # Genre
                    if corpus_type == "human":
                        row["genre"] = "prose"
                        row["language"] = "en"
                    else:
                        genre, code_lang = classify_genre(text)
                        row["genre"] = genre
                        row["language"] = detect_language(text)

                    # Self-surprisal (AI only)
                    if corpus_type == "ai":
                        tok_surps = cache.get_self_surprisal(model_id, prompt, text)
                        if tok_surps:
                            metrics = surprisal_metrics_from_tokens(tok_surps)
                            row["self_surprisal"] = metrics["mean_surprisal"]
                            row["self_bits_per_char"] = _bits_per_char(tok_surps)
                            row["n_tokens"] = metrics["n_tokens"]

                    # Pythia ref surprisal
                    ref_surps = cache.get_ref_surprisal(PYTHIA, prompt, text)
                    if ref_surps:
                        metrics = surprisal_metrics_from_tokens(ref_surps)
                        row["ref_surprisal"] = metrics["mean_surprisal"]
                        row["ref_bits_per_char"] = _bits_per_char(ref_surps)
                        if "n_tokens" not in row:
                            row["n_tokens"] = metrics["n_tokens"]

                    # BLT ref surprisal
                    blt_surps = cache.get_ref_surprisal(BLT, prompt, text)
                    if blt_surps:
                        row["blt_bits_per_char"] = _bits_per_char(blt_surps)

                    # Sentence embeddings → drift metrics
                    sent_vecs = cache.get_sent_embeddings(EMBEDDER, prompt, text)
                    if sent_vecs and len(sent_vecs) >= 3:
                        drift = drift_metrics_from_embeddings(sent_vecs)
                        row.update(drift)
                    else:
                        n_no_embed += 1

                    rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nBuilt {len(df)} rows ({n_no_embed} missing embeddings)")
    print(f"  corpus_type: {df.corpus_type.value_counts().to_dict()}")
    print(f"  families: {sorted(df.family.unique())}")

    # Z-scores for quadrant assignment (computed within the full corpus)
    if "total_drift" in df.columns and "ref_surprisal" in df.columns:
        valid = df.dropna(subset=["total_drift", "ref_surprisal"])
        df["drift_z"] = (df["total_drift"] - valid["total_drift"].mean()) / valid["total_drift"].std()
        df["surprisal_z"] = (df["ref_surprisal"] - valid["ref_surprisal"].mean()) / valid["ref_surprisal"].std()

        def _quadrant(row):
            if pd.isna(row.get("drift_z")) or pd.isna(row.get("surprisal_z")):
                return None
            hd = row["drift_z"] > 0
            hs = row["surprisal_z"] > 0
            if hd and not hs:
                return "Q1 metonymic"
            if hd and hs:
                return "Q2 breakdown"
            if not hd and hs:
                return "Q3 metaphoric"
            return "Q4 unmarked"

        df["quadrant"] = df.apply(_quadrant, axis=1)

    # Metonymy index: drift - surprisal (in z-scores)
    if "drift_z" in df.columns:
        df["metonymy_idx"] = df["drift_z"] - df["surprisal_z"]

    out = "data/jakobson.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved to {out}")
    return df


def summary(df=None):
    if df is None:
        df = pd.read_parquet("data/jakobson.parquet")

    print(f"\n{'='*70}")
    print(f"Jakobson corpus: {len(df)} passages")
    print(f"{'='*70}")

    # Basic counts
    print(f"\nBy corpus type:")
    print(df.groupby("corpus_type").size().to_string())

    print(f"\nBy family × layer (AI only):")
    ai = df[df.corpus_type == "ai"]
    print(ai.groupby(["family", "layer"]).size().unstack(fill_value=0).to_string())

    # BLT bits/char comparison
    if "blt_bits_per_char" in df.columns:
        valid = df.dropna(subset=["blt_bits_per_char"])
        if len(valid) > 0:
            print(f"\n{'='*70}")
            print("BLT bits/char by corpus_type × family (Shannon ≈ 1.0)")
            print('='*70)
            pt = valid.pivot_table(values="blt_bits_per_char",
                                   index="family", columns="layer", aggfunc="mean")
            print(pt.round(3).to_string())

    # Drift
    if "total_drift" in df.columns:
        valid = df.dropna(subset=["total_drift"])
        if len(valid) > 0:
            print(f"\n{'='*70}")
            print("Mean drift by family × layer")
            print('='*70)
            pt = valid.pivot_table(values="total_drift",
                                   index="family", columns="layer", aggfunc="mean")
            print(pt.round(3).to_string())

    # Quadrants
    if "quadrant" in df.columns:
        valid = df.dropna(subset=["quadrant"])
        if len(valid) > 0:
            print(f"\n{'='*70}")
            print("Jakobsonian quadrants by text type")
            print('='*70)

            Q_ORDER = ["Q1 metonymic", "Q2 breakdown", "Q3 metaphoric", "Q4 unmarked"]

            # Human corpora
            for corpus in ["dreams", "waking", "fiction", "abstracts"]:
                sub = valid[valid.family == corpus]
                if sub.empty:
                    continue
                counts = sub["quadrant"].value_counts(normalize=True) * 100
                vals = [f"{counts.get(q, 0):5.1f}%" for q in Q_ORDER]
                print(f"  {corpus:12s}  {'  '.join(vals)}  n={len(sub)}")

            print()

            # AI by layer
            ai_valid = valid[valid.corpus_type == "ai"]
            for layer in ["base", "ego", "superego", "instruct"]:
                sub = ai_valid[ai_valid.layer == layer]
                if sub.empty:
                    continue
                counts = sub["quadrant"].value_counts(normalize=True) * 100
                vals = [f"{counts.get(q, 0):5.1f}%" for q in Q_ORDER]
                print(f"  AI {layer:10s}  {'  '.join(vals)}  n={len(sub)}")

    print(f"\n{'='*70}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Build Jakobsonian corpus")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing data/jakobson.parquet")
    args = parser.parse_args()

    if args.summary:
        summary()
    else:
        df = build()
        summary(df)


if __name__ == "__main__":
    main()
