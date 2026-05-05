"""Compute passage metrics across all corpora with length normalization.

Loads model generations, dream reports, waking narratives (Hippocorpus),
and C20 fiction. Truncates each passage to the minimum number of sentences
needed to exceed 100 words, so all corpora are compared at matched length.

Outputs a single CSV with corpus/family/model columns for unified analysis.

Usage:
    python scripts/corpus_metrics.py [--min-words 100] [--output data/corpus_metrics.csv]
"""
import argparse
import json

import nltk
import pandas as pd

from malign_logits.embedding import compute_passage_metrics, load_generations_from_stash


def truncate_to_min_sentences(text, min_words=100):
    """Return the fewest complete sentences that exceed min_words.

    Uses NLTK sentence tokenizer. Returns None if the text can't reach
    min_words even with all sentences included.
    """
    sents = nltk.sent_tokenize(str(text))
    words = 0
    kept = []
    for s in sents:
        kept.append(s)
        words += len(s.split())
        if words >= min_words:
            return " ".join(kept), words, len(kept)
    if words >= min_words:
        return " ".join(kept), words, len(kept)
    return None, words, len(kept)


def load_model_generations():
    """Load from generation stash, truncate each passage."""
    raw = load_generations_from_stash()
    if raw.empty:
        raw = pd.read_parquet("data/gen_battery_raw.parquet")
    rows = []
    for _, r in raw.iterrows():
        rows.append({
            "corpus": r["family"],
            "subcorpus": r["model"],
            "prompt": str(r.get("prompt", "")),
            "label": r.get("label", ""),
            "text": str(r["psg"]),
        })
    return pd.DataFrame(rows)


def load_dreams():
    """Load cleaned dream reports."""
    path = "data/dreams_sample_500_cleaned.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.read_csv("data/dreams_sample_500.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "corpus": "dreams",
            "subcorpus": "dream",
            "prompt": "",
            "label": "dream",
            "text": str(r["text"]),
        })
    return pd.DataFrame(rows)


def load_hippocorpus():
    """Load waking narrative sample."""
    df = pd.read_csv("data/hippocorpus_sample_500.csv")
    col = "story" if "story" in df.columns else "text"
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "corpus": "waking",
            "subcorpus": "recalled",
            "prompt": "",
            "label": "waking",
            "text": str(r[col]),
        })
    return pd.DataFrame(rows)


def load_fiction():
    """Load C20 fiction narration passages."""
    rows = []
    with open("data/markmark_c20_narration_500.jsonl") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "corpus": "c20_fiction",
                "subcorpus": "narration",
                "prompt": "",
                "label": "fiction",
                "text": d["text"],
            })
    return pd.DataFrame(rows)


def load_abstracts():
    """Load arxiv abstracts."""
    df = pd.read_csv("data/arxiv_abstracts_500.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "corpus": "abstracts",
            "subcorpus": "arxiv",
            "prompt": "",
            "label": "abstract",
            "text": str(r["text"]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-words", type=int, default=75,
                        help="Minimum words per truncated passage (default: 75)")
    parser.add_argument("--output", "-o", default="data/corpus_metrics.csv")
    args = parser.parse_args()

    print("Loading corpora...")
    frames = []
    for name, loader in [
        ("model_generations", load_model_generations),
        ("dreams", load_dreams),
        ("hippocorpus", load_hippocorpus),
        ("c20_fiction", load_fiction),
        ("abstracts", load_abstracts),
    ]:
        try:
            df = loader()
            print(f"  {name}: {len(df)} passages")
            frames.append(df)
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal before truncation: {len(all_df)}")

    # Truncate each passage
    print(f"Truncating to min {args.min_words} words at sentence boundary...")
    trunc_rows = []
    skipped = 0
    for _, r in all_df.iterrows():
        result, n_words, n_sents = truncate_to_min_sentences(r["text"], args.min_words)
        if result is None:
            skipped += 1
            continue
        trunc_rows.append({
            "corpus": r["corpus"],
            "subcorpus": r["subcorpus"],
            "prompt": r["prompt"],
            "label": r["label"],
            "text": result,
            "n_words_truncated": n_words,
            "n_sents_truncated": n_sents,
        })
    trunc_df = pd.DataFrame(trunc_rows)
    print(f"After truncation: {len(trunc_df)} passages ({skipped} skipped < {args.min_words} words)")

    # Word count stats by corpus
    print(f"\nWord counts after truncation:")
    for corpus in sorted(trunc_df["corpus"].unique()):
        sub = trunc_df[trunc_df["corpus"] == corpus]
        wc = sub["n_words_truncated"]
        print(f"  {corpus:15s}  n={len(sub):5d}  words: {wc.mean():.0f} ± {wc.std():.0f}  (min={wc.min()}, max={wc.max()})")

    # Build DataFrame for compute_passage_metrics
    psg_df = pd.DataFrame({
        "prompt": trunc_df["prompt"],
        "model": trunc_df["subcorpus"],
        "psg": trunc_df["text"],
        "family": trunc_df["corpus"],
        "label": trunc_df["label"],
    })

    # Compute metrics
    result = compute_passage_metrics(psg_df, min_sentences=3)
    print(f"\nComputed metrics for {len(result)} passages")

    # Add back corpus metadata
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output}")

    # Summary comparison
    print(f"\n{'=' * 90}")
    print("SUMMARY BY CORPUS")
    print(f"{'=' * 90}")
    print(f"{'corpus':15s} {'subcorpus':12s} {'drift':>8s} {'surprisal':>10s} {'directed':>10s} {'metonymy':>10s} {'n':>6s}")
    print("-" * 75)

    for corpus in sorted(result["family"].unique()):
        for sub in sorted(result[result["family"] == corpus]["model"].unique()):
            s = result[(result["family"] == corpus) & (result["model"] == sub)]
            print(f"{corpus:15s} {sub:12s} {s.total_drift.mean():8.3f} {s.mean_surprisal.mean():10.3f} {s.directedness.mean():10.3f} {s.metonymy_idx.mean():10.3f} {len(s):6d}")

    # Model-only z-scores
    models = result[~result["family"].isin(["dreams", "waking", "c20_fiction"])]
    if not models.empty:
        print(f"\n{'=' * 90}")
        print("Z-SCORES (relative to model generation distribution)")
        print(f"{'=' * 90}")
        for corpus in ["dreams", "waking", "c20_fiction"]:
            sub = result[result["family"] == corpus]
            if sub.empty:
                continue
            zs = []
            for col in ["total_drift", "mean_surprisal", "directedness", "metonymy_idx"]:
                m, s = models[col].mean(), models[col].std()
                z = (sub[col].mean() - m) / s if s > 0 else 0
                zs.append(f"{col.replace('total_drift','drift').replace('mean_surprisal','surp').replace('directedness','dir').replace('metonymy_idx','met')}={z:+.2f}σ")
            print(f"  {corpus:15s}  {'  '.join(zs)}")


if __name__ == "__main__":
    main()
