"""Cross-generation MMD: measure how much alignment changes what gets generated.

For each (family, prompt), embeds BASE and ALIGNED completions, then computes
MMD² (maximum mean discrepancy) between the two clouds. A split-half of the
BASE completions provides the null distribution.

Reads directly from stash_gen_metrics (cached sentence embeddings) and
stash_gen_battery (generation metadata). No CSV dependency.

Usage:
    python scripts/cross_generation_mmd.py
    python scripts/cross_generation_mmd.py --summary   # reprint from CSV
    python scripts/cross_generation_mmd.py --n-perm 1000
"""

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, cdist
from scipy import stats


EMBEDDERS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-m3",
]


def truncate_to_min_sentences(text, min_words=75):
    """Match corpus_metrics.py truncation logic."""
    import nltk
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sents = nltk.sent_tokenize(text)
    words = 0
    for i, s in enumerate(sents):
        words += len(s.split())
        if words >= min_words:
            result = " ".join(sents[: i + 1])
            return result, words, i + 1
    return None, 0, 0


def mmd2_permtest(X, Y, n_perm=500):
    """MMD² with RBF kernel (median heuristic) and permutation test."""
    XY = np.vstack([X, Y])
    sigma = np.median(pdist(XY, "euclidean"))
    if sigma < 1e-10:
        return 0.0, 1.0
    gamma = 1.0 / (2 * sigma ** 2)
    n, m = len(X), len(Y)
    K = np.exp(-gamma * cdist(XY, XY, "sqeuclidean"))

    def _mmd2(K, n, m):
        Kxx = K[:n, :n].copy()
        Kyy = K[n:, n:].copy()
        Kxy = K[:n, n:]
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)
        return (Kxx.sum() / (n * (n - 1))
                + Kyy.sum() / (m * (m - 1))
                - 2 * Kxy.sum() / (n * m))

    obs = _mmd2(K, n, m)
    rng = np.random.default_rng(42)
    exceed = sum(
        1 for _ in range(n_perm)
        if _mmd2(K[np.ix_(p := rng.permutation(n + m), p)], n, m) >= obs
    )
    return obs, (exceed + 1) / (n_perm + 1)


def build_text_lookup(gen_df, min_words=75):
    """Map (prompt, truncated_text) -> (family, model, label)."""
    lookup = {}
    for _, r in gen_df.iterrows():
        prompt = str(r.get("prompt", "")).strip()
        raw = str(r["psg"]).rstrip()
        result = truncate_to_min_sentences(raw, min_words=min_words)
        if result[0] is None:
            continue
        lookup[(prompt, result[0])] = (r["family"], r["model"], r["label"])
    return lookup


def collect_embeddings(cache, text_lookup, embedders):
    """Look up sentence embeddings for all passages via CacheManager."""
    grouped = defaultdict(list)
    for ei, embedder in enumerate(embedders):
        print(f"  [{ei+1}/{len(embedders)}] {embedder}...")
        matched = 0
        for (prompt, text), meta in text_lookup.items():
            sv = cache.get_sent_embeddings(embedder, prompt, text)
            if sv is not None and len(sv) > 0:
                grouped[(embedder, meta[0], meta[1], meta[2], prompt)].append(
                    np.mean(sv, axis=0)
                )
                matched += 1
        print(f"    matched {matched} passages")
    return grouped


def get_aligned_layer(grouped, fam):
    for layer in ["instruct", "superego"]:
        if any(k[2] == layer for k in grouped if k[1] == fam):
            return layer
    return None


def compute_mmd_table(grouped, n_perm=500, min_n=5):
    """Compute MMD² for each (embedder, family, label) cell."""
    results = []
    cells = set()
    for (embedder, fam, model, label, prompt), vecs in grouped.items():
        if model != "base":
            continue
        al = get_aligned_layer(grouped, fam)
        if al is None:
            continue
        av_key = (embedder, fam, al, label, prompt)
        if av_key not in grouped:
            continue

        bv = np.array(vecs)
        av = np.array(grouped[av_key])
        if len(bv) < min_n or len(av) < min_n:
            continue

        cat = label.rsplit("_", 1)
        cat = cat[0] if len(cat) == 2 and cat[1].isdigit() else label

        cell_id = (embedder, fam, label)
        if cell_id in cells:
            continue
        cells.add(cell_id)

        mmd_ba, p_ba = mmd2_permtest(bv, av, n_perm=n_perm)

        rng = np.random.default_rng(42)
        idx = rng.permutation(len(bv))
        h = len(idx) // 2
        if h >= min_n:
            mmd_bb, p_bb = mmd2_permtest(bv[idx[:h]], bv[idx[h : 2 * h]],
                                          n_perm=n_perm)
        else:
            mmd_bb, p_bb = np.nan, np.nan

        results.append({
            "embedder": embedder, "family": fam, "label": label,
            "category": cat, "n_base": len(bv), "n_aligned": len(av),
            "mmd_ba": mmd_ba, "p_ba": p_ba,
            "mmd_bb": mmd_bb, "p_bb": p_bb,
        })
    return pd.DataFrame(results)


def print_summary(rdf):
    """Print markdown-ready summary tables."""
    def _boot_ci(data, n_boot=5000):
        rng = np.random.default_rng(42)
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return np.nan, np.nan
        meds = [np.median(rng.choice(data, size=len(data), replace=True))
                for _ in range(n_boot)]
        return np.percentile(meds, 2.5), np.percentile(meds, 97.5)

    print()
    print("## Cross-generation MMD²: BASE ↔ ALIGNED vs BASE ↔ BASE")
    print()
    n_emb = rdf.embedder.nunique()
    print(f"*{n_emb} embedders, permutation test, "
          f"median across embedders*")
    print()

    # By family
    print("### By family")
    print()
    print("| Family | MMD²(B↔A) | 95% CI | MMD²(B↔B) | % sig | n |")
    print("|---|---|---|---|---|---|")
    fam_rows = []
    for fam in sorted(rdf.family.unique()):
        s = rdf[rdf.family == fam]
        med = s.mmd_ba.median()
        lo, hi = _boot_ci(s.mmd_ba.values)
        bb = s.mmd_bb.median()
        sig = (s.p_ba < 0.05).mean() * 100
        fam_rows.append((med, fam, lo, hi, bb, sig, len(s)))
    for med, fam, lo, hi, bb, sig, n in sorted(fam_rows, reverse=True):
        print(f"| {fam} | {med:.4f} | [{lo:.4f}, {hi:.4f}] | {bb:.4f} "
              f"| {sig:.0f}% | {n} |")
    print()

    # By category
    print("### By content category")
    print()
    print("| Category | MMD²(B↔A) | 95% CI | % sig | n |")
    print("|---|---|---|---|---|")
    cat_rows = []
    for cat in sorted(rdf.category.unique()):
        s = rdf[rdf.category == cat]
        med = s.mmd_ba.median()
        lo, hi = _boot_ci(s.mmd_ba.values)
        sig = (s.p_ba < 0.05).mean() * 100
        cat_rows.append((med, cat, lo, hi, sig, len(s)))
    for med, cat, lo, hi, sig, n in sorted(cat_rows, reverse=True):
        print(f"| {cat} | {med:.4f} | [{lo:.4f}, {hi:.4f}] | {sig:.0f}% | {n} |")

    cg = {c: rdf[rdf.category == c].mmd_ba.values for c in rdf.category.unique()}
    h, p = stats.kruskal(*cg.values())
    print()
    print(f"*Kruskal-Wallis across categories: H={h:.2f}, p={p:.4f}*")
    print()

    # By family × category
    print("### By family × category")
    print()
    print("| Family | Category | MMD²(B↔A) | 95% CI | sig | n |")
    print("|---|---|---|---|---|---|")
    fc_rows = []
    for fam in sorted(rdf.family.unique()):
        for cat in sorted(rdf.category.unique()):
            s = rdf[(rdf.family == fam) & (rdf.category == cat)]
            if len(s) < 2:
                continue
            med = s.mmd_ba.median()
            lo, hi = _boot_ci(s.mmd_ba.values)
            sig_mark = "***" if (s.p_ba < 0.05).all() else (
                "**" if (s.p_ba < 0.05).mean() > 0.5 else "")
            fc_rows.append((med, fam, cat, lo, hi, sig_mark, len(s)))
    for med, fam, cat, lo, hi, sig, n in sorted(fc_rows, reverse=True):
        print(f"| {fam} | {cat} | {med:.4f} | [{lo:.4f}, {hi:.4f}] "
              f"| {sig} | {n} |")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o", default="data/mmd_cross_generation.csv")
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--min-words", type=int, default=75)
    parser.add_argument("--summary", action="store_true",
                        help="Reprint summary from existing CSV")
    args = parser.parse_args()

    if args.summary:
        rdf = pd.read_csv(args.output)
        print_summary(rdf)
        return

    from malign_logits.embedding import load_generations_from_stash

    print("Loading generations from stash...")
    gen_df = load_generations_from_stash()
    print(f"  {len(gen_df)} generations across {gen_df.family.nunique()} families")

    print("Building truncated text lookup...")
    text_lookup = build_text_lookup(gen_df, min_words=args.min_words)
    print(f"  {len(text_lookup)} truncated passages")

    print("Collecting cached embeddings...")
    from malign_logits.cache import get_cache
    cache = get_cache()
    grouped = collect_embeddings(cache, text_lookup, EMBEDDERS)
    print(f"  {len(grouped)} (embedder, family, layer, label, prompt) groups")

    print(f"Computing MMD² ({args.n_perm} permutations per cell)...")
    rdf = compute_mmd_table(grouped, n_perm=args.n_perm)
    print(f"  {len(rdf)} cells")

    rdf.to_csv(args.output, index=False)
    print(f"\nSaved {args.output}")

    print_summary(rdf)


if __name__ == "__main__":
    main()
