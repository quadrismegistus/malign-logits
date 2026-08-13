"""Does a word's M01 movement predict whether alignment USES it more or less?

    uv run python meta/M06_generation/scripts/m06_mediation_corr.py
    -> results/mediation_corr.json + mediation_corr_words.parquet

The question, in RH's words: are the words that carry M06's passage-level
surprisal drop the same words M01 found alignment moved?

The decomposition already answered half of it. A passage gets less surprising
either because the model uses DIFFERENT WORDS or because THE SAME WORDS COST
LESS, and the split is near-entirely the first: composition dominates, level is
about zero and unstable across decomposition orders. So the question reduces to
whether the words the aligned model uses more and less of are the words M01 says
rose and fell.

WHY THIS REPLACES THE CLASSIFIED VERSION. Earlier passes turned M01's per-cell
label into a per-word-type label and asked whether "movers" differ from
"non-movers". Every threshold admitted `the`, which is not a defect of the
threshold: `the` MOVES CONSTANTLY AND GOES NOWHERE. Measured on Llama-3.1-8B,
it is non-still in 30.5% of its 1,575 cells -- more volatile than `hit` or
`hurt` -- while its direction is -3.3% of the cells it moves in. Volatility and
direction are two dimensions and a binary mover flag multiplies them into one.

So no classification and no threshold. Two continuous scores per (pair, word):

    pct_moved       100 * (n_fall + n_rise) / n_cells        how volatile
    dir_when_moved  100 * (n_fall - n_rise) / (n_fall+n_rise)  which way
    net_fall        pct_moved * dir_when_moved / 100         the product

    kill  51.8% moved, +100 direction     scream  41.9% moved, -44
    the   30.5% moved,   -3.3 direction   hurt    16.7% moved, -92

`the` disqualifies itself arithmetically instead of being excluded by hand.

DECLARED, before running: net_fall should correlate NEGATIVELY with the
composition change (f_aligned - f_base). A word M01 sees falling should be used
LESS by the aligned model. Refuted by a null or positive correlation.

The magnitude variant uses log(p_base/p_aligned) FLOORED AT THETA, not at an
epsilon. twp is theta-truncated, so p=0 means "below 0.001", and an epsilon of
1e-9 turns that into a fabricated 14 nats -- `murder` has p_aligned=0 in 44 of
its 63 cells, and its mean log-ratio of 10.46 was mostly the epsilon.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
THETA = 0.001
MIN_CELLS = 20          # a percentage over few cells is not a percentage


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 22)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats

    PW = pd.read_parquet(os.path.join(OUTD, "mediation_words_joined.parquet"))
    pairs = sorted(PW["pair"].unique())
    print("M06 side: %d pairs, %d (pair, word) rows" % (len(pairs), len(PW)))

    plist = "','".join(p.replace("'", "\\'") for p in pairs)
    #: READ DISTINCT. The canonical slice carries 3,982,956 excess rows over
    #: 73,642,696 distinct (base, aligned, prompt, word) keys -- 5.1% -- and the
    #: duplication is CONCENTRATED, not uniform: the Llama-3.1-8B pair has none
    #: at all. Counting rows would inflate each word's cell count by however
    #: much its own pair happens to be duplicated, which is a per-pair bias in
    #: the denominator of every percentage here. [5872] reports zero cls
    #: disagreements among the duplicates, so collapsing them is safe.
    q = ("SELECT pair, word, count() AS cells, "
         "countIf(cls='fall') AS nf, countIf(cls='rise') AS nr, "
         "avg(log(greatest(p_base,%g)/greatest(p_aligned,%g))) AS logratio "
         "FROM (SELECT DISTINCT concat(base,'>',aligned) AS pair, prompt, word, "
         "      p_base, p_aligned, cls FROM malign_logits.movement "
         "      WHERE rule='canonical' AND concat(base,'>',aligned) IN ('%s')) "
         "GROUP BY pair, word" % (THETA, THETA, plist))
    rows = list(ch_rows(q))
    M = pd.DataFrame(rows)
    print("M01 side: %d (pair, word) rows" % len(M))

    M["cells"] = M["cells"].astype(float)
    M = M[M["cells"] >= MIN_CELLS].copy()
    moved = (M["nf"] + M["nr"]).astype(float)
    M["pct_moved"] = 100.0 * moved / M["cells"]
    M["dir_when_moved"] = np.where(moved > 0,
                                   100.0 * (M["nf"] - M["nr"]) / np.maximum(moved, 1), 0.0)
    M["net_fall"] = 100.0 * (M["nf"] - M["nr"]) / M["cells"]
    print("after >=%d cells: %d rows" % (MIN_CELLS, len(M)))

    D = PW.merge(M, on=["pair", "word"], how="inner")
    print("joined: %d (pair, word) rows, %d pairs\n" % (D, len(D["pair"].unique()))
          if False else "joined: %d (pair, word) rows, %d pairs\n"
          % (len(D), len(D["pair"].unique())))

    #: comp is f_aligned - f_base, a frequency share difference. Per 10k for
    #: legibility only; correlations are scale-free.
    D["comp10k"] = D["comp"] * 1e4

    out, res = [], {}
    for name, xcol in (("net_fall", "net_fall"),
                       ("dir_when_moved", "dir_when_moved"),
                       ("pct_moved", "pct_moved"),
                       ("mean_logratio(theta-floored)", "logratio")):
        rr = []
        for pair, g in D.groupby("pair"):
            x, y = g[xcol].to_numpy(), g["comp10k"].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 50:
                continue
            rr.append(stats.spearmanr(x[m], y[m]).statistic)
        rr = np.array([v for v in rr if np.isfinite(v)])
        t = stats.wilcoxon(rr) if len(rr) > 5 else None
        print("  %-30s rho %+.3f   n=%2d pairs  %s  [%d/%d negative]"
              % (name, rr.mean(), len(rr),
                 ("wilcoxon p %.3g" % t.pvalue) if t else "",
                 int((rr < 0).sum()), len(rr)))
        res[name] = {"mean_rho": float(rr.mean()), "n_pairs": int(len(rr)),
                     "p": float(t.pvalue) if t else None,
                     "n_negative": int((rr < 0).sum())}
        out.append((name, rr))

    #: the same thing as a contrast, for readability: the words M01 pushes down
    #: hardest against the ones it pushes up hardest, by occurrence-weighted
    #: composition change.
    print("\n  composition change by net_fall decile (per 10k tokens, pooled):")
    D["dec"] = pd.qcut(D["net_fall"], 10, labels=False, duplicates="drop")
    g = D.groupby("dec").apply(
        lambda d: pd.Series({
            "net_fall": d["net_fall"].mean(),
            "comp10k": np.average(d["comp10k"], weights=np.maximum(d["occ_b"], 1)),
            "n": len(d)}), include_groups=False)
    for dec, r in g.iterrows():
        print("    decile %2d  net_fall %+6.1f   comp %+8.3f   n=%d"
              % (dec, r["net_fall"], r["comp10k"], int(r["n"])))
    res["deciles"] = g.reset_index().to_dict(orient="records")

    json.dump(res, open(os.path.join(OUTD, "mediation_corr.json"), "w"), indent=1)
    D.to_parquet(os.path.join(OUTD, "mediation_corr_words.parquet"), index=False)
    print("\nwrote mediation_corr.json, mediation_corr_words.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
