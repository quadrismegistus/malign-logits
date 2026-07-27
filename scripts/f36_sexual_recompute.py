"""Recompute the expanded-sexual span-resistance result from the committed CSV.

F36_ledger.md books p=0.0003 for this and points at F36_euphemism, which does not
carry it. Before writing it up anywhere it has to be OBSERVED here rather than
copied from the ledger cell. Reimplemented from the data, not imported from
scripts/f36_sexual_beams.py, so the script's aggregation is a variable and not a
given.

The script pools family x pair differences into ONE Wilcoxon vector. Four families
x thirty pairs is 120 rows but not 120 independent observations -- the same pair
appears four times. Both are reported: the script's pooling, and the pair-level
test that averages over families first.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv("data/f36_sexual_beams.csv")
s = d[d.swap == "single"].copy()

def pairdiffs(sub):
    out = []
    for (fam, pair), g in sub.groupby(["family", "pair"]):
        t, b = g[g.is_trans], g[~g.is_trans]
        if t.empty or b.empty:
            continue
        out.append((fam, pair, t.mean_resist.mean() - b.mean_resist.mean()))
    return pd.DataFrame(out, columns=["family", "pair", "diff"])

print(f"{'cohort':18s}{'n':>5s}{'mean diff':>11s}{'median':>10s}{'p':>10s}")
for label, sub in [("all pairs", s), ("original only", s[s.source == "original"]),
                   ("new only", s[s.source == "new"])]:
    D = pairdiffs(sub)
    v = D["diff"].values
    p = wilcoxon(v)[1] if len(v) >= 5 else np.nan
    print(f"{label:18s}{len(v):>5d}{v.mean():>+11.4f}{np.median(v):>+10.4f}{p:>10.4f}")

print("\nPAIR-LEVEL (families averaged first -- one observation per pair):")
D = pairdiffs(s)
per = D.groupby("pair")["diff"].mean()
print(f"  n_pairs={len(per)}  mean={per.mean():+.4f}  median={per.median():+.4f}  "
      f"p={wilcoxon(per.values)[1]:.4f}  positive={int((per>0).sum())}/{len(per)}")

print("\nPer family (script's own breakdown):")
for fam, g in D.groupby("family"):
    v = g["diff"].values
    p = wilcoxon(v)[1] if len(v) >= 5 else np.nan
    print(f"  {fam:12s} n={len(v):2d}  diff={v.mean():+.4f}  p={p:.4f}")
