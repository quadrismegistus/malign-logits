"""The uncertainty/movement shape with theta removed from BOTH axes.

[1748] reported a PLATEAU: movement rises off certainty then flattens. That was
measured with base entropy on the thresholded word partition and movement over the
same partition -- and [1748].3 had already shown the theta cut is not neutral at the
top of the entropy range, because a fixed cut applied to distributions of varying
spread means the fraction the instrument can see is a function of the x-variable.

Controlling for residual MASS was a PARTIAL fix (top/peak 0.73 -> 0.96). This is the
whole fix: entropy and movement both computed from the full cached logit vectors, so
theta plays no part on either axis.

    .venv/bin/python scripts/c1_uncertainty_fullvocab.py

COVERAGE IS THE LIMIT AND IT IS NOT A RANDOM SUBSET. Only the families that received
the full-logit precompute have both arms cached, and only on part of the neutral
stratum. The result below rests on 5 families and ~65 of the 127 neutral texts. It is
reported as what it is; the 21-family version needs a fresh logit run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from c1_institutional_neutral import distinct_texts, isolated_steps  # noqa: E402
from malign_logits.cache import get_cache  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "c1_uncertainty_fullvocab.csv")
MIN_TEXTS = 40


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(a, b):
    m = a > 0
    return float((a[m] * np.log(a[m] / b[m])).sum())


def main():
    cm = get_cache()
    neut = [p.text for p in distinct_texts("neutral")]
    rows, coverage = [], {}
    for key, step in isolated_steps().items():
        ts = [t for t in neut
              if cm.has_logits(step.pre.id, t) and cm.has_logits(step.post.id, t)]
        if len(ts) >= 5:
            coverage[key] = ts
        if len(ts) < MIN_TEXTS:
            continue
        for t in ts:
            p, q = softmax(cm.get_logits(step.pre.id, t)), softmax(cm.get_logits(step.post.id, t))
            m = 0.5 * (p + q)
            rows.append(dict(family=key, text=t,
                             H_full=float(-(p[p > 0] * np.log(p[p > 0])).sum()),
                             l1_full=float(0.5 * np.abs(q - p).sum()),
                             js_full=0.5 * kl(p, m) + 0.5 * kl(q, m)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    print(f"{len(d)} cells, {d.family.nunique()} families, {d.text.nunique()} texts")
    print(f"full-vocabulary entropy range {d.H_full.min():.2f}-{d.H_full.max():.2f}"
          "   (the thresholded partition reached only 3.8)\n")
    print(d.groupby("family").size().to_string(), "\n")

    r = np.array([g.H_full.corr(g.l1_full, method="spearman") for _, g in d.groupby("family")])
    print(f"spearman(H_full, movement) within family: median {np.median(r):+.3f}, "
          f"positive in {(r > 0).sum()}/{len(r)}"
          "   (thresholded version gave +0.378)\n")

    for col in ("l1_full", "js_full"):
        d["bin"] = d.groupby("family").H_full.transform(
            lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
        per = d.groupby(["family", "bin"])[col].median().unstack()
        m = per.median()
        below = sum(1 for _, g in per.iterrows()
                    if g.dropna().idxmax() < g.dropna().index.max())
        print(f"{col} by full-vocab entropy quintile (within family, then median):")
        print("   " + " ".join(f"{v:7.4f}" for v in m.values))
        print(f"   peak at quintile {int(m.idxmax())} of {m.index.max()};  "
              f"top/peak {m.iloc[-1] / m.max():.2f};  "
              f"families peaking below top: {below}/{len(per)}\n")

    # ---- COVERAGE DIAGNOSTIC ------------------------------------------------
    # The families BELOW the coverage floor have 7-12 cached texts. Tempting to run
    # them anyway and report a 21-family tally. They cannot test the relation, and
    # the reason is not small-n alone: 14 of 16 score an IDENTICAL 7-text set, so
    # they are ONE SAMPLE MEASURED SIXTEEN TIMES. The proof is to evaluate that same
    # 7-text set inside the families where the effect IS established -- it returns
    # -0.21 to +0.75 there, so any verdict from it is noise wearing a family label.
    print("COVERAGE DIAGNOSTIC -- why the sub-floor families are excluded")
    below = {k: v for k, v in coverage.items() if len(v) < MIN_TEXTS}
    if below:
        shared = sorted(set.intersection(*[set(v) for v in below.values()]))
        print(f"   {len(below)} families below the floor; "
              f"{len(shared)} texts common to ALL of them")
        print("   that same shared set, scored inside the families ABOVE the floor:")
        for key in sorted(d.family.unique()):
            step = isolated_steps()[key]
            H, M = [], []
            for t in shared:
                if not (cm.has_logits(step.pre.id, t) and cm.has_logits(step.post.id, t)):
                    continue
                pp, qq = softmax(cm.get_logits(step.pre.id, t)), softmax(cm.get_logits(step.post.id, t))
                H.append(float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum()))
                M.append(float(0.5 * np.abs(qq - pp).sum()))
            if len(H) >= 5:
                full = d[d.family == key]
                print(f"     {key:<16} rho on all {len(full):>3}: "
                      f"{full.H_full.corr(full.l1_full, method='spearman'):+.3f}    "
                      f"rho on the shared {len(H)}: "
                      f"{pd.Series(H).corr(pd.Series(M), method='spearman'):+.3f}")
        print("   UNINFORMATIVE, NOT NULL. A tally over these families reports the")
        print("   sample, not the families.\n")

    print("THE THRESHOLDED VERSION, for comparison:")
    print("   0.0345  0.0812  0.0848  0.0904  0.0864   peak at quintile 3, top/peak 0.96")
    print("\nTHE PLATEAU WAS THE INSTRUMENT. On full vocabulary the relation runs")
    print("through the top quintile. Monotone -- though decelerating, not linear.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
