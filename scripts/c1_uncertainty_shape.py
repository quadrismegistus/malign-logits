"""Where does alignment intervene? The shape of movement against base uncertainty.

[1745].3 proposed that the entropy/movement relation is not a metric artefact but a
fact about alignment: it does not revise CONFIDENT predictions, because a certain
base offers no competing candidate for a preference signal to promote. [1746].4
answered that "in proportion to uncertainty" predicts a MONOTONE rise and the data
show a plateau, which is a different mechanism.

    .venv/bin/python scripts/c1_uncertainty_shape.py

THE POPULATION IS THE NEUTRAL STRATUM ALONE, and that is the point. C1's confound is
that entropy tracks DOMAIN -- its institutional arm is uniformly high-entropy advice
frames. Inside the neutral stratum domain is CONSTANT by construction while entropy
varies from 0.0 to 3.8, so the mechanism can be asked here without the confound that
makes it unaskable across strata ([1746].3).

TWO CONTROLS, because the first shape this produced was wrong:

  FAMILY COMPOSITION   Deciles are cut WITHIN each family and then averaged, so a
                       trend cannot come from high-entropy cells being drawn
                       disproportionately from families that move little overall.

  THE THETA CUT        High-entropy distributions spread mass below theta into the
                       RESIDUAL bin, where word-level movement is not measured. That
                       would depress measured movement at the top of the range and
                       MANUFACTURE a downturn. Restricting to low-residual cells is
                       what separates a real decline from the instrument's own cut --
                       and it removed most of the decline that was there.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from c1_institutional_neutral import distinct_texts, isolated_steps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "c1_uncertainty_shape.csv")


def entropy(d):
    p = np.array([v for v in d.values() if v > 0])
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def collect():
    neut = [p.text for p in distinct_texts("neutral")]
    rows = []
    for key, step in isolated_steps().items():
        for t in neut:
            c = step.cell(t)
            if c is None or not c.is_present:
                continue
            pre, post = dict(c.pre.probs), dict(c.post.probs)
            ws = set(pre) | set(post)
            rows.append(dict(
                family=key, text=t, H=entropy(pre),
                departed=0.5 * sum(abs(post.get(w, 0.0) - pre.get(w, 0.0)) for w in ws),
                residual_pre=float(c.pre.residual)))
    return pd.DataFrame(rows)


def shape(d, nbins, label):
    """Within-family bins, then the median across families. Returns the bin medians."""
    d = d.copy()
    d["bin"] = d.groupby("family").H.transform(
        lambda s: pd.qcut(s, nbins, labels=False, duplicates="drop"))
    per = d.groupby(["family", "bin"]).departed.median().unstack()
    m = per.median()
    below = sum(1 for _, g in per.iterrows()
                if g.dropna().idxmax() < g.dropna().index.max())
    print(f"  {label}  (n={len(d)} cells)")
    print(f"     bin     " + " ".join(f"{i:>6}" for i in m.index))
    print(f"     median  " + " ".join(f"{v:6.4f}" for v in m.values))
    print(f"     peak at bin {int(m.idxmax())} of {m.index.max()};  "
          f"top/peak = {m.iloc[-1] / m.max():.2f};  "
          f"families peaking below the top: {below}/{len(per)}")
    return m


def main():
    d = collect()
    d.to_csv(OUT, index=False)
    print(f"NEUTRAL STRATUM ONLY -- domain held constant, entropy free to vary")
    print(f"{len(d)} cells, {d.family.nunique()} families, {d.text.nunique()} texts, "
          f"H {d.H.min():.2f}-{d.H.max():.2f}\n")

    print("1. IS THE RELATION REAL, WITHIN FAMILY?")
    r = np.array([g.H.corr(g.departed, method="spearman") for _, g in d.groupby("family")])
    print(f"   spearman(H, departed) within family: median {np.median(r):+.3f}  "
          f"range {r.min():+.3f} to {r.max():+.3f}   positive in {(r > 0).sum()}/{len(r)}\n")

    print("2. THE SHAPE, AND THE CONTROL THAT CHANGES IT")
    shape(d, 10, "ALL cells, entropy deciles          ")
    print()
    print(f"   spearman(H, residual_pre) = {d.H.corr(d.residual_pre, method='spearman'):+.3f}"
          "   <- why the theta cut matters")
    lo = d[d.residual_pre <= d.residual_pre.median()]
    print()
    shape(lo, 5, "LOW-RESIDUAL cells, entropy quintiles")

    print("\n3. READING")
    print("   The relation is REAL and universal: positive within every family.")
    print("   The shape is NOT 'in proportion to uncertainty' -- that predicts a")
    print("   monotone rise. It is a SHARP RISE OFF CERTAINTY THEN A PLATEAU.")
    print("   Most of the apparent high-entropy DECLINE was the theta cut: it")
    print("   largely disappears once residual mass is controlled.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
