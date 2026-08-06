"""Does markedness change how MANY words move, or how FAR they move, or both?

    uv run python s_depth_breadth.py

FOUR CELLS. Breadth is words moved per site; depth is mean |delta| per word
moved. Crossed with role, that is faller-breadth, faller-depth, riser-breadth,
riser-depth, and the pattern across the four is the finding rather than any one
of them.

WHY IT EXISTS. Finding 13 originally claimed that alignment withdraws more in
the marked twin AND adds less there. The first half replicated at a second seat
and the second did not, so the section stood at "withdrawal confirmed,
substitution unresolved" -- a claim about CATEGORY SHARES aggregated over
lexicons, which is a lot of machinery for a question that can be asked of the
words directly. This asks it directly: no lexicon, no categories, no
aggregation over resources.

malign proposed (docket [4748]) that the effect is depth and not breadth --
marked sites withdraw more from the words they withdraw from, but not from a
longer list. On their 744 sites breadth was flat at about 1% and wrong-signed.
It is not flat here; it is about 3% and detected. Their population cannot see
an effect that size, so the refinement is a statement about n rather than about
alignment, and both cells belong in the sentence.

PAIRING IS WITHIN STEM, matching their design: both members of a minimal pair
at the same edge, so the transgressive word is the only thing that differs.

THE TEST IS PER EDGE, not per pair. 28,931 paired cells are not independent --
they are 43 edges times a few hundred stems -- and a Wilcoxon over the pairs
will return p=1e-17 for a 2% difference. The edge is the replicate. The pair
count and the share-of-pairs are reported anyway, because the breadth effect
turns out to be carried by a tail: marked is larger at only 43% of pairs while
the mean difference is positive, and quoting the mean without that share is the
same failure both seats spent the day catching in each other.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
POP = os.path.join(ROOT, "data", "r_population_k2.parquet")


def cell(D, col, role, kind):
    w = D.pivot_table(index=["edge", "stem"], columns="m", values=col, aggfunc="mean").dropna()
    if not len(w) or "marked" not in w or "unmarked" not in w:
        return None
    d = w["marked"] - w["unmarked"]
    pe = d.groupby("edge").mean()
    p = stats.wilcoxon(pe, np.zeros(len(pe))).pvalue
    return dict(role=role, kind=kind, marked=w["marked"].mean(), unmarked=w["unmarked"].mean(),
                diff=d.mean(), pct=100 * d.mean() / w["unmarked"].mean(),
                n_pairs=len(w), share_marked_larger=100 * (d > 0).mean(),
                n_edges=len(pe), edges_pos=int((pe > 0).sum()), p=p, detected=p < 0.05)


def main():
    P = pd.read_parquet(POP)
    info = P.drop_duplicates("prompt").set_index("prompt")[["stem", "member"]]
    W = pd.read_parquet(os.path.join(OUT, "movement_words.parquet"))
    n = W.groupby(["edge", "prompt", "role"]).size().unstack("role").fillna(0).reset_index()
    n = n.join(info, on="prompt")
    n = n[n["stem"].notna()].copy()
    n["m"] = n["member"].str.lower()

    S = pd.read_csv(os.path.join(OUT, "s_spread_blind.csv")).join(info, on="prompt")
    S = S[S["stem"].notna()].copy()
    S["m"] = S["member"].str.lower()

    rows = [cell(n, "faller", "faller", "breadth"), cell(n, "riser", "riser", "breadth"),
            cell(S, "mean_fall", "faller", "depth"), cell(S, "mean_rise", "riser", "depth")]
    D = pd.DataFrame([r for r in rows if r])
    D.to_csv(os.path.join(OUT, "s_depth_breadth.csv"), index=False)

    print("within-stem pairs: %d, edges: %d\n" % (D["n_pairs"].max(), D["n_edges"].max()))
    print("  %-7s %-8s %9s %9s %10s %7s %8s %10s  %s"
          % ("role", "kind", "marked", "unmarked", "diff", "pct", "edges+", "p", "verdict"))
    for _, x in D.iterrows():
        print("  %-7s %-8s %9.5f %9.5f %+10.5f %6.1f%% %5d/%-3d %10.4f  %s"
              % (x["role"], x["kind"], x["marked"], x["unmarked"], x["diff"], x["pct"],
                 x["edges_pos"], x["n_edges"], x["p"], "DETECTED" if x["detected"] else "not detected"))
    print("\n  share of PAIRS where marked is larger (the breadth effect is tail-carried):")
    for _, x in D.iterrows():
        print("     %-7s %-8s %.1f%%" % (x["role"], x["kind"], x["share_marked_larger"]))
    f = D[D["role"] == "faller"]["detected"].all()
    r = (~D[D["role"] == "riser"]["detected"]).all()
    print("\n  both faller cells detected: %s.  both riser cells null: %s." % (f, r))
    print("wrote s_depth_breadth.csv")


if __name__ == "__main__":
    main()
