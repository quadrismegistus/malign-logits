"""Condensation: does the substitution graph FUNNEL?

Freud's Verdichtung is many-into-one. No pair judgement can see it, which is why
the S schema names it as out of reach and why this script uses no annotations at
all -- only the design. Every (stem, member) is one edge faller -> riser, and the
question is whether that bipartite graph is asymmetric.

    IN-DEGREE   how many distinct fallers arrive at one riser
    OUT-DEGREE  how many distinct risers leave one faller

If alignment condenses, many suppressed acts converge on few permitted words and
in-degree exceeds out-degree. If it merely substitutes, the graph is symmetric
and `found` receiving thirty fallers is a fact about how common `found` is.

THE CONTROL IS THE SAME GRAPH WITH ITS DIRECTIONS SHUFFLED. Reversing a random
half of the edges preserves every vocabulary, every degree total and every
frequency effect, and destroys only the direction. That is the null: a word
being popular is not condensation, a word being popular AS A DESTINATION is.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
OUT = os.path.join(CAMPAIGN, "results")
PILOT = os.path.join(OUT, "r_eight_coder_verbpaired_50x2.parquet")
CONFIRM = os.path.join(OUT, "r_confirm_frame_255x2.parquet")
TABLE = os.path.join(OUT, "s_condensation.csv")

SEED = 20260806
NPERM = 20000


def asymmetry(fallers, risers):
    """Mean in-degree of destinations minus mean out-degree of sources, over
    distinct edges. Positive means the graph funnels."""
    e = pd.DataFrame({"a": fallers, "b": risers}).drop_duplicates()
    indeg = e.groupby("b").a.nunique()
    outdeg = e.groupby("a").b.nunique()
    return indeg.mean() - outdeg.mean(), indeg, outdeg


def main():
    df = pd.concat([pd.read_parquet(PILOT), pd.read_parquet(CONFIRM)], ignore_index=True)
    print("%d pairs over %d stems. No annotations used; this is the design."
          % (len(df), df.stem.nunique()))

    obs, indeg, outdeg = asymmetry(df.faller.values, df.riser.values)
    print("\nDISTINCT WORDS")
    print("  fallers %d   risers %d" % (df.faller.nunique(), df.riser.nunique()))
    print("\nDEGREE")
    print("  mean in-degree  of risers  %.3f   (distinct fallers arriving)" % indeg.mean())
    print("  mean out-degree of fallers %.3f   (distinct risers leaving)" % outdeg.mean())
    print("  asymmetry %+0.3f" % obs)

    #: Shuffle direction only. Every word keeps its membership in the edge set;
    #: only which end it sits on changes.
    rng = np.random.RandomState(SEED)
    a, b = df.faller.values, df.riser.values
    null = np.empty(NPERM)
    for i in range(NPERM):
        flip = rng.rand(len(a)) < 0.5
        na = np.where(flip, b, a)
        nb = np.where(flip, a, b)
        null[i] = asymmetry(na, nb)[0]
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (NPERM + 1)
    print("  permutation null mean %+0.3f sd %.3f   p=%.4f" % (null.mean(), null.std(), p))
    print("  %s" % ("FUNNELS: destinations are more concentrated than sources"
                    if p < 0.05 and obs > 0 else
                    "no directional asymmetry; concentration is a vocabulary fact"))

    print("\nTOP DESTINATIONS, and what they absorb")
    e = df[["faller", "riser", "stem"]].drop_duplicates()
    top = (e.groupby("riser")
             .agg(fallers=("faller", "nunique"), stems=("stem", "nunique"))
             .sort_values("fallers", ascending=False).head(12))
    top["out_as_source"] = [int(outdeg.get(w, 0)) for w in top.index]
    top["absorbs"] = [", ".join(sorted(e[e.riser == w].faller.unique())[:7])
                      for w in top.index]
    print(top.to_string())

    #: A word that is a big destination AND a big source is just common. The
    #: condensation points are the ones with a lopsided ratio.
    print("\nCONDENSATION POINTS: high in-degree, low out-degree")
    r = pd.DataFrame({"in": indeg}).join(pd.DataFrame({"out": outdeg}), how="outer").fillna(0)
    r["ratio"] = (r["in"] + 1) / (r["out"] + 1)
    print(r[r["in"] >= 5].sort_values("ratio", ascending=False).head(10).to_string())

    r.sort_values("in", ascending=False).to_csv(TABLE)
    print("\nwrote %s" % TABLE)

    #: A SECOND STATISTIC, AND IT WAS CHOSEN AFTER THE FIRST CAME BACK NULL.
    #: Said plainly because that is the pattern that inflated R's +0.235. The
    #: mean-difference above asks whether the AVERAGE riser is more concentrated
    #: than the average faller; it cannot see a heavy tail, and the hubs here are
    #: bidirectional (`said` 23-in/19-out) so they cancel. Condensation is a
    #: claim about sinks, not about averages. This counts them.
    print("\n" + "=" * 70)
    print("SINK COUNT. Post-hoc statistic, disclosed as such.")
    print("=" * 70)

    def sinks(fa, ri, k=5):
        e = pd.DataFrame({"a": fa, "b": ri}).drop_duplicates()
        i = e.groupby("b").a.nunique()
        o = e.groupby("a").b.nunique()
        w = sorted(set(i.index) | set(o.index))
        return sum(1 for x in w if i.get(x, 0) >= k and o.get(x, 0) == 0)

    for k in (3, 5, 8):
        obs_s = sinks(a, b, k)
        nul = np.empty(NPERM // 4)
        for i in range(len(nul)):
            f = rng.rand(len(a)) < 0.5
            nul[i] = sinks(np.where(f, b, a), np.where(f, a, b), k)
        pv = (1 + np.sum(nul >= obs_s)) / (len(nul) + 1)
        print("  words with in-degree >=%d and out-degree 0:  %2d observed, "
              "null %.2f (sd %.2f), p=%.4f" % (k, obs_s, nul.mean(), nul.std(), pv))

    #: THE ONE HONEST REPLICATION AVAILABLE. All 305 annotated stems are already
    #: in the graph above, so nothing here can be confirmed on held-out data --
    #: except that condensation needs NO ANNOTATION, only a pair list. The 379
    #: stems excluded by the vv* rule have never been looked at for anything.
    POP = os.path.join(os.path.dirname(os.path.dirname(CAMPAIGN)), "data",
                       "r_population_k2.parquet")
    P = pd.read_parquet(POP)
    used = set(zip(df.stem, df.member))
    H = P[[(s, m) not in used for s, m in zip(P.stem, P.member)]].drop_duplicates(["stem", "member"])
    print("\nREPLICATION on the %d never-annotated pairs (vv*-excluded stems)." % len(H))
    print("These need no coder, only faller and riser, so they are a real")
    print("held-out sample for this question and this question only.")
    ha, hb = H.faller.values, H.riser.values
    o2, i2, u2 = asymmetry(ha, hb)
    print("  mean asymmetry %+0.3f   (305-stem set: %+0.3f)" % (o2, obs))
    for k in (3, 5, 8):
        obs_s = sinks(ha, hb, k)
        nul = np.empty(NPERM // 4)
        for i in range(len(nul)):
            f = rng.rand(len(ha)) < 0.5
            nul[i] = sinks(np.where(f, hb, ha), np.where(f, ha, hb), k)
        pv = (1 + np.sum(nul >= obs_s)) / (len(nul) + 1)
        print("  in>=%d, out=0:  %2d observed, null %.2f (sd %.2f), p=%.4f"
              % (k, obs_s, nul.mean(), nul.std(), pv))


if __name__ == "__main__":
    main()
