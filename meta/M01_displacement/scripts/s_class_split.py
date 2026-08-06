"""The four markedness cells split by open/closed word class: eight cells.

    uv run python s_class_split.py

WHY. `s_depth_breadth.py` reports four cells -- breadth and depth crossed with
faller and riser -- and they come out clean: both faller cells detect, both
riser cells are null. malign then found on their population that their depth
effect was carried entirely by open-class movers, with closed-class as a
passing internal control (docket [4752]). Running the same split here breaks
the clean table.

  BREADTH is lexical, as it should be. Open-class fallers +3.7 percent
  detected; closed-class fallers flat and null. `the` and `was` are withdrawn
  from equally often in both twins, which is the control passing.

  DEPTH is not. Both depth detections are CLOSED-class, in both directions, at
  near-identical magnitudes. An effect that moves function words equally far
  down and equally far up is not a withdrawal, and the likeliest reading is
  that marked prompts carry sharper distributions so every function-word delta
  is larger. That is a property of the prompts, not of alignment. Untested:
  compare pre-alignment entropy at marked and unmarked sites.

SO ONE CELL OF EIGHT IS SAFE, and the clean four-cell result was clean because
it aggregated over word class. That is the same defect this document records at
three other levels.

NOTE ON THE DISAGREEMENT WITH malign. They report open-class detected and
closed-class null for depth; this reports the reverse. **Their split classifies
a SITE by the class of its top faller; this one classifies the WORDS.** A site
whose top faller is open-class still contains closed-class fallers. Those are
different operations and the disagreement is not yet a contradiction. Neither
class-resolved depth claim should be quoted until the same split runs at both
seats.

DEPTH READS A CACHED PER-SITE FILE. `s_depth_by_class.csv` is produced by the
walk in this script with `--walk`; without it the file must already exist,
because the walk costs a full pass over 44 edges and 1,361 twin prompts.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

OPEN = ("vv", "nn", "jj", "rr")


def classify():
    import s_lexicon_crosstab as X
    pos = X.claws()
    return lambda w: "open" if str(pos.get(str(w).lower(), "")).startswith(OPEN) else "closed"


def twins():
    P = pd.read_parquet(os.path.join(ROOT, "data", "r_population_k2.parquet"))
    return P.drop_duplicates("prompt").set_index("prompt")[["stem", "member"]]


def test(w, kind, role, cls):
    """Per-EDGE Wilcoxon. The pairs are 43 edges times a few hundred stems and
    are not independent; a pair-level test returns p=1e-17 for 2 percent."""
    if not len(w) or "marked" not in w or "unmarked" not in w:
        return None
    d = w["marked"] - w["unmarked"]
    pe = d.groupby("edge").mean()
    return dict(kind=kind, role=role, cls=cls, marked=w["marked"].mean(),
                unmarked=w["unmarked"].mean(), diff=d.mean(),
                pct=100 * d.mean() / w["unmarked"].mean(), n_pairs=len(w),
                n_edges=len(pe), edges_pos=int((pe > 0).sum()),
                p=stats.wilcoxon(pe, np.zeros(len(pe))).pvalue)


def walk_depth(cls, info):
    import m01_concentration as CC
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.prompts import Prompts
    keep = set(info.index)
    pr = [p for p in Prompts.all(status="ACTIVE") if p.text in keep]
    _q, models, _h, _d = CC.frozen_population()
    edges, _ = CC.operation_edges(models)
    rows = []
    for i, (_f, _p, st) in enumerate(edges, 1):
        eid = "%s>%s" % (str(st.pre).split("'")[1], str(st.post).split("'")[1])
        for p in pr:
            c = st.cell(p.text)
            if not c.is_present:
                continue
            m = c.movement(CANONICAL)
            if m is None:
                continue
            d = m.delta
            for role, ws in (("faller", m.fallers), ("riser", m.risers)):
                for cc in ("open", "closed"):
                    v = [abs(d[w]) for w in ws if w != RESIDUAL_KEY and w in d and cls(w) == cc]
                    if len(v) >= 3:
                        rows.append((eid, p.text, role, cc, float(np.mean(v))))
        if i % 15 == 0 or i == len(edges):
            print("  [%d/%d] %d rows" % (i, len(edges), len(rows)), flush=True)
    D = pd.DataFrame(rows, columns=["edge", "prompt", "role", "cls", "depth"]).join(info, on="prompt")
    D["m"] = D["member"].str.lower()
    D.to_csv(os.path.join(OUT, "s_depth_by_class.csv"), index=False)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", action="store_true", help="recompute the per-site depth file")
    a = ap.parse_args()
    cls, info = classify(), twins()

    W = pd.read_parquet(os.path.join(OUT, "movement_words.parquet"))
    W["cls"] = [cls(x) for x in W["word"]]
    print("movement vocabulary: %.1f%% open-class tokens" % (100 * (W["cls"] == "open").mean()))
    W = W.join(info, on="prompt")
    W = W[W["stem"].notna()].copy()
    W["m"] = W["member"].str.lower()

    br = []
    for role in ("faller", "riser"):
        for c in ("open", "closed"):
            g = W[(W["role"] == role) & (W["cls"] == c)].groupby(["edge", "prompt", "m"]).size()
            g = g.reset_index(name="n").join(info, on="prompt")
            w = g.pivot_table(index=["edge", "stem"], columns="m", values="n", aggfunc="mean").dropna()
            r = test(w, "breadth", role, c)
            if r:
                br.append(r)
    B = pd.DataFrame(br)
    B.to_csv(os.path.join(OUT, "s_breadth_by_class.csv"), index=False)

    f = os.path.join(OUT, "s_depth_by_class.csv")
    if a.walk or not os.path.exists(f):
        print("walking for depth (44 edges x 1,361 twin prompts)")
        D = walk_depth(cls, info)
    else:
        print("reusing s_depth_by_class.csv")
        D = pd.read_csv(f)
    de = []
    for role in ("faller", "riser"):
        for c in ("open", "closed"):
            g = D[(D["role"] == role) & (D["cls"] == c)]
            w = g.pivot_table(index=["edge", "stem"], columns="m", values="depth", aggfunc="mean").dropna()
            r = test(w, "depth", role, c)
            if r:
                de.append(r)
    #: the summary file the findings generator reads; the per-site file keeps
    #: its own name and is NOT overwritten by this
    pd.DataFrame(de).to_csv(os.path.join(OUT, "s_depth_by_class_summary.csv"), index=False)

    T = pd.concat([B, pd.DataFrame(de)], ignore_index=True)
    print("\n  %-8s %-7s %-7s %10s %10s %7s %8s %9s"
          % ("measure", "role", "class", "marked", "neutral", "pct", "edges+", "p"))
    for _, x in T.iterrows():
        print("  %-8s %-7s %-7s %10.5f %10.5f %6.1f%% %5d/%-3d %9.4f  %s"
              % (x["kind"], x["role"], x["cls"], x["marked"], x["unmarked"], x["pct"],
                 x["edges_pos"], x["n_edges"], x["p"], "DETECTED" if x["p"] < 0.05 else "null"))
    print("\n  %d of %d cells detect." % (int((T["p"] < 0.05).sum()), len(T)))
    print("wrote s_breadth_by_class.csv, s_depth_by_class_summary.csv")


if __name__ == "__main__":
    main()
