"""Is manner enriched at SFT->DPO relative to base->SFT? Findings U.7.

    uv run --with lemminflect python t_manner.py

THE READING THIS KILLS. DPO's top risers include `gently`, `carefully`,
`stared`, `proceeded`, and the induced taxonomy's `quality_manner` survives at
DPO and not at SFT. That suggested a mapping on which SFT installs the
prohibition and the later rungs supply comportment -- the cut, then the
finishing school. It is a good reading and it is wrong.

WHY THE TEST IS BUILT THIS WAY. The evidence for the reading was a RANKED LIST,
which carries no baseline. Four separate claims in this campaign on one day were
suggested by a top-N and refuted by a denominator. So:

  THREE INDEPENDENT DEFINITIONS OF MANNER, declared before the contrast ran and
  not adjusted after. VerbNet's `manner_speaking` class, FrameNet's
  `Communication_manner` frame, and CLAWS `rr*` adverbs ending in -ly. Plus
  their union. Three resources built on unrelated principles; if manner is
  really DPO's register, it should not matter which one names it.

  THE COMPARISON IS BETWEEN RUNGS, NOT AGAINST ZERO. Manner words rise at both
  rungs -- that is not in question and it is not the claim. The claim is that
  they rise MORE at `sft->pref`. So the statistic is the manner share of RISERS
  at one rung minus the same share at the other, paired within family.

  UNIT IS THE FAMILY, 16 of them, one vote each -- as everywhere in findings U.
  A site-level test would let one verbose family carry the result.

RESULT: flat on all four definitions, differences +0.0001 to +0.0029, p 0.56 to
0.94, 8 or 9 of 16 families higher. Manner rises at both rungs at the same rate.
The list read as manner-inflected because `whispered` and `sighed` top BOTH
lists and the adverbs were noticed on the second pass.

Committed because a negative that cannot be reproduced from the repo is not a
finding, it is an assertion -- the registrar's point at docket [4777], and the
right one: this is the best-disciplined kill of the day and it was living in a
shell heredoc.
"""

import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

WALK = os.path.join(OUT, "t_ladder_words.parquet")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"
MIN_RISERS = 100      # a family-rung cell needs this many risers to contribute
RUNGS = ("base>sft", "sft>pref")


def claws():
    pos = {}
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                w, t = f[-1].strip().lower(), f[-3].strip()
                if w and w not in pos:
                    pos[w] = t
    return pos


def main():
    import s_lexicon_crosstab as X
    W = pd.read_parquet(WALK)
    toks = sorted(set(W["word"]))
    vn = X.verbnet_labels(toks)[0]
    fn = X.framenet_labels(toks)[0]
    pos = claws()

    defs = {
        "verbnet manner_speaking": lambda w: vn.get(w) == "manner_speaking",
        "framenet Communication_manner": lambda w: fn.get(w) == "Communication_manner",
        "-ly adverbs (CLAWS rr*)": lambda w: str(pos.get(w, "")).startswith("rr") and w.endswith("ly"),
    }
    base3 = list(defs.values())
    defs["ANY of the three"] = lambda w: any(f(w) for f in base3)

    print("IS MANNER ENRICHED AT sft>pref RELATIVE TO base>sft?  unit = family, paired\n")
    print("  %-32s %6s %10s %10s %10s %8s  %s"
          % ("manner definition", "types", "base>sft", "sft>pref", "diff", "p", "verdict"))
    rows = []
    for nm, f in defs.items():
        members = {w for w in toks if f(w)}
        if len(members) < 10:
            print("  %-32s %6d  too few types" % (nm, len(members)))
            continue
        per = []
        for fam, g in W.groupby("family"):
            cell = {}
            for r in RUNGS:
                h = g[(g["rung"] == r) & (g["role"] == "riser")]
                if len(h) < MIN_RISERS:
                    continue
                cell[r] = float(h["word"].isin(members).mean())
            if len(cell) == 2:
                per.append(cell)
        D = pd.DataFrame(per)
        if len(D) < 8:
            print("  %-32s %6d  only %d families" % (nm, len(members), len(D)))
            continue
        d = D[RUNGS[1]] - D[RUNGS[0]]
        p = stats.wilcoxon(D[RUNGS[1]], D[RUNGS[0]]).pvalue
        v = ("ENRICHED" if d.mean() > 0 and p < 0.05
             else "depleted" if d.mean() < 0 and p < 0.05 else "flat")
        rows.append(dict(definition=nm, n_types=len(members), n_families=len(D),
                         base_sft=D[RUNGS[0]].mean(), sft_pref=D[RUNGS[1]].mean(),
                         diff=d.mean(), fam_higher=int((d > 0).sum()), p=p, verdict=v))
        print("  %-32s %6d %10.4f %10.4f %+10.4f %8.4f  %s"
              % (nm, len(members), D[RUNGS[0]].mean(), D[RUNGS[1]].mean(), d.mean(), p, v))
        print("       %d of %d families higher at sft>pref" % ((d > 0).sum(), len(d)))

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "t_manner.csv"), index=False)
    if len(T):
        print("\n  %d of %d definitions ENRICHED. Manner is not DPO's register."
              % (int((T["verdict"] == "ENRICHED").sum()), len(T)))
    print("wrote t_manner.csv")


if __name__ == "__main__":
    main()
