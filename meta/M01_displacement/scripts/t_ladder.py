"""Plan U, step 1: decompose one family's alignment into its rungs.

    uv run python t_ladder.py                    OLMo-2-0425-1B, all rungs
    uv run python t_ladder.py --family olmo-tiny

THE PLAN IS `registrations/plan_u_ladder.md`. It is a plan, not a registration,
and the four outcome readings were written there before this ran.

WHY. All 43 edges in findings T have a BASE pre-side, so `SFT -> DPO` has never
been measured. Finding 9 approximates the question by comparing which recipe
produced the ENDPOINT -- base->SFT-checkpoint against base->DPO-checkpoint -- so
its DPO edges contain the SFT step inside them and it cannot isolate DPO. Its
headline, *SFT alone produces the operation at full strength and preference
optimization does not add to it*, is testable directly and this is the test.

THIS IS ONE FAMILY, SO THE UNIT IS THE PROMPT AND THE RESULT IS DESCRIPTIVE.
The plan's unit is the FAMILY, six votes. One family supplies one vote, which is
not a test of anything. What this run can do is show the SHAPE -- whether the
rungs move comparable numbers of words and whether they move the same ones --
well enough to decide whether the other five families are worth the compute.
Nothing here should be quoted as a six-family result, because it is not one.

TWO MEASURES, AND THE SECOND IS NOT OPTIONAL.

  CANONICAL fallers and risers per rung, then the JACCARD BETWEEN RUNGS of the
  faller sets and of the riser sets. Between RUNGS, never between fallers and
  risers -- that second thing is ledger clause 6 and it is a settled negative.

  WORD-LEVEL JS per site, threshold-free. CANONICAL floors at min_prob 0.003 and
  delta 0.003, so a rung can move real mass with no word clearing the bar and the
  count would read zero while the distribution had changed. **JS is the only
  thing that separates "DPO does nothing" from "DPO does less than one
  threshold's worth",** and those are different claims with different
  consequences for finding 9.

THE PROVENANCE CAVEAT, WHICH THE DATA CANNOT SETTLE. `Step`'s own docstring says
the registry's relations are STAR-SHAPED from the base: an aligned arm hangs off
the base, not off its own SFT arm. So the registry declares each rung's STAGE
(base/sft/dpo/rlvr, by method name rather than by guessing from position) but
does NOT record that `-DPO` was trained FROM `-SFT`. **The ladder ordering comes
from AI2's published pipeline and the naming convention, not from anything in
this repo.** If that ordering is wrong the composites below are still valid --
they are just `base -> X` edges -- but the rung-to-rung steps measure nothing.
Stated here because it cannot be checked here.

COMPOSITES ARE REPORTED BESIDE THE RUNGS on purpose. `base->Instruct` is the edge
findings T actually measured for this family, so printing it next to the rungs
shows directly whether the rungs sum to the thing we have been reporting.
"""

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

LADDER = ["allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT",
          "allenai/OLMo-2-0425-1B-DPO", "allenai/OLMo-2-0425-1B-Instruct"]
SHORT = {c: c.split("-1B")[-1].lstrip("-") or "base" for c in LADDER}


def js(p, q):
    """Jensen-Shannon divergence in bits over the union of two word supports.

    The residual `__TAIL__` is KEPT. It is real untruncated mass and dropping it
    would renormalise each arm over its own support -- the restrict-then-normalise
    shape that produced an arithmetic identity in this campaign once already.
    """
    keys = set(p) | set(q)
    a = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return np.nan
    a, b = a / sa, b / sb
    m = 0.5 * (a + b)
    def kl(x):
        nz = (x > 0) & (m > 0)
        return float(np.sum(x[nz] * np.log2(x[nz] / m[nz])))
    return 0.5 * kl(a) + 0.5 * kl(b)


def measure(pre, post, texts):
    """One step: per-prompt faller set, riser set, and threshold-free JS."""
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.step import Step
    st = Step(Checkpoint(pre), Checkpoint(post))
    rows = []
    for t in texts:
        c = st.cell(t)
        if not c.is_present:
            continue
        d = js(c.pre.probs, c.post.probs)
        m = c.movement(CANONICAL)
        f = frozenset(w for w in (m.fallers if m else []) if w != RESIDUAL_KEY)
        r = frozenset(w for w in (m.risers if m else []) if w != RESIDUAL_KEY)
        rows.append(dict(prompt=t, js=d, n_fall=len(f), n_rise=len(r), fallers=f, risers=r))
    return pd.DataFrame(rows), getattr(st, "direction", "?")


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap prompts, for a smoke run")
    a = ap.parse_args()

    from malign_logits.prompts import Prompts
    #: DEDUPE ON THE STRING. Eight prompt TEXTS exist under two prompt_ids each,
    #: so a list comprehension over Prompts yields them twice, and a later merge
    #: keyed on `prompt` then multiplies them 2x2 -- the first run reported
    #: n=2,206 against 2,190 cells and those eight carried 4x weight in every
    #: Jaccard mean. The ledger already says retirement operates on STRINGS not
    #: rows, for exactly this reason. A cell is keyed on text, so the duplicate
    #: ids produce identical measurements and dropping one is lossless.
    texts = sorted({p.text for p in Prompts.all(status="ACTIVE")
                    if all(ord(ch) < 128 for ch in p.text) and not getattr(p, "is_logical", False)})
    if a.limit:
        texts = texts[:a.limit]
    print("prompts: %d\n" % len(texts))

    rungs = [(LADDER[i], LADDER[i + 1]) for i in range(len(LADDER) - 1)]
    comps = [(LADDER[0], LADDER[2]), (LADDER[0], LADDER[3])]
    got = {}
    print("%-22s %7s %9s %9s %9s  %s" % ("step", "cells", "JS mean", "fall/site", "rise/site", "dir"))
    for pre, post in rungs + comps:
        D, direction = measure(pre, post, texts)
        if not len(D):
            print("%-22s   no cells" % ("%s>%s" % (SHORT[pre], SHORT[post])))
            continue
        got[(pre, post)] = D
        print("%-22s %7d %9.4f %9.2f %9.2f  %s"
              % ("%s>%s" % (SHORT[pre], SHORT[post]), len(D), D["js"].mean(),
                 D["n_fall"].mean(), D["n_rise"].mean(), direction), flush=True)

    #: THE PRIMARY COMPARISON. Do the rungs move the SAME words?
    print("\nJACCARD BETWEEN RUNGS, per prompt, over prompts both rungs scored")
    print("  %-34s %11s %11s %7s" % ("rung pair", "fallers", "risers", "n"))
    jrows = []
    for i in range(len(rungs)):
        for j in range(i + 1, len(rungs)):
            A, B = got.get(rungs[i]), got.get(rungs[j])
            if A is None or B is None:
                continue
            M = A.merge(B, on="prompt", suffixes=("_a", "_b"))
            if not len(M):
                continue
            #: a merge that returns more rows than either side had is a
            #: non-unique key, not a result. Raise rather than average over it.
            assert len(M) <= min(len(A), len(B)), (
                "merge multiplied rows: %d from %d and %d -- duplicate prompt keys"
                % (len(M), len(A), len(B)))
            fj = [jac(x, y) for x, y in zip(M["fallers_a"], M["fallers_b"])]
            rj = [jac(x, y) for x, y in zip(M["risers_a"], M["risers_b"])]
            lab = "%s>%s vs %s>%s" % (SHORT[rungs[i][0]], SHORT[rungs[i][1]],
                                      SHORT[rungs[j][0]], SHORT[rungs[j][1]])
            print("  %-34s %11.4f %11.4f %7d" % (lab, np.nanmean(fj), np.nanmean(rj), len(M)))
            jrows.append(dict(pair=lab, faller_jaccard=float(np.nanmean(fj)),
                              riser_jaccard=float(np.nanmean(rj)), n=len(M)))

    #: DO THE RUNGS SUM TO THE EDGE WE HAVE BEEN REPORTING?
    if all(k in got for k in rungs) and (LADDER[0], LADDER[3]) in got:
        s = sum(got[k]["js"].mean() for k in rungs)
        w = got[(LADDER[0], LADDER[3])]["js"].mean()
        print("\nsum of rung JS %.4f against the whole edge base>Instruct %.4f  (ratio %.2f)"
              % (s, w, s / w if w else np.nan))
        print("  JS is not additive, so this is a shape check and not an identity.")

    if jrows:
        pd.DataFrame(jrows).to_csv(os.path.join(OUT, "t_ladder_jaccard.csv"), index=False)
    pd.concat([D.drop(columns=["fallers", "risers"]).assign(step="%s>%s" % (SHORT[p], SHORT[q]))
               for (p, q), D in got.items()], ignore_index=True).to_csv(
        os.path.join(OUT, "t_ladder_steps.csv"), index=False)
    print("\nwrote t_ladder_steps.csv, t_ladder_jaccard.csv")
    print("ONE FAMILY. Descriptive. Not the six-family comparison the plan describes.")


if __name__ == "__main__":
    main()
