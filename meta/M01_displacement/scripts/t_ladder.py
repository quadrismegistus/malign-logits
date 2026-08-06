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

#: DERIVED FROM THE REGISTRY, never hand-listed. The first version named six
#: families by grepping for ones this seat recognised and MISSED TEN --
#: archangel-dpo, minicpm, map-neo, redpajama, stablelm, ct-llm, olmo-hybrid,
#: beaver, pythia, olmoe. `operation_edges` carries the same warning for the
#: same reason: a hand-enumerated candidate set is not a derivation.
PREF = {"dpo", "kto", "ppo", "slic", "orpo", "simpo", "rlhf"}
STAGE_ORDER = ["base", "sft", "pref", "rlvr"]


def ladders(scored):
    """family -> {stage: checkpoint}, for families with base, sft and a
    preference stage all scored. `tulu` has no base of its own -- its base is
    meta-llama/Llama-3.1-8B in the `llama` family -- so it is stitched across
    families explicitly rather than dropped or silently skipped."""
    import json
    R = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    recs = R["models"] if isinstance(R, dict) and "models" in R else R
    rows = list(recs.values() if isinstance(recs, dict) else recs)
    fam = collections.defaultdict(dict)
    for r in rows:
        mid = r.get("model_id") or r.get("id") or r.get("name")
        if mid not in scored:
            continue
        s = r.get("stage")
        key = "pref" if s in PREF else s
        if key in STAGE_ORDER and key not in fam[r.get("family", "?")]:
            fam[r.get("family", "?")][key] = mid
    if "tulu" in fam and "base" not in fam["tulu"]:
        b = "meta-llama/Llama-3.1-8B"
        if b in scored:
            fam["tulu"]["base"] = b          # cross-family base, declared not inferred
    return {f: v for f, v in fam.items()
            if {"base", "sft", "pref"} <= set(v)}


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
    ap.add_argument("--family", default=None, help="run one family only")
    a = ap.parse_args()

    from malign_logits.cache import open_stash
    from malign_logits.prompts import Prompts

    #: DEDUPE ON THE STRING. Eight prompt TEXTS exist under two prompt_ids each,
    #: so iterating Prompts yields them twice and a merge keyed on `prompt` then
    #: multiplies them 2x2 -- the first run reported n=2,206 against 2,190 cells
    #: and those eight carried 4x weight in every Jaccard mean.
    texts = sorted({p.text for p in Prompts.all(status="ACTIVE")
                    if all(ord(ch) < 128 for ch in p.text) and not getattr(p, "is_logical", False)})
    if a.limit:
        texts = texts[:a.limit]

    ls = open_stash(os.path.join(ROOT, "data", "raw", "cache", "logits"))
    scored = {k.get("model") for k in ls.keys() if isinstance(k, dict) and k.get("model")}
    L = ladders(scored)
    if a.family:
        L = {k: v for k, v in L.items() if k == a.family}
    print("prompts %d   families with base+sft+preference scored: %d\n" % (len(texts), len(L)))

    steps, jrows = [], []
    for f, ck in sorted(L.items()):
        chain = [ck[s] for s in STAGE_ORDER if s in ck]
        names = [s for s in STAGE_ORDER if s in ck]
        rungs = [(chain[i], chain[i + 1], "%s>%s" % (names[i], names[i + 1]))
                 for i in range(len(chain) - 1)]
        whole = (chain[0], chain[-1], "%s>%s WHOLE" % (names[0], names[-1]))
        got = {}
        for pre, post, lab in rungs + [whole]:
            D, direction = measure(pre, post, texts)
            if not len(D):
                continue
            got[lab] = D
            steps.append(dict(family=f, step=lab, pre=pre, post=post, cells=len(D),
                              js=float(D["js"].mean()), fall=float(D["n_fall"].mean()),
                              rise=float(D["n_rise"].mean()),
                              faller_share=float(D["n_fall"].sum() / max(D["n_fall"].sum() + D["n_rise"].sum(), 1)),
                              direction=direction))
            print("  %-14s %-22s %6d cells  JS %.4f  fall %5.2f  rise %5.2f"
                  % (f, lab, len(D), D["js"].mean(), D["n_fall"].mean(), D["n_rise"].mean()), flush=True)
        #: the primary question: do consecutive rungs move the SAME words?
        labs = [l for _, _, l in rungs if l in got]
        for i in range(len(labs) - 1):
            A, B = got[labs[i]], got[labs[i + 1]]
            M = A.merge(B, on="prompt", suffixes=("_a", "_b"))
            assert len(M) <= min(len(A), len(B)), "merge multiplied rows: duplicate prompt keys"
            if not len(M):
                continue
            jrows.append(dict(family=f, pair="%s | %s" % (labs[i], labs[i + 1]),
                              faller_jaccard=float(np.nanmean([jac(x, y) for x, y in zip(M["fallers_a"], M["fallers_b"])])),
                              riser_jaccard=float(np.nanmean([jac(x, y) for x, y in zip(M["risers_a"], M["risers_b"])])),
                              n=len(M)))

    S = pd.DataFrame(steps); J = pd.DataFrame(jrows)
    S.to_csv(os.path.join(OUT, "t_ladder_steps.csv"), index=False)
    J.to_csv(os.path.join(OUT, "t_ladder_jaccard.csv"), index=False)

    #: UNIT = THE FAMILY. One vote each, never pooled sites.
    print("\n%s\nACROSS FAMILIES, one vote each\n%s" % ("=" * 74, "=" * 74))
    print("  %-14s %7s %9s %9s %9s" % ("rung", "families", "JS med", "fall med", "fallshare"))
    for lab, g in S[~S["step"].str.contains("WHOLE")].groupby("step"):
        print("  %-14s %7d %9.4f %9.2f %8.1f%%"
              % (lab, len(g), g["js"].median(), g["fall"].median(), 100 * g["faller_share"].median()))
    if len(J):
        print("\n  consecutive-rung Jaccard, median over families:")
        for lab, g in J.groupby("pair"):
            print("    %-30s fallers %.4f  risers %.4f  (%d families)"
                  % (lab, g["faller_jaccard"].median(), g["riser_jaccard"].median(), len(g)))
    print("\nwrote t_ladder_steps.csv (%d rows), t_ladder_jaccard.csv (%d)" % (len(S), len(J)))


if __name__ == "__main__":
    main()
