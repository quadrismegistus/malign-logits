#!/usr/bin/env python3
"""Invoke Registration D's STAGE 1. A RUNNER, NOT A PRODUCER.

WHY THIS FILE EXISTS RATHER THAN AN EDIT TO `pairs_d.py`
--------------------------------------------------------
`pairs_d.py` is FROZEN at `0393a14addf54815` ([3320]) and its `main()`
deliberately refuses everything but the self-test. Wiring the stage-1 call into
it would change the bytes the pen froze and the auditor cleared — the [3162]
rule: **a live need is not a reason to edit a dead artifact.**

So the frozen file stays a LIBRARY and this thin caller supplies the two things
a library cannot: WHERE the artifact goes and WHICH seed. Both are declared
here, in the open, and neither touches an analytic choice.

    seed        20260731, inherited from m01_registration_b.py:79 via §D5's
                seeding clause. NOT chosen here.
    out         results/result_d_stage1.json

**THIS RUNNER COMPUTES NOTHING.** Every number in the artifact comes from
`pairs_d.build()` and `pairs_d.stage1()` at the frozen hash. If this file were
deleted the read would be unaffected; if `pairs_d.py` changed, it would not.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pairs_d as D

PRODUCER_SHA16 = "0393a14addf54815"      #: frozen [3320]
SEED = 20260731                          #: §D5, inherited from B:79
OUT = os.path.join(CAMPAIGN, "results", "result_d_stage1.json")


def main():
    #: THE PRODUCER'S OWN BYTES, CHECKED BEFORE IT IS USED. A runner that
    #: imports a frozen artifact without verifying it is trusting its own
    #: import path.
    got = hashlib.sha256(open(os.path.join(HERE, "pairs_d.py"), "rb")
                         .read()).hexdigest()[:16]
    if got != PRODUCER_SHA16:
        raise SystemExit(f"REFUSING: pairs_d.py is {got}, frozen is "
                         f"{PRODUCER_SHA16}. The producer moved.")
    print(f"producer {got}  VERIFIED against the freeze")

    built = D.build()
    r = built["roster"]
    print(f"\nROSTER, re-derived per §A8.2 -- the RULE is retained, not the set")
    print(f"  prompts {r['n_prompts']:,}  sha {r['prompts_sha16']}  "
          f"frozen {r['frozen_prompts_sha16']}")
    print(f"  models  {r['n_models']:,}      sha {r['models_sha16']}  "
          f"frozen {r['frozen_models_sha16']}")
    print(f"  DRIFT   {len(r['drift'])} item(s)  <- bound by NAME, §A8.2b")
    print(f"\nedges {len(built['edges'])}   pairs {len(built['pairs'])}   "
          f"prompts with >=1 qualifying cell {len(built['cells']):,}")
    print("\ncollection diagnostics:")
    for k, v in built["diag"].most_common(8):
        print(f"    {v:>8,}  {k}")

    payload, sha = D.stage1(built, OUT, seed=SEED)
    print(f"\nSTAGE 1 ARTIFACT  {OUT}")
    print(f"  sha256[:16]     {sha}")
    print("\n  per arm, per threshold point -- SDs and RAW MDEs, NO D, NO p:")
    for name, arm in payload["arms"].items():
        print(f"    {name}  ({arm['dimension']}, dir {arm['direction']:+d}, "
              f"resid {arm['residualisation']})")
        for t, cell in sorted(arm["per_t"].items()):
            if cell["status"] != "ok":
                print(f"      t={t}  n={cell['n']:<4} {cell['status']}")
            else:
                print(f"      t={t}  n={cell['n']:<4} "
                      f"sd {cell['sd_D_pair']:.5f}  "
                      f"raw MDE {cell['raw_mde']:.5f}  "
                      f"min attainable p {cell['min_attainable_p']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
