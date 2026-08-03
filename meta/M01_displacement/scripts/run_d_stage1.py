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

#: STALE-NOT-SUPERSEDED, corrected under [3856].2. This pin was NOT moved by
#: tonight's repair -- it was already wrong. `90d23ead9000c56c` predates
#: `stage2`, `reading_rule`, `read_point` and `mde_reading`, four analysis
#: functions that did not exist in it (AST-compared, [3853]); it stopped being
#: HEAD at 02427360, 2026-08-03 11:22. THE AUTHORITY IS THE REGISTRATION, never
#: per-runner drift: Registration D freezes `pairs_d.py` at 84011269d00eea6b and
#: stage 2's runner already gated on exactly that.
#:
#: PROVEN, not assumed ([3868]): stage 1 re-run to a scratch path under BOTH
#: producers x BOTH instrument states. `84011269d00eea6b` x movement
#: `28541cced0ec081b` (pre-repair, 60d605c1) reproduces the artifact of record
#: `9ae70405b23d96fe` EXACTLY. The same producer under the REPAIRED movement
#: gives `15b529a7d9261c8b` -- so the artifact's non-reproduction at HEAD was
#: the instrument moving, not the producer.
PRODUCER_SHA16_STALE = "90d23ead9000c56c"   #: never produced this artifact
PRODUCER_SHA16 = "84011269d00eea6b"      #: Registration D's frozen producer
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

    #: CUSTODY AROUND THE WRITE, [3830]/[3851]. The artifact is chmod a-w and
    #: that is the point. ESCROW FIRST, read-only, and unlock only after the
    #: escrow exists: if the write fails between unlock and lock, the escrow is
    #: the copy that survives. ANNOUNCED, never silent -- an unlock nobody sees
    #: is the lock not being there.
    if os.path.exists(OUT):
        prior = open(OUT, "rb").read()
        ph = hashlib.sha256(prior).hexdigest()[:16]
        esc_dir = os.path.join(CAMPAIGN, "results", "superseded")
        os.makedirs(esc_dir, exist_ok=True)
        dst = os.path.join(esc_dir, f"result_d_stage1.PREFIX-{ph}.json")
        if not os.path.exists(dst):
            with open(dst, "wb") as fh:
                fh.write(prior)
            os.chmod(dst, 0o444)
        print(f"\n  escrowed prior artifact @ {ph} -> {os.path.basename(dst)}")
        print(f"  UNLOCKING {os.path.basename(OUT)} for the re-emit")
        os.chmod(OUT, 0o644)

    payload, sha = D.stage1(built, OUT, seed=SEED)
    os.chmod(OUT, 0o444)
    print(f"  RE-LOCKED a-w")
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
