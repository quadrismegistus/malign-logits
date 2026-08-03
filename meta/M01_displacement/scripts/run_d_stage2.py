#!/usr/bin/env python3
"""Invoke Registration D's STAGE 2. A RUNNER, NOT A PRODUCER.

Same discipline as `run_d_stage1.py`: the frozen producer stays a library and
this caller supplies only WHERE the artifact goes and WHICH seed — both
declared, neither an analytic choice. It verifies the producer's own bytes
before importing it, and `stage2` itself refuses without stage 1's artifact.
"""
import hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pairs_d as D

PRODUCER_SHA16 = "84011269d00eea6b"     #: frozen [3366], three seats
STAGE1_SHA16 = "9ae70405b23d96fe"       #: cleared [3355]/[3356]
SEED = 20260731                         #: §D5, inherited from B:79
S1 = os.path.join(CAMPAIGN, "results", "result_d_stage1.json")
OUT = os.path.join(CAMPAIGN, "results", "result_d_stage2.json")

got = hashlib.sha256(open(os.path.join(HERE, "pairs_d.py"), "rb")
                     .read()).hexdigest()[:16]
if got != PRODUCER_SHA16:
    raise SystemExit(f"REFUSING: pairs_d.py is {got}, frozen is {PRODUCER_SHA16}")
print(f"producer {got}  VERIFIED", flush=True)

built = D.build()
out, sha = D.stage2(built, S1, STAGE1_SHA16, OUT, seed=SEED)
print(f"stage-1 gate PASSED against {STAGE1_SHA16}", flush=True)
print(f"\nSTAGE 2 ARTIFACT  {OUT}\n  sha256[:16]  {sha}\n", flush=True)

for name in ("h1_signed", "arousal", "val_extrem", "dom_extrem"):
    a = out["arms"][name]
    if a.get("status") == "NOT TESTED":
        print(f"{name}  (family {a['family']})  *** NOT TESTED *** {a['why']}")
        continue
    p0 = a["per_t"]["0.00"]
    print(f"{name}  (family {a['family']})")
    print(f"    PRIMARY t=0.00  n {p0['n']}  D {p0['D']:+.5f}  "
          f"p {p0['p']:.5f} ({p0['p_convention']})  reject {p0['reject']}")
    print(f"      A_marked {p0['A_marked']:+.5f}   "
          f"A_unmarked {p0['A_unmarked']:+.5f}")
    print(f"    §D3 {a['reading_rule']['verdict']}: {a['reading_rule']['why']}")
    print(f"    §D6d {a['mde_reading']['reading']}")
    for t in ("0.01", "0.02", "0.05", "0.10", "0.20"):
        c = a["per_t"][t]
        if c["status"] != "ok":
            print(f"      t={t}  {c['status']}")
        else:
            print(f"      t={t}  n {c['n']:<4} D {c['D']:+.5f}  "
                  f"p {c['p']:.5f}  collapsed {c['collapsed']}")
