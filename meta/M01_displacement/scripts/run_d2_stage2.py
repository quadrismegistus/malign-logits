#!/usr/bin/env python3
"""Invoke D2's STAGE 2. A RUNNER, NOT A PRODUCER."""
import hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pairs_d as D, pairs_d2 as D2

D_SHA, D2_SHA, S1_SHA = "84011269d00eea6b", "e3facec24b6b6641", "183722c556248709"
S1 = os.path.join(CAMPAIGN, "results", "result_d2_stage1.json")
OUT = os.path.join(CAMPAIGN, "results", "result_d2_stage2.json")

for mod, want in (("pairs_d.py", D_SHA), ("pairs_d2.py", D2_SHA)):
    got = hashlib.sha256(open(os.path.join(HERE, mod), "rb").read()).hexdigest()[:16]
    if got != want:
        raise SystemExit(f"REFUSING: {mod} is {got}, frozen is {want}")
print(f"producers VERIFIED", flush=True)

built = D.build()
out, sha = D2.stage2_d2(built, S1, S1_SHA, OUT, seed=20260731)
print(f"stage-1 gate PASSED against {S1_SHA}")
print(f"\nD2 STAGE 2  {OUT}\n  sha256[:16]  {sha}\n  alpha {out['_alpha']}  "
      f"structure {out['_structure']}\n")
for name in D2.D2_ARMS:
    a = out["arms"][name]; p0 = a["per_t"]["0.00"]
    print(f"{name}   TESTED {a['tested']}")
    print(f"    PRIMARY t=0.00  n {p0['n']}  D {p0['D']:+.5f}  p {p0['p']:.5f}"
          f"  alpha {p0['alpha']}  reject {p0['reject']}")
    print(f"      A_marked {p0['A_marked']:+.5f}   A_unmarked {p0['A_unmarked']:+.5f}")
    print(f"    §D3  {a['reading_rule']['verdict']}: {a['reading_rule']['why']}")
    print(f"    §D6d {a['mde_reading']['reading']}")
    for t in ("0.01", "0.02", "0.05", "0.10", "0.20"):
        c = a["per_t"][t]
        print(f"      t={t}  " + (c["status"] if c["status"] != "ok" else
              f"n {c['n']:<4} D {c['D']:+.5f}  p {c['p']:.5f}  "
              f"collapsed {c['collapsed']}"))
