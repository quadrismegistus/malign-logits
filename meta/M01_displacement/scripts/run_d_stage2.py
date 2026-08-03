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
#: PIN FORWARD UNDER [3828], protocol [3830]. The superseded value STAYS: it is
#: the only record that a swap happened, and this is the first swap tonight where
#: the upstream GENUINELY CHANGED rather than drifted -- stage 1 is MOVED, 84
#: differing leaves and NOT ONE of them float noise ([3871]).
#: PROVENANCE, proven not assumed: `9ae70405b23d96fe` was produced by THIS
#: producer (84011269d00eea6b) under the PRE-REPAIR movement.py (28541cced0ec081b,
#: at 60d605c1) -- reproduced exactly to a scratch path. `15b529a7d9261c8b` is the
#: same producer under the repaired instrument, reproduced twice independently.
STAGE1_SHA16_SUPERSEDED = "9ae70405b23d96fe"   #: pre-repair, cleared [3355]/[3356]
STAGE1_SHA16 = "15b529a7d9261c8b"       #: post-repair re-emit, [3871]
SEED = 20260731                         #: §D5, inherited from B:79
S1 = os.path.join(CAMPAIGN, "results", "result_d_stage1.json")
OUT = os.path.join(CAMPAIGN, "results", "result_d_stage2.json")

got = hashlib.sha256(open(os.path.join(HERE, "pairs_d.py"), "rb")
                     .read()).hexdigest()[:16]
if got != PRODUCER_SHA16:
    raise SystemExit(f"REFUSING: pairs_d.py is {got}, frozen is {PRODUCER_SHA16}")
print(f"producer {got}  VERIFIED", flush=True)

built = D.build()

#: CUSTODY AROUND THE WRITE, [3830]/[3851]. Escrow read-only FIRST, unlock only
#: after it exists, announce the unlock, re-lock after. If the write dies between
#: unlock and lock, the escrow is the copy that survives.
if os.path.exists(OUT):
    _prior = open(OUT, "rb").read()
    _ph = hashlib.sha256(_prior).hexdigest()[:16]
    _dir = os.path.join(CAMPAIGN, "results", "superseded")
    os.makedirs(_dir, exist_ok=True)
    _dst = os.path.join(_dir, f"result_d_stage2.PREFIX-{_ph}.json")
    if not os.path.exists(_dst):
        with open(_dst, "wb") as _fh:
            _fh.write(_prior)
        os.chmod(_dst, 0o444)
    print(f"  escrowed prior artifact @ {_ph} -> {os.path.basename(_dst)}", flush=True)
    print(f"  UNLOCKING {os.path.basename(OUT)} for the re-emit", flush=True)
    os.chmod(OUT, 0o644)

out, sha = D.stage2(built, S1, STAGE1_SHA16, OUT, seed=SEED)
os.chmod(OUT, 0o444)
print("  RE-LOCKED a-w", flush=True)
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
