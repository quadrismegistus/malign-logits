#!/usr/bin/env python3
"""Invoke D2's STAGE 1 at alpha 0.025. A RUNNER, NOT A PRODUCER."""
import hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pairs_d as D, pairs_d2 as D2

D_SHA, D2_SHA = "84011269d00eea6b", "e3facec24b6b6641"
SEED = 20260731
OUT = os.path.join(CAMPAIGN, "results", "result_d2_stage1.json")

for mod, want in (("pairs_d.py", D_SHA), ("pairs_d2.py", D2_SHA)):
    got = hashlib.sha256(open(os.path.join(HERE, mod), "rb").read()).hexdigest()[:16]
    if got != want:
        raise SystemExit(f"REFUSING: {mod} is {got}, frozen is {want}")
    print(f"{mod:14s} {got}  VERIFIED", flush=True)

built = D.build()

#: CUSTODY AROUND THE WRITE, [3830]/[3851]. Escrow read-only FIRST, unlock only
#: after it exists, announce it, re-lock after. THIRD runner tonight to need
#: this and the third to hit PermissionError without it -- the lock is doing its
#: job every time, and the block is inline rather than shared because the method
#: freeze forbids new machinery mid-campaign.
if os.path.exists(OUT):
    _prior = open(OUT, "rb").read()
    _ph = hashlib.sha256(_prior).hexdigest()[:16]
    _dir = os.path.join(CAMPAIGN, "results", "superseded")
    os.makedirs(_dir, exist_ok=True)
    _dst = os.path.join(_dir, f"result_d2_stage1.PREFIX-{_ph}.json")
    if not os.path.exists(_dst):
        with open(_dst, "wb") as _fh:
            _fh.write(_prior)
        os.chmod(_dst, 0o444)
    print(f"  escrowed prior artifact @ {_ph} -> {os.path.basename(_dst)}", flush=True)
    print(f"  UNLOCKING {os.path.basename(OUT)} for the re-emit", flush=True)
    os.chmod(OUT, 0o644)

payload, sha = D2.stage1_d2(built, OUT, seed=SEED)
os.chmod(OUT, 0o444)
print("  RE-LOCKED a-w", flush=True)
print(f"\nD2 STAGE 1  {OUT}\n  sha256[:16]  {sha}", flush=True)

#: THE PRE-REGISTERED KNOWN ANSWER, [3378].3 -- the SDs are alpha-independent
#: and MUST reproduce D's stage 1 exactly. A deviation is a STOP.
#: SUPERSEDED, NOT OVERWRITTEN ([3895], extending [3830]'s pin discipline to
#: EXPECTATIONS). This block encodes the PRE-REPAIR world: it was derived from
#: D stage 1 @ 9ae70405b23d96fe, produced under movement.py 28541cced0ec081b
#: (at 60d605c1) before the residual-as-faller repair. It FIRED TEN OF TEN on
#: the repaired run ([3894]) -- which is the chain working: D2's known answer
#: detected D's movement exactly where it had to, and a PASS here would have
#: meant D2 was not reading D stage 1 at all.
EXPECT_SUPERSEDED = {
 "val_extrem": {"0.00":(632,0.15864),"0.01":(632,0.15864),"0.02":(632,0.15864),
                "0.05":(629,0.15819),"0.10":(287,0.17895)},
 "dom_extrem": {"0.00":(632,0.17548),"0.01":(632,0.17548),"0.02":(632,0.17548),
                "0.05":(629,0.17584),"0.10":(287,0.19654)}}

#: THE LIVE EXPECTATION, DERIVED FROM THE INPUT'S RECORD AND NOT FROM THE RUN IT
#: GUARDS ([3895]): read out of `result_d_stage1.json` @ 15b529a7d9261c8b --
#: the repaired D stage 1, escrowed and two-seat verified at [3885]/[3886].
#: Pasting the numbers the failing run printed would make the check a copy of
#: its own subject; taking them from the upstream artifact keeps it independent.
EXPECT = {
 "val_extrem": {"0.00":(632,0.1561),"0.01":(632,0.1561),"0.02":(632,0.1561),
                "0.05":(629,0.1556),"0.10":(292,0.17245)},
 "dom_extrem": {"0.00":(632,0.1757),"0.01":(632,0.1757),"0.02":(632,0.1757),
                "0.05":(629,0.17605),"0.10":(292,0.19632)}}
print("\nPRE-REGISTERED CHECK ([3378].3) -- SDs are alpha-independent:")
allok = True
for arm, pts in EXPECT.items():
    for t, (en, esd) in pts.items():
        c = payload["arms"][arm]["per_t"][t]
        m = (c["n"] == en) and abs(c["sd_D_pair"] - esd) < 5e-6
        allok = allok and m
        print(f"   {arm:11s} t={t}  n {c['n']:<4} (exp {en:<4}) "
              f"sd {c['sd_D_pair']:.5f} (exp {esd:.5f})  {'MATCH' if m else '*** MISS'}")
print(f"\n  ALL TEN SDs AND SIX n's MATCH: {allok}")

f = payload["falsifier"]
print(f"\n§2 FALSIFIER  threshold {f['threshold']}  tripped_at {f['tripped_at']}")
print(f"  STRUCTURE STANDS: {f['structure_stands']}")
print("\nraw MDEs AT alpha 0.025, and the §D6d comparator is 0.025:")
for arm in D2.D2_ARMS:
    for t, c in sorted(payload["arms"][arm]["per_t"].items()):
        if c["status"] != "ok":
            print(f"   {arm:11s} t={t}  {c['status']}")
        else:
            print(f"   {arm:11s} t={t}  n {c['n']:<4} MDE {c['raw_mde']:.5f}"
                  f"  quotable-on-null {c['raw_mde'] < 0.025}"
                  f"  collapsed {c['collapsed']}")
