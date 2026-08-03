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
payload, sha = D2.stage1_d2(built, OUT, seed=SEED)
print(f"\nD2 STAGE 1  {OUT}\n  sha256[:16]  {sha}", flush=True)

#: THE PRE-REGISTERED KNOWN ANSWER, [3378].3 -- the SDs are alpha-independent
#: and MUST reproduce D's stage 1 exactly. A deviation is a STOP.
EXPECT = {
 "val_extrem": {"0.00":(632,0.15864),"0.01":(632,0.15864),"0.02":(632,0.15864),
                "0.05":(629,0.15819),"0.10":(287,0.17895)},
 "dom_extrem": {"0.00":(632,0.17548),"0.01":(632,0.17548),"0.02":(632,0.17548),
                "0.05":(629,0.17584),"0.10":(287,0.19654)}}
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
