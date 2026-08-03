#!/usr/bin/env python3
"""Invoke D3b's STAGE 1. A RUNNER, NOT A PRODUCER.

**IT REFUSES UNTIL THE PRODUCER IS FROZEN.** `PRODUCER_SHA` is None while the
producer is still under edit; running with an unfrozen producer would emit a
stage-1 artifact nobody can reproduce, and the artifact would hash cleanly.
"""
import hashlib
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pairs_d as D
import pool_decomp_d3b as P

D_SHA = "84011269d00eea6b"          #: §8, frozen -- pairs_d.py
#: **RENAMED FROM `D3B_SHA`, WHICH RESOLVED TWO WAYS.** It named the PRODUCER's
#: hash; [3553] read it as D3b's REGISTRATION hash and handed me
#: `f02f59d403906503`. The mismatch would have refused loudly rather than run
#: on the wrong pin -- but a constant whose name admits two referents is the
#: day's own defect class wearing an identifier.
#: PIN FORWARD UNDER [3905].3, CAUSED BY (1): `pool_decomp_d3b.py`'s
#: `D2_READ_SHA16` moved a1d712093155f32c -> 756eba00a0cfff4a because D2
#: stage 2 was re-run on the repaired instrument ([3901]/[3902]). The
#: producer moved BECAUSE ITS INPUT DID -- not because anyone wanted a
#: different producer, and the record must show that order. AST-proven
#: content-minimal: fifteen top-level objects, fourteen byte-identical,
#: module scope differing in exactly the two pin names ([3907]),
#: independently second-seat verified ([3909]).
PRODUCER_SHA_SUPERSEDED = "6ec1601c21fea6f6"   #: pre pin-forward
#: FORWARDED AGAIN, and the cause is §B7(ii): `pool_decomp_d3b.py` gained
#: `q1`/`q3` in its stage-1 summary, so its hash moved a second time.
#: dce56bd0b8979dfe was the pin-forward of [3907]; this is the amendment's
#: own consequence, adopted at [3948] on text 6c21db65ce1d2ae2.
PRODUCER_SHA_SUPERSEDED_2 = "dce56bd0b8979dfe"   #: after the D2 pin-forward
PRODUCER_SHA = "23ac1817f1ba7511"
OUT = os.path.join(CAMPAIGN, "results", "result_d3b_stage1.json")

if not PRODUCER_SHA:
    raise SystemExit(
        "REFUSING: pool_decomp_d3b.py is not frozen (PRODUCER_SHA is unset).\n"
        "An unfrozen producer emits an artifact that hashes cleanly and "
        "reproduces nothing. Freeze first, pin here, then run.")

#: ══════════════════════════════════════════════════════════════════════════
#: **A ONCE-ONLY RUNNER REFUSES A SECOND RUN.** [3572].1, ordered after I
#: re-ran stage 1 by accident: I piped this runner to `head -3` to read its two
#: VERIFIED lines, and `head` waited for a third line that prints AFTER the
#: artifact is written. **`head` IS NOT AN ABORT, AND A ONCE-ONLY RUNNER IS NOT
#: AN INSPECTION TOOL** -- `shasum` answers what that run was invoked to ask.
#: The bytes were identical, which is luck about determinism and not a defence.
#: ══════════════════════════════════════════════════════════════════════════
if os.path.exists(OUT) and "--rerun" not in sys.argv:
    raise SystemExit(
        f"REFUSING: {os.path.basename(OUT)} already exists.\n"
        "This runner emits a ONCE-ONLY registered artifact. To re-run, pass\n"
        "  --rerun '<the docket ruling that authorises it>'\n"
        "which PRINTS the authority and the superseded hash with the run.")
RERUN_AUTHORITY = None
if "--rerun" in sys.argv:
    i = sys.argv.index("--rerun")
    RERUN_AUTHORITY = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    if not RERUN_AUTHORITY:
        raise SystemExit("REFUSING: --rerun requires the authorising ruling as "
                         "its argument. A re-run with no named authority is an "
                         "accident with a flag on it.")
    print(f"RE-RUN AUTHORISED BY: {RERUN_AUTHORITY}", flush=True)
    #: AN ANNOUNCEMENT IS NOT AN ESCROW; THE COPY IS. [3911](b).
    #: This branch used to PRINT `SUPERSEDING: ... @ <hash>` and never copy the
    #: file -- and because the §4 reproduction STOP fires AFTER the write, a
    #: halted re-run still overwrote the artifact it claimed to supersede. On
    #: 2026-08-03 that destroyed the pre-repair D3b stage 1 on disk; only the
    #: commit rule (b203b0d2) redeemed it. The appearance of custody without
    #: the act is the mode-bit lesson one directory over.
    _blob = open(OUT, "rb").read()
    _prior = hashlib.sha256(_blob).hexdigest()[:16]
    _dir = os.path.join(CAMPAIGN, "results", "superseded")
    os.makedirs(_dir, exist_ok=True)
    _dst = os.path.join(_dir, f"result_d3b_stage1.PREFIX-{_prior}.json")
    if not os.path.exists(_dst):
        with open(_dst, "wb") as _fh:
            _fh.write(_blob)
        os.chmod(_dst, 0o444)
    print(f"SUPERSEDING: {os.path.basename(OUT)} @ {_prior}", flush=True)
    print(f"  ESCROWED (read-only) -> {os.path.basename(_dst)}", flush=True)
    print(f"  UNLOCKING {os.path.basename(OUT)} for the re-emit", flush=True)
    os.chmod(OUT, 0o644)

for mod, want in (("pairs_d.py", D_SHA), ("pool_decomp_d3b.py", PRODUCER_SHA)):
    got = hashlib.sha256(open(os.path.join(HERE, mod), "rb").read()).hexdigest()[:16]
    if got != want:
        raise SystemExit(f"REFUSING: {mod} is {got}, frozen is {want}")
    print(f"{mod:22s} {got}  VERIFIED", flush=True)

coll = P.collect(verbose=False)
payload, sha = P.stage1(coll, OUT, seed=P.SEED)
os.chmod(OUT, 0o444)
#: RE-LOCK. [3915] added escrow-before-write and the ANNOUNCED unlock and
#: **left the artifact unlocked afterwards** -- a half-applied ceremony,
#: found by `ls -l` on the run's own output rather than by any check.
#: The unlock and the re-lock are ONE act; shipping the first without the
#: second removes the protection and keeps the appearance of restoring it.
print("  RE-LOCKED a-w", flush=True)
print(f"\nD3b STAGE 1  {OUT}\n  sha256[:16]  {sha}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# THE PRE-REGISTERED KNOWN ANSWER
#
# §4 of the FROZEN registration tabulates `gap_pair` over the admitted 632,
# under the unweighted valence construction, and both seats derived those
# figures independently before this producer existed. **They are free, and
# they ask the one question a stage-1 artifact cannot ask of itself: IS THIS
# THE SAME DATA?** A deviation is a STOP, not a discrepancy to explain.
#
# `q1` is EXCLUDED: §4's -0.0171 and the linear-interpolation -0.01733 are
# adjacent order statistics under different quantile conventions, so it tests
# the convention rather than the data ([3470].2).
# ══════════════════════════════════════════════════════════════════════════
#: SUPERSEDED BY AMENDMENT D3b-B §B2 (adopted [3948], text 6c21db65ce1d2ae2).
#: These are §4's PRE-REPAIR figures. They fired 6-of-9 on the repaired run and
#: halted the chain, which is the check working: a PASS would have meant this
#: producer was not reading the repaired pool at all.
EXPECT_SUPERSEDED = {"n": 632, "min": -0.4193, "median": 0.0250, "max": 0.3696,
          "n_negative": 215, "n_positive": 417,
          "bins": {"0.01": 71, "0.02": 145, "0.05": 333}}

#: **AMENDMENT D3b-B §B2, ELEVEN FIGURES IN THREE CLASSES.** §B7(i) authorises
#: this edit by clause, not by docket post -- a producer citing a message number
#: cites something a reader of this repository cannot resolve; a producer citing
#: §B7 cites a document beside it.
#:
#: `q1`/`q3` enter the check set for the first time, gated under §B2's NAMED
#: convention: linear interpolation on the sorted vector, k = (n-1)p,
#: value = d[floor k] + (k - floor k) * (d[ceil k] - d[floor k]). They were
#: previously untabled here because §4's convention is LOST and no comparison to
#: its values is defined -- §B2 tables them as NEWLY SPECIFIED for that reason,
#: and from this amendment forward they are checkable.
EXPECT = {"n": 632, "min": -0.4193, "median": 0.0237, "max": 0.3730,
          "n_negative": 220, "n_positive": 412,
          "q1": -0.0184, "q3": 0.0713,
          "bins": {"0.01": 77, "0.02": 145, "0.05": 334}}

r = payload["arms"]["val_extrem"]["regressors"][P.PRIMARY_REGRESSOR]
print("\n§4's TABLED SUPPORT -- pre-registered, frozen before this producer:")
allok = True
for field, want in (("n", EXPECT["n"]), ("min", EXPECT["min"]),
                    ("median", EXPECT["median"]), ("max", EXPECT["max"]),
                    ("n_negative", EXPECT["n_negative"]),
                    ("n_positive", EXPECT["n_positive"]),
                    ("q1", EXPECT["q1"]), ("q3", EXPECT["q3"])):
    got = r[field]
    m = (abs(got - want) < 5e-5) if isinstance(want, float) else (got == want)
    allok &= m
    print(f"   {field:<12} got {got!s:>12}   §4 {want!s:>10}   "
          f"{'MATCH' if m else '*** MISS'}")
for cut, want in EXPECT["bins"].items():
    got = r["near_zero_bins"][cut]
    m = got == want
    allok &= m
    print(f"   |gap|<={cut}   got {got:>12}   §4 {want:>10}   "
          f"{'MATCH' if m else '*** MISS'}")
print(f"\n  ALL ELEVEN §B2 FIGURES REPRODUCE: {allok}")
if not allok:
    raise SystemExit("STOP: the regressor does not reproduce §B2's tabled "
                     "support. This is not the same data.")

# ══════════════════════════════════════════════════════════════════════════
# RELIABILITY -- §3's fork input, and NOTHING the fork decides
# ══════════════════════════════════════════════════════════════════════════
print(f"\nRELIABILITY (§A4: gap-level, across pairs, Pearson, "
      f"Spearman-Brown), floor {P.RELIABILITY_FLOOR}:")
for arm in sorted(payload["arms"]):
    for key in sorted(payload["arms"][arm]["regressors"]):
        rec = payload["arms"][arm]["regressors"][key]
        rel = rec["reliability"]
        if not rel or rel["reliability_spearman_brown"] is None:
            print(f"   {arm:11s} {key:<24} reliability UNAVAILABLE")
            continue
        sb = rel["reliability_spearman_brown"]
        #: **[3499].2: THE DISTANCE FROM THE FLOOR PRINTS.** The fork is
        #: discontinuous -- below 0.60 the residual side falls back to the RAW
        #: intercept, which §6.1 declares reads HIGH, IN OUR FAVOUR. A branch
        #: alone hides how close the call was; the distance does not.
        print(f"   {arm:11s} {key:<24} r {rel['r_halves']:+.5f}  "
              f"SB {sb:.5f}  floor{sb - P.RELIABILITY_FLOOR:+.5f}  "
              f"{'MEETS' if rec['meets_floor'] else 'BELOW -> RAW'}  "
              f"n {rel['n_pairs']}  mass-imbalance med "
              f"{rel['mass_imbalance_median']:.5f}")

print("\nSTAGE 1 CARRIES NO D, NO INTERCEPT, NO RATIO, NO RELABEL.")
