#!/usr/bin/env python3
"""Invoke D3b's STAGE 2 — the bracket. A RUNNER, NOT A PRODUCER.

**THE GATE IS GIVEN THE SIXTEEN.** `stage1` returns the FULL digest and
`require_stage1` compares `hexdigest()[:16]`; a caller piping one into the other
REFUSES. That mislabel is recorded at [3558].4 and the frozen producer is not
edited for it — **the truncation is the caller's job and it happens HERE, once,
with the value verified against the file by `shasum` rather than copied from a
print statement whose label said sixteen and whose value was sixty-four.**
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pairs_d as D
import pool_decomp_d3b as P

D_SHA = "84011269d00eea6b"           #: §8, frozen -- pairs_d.py
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
STAGE1 = os.path.join(CAMPAIGN, "results", "result_d3b_stage1.json")
#: PIN FORWARD, and it is the FIRST time this constant could legitimately
#: move: until [3953] there was no CERTIFIED post-repair stage 1 to point
#: at -- the run of [3910] wrote an artifact its own STOP refused, and
#: forwarding to an artifact whose producer declined to certify it is the
#: laundering [3905] drew the line against. `3a7219afddb02569` is the run
#: whose eleven-figure STOP PASSED against Amendment D3b-B §B2.
#: The two superseded values, in order and both kept:
#:   ce6b215c5af138bc  pre-repair, the frozen chain
#:   f0cfe6d4ec61bb1c  written by the HALTED run; NEVER CERTIFIED, escrowed
#:                     only so the record shows what the STOP refused
STAGE1_SHA16_SUPERSEDED = "ce6b215c5af138bc"   #: pre-repair, frozen chain
STAGE1_SHA16_UNCERTIFIED = "f0cfe6d4ec61bb1c"  #: halted run, refused by its own STOP
STAGE1_SHA16 = "3a7219afddb02569"    #: STOP PASSED, eleven of eleven, [3953]
OUT = os.path.join(CAMPAIGN, "results", "result_d3b_stage2.json")

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
    #: AN ANNOUNCEMENT IS NOT AN ESCROW; THE COPY IS. [3911](b), applied here
    #: BEFORE this runner ever ran -- stage 1 carried the identical branch and
    #: it cost an overwrite on 2026-08-03, redeemed only by the commit rule.
    #: Same defect, same fix, found by auditing the sibling rather than by
    #: repeating the loss.
    _blob = open(OUT, "rb").read()
    _prior = hashlib.sha256(_blob).hexdigest()[:16]
    _dir = os.path.join(CAMPAIGN, "results", "superseded")
    os.makedirs(_dir, exist_ok=True)
    _dst = os.path.join(_dir, f"result_d3b_stage2.PREFIX-{_prior}.json")
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
payload, sha = P.stage2(coll, STAGE1, STAGE1_SHA16, OUT, seed=P.SEED)
os.chmod(OUT, 0o444)
#: RE-LOCK -- THE SECOND HALF. Stage 1 carried the identical omission and
#: it was found by `ls -l` on the run's output, not by a check ([3953]).
#: Same branch, same gap, fixed here BEFORE this runner ever emitted --
#: the sibling sweep that [3915] should have done for the escrow and did.
print("  RE-LOCKED a-w", flush=True)
print(f"\nD3b STAGE 2  {OUT}\n  sha256[:16]  {sha[:16]}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# THE BRACKET.  §7 / §A5: both sides on ONE denominator -- D2's own D, which
# IS mean(D_pair), so the two shares are directly comparable and their sum is
# interpretable.  NO significance test.  NO verdict language.  D3b reports two
# bounds and their gap; it does not re-adjudicate D2.
# ══════════════════════════════════════════════════════════════════════════
for arm in sorted(payload["arms"]):
    a = payload["arms"][arm]
    print(f"\n{'=' * 74}\n{arm}   n {a['n']}   D2 observed {a['D2_observed']:+.5f}"
          f"   mean(D_pair) {a['D_pair_mean']:+.8f}")

    print("\n  CONFOUND SIDE -- relabel at maximal sorting (UPPER BOUND on the "
          "pool-associated share)")
    for key in P.RELABEL_SORT_KEYS:
        r = a["confound_side"].get(key)
        if not r:
            print(f"    {key:<24} UNAVAILABLE")
            continue
        print(f"    {key:<24} {r['role']:<15} D {r['D_relabelled']:+.6f}"
              f"   ratio {r['ratio_to_D2']:+.4f}"
              f"   flipped {r['n_flipped']}  ties {r['n_ties']}"
              f"  concord(data) {r['concordance_in_data']:.4f}")

    print("\n  RESIDUAL SIDE -- intercept (UPPER BOUND on the pool-independent "
          "share)")
    for key, rec in sorted(a["residual_side"].items()):
        f = rec["fit"]
        if not f:
            print(f"    {key:<24} FIT UNAVAILABLE")
            continue
        share = rec["pool_independent_share"]
        print(f"    {key:<24} {rec['role']:<12} b1 {f['b1']:+.6f}"
              f"   raw b0 {f['b0']:+.6f}")
        print(f"    {'':<24} reliability {rec['reliability']:.5f}"
              f"  {'MEETS' if rec['meets_floor'] else 'BELOW -> RAW'}"
              f"   b1_corr {rec['disattenuated']['b1_corrected']:+.6f}"
              f"   b0_corr {rec['disattenuated']['b0_corrected']:+.6f}")
        print(f"    {'':<24} REPORTED b0 {rec['reported_b0']:+.6f}"
              f"  [{rec['reported_estimator']}]"
              f"   SHARE b0/mean(D_pair) {share:+.4f}")

    #: **THE GLOSS IS COMPUTED, NOT BOILERPLATE.** The first run printed
    #: "a sum above 1 is expected" unconditionally, including under the
    #: dominance sum of -0.3729 where it is nonsense ([3562].4a). A static
    #: explanation attached to a computed value it does not fit.
    prim = a["confound_side"].get(P.RELABEL_PRIMARY)
    res = a["residual_side"][P.PRIMARY_REGRESSOR]
    if prim and res["pool_independent_share"] is not None:
        print(f"\n  THE BRACKET, primary construction, one denominator:")
        print(f"    pool-ASSOCIATED   upper bound  {prim['ratio_to_D2']:+.4f}")
        print(f"    pool-INDEPENDENT  upper bound  "
              f"{res['pool_independent_share']:+.4f}")
        tot = prim["ratio_to_D2"] + res["pool_independent_share"]
        why = ("both bounds read HIGH (§6.1, §6.3), so a sum ABOVE 1 is "
               "expected and is not a contradiction" if tot > 1.0 else
               "the pool-associated bound is NEGATIVE, so the sum is not a "
               "partition of anything -- read the two bounds, not the sum"
               if prim["ratio_to_D2"] < 0 else
               "the two bounds do not exhaust D2; the sum is reported, not "
               "interpreted")
        print(f"    the two bounds SUM to {tot:+.4f}  -- {why}")

print(f"\n{'=' * 74}")
print("NO SIGNIFICANCE TEST. NO VERDICT. D2 STANDS AS READ (§7).")
