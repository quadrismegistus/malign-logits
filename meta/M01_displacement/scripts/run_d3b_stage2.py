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
PRODUCER_SHA = "6ec1601c21fea6f6"    #: FROZEN, committed, locked
STAGE1 = os.path.join(CAMPAIGN, "results", "result_d3b_stage1.json")
STAGE1_SHA16 = "ce6b215c5af138bc"    #: verified by shasum, three times
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
    _prior = hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16]
    print(f"SUPERSEDING: {os.path.basename(OUT)} @ {_prior}", flush=True)

for mod, want in (("pairs_d.py", D_SHA), ("pool_decomp_d3b.py", PRODUCER_SHA)):
    got = hashlib.sha256(open(os.path.join(HERE, mod), "rb").read()).hexdigest()[:16]
    if got != want:
        raise SystemExit(f"REFUSING: {mod} is {got}, frozen is {want}")
    print(f"{mod:22s} {got}  VERIFIED", flush=True)

coll = P.collect(verbose=False)
payload, sha = P.stage2(coll, STAGE1, STAGE1_SHA16, OUT, seed=P.SEED)
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
