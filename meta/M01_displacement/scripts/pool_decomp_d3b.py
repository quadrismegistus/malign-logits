#!/usr/bin/env python3
"""D3b: decomposing D2's effect against pool extremity. FROZEN f02f59d403906503.

**THIS PRODUCER IS WRITTEN AGAINST THE FROZEN REGISTRATION AND CITES IT BY
SECTION AT EVERY DECISION.** Where the registration is silent, this file RAISES
rather than chooses -- see `RELABEL_SORT_KEY`.

    §2   the bracket: relabel (confound side) and intercept (residual side)
    §3   the estimator fork, the 0.60 reliability floor, the weighting fork
    §7   the reading rule and the DECLARED RATIO
    §8   population, D's frozen producer, C v6's edge -- all inherited

**STAGE SEPARATION, inherited from D's two-stage split and for D's reason:**
§3's floor turns the reliability number into a fork, so the reliability MUST be
on the record before any b0 exists. STAGE 1 emits reliability and the regressor
diagnostics and NO D, NO b0, NO ratio. STAGE 2 refuses to run without stage 1's
hash.
"""

import collections
import hashlib
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pairs_d as D                      #: FROZEN 84011269d00eea6b, §8

#: D3b RE-FROZEN after Amendment A. Verified at this seat: three consecutive
#: re-hashes, 19,102 bytes, -r--r--r--, in HEAD.
REGISTRATION_SHA16 = "f02f59d403906503"
AMENDMENT_A_SHA16 = "b69b3e7d3e5edf68"

#: §D6b's arms, by name. D2 read val_extrem and dom_extrem; D3b decomposes both.
D3B_ARMS = ("val_extrem", "dom_extrem")

#: **THE BENCHMARK IS READ FROM D2's ARTIFACT, NEVER TYPED.** It was a dict of
#: `{"val_extrem": 0.01511, ...}` copied from a docket post -- the ROUNDED
#: PRINTED values -- while the residual side divided by `mean(D_pair)` from the
#: data. Two denominators 0.03% apart under a sentence claiming ONE scale
#: ([3564]). The numbers were nearly right and the stated property was false.
#:
#: **AND THE FIX IS A CHECK, NOT A BETTER CONSTANT.** Sourcing it from
#: `D_pair_mean` would fix the arithmetic and destroy the evidence that D2's
#: RESULT and this producer's MEAN are the same number. They are bit-identical
#: on both arms; that is a FACT and it is asserted, not assumed. A mismatch
#: means the two populations are not the same 632 -- a finding, not a rounding
#: question -- and stage 2 STOPS.
D2_READ = "result_d2_stage2.json"
D2_READ_SHA16 = "a1d712093155f32c"     #: the read cited in D3b's own preamble


def d2_observed():
    """D2's D per arm, at STORED precision, from a hash-gated artifact."""
    path = os.path.join(CAMPAIGN, "results", D2_READ)
    with open(path) as fh:
        blob = fh.read()
    got = hashlib.sha256(blob.encode()).hexdigest()[:16]
    if got != D2_READ_SHA16:
        raise SystemExit(
            f"BENCHMARK GATE FAILED: {D2_READ} hashes {got}, expected "
            f"{D2_READ_SHA16}. D3b does not decompose an unidentified result.")
    d = json.loads(blob)
    return {a: d["arms"][a]["per_t"]["0.00"]["D"] for a in D3B_ARMS}


#: §3's floor, declared in the registration BEFORE any reliability existed.
RELIABILITY_FLOOR = 0.60

#: §3/§4. `gap_pair` is defined in §4 as mean_abs_z (MARKED - UNMARKED); its
#: tabled figures (n 632, NEG 215, min -0.4193) reproduce ONLY under the
#: unweighted form, which is what pins this. The |delta|-weighted gap is §3's
#: fixed-reading SENSITIVITY.
PRIMARY_REGRESSOR = "mean_abs_z_unweighted"
SENSITIVITY_REGRESSOR = "mean_abs_z_weighted"

#: **§2 SAYS "reassign role BY POOL EXTREMITY" AND NAMES NO MEASURE.**
#: Three constructions give three sorts and therefore three `relabelled D`
#: values -- the numerator of §7's declared ratio. Raised at [3475]; until the
#: pen rules, EVERY construction is computed and NONE is primary. Setting this
#: to a single key is the ruling's job, not this file's.
RELABEL_SORT_KEYS = ("mean_abs_z_unweighted", "tail_ge_1_unweighted",
                     "mean_abs_z_weighted")
#: §A2 IN FORCE (D3b-A, RH 2026-08-03): the sort key is §4's declared
#: regressor. GROUND: the bracket's two sides must read the SAME measure or the
#: decomposition compares unlike quantities -- internal coherence, not a number.
RELABEL_PRIMARY = PRIMARY_REGRESSOR

DIM_OF_ARM = {"val_extrem": "valence", "dom_extrem": "dominance"}

SEED = 20260803


# ══════════════════════════════════════════════════════════════════════════
# pools
# ══════════════════════════════════════════════════════════════════════════
def member_pool(built, text, dim):
    """Every qualifying word of one member, as (abs z, weight).

    **THE POOL IS THE POOL THE STATISTIC READS** -- `built["cells"]` after the
    function-word filter, the norm lookup and the >=3-per-role bar. Profiling
    any other pool would answer a question nobody asked (pool_extremity.py's
    rule, kept).
    """
    vals, wts = [], []
    for c in built["cells"].get(text, {}).values():
        for z, w in zip(c["zs"], c["ws"]):
            vals.append(abs(z[dim]))
            wts.append(w)
    return vals, wts


def pool_stat(vals, wts, key):
    """One scalar per member under a NAMED construction. No default."""
    if not vals:
        return None
    if key == "mean_abs_z_unweighted":
        return st.mean(vals)
    if key == "mean_abs_z_weighted":
        tw = sum(wts)
        return (sum(v * w for v, w in zip(vals, wts)) / tw) if tw > 0 else None
    if key == "tail_ge_1_unweighted":
        return st.mean([1.0 if v >= 1.0 else 0.0 for v in vals])
    raise ValueError(f"unknown pool construction: {key!r}")


def gaps(built, pairs, dim, key):
    """MARKED - UNMARKED per pair under one construction. None if either empty."""
    out = {}
    for pid, mem in pairs.items():
        a = pool_stat(*member_pool(built, mem["MARKED"], dim), key)
        b = pool_stat(*member_pool(built, mem["UNMARKED"], dim), key)
        out[pid] = None if (a is None or b is None) else a - b
    return out


# ══════════════════════════════════════════════════════════════════════════
# reliability -- §3, a STAGE 1 quantity
# ══════════════════════════════════════════════════════════════════════════
def mass_balanced_split(keys, sizes):
    """Split an edge set into halves with MASS balanced. §3 / §A4, both declared.

    Greedy longest-processing-time: size DESCENDING, then **§A4's tie-break --
    CELL KEY (family, position) ASCENDING, NO RANDOM SEED.** A key-order
    tie-break needs no constant, cannot be mis-transcribed, and is reproducible
    by anyone holding the data without holding the document.

    **THE EQUAL-MASS STEP IS DECLARED TOO (§A4), because the greedy rule is
    undefined at its own first step: at the first edge both halves hold zero
    and EVERY split hits it.** On equal mass, the half holding FEWER EDGES; if
    the counts are equal too, half A. Derived from an ordering already
    declared, so no new constant enters.

    §3's reason for balancing at all: split-half assumes parallel forms, an
    unbalanced split UNDERSTATES reliability, and understating it makes the
    disattenuation OVERSHOOT -- a bias against the residual side.
    """
    ha, hb, ma, mb = [], [], 0, 0
    for k in sorted(keys, key=lambda k: (-sizes[k], k)):
        if ma < mb:
            to_a = True
        elif mb < ma:
            to_a = False
        elif len(ha) != len(hb):           #: equal mass -> fewer edges
            to_a = len(ha) < len(hb)
        else:                              #: equal mass AND equal count -> A
            to_a = True
        if to_a:
            ha.append(k); ma += sizes[k]
        else:
            hb.append(k); mb += sizes[k]
    return ha, hb, ma, mb


def split_half_reliability(built, pairs, dim, key, admitted=None):
    """Spearman-Brown-corrected split-half reliability of `gap_pair`. §3 / §A4.

    **THE ESTIMAND IS THE PAIR'S GAP, NOT A MEMBER'S MEAN (§A4).** The
    disattenuation divides by the reliability of the regressor, and the
    regressor IS the difference. Member-level reliability runs higher, so using
    it UNDER-corrects -- leaving attenuation that §6.1 declares reads HIGH, IN
    OUR FAVOUR. The ambiguity had a direction.

    **THE SPLIT IS SHARED OVER THE UNION OF THE PAIR'S EDGES (§A4), NOT
    PER-MEMBER.** Under per-member splitting `half A` was not a referent: M's
    "A" and U's "A" were independently-constructed halves related only by
    construction order, so the half-gap differenced two arbitrarily-associated
    things and the code answered by having a loop order. Edge sets differ in
    631 of 632 pairs. **The union partitions EACH member's full edge set, so
    every half-gap is a miniature of the gap the regression consumes**; the
    intersection would drop a median of seven edges and 17 pairs against 8.

    Mass is POOLED word count (MARKED + UNMARKED at that edge); a member absent
    at an edge contributes none. Correlation is PEARSON -- Spearman-Brown
    presumes it.
    """
    ids = list(pairs) if admitted is None else list(admitted)
    xa, xb, skipped = [], [], 0
    imb = []
    for pid in ids:
        mem = pairs[pid]
        per = {r: built["cells"].get(t, {}) for r, t in mem.items()}
        union = set(per["MARKED"]) | set(per["UNMARKED"])
        if len(union) < 2:
            skipped += 1
            continue
        sizes = {e: (len(per["MARKED"][e]["zs"]) if e in per["MARKED"] else 0)
                    + (len(per["UNMARKED"][e]["zs"]) if e in per["UNMARKED"] else 0)
                 for e in union}
        ha, hb, ma, mb = mass_balanced_split(union, sizes)
        if (ma + mb) > 0:
            imb.append(abs(ma - mb) / (ma + mb))
        halves = {}
        empty = False
        for lab, es in (("A", ha), ("B", hb)):
            for role in ("MARKED", "UNMARKED"):
                vals, wts = [], []
                for e in es:
                    c = per[role].get(e)
                    if not c:
                        continue
                    for z, w in zip(c["zs"], c["ws"]):
                        vals.append(abs(z[dim])); wts.append(w)
                v = pool_stat(vals, wts, key)
                #: §A4's SKIP PREDICATE is on the REALIZED halves -- either
                #: member contributing no edge to either half -- not on a cell
                #: count. A member with 3 edges can still land them all in one
                #: half, so the realized skip count is >= the 8 that cannot
                #: satisfy it under ANY assignment.
                if v is None:
                    empty = True
                halves[(lab, role)] = v
        if empty:
            skipped += 1
            continue
        xa.append(halves[("A", "MARKED")] - halves[("A", "UNMARKED")])
        xb.append(halves[("B", "MARKED")] - halves[("B", "UNMARKED")])

    if len(xa) < 3:
        return None
    a, b = np.asarray(xa, float), np.asarray(xb, float)
    if a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])                     #: PEARSON, §A4
    sb = (2 * r) / (1 + r) if r > -1 else None             #: Spearman-Brown
    return {"n_pairs": len(xa), "n_skipped": skipped,
            "r_halves": r, "reliability_spearman_brown": sb,
            #: §A4: a near-miss is a NUMBER, not a branch. The fork is
            #: discontinuous and jumps IN OUR FAVOUR below the floor (§A7).
            "distance_from_floor": (None if sb is None
                                    else sb - RELIABILITY_FLOOR),
            "mass_imbalance_median": (st.median(imb) if imb else None),
            "mass_imbalance_max": (max(imb) if imb else None)}


# ══════════════════════════════════════════════════════════════════════════
# the two bracket sides -- §2
# ══════════════════════════════════════════════════════════════════════════
def ols(x, y):
    """b0, b1 and the pieces disattenuation needs. Plain, no library fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx = float(((x - mx) ** 2).sum())
    if sxx == 0:
        return None
    b1 = float(((x - mx) * (y - my)).sum() / sxx)
    b0 = float(my - b1 * mx)
    resid = y - (b0 + b1 * x)
    return {"n": n, "b0": b0, "b1": b1,
            "mean_x": float(mx), "mean_y": float(my),
            "sd_x": float(x.std(ddof=1)), "sd_y": float(y.std(ddof=1)),
            "rmse": float(math.sqrt((resid ** 2).mean()))}


def disattenuate(fit, reliability):
    """§3's PRIMARY estimator: b1 corrected for regressor unreliability.

    b1_true = b1_observed / reliability, then b0 re-derived through the means
    so the fitted line still passes through (mean_x, mean_y).

    **§6.1 IS THE REASON THIS IS PRIMARY AND SAYS WHICH WAY THE RAW ONE ERRS:**
    attenuation drags b0 toward the unadjusted mean, so the RAW intercept
    OVERSTATES the pool-independent share -- in our favour. The corrected form
    is the one whose bias does not run systematically our way (§3).
    """
    if not fit or not reliability or reliability <= 0:
        return None
    b1c = fit["b1"] / reliability
    return {"reliability": reliability, "b1_corrected": b1c,
            "b0_corrected": float(fit["mean_y"] - b1c * fit["mean_x"])}


def relabel_D(built, pairs, arm_A, dim, key, admitted):
    """§2's CONFOUND SIDE at FULL STRENGTH: role reassigned by pool extremity.

    Every pool and every movement stays exactly as measured; only the ROLE
    LABEL moves, and it moves to make role/pool concordance 1.0 -- the confound
    turned all the way up. **§6.3: this READS HIGH by construction, which is
    what makes it an UPPER BOUND on the pool-associated share rather than an
    estimate of it.**

    Ties (equal pool stat) keep the data's own labelling: a tie carries no
    pool-order information, so inventing one would manufacture concordance the
    data does not have.
    """
    vals, flipped, ties = [], 0, 0
    for pid in admitted:
        mem = pairs[pid]
        am = arm_A.get(mem["MARKED"]), arm_A.get(mem["UNMARKED"])
        if not am[0] or not am[1]:
            continue
        A_m = st.mean(am[0].values())
        A_u = st.mean(am[1].values())
        p_m = pool_stat(*member_pool(built, mem["MARKED"], dim), key)
        p_u = pool_stat(*member_pool(built, mem["UNMARKED"], dim), key)
        if p_m is None or p_u is None:
            continue
        if p_m > p_u:
            vals.append(A_m - A_u)
        elif p_u > p_m:
            vals.append(A_u - A_m); flipped += 1
        else:
            vals.append(A_m - A_u); ties += 1
    if not vals:
        return None
    return {"n": len(vals), "D_relabelled": float(st.mean(vals)),
            "n_flipped": flipped, "n_ties": ties,
            "concordance_in_data": (len(vals) - flipped - ties) / len(vals)}


# ══════════════════════════════════════════════════════════════════════════
# collection
# ══════════════════════════════════════════════════════════════════════════
def collect(built=None, verbose=False):
    """Everything both stages read, computed once. Returns; writes nothing."""
    if built is None:
        built = D.build(verbose=verbose)
    out = {"built": built, "pairs": built["pairs"], "arms": {}}
    for arm in D.ARMS:
        if arm[0] not in D3B_ARMS:
            continue
        arm_A, beta = D.arm_values(built["cells"], arm, None)
        rows = D.assemble(built, arm_A)
        #: **`admitted_at` RETURNS ROWS, NOT IDS.** Both stages and the relabel
        #: index `pairs` by id, so the id list is derived here once. The smoke
        #: test caught this passing rows into a dict subscript; 29 self-tests
        #: did not, because every one of them built its own inputs -- A TEST
        #: THAT CONSTRUCTS ITS ARGUMENTS CANNOT CATCH THE CALLER PASSING THE
        #: WRONG SHAPE.
        admitted_rows = D.admitted_at(rows, 0.0)
        out["arms"][arm[0]] = {
            "arm": arm, "arm_A": arm_A, "rows": rows,
            "admitted_rows": admitted_rows,
            "admitted": [r["pair_id"] for r in admitted_rows],
            "by_id": {r["pair_id"]: r for r in rows},
            "dim": DIM_OF_ARM[arm[0]],
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 -- reliability and regressor diagnostics. NO D. NO b0. NO RATIO.
# ══════════════════════════════════════════════════════════════════════════
def stage1(coll, out_path, seed=SEED):
    """§3's reliability, on the record BEFORE any intercept exists.

    **WHY THE SPLIT EXISTS HERE, and it is D's reason unchanged:** §3's floor
    turns reliability into a FORK -- below 0.60 the corrected intercept is
    reported UNSTABLE and the residual side falls back to the raw one. A fork
    whose input is computed alongside its output is a rule chosen after the
    answer is visible. This stage emits the input and nothing the fork decides.
    """
    payload = {
        "_what": "D3b STAGE 1. Reliability of the regressor + regressor "
                 "diagnostics. NO D, NO intercept, NO ratio, NO relabel.",
        "_registration": REGISTRATION_SHA16,
        "_producer_d": D.__file__,
        "_seed": seed,
        "_floor": RELIABILITY_FLOOR,
        "arms": {},
    }
    for name, a in coll["arms"].items():
        rec = {"n_admitted": len(a["admitted"]), "regressors": {}}
        for key in sorted(set(RELABEL_SORT_KEYS)
                          | {PRIMARY_REGRESSOR, SENSITIVITY_REGRESSOR}):
            g = gaps(coll["built"], coll["pairs"], a["dim"], key)
            vals = [g[p] for p in a["admitted"] if g.get(p) is not None]
            #: §A4: the correlation runs over the ADMITTED pairs -- the
            #: population the corrected estimator is fit on -- with the
            #: all-pairs figure BESIDE IT as a declared sensitivity.
            rel = split_half_reliability(coll["built"], coll["pairs"],
                                         a["dim"], key, a["admitted"])
            rel_all = split_half_reliability(coll["built"], coll["pairs"],
                                             a["dim"], key, None)
            rec["regressors"][key] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "median": st.median(vals) if vals else None,
                "n_negative": sum(1 for v in vals if v < 0),
                "n_positive": sum(1 for v in vals if v > 0),
                #: §4's tabled support, so the runner has a PRE-REGISTERED
                #: KNOWN ANSWER. These figures are in the FROZEN registration
                #: and were derived independently at both seats, so they cost
                #: nothing and they ask the only question a stage-1 artifact
                #: cannot ask of itself: IS THIS THE SAME DATA?
                "near_zero_bins": {f"{c:.2f}": sum(1 for v in vals
                                                   if abs(v) <= c)
                                   for c in (0.01, 0.02, 0.05)},
                "reliability": rel,
                "reliability_all_pairs": rel_all,     #: §A4 sensitivity
                "meets_floor": (None if not rel or
                                rel["reliability_spearman_brown"] is None
                                else rel["reliability_spearman_brown"]
                                >= RELIABILITY_FLOOR),
            }
        payload["arms"][name] = rec

    blob = json.dumps(payload, indent=2, sort_keys=True)
    with open(out_path, "w") as fh:
        fh.write(blob)
    return payload, hashlib.sha256(blob.encode()).hexdigest()


def require_stage1(path, expect_sha16):
    """STAGE 2 REFUSES WITHOUT STAGE 1'S HASH. Returns (payload, observed).

    **TWO SHAPES TAKEN FROM [3500].3, WHICH FOUND THEM IN THE EARLIER RUNNERS:**

    FAILS CLOSED. An ABSENT expectation raises rather than skipping the check.
    D's runners read `if expect_sha16 and got != expect_sha16`, so a falsy
    expected hash silently disabled the gate; it was populated everywhere, so
    nothing was ever wrong -- **and a gate that can be switched off by an empty
    string is not a gate, it is a gate-shaped default.**

    RETURNS WHAT IT FOUND. The caller prints the OBSERVED hash, never the
    expected constant. **A success line echoing its own expectation would have
    printed "PASSED" even under the fail-open shape above.**
    """
    if not expect_sha16:
        raise SystemExit("STAGE 1 GATE: no expected hash supplied. "
                         "An absent expectation is a REFUSAL, not a skip.")
    with open(path) as fh:
        blob = fh.read()
    got = hashlib.sha256(blob.encode()).hexdigest()[:16]
    if got != expect_sha16:
        raise SystemExit(
            f"STAGE 1 GATE FAILED: {path} hashes {got}, expected {expect_sha16}. "
            "Stage 2 does not run.")
    return json.loads(blob), got


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 -- the bracket
# ══════════════════════════════════════════════════════════════════════════
def stage2(coll, stage1_path, stage1_sha16, out_path, seed=SEED):
    """§2's two sides, §3's fork resolved by stage 1's number, §7's ratio."""
    s1, observed = require_stage1(stage1_path, stage1_sha16)
    #: the gate prints WHAT IT FOUND, not what it hoped for ([3500].3)
    print(f"stage-1 gate PASSED; {stage1_path} OBSERVED {observed}")
    payload = {
        "_what": "D3b STAGE 2. The bracket: relabel (upper bound on the "
                 "pool-associated share) and intercept (upper bound on the "
                 "pool-independent share). NO significance test (§7).",
        "_registration": REGISTRATION_SHA16,
        "_stage1_sha16": stage1_sha16,
        "_relabel_primary": RELABEL_PRIMARY,
        "arms": {},
    }
    observed = d2_observed()
    payload["_benchmark_source"] = f"{D2_READ} @ {D2_READ_SHA16}, stored precision"
    for name, a in coll["arms"].items():
        s1a = s1["arms"][name]
        y = [a["by_id"][p]["D_pair"] for p in a["admitted"]]
        #: **THE FREE CROSS-ARTIFACT KNOWN ANSWER.** D2's result and this
        #: producer's mean are two quantities computed by two producers on two
        #: occasions; §A5 says they are the same number. BIT-equality, not a
        #: tolerance -- a tolerance would pass the very rounding this replaces.
        mean_here = float(np.mean(y))
        if observed[name] != mean_here:
            raise SystemExit(
                f"BENCHMARK MISMATCH on {name}: D2's stored D is "
                f"{observed[name]!r}, mean(D_pair) here is {mean_here!r}. "
                "The populations are not the same 632. STOP -- this is a "
                "finding, not a rounding question.")
        rec = {"n": len(y), "D2_observed": observed[name],
               "D_pair_mean": float(np.mean(y)), "residual_side": {},
               "confound_side": {}}

        #: RESIDUAL SIDE -- §2, §3's fork
        for key, role in ((PRIMARY_REGRESSOR, "PRIMARY"),
                          (SENSITIVITY_REGRESSOR, "SENSITIVITY")):
            g = gaps(coll["built"], coll["pairs"], a["dim"], key)
            xs = [g[p] for p in a["admitted"]]
            keep = [i for i, v in enumerate(xs) if v is not None]
            fit = ols([xs[i] for i in keep], [y[i] for i in keep])
            rel = (s1a["regressors"][key]["reliability"] or {}).get(
                "reliability_spearman_brown")
            dis = disattenuate(fit, rel)
            meets = s1a["regressors"][key]["meets_floor"]
            rec["residual_side"][key] = {
                "role": role, "fit": fit, "reliability": rel,
                "meets_floor": meets, "disattenuated": dis,
                #: §3's floor, applied from stage 1's number, not re-decided
                "reported_b0": (dis["b0_corrected"] if (dis and meets)
                                else (fit["b0"] if fit else None)),
                "reported_estimator": ("disattenuated" if (dis and meets)
                                       else "RAW (corrected UNSTABLE, §3 floor)"),
            }
            #: §A5's DECLARED QUANTITY, NO THRESHOLD: the POOL-INDEPENDENT
            #: SHARE. Denominator is mean(D_pair) over THE SAME PAIRS THE
            #: REGRESSION FITS -- which IS D2's D, since the sign-flip
            #: statistic is that mean. So this share and §7's ratio sit on ONE
            #: denominator and both bracket sides are directly comparable.
            #: SIGNED, no absolute value: mean(D_pair) is POSITIVE on both arms,
            #: so a NEGATIVE share means the pool-independent component runs
            #: OPPOSITE the effect -- a finding, not something to hide.
            #: NOT bounded to [0,1]; a share outside it is a real outcome.
            b0 = rec["residual_side"][key]["reported_b0"]
            den = float(np.mean([y[i] for i in keep])) if keep else None
            rec["residual_side"][key]["denominator_mean_D_pair"] = den
            rec["residual_side"][key]["pool_independent_share"] = (
                None if (b0 is None or not den) else b0 / den)

        #: CONFOUND SIDE -- §2. Every construction until §2's silence is ruled.
        for key in RELABEL_SORT_KEYS:
            r = relabel_D(coll["built"], coll["pairs"], a["arm_A"],
                          a["dim"], key, a["admitted"])
            if r:
                r["ratio_to_D2"] = r["D_relabelled"] / observed[name]
            if r:
                r["role"] = ("PRIMARY (§A2)" if key == RELABEL_PRIMARY
                             else "sensitivity")
            rec["confound_side"][key] = r
        payload["arms"][name] = rec

    blob = json.dumps(payload, indent=2, sort_keys=True)
    with open(out_path, "w") as fh:
        fh.write(blob)
    return payload, hashlib.sha256(blob.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════
# self-tests
# ══════════════════════════════════════════════════════════════════════════
def selftest(verbose=True):
    ok, fail = 0, []
    src = open(__file__).read()

    def check(label, cond):
        nonlocal ok
        if cond:
            ok += 1
            if verbose:
                print(f"  ok   {label}")
        else:
            fail.append(label)
            print(f"  FAIL {label}")

    #: --- pool_stat: three named constructions, no default ---
    vals, wts = [0.0, 1.0, 3.0], [0.5, 1.5, 2.5]
    check("mean_abs_z_unweighted is the plain mean",
          abs(pool_stat(vals, wts, "mean_abs_z_unweighted") - 4/3) < 1e-12)
    check("mean_abs_z_weighted is mass-weighted",
          abs(pool_stat(vals, wts, "mean_abs_z_weighted")
              - (0*0.5 + 1*1.5 + 3*2.5)/4.5) < 1e-12)
    check("tail_ge_1 counts >= 1 inclusive",
          abs(pool_stat(vals, wts, "tail_ge_1_unweighted") - 2/3) < 1e-12)
    try:
        pool_stat(vals, wts, "whatever")
        check("unknown construction raises", False)
    except ValueError:
        check("unknown construction raises", True)
    check("empty pool is None", pool_stat([], [], "mean_abs_z_unweighted") is None)

    #: --- ols against a known line ---
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 7.0]                     #: y = 1 + 2x exactly
    f = ols(x, y)
    check("ols recovers the intercept", abs(f["b0"] - 1.0) < 1e-12)
    check("ols recovers the slope", abs(f["b1"] - 2.0) < 1e-12)
    check("ols returns None on a constant regressor", ols([1, 1, 1], y[:3]) is None)

    #: --- disattenuation: direction and the line through the means ---
    d = disattenuate(f, 0.5)
    check("disattenuation DOUBLES b1 at reliability 0.5",
          abs(d["b1_corrected"] - 4.0) < 1e-12)
    check("corrected line still passes through (mean_x, mean_y)",
          abs((d["b0_corrected"] + d["b1_corrected"] * f["mean_x"])
              - f["mean_y"]) < 1e-12)
    check("|b0| GROWS when b1 is corrected upward on positive-mean x",
          abs(d["b0_corrected"]) > abs(f["b0"]))
    check("reliability 1.0 is the identity",
          abs(disattenuate(f, 1.0)["b0_corrected"] - f["b0"]) < 1e-12)
    check("disattenuate refuses reliability 0", disattenuate(f, 0.0) is None)

    #: --- mass-balanced split ---
    sizes = {"a": 10, "b": 9, "c": 8, "d": 1}
    ha, hb, ma, mb = mass_balanced_split(list(sizes), sizes)
    #: **§A3: NO RANDOM SEED REACHES THE SPLIT.** Checked structurally, because
    #: the defect it replaces (a per-process-salted seed) produced two different
    #: reliabilities from identical inputs and no output-level test saw it.
    _split_src = src[src.index("def mass_balanced_split("):
                     src.index("def split_half_reliability(")]
    check("the split takes no seed parameter",
          "def mass_balanced_split(keys, sizes)" in _split_src)
    check("the split uses no RNG", "default_rng" not in _split_src
          and "random" not in _split_src)
    check("the split's tie-break is the cell key ascending",
          "(-sizes[k], k)" in _split_src)
    check("split covers every cell exactly once",
          sorted(ha + hb) == sorted(sizes) and not (set(ha) & set(hb)))
    check("greedy split balances mass to within the largest cell",
          abs(ma - mb) <= max(sizes.values()))
    #: the tie-break is the whole point: equal sizes must split by KEY, and the
    #: answer must not move when the input order does
    tied = {"z": 5, "a": 5, "m": 5, "b": 5}
    r1 = mass_balanced_split(["z", "a", "m", "b"], tied)
    r2 = mass_balanced_split(["b", "m", "a", "z"], tied)
    check("equal-size cells split by key, independent of input order", r1 == r2)
    check("the key-ordered split takes the alphabetical first", r1[0][0] == "a")

    #: --- STAGE SEPARATION, structurally ---
    #: **THE CHECK IS ON IDENTIFIERS, NOT WORDS, AND THE FIRST VERSION WAS NOT.**
    #: Grepping stage 1 for "ratio" matched its own `_what` disclaimer -- the
    #: sentence saying it computes NO ratio. A stage-separation test that a
    #: DISCLAIMER can fail is testing prose; these strings cannot occur except
    #: as calls or subscripts.
    s1 = src[src.index("def stage1("):src.index("def require_stage1(")]
    for forbidden in ('["D_pair"]', "relabel_D(", "disattenuate(",
                      #: `D2_OBSERVED` was in this list and the identifier no
                      #: longer exists, so that entry had become a check that
                      #: CANNOT FAIL. Renaming a forbidden symbol silently
                      #: retires the guard that forbade it.
                      "ratio_to_D2", "d2_observed("):
        check(f"stage1 contains no {forbidden}", forbidden not in s1)
    _s2 = src[src.index("def stage2("):src.index("def selftest(")]
    check("stage2 calls the gate", "require_stage1(" in _s2)
    #: **THE GUARDS MUST NAME SYMBOLS STAGE 2 ACTUALLY USES, or the list
    #: decays into prose.** `D2_OBSERVED` sat in the forbidden list after the
    #: identifier was renamed away, so that entry could no longer fail --
    #: RENAMING A FORBIDDEN SYMBOL SILENTLY RETIRES THE GUARD THAT FORBADE IT.
    #: Checked against stage 2's own body: a symbol stage 1 must not contain is
    #: only a guard if stage 2 contains it.
    for sym in ("relabel_D", "disattenuate", "ratio_to_D2", "d2_observed"):
        check(f"stage2 uses the symbol {sym!r} that stage1 is forbidden",
              sym in _s2)
    check("stage2 ASSERTS the benchmark against the local mean",
          "BENCHMARK MISMATCH" in _s2 and "!=" in _s2)
    _gate = src[src.index("def require_stage1("):
                src.index("# ═", src.index("def require_stage1("))]
    check("the gate compares hashes and exits", "SystemExit" in _gate)
    #: [3500].3's two shapes, checked BEHAVIOURALLY -- a gate that can be
    #: disabled by an empty string is a gate-shaped default
    import tempfile as _tf
    with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write('{"x": 1}'); _tmp = fh.name
    _real = hashlib.sha256(open(_tmp).read().encode()).hexdigest()[:16]
    for absent in (None, "", 0):
        try:
            require_stage1(_tmp, absent); check(f"absent expectation {absent!r} REFUSES", False)
        except SystemExit:
            check(f"absent expectation {absent!r} REFUSES", True)
    try:
        require_stage1(_tmp, "0" * 16); check("wrong expectation refuses", False)
    except SystemExit:
        check("wrong expectation refuses", True)
    _payload, _obs = require_stage1(_tmp, _real)
    check("the gate RETURNS the observed hash, not the expected constant",
          _obs == _real and _payload == {"x": 1})
    check("stage2 prints the OBSERVED hash",
          "OBSERVED {observed}" in src)
    os.unlink(_tmp)

    #: --- §A2 IN FORCE: the sort key is the DECLARED REGRESSOR ---
    check("RELABEL_PRIMARY is §4's declared regressor (§A2)",
          RELABEL_PRIMARY == PRIMARY_REGRESSOR)
    check("the primary sort key is one of the tabled constructions",
          RELABEL_PRIMARY in RELABEL_SORT_KEYS)

    #: --- §A4's EQUAL-MASS RULE, which the greedy rule left undefined ---
    #: at the first edge both halves hold zero, so EVERY split hits it
    one = mass_balanced_split(["b", "a"], {"a": 5, "b": 5})
    check("equal size -> key order decides, 'a' first", one[0][0] == "a")
    check("first edge goes to half A (equal mass, equal count)", one[0] == ["a"])
    check("the second edge goes to B (A now heavier)", one[1] == ["b"])
    #: equal mass, UNEQUAL counts -> the half holding FEWER edges
    two = mass_balanced_split(["p", "q", "r"], {"p": 4, "q": 2, "r": 2})
    check("equal mass with unequal counts -> fewer edges wins",
          two[0] == ["p"] and sorted(two[1]) == ["q", "r"])
    check("the declared rule balances mass exactly here", two[2] == two[3] == 4)
    check("every sort key is a construction pool_stat knows",
          all(pool_stat([1.0], [1.0], k) is not None for k in RELABEL_SORT_KEYS))

    #: --- §A4's SHARED-OVER-UNION SPLIT, on a case that DISCRIMINATES ---
    #: Each pair's members share ONE edge, so the INTERSECTION has <2 and an
    #: intersection implementation skips every pair and returns None. The UNION
    #: has 3 edges and every member reaches both halves, so all three compute.
    #: **A test that both readings pass tests nothing; this one separates them.**
    def _cell(zs):
        return {"zs": [{"valence": v} for v in zs], "ws": [1.0] * len(zs)}
    ubuilt = {"cells": {}}
    upairs = {}
    for i, off in enumerate((0.0, 0.7, 1.4)):
        m, u = f"M{i}", f"U{i}"
        ubuilt["cells"][m] = {("f", "e1"): _cell([1.0 + off, 2.0]),
                              ("f", "e2"): _cell([0.5 + off])}
        ubuilt["cells"][u] = {("f", "e1"): _cell([0.4, 1.1 + off]),
                              ("f", "e3"): _cell([2.2 - off])}
        upairs[f"p{i}"] = {"MARKED": m, "UNMARKED": u}
    rel = split_half_reliability(ubuilt, upairs, "valence",
                                 "mean_abs_z_unweighted")
    check("union split computes pairs whose INTERSECTION is too small",
          rel is not None and rel["n_pairs"] == 3 and rel["n_skipped"] == 0)
    check("the reliability record carries the distance from the floor",
          rel["distance_from_floor"] is not None
          and abs(rel["distance_from_floor"]
                  - (rel["reliability_spearman_brown"] - RELIABILITY_FLOOR)) < 1e-12)
    #: a member that reaches only ONE half is skipped -- the REALIZED-halves
    #: predicate, not a cell count
    ubuilt["cells"]["U0"] = {("f", "e1"): _cell([0.4, 1.1])}
    rel2 = split_half_reliability(ubuilt, upairs, "valence",
                                  "mean_abs_z_unweighted")
    check("a member absent from one half skips the pair", rel2 is None
          or rel2["n_skipped"] >= 1)

    #: --- relabel: a hand-built case with a known answer ---
    built = {"cells": {
        "M": {"e": {"zs": [{"valence": 2.0}, {"valence": 2.0}], "ws": [1.0, 1.0]}},
        "U": {"e": {"zs": [{"valence": 0.0}, {"valence": 0.0}], "ws": [1.0, 1.0]}},
        "P": {"e": {"zs": [{"valence": 0.0}, {"valence": 0.0}], "ws": [1.0, 1.0]}},
        "Q": {"e": {"zs": [{"valence": 5.0}, {"valence": 5.0}], "ws": [1.0, 1.0]}},
    }}
    prs = {"p1": {"MARKED": "M", "UNMARKED": "U"},     #: pool order AGREES
           "p2": {"MARKED": "P", "UNMARKED": "Q"}}     #: pool order DISAGREES
    arm_A = {"M": {"e": 1.0}, "U": {"e": 0.0},
             "P": {"e": 1.0}, "Q": {"e": 0.0}}
    r = relabel_D(built, prs, arm_A, "valence", "mean_abs_z_unweighted",
                  ["p1", "p2"])
    #: **THE SHAPE CONTRACT, ADDED AFTER THE SMOKE TEST FOUND IT.** `collect`
    #: must hand the relabel PAIR IDS; `D.admitted_at` returns ROWS. The tests
    #: below build their own id list and so could never have caught it.
    check("collect exposes ids and rows under distinct names",
          "admitted_rows" in src[src.index("def collect("):src.index("# ═", src.index("def collect("))]
          and '"admitted": [r["pair_id"]' in src)
    check("relabel flips exactly the discordant pair", r["n_flipped"] == 1)
    check("relabel D is the mean after flipping: (1 + -1)/2 = 0",
          abs(r["D_relabelled"] - 0.0) < 1e-12)
    check("relabel reports the data's concordance", r["concordance_in_data"] == 0.5)
    #: a TIE keeps the data's labels rather than inventing concordance
    built["cells"]["Q"] = built["cells"]["P"]
    r2 = relabel_D(built, prs, arm_A, "valence", "mean_abs_z_unweighted",
                   ["p1", "p2"])
    check("a tie keeps the data's own labelling", r2["n_ties"] == 1
          and r2["n_flipped"] == 0)
    check("a tie is NOT counted as concordant",
          abs(r2["concordance_in_data"] - 0.5) < 1e-12)

    #: **THE PRINTED COUNT NAMES ITS UNIT AND THE TOTAL IS HOISTED.**
    #: `audit_d.check_counts_named_beside_fields` flagged the original line for
    #: emitting a bare length inside an f-string: "35/35 passed" names nothing
    #: counted. Hoisting the total removes the call from the emitting line
    #: entirely, which satisfies the heuristic's PURPOSE and not only its regex.
    n_checks = ok + len(fail)
    print(f"\n{ok}/{n_checks} checks passed"
          + ("" if not fail else f"; FAILED: {fail}"))
    return not fail


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
