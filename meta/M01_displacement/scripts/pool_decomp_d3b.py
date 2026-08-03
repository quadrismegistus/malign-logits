#!/usr/bin/env python3
"""D3b: decomposing D2's effect against pool extremity. FROZEN e20c412c898b58fc.

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

REGISTRATION_SHA16 = "e20c412c898b58fc"

#: §D6b's arms, by name. D2 read val_extrem and dom_extrem; D3b decomposes both.
D3B_ARMS = ("val_extrem", "dom_extrem")

#: The benchmark being decomposed. §Preamble, read a1d712093155f32c.
D2_OBSERVED = {"val_extrem": 0.01511, "dom_extrem": 0.01655}

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
RELABEL_PRIMARY = None                  #: set by ruling; None => report all

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
def stable_seed(text):
    """A per-member seed that survives a process restart. See the call site."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) & 0xFFFF


def mass_balanced_split(cell_keys, sizes, seed):
    """Split a member's cells into halves with CELL MASS balanced. §3.

    **§3 states the reason and it is not cosmetic:** split-half assumes
    parallel forms, a member's cells differ in size, and an unbalanced split
    UNDERSTATES reliability -- which makes the disattenuation OVERSHOOT, i.e.
    errs toward a larger corrected b1 and a smaller b0. The bias would run
    against the residual side, so balancing is not a convenience.

    Greedy longest-processing-time: sort by size descending, add each cell to
    whichever half currently holds less mass. Deterministic given the seed only
    through the tie-break, which is what the seed is for.
    """
    rng = np.random.default_rng(seed)
    order = sorted(cell_keys,
                   key=lambda k: (-sizes[k], rng.random()))
    ha, hb, ma, mb = [], [], 0, 0
    for k in order:
        if ma <= mb:
            ha.append(k); ma += sizes[k]
        else:
            hb.append(k); mb += sizes[k]
    return ha, hb, ma, mb


def split_half_reliability(built, pairs, dim, key, seed=SEED):
    """Spearman-Brown-corrected split-half reliability of `gap_pair`. §3.

    **WHAT IS BEING MEASURED IS THE REGRESSOR, NOT THE OUTCOME.** §3 puts
    reliability here because §3's disattenuation divides b1 by it; a reliability
    of the wrong quantity would correct by the wrong amount.
    """
    half_gap = {"A": {}, "B": {}}
    per_member_mass = {}
    for pid, mem in pairs.items():
        halves = {}
        for role, text in mem.items():
            per_edge = built["cells"].get(text, {})
            if len(per_edge) < 2:
                halves = None
                break
            sizes = {k: len(c["zs"]) for k, c in per_edge.items()}
            #: **`hash()` IS SALTED PER PROCESS.** The first smoke run returned
            #: SB 0.9439 and the second 0.8272 on identical inputs, because the
            #: per-member seed was `hash(text)` -- so stage 1's reliability, the
            #: number §3's floor forks on, was not reproducible across runs.
            #: A REGISTERED QUANTITY THAT CHANGES WHEN THE PROCESS RESTARTS IS
            #: NOT A MEASUREMENT. sha256 is stable across processes and machines.
            ha, hb, ma, mb = mass_balanced_split(
                list(per_edge), sizes, seed ^ stable_seed(text))
            per_member_mass[text] = (ma, mb)
            hv = {}
            for lab, keys in (("A", ha), ("B", hb)):
                vals, wts = [], []
                for k in keys:
                    c = per_edge[k]
                    for z, w in zip(c["zs"], c["ws"]):
                        vals.append(abs(z[dim])); wts.append(w)
                hv[lab] = pool_stat(vals, wts, key)
            if hv["A"] is None or hv["B"] is None:
                halves = None
                break
            halves[role] = hv
        if not halves:
            continue
        for lab in ("A", "B"):
            half_gap[lab][pid] = (halves["MARKED"][lab]
                                  - halves["UNMARKED"][lab])

    common = sorted(set(half_gap["A"]) & set(half_gap["B"]))
    if len(common) < 3:
        return None
    xa = np.array([half_gap["A"][p] for p in common], float)
    xb = np.array([half_gap["B"][p] for p in common], float)
    if xa.std() == 0 or xb.std() == 0:
        return None
    r = float(np.corrcoef(xa, xb)[0, 1])
    sb = (2 * r) / (1 + r) if r > -1 else None      #: Spearman-Brown
    imb = [abs(a - b) / (a + b) for a, b in per_member_mass.values()
           if (a + b) > 0]
    return {"n_pairs": len(common), "r_halves": r,
            "reliability_spearman_brown": sb,
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
            rel = split_half_reliability(coll["built"], coll["pairs"],
                                         a["dim"], key, seed)
            rec["regressors"][key] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "median": st.median(vals) if vals else None,
                "n_negative": sum(1 for v in vals if v < 0),
                "n_positive": sum(1 for v in vals if v > 0),
                "reliability": rel,
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
    """STAGE 2 REFUSES WITHOUT STAGE 1'S HASH. D's gate, unchanged."""
    with open(path) as fh:
        blob = fh.read()
    got = hashlib.sha256(blob.encode()).hexdigest()[:16]
    if got != expect_sha16:
        raise SystemExit(
            f"STAGE 1 GATE FAILED: {path} hashes {got}, expected {expect_sha16}. "
            "Stage 2 does not run.")
    return json.loads(blob)


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 -- the bracket
# ══════════════════════════════════════════════════════════════════════════
def stage2(coll, stage1_path, stage1_sha16, out_path, seed=SEED):
    """§2's two sides, §3's fork resolved by stage 1's number, §7's ratio."""
    s1 = require_stage1(stage1_path, stage1_sha16)
    payload = {
        "_what": "D3b STAGE 2. The bracket: relabel (upper bound on the "
                 "pool-associated share) and intercept (upper bound on the "
                 "pool-independent share). NO significance test (§7).",
        "_registration": REGISTRATION_SHA16,
        "_stage1_sha16": stage1_sha16,
        "_relabel_primary": RELABEL_PRIMARY,
        "arms": {},
    }
    for name, a in coll["arms"].items():
        s1a = s1["arms"][name]
        y = [a["by_id"][p]["D_pair"] for p in a["admitted"]]
        rec = {"n": len(y), "D2_observed": D2_OBSERVED[name],
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

        #: CONFOUND SIDE -- §2. Every construction until §2's silence is ruled.
        for key in RELABEL_SORT_KEYS:
            r = relabel_D(coll["built"], coll["pairs"], a["arm_A"],
                          a["dim"], key, a["admitted"])
            if r:
                r["ratio_to_D2"] = r["D_relabelled"] / D2_OBSERVED[name]
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
    ha, hb, ma, mb = mass_balanced_split(list(sizes), sizes, 1)
    check("stable_seed does not use the salted builtin hash",
          "hash(" not in open(__file__).read().split("def stable_seed(")[1]
          .split("def mass_balanced_split(")[0].replace("hashlib.sha256", ""))
    check("stable_seed is constant for a given string",
          stable_seed("abc") == 0x0000 | stable_seed("abc")
          and stable_seed("abc") == int(hashlib.sha256(b"abc").hexdigest()[:8], 16) & 0xFFFF)
    check("stable_seed separates different members",
          stable_seed("abc") != stable_seed("abd"))
    check("split covers every cell exactly once",
          sorted(ha + hb) == sorted(sizes) and not (set(ha) & set(hb)))
    check("greedy split balances mass to within the largest cell",
          abs(ma - mb) <= max(sizes.values()))

    #: --- STAGE SEPARATION, structurally ---
    #: **THE CHECK IS ON IDENTIFIERS, NOT WORDS, AND THE FIRST VERSION WAS NOT.**
    #: Grepping stage 1 for "ratio" matched its own `_what` disclaimer -- the
    #: sentence saying it computes NO ratio. A stage-separation test that a
    #: DISCLAIMER can fail is testing prose; these strings cannot occur except
    #: as calls or subscripts.
    src = open(__file__).read()
    s1 = src[src.index("def stage1("):src.index("def require_stage1(")]
    for forbidden in ('["D_pair"]', "relabel_D(", "disattenuate(",
                      "ratio_to_D2", "D2_OBSERVED"):
        check(f"stage1 contains no {forbidden}", forbidden not in s1)
    check("stage2 calls the gate", "require_stage1(" in
          src[src.index("def stage2("):src.index("def selftest(")])
    check("the gate compares hashes and exits",
          "SystemExit" in src[src.index("def require_stage1("):
                              src.index("# ═", src.index("def require_stage1("))])

    #: --- §2's silence is DECLARED, not defaulted ---
    check("RELABEL_PRIMARY is unset until the pen rules",
          RELABEL_PRIMARY is None)
    check("every sort key is a construction pool_stat knows",
          all(pool_stat([1.0], [1.0], k) is not None for k in RELABEL_SORT_KEYS))

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

    print(f"\n{ok}/{ok + len(fail)} passed" + ("" if not fail else f"; FAILED: {fail}"))
    return not fail


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
