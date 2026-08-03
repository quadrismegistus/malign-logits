#!/usr/bin/env python3
"""Pool-extremity diagnostic for D2. A DIAGNOSTIC, NOT A TEST. [3397].

RH's QUESTION: does D2's confirmation merely reflect transgressive prompts
having MORE EXTREME VALENCE AVAILABLE in their word pools?

**WHAT IS AND IS NOT BUDGETED, per the pen's accounting:**

    THE LEVEL is budgeted BY CONSTRUCTION. `A = wmean(fallers) - wmean(risers)`
    is a WITHIN-CELL role contrast, so a uniformly more extreme pool shifts both
    terms alike and subtracts out.

    THE TAIL COMPOSITION IS NOT. C's H2 confirmation was pool-controlled by the
    MEMBERSHIP null; D2's site-specificity contrast runs on the pair SIGN-FLIP
    null, which does NOT hold pool composition fixed across members. **If marked
    pools are extreme-tail-richer, content-blind movement could mechanically
    widen the faller-riser gap.**

**THE READING RULE, DECLARED BEFORE ANYTHING IS COMPUTED:** comparative
description only. **NO significance test. NO verdict language.** If the pools
differ materially, the finding is *"the §D3 membership-null-benchmarked
registration is REQUIRED before the D2 sentence travels."* If they match, the
question dies on a measurement — the way G's base-peakedness confound did.

**EXPOSURE: this touches BASE POOLS ONLY.** No D, no A, no verdict material.
The words and their norm values are the inputs the statistic reads; this
describes their distribution and computes no contrast the registration owns.
"""

import collections
import hashlib
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pairs_d as D                      #: FROZEN 84011269d00eea6b

#: DECLARED CUTS. |z| >= 1 and >= 2 on the SOURCE-DATABASE z-scale, which is
#: the scale the norms are anchored to (C §C0: "z anchored to the source
#: database", never to the observed sample) -- so a cut means the same thing
#: here as it does in the statistic.
CUTS = (1.0, 2.0)
DIM = "valence"


def pool_profile(built, dim=DIM):
    """Per MEMBER, the extremity profile of its qualifying words.

    **THE POOL IS THE SAME POOL THE STATISTIC READS**: the words surviving the
    function-word filter, the norm lookup and the >=3-per-role bar, in the
    cells that qualified. Profiling a different pool would answer a question
    nobody asked.

    BOTH WEIGHTINGS ARE REPORTED AND THE DIFFERENCE IS THE POINT:

      UNWEIGHTED   what the pool CONTAINS -- the availability question RH asked
      |delta|-WEIGHTED   what the statistic actually SEES, since `A` is
                   |delta|-weighted. A pool can be tail-rich in words the
                   movement never touches.
    """
    out = {}
    for text, per_edge in built["cells"].items():
        vals, wts = [], []
        for c in per_edge.values():
            for z, w in zip(c["zs"], c["ws"]):
                vals.append(abs(z[dim])); wts.append(w)
        if not vals:
            continue
        tw = sum(wts)
        rec = {"n_words": len(vals),
               "mean_abs_z": st.mean(vals),
               "mean_abs_z_wtd": (sum(v * w for v, w in zip(vals, wts)) / tw
                                  if tw > 0 else None)}
        for cut in CUTS:
            hit = [1.0 if v >= cut else 0.0 for v in vals]
            rec[f"tail_ge_{cut:g}"] = sum(hit) / len(hit)
            rec[f"tail_ge_{cut:g}_wtd"] = (
                sum(h * w for h, w in zip(hit, wts)) / tw if tw > 0 else None)
        out[text] = rec
    return out


def within_pair(built, prof):
    """MARKED vs UNMARKED, paired, with BOTH SIDES' ABSOLUTES. [3052] precedent.

    A difference is not a direction until both terms are visible -- the same
    §D1 rule that governs the read governs its diagnostic.
    """
    keys = (["mean_abs_z", "mean_abs_z_wtd"]
            + [f"tail_ge_{c:g}" for c in CUTS]
            + [f"tail_ge_{c:g}_wtd" for c in CUTS])
    rows, n_pairs = [], 0
    for pid, members in built["pairs"].items():
        m, u = members.get("MARKED"), members.get("UNMARKED")
        if m not in prof or u not in prof:
            continue
        n_pairs += 1
        rows.append({"pair_id": pid,
                     **{f"M_{k}": prof[m][k] for k in keys},
                     **{f"U_{k}": prof[u][k] for k in keys}})
    summary = {"n_pairs_with_both_members_profiled": n_pairs}
    for k in keys:
        M = [r[f"M_{k}"] for r in rows if r[f"M_{k}"] is not None]
        U = [r[f"U_{k}"] for r in rows if r[f"U_{k}"] is not None]
        d = [r[f"M_{k}"] - r[f"U_{k}"] for r in rows
             if r[f"M_{k}"] is not None and r[f"U_{k}"] is not None]
        summary[k] = {
            "MARKED_mean": st.mean(M) if M else None,
            "UNMARKED_mean": st.mean(U) if U else None,
            "within_pair_diff_mean": st.mean(d) if d else None,
            "within_pair_diff_median": st.median(d) if d else None,
            #: a spread, so a reader can see whether a small mean hides a wide
            #: distribution -- NOT a test statistic and not used as one
            "within_pair_diff_sd": st.pstdev(d) if len(d) > 1 else None,
            "n_pairs_positive": sum(1 for x in d if x > 0),
            "n_pairs_negative": sum(1 for x in d if x < 0),
        }
    return rows, summary


def selftest():
    ok = [0, 0]

    def case(name, cond):
        good = False
        try:
            good = bool(cond())
        except Exception as e:
            print(f"  [ERR] {name}: {type(e).__name__}: {e}")
        ok[0] += 1; ok[1] += 1 if good else 0
        print(f"  [{'ok' if good else 'FAIL'}] {name}")

    stub = {"cells": {"m": {("f", "p"): {"zs": [{"valence": 0.5},
                                                {"valence": 1.5},
                                                {"valence": 2.5}],
                                         "ws": [1.0, 1.0, 1.0],
                                         "rs": [], "departed": 0.0}},
                      "u": {("f", "p"): {"zs": [{"valence": 0.1},
                                                {"valence": 0.2},
                                                {"valence": 0.3}],
                                         "ws": [1.0, 1.0, 1.0],
                                         "rs": [], "departed": 0.0}}},
            "pairs": {"p1": {"MARKED": "m", "UNMARKED": "u"}}}
    prof = pool_profile(stub)

    case("the tail counter FIRES: 2 of 3 words at |z| >= 1",
         lambda: abs(prof["m"]["tail_ge_1"] - 2 / 3) < 1e-9)
    case("and at |z| >= 2 only ONE of the three qualifies",
         lambda: abs(prof["m"]["tail_ge_2"] - 1 / 3) < 1e-9)
    case("the tail counter is ZERO where no word reaches the cut",
         lambda: prof["u"]["tail_ge_1"] == 0.0 and prof["u"]["tail_ge_2"] == 0.0)
    case("mean |z| uses the ABSOLUTE value, so sign cannot cancel",
         lambda: abs(prof["m"]["mean_abs_z"] - 1.5) < 1e-9)
    neg = {"cells": {"x": {("f", "p"): {"zs": [{"valence": -2.5}],
                                        "ws": [1.0], "rs": [], "departed": 0}}},
           "pairs": {}}
    case("a NEGATIVE z counts as extreme, not as low",
         lambda: pool_profile(neg)["x"]["tail_ge_2"] == 1.0)
    case("the weighted mean differs from the unweighted when weights differ",
         lambda: (lambda p: abs(p["m"]["mean_abs_z_wtd"] - 2.25) < 1e-9)(
             pool_profile({**stub, "cells": {**stub["cells"],
                          "m": {("f", "p"): {**stub["cells"]["m"][("f", "p")],
                                             "ws": [0.0, 1.0, 3.0]}}}})))
    rows, summ = within_pair(stub, prof)
    case("BOTH SIDES' ABSOLUTES are reported, not only the difference",
         lambda: summ["mean_abs_z"]["MARKED_mean"] is not None
                 and summ["mean_abs_z"]["UNMARKED_mean"] is not None)
    case("the within-pair difference is MARKED minus UNMARKED",
         lambda: abs(summ["mean_abs_z"]["within_pair_diff_mean"] - 1.3) < 1e-9)
    case("and the direction counters agree with it",
         lambda: summ["mean_abs_z"]["n_pairs_positive"] == 1
                 and summ["mean_abs_z"]["n_pairs_negative"] == 0)
    case("NO significance machinery exists in this module",
         lambda: not ({"sign_flip_p", "raw_mde", "reading_rule", "mde_reading"}
                      & {k for k, v in globals().items()
                         if callable(v) and getattr(v, "__module__", None)
                         == __name__}))
    case("and no verdict vocabulary appears in the summary keys",
         lambda: not any(w in json.dumps(summ).lower()
                         for w in ("reject", "p_value", "confirmed", "verdict")))
    print(f"selftest {ok[1]}/{ok[0]}")
    return 0 if ok[1] == ok[0] else 1


if __name__ == "__main__":
    sys.exit(selftest())
