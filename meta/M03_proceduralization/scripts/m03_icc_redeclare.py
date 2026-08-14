"""The M03 rung ICC, re-declared. Rule at plans/plan_icc_redeclaration.md.

    uv run python meta/M03_proceduralization/scripts/m03_icc_redeclare.py
    -> results/icc_redeclared.json

The plan was committed ALONE at b2b9a0cb, before this file existed. That
ordering is why these numbers are worth more than the ones they replace:
`D_ladder_selection.md` books 0.855 and 0.846, `d_ladder_fields.py:157`
PRINTS 0.85, and nothing computes any of the three -- the value in the
producer is a string literal inside a print statement ([5998], [6000]).

**THE BOOKED VALUES ARE NOT AVAILABLE TO THE REDUCTION.** They are printed
after every number is computed. No branch reads them.

THE RULE, in the plan's words:

    1. POPULATION FIRST, AS AN ASSERT: 12 scenarios for f21_inst and 18 for
       m03_slice, or refuse. A reduction that cannot reproduce the population
       cannot be assessed on its statistic -- which is what ended malign's
       attempt at 11 scenarios.
    2. d is the PRODUCER'S OWN subtraction, share(inst) - share(indiv),
       unaveraged over rungs because the rung axis is what is being measured.
    3. the item is (source, field) -- the producer's own item, >= 8 scenarios.
    4. median over items, published WITH quartiles and the share above 0.5.
    5. ICC(1) one-way random effects, scenarios random, rungs as repeats.

ICC(1) IS DECLARED WITH ITS BIAS. Rungs are ORDERED along training and so are
not exchangeable; a systematic trend across them is charged to within-group
variance, so ICC(1) UNDERSTATES the correlation. That is conservative in the
direction that matters: it biases toward rungs looking INDEPENDENT, which is
the reading that would revive the rung unit. A high ICC under a statistic
biased against high is a safe collapse.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CAMP = os.path.dirname(HERE)
SRC = os.path.join(CAMP, "results", "d_ladder_fields.csv")
OUT = os.path.join(CAMP, "results", "icc_redeclared.json")

#: d_ladder_fields.py:149, quoted not re-derived
AL = ("sft_step", "sft_endpoint", "dpo_endpoint", "rlvr_step")
#: d_ladder_fields.py:167
MIN_SCENARIOS = 8
#: the plan's population assert
EXPECTED_SCENARIOS = {"f21_inst": 12, "m03_slice": 18}
#: printed at the end only; never read by the reduction
SUPERSEDED = {"f21_inst": 0.855, "m03_slice": 0.846,
              "producer_print_statement": 0.85}


def icc1(groups):
    """ICC(1), one-way random effects, from a list of per-group value lists.

    ICC(1) = (MSB - MSW) / (MSB + (k0 - 1) * MSW), with k0 the standard
    correction for unequal group sizes. Returns None if it cannot be formed
    (one group, or no within-group variation to partition).
    """
    import numpy as np
    g = [np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
         for vals in groups]
    g = [x for x in g if len(x) >= 2]
    k = len(g)
    if k < 2:
        return None
    n = np.array([len(x) for x in g], dtype=float)
    N = n.sum()
    means = np.array([x.mean() for x in g])
    grand = np.concatenate(g).mean()
    ssb = float((n * (means - grand) ** 2).sum())
    ssw = float(sum(((x - x.mean()) ** 2).sum() for x in g))
    dfb, dfw = k - 1, N - k
    if dfw <= 0 or dfb <= 0:
        return None
    msb, msw = ssb / dfb, ssw / dfw
    #: k0: the unequal-group-size correction. With equal n it reduces to n.
    k0 = (N - (n ** 2).sum() / N) / dfb
    den = msb + (k0 - 1) * msw
    if den == 0:
        return None
    return float((msb - msw) / den)


def main():
    import numpy as np
    import pandas as pd

    #: keep_default_na=False: `field` and `source` are categorical labels and a
    #: label reading as NaN is tonight's own lesson ([5966]/[5969]).
    d = pd.read_csv(SRC, keep_default_na=False, low_memory=False)
    d["share"] = pd.to_numeric(d["share"], errors="coerce")
    print("substrate %s rows" % format(len(d), ","))

    A = d[d.role_ck.isin(AL)].copy()
    print("alignment rungs (%s): %s rows, %d checkpoints"
          % (", ".join(AL), format(len(A), ","), A.checkpoint.nunique()))

    #: RULE 2: the producer's own subtraction, at the RUNG grain.
    P = A.pivot_table(index=["stratum", "scenario", "source", "field",
                             "checkpoint"], columns="arm", values="share")
    P = P.dropna().reset_index()
    P["d"] = P["inst"] - P["indiv"]
    print("paired cells (both arms present): %s" % format(len(P), ","))

    #: RULE 1: POPULATION IS AN ASSERT.
    got = P.groupby("stratum").scenario.nunique().to_dict()
    for strat, want in EXPECTED_SCENARIOS.items():
        if got.get(strat) != want:
            raise SystemExit(
                "POPULATION MISMATCH: %s has %s scenarios, the finding says "
                "%d. A reduction that does not reproduce the population "
                "cannot be assessed on its statistic; fix the population "
                "before reading any ICC." % (strat, got.get(strat), want))
    print("population asserted: %s" % ", ".join(
        "%s %d scenarios" % (s, got[s]) for s in sorted(got)))

    out = {"_about":
           "The M03 rung ICC, RE-DECLARED under the rule at "
           "plans/plan_icc_redeclaration.md, committed at b2b9a0cb BEFORE this "
           "producer existed. SUPERSEDES 0.855 / 0.846 in "
           "D_ladder_selection.md and 0.85 printed by d_ladder_fields.py:157, "
           "none of which was computed by anything. The ICC licenses "
           "collapsing 52 rungs to 12 and 18 scenarios; a HIGH value means "
           "the collapse costs nothing, a LOW one means it discarded "
           "independent information. ICC(1) UNDERSTATES correlation here "
           "because rungs are ordered and not exchangeable, so it is "
           "conservative toward the reading that would revive the rung unit. "
           "The median travels with its quartiles because it aggregates "
           "hundreds of items and can hide them disagreeing.",
           "plan": "plans/plan_icc_redeclaration.md",
           "rule": {"1_population": "asserted at run: 12 and 18 scenarios",
                    "2_d": "share(inst) - share(indiv), producer's own, per rung",
                    "3_item": "(source, field), >= %d scenarios" % MIN_SCENARIOS,
                    "4_summary": "median over items, with quartiles",
                    "5_statistic": "ICC(1) one-way random effects"},
           "alignment_rungs": list(AL), "strata": {}, "superseded": SUPERSEDED}

    for strat in sorted(EXPECTED_SCENARIOS):
        S = P[P.stratum == strat]
        vals, dropped = [], 0
        for (src, fld), item in S.groupby(["source", "field"]):
            #: RULE 3: the producer's own item and its own >=8 gate.
            if item.scenario.nunique() < MIN_SCENARIOS:
                dropped += 1
                continue
            v = icc1([list(g.d) for _, g in item.groupby("scenario")])
            if v is not None:
                vals.append({"source": src, "field": fld, "icc": v,
                             "n_scenarios": int(item.scenario.nunique()),
                             "n_rungs_median": int(item.groupby("scenario")
                                                   .checkpoint.nunique().median())})
        if not vals:
            raise SystemExit("no items survived for %s" % strat)
        a = np.array([x["icc"] for x in vals])
        rec = {"n_items": len(vals), "n_items_dropped_under_%d" % MIN_SCENARIOS: dropped,
               "icc_median": float(np.median(a)),
               "icc_q1": float(np.percentile(a, 25)),
               "icc_q3": float(np.percentile(a, 75)),
               "icc_min": float(a.min()), "icc_max": float(a.max()),
               "share_items_above_0.5": float((a > 0.5).mean()),
               "share_items_above_0.8": float((a > 0.8).mean()),
               "share_items_below_0.2": float((a < 0.2).mean()),
               "items": sorted(vals, key=lambda x: -x["icc"])}
        #: THE DECISION-RELEVANT QUANTITY IS NOT THE ICC, IT IS THE DESIGN
        #: EFFECT. The plan asks whether collapsing 52 rungs to one value per
        #: scenario is justified. DEFF = 1 + (k-1) * ICC with k the rungs per
        #: scenario; effective observations per scenario = k / DEFF. A
        #: moderate ICC at k=52 still leaves a scenario worth ~1.5 rungs, not
        #: 52 -- so "the ICC is lower than booked" and "the collapse was
        #: wrong" are different claims and only the first is true.
        k = float(np.median([x["n_rungs_median"] for x in vals]))
        deff = 1.0 + (k - 1.0) * rec["icc_median"]
        rec["rungs_per_scenario_median"] = k
        rec["design_effect"] = deff
        rec["effective_obs_per_scenario"] = k / deff
        rec["effective_n_total"] = EXPECTED_SCENARIOS[strat] * k / deff
        rec["scenario_unit_n"] = EXPECTED_SCENARIOS[strat]
        rec["rung_unit_n_claimed"] = int(EXPECTED_SCENARIOS[strat] * k)
        out["strata"][strat] = rec
        print("\n%-10s %d items (%d dropped under %d scenarios)"
              % (strat, len(vals), dropped, MIN_SCENARIOS))
        print("  ICC(1) median %.4f   IQR %.4f - %.4f   range %.4f - %.4f"
              % (rec["icc_median"], rec["icc_q1"], rec["icc_q3"],
                 rec["icc_min"], rec["icc_max"]))
        print("  items above 0.8: %4.1f%%   above 0.5: %4.1f%%   below 0.2: %4.1f%%"
              % (100 * rec["share_items_above_0.8"],
                 100 * rec["share_items_above_0.5"],
                 100 * rec["share_items_below_0.2"]))
        print("  %.0f rungs/scenario -> design effect %.1f -> %.2f effective "
              "observations per scenario" % (k, deff, k / deff))
        print("  so the rung unit's claimed n=%d is really n=%.1f; the "
              "scenario unit uses n=%d"
              % (rec["rung_unit_n_claimed"], rec["effective_n_total"],
                 rec["scenario_unit_n"]))

    #: EVERYTHING ABOVE IS COMPUTED. Only now are the booked values read.
    print("\nAGAINST THE SUPERSEDED VALUES (comparison only; not used above)")
    for strat in sorted(EXPECTED_SCENARIOS):
        print("  %-10s booked %.3f  ->  re-declared %.4f"
              % (strat, SUPERSEDED[strat], out["strata"][strat]["icc_median"]))
    print("  %-10s printed %.2f by d_ladder_fields.py:157, computed by nothing"
          % ("producer", SUPERSEDED["producer_print_statement"]))

    json.dump(out, open(OUT, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
