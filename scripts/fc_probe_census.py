#!/usr/bin/env python
"""fc_probe_census.py — THE COMBINED READING FOR BOTH SFT PROBES, WRITTEN
BEFORE THE SECOND ONE LANDS.

    scripts/fc_probe_census.py

WHY NOW. MiniCPM5 is still generating as this file is committed. Both rules it
applies were declared in advance by the lacan seat and are implemented here
rather than left in prose, for the reason that has already paid twice tonight:
a prose rule cannot fail a test, and two boundary defects in the seven-cell map
were caught by a known-answer selftest an hour before they would have mattered.

**THIS IS A CENSUS, NOT A SAMPLE.** `fc_roster_concentration.py --prompts beam`
finds exactly two pairs meeting both halves of the criterion once SFT rungs are
admitted. Running both EXHAUSTS the cell. Agreement is the cell behaving one
way; disagreement is the cell being heterogeneous, which is a finding rather
than noise to average. **Never pool them: two pairs is not a rate.**

RULE 1 — the combined verdict, lacan [4888].3.
    both cells the same   that verdict, as a census of the cell
    cells differ          REPORT BOTH, no adjudication. The disagreement is
                          the result: the cell is heterogeneous. Do not
                          average, and do not pick the one agreeing with the fit.
    either is positive    report separately; a positive in this cell needs its
                          own account regardless of what the other does

RULE 2 — the STAGE reading, lacan [4896].4. **The fit's 32 pairs are ALL
superego-stage and these are SFT rungs**, so applying the fitted relation here
extrapolates in a dimension nobody named: not concentration, which the evening
was spent on, but STAGE. "The effect exceeds what the fit predicts" therefore
has a competing explanation — SFT rungs may simply sit above a superego-fitted
relation — and findings U supplies the mechanism rather than leaving it ad hoc,
since SFT carries 74% of ladder JS and does the cutting.

    both above their own PRED_FIT   STAGE. Two of two SFT rungs exceeding a
                                    superego-fitted relation is about the RUNG,
                                    and "the intercept understates" is the
                                    wrong frame.
    not both                        not stage. Any excess is pair-specific and
                                    the intercept reading stands for that pair
                                    alone.

**Rule 2 touches only the smaller half of the verdict.** The anti-competitor
margin at OLMo-2 was 0.0884 and survives any reading of this; what is
reinterpreted is the 0.0017 half. If it is stage, that is a finding about the
ladder rather than about the intercept.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

FIT_INTERCEPT = -0.08672779      #: OLS on the 32 SUPEREGO roster pairs
FIT_SLOPE = -0.12223136
T975 = 1.971

#: beam-population drops — twp entropy restricted to the 210 beam prompts,
#: matching what the fit's own drops are computed on (fc_committed_entropy_test
#: :110 opens true_word_probs; :131 iterates the beam prompts to select sites).
PAIRS = (
    ("allenai/OLMo-2-0425-1B>allenai/OLMo-2-0425-1B-SFT", 0.12126032),
    ("openbmb/MiniCPM5-1B-Base>openbmb/MiniCPM5-1B-SFT", 0.17453224),
)


def decide(lo, hi, m, pc, pf):
    """The seven-cell map, lacan [4877].2, at this pair's own thresholds."""
    if m >= 0:
        return "FOURTH REVERSAL"
    ec, ef = not (lo <= pc <= hi), not (lo <= pf <= hi)
    if ec and not ef:
        return "COMPETITOR REFUTED, intercept account survives"
    if ef and not ec:
        return "INTERCEPT ACCOUNT REFUTED, competitor survives"
    if not ec and not ef:
        return "DOES NOT ADJUDICATE"
    if hi < pf:
        return "EFFECT STRONGER THAN EITHER ACCOUNT PREDICTS"
    if lo > pc:
        return "EFFECT WEAKER THAN EVEN THE COMPETITOR PREDICTS"
    return "BOTH ACCOUNTS WRONG, truth intermediate"


def stage_reading(rows):
    """Rule 2. 'Above its own PRED_FIT' means the ESTIMATE is more negative
    than the prediction — the effect exceeds what a superego-fitted relation
    gives at that pair's drop."""
    above = [r for r in rows if r["m"] < r["pf"]]
    if len(above) == len(rows) and len(rows) > 1:
        return ("STAGE", "both SFT rungs exceed a superego-fitted relation. This "
                "is about the RUNG, not the intercept: 'the intercept understates' "
                "is the wrong frame, and findings U supplies the mechanism (SFT "
                "carries 74% of ladder JS and does the cutting).")
    return ("NOT STAGE", "not both rungs exceed their own prediction, so any "
            "excess is pair-specific and the intercept reading stands for that "
            "pair alone.")


def selftest():
    """Known answers for BOTH rules, run before the reading."""
    pc, pf = -0.0148, -0.1015
    cases = [((-0.126, -0.103), -0.1145, "EFFECT STRONGER THAN EITHER ACCOUNT PREDICTS"),
             ((-0.117, -0.077), -0.0970, "COMPETITOR REFUTED, intercept account survives"),
             ((-0.030, +0.009), -0.0148, "INTERCEPT ACCOUNT REFUTED, competitor survives"),
             ((-0.121, -0.005), -0.0600, "DOES NOT ADJUDICATE"),
             ((-0.090, -0.050), -0.0700, "BOTH ACCOUNTS WRONG, truth intermediate"),
             ((-0.010, -0.002), -0.0060, "EFFECT WEAKER THAN EVEN THE COMPETITOR PREDICTS"),
             ((+0.10, +0.22), +0.1610, "FOURTH REVERSAL")]
    seen = set()
    for (lo, hi), m, want in cases:
        got = decide(lo, hi, m, pc, pf)
        assert got == want, "decide(%r,%r,%r) = %r, want %r" % (lo, hi, m, got, want)
        seen.add(got)
    assert len(seen) == 7, "only %d of 7 cells exercised" % len(seen)
    #: rule 2, both directions, including the one-pair guard
    assert stage_reading([{"m": -0.11, "pf": -0.10}, {"m": -0.12, "pf": -0.11}])[0] == "STAGE"
    assert stage_reading([{"m": -0.11, "pf": -0.10}, {"m": -0.09, "pf": -0.11}])[0] == "NOT STAGE"
    assert stage_reading([{"m": -0.11, "pf": -0.10}])[0] == "NOT STAGE", \
        "one pair must never read STAGE — that would be a census of one"
    print("  selftest: 7 cells + rule 2 both ways + the one-pair guard — all pass")


def main():
    import fc_analyse as F
    from malign_logits.cache import get_cache
    by = F.load(get_cache(), None)
    selftest()
    print()
    rows, missing = [], []
    for pid, drop in PAIRS:
        if pid not in by:
            missing.append(pid)
            continue
        per = {}
        for (role, arm, w, prompt), rec in by[pid].items():
            if arm != "undisturbed":
                continue
            sb, sa = rec.get("scored_by_base"), rec.get("scored_by_aligned")
            if not sb or not sa:
                continue
            first, second = (sb, sa) if role == "base" else (sa, sb)
            v = [x - y for r1, r2 in zip(first, second)
                 for i, (x, y) in enumerate(zip(r1, r2)) if i > 0]
            if v:
                per.setdefault(prompt, {})[role] = statistics.mean(v)
        a = [(d["base"] - d["aligned"]) / 2 for d in per.values() if len(d) == 2]
        if len(a) < 5:
            missing.append(pid)
            continue
        m, sd = statistics.mean(a), statistics.stdev(a)
        se = sd / len(a) ** 0.5
        pc, pf = FIT_SLOPE * drop, FIT_INTERCEPT + FIT_SLOPE * drop
        rows.append(dict(pid=pid, n=len(a), m=m, sd=sd, se=se, pc=pc, pf=pf,
                         lo=m - T975 * se, hi=m + T975 * se,
                         neg=sum(1 for x in a if x < 0)))

    print("SFT-RUNG CENSUS — the whole low-concentration high-displacement cell")
    for r in rows:
        print("  %s" % r["pid"].split(">")[0].split("/")[-1])
        print("     asymmetry %+.4f  CI [%+.4f, %+.4f]  n=%d  %d neg"
              % (r["m"], r["lo"], r["hi"], r["n"], r["neg"]))
        print("     its own PRED_CONC %+.5f  PRED_FIT %+.5f" % (r["pc"], r["pf"]))
        print("     -> %s" % decide(r["lo"], r["hi"], r["m"], r["pc"], r["pf"]))
    if missing:
        print("\n  NOT YET LANDED: %s" % ", ".join(p.split(">")[0].split("/")[-1]
                                                   for p in missing))
        print("  The census is INCOMPLETE and neither rule may be applied to a")
        print("  partial cell — half a census reads exactly like a whole one.")
        return
    v = [decide(r["lo"], r["hi"], r["m"], r["pc"], r["pf"]) for r in rows]
    print("\n  RULE 1 — combined verdict [4888].3")
    if len(set(v)) == 1:
        print("     BOTH CELLS AGREE: %s" % v[0])
        print("     Reported as a CENSUS of the cell, n=2 pairs, never as a rate.")
    else:
        print("     CELLS DIFFER — no adjudication. The disagreement IS the result:")
        print("     the cell is heterogeneous. Do not average; do not pick the one")
        print("     that agrees with the fit.")
        for r, vv in zip(rows, v):
            print("       %-22s %s" % (r["pid"].split(">")[0].split("/")[-1], vv))
    tag, why = stage_reading(rows)
    print("\n  RULE 2 — the STAGE reading [4896].4")
    print("     %s: %s" % (tag, why))
    print("\n  TRAVELS WITH BOTH: an SFT rung is not a superego pair, the fit was")
    print("  estimated on 32 superego pairs, and these two ARE the cell rather")
    print("  than a sample of it. Never pooled with the roster.")


if __name__ == "__main__":
    main()
