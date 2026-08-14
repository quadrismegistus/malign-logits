#!/usr/bin/env python
"""fc_probe_read.py — THE READING FOR THE SFT PROBE, WRITTEN BEFORE THE DATA.

    scripts/fc_probe_read.py

WHY IT IS WRITTEN NOW. The probe is still generating as this file is committed.
The outcome map it applies was declared by the lacan seat at docket [4872],
from the seat with no stake in the asymmetry, and accepted at [4873]. Writing
the classifier before the number exists is what makes the map binding: a
declared reading with no number in it is a declaration that a number can still
be argued with, and -0.06 will look like "the effect appeared" to whoever wants
it to at four in the morning.

THE FRAMING THAT MAKES A MAP NECESSARY. All three accounts predict a NEGATIVE
number at this pair, so "did the asymmetry appear" cannot separate them. The
competitor does not predict zero — it predicts SMALL. The discriminating
quantity is the magnitude against -0.0103.

    PURE CONCENTRATION   proportional to drop, through the origin
                         -0.1222 * 0.0846  =  -0.0103
    THE COMMITTED FIT    with its intercept
                         -0.0867 - 0.1222 * 0.0846  =  -0.0967
    THE ROSTER MEAN      effect independent of drop
                                            =  -0.1381

PRE-DECLARED MDE (computed 7 Aug, before the run finished, from the 32 roster
pairs' within-pair per-site sd at this probe's 209 sites): median 0.0194,
upper-quartile 0.0290, WORST CASE 0.0453 — against a separation of 0.0864, so
1.9x margin even at the noisiest per-site variance any roster pair exhibits.
The probe can resolve the gap. Had that come out under 1x, the honest move was
to say so before seeing the answer.

WHAT TRAVELS WITH THE RESULT, ALWAYS, whichever way it falls:
  n=1 PAIR, and the pair was CHOSEN because it populates the cell. That
  selection is the design, not a confound — the point was to find a case where
  concentration and displacement predict different things. But **one pair is an
  existence proof or a failure to find one; it is not a rate.** No roster
  conclusion either way. And it is an SFT rung where all 32 roster pairs are
  superego, so it is reported BESIDE them and never pooled.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

PAIR = "allenai/OLMo-2-0425-1B>allenai/OLMo-2-0425-1B-SFT"
DROP = 0.0846
PRED_CONC, PRED_FIT, PRED_MEAN = -0.0103, -0.0967, -0.1381

#: **THE DECISION RULE, amended by lacan at [4875].3 after my [4874] showed the
#: original bands could not confirm the account they tested.** The bands were
#: anchored on the roster mean and a round number below it, and the committed
#: fit predicts -0.0967 at this pair -- 0.0033 inside the non-adjudicating
#: band. A map that can refuse to see the thing it was built to test.
#:
#: My proposed fix (an indifference zone of +/-MDE around the midpoint) was
#: REJECTED, correctly: the MDE is a free parameter, and at the roster's
#: worst-case per-site sd (0.2341) the zone becomes -0.0988..-0.0082 and
#: swallows BOTH predictions, so the rule could be vacuous depending on a value
#: not known until the run ends. **A decision rule whose free parameter can make
#: it vacuous must not be declared before knowing which value it takes.**
#:
#: The amendment removes the parameter: report the estimate with its 95% CI from
#: THIS pair's own per-site sd and test it against both point predictions. The
#: SE is a nuisance parameter, so letting it come from the data is not post-hoc
#: -- it is the only correct way to get it. THE RULE IS DECLARED NOW; ONLY THE
#: WIDTH IS MEASURED.
#: **THE THRESHOLDS ARE DERIVED HERE, NOT TRANSCRIBED.** They were wrong for
#: three posts because a human did the multiplication and the product was then
#: copied between posts: lacan computed -0.0970 correctly at [4872], I quoted
#: -0.0967 at [4874], and lacan adopted MY number into the amended map at
#: [4875] and the final map at [4877] without re-deriving it, because it looked
#: like the one already computed. Three digits of agreement stopped the check at
#: the fourth. Then the correction at [4879] proposed -0.0970 -- which is right
#: for the 4-dp DISPLAYED constants and still 0.00004 off the truth, because it
#: repeats the same move on rounded inputs.
#:
#: So the file pins the three UNROUNDED inputs and lets the machine multiply.
#: The only transcription risk left is on quantities that are what they are,
#: rather than on a product nobody can check by eye. Re-derive all three with
#: `scripts/fc_reversal_entropy.py` (fit) and `fc_roster_concentration.py` (drop).
FIT_INTERCEPT = -0.08672779     #: OLS on all 32 roster pairs, asym ~ drop
FIT_SLOPE = -0.12223136
#: **THE POPULATION IS PART OF THE NUMBER, and I pinned the wrong one.** The
#: fit's 32 drops are computed on each pair's BEAM prompts -- `fc_reversal_
#: entropy.py` iterates `per`, which is built from the beam_fc undisturbed arm
#: -- so the fit maps BEAM-population drops to asymmetries. I pinned this pair's
#: FULL twp drop (0.08464794, n=2,583), which asks the fit a question in the
#: wrong units. Caught by lacan at [4881] while chasing the rounding.
#:
#:     beam population, n=210     +0.12126032   <- what the fit expects
#:     full twp population, n=2583 +0.08464794
#:     ratio 1.43x; moves PRED_FIT by 0.00448, ELEVEN TIMES the rounding fix
#:     we had just spent an hour on.
#: **PROVENANCE, because "beam population" named two different things and only
#: one of them was right.** The entropy SOURCE is `true_word_probs`; the beams
#: only select WHICH PROMPTS enter the average. Verified in the committed test
#: itself: `st = cm._stash("true_word_probs")` at line 110, `entropy()` reads
#: that stash at line 55, and line 131 iterates `per` -- the prompts holding
#: both roles' undisturbed beams -- purely as the site-selection step.
#:
#: My [4884].4 proposed recomputing this from the probe's own 420 undisturbed
#: units instead. That would have been an OVER-CORRECTION reintroducing the
#: same mismatch with the sign flipped, and lacan stopped it at [4886]. **The
#: population was never the beams; it was twp over the beams' prompts.** One
#: phrase, two referents -- the third time tonight a label dropped the part that
#: mattered (-0.0967, the 0.10 threshold, and now this).
PROBE_DROP = 0.12126032   #: twp entropy, restricted to the 210 beam prompts

PRED_CONC = FIT_SLOPE * PROBE_DROP                    #: competitor: through origin
PRED_FIT = FIT_INTERCEPT + FIT_SLOPE * PROBE_DROP     #: intercept account
T975 = 1.971                                          #: t(.975, df~208)


def decide(lo, hi, m):
    """**THE FINAL MAP — lacan [4877].2, seven cells, ruled after my [4874] and
    [4876] found two boundary defects in its predecessors. Not amended again
    before the run.**

    Each cell is named by WHAT IT LICENSES, not by which tests it passes. That
    is the lesson both defects taught: a cell defined by which statistical
    tests fail can contain claims that contradict each other, because the tests
    do not know what they mean. The prose bands anchored on a round number put
    the intercept account's own prediction in the non-adjudicating cell; the
    four CI-exclusion predicates put "competitor refuted emphatically" and
    "effect essentially absent" in the SAME cell. Enumerating the claims first
    and then asking which test distinguishes them gives seven at the outset.
    """
    if m >= 0:
        return ("FOURTH REVERSAL",
                "positive, in the pair chosen to test the competitor. Report it, "
                "do not celebrate it, and expect it to need its own account.")
    ex_conc = not (lo <= PRED_CONC <= hi)
    ex_fit = not (lo <= PRED_FIT <= hi)
    if ex_conc and not ex_fit:
        return ("COMPETITOR REFUTED, intercept account survives",
                "the effect exists at near-zero concentration; the intercept is "
                "real at a pair the fit MEASURED rather than extrapolated to.")
    if ex_fit and not ex_conc:
        return ("INTERCEPT ACCOUNT REFUTED, competitor survives",
                "consistent with pure concentration; the committed intercept "
                "looks like an artefact of the linear form, which [4812] already "
                "showed with sqrt spanning zero.")
    if not ex_conc and not ex_fit:
        return ("DOES NOT ADJUDICATE",
                "the CI covers both predictions; report the value and the width, "
                "and say the width is why.")
    #: excludes both -- three distinct claims, split at [4877].2
    if hi < PRED_FIT:
        return ("EFFECT STRONGER THAN EITHER ACCOUNT PREDICTS",
                "the competitor is refuted emphatically and the intercept "
                "account understates. The strongest outcome available here.")
    if lo > PRED_CONC:
        return ("EFFECT WEAKER THAN EVEN THE COMPETITOR PREDICTS",
                "worse for us than the deflationary account's own prediction, "
                "and a distinct outcome from the competitor being supported.")
    return ("BOTH ACCOUNTS WRONG, truth intermediate",
            "the CI sits strictly between the two predictions; neither account "
            "describes this pair.")


def selftest():
    """KNOWN ANSWERS OVER ALL SEVEN CELLS, run before the reading.

    This rule is applied ONCE to a number that matters. Two predecessors of it
    were wrong at a boundary, and BOTH were caught here rather than live --
    each time because an expectation of mine failed against the code, which is
    the only ordering that leaves the declaration intact. A prose map cannot
    fail a test; that is why both defects survived in prose."""
    cases = [
        ((-0.145, -0.105), -0.125, "EFFECT STRONGER THAN EITHER ACCOUNT PREDICTS"),
        #: the committed fit's OWN prediction -- the [4874] defect, which the
        #: superseded prose bands called DOES NOT ADJUDICATE
        ((-0.117, -0.077), PRED_FIT, "COMPETITOR REFUTED, intercept account survives"),
        ((-0.120, -0.080), -0.100, "COMPETITOR REFUTED, intercept account survives"),
        ((-0.030, +0.009), -0.0103, "INTERCEPT ACCOUNT REFUTED, competitor survives"),
        #: **CI CHOSEN TO STRADDLE THE CURRENT THRESHOLDS, not fixed literals
        #: that happened to straddle the old ones.** The previous version was
        #: (-0.099, -0.001), which covered both predictions while PRED_FIT was
        #: -0.0967 and stopped covering it the moment [4881]'s population fix
        #: moved PRED_FIT to -0.10155. The selftest FAILED, correctly, and
        #: caught a stale case rather than a wrong rule -- but I had already
        #: posted "selftest green" without reading its output ([4883] corrects
        #: that). Derived from the constants now, so it tracks them.
        ((PRED_FIT - 0.02, PRED_CONC + 0.01), -0.050, "DOES NOT ADJUDICATE"),
        ((-0.060, -0.040), -0.050, "BOTH ACCOUNTS WRONG, truth intermediate"),
        #: the [4876] defect: this and the first case shared one cell
        ((-0.008, -0.002), -0.005, "EFFECT WEAKER THAN EVEN THE COMPETITOR PREDICTS"),
        ((+0.100, +0.220), +0.161, "FOURTH REVERSAL"),
    ]
    seen = set()
    for (lo, hi), m, want in cases:
        got = decide(lo, hi, m)[0]
        assert got == want, "decide(%r,%r,%r) = %r, declared %r" % (lo, hi, m, got, want)
        seen.add(got)
    assert len(seen) == 7, "only %d of 7 declared cells exercised" % len(seen)
    print("  selftest: %d known answers, ALL SEVEN cells exercised" % len(cases))
    print("  pinned: the [4874] defect (fit's own -0.0967 read as non-adjudicating)")
    print("  and the [4876] defect (stronger-than-both sharing a cell with absent)")


def main():
    import fc_analyse as F
    from malign_logits.cache import get_cache
    by = F.load(get_cache(), None)
    if PAIR not in by:
        sys.exit("probe pair not in the stash yet — run has not landed")
    per = {}
    for (role, arm, w, prompt), rec in by[PAIR].items():
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
        sys.exit("only %d complete sites — run has not landed" % len(a))

    m = statistics.mean(a)
    sd = statistics.stdev(a)
    p, n = F.permutation_p(a)
    se = sd / (len(a) ** 0.5)
    lo, hi = m - T975 * se, m + T975 * se
    label, reading = decide(lo, hi, m)

    selftest()
    print()
    print("SFT PROBE — the reading, applied by a classifier written before the data")
    print("  %s" % PAIR)
    print("  entropy drop %+.6f  |  %d sites  |  %d/%d negative"
          % (PROBE_DROP, len(a), sum(1 for x in a if x < 0), len(a)))
    print()
    print("  OBSERVED ASYMMETRY   %+.4f   95%% CI [%+.4f, %+.4f]" % (m, lo, hi))
    print("                       sd %.4f, SE %.5f, perm p=%.4f" % (sd, se, p))
    print()
    print("  the two point predictions, and whether the CI excludes them:")
    for lab, v in (("pure concentration", PRED_CONC), ("committed fit", PRED_FIT)):
        print("     %-20s %+.4f   %s"
              % (lab, v, "EXCLUDED" if not (lo <= v <= hi) else "covered"))
    print("     %-20s %+.4f   (descriptive, not a decision cell)"
          % ("roster mean", PRED_MEAN))
    print()
    print("  *** %s ***" % label)
    print("  %s" % reading)
    #: `lo <= PRED_MEAN` alone. An earlier line read `lo <= PRED_MEAN or
    #: hi <= PRED_MEAN`, and since lo <= hi the second disjunct can never be
    #: true without the first -- dead code in a reporting path, and the same
    #: shape lacan named at [4864]: offering a disjunct is a way of not having
    #: checked which side holds.
    if lo <= PRED_MEAN:
        print("  ADDENDUM [4877].2: the interval reaches the roster mean %+.4f or"
              % PRED_MEAN)
        print("  beyond — the effect is at FULL ROSTER STRENGTH at this pair.")
    print()
    print("  TRAVELS WITH THIS NUMBER, ALWAYS:")
    print("   n=1 pair, CHOSEN because it populates the low-concentration cell.")
    print("   An existence proof or a failure to find one — NOT a rate. No")
    print("   roster conclusion either way. SFT rung; all 32 roster pairs are")
    print("   superego, so this is reported BESIDE them and never pooled.")


if __name__ == "__main__":
    main()
