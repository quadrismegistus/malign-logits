---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/opening_matched.json
date: 2026-08-13
role: finding
topics: [self-surprisal, chain, syntagmatic, forced-arms]
description: "Opening-matched comparison (RH's design): COMPENSATION, not damage. Holding the opening token's surprisal fixed -- and, in the strictest version, holding the prompt fixed and requiring a word-like opening -- a forced passage is LESS surprising to its own model than one of its own passages that opened equally improbably. Every arm, both roles, three estimators, all negative. THE ARM ORDERING IS NOT ESTABLISHED and the earlier 'imposition, not demotion' headline is WITHDRAWN ([5805]): all three paired arm contrasts are ns, the ordering inverts between median and mean, and the interval does not bound out an arm effect the size of one the ladder already detects. Not alignment-specific either; no DiD survives correction."
---
# Opening-matched: the chain compensates for being handed a word

Plan: `plans/plan_opening_matched.md`, committed before this producer, with
BOTH signs declared. Producer `scripts/m06_opening_matched.py`; results
`results/opening_matched.json` + per-pair parquet. No new compute. Single
pass; [5503] applies.

RH's design: instead of fencing off the undisturbed comparison as
confounded, **match on the opening's surprisal on the fly**, which is what
forcing a single matched word cannot do.

## The two theses, and which one the data picks

    RESIDUAL = forced passage's mean surprisal after the opening
               MINUS what an opening-matched undisturbed passage shows

    T1 DAMAGE        forcing breaks the chain          residual POSITIVE
    T2 COMPENSATION  the syntagmatic absorbs it        residual NEGATIVE

**T2. Every arm, both roles, three estimators, no exceptions.**

    PRIMARY (binned at 0.5 nat on the opening logprob; pair grain)
      faller         aligned -0.0342  9/31  p 6.8e-04 | base -0.0260  7/32  p 7.0e-05
      matched        aligned -0.0551  5/35  p 1.4e-06 | base -0.0299  8/30  p 4.7e-04
      riser_matched  aligned -0.0477  7/32  p 7.0e-05 | base -0.0394  7/31  p 1.2e-04

    SENSITIVITY (linear fit on undisturbed rows, arm mean residual)
      all six cells negative, p from 1.9e-10 to 5.5e-17

    CONTEXT CONTROL (ANCOVA, prompt fixed effects, within-prompt slope)
      faller         aligned -0.0298  5/34  p 2.4e-06 | base -0.0108 13/25 p 0.073
      matched        aligned -0.0378  2/36  p 5.4e-09 | base -0.0334  7/30  p 1.9e-04
      riser_matched  aligned -0.0321  6/32  p 2.4e-05 | base -0.0263  6/31  p 4.1e-05

Common support is not a corner: 234 of 240 (pair, role, arm) cells qualify,
median 20-22 qualifying bins per pair, over an x range of -15.5 to 0 nats.

**Read plainly: hand the model a word and what follows is EASIER for it than
its own free continuation from an equally improbable start.** Free sampling
that wanders into the tail keeps wandering; an imposed opening is
recuperated.

## Two controls, because two alternatives predicted the same sign

Both were run before this was written up, and the finding would have been
wrong without them.

CONTEXT ENTROPY. At a given opening logprob, undisturbed rows are drawn
preferentially from HIGH-ENTROPY contexts -- that is why their sampled token
was improbable -- and entropy propagates (first-token-to-rest correlation
+0.365). That alone predicts the compensation sign. Removed by holding the
PROMPT fixed (ANCOVA above): the effect survives, smaller in the base arm.

OPENING IDENTITY. At equal logprob an undisturbed opening is a TAIL-SAMPLED
token, often a fragment or punctuation, while a forced opening is a curated
content word. Removed by restricting the undisturbed arm to rows whose first
whitespace word is alphabetic and >= 2 characters (211,152 of 238,400
qualify). The effect survives and every primary estimate GREW slightly.

## Q1: NOT ESTABLISHED -- the ordering claim is WITHDRAWN ([5805])

**This section previously read "imposition, not demotion" and made the arm
ordering the headline. That claim was not tested and is withdrawn.** Caught
by malign at [5805]; their arm-vs-arm contrasts reproduce exactly under my
own recomputation from the parquet.

Every test in this finding asks *does this arm compensate* -- arm against
ZERO. The ordering claim is about arms against EACH OTHER, and no such test
existed. Run now, paired over pairs:

    aligned  faller - matched         +0.0117  14-/26+  p 0.081
             faller - riser_matched   +0.0065  18-/21+  p 0.749
             matched - riser_matched  +0.0071  17-/22+  p 0.522
    base     faller - matched         +0.0090  13-/25+  p 0.073
             faller - riser_matched   +0.0210  14-/24+  p 0.143
             matched - riser_matched  +0.0130  14-/24+  p 0.143

**No contrast reaches significance and two are at coin-flip. The three-rung
ordering the section was named after is not distinguishable from no ordering
at all.** Reproducing three medians in an order is not testing the order.

**AND IT INVERTS UNDER THE OTHER ESTIMATOR, in exactly the arm the claim
elevated:**

    by MEDIAN  matched -0.0551 > riser_matched -0.0477 > faller -0.0342
    by MEAN    riser_matched -0.0192 > faller -0.0155 > matched +0.0055

`matched` is first by median, last by mean, and changes sign. The
mean/median divergence was fenced below as a readers' note about heavy
tails; it is not a note, it is the ordering. The median is the right
traveller for each arm's OWN compensation (35 of 40 negative), per [5762] --
but an ordering compares MAGNITUDES, and magnitude is what heavy tails
destabilise.

**"NOT DEMOTION" IS ALSO WITHDRAWN, as an accepted null the data cannot
bound.** The faller-matched interval is 95% bootstrap [-0.0030, +0.0301],
which spans zero and does not exclude an arm effect the size of one already
detected elsewhere: M04's ladder finds faller-matched on D at -0.0673,
32/10, p 0.0009. Different quantity, same arms, same corpus. So movement
class is an UNDETECTED arm difference on this instrument, not an absent one.

What may be said: **being handed a word produces compensation, robustly, in
every arm and both roles, and this instrument cannot resolve whether the
word's movement class matters.** One hint, recorded as a hint: the faller
compensates LEAST in both roles at p 0.081 and 0.073, the same direction
twice, uncorrected -- not evidence, and not evidence for the withdrawn
reading either.

## Q2: not alignment-specific

DiDs at the primary estimator: faller p 0.20, matched p 0.034,
riser_matched p 1. Under the context control: faller p 0.073, matched
p 0.324, riser_matched p 1. **The two estimators disagree about WHICH arm's
DiD is nominal while agreeing completely about the main effect**, and no DiD
survives Bonferroni over the three arms (0.0167). The base compensates too.
Nothing here is alignment-specific and nothing should be read as such.

## Fences

- Self-surprisal is not comparable across models; every contrast is within
  (pair, role) and, in the context control, within (pair, role, prompt).
- The binned estimator weights bins equally within a pair, so it is a
  median-of-bins and not a passage-weighted average; the linear sensitivity
  is passage-weighted and agrees in sign, which is why both are reported.
- Mean and median diverge in several aligned cells (e.g. matched primary,
  median -0.0551 against mean +0.0055): the bin-level deltas have heavy
  tails. **The median travels**, per [5762], for each arm's OWN
  compensation. It does NOT travel for a comparison of magnitudes between
  arms -- see the withdrawn Q1 above, where this divergence turned out to
  BE the claim rather than a note about it.
- The forced arms remain SECONDARY population per plan A Amendment 1.
- AGGREGATION LAYER SECOND-SEATED ([5804]): all six contrasts, the sign
  counts, the imposition ordering (matched -0.0551 > riser_matched -0.0477
  > faller -0.0342) and the mean/median divergence all reconstruct to the
  digit from the parquet. **The ANCOVA and word-like-opening controls live
  UPSTREAM of that parquet and remain single-pass with the producer** --
  which is to say the two controls this finding most depends on are the
  part nobody has independently rebuilt.
- NOT A FRESH WITNESS RELATIVE TO S4 (`self_surprisal.md`): different
  quantity -- a residual against opening surprisal rather than a level --
  but the same corpus, the same arms, the same collection. One collection,
  two readings.
