---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/opening_matched.json
date: 2026-08-13
role: finding
topics: [self-surprisal, chain, syntagmatic, forced-arms]
description: "Opening-matched comparison (RH's design): COMPENSATION, not damage. Holding the opening token's surprisal fixed -- and, in the strictest version, holding the prompt fixed and requiring a word-like opening -- a forced passage is LESS surprising to its own model than one of its own passages that opened equally improbably. Every arm, both roles, three estimators, all negative. And it is IMPOSITION rather than DEMOTION: the flat non-mover shows it as strongly as the faller. DiDs do not survive correction; the effect is not alignment-specific."
---
# Opening-matched: the chain compensates, and it is about imposition not demotion

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

## Q1: imposition, not demotion

The three arms are matched on ALIGNED probability, so an arm difference
would separate demotion from imposition per se. **There is none of the right
shape**: `matched` -- the flat non-mover -- shows the LARGEST aligned
compensation (-0.0551 primary, -0.0378 under the context control), and the
faller the smallest. So what produces compensation is being handed a word,
not the word's movement class. This is the cleanest separation in the
forced series, and it runs against the intuition that a demoted word is
what disturbs the chain.

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
  tails. **The median travels**, per [5762]; the means are printed beside it
  and the divergence is stated rather than smoothed.
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
