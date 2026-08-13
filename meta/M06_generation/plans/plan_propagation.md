---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades
date: 2026-08-13
role: plan
topics: [syntagmatic, propagation, forced-arms, chain]
description: "Plan: THE PROPAGATION SLOPE — RH's reframe. The question is not alignment-specificity but a fact about language models: does forcing an IMPROBABLE word damage the chain or does the syntagm absorb it? The arms design holds probability FIXED, so it is the wrong instrument; but q is known for all four arms and `riser` sits +3.67 log2 above the rest, so the slope dy/d(log q) is computable. Compared against the same slope in undisturbed generation, which is the self-selected baseline."
---
# Plan: the propagation slope

RH, 2026-08-13, in session, after the alignment-specific reading came back
narrow:

> *"But this is only because we took as our null a thesis about
> alignment-specificity -- whether falling differs from falling-matched. What
> if the hypothesis is about language models generally: forcing an improbable
> word -> (h1) makes the syntagm compensate or (h2) damages it"*

**The arms design is the wrong instrument for that question and it is worth
saying why.** `faller`, `matched` and `riser_matched` are MATCHED ON ALIGNED
PROBABILITY -- the corpus isolates movement DIRECTION at constant
probability. RH's hypothesis needs probability to VARY. So every contrast run
on this corpus so far has been blind to it by construction.

Two things make it answerable anyway: `riser` sits **+3.67 log2** above the
other three (the property that makes it a confound for direction tests makes
it the manipulation here), and every arm word carries a full-vocabulary
probability `q` in the frozen table -- verified conserved across the whole
`twp_residual` store (605,550 rows, median 1.0000002).

## The quantity

    x = log2 q of the FORCED word, under the scoring model
        (aligned rows: q; base rows: q - delta, the arms table's own route)
    y = mean logprob of the continuation
        (= the whole array; the forced word is not in it, [5811])

    SLOPE b = dy/dx, fitted WITHIN (pair, prompt, role) across the four arms

`b` is the fraction of the opening's improbability that propagates into what
follows.

    b ~ 1   the chain inherits the imposition entirely
    b ~ 0   the chain absorbs it entirely -- COMPENSATION
    b > 0 substantially, but < 1   partial propagation; the interesting case,
          and the number is the finding

## Directions, declared

  H1 COMPENSATION: b is near zero -- an improbable forced word costs the
     continuation little or nothing.
  H2 DAMAGE: b is substantially positive -- the less probable the forced
     word, the harder the continuation.

**The reference that makes b interpretable, and it already exists:** in
UNDISTURBED generation the within-prompt slope of continuation logprob on
the sampled opening's logprob is **+0.016 to +0.024** (79 fitted lines,
`opening_matched` ANCOVA). That is roughly **2% propagation** for a
SELF-SELECTED opening. So:

  H3 (the comparison, and the one that actually separates imposition from
     improbability): **b_forced vs b_undisturbed ~ 0.02.** If they are
     equal, an imposed improbable word behaves exactly like a
     self-sampled improbable one and there is nothing special about
     imposition. If b_forced is materially larger, imposition propagates
     more than sampling does -- which is the damage claim in its
     defensible form.

  Q1 (open): whether b differs by role. Tested as a paired contrast, not
     read off two point estimates ([5805]).

## Fences

- Four points per cell is a thin regression; cells with fewer than 3 arms
  present are dropped, and the per-cell fit quality is reported before any
  slope is quoted.
- `q` is a WORD probability and the continuation is scored in TOKENS. For
  single-token words these coincide ([5818]); for multi-token words `log q`
  is the word's total mass and the comparison is approximate. The
  single-token share is measured and reported, and the slope is reported
  BOTH pooled and restricted to single-token arm words.
- The undisturbed reference slope comes from a different fit (token logprob
  as x, within-prompt), so H3 is a comparison of like-shaped quantities and
  not of identical estimators; it is read as an order-of-magnitude test.
- Nothing here revisits the alignment-specific question, which is answered
  and narrow.

Producer: `scripts/m06_propagation.py`. Results: `results/propagation.json`
+ per-cell parquet.
