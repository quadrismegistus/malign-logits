---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/propagation.json
date: 2026-08-13
role: finding
topics: [syntagmatic, propagation, forced-arms, chain]
description: "RH's reframe answered: forcing an improbable word DOES damage the chain, and the damage is ~1%. Propagation slope +0.0083 aligned / +0.0073 base nats-per-bit (37/3 and 36/4 of 40 pairs, p 2.0e-08 and 1.9e-07) = about 1.3% of the opening's improbability reaching the continuation. The undisturbed self-sampled baseline is 1.6-2.4%, so an IMPOSED improbable word propagates no more than a self-sampled one. H2 in direction, H1 in magnitude: the syntagm absorbs roughly 99% of a paradigmatic imposition. Role difference marginal and variant-dependent after the inherited-predicate repair (p 0.081 pooled / 0.039 single-token, worth 0.003 nats-per-bit) -- not quotable as an alignment effect."
---
# The propagation slope: the chain absorbs about 99% of an imposition

Plan: `plans/plan_propagation.md`, committed before this producer, with H1,
H2 and H3 declared. Producer `scripts/m06_propagation.py`; results
`results/propagation.json` + per-pair parquet. No new compute. Single pass;
[5503] applies.

RH's reframe, after the alignment-specific reading came back narrow: *"what
if the hypothesis is about language models generally: forcing an improbable
word -- (h1) makes the syntagm compensate, or (h2) damages it?"*

**The arms design is blind to this by construction and that is worth stating
first.** `faller`, `matched` and `riser_matched` are MATCHED ON ALIGNED
PROBABILITY; the corpus isolates movement direction at constant probability,
which is exactly the variable this question needs to vary. `riser` sits
+3.67 log2 above the other three -- the property that makes it a confound for
direction tests makes it the manipulation here -- and every arm word carries
a full-vocabulary `q`, so the slope is computable after all.

## The measurement

    x = log2 q of the forced word under the SCORING model
        (aligned rows q; base rows q - delta, the arms table's own route)
    y = mean logprob of the continuation (the whole array; the forced word
        is not in it, [5811])
    b = dy/dx within (pair, prompt, role), one point per arm

    601,324 rows | 10,658 cells | 9,423 fitted
    median log2-q spread within a cell: 3.78 bits

    ALL ARM WORDS        aligned +0.0083  37/3   p 2.0e-08  (40 pairs)
                         base    +0.0073  36/4   p 1.9e-07
                         aligned - base +0.0039  p 0.081
    SINGLE-TOKEN ONLY    aligned +0.0081  38/2   p 1.5e-09
                         base    +0.0068  32/8   p 1.8e-04
                         aligned - base +0.0029  p 0.039

**CORRECTED before these numbers were second-seated ([5828]): the first run
carried an INHERITED PREDICATE.** I required the reconstructed base
probability `p > 0` for every arm word, but `p` is used only by the BASE
role, which takes it as its x; the ALIGNED fit never touches it. The filter
dropped 8.4% of arm-words and did so ARM-ASYMMETRICALLY -- **23.5% of
`riser_matched` against 0.0% of `faller`**, because a word that rose from
p=0 is exactly the kind that fails it -- so it removed the most extreme
risers from the aligned regression's high-q end. Same defect malign found
in `ladder_confirm.py` the same hour; theirs cost ~23% of a magnitude, mine
cost little (aligned +0.0088 to +0.0083) but the exposure was structural,
not small. Numbers above are the repaired run.

## Both hypotheses are right, at different scales

**H2 IN DIRECTION.** The slope is positive and it is not marginal: 35 or 36
of 40 pairs, p ~1e-06 to 1e-07, on both arms and under the single-token
restriction. A less probable forced word does leave a less predictable
continuation. The chain is not indifferent to what is put in it.

**H1 IN MAGNITUDE, overwhelmingly.** b is in nats-per-BIT, so per nat of
opening improbability the continuation loses **b / ln 2 ~ 0.013 nats**.
**About 1.3% of the imposition reaches the continuation, and roughly 99% is
absorbed.** The damage is real, robust, and almost nothing.

**H3: IMPOSITION COSTS NOTHING EXTRA.** The same slope for a SELF-SAMPLED
opening in undisturbed generation is **0.016 to 0.024 nats per nat**
(`opening_matched`'s within-prompt ANCOVA, 79 fitted lines). The forced
slope, 0.013, is the same order and if anything SMALLER. **An imposed
improbable word propagates no more than one the model chose itself** --
which is what the architecture requires if provenance is invisible to the
forward pass (malign's argument, [5810] §1), and this is the first version
of that prediction that is actually testable rather than broken.

**And it is essentially not alignment, with one honest wrinkle.** aligned -
base is +0.0039 (p 0.081) pooled and +0.0029 (p 0.039) single-token. The
repaired run moved this from clearly null to marginal, and the two variants
disagree across 0.05. **The difference, if real, is 0.003 nats-per-bit -- a
third of a percent of propagation -- so it changes nothing about the reading
and should not be quoted as an alignment effect on a single variant crossing
a threshold.** Both arms absorb at essentially the same rate.

## What it means, stated at the right scope

This is a fact about autoregressive language models, not about alignment: put
an improbable word in a model's mouth and roughly ninety-nine percent of that
improbability does not survive into what follows. The syntagmatic axis is
extraordinarily absorbent, and it absorbs an external imposition exactly as
readily as its own unlikely choice.

It also explains, retrospectively, why every forced-arm probe in this
campaign returned null on the demoted side. **Composition (I5), level
(ascent), trajectory and third-party predictability (F3), self-surprisal
(S3): all of them were looking for a downstream trace of an imposition that
propagates at about one percent.** The nulls were not insensitivity to
alignment; they were the chain doing what this slope says it does.

## Fences

- `q` is a WORD probability, the continuation is scored in TOKENS. 68.7% of
  arm words are single-token under the pythia tokenizer, where the two
  coincide; the restricted variant is reported and agrees on the slope.
- Four points per cell is a thin regression. Cells with fewer than three arms
  present are dropped; the within-cell log2-q spread is
  reported so the slope is not read off a corner (median 3.78 bits).
- The undisturbed reference comes from a different fit (token logprob as x,
  within prompt) so H3 is an order-of-magnitude comparison of like-shaped
  quantities, not of identical estimators. A single-instrument version is
  owed if the comparison is ever leaned on.
- Per-pair rows persisted for reconstruction ([5820]).
