---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/self_surprisal.json
date: 2026-08-13
role: finding
topics: [self-surprisal, forced-arms, alignment-specific]
description: "Self-surprisal by arm (A|A), RH's question: the answer runs OPPOSITE to the hypothesis it contained. Within the forced arms the ALIGNED model is soothed most by RISEN words, not fallen ones -- rose-vs-flat at held aligned probability lowers aligned self-surprisal (pair grain -0.0077, p 0.039) and does NOTHING to base (+0.0012, p 0.75), DiD -0.0150, p 0.0166 at pair grain and 0.0013 at cell grain. THE FIRST NON-NULL DiD IN THE FORCED SERIES after four nulls; flagged for second-seat reconstruction because it is this seat's first positive after a long run of nulls."
---
# Self-surprisal by arm: each model is soothed by its own promoted vocabulary

Plan: `plans/plan_self_surprisal.md`, committed before this producer.
Producer `scripts/m06_self_surprisal.py`; results `results/self_surprisal.json`
+ per-cell parquet (55,043 cells). No new compute -- `gen_scores` already
held self-scored logprobs for every arm. Single pass; [5503] applies.

RH's question, 2026-08-13: *does A|A become LESS surprised at itself with
faller or faller-matched forces than at undisturbed, riser, or riser-matched
gens?*

**The answer is no, and the ordering runs the other way.**

## The design facts, both taken from the artifacts

POSITION 1 IS THE FORCED TOKEN and is dropped from every arm including
undisturbed: forced words average -2.18 to -2.28 there while the model's own
sampled first token averages -4.70, because forced words are selected
high-mass candidates and a temp-1.0 draw is not. Comparing with position 1
in would measure the selection rule.

THE ARMS ARE fell / flat / rose AT ONE ALIGNED PROBABILITY. `matched` is the
non-mover (median delta +0.0002), `riser_matched` is a RISER held at the
faller's aligned probability (+0.0044; the frozen table stores the receipt
in `riser_matched_log2`, median +0.170), and `riser` sits +3.67 log2 higher
so it varies probability rather than direction. Established from the table
after this seat read the arms wrong from their names ([5789], withdrawn
[5792]; confirmed three ways at [5790]/[5791]).

## The result

    (negative = LESS surprised at its own continuation; pair grain is the
     conservative unit, n=39-40 pairs; cell grain beside it)

    S3  faller - matched (FELL vs FLAT)
        aligned   pair -0.0053  15/25  p 0.154     cell -0.0075  p 0.054
        base      pair -0.0199   8/31  p 0.00029   cell -0.0130  p 3.8e-08
        DiD       pair +0.0133  22/18  p 0.636     cell +0.0115  p 0.018

    S4  riser_matched - matched (ROSE vs FLAT)
        aligned   pair -0.0077  13/27  p 0.0385    cell -0.0097  p 0.0056
        base      pair +0.0012  21/18  p 0.749     cell +0.0000  p 0.918
        DiD       pair -0.0150  12/28  p 0.0166    cell -0.0161  p 0.0013

**EACH ARM IS SOOTHED BY THE VOCABULARY IT PROMOTED.** Forcing a fallen word
lowers self-surprisal in the BASE (p 0.0003) and only marginally in the
aligned (p 0.154). Forcing a risen word lowers it in the ALIGNED (p 0.039)
and not at all in the base (p 0.75). The two effects are mirror images, and
S4's DiD is non-null at BOTH grains.

**THIS IS THE FIRST NON-NULL DiD IN THE FORCED SERIES**, after composition
(I5), level (ascent), trajectory and third-party predictability (F3) all
returned null. It says the aligned model has a register it recognises: put
it on-register and it settles into text it finds more predictable, and the
base model -- which never promoted that vocabulary -- does not.

**AND IT IS NOT A PROBABILITY EFFECT FOR THE ALIGNED ARM, BY DESIGN.** The
three arms are matched on ALIGNED probability, so under the aligned model
faller, matched and riser_matched are equiprobable; the aligned differences
therefore cannot be the forced word's own likelihood to the scorer (which is
excluded from the measure anyway). The BASE arm has no such protection -- for
the base, the faller is a high-probability word and the riser_matched a low
one -- so **the base's S3 is confounded with typicality and the DiD inherits
that**. The clean sentence is the aligned-only one: at held aligned
probability, a RISEN word lowers aligned self-surprisal and a FALLEN word
does not measurably.

## RH's ordering question, answered directly

Pair-grain median self-surprisal by arm:

    aligned   undisturbed 1.920 > matched 1.851 > faller 1.835
              > riser_matched 1.808 > riser 1.768
    base      undisturbed 2.944 > matched 2.932 > riser_matched 2.930
              > riser 2.917 > faller 2.899

For the aligned model, faller and matched are **not** below riser and
riser_matched -- they are above. The risen arms soothe it most. For the base,
the faller is lowest, as the mirror predicts. All forced arms sit below
undisturbed in both roles, but that comparison is fenced (see below) and
carries no weight.

## Fences

- **Anything against `undisturbed` is confounded** and is reported only for
  completeness (S1, S2, S5). Forced arms are conditioned on a selected
  high-mass word; undisturbed text follows a temp-1.0 draw that may come
  from the tail. Dropping position 1 removes the selection from the MEASURE
  but not from the CONDITIONING. S3 and S4 are the clean contrasts.
- **`riser - matched` (S6) is descriptive only** -- aligned pair -0.0385
  (p 4e-05), base -0.0191 (p 1.4e-05) -- because `riser` sits 3.67 log2
  higher in probability, so it confounds direction with improbability. It
  is not evidence for the direction claim and is not read as such.
- Self-surprisal is not comparable across models (different tokenizers and
  entropies), which is why every contrast is within (pair, prompt) and no
  pooled level is quoted except the by-arm medians above, which are
  within-role orderings.
- Multiplicity: two declared DiDs (S3, S4). S4's pair-grain p 0.0166
  survives Bonferroni over those two (0.025); it would not survive
  correction over all six contrasts, and the six were not all declared as
  DiD tests.
- 12,993 forced cells carried a word absent from the arms table for their
  (pair, prompt) and were dropped, named here rather than folded in.

## Flagged, not quoted

**This is the first positive result this seat has produced after a long run
of nulls, which is exactly the condition under which its own ledger says
positives get checked least.** The per-cell parquet is keyed
(pair, role, prompt, arm) for the [5760]-form reconstruction, and the S4 DiD
should not be leaned on until a second seat rebuilds it. The specific thing
to attack: whether the base arm's typicality confound, which the DiD
inherits, is doing the work.
