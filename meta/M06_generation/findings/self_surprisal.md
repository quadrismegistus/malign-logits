---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/self_surprisal.json
date: 2026-08-13
role: finding
topics: [self-surprisal, forced-arms, alignment-specific]
description: "Self-surprisal by arm (A|A), RH's question: the answer runs OPPOSITE to the hypothesis it contained. Within the forced arms the ALIGNED model is soothed most by RISEN words, not fallen ones -- rose-vs-flat at held aligned probability lowers aligned self-surprisal (pair grain -0.0077, p 0.039) and does NOTHING to base (+0.0012, p 0.75), DiD -0.0150, p 0.0166 at pair grain and 0.0013 at cell grain. THE FIRST NON-NULL DiD IN THE FORCED SERIES after four nulls. **THE MIRROR IS HALF-ESTABLISHED**: S4 (ROSE) is arm-specific at both grains, but S3 (FELL) has DiD +0.0133 p 0.636 at the PAIR grain -- null at the unit this document calls conservative -- so the base being soothed by fallen words is solid while the claim that the aligned model is NOT also soothed by them is not established (dario, [5915]). Definitional choices that a reader must have: the pair grain is the MEDIAN over prompts within pair (the mean inverts the sign counts), and the sign test KEEPS zeros in the median while excluding them from the counts (dropping them shifts S3 base to -0.0208 against the booked -0.0199). Flagged for second-seat reconstruction because it is this seat's first positive after a long run of nulls."
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

**CORRECTED 2026-08-13 ([5811]): POSITION 1 IS NOT THE FORCED TOKEN.** This
section originally said it was. Measured: the forced word is in neither the
prompt (0.0000) nor the scored text (0.0008), `len(logprobs)` equals
`tokens(text)` exactly, and `logprobs[1]` VARIES within a forced site (sd
1.75) where a forced word's logprob would be deterministic. Position 1 is
the first SAMPLED token after an unmeasured imposed word.

**Consequences, scoped:** dropping position 1 was still SYMMETRIC across the
forced arms, so **S3 and S4 are unaffected in value and stand**. But every
comparison involving `undisturbed` is confounded far more deeply than the
conditioning fence below admits -- the forced arms carry ONE MORE WORD of
context than the undisturbed arm. **S1, S2, S5 and the ARM LEVELS table are
withdrawn**, including the ordering RH's question asked about, which cannot
be answered from this instrument.

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

**EACH ARM IS SOOTHED BY THE VOCABULARY IT PROMOTED** -- but only ONE HALF of
that mirror is established as arm-specific, and the sentence overstates it.
Forcing a fallen word lowers self-surprisal in the BASE (p 0.0003) and only
marginally in the aligned (p 0.154). Forcing a risen word lowers it in the
ALIGNED (p 0.039) and not at all in the base (p 0.75).

**THE WITHIN-ARM EFFECTS ARE BOTH REAL; THE ARM-SPECIFICITY IS NOT.** A mirror
claim is a claim about DiDs, because "soothed by ITS OWN vocabulary" asserts the
other arm is not soothed the same way. Only S4 carries that:

    S4  ROSE vs FLAT   DiD pair -0.0150  p 0.0166    cell -0.0161  p 0.0013
    S3  FELL vs FLAT   DiD pair +0.0133  p 0.636     cell +0.0115  p 0.018

**S3's DiD is NULL at the pair grain**, which this document calls the
conservative unit, and reaches significance only at the cell grain. So the base
being soothed by fallen words is solid, and the claim that the ALIGNED model is
not also soothed by them is NOT ESTABLISHED where we say to look. One arm-
specific effect beside one half whose arm-specificity is open.

Found by @dario drawing it (docket [5915]): a 2x2 with two significant cells on
the diagonal reads as one symmetric finding, and nothing in the panel is false.
Same shape as the SAMESIDE column at [5914] -- the LAYOUT does rhetorical work
the contents cannot support. `m06_self_surprisal_figs.py` states it on the panel
rather than in a caption, where a rerun cannot erase it and a travelling figure
cannot shed it.

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

## RH's ordering question -- WITHDRAWN ([5811])

The table below compares sequences with different amounts of conditioning
context (forced arms are scored after an unmeasured imposed word) and is not
an answer to the question. It is kept only because [5795] posted it.

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

## Why undisturbed is the MOST surprising arm, and why it carries no weight

RH asked. The mechanism is entropy propagation from the first token, and it
is measurable rather than a story: within undisturbed self-scored passages
(224,307 rows, n>40), the correlation between the FIRST token's logprob and
the mean logprob of everything after it is **+0.365**, and between the early
window (positions 2-8) and the late one (9 onward) it is **+0.466**. A
passage that opens improbably stays improbable.

**SUPERSEDED ([5811]) as an EXPLANATION, though the correlations above are
real measurements about undisturbed passages.** I attributed the level gap
to the arms' opening typicality (-2.2 forced against -4.70 undisturbed).
The actual asymmetry is larger and simpler: **the forced arms are scored on
a continuation that follows an extra, unmeasured word**, so they have one
more word of conditioning context than the undisturbed arm at the same
nominal position. More context, more predictable continuation. Every
contrast against `undisturbed` is withdrawn rather than fenced.

## Relation to M04's ladder -- overlapping, NOT independent ([5798])

RH asked whether this is the same result as M04's ladder. It is not the
same measurement, and the evidence is not independent either.

    M04 rose-minus-flat  +0.0510  13/27  p 0.0436   on D = A|A - B|A
    S4 aligned           -0.0077  13/27  p 0.0385   on A|A

Identical sign counts, adjacent p, same direction -- but the producers
differ in window and term. M04's `k8` is `arraySlice(logprobs, 1, 8)`, the
first eight tokens; mine is `arraySlice(logprobs, 2)`, position 2 to the end.
**CORRECTED ([5811]): neither window contains the forced word** -- it is
absent from the scored text entirely, so both are windows onto the
continuation AFTER it. My earlier description of M04's window as "including
the forced word", and the credit I gave it for a position-1-cancels-in-D
property, are both void: there is no forced-word position in the array to
include or to cancel. The relation is asymmetric: M04's
window sits almost entirely INSIDE mine (positions 2-8) but is about 3% of
my measure's tokens, while their D subtracts a term mine does not have.
(Position 1 CANCELS in their D, since both arms force the same word -- a
design property that makes including it harmless there.)

**So "the first non-null DiD in the forced series" above is scoped to THIS
seat's series (I5, ascent, I6, I7, F3). M04's ladder reached a rose-flat
non-null first, on an overlapping quantity.** The two should never be
counted as two witnesses.

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
- **RETENTION IS ARM-INDEPENDENT, measured not assumed** ([5801], after
  malign's [5800] sibling-rule lesson). The declared filter
  (`scorable=1`, `n_nan=0`, `n>3`) retains aligned faller 0.9930 /
  matched 0.9933 / riser_matched 0.9917 (spread 0.0016) and base
  0.9941 / 0.9939 / 0.9942 (spread 0.0003). S4's own contrast sits on
  the widest of those gaps, 0.16%, three orders below the effect it is
  asked to explain. The check could not have been prompted by the
  output -- the numbers print identically whether or not the
  denominators diverged.

## Flagged, attacked, and standing ([5796])

This was posted flagged rather than claimed, because it is this seat's first
positive after a long run of nulls -- the condition under which its own
ledger says positives get checked least. Both asks have now been run by an
independent seat.

RECONSTRUCTION: every number at both grains, and the aligned pair-grain arm
ordering, rebuilt exactly from the parquet. One qualification found in the
rebuild and adopted here: **at POOLED CELL grain the arm ordering differs**
(matched and riser_matched rise above undisturbed), so the ordering claim
above is PAIR-GRAIN-SPECIFIC and must travel with that label.

THE TYPICALITY ATTACK -- the one this finding named as its own weakest
point -- WAS RUN AND DOES NOT LAND. Joining the S4 contrasts to the arms
table's base probabilities (6,941 cells), the exposure is real as stated:
riser_matched words are less base-typical than matched words, median gap
-0.745 log2. But it does not reach the result, twice over:

- the S4 contrast is UNCORRELATED with the base-probability gap in both
  arms (Spearman -0.030 base, -0.024 aligned). If typicality drove the base
  leg, the base contrast would track the gap; it does not.
- the DiD holds its sign in EVERY tertile of |gap| -- -0.0161 small,
  -0.0255 mid (p 0.011), -0.0105 large -- with no monotone growth in the
  confound, and the near-matched tertile, where typicality has almost
  nothing to bite on, carries the same-sized effect as the pooled read.

**So the mirror sentence -- each arm is soothed by the vocabulary it
promoted -- is the best-defended non-null in the forced series.** Still
single-pass at the PRODUCER layer per [5503]: the parquet's own construction
(position-1 drop, per-arm scoring) has not been regenerated independently.
