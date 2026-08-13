---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades
date: 2026-08-13
role: plan
topics: [self-surprisal, forced-arms, ladder, m04-adjacent]
description: "Plan: SELF-SURPRISAL BY ARM (A|A) — does a model become LESS surprised at its own continuation when a faller or its matched control is forced, compared with undisturbed and the risen arms? RH's question, 2026-08-13. No new compute: gen_scores already holds self-scored logprobs for the passage corpus including all forced arms. Arm semantics established from the frozen table's own medians, not from the arm names ([5790]/[5791]/[5792])."
---
# Plan: self-surprisal by arm — is the model less surprised at itself after a forced faller?

RH, 2026-08-13, in session: *"Does A|A (aligned on aligned completions)
become less surprised at itself with faller or faller-matched forces than at
undisturbed, riser, or riser-matched gens?"*

This is a different question from F3 and a sharper one. F3 measured
predictability to a THIRD PARTY (GPT-2) and found a forced faller makes the
continuation less surprising in both arms. A|A asks whether the model finds
its own continuation more predictable **to itself** — which, if it holds,
is retreat into its own high-probability register rather than a fact about
corpus typicality.

**ARM SEMANTICS, ESTABLISHED FROM THE TABLE RATHER THAN THE NAMES, because
this seat got them wrong once today ([5789], withdrawn at [5792]).** Median
arm delta over all 8,169 cells, computed here and reproducing malign's
[5790] and registrar's independent [5791] to within 0.03:

    faller         -0.0152   FELL
    matched        +0.0002   FLAT -- this is the non-mover
    riser_matched  +0.0044   ROSE, held at the FALLER's aligned probability
                             (median log2(q/q_faller) = +0.162; the frozen
                             table stores this receipt as its own field)
    riser          +0.0454   ROSE, and +3.668 log2 HIGHER in probability

So `faller / matched / riser_matched` is a three-rung ladder at ONE
probability -- fell / flat / rose -- and `riser` varies probability rather
than direction. **This plan therefore uses the three-rung ladder and does
NOT treat `riser - matched` as a direction test**, which would confound
direction with 3.7 log2 of improbability. F3's riser_matched null was a
real riser result all along.

## Population and why there is no new compute

`gen_scores`, `corpus='passage'`, **`model = scorer`** (self-scoring),
`scorable=1`, `n_nan=0`, `n>3`. Coverage verified before this plan:
904,544 forced and 238,400 undisturbed self-scored rows over 84 models.
Arms from the committed arms table keyed (pair, prompt, forced_word);
`pair` supplies base/aligned role by splitting on `>`. SmolLM2 excluded,
deepseek fenced (text-grain fence does not bind logprobs, but the pair is
excluded anyway so this instrument's population matches F3's).

## The exclusion that decides whether the measure means anything

**THE FORCED TOKEN IS AT POSITION 1 AND MUST BE DROPPED.** Measured before
this plan was written, self-scored passage rows with n>25:

    position 1   forced -2.281   undisturbed -4.726
    position 2   forced -3.425   undisturbed -3.168
    positions 3, 5, 20                 converged

Position 1 is the forced word: forced words are SELECTED high-mass
candidates (mass >= 0.001, matched on aligned probability), while the
model's own first token is a temperature-1.0 draw from the whole
distribution, so its mean logprob is near the negative entropy. Comparing
arms with position 1 included would therefore measure the selection rule,
not the model's relation to its own text. **Position 1 is dropped from
EVERY arm including undisturbed**, so the slice is identical across arms;
the position-1 values are reported separately as their own descriptive row,
never inside the headline. Position 2's small forced-vs-unforced gap
(-3.425 vs -3.168) is the one-token perturbation cost and is reported, not
removed.

## Measures and unit

Self-surprisal = mean negative logprob over positions 2..n. Cell =
(pair, prompt, role, arm) mean over its sequences. Primary contrasts,
paired per (pair, prompt) within role, and reported at BOTH grains: cell
grain (large n) and PAIR grain (n<=41, the conservative unit and the one
the campaign's variance probe [5766] says survives).

    S1  faller        - undisturbed
    S2  matched       - undisturbed
    S3  faller        - matched          <- FELL vs FLAT, at one probability
    S4  riser_matched - matched          <- ROSE vs FLAT, at one probability
    S5  riser_matched - undisturbed
    S6  riser         - matched          reported ONLY as descriptive; it
                                         confounds direction with +3.7 log2
                                         of probability and is not a
                                         direction test
    DiD on S3 and S4: aligned excess minus base excess.

## Directions, declared before any number

  P1 (directional, from F3's third-party result): a forced faller LOWERS
     self-surprisal relative to its matched control (S3 negative) in at
     least one arm. F3 found the continuation more predictable to GPT-2;
     P1 says the model agrees about its own text.
  Q1 (open, no direction): whether the effect is DIRECTION-SPECIFIC --
     S3 (fell vs flat) non-null while S4 (rose vs flat) is null, both at
     the same aligned probability. F3 found exactly that shape on
     third-party surprisal; whether the model's own scoring agrees is
     open. If both move it is "any movement"; if neither, the self-scored
     metric is insensitive here.
  Q2 (open): whether the DiD is null, as it has been on composition,
     level, trajectory and third-party predictability. A NON-null DiD
     here would be the first alignment-specific forced-arm result in the
     campaign, and would mean the aligned model's retreat into its own
     register is something the base does not do.
  Q3 (open, descriptive): the ordering RH's question implies — whether
     faller and matched sit BELOW undisturbed, riser and riser_matched.

## Fences

Self-surprisal is not comparable ACROSS models (different tokenizers,
different entropies), which is why every contrast is within (pair, prompt)
and never a pooled level. The undisturbed arm's continuation is sampled
from the model's own distribution while forced arms are conditioned on a
selected word: dropping position 1 removes the selection from the measure
but NOT the conditioning, so S1/S2/S4/S5 compare differently-conditioned
text and are read as such. S3 and S4 are the clean within-forced contrasts,
because only they hold aligned probability fixed while varying direction.

Producer: `scripts/m06_self_surprisal.py`. Results:
`results/self_surprisal.json` + per-cell parquet.
