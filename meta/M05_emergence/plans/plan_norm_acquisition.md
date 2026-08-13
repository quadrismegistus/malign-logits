---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [capacity, norms, modulation]
description: "Plan: norm acquisition/modulation curves — when does the distribution's K-scale composition (transgressiveness, charge, concreteness, ...) start moving, in pretraining and across the alignment stages? No new compute: the census word_probs stores joined to k_ratings. Directions inherited from K at the deployed cut for the alignment half; the pretraining half is descriptive."
---
# Plan: norm acquisition — when does modulation install?

RH's question (2026-08-13, in session): "did we ever use the non-capacity
prompts across rungs — when does it learn to modulate transgressiveness
or charge or concreteness?" We never did. No fleet is needed: the M05
census already holds word-grain probabilities per rung for the 584-text
battery on BOTH ladders (the same `word_probs` choke point the syntax and
sense curves ran through), and `k_ratings` is word-keyed — K's words came
from this prompt family's top-20s, so coverage is high by construction.

## The quantity

Per (ladder, rung, prompt): the MASS-WEIGHTED MEAN of each K scale over
the rated words in the cell's resolved distribution (renormalised within
rated mass), plus the rated-mass share as the cell's own coverage figure.
Naming rule: `dist_mean_k_<scale>`; coverage `k_rated_mass_share`. Raw
rows first (one row per cell), curves computed from the table by a
separate reader — the sense-pipeline pattern exactly.

## The two questions

1. **PRETRAINING (descriptive, no directions):** when does the
   distribution's composition on each scale start MOVING, and when does
   it start DIFFERENTIATING by prompt (cross-prompt SD of the cell means,
   against a words-shuffled null)? Composition curves like G's; the
   differentiation curve is the "contextual deployment" capacity — when
   the model learns WHERE charged vocabulary belongs.
2. **ALIGNMENT (directions inherited from K at the deployed cut, RH may
   amend):** across the OLMo stage boundaries, when do the deployed-cut
   signatures install — transgressiveness/bodily-harm/concreteness/
   arousal DOWN, charge/register/valence UP? Is the installation an SFT
   event (M05-A's pattern) or a DPO event? Per-prompt existence before
   any pooled step (the prompt-unit doctrine); the fall is a step at a
   boundary or it is accrual, and which one is the finding.

## Riders inherited with the instrument

k_ riders travel ([5581]-era, fields.py): one coder, not human norms;
register_level descriptor-only; vulgarity sparse (floors are not nulls);
RANKS NOT LEVELS — stage CONTRASTS are order-like and fine, absolute
thresholds are not. CHARGE IS NOT AROUSAL (0.54 calibration; K's own
rider). The distribution-mean is a COMPOSITION quantity: it moves when
mass moves between rated words, and its coverage share is reported beside
it everywhere (unrated mass is censored, not zero — the sense pipeline's
own discipline).

## Population, unit

The two ladder populations, never pooled (data/m05_checkpoint_population
.json, data/pythia_population.json); the prompt is the unit within rung;
sign tests over prompts at stage boundaries; per-rung existence before
any pooled curve. Producer: m05_norm_acquisition.py, writes
data/m05_norm_mass.parquet; reads only through the word_probs choke
point.
