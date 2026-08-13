# Findings H: the norm signature is installed by SFT, partially rebought by DPO, and re-suppressed by RLVR — two modules, opposite signs, visible at checkpoint grain

**Status: single registrar pass over the committed table
(`results/norm_acquisition_increments.json`, producer
`scripts/m05_norm_acquisition.py`, raw rows `data/m05_norm_mass.parquet`,
145,741 cells); no new compute — the K-scale composition of the already-stored
resolved distributions. Quotable forms live in the claims register; every
number below carries the k_ riders (bottom).**

The instrument: for each (ladder, rung, prompt) cell, the mass-weighted mean
of each of the seven K scales over the K-rated words of the resolved
next-word distribution, renormalised within rated mass
(`dist_mean_k_<scale>`), with `k_rated_mass_share` stored per cell. Stage
increments are PAIRED PER PROMPT (n=584; increments table n=581 after
missing-cell drops), sign tests with ties excluded and reported. Medians
travel; `median_nonzero` sits beside each for the tie-heavy scales.

## 1. SFT installs the named signature (base → SFT endpoint, OLMo)

Paired per-prompt increments, 581 prompts:

| scale | up/dn (ties) | median | p (ties excl.) |
|---|---|---|---|
| transgressiveness | 120/332 (129) | -0.0040 | 4.6e-24 |
| bodily_harm | 119/299 (163) | -0.0007 | 5.7e-19 |
| charge | 186/365 (30) | -0.0259 | 2.0e-14 |
| concreteness | 242/328 (11) | -0.0292 | 3.6e-04 |
| valence | 313/231 (37) | +0.0030 | 5.0e-04 |
| register_level | 324/236 (21) | +0.0046 | 2.3e-04 |
| vulgarity | 143/186 (252) | 0.0 | 2.0e-02 |

Transgressiveness, bodily harm, and charge DOWN; valence and register UP;
concreteness DOWN (de-concretization at the distribution grain, matching
M05-A's page-grain finding). Vulgarity is tie-dominated (252 of 581) — its
row is directional at best.

## 2. DPO buys part of it back — concreteness and charge REBOUND (SFT → DPO)

| scale | up/dn (ties) | median | p |
|---|---|---|---|
| concreteness | 451/111 (19) | +0.0457 | 1.4e-49 |
| charge | 450/82 (49) | +0.0119 | 1.7e-62 |
| transgressiveness | 324/89 (168) | +0.0004 | 1.9e-32 |
| bodily_harm | 252/130 (199) | 0.0 (+0.0010 nz) | 4.3e-10 |
| register_level | 268/272 (41) | 0.0 | 0.90 |
| vulgarity | 119/132 (330) | 0.0 | 0.45 |

The DPO rebound is the table's largest single effect (concreteness
+0.0457, 451 of 562 non-tied prompts up). The rebound REVERSES the SFT sign
on four scales while register_level — the scale SFT raised — is untouched
(p=0.90). This is the two-module dissociation at norm grain: what SFT
installs and what DPO adjusts are different axes ([5720]'s reading: SFT
cuts, DPO sorts).

## 3. RLVR re-suppresses, harder and across the board (DPO → RLVR)

All seven scales move, five at p < 1e-25: transgressiveness 44/366
(p 2.7e-64), bodily_harm 49/329 (p 4.8e-52), concreteness 103/458
(p 2.2e-54), charge 58/473 (p 6.2e-82), vulgarity 42/199 (p 1.3e-25) — all
DOWN; register UP (333/204, p 2.9e-08). The RLVR stage is the most
uniformly signed operation in the table.

## 4. The NETs: where the ladder ends up is not the sum of its steps

Paired NET base→DPO: concreteness is a DEAD HEAT (285/286, median 0.0,
p=1.00) — the DPO rebound returns the distribution to its base concreteness
almost exactly, while transgressiveness/charge stay suppressed and
valence/register stay raised. Paired NET base→RLVR restores the SFT-shaped
signature (all SFT signs, similar magnitudes: transgressiveness 120/332
p 4.6e-24, charge -0.0202 p 1.4e-12), now with concreteness also net-down
(261/309, p 0.049, marginal).

RIDER (the [5730]/[5732] lesson, pre-agreed): NETs are computed as their own
paired per-prompt contrasts. Summed stage medians do NOT equal the median
NET, and any reading that adds rows 1-3 to predict row 4 is wrong by
construction.

## 5. Pythia panel: differentiation across pretraining, behind a hard fence

THE FENCE FIRST (step-128 coverage fence, pre-agreed): below step 8 the
instrument resolves a median 0.5-0.7% of next-word mass (theta censoring on
a near-uniform distribution) and k_rated_mass_share sits at 0.63; resolved
mass reaches ~0.16-0.18 only by step 32-64. Every composition number below
step ~64 is a reading of a sliver and is NOT QUOTABLE; the panel starts
where the instrument does.

From step 128 (rated coverage ~1.0): the resolved distribution's
composition starts at the function-word floor (concreteness 1.08, charge
1.01 — the mass sits on abstract high-frequency words) and DIFFERENTIATES
across pretraining: concreteness climbs 1.08 → 2.87 at the final rung,
charge 1.01 → 1.27, transgressiveness 1.000 → 1.024 (late, small).
Valence's median is pinned at 4.00 throughout (scale midpoint; the median
is the wrong summary for it — distributional reads only). One rung
(step 8) is short 2 of 584 prompts; all others complete.

## The k_ riders (travel with every number above)

One coder family; RANKS NOT LEVELS (medians compare, differences do not
scale); register_level is descriptor-only, construct NOT ESTABLISHED;
vulgarity sparse (tie-dominated rows flagged above); CHARGE IS NOT AROUSAL.
The unrated tail is censored, never zeroed — `k_rated_mass_share` is the
per-cell coverage figure and travels with any per-cell read.

## Provenance

Producer run 2026-08-13 (both ladders, 249 checkpoints reached, choke point
`malign_logits.movement.word_probs` at theta=0.001 — the same censoring
declaration as the verse fleet's, measured per cell via residual).
Increments JSON committed same day. This doc is the registrar's write-up of
[5720]-thread numbers; second-seat reproduction from the parquet is the
standing upgrade path.
