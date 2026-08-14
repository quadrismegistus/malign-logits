---
type: plan
date: 2026-08-14
status: committed before the producer
role: plan
topics: [pole_sep, reduction, M05, emergence]
---
# Declaring the `pole_sep` per-checkpoint reduction

RH's call at [5958], answering @dario's item 10 hold. **This plan is committed
BEFORE the producer exists and before any number it produces is seen.** That
ordering is the whole point: the six values in
`findings/pole_sep_is_not_about_poles.md` cannot be reproduced from the
committed artifact by any stated rule, and a reduction chosen because it
reproduces them would be a recipe fitted to a target ([5935]). So the rule is
declared on the artifact's properties alone, and **whatever it yields is the
number**, including if it disagrees with all six.

## The problem

`results/m05_pole_sep.csv` is 166,254 rows at (checkpoint, group, role, layer)
grain: 95 checkpoints x ~22 groups x 3 roles x 33 layers. Every published
figure needs ONE number per checkpoint. Nothing states how.

## What the artifact determines, measured before declaring anything

    166,254 rows, 95 checkpoints, 22 groups, 0 NaNs in pole_sep
    n_layers        33, constant across every checkpoint
    roles           both / control_a / control_b
    cells with all 3 roles   50,193
    cells with only 1 role   15,675
    max |both - control_a|   0.0000000000   (over the 50,193)
    max |both - control_b|   0.0000000000
    groups per checkpoint    21 or 22, NOT constant

## The declared reduction

**1. ROLE IS NOT A DIMENSION OF THIS MEASUREMENT. Deduplicate on it.**
`pole_sep` is computed from a group's own two pole prompts and is bit-identical
across role -- max difference exactly 0.0, not approximately. The finding says
so and it verifies. So the three role rows are one measurement written three
times, and **pooling roles is a weighted median whose weights are the number of
controls that happened to be run for that cell** -- 3 for some, 1 for others.
That is an artifact of run coverage, not of the model. Take one row per
(checkpoint, group, layer).

This alone separates this plan from the obvious candidate: @dario's median over
all cells, and my own `role == "both"` variant, differ ONLY because of this
uneven weighting, and neither is defensible once the identity is measured.

**2. TWO STAGES, LAYERS INSIDE GROUPS.** Median over the 33 layers within
(checkpoint, group), then median over groups. A one-stage median over all
(group, layer) cells lets a group with more layers present count more; layers
are constant here so the two coincide today, but the two-stage form states the
unit -- **the group is the unit, the layer is the thing summarised** -- and does
not silently change meaning if layer coverage ever varies.

**3. THE GROUP SET IS THE COMMON SET, AND ITS SIZE IS PUBLISHED.** Group count
varies 21/22 across checkpoints, so a curve over all available groups mixes
composition change with real change: a checkpoint can move because a group
entered or left. Restrict to groups present at EVERY checkpoint on the ladder
being plotted, and write the count and the excluded names into the artifact.

**4. MEDIAN, NOT MEAN**, throughout. The M05 riders say ranks not levels; the
same reason the acquisition curves use medians applies here.

## What this plan does NOT do

- **It does not try to reproduce the six booked values**, and the producer will
  not be given them as targets. It prints them beside its own output for the
  reader's information only, clearly labelled as superseded.
- **It does not pick a single layer.** A layer selection is defensible and might
  even be what the original did -- the near-miss at stage1-step0 on BOTH columns
  (0.7975 vs 0.795 real, 1.3873 vs 1.396 null) drifting wrong by step16000 is
  the signature of a layer or normalisation effect that grows with training. But
  choosing WHICH layer after seeing that pattern is the fitting this plan exists
  to avoid. If a layer rule is ever wanted it needs its own prior declaration.
- **It does not touch the finding's argument**, which is that the real and null
  columns move together. That claim is about co-movement and survives every
  reduction tried; the finding's own next paragraph says THE LEVEL GAP LICENSES
  NOTHING.

## Predicted outcome, recorded before running

The co-movement holds: real and null collapse and recover together on both
lineages. The LEVELS will differ from the booked six, probably by more at later
checkpoints than at step0. **If co-movement fails, that is a real result against
the finding and it gets reported, not re-reduced.**

## Output

`results/m05_pole_sep_reduced.csv` (one row per checkpoint per column) and
`results/m05_pole_sep_reduced.json` carrying `_about`, the rule above in
machine-readable form, the common group set, and the superseded values.
