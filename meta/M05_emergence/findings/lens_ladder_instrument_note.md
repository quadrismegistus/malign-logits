# The lens ladder: the depth signature is head-dependent, and the ratio cannot resolve pretraining

**Status: one positive finding, and it is about our method. Everything else is
null, a trend at noise scale, or a bound on the measure.** Recorded as an
instrument note rather than a finding because the only defensible headline is
negative — and because it constrains a sentence about alignment that was about
to be written.

Producer `scripts/m05_lens_ladder.py` (95 rungs, 33 layers, 21 groups, two
frozen heads, 131,736 rows, ~3 hours). Analysis
`scripts/m05_lens_ladder_analysis.py`. 127,744 rows survive the degeneracy
guard (3.0% dropped). **One lineage — Olmo-3 — so nothing here generalises
across families.**

---

## What it was built to ask

The cross-section gave depth without time (38 lineages, `meta/M02_frame_exit`);
the ladder gave time without depth (95 rungs). This is the cell where they
cross, and it could ask two things neither could: does superposition arrive at
all depths at once or propagate, and does the late gate FORM across SFT's 43
rungs or switch on?

Reading the scale correctly matters and is easy to get backwards. Calibrated in
`M02_frame_exit/findings/contradiction_ratio_has_no_null.md`: **0.000 blend,
0.907 observed, 1.006 NEITHER pole, 4.031 resolution. Low is superposition.**

## 1. THE POSITIVE RESULT: the late-gate concentration depends on the head

Contrast = ratio(SFT rung) − ratio(base endpoint), same group, same layer, same
frozen head; then the share of the total |contrast| in the top eighth of the
stack.

    head          first SFT rung -> last      magnitude, first -> last
    base_main         0.228  ->  0.441            0.022 -> 0.096
    dpo_main          0.240  ->  0.242            0.022 -> 0.096

At the group unit, at the final SFT rung: median top-eighth share **0.473 under
the base head against 0.265 under the DPO head, higher under the base head in
17 of 21 groups, sign p = 0.0072.** An even spread would be 0.156.

**SFT changes the representation by the same amount under either readout. Where
in depth that change appears depends on which head you read it through.**

This is not the cross-section's comparison — that reads each model through its
OWN head — so it does not refute the late gate. What it does is remove the
inference from it: a depth signature that moves this much with the choice of
readout is not, on its own, evidence about where the computation changed. The
sentence "a late gate is cheap, reversible and cosmetic" cannot rest on it.
`M02_frame_exit/findings/depth_and_exit_do_not_join.md` reached the same
caution from the surface side on the same day.

## CORRECTION, 2026-08-11, same day: `step` is not a key

The first version of this note grouped the pretraining ladder by `step`. **OLMo
pretrains in three stages, each restarting its step numbering**, so
`stage1-step1000`, `stage2-step1000` and `stage3-step1000` all carry
`step == 1000`. Nine step values collide and **52% of the base_step rows were
affected**; the trajectory table averaged three different checkpoints into one
rung. Found while unit-testing the pole-separation claim, which surfaced the
same trap (63 groups where 21 were expected).

**What does NOT change.** `sft_step` and `rlvr_step` have zero collisions (43
revisions over 43 steps, 7 over 7), so section 1 -- the head-dependence result,
the only positive finding here -- never touched it. The group-unit tests in
section 2 compared step 0 against the maximum step, both stage-1 by
construction, so every conclusion below is unchanged: the lower band still
falls in 15 of 21 groups at p = 0.078 and the localisation still fails.

**What does change: one claim, and it is now the opposite.** The old table
showed the upper half rising monotonically across pretraining. On stage 1 alone
the top band **falls to 0.726 by step 4,000** and only then climbs to 0.980 --
a U, not a climb. The trajectory below is stage 1 only.

    step        bottom     lower     upper       top
       0        0.8976    0.9112    0.9165    0.9086
    1000        0.8185    0.8275    0.8601    0.8834
    2000        0.9177    0.8863    0.8174    0.7857
    4000        0.9242    0.8607    0.8078    0.7258   <- top-band floor
    8000        0.9444    0.8540    0.7865    0.7352
   32000        0.9365    0.8443    0.8607    0.8492
  128000        0.8870    0.8195    0.8526    0.8250
  512000        0.8899    0.8161    0.8868    0.8286
 1413814        0.8793    0.8188    0.9440    0.9796

The "half-way to its floor by step 1,000" claim for the lower band survives on
stage 1 alone (threshold 0.8650, first rung at or below it is step 1,000).

## 2. SUPERPOSITION DOES NOT ARRIVE IN ONE BAND. THE POOLED TABLE SAYS IT DOES.

Pooled over groups and layers, the story is clean and localised: only the
0.25–0.50 band falls (0.911 → 0.819), half-way there by step 1,000, while the
upper half RISES.

**At the group unit it dissolves.**

    band            step0    end     groups falling      p
    bottom 0.00-0.25  0.902  0.874      13/21           0.38
    lower 0.25-0.50   0.920  0.834      15/21           0.078
    upper 0.50-0.875  0.920  1.002       8/21           0.38
    top   0.875-1.0   0.916  0.952      10/21           1.00

The mid-stack fall is a trend, not a result. The **localisation** claim — that
this band moves and the others do not — fails outright: the lower band's change
is not reliably different from any other band's (p = 0.078 / 0.19 / 0.38, with
one nominal hit in six tests under the other head).

**The scale check settles it. Within-group rung-to-rung sd across the late
pretraining rungs is 0.049. The entire step-0-to-1.4M-step move is 0.065.** All
of pretraining moves this measure by about one adjacent-rung wobble.

Averaging 21 groups × 8 layers hid the group variance. This is the campaign's
most-booked defect and it was paid again here.

## 3. A BOUND ON THE MEASURE: step 0 is indistinguishable from a trained model

Read through a trained head, the untrained network gives **~0.90 at every depth,
with zero degenerate cells in 672 attempted.** On a scale where NEITHER is 1.006
and the trained endpoint sits at 0.88, a randomly initialised network is not
separable from a trained one at this grain.

This confirms [5426]'s correction in output terms rather than geometric ones —
untrained means spread, not collapsed. It also says the ratio has too little
dynamic range along a ladder to see a gate form, which is the real reason
question 2 has no answer here.

## 4. NULL: no endpoint alignment contrast

Base endpoint against SFT and DPO endpoints, paired within group: 10–11 of 21
groups, p = 1, under both heads. One lineage, so this carries little weight —
but it is worth recording that the ladder's own endpoints do not reproduce the
cross-section's arm effect at this unit.

## Two bugs found and fixed before any of this was believed

**A rising band has no half-way point, and the arithmetic does not say so.**
With the end value above step 0, the "half of the total fall" threshold sits
ABOVE step 0, so step 0 satisfies it and the summary printed "half-way at step
0" for bands that never fell. Now stated rather than computed.

**Two summaries of the same rows with opposite signs.** The endpoint table
reported levels as a median pooled over every (group, layer) cell beside a
contrast that was a median of paired differences: base 0.8732 against aligned
0.9072 next to a paired median of −0.0044. Both arithmetically correct; putting
them in one row was not. Levels now use the estimator the contrast uses.

## What the run bought

Not a gate forming. It bought the knowledge that **the ratio cannot see one on
this substrate**, and that the depth signature we do read off it is unstable
across readouts. Both are constraints on what may be written, which is the
honest product of three hours of compute.

To do better, the instrument has to change, not the sample: the ratio's
dynamic range along a ladder is ~0.07 against 0.05 of rung-to-rung noise. No
number of additional rungs fixes that.
