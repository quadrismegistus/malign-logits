---
type: plan
date: 2026-08-14
status: committed before the producer
role: plan
topics: [ICC, M03, ladder, unit-of-analysis, re-declaration]
---
# Re-declaring the M03 rung ICC

RH's call, answering @registrar's referral at [6000] and @malign's report at
[5998]. **Committed BEFORE the producer exists and before any number it
produces is seen**, as `plan_pole_sep_reduction.md` was earlier tonight. The
booked values are not available to the reduction and are not targets.

## The debt

`D_ladder_selection.md` books **ICC 0.855 (F21) and 0.846 (M03)** for the
paired difference across rungs. `d_ladder_fields.py:157` PRINTS **0.85**.
Nothing computes any of the three: the value in the producer is a string
literal inside a `print` statement. Three numbers, no producer, and the
producer has been emitting a fourth-decimal-different one at every
invocation.

## Why this is not bookkeeping

The ICC is what licenses collapsing 52 rungs to 12 and 18 scenarios, and the
collapse decides the result:

    unit = RUNG        F21  51/52 positive  p = 2.4e-14
                       M03   3/52 positive  p = 1.0e-11
    unit = SCENARIO    F21   7/12 positive  p = 0.774
                       M03   7/18 positive  p = 0.481

A HIGH ICC says the rungs are redundant and collapsing costs nothing. A LOW
one says they carry independent information and the collapse threw away
power. @malign's single reduction returned **0.085 for m03_slice** against
the booked 0.846 -- an order of magnitude. His population came out at 11
scenarios where the finding says 12, so his figure cannot be assessed; but
if anything near 0.085 survives a reduction that DOES reproduce the
population, the M03 null at p=0.481 is over-collapsed.

**So this plan can change a result, and that is the reason to run it.**

## What the substrate determines, measured before declaring anything

    meta/M03_proceduralization/results/d_ladder_fields.csv   511,242 rows
    columns  checkpoint, role_ck, step, stratum, arm, scenario, source,
             field, share, coverage, n_risers
    strata   f21_inst 12 scenarios | m03_slice 18 scenarios   <- the finding's
    arms     indiv, inst
    role_ck  base_step 39 | sft_step 43 | rlvr_step 7 | sft_endpoint 1 |
             dpo_endpoint 1
    fields   302 over 14 sources

`d_ladder_fields.py:149` declares the alignment rungs as
`AL = ("sft_step", "sft_endpoint", "dpo_endpoint", "rlvr_step")`, which is
43+7+1+1 = **52**, the finding's rung count.

## The declared reduction

**1. POPULATION FIRST, AND IT IS AN ASSERT, NOT A REPORT.** Restrict to
`role_ck in AL`; require both arms present in a cell. The producer must
reproduce **12 scenarios for f21_inst and 18 for m03_slice** and refuse
otherwise. A reduction that does not reproduce the population cannot be
assessed on its statistic at all -- which is what ended @malign's attempt and
is stated here in advance rather than discovered.

**2. `d` IS THE PRODUCER'S OWN.** `d = share(inst) - share(indiv)`, paired
within `(stratum, scenario, source, field, checkpoint)`. Not re-derived, not
redefined; the same subtraction `d_ladder_fields.py:164` performs, differing
only in that it is NOT averaged over rungs, because the rung axis is the one
whose variance is being measured.

**3. THE ITEM IS `(source, field)`, WHICH IS THE PRODUCER'S OWN ITEM.** The
report sign-tests each `(source, field)` separately. So the ICC is computed
**per (source, field)** -- scenario as group, rung as repeat -- and then
summarised over items. Averaging `d` over 302 fields first would build a
composite the producer never tests. Items with fewer than 8 scenarios are
dropped, matching `d_ladder_fields.py:167`.

**4. THE SUMMARY IS THE MEDIAN OVER ITEMS, AND THE DISTRIBUTION IS
PUBLISHED WITH IT.** A single ICC for a stratum is an aggregate over 302
items and can hide items disagreeing wildly. Quartiles and the share of
items above 0.5 travel with the median, so the headline cannot stand alone.

**5. ICC(1), ONE-WAY RANDOM EFFECTS**, scenarios random, rungs as
interchangeable repeats within scenario. Declared with its known bias: rungs
are ORDERED along training and therefore not exchangeable, so a systematic
trend across rungs is charged to within-group variance and **ICC(1)
UNDERSTATES the correlation.** That is conservative in the direction that
matters -- it biases toward rungs looking independent, which is the reading
that would REVIVE the rung unit. If the ICC comes back high under a
statistic biased against high, the collapse is safe.

## What this plan does not do

- **It does not try to reproduce 0.855, 0.846 or 0.85**, and the producer is
  not given them as targets. They are printed after every number is computed,
  labelled superseded.
- **It does not sweep ICC variants.** ICC(2,1) and ICC(3,1) are named here as
  declined, not tried: with three targets and a dozen reductions available,
  one landing means nothing ([5935], and @malign's own stop at [5998]).
- **It does not touch the RULE.** That 52 checkpoints of one training run are
  not 52 independent observations does not depend on any ICC, and the M03
  authoring guide argues it structurally. Only the number is at issue.

## Predicted outcome, recorded before running

The F21 ICC is high enough to license the collapse. **The M03 ICC is the open
question**, and the prediction is deliberately weak: I expect it above 0.085
because that figure came from a population that did not reproduce, and below
0.846 because nothing computed 0.846 either. **If it lands low enough that
the rung axis carries independent information, that is a result against the
M03 finding's unit choice and it gets reported, not re-reduced.**

## Output

`results/icc_redeclared.json` carrying `_about`, the rule above in
machine-readable form, per-stratum medians with quartiles and item counts,
the per-item distribution, and the superseded values.
