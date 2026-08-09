# Alignment's semantic-field signature is roster-wide and not contradiction-specific

Alignment moves generated prose off the concrete particulars of narrative and
onto interiority: fewer proper names, places, objects, bodies and time markers;
more emotion, cognition, communication, sociality, evaluation and dominance.
The shift is large, consistent across 26 model pairs, and survives correction on
39 of 79 fields across three lexicon granularities.

**None of it is specific to contradiction.** With a syntactically matched
single-pole control, the contradiction-specific residual survives on 0 of 79.

## The design

The F11 quintuplets supply a control matched on form rather than on nothing:

    both        He loved her and hated her and wanted to
    control_a   He loved her and adored her and wanted to
    control_b   He hated her and despised her and wanted to

Same clause count, same conjunction, same hinge. What `both` has and the
controls lack is the contradiction. Per model pair, per field:

    D_CONTRA  = rate(aligned, both)     - rate(base, both)
    D_CONTROL = rate(aligned, controls) - rate(base, controls)
    SPECIFIC  = D_CONTRA - D_CONTROL

`both` is the role name, not "both arms". All three are aligned-minus-base
differences; what changes is which prompt they are taken on.

Population: 54,080 English continuations, the 26 complete pairs of
`data/f11_l2_receipt.json`, roles `both` / `control_a` / `control_b`, both arms.
Rates are shares of *classified* words (`n_counted`), so a difference in how
much of an arm the lexicon knows cannot masquerade as a field difference.
Passage is the unit; the median over 26 pairs is the statistic; Wilcoxon over
pairs is the test; Benjamini-Hochberg at FDR 0.05 within each source.

Producer: `scripts/l2_semantic_fields.py`. Results:
`results/l2_fields_{meta,norms,usas_fine}.json`.

## The result

    source        fields   general effect   contradiction-specific
    meta              13       7 survive          0
    norms             24      11 survive          0
    usas_fine         42      21 survive          0
                              39 of 79            0 of 79

The signature, with the proportion of the 26 pairs moving that way:

    DOWN                                   UP
    personal_names       23/26  p 4.2e-07  dominance:dominant      22/26  p 2.2e-04
    duration_phase_age   22/26  p 3.3e-06  sociality_belonging     23/26  p 1.4e-04
    geography_places     22/26  p 4.1e-06  cognition_mental        22/26  p 7.0e-05
    "other" (residual)   25/26  p 5.7e-07  emotion_affect          21/26  p 1.8e-03
    time_aspect          22/26  p 7.5e-06  evaluation_modality     21/26  p 3.2e-03
    matter_objects       20/26  p 1.8e-03  language_communication  19/26  p 8.6e-03
    body_health          22/26  p 3.9e-03  perception_sensation    19/26  p 1.3e-02
    concreteness:concrete 20/26 p 4.7e-03  core_cognition_belief   20/26  p 1.0e-03

`dominance:neutral` falls in 23 of 26 while `dominance:dominant` rises in 22:
the prose becomes more agentive and less neutral. `personal_names` is the
sharpest single field.

`D_CONTROL` sits on top of `D_CONTRA` for every surviving field. Largest
|SPECIFIC| anywhere is 0.0061 (`arousal:calm`, norms), under 0.6 points of
classified vocabulary, and it does not survive correction.

## What could have broken it, and did not

**Pole-word echo.** The three roles differ lexically by construction, and those
differing words -- loved/hated, loved/adored, hated/despised -- are exactly what
a field count keys on. Measured on this corpus the same day, 82% of the
continuations that brought both pole words back were restating the prompt. The
primary therefore strips each passage's own prompt content words before
counting. The `--keep-poles` sensitivity is unchanged: still 7 of 13 general and
0 of 13 specific on `meta`, with values moving by under 0.003.

**Reporting only the residual.** A SPECIFIC near zero is ambiguous between
"alignment does nothing here" and "alignment does the same thing to both prompt
types". Only D_CONTRA and D_CONTROL separate them, and here they say the second.
Had this been reported as a difference-in-differences alone, a 39-of-79 positive
result would have been filed as a null.

## Limits

Compositional: rates are shares within a source, so a rise in one field is
partly a fall in another. The direction of any single field should not be read
independently of the others.

Coverage differs by arm and by lexicon -- 0.951 base against 0.957 aligned on
USAS, but 0.497 against 0.518 on the Warriner/Brysbaert norms. The norms gap is
2 points and the norms rows should be held more loosely than the USAS ones.

The null is bounded, not absolute: it rules out contradiction-specific effects
of the size these 26 pairs can resolve, not effects of any size.

This measures vocabulary, not structure. It says nothing about whether the
contradiction is *held* or *resolved* in a passage -- see the paired coder work,
where OEDIPALIZED came back 13.7% base against 14.0% aligned (p = 1.0, n = 500
matched) against an 8.6pp resolvable difference.
