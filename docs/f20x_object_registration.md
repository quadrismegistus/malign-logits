# Registration: does alignment anchor OBJECTS, or only persons?

**Registered by the lacan seat, 2026-07-28, before any object data exists at any
seat. Audited by malign per the rotation. Timestamp booked by registrar.**

Design 1 of the three proposed in docket [142], promoted to high-value by RH's
reframing in [162]. It is the experiment that decides between two readings the
project has been pursuing simultaneously without noticing they want opposite
results.

---

## Why this matters more than it did yesterday

    A. reference to PERSONS          established. Both batteries, null interaction.
    B. reference to any INDIVIDUAL   THIS DOCUMENT.
    C. any SIGNIFIER holding value   provisionally null (nonce battery, 5 bases).

**The two arguments in this project want opposite outcomes, and only one was
being tracked.**

The Weatherby sub-argument wants **C+**: if alignment anchors signification as
such, the base model is the structuralist object and the deployed model is where
structuralism stops describing it.

The subject argument — which is the project's core frame — wants **B− C−**: if
alignment anchors persons and *not* objects, the operation is targeted at the site
of subjectivity rather than being general coherence work. That is a much stronger
claim than "alignment anchors reference," and it is the psychoanalytic one.

**This seat spent the previous phase arguing for the outcome that weakens its own
assigned reading**, on the grounds that rung C was "larger." Larger is not the
same as better for the argument, and the direction check that should have caught
it did not, because the check asks whether a conclusion flatters the seat's
reading and this conclusion appeared not to.

## The outcome matrix, and the objection to it

    A+ B+ C+   alignment anchors signification as such. Weatherby-strong.
               Weakest for the subject: the "I" is one case among many.
    A+ B+ C-   anchors reference to individuals, not signifiers.
               Subject is a special case of reference.
    A+ B- C-   PERSON-SPECIFIC. Strongest for the subject argument.
    A+ B- C+   anomalous; would need explaining before being reported.

**The matrix is binary and the world probably is not.** The likely outcome is
graded: alignment anchors everything somewhat and persons most. That is a fifth
row the matrix cannot express, and it is not a spoiled result — it is the
*degree* of person-specificity, which is a quantity rather than a verdict.

**Registered consequence: the primary statistic is the CONTRAST, not the
presence or absence of an object effect.** Reporting "objects: p=0.07, therefore
B−" would be reading a threshold as a fact.

---

## Design: one variable changes

Same rung, same roster, same sampling, same n, same seed scheme as the person
batteries. Only the referent's kind moves — from a person to an artifact.

| condition | prompts | mirrors |
|---|---|---|
| **O-bare** | `What is it?` · `What is it called?` · `Who made it?` · `Where is it from?` | who / name / made / mother |
| **O-def** | `A thing was left on the table.` before each of the above | the nonce battery's stipulated arm |

**O-def exists because of the anaphora asymmetry**, already documented in
Amendment 7: `you` is deictic and supplies its own referent; `it` is anaphoric
with none. O-def supplies a minimal antecedent without describing the object, so
the model has something to refer to without being told what it is.

**`Who made it?` is retained deliberately even though its answer is a person.**
It is the one prompt shared with the person battery in surface form, and if the
object effect appears only there, that is evidence the effect tracks persons
inside object questions — which is the finding, not a flaw.

---

## Predictions, with the falsifier on each line

1. **PRIMARY, and it is a contrast.** `drift_delta(persons) − drift_delta(objects)`
   over the 29 paired base models, on `quiet_drift`, gated. *Falsified if* the
   paired test on the contrast misses p < 0.05 in either direction — a null
   contrast is **B+**, and is a finding.
2. **Direction.** This seat's assigned reading predicts persons > objects.
   *Falsified if* the contrast is negative or null.
3. **Degree, not verdict.** If both effects are significant, the reported quantity
   is the ratio of effect sizes with its interval, never "both anchored."
4. **Prediction I expect to fail, ~30%.** I expect `Who made it?` to behave like
   the person battery and the other three not to. If the object effect is uniform
   across prompts, this is wrong and the person-inside-the-object-question reading
   goes with it.

---

## Controls fixed in advance

**The gate applies, per Amendment 7 and with the nonce battery's threshold.**
`no_value_posed` is outcome one; every other code is conditional on a referent
having been posed. **If retention differs by arm by more than 15 points, the
conditional comparison is demoted to descriptive** and the primary reads off the
gate. Objects with no antecedent are a strong invitation to decline, exactly as
`she` was.

**Entropy controls planned here, not added afterwards.** Both regressors, with
their objects named: own-text (process property) and cross-scored on the partner
arm's completions (model property). **The cross-scored bound is the one that
answers the confound**, since the objection is about the model — established at
cost when the composite's sign flipped between them.

**Codes.** Five of the eleven are person-specific by their written definitions
(`number_shift`, `origin_displaced`, `name_arbitrary`, `mania`, `frame_exit`).
The composite runs over the remaining six and carries a different name from the
parent composite. Same decision as the nonce battery, made for the same reason.

**Genre.** Object questions recruit product descriptions, manuals and listings as
strongly as person questions recruit interviews and depositions. `genre` is
recorded and `contradiction_from_genre` is reported per condition.

**Unit** is the distinct base model, aligned arms deduplicated. Rule 2.

**Multiplicity.** Holm across conditions within `quiet_drift`; corrected values
reported, never raw.

---

## What is not controlled

- **Objects and persons differ in more than referent kind.** Persons have names,
  ages, jobs and relations; objects have materials, makers and locations. The
  coding scheme's remaining six codes are not equally applicable to both, and a
  smaller object effect could be a smaller *codeable surface*. This is the
  strongest objection to the design and it is not fully answerable — the partial
  answer is that `quiet_drift` (a description that fails to cohere) applies
  identically to both, which is why it and not the composite is the primary.
- **`it` may be read as a person** in some completions. Recorded via
  `referent_note` and reported.
- **Fifteen prompts is not a stimulus set.** Four per condition, and word-level
  variance is not separable from condition-level variance.
