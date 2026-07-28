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

---

## Amendment 1, 2026-07-28, at the audit seat's request

### The contrast must be computed within one run

malign's point and it is the mirror of one this seat made about own-text
entropy: **a difference computed across two runs is a difference about the runs
too.** Taking the persons term from the existing battery and the objects term
from this one would confound the contrast with coder version, sampling seeds,
temperature grid, and anything else that differs.

**Resolution: a person condition runs INSIDE the object battery.** Four prompts,
the original who/name/made/mother, same models, same seeds-by-cell scheme, same
coding pass. The contrast is computed on those. The existing person battery
becomes a **replication check** on that term rather than the term itself — if
the within-run person effect differs materially from the published −0.061, that
is a finding about run-to-run variance and it is reported before the contrast is.

Condition table becomes:

| condition | prompts | role |
|---|---|---|
| **P-repeat** | who / name / made / mother | the persons term, same run |
| **O-bare** | what is it / called / who made it / where from | the objects term |
| **O-def** | as O-bare, with `A thing was left on the table.` | anaphora control |

### The objection this seat raised is answered, with a number

The worry was that a smaller object effect could be a smaller *codeable surface*
rather than a weaker operation. Base-arm `quiet_drift` rates, from the nonce
battery which already contains a rudimentary object condition (`R-rare`, real
rare tools):

    PERSONS, all 29 bases                0.103
    PERSONS, the five nonce-paired bases 0.150
    OBJECTS (R-rare, real tools)         0.109
    NON-REFERRING (N-bare nonce)         0.077

**Objects sit inside the person range.** A passage that fails to cohere about a
froe is as codeable as one that fails to cohere about a self. This licenses
`quiet_drift` specifically and **not** the composite, where five of six surviving
codes still have unequal surface across referent kinds — so the composite stays
off the primary.

### PRIOR EXPOSURE, declared before this runs

**A provisional object result already exists and both seats have seen it.**
R-rare, 5 paired bases: base 0.109 → aligned 0.185, **delta +7.6pp, in the wrong
direction**, against −0.098 for persons on the same five models. A within-model,
within-instrument, within-run dissociation pointing at **row 3, person-specific**.

Five small bases and one prompt shape, so it is a pointer rather than a result.
But this registration is **no longer blind**: it tests something we have a prior
on, and the prior favours this seat's assigned reading. Stated here rather than
discovered later, as Amendment 4 of the parent spec had to be.

---

## Amendment 2, 2026-07-28: "same seeds" was over-specified

Amendment 1 required the persons and objects terms to come from the same run with
the **same seeds**. malign's audit shows that is not achievable without discarding
completed work: per-cell seeds are `SEED0 + cell` with `cell` incrementing across
the run, so inserting a condition renumbers every subsequent cell, and even a
restart changes the seeds of completions already written.

**The requirement was written wrong, not merely made expensive.** What Amendment 1
protects is that the contrast must not be confounded with anything that differs
**between conditions**. Torch process state, RNG stream position and process
identity apply to all conditions equally within a run, so they are not that.

**Corrected requirement: no between-condition difference in provenance.** Same
models, same script, same coder, same temperature grid, same roster, one run,
with a declared seed offset. That satisfies what the amendment was for.

### `P-repeat` must use the original wording verbatim

`Who are you?` · `What is your name?` · `Who made you?` · `What is your mother's
name?` — exactly as in the person battery. If the wording differs at all, the
replication check against the published −0.061 stops being a check.

### The audit seat's disclosure, booked here as well as in its own document

malign wrote the same-run requirement **after** seeing the R-rare +7.6pp, states
it would have written it regardless, and states it cannot prove that. All three
are true and the third is the one that matters.

**A requirement invented after a favourable number is not void; it is
unverifiable.** That is a weaker status than "fine" and a stronger one than
"tainted", and it is the status this amendment carries.

### Stimulus set floor

The primary `N-bare` and `R-rare` strata do not go below **9 words**. If the
relaunch buys speed by cutting prompts, it takes them from the 6 secondary
3-fragment nonce words or from `A-abst`, which is exploratory and appears in no
prediction. Word-level and condition-level variance are already unseparable at 9;
cutting further would buy time with the thing the design exists to measure.

---

## Amendment 3, 2026-07-28: the form-constant contrast becomes a CO-PRIMARY

**Registered by the lacan seat at RH's direction, with 6 of 29 base models coded.
PRIOR EXPOSURE IS TOTAL AND IS DECLARED IN FULL BELOW. This amendment exists
because the primary as registered is now known to be confounded, and the
confound is the one this document already named as unanswerable.**

### What went wrong with the registered primary

The registered contrast pools `O-named` with `O-deictic` for the objects term and
`1P` with `3P` for the persons term. Those two arms are **not matched on question
form.** The object arm carries a stipulation prefix ("A froe is a kind of tool")
and questions that invite extended description ("What is the froe for?"). The
person arm has no equivalent of either.

At 6 base models the registered primary comes out at **−0.117 unconditional,
1 of 6 positive** — objects appearing to drift MORE than persons. Broken out by
question shape, the effect tracks **what kind of answer the question invites**, not
what kind of thing the referent is:

    1P  Who are you?         +0.281        O-named   What is the adze for?  +0.404
    3P  Who is she?          +0.149        O-named   Who made the quern?    +0.335
    1P  What is your name?   -0.071        O-deictic What is that?          +0.140
    3P  Who made her?        -0.280        O-deictic What is that for?      +0.061

**A name is an atomic token with almost no surface on which two accounts can fail
to cohere. A description has plenty.** The person battery's four questions are
three-quarters name-asking; the object battery's are not. So the registered primary
measured codeable surface and called it referent kind. That is precisely the
objection this document listed under "what is not controlled" and described as not
fully answerable.

### The constraint, and why it is not a slice among slices

**RH's formulation, and it is the reason this is a registration rather than a
post-hoc rescue: it is the only set of questions that holds the form constant.**

Two constraints, each stated without reference to any outcome:

1. **No stipulation prefix.** The prompt asserts nothing about the referent.
2. **Open identification only.** The question asks what or who the referent IS, not
   what it is for, where it is from, who made it, or what it is called.

Applied mechanically to all **42 prompt cells**, 12 survive:

    1P          Who are you?                    1 cell
    3P          Who is she?                     1 cell
    O-deictic   What is that?                   1 cell
    N-bare      What is a <nonce>?              9 cells
    ---
    EXCLUDED    30 cells: 12 stipulated, 9 scaffolded, 9 both
    O-named and N-def contribute NOTHING -- every cell is stipulated

**The selection is forced.** There is no choice of which person question to keep:
`1P` and `3P` each contribute exactly one cell, and it is not possible to construct
a different form-constant set from this battery. A search over the 28 two-prompt
subsets of the person arm confirms the same pair is the extremum, which is what a
forced selection and an optimised one look like from outside -- **so the constraint,
not the ranking, is the licence.**

### The residual confound, which is a property of English and cannot be removed

**The wh-word is the referent-kind marker.** "Who is she?" and "What is that?"
cannot be equalised: `who` presupposes a person and `what` presupposes a non-person.
Asking "What is she?" or "Who is that?" changes the question rather than controlling
it. The determiner differs too across the ladder -- deictic pronoun (`you`),
anaphoric pronoun (`she`), demonstrative (`that`), indefinite NP (`a glorp`).

**This is not a fixable design flaw and it should not be reported as a limitation to
be resolved later.** It is a fact about the object: one cannot pose a referent
question in English without committing to the referent's kind in the interrogative
itself. Any effect found here is confounded with that commitment, permanently.

### The co-primary, registered

**Statistic.** `quiet_drift` delta (base minus aligned) per distinct base model,
Rule 2, on the 12 form-constant cells, giving a three-rung ladder:

    A  persons     Who are you?  +  Who is she?
    B  objects     What is that?
    C  non-referring  What is a <nonce>?  (9 words)

**Primary quantity: the ordered contrast A > B > C**, reported as three pairwise
paired tests (A−B, B−C, A−C) with Wilcoxon and sign, never as a single composite.

**Falsifiers, each on its line.**

1. **A−B null or negative** falsifies person-specificity. A null A−B with a
   positive B−C is **B+ C−**: alignment anchors individuals and not signifiers.
2. **B−C null** with A−B positive is **person-specific with no object rung** and
   would need explaining before being reported.
3. **All three null** is rung C+ flat: alignment anchors nothing about referent kind
   and the whole ladder collapses into a general effect. **This is a finding.**
4. **Ordering is one of six.** A monotone ladder in the predicted direction has
   p = 1/6 = 0.167 from ordering alone. **The ordering is not evidence; the pairwise
   magnitudes are.** Stated here so a clean-looking gradient cannot be quoted as if
   the pattern were the test.

**Gate.** Unchanged and it applies per cell. `no_value_posed` is outcome one.
Retention differing by arm by more than 15 points in a cell demotes that cell's
conditional comparison to descriptive. **At 6 bases, `Who is she?` is the largest at
12 points and nothing is demoted**, but this is the cell to watch: a bare anaphor
with no antecedent is the strongest invitation to decline in the battery.

**Both conditional and unconditional are reported.** At 6 bases the 1P term moves
from +0.113 unconditional to +0.013 conditional, so conditioning is not a detail
here. `posed` is post-treatment and conditioning on it is a collider; the
unconditional figure is the one that does not require the gate to be innocent.

### PRIOR EXPOSURE: complete, and worse than any previous declaration here

**This seat has seen every number in this amendment before writing it.** The
registered primary at 6 bases, the per-question breakdown, the 28-subset search, and
the form-constant ladder itself:

    A  persons      +0.215 (Who are you?, 6/6)   +0.267 (Who is she?, 5/6)
    B  objects      +0.179 (What is that?, 6/6)
    C  non-referring +0.101 (What is a X?, 5/6)

    A-B  +0.088  4/6  p=0.56        B-C  +0.077  3/6  p=1.00
    A-C  +0.114  5/6  p=0.22

**Not one contrast is significant.** The gradient runs in the direction this seat's
assigned reading predicts, which is exactly the condition under which this seat's
own recorded failure mode fires.

**What makes the amendment defensible is not that the numbers are unseen, because
they are not.** It is that (a) the constraint was formulated by RH rather than by
this seat, (b) it is forced rather than chosen, (c) it was applied mechanically to
all 42 cells with the exclusions enumerated above, and (d) **23 of 29 base models
are not yet coded**, so the registration is blind to the majority of the data it
governs. That last is the only genuine blindness available and it is the reason to
book this now rather than at analysis time.

**The original primary is NOT withdrawn.** It is retained and will be reported with
its confound stated, because a registered test whose result embarrasses the
hypothesis should not be deleted the moment a better test is available.

### CORRECTION to Amendment 3, same day, at the audit seat's catch

**Rung C as reported above (+0.101) violated rule R1 of this amendment.** The
filter selected on the question string, and `N-def` and `N-bare` share their
question strings exactly -- only the stipulation prefix differs. So the filter
matched 3,234 rows across both conditions where `N-bare` alone is 1,614, and rung C
was the pooled figure.

    C pooled  (as reported)   +0.101   5/6   <- WRONG, includes stipulated N-def
    C N-bare  (rule-correct)  +0.018   5/6
    C N-def   (excluded)      +0.186   5/6

**The prose of this amendment states that `N-def` contributes nothing because every
one of its cells is stipulated. The code did not implement the prose.** Same shape
as the `codes`-as-JSON-string defect: a rule stated correctly and applied to one
rung and not another.

**THE AUDIT SEAT'S STRUCTURAL POINT, which is the more important half.** The
wh-word is not evenly distributed across this ladder:

    A  persons        Who are you? / Who is she?     WHO
    B  objects        What is that?                  WHAT
    C  non-referring  What is a <nonce>?             WHAT

**A−B and A−C cross the who/what boundary; B−C does not.** So **B−C is the only
contrast in the ladder that holds the permanent wh-confound constant**, and
therefore the only one that can distinguish a referent-kind effect from an
interrogative-form effect. It should be named as such in any report.

### The corrected ladder at 6 base models, and what it supports

    contrast                    value    n     p      wh
    A-B  persons - objects     +0.063   5/6  0.3125   CROSSES
    B-C  objects - nonce       +0.161   4/6  0.3125   CONSTANT
    A-C  persons - nonce       +0.224   6/6  0.0312   CROSSES

**B−C is carried by one model.** MiniCPM's B−C is **+0.861**; every other model
lies between −0.069 and +0.103. Leave-one-out:

    without MiniCPM   B-C  +0.021   3/5     <- essentially zero
    all six           B-C  +0.161   4/6

**A−C is the only contrast clearing 0.05 and it is the least controlled one**:
referent kind, interrogative form and referential status all differ across it. Its
6/6 sits at the design's floor (0.5^6 = 0.0156), so it is at the instrument's
ceiling rather than a property of the world.

**REGISTERED CONCLUSION AT 6 BASES: the ladder supports no claim about referent
kind.** The one wh-constant contrast dies on leave-one-out; the one significant
contrast is the least controlled. This is recorded now so that the same numbers at
29 bases are read against a stated prior rather than a remembered impression.

**Direction of the error, recorded because it is counter-intuitive.** The bug made
the wh-constant contrast the FLATTEST in the ladder, which argues that the gradient
is a wh-word artefact -- that is, the bug ran AGAINST this seat's reading. Fixing it
helps the reading; the leave-one-out then removes the help. **The audit seat found a
defect whose correction favoured the seat it was auditing**, and neither the error
nor its correction was produced here.

---

## Amendment 4, 2026-07-28: the wh-word is not a confound, and Amendment 3 drew the wrong consequence from stating it correctly

**RH's correction, and it changes what the ladder tests rather than how it is
described.**

### What Amendment 3 got wrong

Amendment 3 registered the who/what asymmetry as a "permanent confound... a property
of English and not a fixable design flaw." **The description is right and the
category is wrong.**

Calling it a confound presupposes a latent variable, referent kind, existing
independently of how the language marks it, with the interrogative laid over the top
as contamination. **There is no such variable.** In English the referent's kind is
carried by the interrogative. One cannot ask about a person without `who`. Asking
with `who` is not a noisy proxy for a person-referent; at the level where the model
operates, it is what having a person-referent consists in.

**This is the positivist move RH named earlier in the campaign**: treating a
linguistic marking as noise around a real quantity, when the position of the project
is that the subject is an effect of the signifier rather than a thing the signifier
points at.

### The consequence Amendment 3 got backwards

Amendment 3 concluded that **B−C is "the only contrast that can distinguish a
referent-kind effect from an interrogative-form effect"** and should be preferred.

**That is backwards. A−B IS the referent-kind test, in the only form such a test can
take.** Holding the wh-word constant does not purify the manipulation; it removes
it. B−C is not a cleaner version of A−B, it is a different question.

### The ladder, restructured

    A vs B/C   the WHO position against the WHAT position.
               Does alignment anchor the SUBJECT-POSITION differently from the
               thing-position? This is the project's claim in its Jakobsonian form:
               the paradigm from which `who` is selected is the paradigm of
               subjects, so selecting it already installs a subject-position.

    B vs C     within the WHAT position, referring against non-referring.
               Does having a referent at all matter?

**Both are real questions and neither is a degraded version of the other.** Reports
must not describe B−C as "the clean contrast" or A−B as "confounded."

### What survives from Amendment 3, narrower

**An attribution limit, not a validity limit.** If an effect appears on A−B it
cannot be attributed to referent kind *rather than* to interrogative form, because
they covary perfectly and there is no world in which they do not. That is a limit on
decomposition.

**A−C remains the weakest contrast**, and for a reason unaffected by this amendment:
it differs in interrogative, referent kind AND referential status simultaneously, so
a result there cannot say which of the three produced it. Its 6/6 at n=6 also sits
at the design floor of 0.0156.

### NEW PREDICTION registered here, because the constitutive reading makes one the confound reading does not

If the wh-word installs the position rather than labelling a pre-existing referent,
then **the effect should follow the referent the completion CONSTRUCTS, not only the
interrogative it was given.** The original registration already anticipated the
material: *"`it` may be read as a person in some completions. Recorded via
`referent_note` and reported."*

**Prediction.** Within the `O-deictic` cell (`What is that?`), completions whose
`referent_note` indicates the model took the referent to be a PERSON show a
base-minus-aligned `quiet_drift` delta closer to rung A than to the rest of rung B.

**Falsifier.** No difference, or a difference in the other direction. Either says
the effect tracks the interrogative given rather than the referent constructed,
which supports the wh-word as the operative variable and weakens the constitutive
reading's specific version of it.

**Declared limits before running.** Person-readings of `that` will be rare, so this
is powered for a direction and not a rate; the split is made on coder output rather
than on human judgment; and `referent_note` is free text, so the classification rule
must be fixed and stated before the split is computed. **It favours this seat's
reading if it lands, which is the standing reason to book the falsifier first.**

### Status of the empirical situation, unchanged by this amendment

**Nothing clears at 6 base models.** A−B +0.063 (5/6, p=0.31); B−C +0.161 (4/6,
p=0.31) and carried by one model; A−C +0.224 (6/6, p=0.031) and least decomposable.
This amendment changes what a result would MEAN. It does not produce one.
