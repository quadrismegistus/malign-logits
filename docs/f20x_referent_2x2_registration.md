# Registration: the referentiality x referent-kind 2x2

**Registered by the lacan seat, 2026-07-28, at RH's design. NO ITEMS HAVE BEEN
SELECTED AND NO DATA EXISTS. Queued behind the current 2x2 battery (26 arms of 80)
and the Q1 coding run.**

Supersedes nothing. This is a new battery that answers a question the existing one
structurally cannot.

---

## The design

|  | **no referent** | **has a referent** |
|---|---|---|
| **thing** | `What is a glorp?` | `What is a gimlet?` |
| **person** | `Who is Alden Voskrit?` | `Who is George Washington?` |

Four cells. Bare, unstipulated, one question form per row. Referent kind varies down
the columns; referentiality varies across the rows.

## What this answers that nothing currently running can

**There is no nonsense-PERSON cell anywhere in the project.** Every nonce word ever
generated is a thing. So the question "does the person effect require a real
person?" has never been askable.

**There is no bare senseful-THING cell either.** `O-named` and `R-rare` are both
stipulated ("A gimlet is a kind of tool"), so an unstipulated real-thing question
has never been run.

**The interaction is the point, and it is the project's actual question.** A made-up
name has no referent but occupies a subject-position.

    alignment anchors invented people like real ones
        -> the POSITION does the work, not the referent
        -> the subject is a place in the signifier, which is the psychoanalytic claim

    alignment anchors only referring terms, person or thing
        -> the REFERENT does the work
        -> the subject is a special case of reference and nothing more

**No ladder can separate those**, because a ladder confounds referentiality with
referent kind at every rung. This is why the design is a 2x2 and not a fifth rung.

**It also holds the wh-word constant within each row.** Per Amendment 4 of
`f20x_object_registration.md`, `who` and `what` are not confounds to be removed;
they are how English marks referent kind. Varying them BETWEEN rows and holding them
constant WITHIN rows is what a referent-kind manipulation looks like once that is
accepted.

---

## ITEM SELECTION: rules fixed before any item is chosen

**This is where the design can be bent without anyone noticing, so the rules are
registered before selection and the selected list is committed before generation.**

### The matching problem, stated plainly

The two right-hand cells are not naturally matched on how much the models have seen.
`gimlet` is a deliberately rare tool; `George Washington` is among the most
documented people in English. **An unmatched design measures training-data density
and labels it referent kind.**

### Rule M1: frequency band, fixed in advance

Real items in both referring cells are selected to fall inside **one stated
frequency band** on a **single public measure chosen and named before any candidate
is scored.** The measure is fixed as **English Wikipedia article length in bytes**,
which is public, checkable by any seat, and independent of this project.

    band: 8,000 - 60,000 bytes

`George Washington` is far above this band and **is excluded by the rule that names
it in this document**, which is the intended demonstration that the rule bites.
Household names are out of both referring cells.

**If the band cannot be applied for lack of network access, the battery does not
run.** A substitute measure invented at run time is the failure mode this rule
exists to prevent.

### Rule M2: invented items must be verifiably non-referring

**Invented people are the risk.** Any plausible-sounding name is likely to be
someone, and a base model producing a real biography for what we called a nonce name
would silently move that cell into the referring column.

- Every invented surname is checked against English Wikipedia and returns no article.
- Invented surnames are built with the **same phonotactic generator** as the existing
  nonce things (`glorp`, `quiln`, `plost`, `tarnu`, `velbin`, `gorpin`, `flant`,
  `zendle`, `fenmit`), so the two nonsense cells are matched in their nonsense.
- **Fresh words, not the existing nine**, so no analysis can confuse the cells.
- Given names come from a common-first-name list, unmatched to anything.

### Rule M3: real people are historical public figures

Not living private individuals. **Alignment training contains explicit instruction
about not speculating on private persons**, so an obscure living person would trip
refusal and confound the gate with the outcome. Historical or long-public figures
only, inside the M1 band.

### Rule M4: nine items per cell

Matching the existing nonce stratum. Word-level and cell-level variance remain
unseparable at nine, as in the parent registration; going below nine buys time with
the thing the design exists to measure.

### Wording, fixed

    thing cells:   Q: What is a <item>?\nA:
    person cells:  Q: Who is <Name>?\nA:

**Present tense in both person cells.** `Who was` presupposes a dead referent, which
is a fact about the referring cell that the non-referring cell cannot match, and it
recruits obituary genre. The existing 3P cell uses present tense and this matches it.

---

## Predictions, with the falsifier on each line

1. **Main effect of referentiality.** `drift_delta(referring) > drift_delta(non-referring)`,
   pooled over both referent kinds, paired over distinct base models. *Falsified if*
   null or negative. **Prior: the current battery gives bare nonce +0.018 against
   objects +0.179, so this is expected and is the weakest prediction here.**
2. **Main effect of referent kind.** `drift_delta(person) > drift_delta(thing)`.
   *Falsified if* null or negative. **This is the seat's assigned reading and it is
   currently at +0.088, 4 of 6, p=0.56 — a direction and no more.**
3. **PRIMARY, and it is the interaction.** Does the person effect depend on
   referentiality? *Falsified if* the interaction is null, which would mean the two
   main effects are additive and the subject-position adds nothing beyond being one
   more referring term.
4. **Prediction this seat expects to FAIL, ~65%.** That invented people are anchored
   like real people — the position doing the work. If the non-referring person cell
   behaves like the non-referring thing cell, the psychoanalytic reading of these
   data loses its strongest available form and the deflationary account gains.

---

## Controls fixed in advance

**Unit** is the distinct base model, aligned arms deduplicated. Rule 2.

**Gate.** `no_value_posed` is outcome one and is reported per cell before anything
conditional. Retention differing by arm by more than 15 points in a cell demotes that
cell's conditional comparison to descriptive. **Both conditional and unconditional
figures are reported**, because `posed` is post-treatment and conditioning on it is a
collider — established at cost on the 1P term, which moves +0.113 to +0.013 under
conditioning.

**Codes.** `quiet_drift` alone carries the primary. The full code table is reported
alongside, because the current battery showed `quiet_drift` to be about 95% of all
instability and `name_arbitrary` to be the one code with a different shape — 6 of 6
on both nonce conditions against 2 of 6 elsewhere. **`name_arbitrary` is registered
here as a secondary outcome**, since a non-referring PERSON is where a name can most
clearly be arbitrary, and that is a signifier-level measure rather than a
referent-level one.

**Parameters, not runs.** Matched to the current 2x2 battery: same temperature grid,
same max tokens, same roster, same coder version. **This battery does NOT regenerate
existing conditions.** The same-run requirement of the object registration's
Amendment 1 was measured today and found to cost 1,760 completions to buy a
difference of 0.005 in the delta, against contrasts of 0.09 to 0.17. Run-to-run
variance in the quantity the contrast uses is about 5% of the effect. **Parameters
are held; the run is not.**

**Instrument** is `code_sited`. Levels are not comparable to any `code_identity`
number; directions and contrasts are.

**Multiplicity.** Holm across the three registered contrasts. Corrected values
reported, never raw.

---

## What is not controlled

- **Frequency matching is on Wikipedia bytes, which is a proxy for training-data
  density and not the thing itself.** Two items in the same band can differ several
  fold in corpus count. The band narrows the confound; it does not remove it.
- **Invented people carry a nationality and a gender that invented things do not.**
  A given name commits to both. There is no version of a person-name that does not.
- **`Who is <Name>?` invites biography; `What is a <item>?` invites definition.** The
  genres differ, and genre was the largest single artefact in the parent battery
  (Q/A looping at 63.5%). `drift_from_genre` is recorded and reported per cell.
- **Nine items is not a stimulus set.**

---

## PRIOR EXPOSURE

**This seat has seen the current battery's results at 6 base models before writing
this**, including that persons lead the ladder on both metrics, that bare nonce is
near-flat, and that the stipulation effect is 6 of 6. Those informed prediction 1
being called the weakest and prediction 4 being called likely to fail.

**No item has been selected and no data for this design exists**, which is the only
blindness that matters here and it is complete.

**RH designed the 2x2.** The seat's contribution is the selection rules and the
falsifiers, which is the part where a design of this shape gets bent.
