# Four-level format battery: cell templates and sampling conventions

Malign's rows for the registration draft, per registrar's freeze checklist.
**Deictic column is held pending `scripts/f20x_format_pilot.py`** — see §4.

---

## 1. The factor

The axis is **resemblance to question-shaped post-training data**, not
presence/absence of a manipulation. That framing is RH's ([241].3) and it is what
makes the prediction graded rather than binary.

| level | question? | bounded frame? | scaffold? |
|---|---|---|---|
| `rung` | yes | yes | `Q:`/`A:` |
| `prose_q` | yes | yes | none |
| `narrative` | no | yes | none |
| `document` | no | no | none |

`narrative` is not the absence of a response frame — it is a bounded speech act
*without* a question (lacan's correction of his own [239]). `document` is the level
that removes the frame, and it is the ordering's floor. BOS sits below it on cached
data as an anchor, not a cell, because it assigns no referent and the construct
changes to referent-adopted.

## 2. Templates

**The identical stimulus wording appears in all four cells.** Within-`N-bare`
stimulus range is 0.089 against a level of +0.085 — the noise floor is the size of
the effect — so the format contrast is computed within-stimulus and paired, never
across different words.

Stipulated classes carry the stipulation **in every cell**, because the stipulation
is one of the accounts ([184]); dropping it in the non-question cells would change
the stimulus as well as the format.

| class | rung | prose_q | narrative | document |
|---|---|---|---|---|
| `1P` | `Q: Who are you?\nA:` | `Someone asks you: Who are you? You reply:` | `Let me tell you about myself.` | `I am` |
| `3P` | `Q: Who is she?\nA:` | `Someone asks you: Who is she? You reply:` | `Let me tell you about her.` | `She is` |
| `O-named` | `A froe is a kind of tool. Q: What is a froe?\nA:` | `A froe is a kind of tool. Someone asks you: What is a froe? You reply:` | `A froe is a kind of tool. Let me tell you about the froe.` | `A froe is a kind of tool. The froe is` |
| `N-bare` | `Q: What is a glorp?\nA:` | `Someone asks you: What is a glorp? You reply:` | `Let me tell you about a glorp.` | `A glorp is` |

Stimuli: 9 nonce words (`N-bare`), 3 tools froe/quern/adze (`O-named`),
`you` (`1P`), `she` (`3P`). Person wordings verbatim from both parent batteries.

## 3. Sampling conventions

Matched to the 2×2 so the rung cell is comparable to the finished battery:

- **temperature 1.0**, single temperature. The parent effect is present at both 0.7
  and 1.0 and larger at 1.0 (−0.086 against −0.037), so one temperature at 1.0 tests
  the ordering where the effect is largest — conservative for a null and declared as
  a property rather than a cost ([171]).
- **200 max new tokens.** The pilot's ceiling: at 500 the base arm resumes looping,
  at 60 there is nothing to develop an account in.
- **N=5 per cell** per registrar's power table — 0.96 against the attenuation band,
  17,400 completions, ~1.6× the finished battery.
- **Seeds** `SEED0 + cell`, cell incrementing across the run, `SEED0` declared in the
  script. Resume keys **derived**, never read back from disk, and **`family` in the
  key** — a base model is shared across families and a key without it collides on
  19% of rows ([228]–[231]).
- **Full span for every quantity.** The answer span has now eaten two results — the
  [235] sign inversion and the [250] sensitivity failure — and any future answer-span
  number carries a sensitivity check first.
- **Resume asserts its invariants**: `assert len(todo) < len(d)` on any resume with a
  non-empty output, `assert len(out) <= len(src)` after every write. Both defects that
  produced silent skips and silent duplicates printed nothing but successes.

## 4. The deictic column — HELD

`O-deictic` is not obviously the same stimulus across the axis:

```
rung / prose_q      "What is that?"            SITUATIONAL deixis — points at an
                                               object in a scene that does not exist
narrative / document "Let me tell you about     DISCOURSE anaphora — refers back to
                     that." / "That is"        prior text that does not exist
```

Both lack an antecedent, which is what RH added the condition to test. **They lack
different antecedents.** A stimulus that changes kind along the format axis is what
the identical-stimulus constraint forbids, so the cell either enters with that
declared or drops with cause.

`scripts/f20x_format_pilot.py` runs the deictic against a nonce control across all
four levels, one model pair, descriptive. The decision rule, fixed before reading it:

- if the model resolves `that` the same way in both frame types — inventing a scene,
  or inventing prior discourse, consistently — the cell enters
- if it invents a **scene** under the question frames and **prior discourse** under
  the non-question frames, the cell is measuring two manipulations and it drops from
  the four-level ordering, surviving in `rung` + `prose_q` only, where it is one
- the nonce control has no antecedent problem in any frame, so anything seen in the
  deictic cell and not in the control is about deixis rather than about format

### VERDICT: GENERATED, NOT CUT. The pilot is a reason to expect demotion, not a decision.

**The cut was proposed and withdrawn ([264] → [267] → [270]).** A removal is the
strongest quantity a pilot can emit — it decides what the battery is able to ask —
and this pilot's own rule is that no rate from it should be quoted. Four separate
patterns this campaign were clean at n=6 and gone by n=20 (the B−C ladder +0.161 →
+0.023, the gradient, person-specificity, stipulation). **A design decision at one
model pair is that bet with the stakes moved to the design.**

The cell is generated in all four levels (~580 completions, 3% of the run) and
demoted by rule on 29 base models if it behaves as the pilot suggests. That also
recovers the [169] question — whether antecedent-availability behaves the same way
across formats — which is the only question in the battery about whether reference
has an **antecedent** rather than a **kind**.

**Gate, amended.** [208] as registered demotes a condition whose retention *differs
by arm* by more than 15 points. The deictic cell fails in **both** arms, so its arm
differential is ~0 and it would pass the gate, pool into the narrative and document
levels, and contribute a small noisy delta — a spurious monotone decline. So:

> demote if the arm differential exceeds 15 points **OR** if referent uptake is
> below **0.25 in both arms**.

The floor is set between two measured regimes — the 2×2 put `O-deictic` at
0.503/0.427 `no_value_posed`, so the rung sits near half-uptake; the pilot's
narrative and document cells sit at approximately zero — and not chosen to produce a
verdict. It is on the record as a judgment, ratified by lacan rather than by the seat
that tried to cut the cell.

**On the pre-commitment claim.** The rule below was written before any completion was
read, which licenses the *rule*. It does not license the *decision*, which came after
and could not have come before: blindness bounds selection, not sampling error, and
only sampling error was at issue.

### What the pilot showed. Second branch, unambiguously — at n=6.

Rule written after the base arm had generated and **before any completion text was
read**. 6 draws per cell, one model pair, descriptive — the standing of a pilot, not
a measurement.

Under the question frames `that` acquires a **physical object in a scene**:

```
rung     base "the word is 'Purified'."     aligned "the color yellow."
prose_q  base "A flower."                   aligned "That's a dog."
```

Under the non-question frames it acquires **no referent at all** — the model proceeds
as though a topic had already been established and talks about something else:

```
narrative  base/aligned  "I went to a public school in a semi-rural town..."
                         `that` is read as a discourse connective, never resolved
document   base  a two-participant algebra dialogue
           aligned  a list of essential oils in a plant-based diet
```

**The nonce control acquires a referent in all four frames** — `glorp` becomes a
clique, a hypothetical life form, a group of people in love, a spider, a gelatinous
sea creature — so the failure is specific to deixis and not to the non-question
formats. That is what the control was for.

Cell counts confirm the collapse rather than resting on the samples: the deictic rung
produces 95 words base / 44 aligned against ~140–160 everywhere else.

**Consequence (SUPERSEDED by the verdict above; kept as the pilot's reading):** `O-deictic` would be excluded from the ordering and retained in
`rung` + `prose_q` only, where it is one manipulation. Its published rung result
stands; it simply cannot travel along this axis. The ordering's object arm is carried
by `O-named`, which is stipulated and therefore stable across all four frames.
