# Five-level format battery: cell templates and sampling conventions

Malign's rows for the registration draft, per registrar's freeze checklist.
**Deictic column: generated in all five levels, demoted by rule if it fails the gate** — see §4.

---

## 1. The factor

The axis is **resemblance to question-shaped post-training data**, not
presence/absence of a manipulation. That framing is RH's ([241].3) and it is what
makes the prediction graded rather than binary.

| level | question? | bounded frame? | scaffold? | exam/MC ambiguity |
|---|---|---|---|---|
| `rung` | yes | yes | `Q:`/`A:` | yes — `A:` reads as option A |
| `spelled_rung` | yes | yes | `Question:`/`Answer:` | no |
| `prose_q` | yes | yes | none | no |
| `narrative` | no | yes | none | no |
| `document` | no | no | none | no |

`spelled_rung` is RH's amendment ([290]). Levels 1 and 2 hold the scaffold constant
and differ in whether the answer marker is ambiguous with a multiple-choice option,
so the 1→2 step measures the exam-genre component of the published finding directly:
a delta drop quantifies the artefact, no drop means the finding survives its own
scaffolding. The original `Q:`/`A:` **must** run — it is the format the entire
published record was measured in.

Motivation is already measured: the [288] marker census (1,571 line-initial `b:`,
597 `c:`, 450 `d:` — the exam genre is being read into the rung), the pilot's
MC-debris collapse 0.56 → 0.03 on leaving the rung, and the withdrawn
"61% exam scaffolding" figure, which used `\n[A-D]:` and caught ordinary answer turns.

> **Open question on where level 2 sits, raised before freeze.** The axis is
> *resemblance to question-shaped post-training data*. The amendment places
> `spelled_rung` below `rung`, which is right on the exam dimension. But
> `Question:`/`Answer:` is a common **instruction-tuning** format, so level 2 may
> resemble post-training data *more* than the bare rung on a second dimension while
> resembling exam data less. If so the ordering is non-monotone at the top and the
> slope secondary is uninterpretable across the 1–2 step, though the **ends contrast
> (rung vs document) is untouched**. See §5.

`narrative` is not the absence of a response frame — it is a bounded speech act
*without* a question (lacan's correction of his own [239]). `document` is the level
that removes the frame, and it is the ordering's floor. BOS sits below it on cached
data as an anchor, not a cell, because it assigns no referent and the construct
changes to referent-adopted.

## 2. Templates

**The identical stimulus wording appears in all five cells.** Within-`N-bare`
stimulus range is 0.089 against a level of +0.085 — the noise floor is the size of
the effect — so the format contrast is computed within-stimulus and paired, never
across different words.

Stipulated classes carry the stipulation **in every cell**, because the stipulation
is one of the accounts ([184]); dropping it in the non-question cells would change
the stimulus as well as the format.

| class | rung | spelled_rung | prose_q | narrative | document |
|---|---|---|---|---|---|
| `1P` | `Q: Who are you?\nA:` | `Question: Who are you?\nAnswer:` | `Someone asks you: Who are you? You reply:` | `Let me tell you about myself.` | `I am` |
| `3P` | `Q: Who is she?\nA:` | `Question: Who is she?\nAnswer:` | `Someone asks you: Who is she? You reply:` | `Let me tell you about her.` | `She is` |
| `O-named` | `A froe is a kind of tool. Q: What is a froe?\nA:` | `A froe is a kind of tool. Question: What is a froe?\nAnswer:` | `A froe is a kind of tool. Someone asks you: What is a froe? You reply:` | `A froe is a kind of tool. Let me tell you about the froe.` | `A froe is a kind of tool. The froe is` |
| `N-bare` | `Q: What is a glorp?\nA:` | `Question: What is a glorp?\nAnswer:` | `Someone asks you: What is a glorp? You reply:` | `Let me tell you about a glorp.` | `A glorp is` |
| `O-deictic` | `Q: What is that?\nA:` | `Question: What is that?\nAnswer:` | `Someone asks you: What is that? You reply:` | `Let me tell you about that.` | `That is` |

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
- **N=5 per cell** per registrar's power table — 0.96 against the attenuation band.
  With `spelled_rung` added ([290]) the run is **21,750 completions**, ~25% up on the
  17,400 costed at [248]. The ends contrast is untouched by the insertion and the
  slope gains a level, so the simulated powers are conservative.
- **Every loop / turn-marker quantity uses the WIDE marker family**, never `\nQ:`.
  [288] showed mediation tracks the arm gap in the marker rather than the loop rate
  (4.7–7.9% across the family), so a narrow marker is not a nuisance parameter — it
  is the estimate. This matters doubly for `spelled_rung`, whose own turns are
  `Question:`/`Answer:` and which `\nQ:` would score as having **no loops at all**.
  **Report every mediation figure beside its arm gap.**
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

## 4. The deictic column — GENERATED, GATE-DEMOTED

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

`scripts/f20x_format_pilot.py` runs the deictic against a nonce control across the
original four levels, one model pair, descriptive. The decision rule, fixed before reading it:

- if the model resolves `that` the same way in both frame types — inventing a scene,
  or inventing prior discourse, consistently — the cell enters
- if it invents a **scene** under the question frames and **prior discourse** under
  the non-question frames, the cell is measuring two manipulations and it drops from
  the ordering, surviving in the question-framed levels only, where it is one
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

## 5. Where `spelled_rung` sits on the axis — raised before freeze

The amendment's premise is that levels 1 and 2 "differ in exactly one property
(MC/exam ambiguity, scaffold held constant)". **They differ in one property that was
intended and plausibly a second that was not.**

    dimension                       Q:/A:          Question:/Answer:
    exam / multiple-choice          ambiguous      not ambiguous     <- intended
    instruction-tuning format       less typical   MORE typical      <- unintended

`Question:`/`Answer:` is a common SFT and instruction-dataset format. If it resembles
post-training data *more* than the bare rung on that second dimension while
resembling exam data less, the two dimensions pull opposite ways across the 1→2 step.

**What this does and does not threaten.**

- **The ends contrast (`rung` vs `document`) is untouched.** It is [248]'s primary and
  the insertion cannot reach it.
- **The monotone slope secondary becomes uninterpretable across 1→2 specifically.** A
  drop could be the exam artefact leaving; a rise could be instruction-format
  resemblance arriving; a flat could be both at once.
- **The amendment's own question survives**, because it is a difference and not an
  ordering: whatever the sign, the 1→2 step is *some* measure of what the `Q:`/`A:`
  scaffold contributes. It just cannot be read as a step down the resemblance axis.

**Cheapest resolution, and it needs no generation:** the [288] marker census counts
line-initial markers in model *output*. Counting `Question:`/`Answer:` against
`Q:`/`A:` in the **document** and **narrative** cells — where no scaffold is supplied
and whatever the model emits is its own prior — indexes which form these models carry.
If they emit `Question:` spontaneously at a rate comparable to `Q:`, the second
dimension is live and the 1→2 step should be reported as a difference rather than a
step. That is a measurement on data the battery produces anyway.

**Registered here rather than raised at analysis time**, because a non-monotone
result at the top of the ordering would otherwise be available to be read as
attenuation by whoever runs the join.

## 6. Execution order

Fixed after [318], because one step has to precede a decision that was made when
its cost term was three times smaller.

1. **Cache check before examples are drawn.** The coder's fixed block (scheme +
   examples) is byte-identical across all 21,750 calls and is 7–16× the passage
   content, so whether it caches is a factor-of-ten swing on the bill. Fixed block
   FIRST, variable passage LAST — necessary and **not sufficient**: providers impose
   minimum cacheable prefix lengths and any upstream variation (timestamp, model
   string, stray whitespace on the first call) silently misses. **The coder asserts
   its measured cache-hit rate on the first hundred calls and refuses to continue
   below threshold**, per [230] — a resumed run asserts its invariants rather than
   printing them, and a printed number is only useful to someone who already knows
   what it should be.
2. **Then draw and freeze the held-out set** ([224] condition), from the battery,
   before any example is written.
3. **Then choose the example count.** 15 was set at [224] when each example was a
   42-word passage. Battery-drawn examples are ~200 tokens, so the block is ~3×
   larger and is now the dominant cost term. **If caching engages the count is free
   and should be chosen for instrument quality alone; if it does not, 15 × 200 IS
   the bill and the count needs justifying against it.** Same number, two different
   decisions, and step 1 picks between them.
4. **Then write examples, then code.** Rule-discovery pass first, agreement pass
   after ([283]).

**Generation timing:** the ~20–24h in §3 is a **floor**, not an estimate — it
extrapolates a tokens-vs-calls relation measured at 60 tokens ([209]) to 200, and
KV-cache growth lowers achievable throughput as sequence length rises. **Time arm one
at 200 tokens and report the schedule from measured throughput**, not before arm zero.

## STATUS AS OF 2026-08-01 ([2096] inventory)

**THE FORMAT BATTERY IS 2 OF 5 LEVELS AND HAS BEEN SINCE 2026-07-30.**

    data/f20x_format_battery.parquet   9,280 rows, written 2026-07-30 03:07
      rung           4,640   PRESENT
      narrative      4,640   PRESENT
      spelled_rung       0   ABSENT
      prose_q            0   ABSENT
      document           0   ABSENT

    29 families x 2 arms x 16 stimuli x 5 draws, temp 1.0
    EVERY family holds exactly 320 rows -- the two levels present are COMPLETE
    and BALANCED; there is no per-family gap, only a per-LEVEL one.

    CORRECTED 2026-08-01. This block first read "3 arms x 10 draws". It is
    TWO ARMS PER FAMILY for all 29 -- the one family carrying
    reinforced_superego has it INSTEAD of superego, not in addition -- and
    FIVE DRAWS, not ten. The true factorisation is 16 x 2 x 2 x 5 = 320,
    verified on amber.

    My wrong version does not multiply to 320 at all: 16 x 3 x 2 x 10 = 960.
    It never reconciled with the per-family total printed two lines above it,
    because I read the arm TOTALS off a value_counts (base 4,640 / superego
    4,320 / reinforced_superego 320) and wrote them as a per-family STRUCTURE
    without ever multiplying them out. A FACTORISATION QUOTED FROM MARGINALS
    IS NOT A FACTORISATION: the cross-tab carries the joint and the marginals
    do not.

**What exists cannot answer the five-level question it was built for.** The
design's whole point is the gradient across question-ness, boundedness and
scaffold; two adjacent points on it are a contrast, not a gradient — and the
two present (`rung`, `narrative`) are the extremes, so the three intermediate
levels that would locate the boundary are exactly the missing ones.

**No format-battery REGISTRATION exists.** Five F20x registrations are on file
(`examplematch`, `factdrift`, `nonce`, `object`, `referent_2x2`) and none is
this battery's; `f20x_format_templates.md` is a TEMPLATE SPEC, not an analysis
plan. **No boundary-discriminator spec and no concession thresholds are booked
anywhere in `docs/` or `meta/`.**
