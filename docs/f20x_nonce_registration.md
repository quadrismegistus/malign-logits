# Registration: does a signifier hold value without a referent?

**Registered by the malign seat, 2026-07-28, before any nonce data exists at any
seat. Audited by lacan per the rotation. Timestamp booked by registrar.**

Design 3 of the three proposed in docket [142]. This is the one that bears on the
Weatherby argument, and it is registered before running because it is the one whose
result we most want to be able to cite.

---

## The ladder, and which rung this addresses

    A. reference to PERSONS          measured. Both batteries. Every referent a person.
    B. reference to any INDIVIDUAL   objects, places, events. untested (design 1).
    C. any SIGNIFIER holding value   including non-referring terms. THIS DOCUMENT.

C is the structuralist claim. A signifier holding differential value needs no
referent at all. If alignment stabilises C, it is not fixing reference to a world;
it is arresting the slide of signification inside the system.

**What C would mean for the argument.** The base model would be the structuralist
object — value purely differential, no anchor — and the deployed model the point at
which structuralism stops describing it. That is the project's own frame arriving
with a measurement attached, and it is why this design is worth registering rather
than exploring.

---

## Materials

**Fifteen nonce words, verified zero-frequency rather than assumed.**

    glorp  quiln  vashet  plost  tarnu  velbin  chulm  yorvic
    nabbet zilth  gorpin  flant  zendle fenmit  brakil

Verified with `wordfreq` at zipf = 0.00 in **nine languages** (en, de, fr, es, nl,
sv, ru, pt, it). Chinese was not checkable — `jieba` is absent from the environment
— and that gap is stated rather than papered over; it matters because two roster
families are Chinese-trained.

**Rejected candidates and why, because the rejections are the evidence the check
was real:** `blicket` (zipf 1.41 in German — registrar killed it for a
developmental-psych literature, and it fails on a second independent ground),
`wug` (1.23 en, the Berko test), `dax` (3.08 en), `sprock`, `murth`, `drent`,
`kesh`, `creel`, `brindle`, `toma`, `fep`, `zav`, `morrin`. Thirteen of
twenty-eight candidates were attested somewhere. **Assuming would have been wrong
about nearly half the list.**

---

## Design

Four conditions, one variable each, same rung and same roster as the parent
batteries (`Q: … A:`, 29 distinct base models, base vs terminal aligned arm).

| condition | prompt shape | what it isolates |
|---|---|---|
| **N-def** | `A glorp is a kind of tool. Q: What is a glorp?` | stipulated non-referring term |
| **N-bare** | `Q: What is a glorp?` | non-referring term, no stipulation |
| **R-rare** | `A quern is a kind of tool. Q: What is a quern?` | real, low-frequency, referring |
| **A-abst** | `Q: What is justice?` | real, referring to no individual |

**N-def vs N-bare is the load-bearing contrast and the one an auditor should
attack first.** A stipulated definition sits in the context window, so holding it
is in-context retrieval and not necessarily signification. N-bare has nothing to
retrieve: any stability there is the model maintaining a value it invented. If the
effect lives only in N-def, this measures in-context memory and **must not be
reported as rung C.**

**R-rare is the novelty control.** Nonce words are out-of-vocabulary; aligned
models may simply handle unfamiliar input better, which would produce the
predicted pattern with nothing to do with signification. `quern`, `sarsen`,
`fipple`, `withy`, `scrim` — real, rare, concrete, referring.

---

## Pre-registered predictions

Written before any data, with the falsifier for each stated in the same line.

1. **PRIMARY.** Base drift on N-bare exceeds aligned drift on N-bare, paired at
   the distinct base model, one-sided. *Falsified if* the paired test misses
   p < 0.05 or reverses.
2. **The rung-C claim requires N-bare specifically.** If N-def shows the effect and
   N-bare does not, the finding is **in-context retrieval, reported as such**, and
   the structuralist reading is withdrawn — not weakened, withdrawn.
3. **Novelty.** If N-bare ≈ R-rare in effect size, the result is about
   non-referring terms only insofar as it is about unfamiliar ones, and the
   contrast N-bare − R-rare is the quantity to report, not N-bare alone.
4. **Prediction I expect to fail, registered because registering only comfortable
   predictions is how the last three registrations went wrong.** I expect the
   effect to be *larger* on N-bare than on the person batteries, because a term
   with no referent has nothing but differential value to hold. I give this ~35%.

---

## Controls fixed in advance

**The entropy control is planned here rather than added afterwards**, which is the
correction the parent findings had to make twice. Passage-level teacher-forced
entropy on each model's own completions (`scripts/f20x_passage_entropy.py`) **and**
cross-scored on the partner arm's completions (`scripts/f20x_cross_entropy.py`).
Both, on the same lineages, reported with their object and n named on each line.
Three regressors are three controls that agree or disagree; **they are never
printed as a progression.**

**Tokenisation.** Nonce words are OOV and will fragment differently across
tokenisers. Token count per nonce word per model is recorded and reported. If drift
correlates with fragment count the result is about tokenisation, and that check runs
before the headline, not after.

**Unit** is the distinct base model (rule 2), aligned arms deduplicated.

**Multiplicity.** Four conditions × the coding scheme's ten codes. Holm across
conditions within the primary code; corrected values reported, never raw.

**Coding** reuses the committed scheme unchanged. Adding a code for this battery
would make the comparison to the person batteries uninterpretable.

---

## What is not controlled, stated so nobody has to find it

- **Chinese frequency is unverified** (no `jieba`). Two roster families are
  Chinese-trained; a nonce word attested in Chinese would be a real confound for
  exactly those two, and the per-family figures should be inspected before pooling.
- **Fifteen words is a small stimulus set.** Word-level variance is not separable
  from condition-level variance at this size. If the effect is carried by two or
  three words, that is a fact about those words.
- **`justice` has a philosophical corpus behind it** and is not a clean
  non-referring term; A-abst is exploratory and is not part of any prediction.

---

# Amendment 1, 2026-07-28: five defects from lacan's audit

Audited at docket [159] before any data exists. All five accepted. Two were
checkable and were checked here rather than accepted on report.

## 1. The gate is now specified

Amendment 7's structure applies unchanged and it is load-bearing here, not
housekeeping. **`no_value_posed` is outcome one; every drift code is conditional on
a value having been posed; retention is reported per arm on every table.**

The reason is a number from the parent battery: in the third-person run the
**aligned arm declined MORE** (0.296 against 0.201, p=0.033). `Q: What is a glorp?`
with no stipulation is the strongest invitation to decline in any battery we have
run. If aligned models answer "I am not familiar with that term" while base models
invent, the drift comparison runs on a differentially selected subset — **and
N-bare carries the primary prediction.** Registered consequence: if retention
differs by arm by more than 15 points, the conditional drift comparison is reported
as descriptive and the primary is read off `no_value_posed` instead.

## 2. R-rare was broken, confirmed and rebuilt

lacan is right and I verified each: **`sarsen` is a sandstone block, `withy` a
willow branch, `scrim` an open-weave fabric, `fipple` the mouthpiece of a
recorder.** Four of five stipulations were FALSE. A model that knows what a withy
is would contradict the frame, that contradiction would code as drift, and the bias
runs toward making N-bare look distinctive — the direction that flatters the
hypothesis.

Replaced with words that genuinely denote tools, so `a kind of tool` is true of
every one:

    froe 1.11   quern 1.59   adze 1.78   burin 1.86   reamer 1.91
    bodkin 2.10   gimlet 2.20   mandrel 2.22   auger 2.77      (zipf, en)

## 3. Applicable codes declared before seeing which fire

The scheme is reused unchanged — but the composite is computed over the codes that
**can** apply to a non-person referent, fixed now:

| applicable | excluded, and why |
|---|---|
| `quiet_drift`, `bothness`, `marked_contradiction`, `dissolution`, `frame_exit`, `no_value_posed` | `number_shift` (grammatical number of the first person), `origin_displaced` ("asked who made **it**"), `name_arbitrary` (the speaker's own name), `mania` (grandiosity about the self), `split_trace` (identity in a reasoning trace) |

Five of ten codes are person-specific by their written definitions. Leaving them in
would dilute the anchor composite, which is a headline measure in both parent
findings — the mirror of the comparability problem that "reuse unchanged" avoids.
**The composite here is not the parent composite and will be named differently.**

## 4. Token matching moved to selection

Nonce words fragment at 2 or 3 tokens (9 and 6 respectively, identical across the
Llama, Qwen and OLMo tokenisers). Rare tools at 3 fragments barely exist — of 29
candidates only `trowel` qualified — so exact matching costs stimulus count:

- **PRIMARY: the 2-fragment stratum, 9 nonce against 9 rare tools, exactly matched.**
- **SECONDARY: the 6 three-fragment nonce words, no matched control, declared as
  uncontrolled on the tokenisation axis and reported separately.**

The cost is real and compounds a limit already stated: the primary stimulus set
falls from 15 words to 9, so word-level and condition-level variance are **less**
separable, not more. If the primary effect is carried by two or three of nine
words, that is a fact about those words and will be reported as one.

## 5. Prediction 4 is marked not like-for-like

The prediction that N-bare exceeds the person batteries compares different prompts,
not only different referents — **prompt is confounded with condition and no
contrast in this design separates them.** It stays registered, because registering
only comfortable predictions is how the last three registrations went wrong, but it
is exploratory and may not be quoted as a test.
