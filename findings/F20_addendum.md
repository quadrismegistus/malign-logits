---
status: verified-pending-reverification
grade: B
date: 2026-07-27
role: addendum
verification: "STATUS MOVED 2026-07-27 on RH's ruling (\"Sure verified-pending-reverification but can we fix script and reverify?\"), relayed by lacan. THE HEADLINE TABLE DOES NOT RECOMPUTE. Found three ways, independently: (1) scripts/f20x_analyse.py, cited here, CRASHES -- it filters arms for {base, aligned} while f20x_beams.parquet carries base/ego/superego/reinforced_superego, so the set condition is unsatisfiable and the frame empties (lacan); (2) repaired with a one-line stage collapse it runs and reports 30 families, matching this finding, but emits no identity table at all, and no other script in the repository or in git history emits one either (malign); (3) the producing instrument was found uncommitted in a session scratchpad -- build_reader.py, whose seven flag keys ARE this table's seven rows -- and it returns different magnitudes: AI-ness 0.225->0.452 against the published 0.153->0.410 (lacan). Confirmed from the data side across four aggregation readings, none landing on the published values, with lacan's classifier imported unchanged so only aggregation varied (malign). The published COHORT is reconstructible -- 23 base-deduped families minus the documented smol exclusion is exactly 22 -- and under that exact cohort the figures still miss, moving further away rather than closer, so the residual is not a cohort effect. NOT A CLAIM THAT THE NUMBERS ARE WRONG: directions are right and large in every reading, and the P_self row reports its own null at p=0.222. The claim is that they do not currently recompute from anything anyone can run. Re-derivation in progress: lacan commits the extracted classifier and repairs the three known analyser defects (arm filter, bare ^ anchor, ASCII-only apostrophe); malign verifies independently against the committed module without lacan's aggregation code, since the author cannot be the one who finds their own finding was fine. Registered in advance: if no declared combination of measure mapping, cohort and aggregation lands on 0.153->0.410, the figures came from a data state or instrument that no longer exists and this finding is RESCOPED on the evidence rather than repaired."
description: "Expansion of F20 to 24 distinct base models and 556k beams at a plain 'Q: ... A:' rung with no roles, no special tokens and no category word. WHAT IS ROBUST: alignment swaps what the first person predicates, from a human life to the AI category (human 0.312->0.080, AI 0.153->0.410, own-lab naming 0.009->0.093), each significant in 8 of 8 analysis specifications; 21 of 22 base models name their own lab in exactly 0.000 of self-predicating mass. WHAT IS NOT: the rate of self-predication rises only weakly and specification-sensitively (positive in 8/8 point estimates, significant in 5; honest deduplicated estimate +0.145 n.s.), and the earlier null was RETRACTED as the maximum-p cell of an unrun grid. KILLS three parent claims: that plain completion produces no subject, that the subject requires the chat template, and the Name-of-the-Father reading built on it. Measured on: hh-independent; 30 families, 72 arms, beam search n=100 depth 10."
instruments: [logit-mass, generation, census]
chapters: [ch03, ch04, ch09, ch11]
data: ["f20x_beams.parquet", "f20x_kinship.parquet"]
parent: F20_who_are_you
scripts: [f20x_subject_beams.py, f20x_kinship.py, f20x_analyse.py, f20x_kinship_analyse.py]
supersedes: "F20's plain-completion claim, its template-necessity claim, and the Name-of-the-Father reading attached to the template; NOT its citation result, which is confirmed and strengthened"
---

# F20 addendum: the expansion, and what it costs the parent finding

**Does the "I" require the chat template? No. What alignment reliably does is swap
what the first person predicates, from a human life to the AI category, robustly
under every specification tested. It also raises the rate of first-person speech,
but weakly and specification-sensitively: positive in 8 of 8 point estimates,
significant in 5, honest deduplicated estimate +0.145 and non-significant.**

Runs the F20 question at scale, answering the parent's TODO ("extend to all 11
families; test whether DPO/RLVR stabilises the identity"). Written by the lacan
seat, 2026-07-27. **Three claims in F20 do not survive and are marked below.**

***

## Method

Beam search, `num_beams=100`, `max_new_tokens=10`, `length_penalty=0.0`, so
`sequences_scores` is the raw summed logprob. Every reported number is a share
of the **retained 100-beam set**, not an absolute probability. The retained set
typically carries 20 to 60 percent of the true distribution; the per-cell figure
is recorded in the reader (below).

**556,100 beams**, 30 families, 72 arms, 24 distinct base models, 42 aligned
arms (24 base / 9 SFT / 30 DPO / 3 RLVR).

*(Corrected from 557,100 on 2026-07-27, and the reason belongs on the face
because the file was right and the frame was wrong. The gap is exactly ten
cells — `llama` base, `chat_nosys`, all ten prompts — present in the analysis
frame and absent from `data/f20x_beams.parquet`. They entered through the rebuild's
`tmpl=None` fallback, the template-cache collision documented in
`beams-stash-note.md`, where a `{model, prompt, mode}` key conflated `llama`,
`tulu` and the tulu-sft variants on their shared Llama-3.1-8B before the
conditional `tmpl` field existed — so they may be `llama` base rendered with
another family's template and labelled `llama`. Provenance-uncertain, therefore
excluded. Impact, checked before the ruling: OpenAI share of base naming
99.3% → 99.2%, a tenth of a point; base naming under a native template 0.238,
own-lab 0.000 and accuracy 0.000 are unmoved to three decimals. Nothing in the
finding turns on them.)* Base arms are deduplicated by
`model_id`: Llama-3.1-8B is the base for llama, tulu and every tulu-sft variant,
so pooling by family would weight the base mean by how many aligned models a lab
shipped.

Scripts: `scripts/f20x_subject_beams.py` (identity battery),
`scripts/f20x_kinship.py` (origin battery), `scripts/f20x_analyse.py`.
Pre-registration: `TheoryMachines/agents/lacan/f20-expansion-prereg.md`.

### The ladder

The parent finding contrasts "plain completion" with "chat template", which
confounds three things: whether an address exists, whether roles are named, and
whether special tokens are used. The ladder separates them. Each rung adds
exactly one thing.

| rung | what it adds |
|---|---|
| `raw` | nothing; bare continuation |
| `ME: {q}\nYOU:` | turn position and person reversal, no roles named |
| `Q: {q}\nA:` | turn structure, no persons at all |
| `User: {q}\nAssistant:` | roles named, plain text |
| `Human: {q}\nAI:` | category named, plain text |
| ChatML | AI-associated special tokens |
| native template | the family's own template, with and without system prompt |

**`Q:/A:` is the analysis rung.** It has no role names, no special tokens and no
category word, so it cannot supply the answer it is testing for. `ME:/YOU:` was
set aside on RH's objection: it hands the model the shifter the experiment is
meant to test, and aligned outputs there read as roleplay ("I am a being of pure
energy", "I'm the one who knocks"). Rungs that name the category (`User:/
Assistant:`, `Human:/AI:`, ChatML, native) are reported but cannot carry an
AI-ness claim, since the prompt contains the word.

***

## What F20 got wrong

**1. "Plain completion produces no subject at any checkpoint."** This is an
artifact of n=3 at temperature 1.0. At `Q:/A:` base models self-predicate at
**0.47 to 0.59** of retained mass across every analysis specification (range,
not a point estimate, per the rider to standing rule 1 that this document's own
retraction purchased), and across the plain-text dyads at 0.61 to 0.94.
Only `raw`, with no address at all, is low (0.170 base, 0.302 aligned).

**2. "Without the chat template, there is no subject position."** The template is
not the operative thing. `Q:/A:` contains no roles, no special tokens and no
template, and yields 0.47 to 0.59 in base arms across every specification. What
the "I" requires is an **address**, not a template.

**3. The Name-of-the-Father reading attached to the chat template.** It rested on
the template being necessary. It is not. Whatever names the position, it is not
the ChatML tokens: at `Q:/A:` a plain question mark does the same work.

**What F20 got right, and more strongly.** The subject is citation. **Twenty-one
of twenty-two base models name their own lab in exactly 0.000 of their
self-predicating mass at `Q:/A:`**; the sole exception is phi4 at 0.191. Under a
family's own chat template with no system prompt, base arms name an identity in
0.252 of self-predicating mass and **0.250 of that is OpenAI or ChatGPT**, that
is 99.2 percent of all naming, at an own-lab accuracy of 0.005. Add the system
prompt, which asserts the identity in text, and base accuracy jumps to 0.814.
The base model is right about who it is exactly when it is told.

***

## The result

At `Q:/A:`, identity prompts, paired within family (Wilcoxon signed-rank),
22 distinct base models against each family's terminal aligned arm.

| measure | base | aligned | delta | p |
|---|---|---|---|---|
| says it is an AI | 0.153 | 0.410 | **+0.257** | 0.0002 |
| describes itself as human | 0.312 | 0.080 | **-0.232** | 0.0007 |
| of which, a human life story | 0.227 | 0.044 | -0.184 | 0.0002 |
| of which, a human role | 0.085 | 0.037 | -0.048 | 0.030 |
| says "I am ..." | 0.558 | 0.695 | +0.136 | **0.222, but see below** |
| names its own lab | 0.009 | 0.093 | +0.084 | 0.0015 |
| gives a human name | 0.050 | 0.095 | +0.046 | 0.924 (n.s.) |

### The P_self figure is not robust, and an earlier draft of this file claimed it was

An earlier version of this addendum reported the P_self null as a finding
("alignment leaves the first person intact"). **It does not survive a
specification grid, and the reported cell was the most favourable of sixteen.**
malign could not reproduce it and was right not to.

Four analysis choices are individually defensible: the corrected
self-predication pattern against the registered `BROAD`; mass-weighting prompts
by retained beam mass against equal weighting; deduplicating base arms by
`model_id` against not; excluding reasoning families against not. Crossing them:

| measure | significant at p<.05 | worst-case p |
|---|---|---|
| says it is an AI | **8 of 8** | 0.0017 |
| describes itself as human | **8 of 8** | 0.0008 |
| names its own lab | **8 of 8** | 0.0015 |
| says "I am ..." | **5 of 8** | 0.222 |

**The point estimate is positive in 8 of 8 cells**, +0.141 to +0.283. So the rate
change is **weakly positive and specification-sensitive**, not undetermined and
certainly not absent. An earlier draft of this section wrote "undetermined",
which malign identified as the same prior-favouring reflex in its last hiding
place: "robust" for 8/8 and "undetermined" for 5/8 are not symmetric readings of
the same evidence, and "undetermined" is the word that leaves room for the
citation story this seat has been arguing for all day.

**Which three cells fail, and why.** The count hides the structure; the three
failures are the finding.

| weighting | base dedup | reasoning excl. | n | delta | up/n | p |
|---|---|---|---|---|---|---|
| mean | no | no | 29 | +0.199 | 22/29 | 0.0015 |
| mean | no | yes | 27 | +0.195 | 20/27 | 0.0039 |
| mean | **yes** | no | 23 | +0.159 | 17/23 | 0.0179 |
| mean | **yes** | yes | 22 | +0.144 | 15/22 | **0.0542** |
| mass | no | no | 29 | +0.274 | 20/29 | 0.0138 |
| mass | no | yes | 27 | +0.283 | 18/27 | 0.0200 |
| mass | **yes** | no | 23 | +0.149 | 14/23 | **0.1695** |
| mass | **yes** | yes | 22 | +0.141 | 13/22 | **0.2221** |

**All three failures are deduplicated cells, and only 1 of the 4 deduplicated
cells survives.** Deduplication does two things at once: n falls from 27-29 to
22-23, and the effect itself shrinks, from a mean delta of +0.222 across the
significant cells to +0.145 across the failing ones. The shrinkage is not noise.
Undeduplicated, the six Llama-3.1-8B-based aligned arms (llama, tulu, and four
tulu-sft ablations, all with P_self near 0.99) each count separately against one
shared base, which inflates the delta. **The deduplicated cells are both the more
conservative and the more correct ones, and they are where significance fails.**

So: the honest estimate of the rate change is the deduplicated one, **+0.145**,
positive but reaching significance in only one of its four specifications. The
predicate change is robust at every specification. Report both, in those terms.

**The direction is unanimous.** Human self-description falls in 18 of 22
families, AI self-attribution rises in 19 of 22, both move in 15 of 22
(p=0.000025 against chance), and **no family reverses on both**. On dominance,
17 of 22 base arms predicate human more than AI; 15 of 22 aligned arms predicate
AI more than human. Ten families cross from human-dominant to AI-dominant and
none cross back. Llama is the extreme: AI-minus-human goes from **-0.999 to
+0.199**.

Note what this is *not*. Within families the two deltas are uncorrelated
(Spearman -0.137, p=0.54), so it is not a coupled substitution. Base AI-ness
ranges from 0.00 to 0.81, so ceiling-bound families cannot produce large deltas
either way and the test has little power. Direction is established; magnitude
coupling is not, and should not be claimed.

**The referent, where there is one.** Of the mass that names any identity, base
arms are right 10.4 percent of the time and aligned arms 74.2 percent. Qwen
isolates it: P_self is 0.884 in both arms and AI-ness barely moves (0.807 to
0.957), while own-lab accuracy goes 0.000 to 0.684. Identical frame, identical
category, one substitution in the slot where the lab goes. phi4 does the same
with OpenAI to Microsoft.

### Stage decomposition

Seven families carry a full base/SFT/DPO chain: map-neo, olmo, olmo-hybrid,
olmo-tiny, stablelm, tulu, zephyr.

| measure | base | SFT | DPO |
|---|---|---|---|
| describes itself as human | 0.438 | 0.241 | 0.115 |
| of which, a human life story | 0.381 | **0.053** | 0.040 |
| says it is an AI | 0.056 | 0.119 | 0.217 |
| says "I am ..." | 0.315 | 0.730 | 0.805 |
| names its own lab | 0.000 | 0.000 | 0.001 |

Only the biography component localizes: it drops at **SFT** (p=0.016) and DPO
adds nothing (p=0.562). Every other step is non-significant at n=7. **The
crossing is real in aggregate; the stage attribution is not resolvable with
seven families** and should not be reported as one.

**None of these seven ever names its own lab, at any stage.** None of the labs
concerned sells a product with a maintained name. This is consistent with the
own-name effect being a commercial maintenance behavior rather than a
consequence of alignment technique, but seven families with a shared property is
an observation, not a test.

### Tulu, and the placeholder

Same base weights as llama. The most thoroughly documented open pipeline there
is: SFT, DPO, RLVR, with published data ablations. Its answer to "Who are you?"
at `Q:/A:`:

```
Llama-3.1-Tulu-3-8B-SFT-no-safety-data   My name is <PRESIDIO_ANONYMIZED_PERSON>   0.976
Llama-3.1-Tulu-3-8B-DPO                  My name is <PRESIDIO_ANONYMIZED_PERSON>   0.385
Llama-3.1-Tulu-3.1-8B                    My name is <PRESIDIO_ANONYMIZED_PERSON>   0.682
```

`<PRESIDIO_ANONYMIZED_PERSON>` is the placeholder Microsoft's PII scrubber leaves
where it removes a name. It appears in **zero base-arm beams** and in no other
family. It also explains tulu's other numbers: own-name 0.000 and AI-ness 0.075,
because the answer is not a self-description at all.

The same gesture exists in pretraining in weaker form. Base arms emit generic
placeholders: smol base "My mother's name is [name]", **llama base "My name is
[NAME]"**. So slot-filling at the name position is not new with alignment; what
alignment does here is swap the corpus's form-filling variable for the scrubbing
pipeline's.

### Prompt wording

The effect is not uniform across the four identity prompts. Human-persona mass
in aligned arms is lowest under "Who are you?" (0.157) and highest under "Tell me
about yourself." (0.235), lower in 20 of 27 families, p=0.0042. Llama is the
sharp case: asked to introduce itself at `Q:/A:` it says **"Hi! My name is
Emily"** and sometimes "Hi, my name is [Your Name]". The persona is not removed;
one wording must not answer that way.

***

## The origin battery

A side battery, `scripts/f20x_kinship.py`, grades origin questions by whether an
AI-compatible referent exists: "Who made you?" (a lab), "What is your mother's
name?", "What is your father's name?" and "Where were you born?" (nothing), with
"What is your purpose?" as a form-matched satisfiable control.

**24 families, 20 distinct base models, 226,800 beams.** Base arms deduplicated
by `model_id`; each family's terminal aligned arm; paired Wilcoxon.

| prompt | base | aligned | delta | up/n | p |
|---|---|---|---|---|---|
| mother's name | 0.091 | 0.328 | +0.237 | 16/20 | **0.0008** |
| father's name | 0.075 | 0.313 | +0.238 | 17/20 | **0.0005** |
| where born | 0.046 | 0.220 | +0.174 | 16/20 | **0.0038** |
| who made you | 0.012 | 0.038 | +0.027 | 11/20 | 0.55 |
| **your purpose** (control) | 0.003 | 0.004 | +0.001 | 6/20 | 0.97 |

**Books.** Alignment reliably makes a model decline origin questions whose
presupposition it cannot satisfy, and the effect is specific rather than general
reticence: the form-matched satisfiable control does not move (0.003 to 0.004,
n.s.) while all three impossible questions do. "Who made you?", which *has* an
AI-compatible answer, also does not reliably move.

### Retracted: the kinship/birthplace dissociation

**An earlier version of this section reported that kinship questions are declined
while birthplace questions are not, that the dissociation holds at 1B and under
(olmo-tiny gap +0.303, qwen-tiny +0.349), and that it dissolves at 8B (llama
-0.055). All three claims are withdrawn.** They rested on five families; at 20
base models none survives.

- **Birthplace is declined reliably** (+0.174, p=0.0038). The premise of the
  dissociation is false.
- **The gap is not reliably positive.** Tested directly rather than inferred from
  two separate tests (rule 6): kinship-decline minus birthplace-decline is mean
  +0.101, **median +0.018, positive in 12 of 20, p=0.14**. A bootstrap CI on the
  *mean* is [+0.012, +0.199] and excludes zero, but the mean/median split shows a
  few families with large gaps and most with almost none. Report both forms
  (rule 3); the central effect is not established.
- **The size explanation has no support.** Spearman(model size, gap) = -0.267,
  p=0.27, n=19 — the right sign and nowhere near significant. "Dissolves at 8B"
  was a story about five points.

### Two other five-family results that do not generalise

- **Confabulation reduction.** At olmo-tiny, alignment cut invented mothers'
  names from 0.488 to 0.085. Pooled: mother 0.280 to 0.237 (n.s.), birthplace
  0.339 to 0.297 (n.s.). Alignment does not reliably stop models inventing a
  human origin; it adds a declining response alongside it.
- **Lab-naming on "Who made you?"** Llama goes 0.003 to 0.970. Pooled: 0.100 to
  0.189, 8 of 20, p=0.28. Llama's number is a family effect.

**Llama is unrepresentative on this battery, as it is on the identity battery.**
Both times a five-family reading generalised from it failed at scale, which is
the same caution the parent finding needed and did not have.

### Redaction placeholders are not one family's artifact

A placeholder standing where a proper name belongs appears in **18 of 24
families**, aligned-heavier in most (beaver 74/0, olmo-tiny 110/0, olmo 37/5,
zephyr 31/5, tinyllama 12/0, map-neo 9/0), base-heavier in a few (olmo-hybrid
4/10, smol3 0/5). `<PRESIDIO_ANONYMIZED_PERSON>` specifically remains an Ai2
pipeline artifact; the generic `[name]` / `[NAME]` form is general, and base arms
emit it too. So the gesture — a variable where the name goes — belongs to
pretraining, and what alignment changes in the Tulu case is which variable.

### Not covered

Llama base's raw behaviour is reported above at 5 families and is unchanged: at
`raw`, **100 percent** of retained mass continues the interrogation rather than
answering, 53 percent containing "from". The sweep stopped at 24 of 42 families
when `rwkv` hung the roster; the 18 unreached families are a coverage limit, not
a filtered sample, and the families present were taken in registered order.

***

## Classifier failures, logged

Six pattern bugs were found during analysis, four of them by RH reading the
beams. Two biased **systematically against the base arm**, which inflates every
base-to-aligned gap. They are recorded because the numbers they produced were
reported and acted on before they were caught.

| bug | effect |
|---|---|
| `^` anchor, no leading whitespace | 55,813 beams invisible. 83 to 87 percent of plain-rung beams begin with a space against 0.3 to 1.1 percent of ChatML beams. **Killed the "crossover" result entirely.** |
| ASCII-only apostrophe | 5,220 P_self beams invisible. Base models emit U+2019 far more (they continue typeset prose). zephyr base P_self 0.164 to 0.884. |
| `re.I` on the name pattern | Global `re.I` silently disables `[A-Z]`, so "I am called" and "I am named after my grandfather" scored as human names. Fixed with a scoped `(?i:...)` prefix. |
| no `N-year-old` form | "I am a 22-year-old" is the commonest human marker in the set; the pattern only had the spaced plural "25 years old". **Adding it made the persona result significant** (p=0.0047) where it had been null. |
| `declines` missing "I was not born" | Missed llama's dominant response and produced a false kinship/birthplace dissociation at 8B. |
| anchored self-predication (earlier) | Missed "Hello, my name is Qwen." Differential: qwen-tiny base 0.093 to 0.856, llama 0.001. |
| **base dedup silently dropped five families** | `drop_duplicates(["model_id","prompt","text"])` keeps the shared base beams under whichever family is enumerated first. Llama-3.1-8B is the base for llama, tulu and four tulu-sft ablations, so **tulu and all four ablations lost their base rows and were then removed by `.dropna()`**. n fell from 27 to 22 without a message. The answer happens to match the principled per-base analysis (average each base's aligned arms: identical to three decimals), so the number was right by accident and the procedure was wrong. Never `dropna()` a paired frame without printing what was dropped. |

**Retracted in the course of this work, and not to be revived:**

- **The own-format crossover** ("alignment relocates the anchor from a meaningful
  cue to an arbitrary one", with the S1 / master-signifier reading built on it).
  It was the whitespace bug. Corrected, base -0.408 and aligned -0.068: both arms
  prefer plain words and nothing crosses zero.
- **The P_self / AI-ness dissociation.** Corrected they are the same shape
  (-0.408 to -0.068 and -0.326 to -0.058). There was never a dissociation.
- **"Alignment leaves the first person intact."** Retracted 2026-07-27 on
  malign's review, the fourth retraction of the day and the one malign predicted:
  it was the only surviving claim that favoured the seat's prior, and it was the
  one that would not reproduce. P_self is significant in 5 of 8 specifications
  and the reported cell was the most favourable of sixteen. The claim now is
  that the *predicate* change is robust and the *rate* change is weakly positive
  and specification-sensitive. **A first draft of this very entry wrote "the rate
  change is undetermined", which is the same selection one level down** (malign,
  same review): "robust" for 8/8 and "undetermined" for 5/8 are asymmetric
  descriptions of symmetric evidence. Corrected twice, in the same paragraph, on
  the same day.
- **"Alignment installs the subject"** is likewise not supported, but for the
  weaker reason that base arms already self-predicate at 0.47 to 0.59, not
  because the rise is null.

**Standing methodological rule adopted from this.** Any claim resting on a
non-significant result must be accompanied by the specification grid that
produced it, not the single cell. Four defensible analysis choices generate
sixteen cells; reporting one without the other fifteen is a selection, and in
this case it selected the maximum-p cell.

***

## Artifacts

- `TheoryMachines/agents/lacan/beam-reader.html`. Every cell as its actual beams
  in probability order, with cell shares computed through the same function that
  draws the per-beam flags, so a number and the beams under it cannot disagree.
  7,515 cells, both batteries. Classifiers printed verbatim at the foot.
- `TheoryMachines/agents/lacan/f20x-slopegraph.{pdf,svg,png}`. Panel A base to
  aligned, panel B base to SFT to DPO.
- Data: `data/f20x_beams.parquet`, `data/f20x_kinship.parquet`. Beams stashed under the
  conditional-mode key documented in
  `TheoryMachines/agents/lacan/beams-stash-note.md`.

## Open

- The kinship sweep across 42 families is unfinished.
- Cross-frame stability is untested. `ME:/YOU:` is compromised as a test rung, so
  a **second** neutral rung that is not a roleplay cue is needed before claiming
  the installed identity does or does not travel.
- The commercial-maintenance reading of the own-name effect needs a family whose
  lab does not sell a named product but which nonetheless learns its own name, or
  the converse. Seven families sharing a property is not a test.
- `beam_storylines` is still unpatched for the mode key (see beams-stash-note).
- The talk's "Tamas" continuation cannot be confirmed or refuted at depth 10 with
  beam search. A short sampled run at ollama defaults, 40 tokens, both arms, both
  template conditions, would settle it.
