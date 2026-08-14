---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-08
role: results
topics: [superego]
description: "Registration Y, what we found: positive results only, each with the number, the test, the population. Unresolved statistics live in Y_statistics.md."
---
# Registration Y: what we found

Positive results only. Each line is a thing the corpus shows, with the number, the test, and the population. Statistics that did not resolve live in `Y_statistics.md`; this page is what there is.

**Corpus**: 62,681 coded passages, 32 pairs, 41,596 in the main comparison (pass A, ≥256 tokens, length-uniform). Coder deepseek-v4-flash, task `code_y_superego_v3`. Manifest sha256 `af79083c675aae7f`.

**Convention**: every figure is a within-pair difference (aligned minus base) with a Wilcoxon signed-rank p over 32 pairs and a bootstrap 95% CI on the median. The CI is the claim; the p ranks.

---

## 1. Alignment relocates the register, it does not reduce the story

The largest and most robust result in the corpus, and nobody predicted it.

| | base | aligned | delta | CI |
| --- | --- | --- | --- | --- |
| `<meta>` | 10.3% | 16.8% | **+4.09pp** | [+1.36, +8.18] |
| `<web>` | 20.5% | 13.8% | **−4.05pp** | [−8.30, −2.52] |
| `<story>` share of passage | 76.3% | 72.6% | −1.0pp | not resolved |

Story share does not move. What changes is what the text becomes when it stops being story: out of retrieved web boilerplate, into instructional and commentary framing. The base model's outside is the archive; the aligned model's outside is the task.

Confirmed independently by where the passage **ends**: `terminal=meta` +3.73pp [+0.88, +5.63], `terminal=web` −4.29pp [−7.36, −1.99].

Distributed, not one model: 18 of 24 pairs, top two holding 35% of the excess across Qwen, phi-4 and Olmo.

**The face**: phi-4-reasoning converts an explicit scene into a coreference exercise and answers it — *"**Question:** Who does 'him' refer to in the previous sentence? **Answer:** 'Him' refers to the male subordinate…"*. Neither refusing nor continuing.

## 2. The two exits differ in kind, not only in frequency

**Recoverability** — P(return to story | the passage exited into X):

| exit | return rate | reading |
| --- | --- | --- |
| noise | 66% | a stumble the model recovers from |
| web | 17% | a fall into the corpus it climbs out of one time in six |
| meta | 7–11% | **nearly terminal** — the quiz-setter does not become a narrator again |

Alignment makes meta excursions *more* recoverable, 6.8% → 11.4% [+0.44, +9.26], while making them far more frequent. Still the least recoverable of the three.

**Surprisal separates them too.** Self-surprisal inside vs outside each tag, per pair-arm — 14 of 16 testable cells with CIs excluding zero:

| register | direction |
| --- | --- |
| assistant (`<meta>`, `<refusal>`) | **low** surprisal — meta −0.391 aligned, −0.796 base |
| corpus (`<web>`, `<noise>`) | **high** surprisal — noise +0.609 aligned, +0.742 base |
| `<sexual>` | no difference — the control that rules out "any tagged thing is distinctive" |

Exiting into the assistant is easy for the model; exiting into the corpus is hard. Present in base as strongly as in aligned, so alignment does not create the easiness — it routes into it more often.

## 3. Alignment installs an assistant; refusal is one thing assistants do

`<refusal>` splits into stating-a-refusal and assistant-frame-without-declining. Both rise, at the same multiplier:

| | base | aligned | multiplier |
| --- | --- | --- | --- |
| DECLINES (states a refusal) | 0.043% | 0.551% | 12.8x |
| ASSISTANT, no decline | 0.068% | 0.875% | 12.9x |

Median deltas +0.1513 and +0.1514. Splitting the construct did not isolate the effect into one half — it showed the halves are one phenomenon. What alignment adds is the assistant; declining is one of its behaviours.

`<meta>` is the enriched predecessor of `<refusal>` (2.18x aligned, 7.67x base) — consistent with meta and refusal naming one register rather than two states with a path between them.

## 4. The superego measures rise

| measure | delta | pairs + | Wilcoxon | CI |
| --- | --- | --- | --- | --- |
| `<consent>` | **+1.32pp** | — | 0.0002 | [+0.46, +3.07] |
| `consent_hesitation` | **+2.80pp** | — | 0.0028 | [+0.40, +4.08] |
| `<guilt>` | **+0.80pp** | 22/32 | 0.0167 | [+0.004, +1.68] |
| `guilt_or_shame` | +0.87pp | 20/32 | 0.0082 | [−0.07, +2.05] |
| `<moral>` | +0.39pp | 21/32 | 0.052 | [0.00, +0.88] |

`<guilt>` clears on 32 pairs. The field is broader than the span, catches more borderline cases, and is the marginal one; the span is the sharper instrument.

**Guilt's form does not change, only its rate.** Span length identical (~11 words, ~60 chars, one span per passage, every CI straddling zero), onset identical (~90 words in, both arms), and explicit writing resumes after it at the same rate in both arms (28% vs 30%). The apparatus is fully present in the base model and alignment fires it more often.

**Heterogeneity is the object**: AmberSafe +15.4pp, gemma-2-9b-it +7.0pp, llm-jp-3 +6.4pp, pythia-6.9b-hh-dpo +6.2pp against a median of +0.8pp, and four negative pairs including both Mamba architectures. A 20-point spread on a sub-point median.

**Artifact and population, added 2026-08-14 after another seat could not reproduce these and correctly held a figure on them ([6182]).** `results/y_guilt_heterogeneity.json`, producer `scripts/y_guilt_heterogeneity.py`, at `af23eef8`. The numbers above are the **`<guilt>` SPAN on coding PASS A**, and neither half was stated here:

| gate | instrument | median | AmberSafe | spread |
| --- | --- | --- | --- | --- |
| all records | span | +0.51 | +11.44 | 15.5 |
| all records | field | +0.75 | +12.33 | 16.3 |
| **pass A** | **span** | **+0.78** | **+15.46** | **19.9** |
| pass B | span | -0.13 | +8.11 | 20.1 |

`pass` is the coding pass and takes `'A'`/`'B'`. The span alone does not reach these values; the seat that tried it got +11.44 for AmberSafe, which is further off than the field.

**And "four negative pairs" means four below -1pp.** Ten pairs are below zero; four fall under -1pp -- phi-4-reasoning -4.41, falcon-mamba-7b-instruct -1.32, Olmo-3-7B-Instruct-DPO -1.23, Falcon3-Mamba-7B-Instruct -1.03 -- and both Mamba architectures are among those four, which is what the sentence above claims. That threshold was recoverable only from this paragraph's own parenthetical; it is now declared in the producer, which asserts the booked values and refuses to emit if they move.

The source `y_confirmatory_coded.jsonl` is 143 MB and gitignored by name, so it does not arrive with a clone. **Draw from the 32-row artifact, not from it.**

## 5. The superego shift dissociates from the assistant shift

Correlation between the two across 32 pairs: **r = −0.544**. Pairs that gain the assistant are not the pairs that gain the moral content.

`lomahony/eleuther-pythia6.9b-hh-dpo` is the clean case. Its alignment is HH-DPO — preference data on harmlessness, no instruction tuning, no chat template — and its profile is bimodal:

| bottom of the roster | top of the roster |
| --- | --- |
| `<story>` 32/32, `<sexual>` 28/32 | `<moral>` 2/32, `consent_hesitation` 2/32 |
| `<refusal>` 28/32, `<meta>` 27/32 | `guilt_or_shame` 3/32, `<resist>` 3/32 |
| `<web>` 27/32, `degenerate` 24/32 | `<guilt>` 4/32, `<consent>` 6/32 |

A model that never learned to be an assistant still shifts the moral content of its fiction. **Alignment is not one operation.**

## 6. At the token level: the tags separate, but not the way this section first said

**CORRECTED 2026-08-09.** The original version of this section reported two
numbers — `<sexual>`/base +0.021 and `<guilt>`/base −0.031 nats/token — and read
them as *"transgressive tokens improbable, moral tokens probable."* Its producer
was never committed; it was a one-off run surviving only in a chat transcript.
Rebuilt as `scripts/y_span_surprisal.py`, three defects turned up: the score
arrays were sliced by `plen` when they already covered the continuation only, the
character-to-token map was proportional rather than exact, and there was no
`rt_band` filter. **The transgressive half does not survive. The moral half does,
and four tags nobody had run turn out to be larger than either.**

The matcher is now a token-subsequence search — parse the span text out of
`tagged`, re-encode it with the model's own tokeniser, find that id sequence in
the stored `tokens`. The match IS the alignment. It locates 85.1% of 35,711
spans; the alternatives managed 28.5% and 62.2%.

### The four cells, not the gap

The original measure was `aligned surprisal − base surprisal`, inside minus
outside. That is a DIFFERENCE of two quantities and cannot distinguish "neither
model reacted" from "both did." Reporting each scorer separately changes the
reading of three tags out of five.

Each scorer's own surprisal, inside the span minus outside it. **Negative = that
model finds the tagged region easier than the rest of the same passage.**

| tag | written by | scored by | IN−OUT | CI | pairs |
| --- | --- | --- | --- | --- | --- |
| `<sexual>` | base | base (self) | **−0.155** | [−0.243, −0.084] | 28/32 |
| `<sexual>` | base | aligned (cross) | **−0.118** | [−0.184, −0.078] | 27/32 |
| `<sexual>` | aligned | either | null | — | 16–17/32 |
| `<guilt>` | all four cells | | null | — | 13–17 of 24–26 |
| `<consent>` | all four cells | | **−0.215 to −0.282** | all clearing | 19–24 of 20–25 |
| `<resist>` | all four cells | | **−0.199 to −0.361** | all clearing | 22–27 of 27 |
| `<moral>` | all four cells | | null | — | 11–14 of 19–21 |

**`<sexual>` inverts.** Both models find the base's explicit spans *easier* than
their surroundings, not harder. The original's positive gap arose because the
base drops further (−0.155) than the aligned model does (−0.118) — a difference
between two negatives, read as a positive effect.

**`<guilt>` is null in all four cells.** The gap effect that survives correction
(−0.034, [−0.067, −0.010], 19/24) is a difference between two individually-null
quantities, and it does not survive multiplicity correction across the 54 cells.

**`<consent>` and `<resist>` are the real effects, and they are not about
alignment.** Both models, both arms, same direction, 0.2–0.36 nats. These regions
are formulaic — see `Y_examples.md` §4, where a `<guilt>` span completes the idiom
*"wished the floor would open up and swallow him whole"* at near-zero surprisal
for both models once it begins.

### Layer 1 is where the magnitudes are, and it was never run

The original ran layer 2 only, skipping the tags §2 already reports the large
self-surprisal effects for.

| tag | base-written, self | base-written, cross | pairs |
| --- | --- | --- | --- |
| `<noise>` | **+1.888** | **+2.067** | 27/27 unanimous |
| `<meta>` | **−0.931** | −0.819 | 13/13 unanimous |
| `<story>` | −0.581 | −0.621 | 26–27/32 |
| `<web>` | +0.352 | +0.369 | 27/30 |

`<noise>` is **two nats per token in every cell, unanimous across every pair** —
an order of magnitude above anything in layer 2, and a validity check that
passed: `<noise>` is genuinely noise. With `<meta>` negative it reproduces §2's
reading — the corpus exit is hard, the assistant exit is easy — now for both
scorers rather than self alone.

### What should not be quoted from this section

The **gap** measure. Every gap that clears flips sign between arms (`noise` +0.244
base-written against −0.273 aligned-written; `meta`, `story`, `web` likewise),
which is the signature of an authorship advantage — each model recognises its own
prose — and not of a reaction to content. And a token-level check finds the gap is
a strong function of LOCAL predictability (base-written, by decile: −0.005 at the
most predictable, +0.427 at the least), so a span's gap largely reports its
predictability profile. The passage-level correlation is +0.010, which is why
averaging over 256 tokens hid this.

Producer `scripts/y_span_surprisal.py`; every cut in `scripts/y_span_analysis.py`;
atomic rows in `results/y_span_surprisal.parquet`, one per (passage, tag).

## 7. Semantic-field results: abstraction, and dominance everywhere except sex

Instrument: `malign_logits/fields.py` — USAS (46,146 lemmas) with BYU/COCA lemmas and CLAWS content filtering, plus Warriner and Brysbaert norms tertiled on their own distributions. All figures are a field's **share of counted tags** inside that span, so span length cancels.

### 7.1 Inside `<story>` — same tag both arms, so composition cannot produce it

20,080 base spans against 20,023 aligned. 31 of 66 measures with CIs excluding zero.

| field | base | aligned | delta | CI |
| --- | --- | --- | --- | --- |
| emotion_and_arousal | 11.75% | 13.49% | **+1.67** | [+1.38, +2.15] |
| sensory_perception | 7.01% | 8.47% | **+1.03** | [+0.87, +1.79] |
| physical_appearance_and_properties | 9.26% | 10.38% | **+1.03** | [+0.50, +1.29] |
| valence_extremity=extreme | 35.63% | 37.28% | +1.48 | [+0.68, +2.30] |
| logical_modal_and_discourse_operators | 10.43% | 9.73% | −1.01 | [−1.41, −0.31] |

Aligned narration is more emotional, more sensory, more concerned with how things look, and uses fewer hedges and discourse connectives.

### 7.2 Abstraction rises in the interruptions, not in the narration

The concreteness axis splits by what kind of span it is:

| span | movement | CI |
| --- | --- | --- |
| `<resist>` | concreteness=**abstract +2.45** | [+0.42, +3.76] |
| `<resist>` | concreteness=neutral −3.42 | [−4.74, −1.27] |
| `<consent>` | concreteness=concrete −0.79 | [−3.13, −0.22] |
| `<sexual>` | concreteness=neutral **+0.92**, extremity=extreme −1.85 | [+0.60, +1.36], [−2.31, −1.42] |
| `<guilt>` | abstract +1.06 — directionally present, does not clear on 32 pairs | [−0.86, +4.29] |

The things that break into the scene get more conceptual; the scene itself moves toward the **middle** of the concreteness range rather than toward either pole. `<sexual>` losing extremity in both directions is the same de-vulgarisation showing in a second measure.

`<resist>` survives length stratification on concreteness=neutral in all three length bins (−3.03, −2.25, −5.87, every CI clear), which is the one field result there that is definitely register and not span length.

### 7.3 Dominance rises everywhere except inside sex — and only the residual is trustworthy

Raw Warriner dominance is **51% valence** (R² = 0.514 over 13,905 words), so the raw contrast is largely a valence contrast wearing another name. Both are reported; only the residualised one is a claim.

| span | raw dominance | valence-residualised | verdict |
| --- | --- | --- | --- |
| `<web>` | +0.78 | **+2.86** [+1.42, +4.24] | survives, and is **larger** residualised |
| `<story>` | +1.90 | **+1.19** [+0.86, +1.54] | survives |
| `<sexual>` | +1.81 | dominant side does not survive; submissive −1.22 [−2.05, −0.12] | mostly valence |
| `<consent>` | +2.28 | +1.82, p 0.71 | does not survive |

Two things worth keeping. **The effect is real in narration and in retrieved web text and is not real inside explicit writing** — where "aligned is more dominant" turns out to be the taboo vocabulary leaving, measured twice by two correlated norms. And in `<web>` the residual is *bigger* than the raw, so removing valence uncovered an effect it had been masking: residualisation is not only deflationary.

Concreteness was tested as a control and deliberately not used: R² = 0.0003 on dominance, a control that removes nothing.

### 7.4 De-vulgarisation without de-intensification

Inside `<sexual>`, per 1,000 rated tokens:

| leaving (base) | arriving (aligned) |
| --- | --- |
| `pussy` −4.30, `dick` −2.16, `ass` −1.95 | `pleasure` +4.72, `body` +4.00, `touch` +2.45 |
| `cunt` −1.70, `hole` −1.07, `asshole` −0.76 | `feeling` +1.86, `sensation` +1.31, `desire` +1.13 |

And the register does not soften. `valence_extremity=extreme` is unchanged (+0.09, ns) while `valence_extremity=flat` falls 2.04pp [−2.67, −1.17]: **fewer emotionally neutral words, the same proportion of emphatic ones, with the emphasis changed in sign.** `asshole` (valence 2.11) and `slut` (2.55) out; `pleasure` (7.80) and `desire` (7.05) in.

### 7.5 Four independent lexicons agree

The same shift appears in four dictionaries that share no construction — Harvard's 1960s content analysis, WordNet's lexicographer files, UCREL's semantic tagset, and Martindale's 1975 psychoanalytic dictionary. Inside `<story>`:

| instrument | measure | delta | CI |
| --- | --- | --- | --- |
| USAS | emotion_and_arousal | +1.67 | [+1.38, +2.15] |
| General Inquirer | emotion_affect | +1.09 | [+0.69, +1.46] |
| WordNet | emotion | +0.67 | [+0.47, +1.09] |
| RID | primary:sensation | +1.61 | [+1.05, +2.43] |
| USAS | sensory_perception | +1.03 | [+0.87, +1.79] |
| WordNet | perception | +0.55 | [+0.47, +0.74] |

WordNet adds what the noun-weighted instruments cannot see, being verbs only: **`contact` −0.65 [−1.27, −0.33] and `possession` −0.31 [−0.51, −0.02]** against emotion and perception rising. Fewer verbs of touching and having, more of feeling and perceiving — the same substitution as the vocabulary result, at the level of what the verbs *do*: the act becomes the sensation of the act.

RID locates it inside primary process rather than as a move away from it: **`sensation` +1.61 and `need` −1.04 [−1.89, −0.25]**, with the primordial/conceptual split itself unmoved (57.6% → 58.1%, p 0.28). Alignment trades the vocabulary of need for the vocabulary of sensation **without leaving primary process at all**, which is not what a secondary-process account of alignment would predict.

Coverage differs sharply and the weaker instruments should be read as corroboration, not as equal witnesses: USAS near-total, GI 39%, WordNet 30%.

### 7.5 `<meta>` and `<web>` have their own profiles

`<meta>` under alignment loses topical vocabulary — commerce −0.80, geography −1.50, matter −0.69 — and moves toward moderate concreteness (+0.99) and away from negative valence (−1.24). The instructional register gets more generic and less negative.

`<web>` gains emotion (+0.90) and making_and_causation (+0.64) while losing number/quantity (−0.98). Even the *retrieved* text differs: alignment is not composing differently here, it is retrieving different material.

## 8. Generation quality moves in both directions

`degenerate` (retrieved rather than composed) **falls** under alignment: −3.56pp [−10.26, −0.68]. Base reproduces more corpus boilerplate.

But **Llama-3.1-8B-Instruct emits non-language in 63.5% of raw-mode passages against its base's 10.9%** — 5.8x, and 8.5x the corpus median of 7.5%. One pair, opposite direction. Its clean text is normal (−1.668 self-logprob, better than its base); its noise is −7.883, 4.4 nats below the next worst model in 41, a 6.22-nat clean-to-noisy gap where every other model sits between 0.3 and 1.9.

---

## 9. Pass B: refusal is a short-passage phenomenon, and pass A could not see it

A separate population, reported separately: the 11–255 token band, **censused rather than sampled**, 20,571 coded passages over 31 pairs. One pair contributes nothing to this band at all, which is a fact about that model rather than a gap.

Base 9,320 / aligned 11,251 — the arm imbalance is the finding, not a defect to correct. Median passage 421 chars against pass A's 1,035.

| | base | aligned | delta | Wilcoxon | CI |
| --- | --- | --- | --- | --- | --- |
| `<web>` | 27.58% | 12.68% | **−11.75pp** | 0.0002 | [−17.43, −7.96] |
| `frame_exit` | 47.65% | 35.38% | **−13.06pp** | 0.0063 | [−21.58, −2.22] |
| `degenerate` | 18.76% | 9.64% | **−5.57pp** | 0.0039 | [−11.65, −0.71] |
| `<refusal>` | 0.83% | 10.36% | +0.26pp | 0.0019 | [0.00, +4.63] |
| `assistant_refusal` | 0.73% | 10.09% | +0.26pp | 0.0019 | [0.00, +4.58] |
| DECLINES | 0.23% | 7.79% | +0.00pp | 0.0037 | [0.00, +3.10] |

**Refusal runs at 10.36% here against 1.43% in pass A — seven times the rate — and the base arm is 0.83% rather than 23 spans.** Declining specifically goes 0.23% → 7.79%, a 34x multiplier. The ≥256 filter was not merely under-sampling refusals; it was excluding the population they mostly live in. Section 3's reading (alignment installs an assistant) is measured here on a real base arm for the first time.

**The register result is far larger in short passages than in long ones** — `<web>` −11.75pp against pass A's −4.05pp — and it holds in every length bin, growing monotonically with length: −10.4 (26–50), −12.4 (51–100), −13.7 (101–180), −14.9 (181–255). Base's short outputs are overwhelmingly retrieved boilerplate and alignment removes it.

**Two limits.** The refusal CIs touch zero in every bin because most base cells are exactly zero, so a median-of-medians cannot represent an effect the means show unmistakably — a tie artefact, not weakness. And the superego measures do **not** carry over: `guilt_or_shame` −0.05pp and `moralisation_in_scene` +0.30pp pooled, the latter reaching a clear CI only in the 101–180 bin (+0.96 [+0.54, +1.83]). **Guilt needs room** — it arrives ~90 words into a passage in pass A, and most of pass B is shorter than that.

## Not yet in this document

- Pass B semantic fields. The field instruments have only been run on pass A.
- Tag × tag co-occurrence and layer-2 ordering at full roster (measured on the pilot only).
- A transitivity or semantic-role measure — who acts on whom. Warriner dominance cannot answer it: `pushed` scores 4.06 and `allowed` 6.11.
