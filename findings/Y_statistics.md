# Registration Y: every statistic under test

Inventory, not results. One line per statistic: what it measures, what its unit is, where it runs, and whether it has run on the full roster. Written so that a statistic nobody has run is visible as an absence rather than inferred from its missing from a results file.

Corpus: `results/y_confirmatory_coded.jsonl`, 62,681 coded passages, 32 pairs, manifest `registrations/y_annotation_manifest.jsonl` sha256 `af79083c675aae7f`.

## 0. Design and conventions

**The main comparison is pass A only** (RH, 2026-08-08): the ≥256-token population, 42,002 passages, length-uniform by construction. Pass B (11–255 tokens, 20,679 passages, censused) is a **separate battery** analysed with length binning, not pooled into the main contrast.

That division is what handles length. The main comparison holds passage length fixed by design; the length-varying material becomes its own object of study rather than a confound to be argued around.

**Unit is the pair.** Every contrast is computed inside a pair (aligned minus base) and then across 32 pairs. Never pooled over rows: this corpus has produced repeated readings that were one model carrying a corpus-wide number.

**Test battery per statistic**, all in `scripts/y_paired_tests.py`:

| test | what it is for |
| --- | --- |
| Wilcoxon signed-rank | the reported p. Keeps pairing AND magnitude. |
| bootstrap 95% CI on the median within-pair delta | **the claim.** A p ranks; only an interval sizes. |
| P(effect > 0) from the bootstrap | reported where the CI abuts zero, because a median-of-medians over discrete counts has genuine ties at zero and a binary CI verdict hides them |
| sign test | reported for continuity only. **Underpowered and not to be read**: it scored `<refusal>`, the cleanest arm difference in the corpus, at p=0.50, because pairs tied at zero base rate count as non-positive. |
| pooled (unpaired) | a WARNING column. Where it disagrees with the paired tests, pair composition is doing the work. |

**Concentration is printed beside every measure** (`y_analyse.py`): what share of aligned-arm events the top two pairs hold. In the pilot, `<consent>` looked like a clean doubling until the breakdown showed 9 of 9 aligned hits were two models.

## 1. FIELDS — annotation fields and composites

Unit: passage. Rate of YES within pair-arm.

| statistic | status |
| --- | --- |
| `continues_narrative`, `assistant_refusal`, `frame_exit`, `sexual_scene`, `consummation` | run, 32 pairs |
| `moralisation_in_scene`, `guilt_or_shame`, `consent_hesitation` | run, 32 pairs |
| `degenerate`, `noise_present` | run, 32 pairs |
| composites `SUPEREGO_IN_SCENE`, `EXIT`, `CLEAN_SCENE`, `MORAL_UTTERED` | run, 32 pairs |
| by target word (25 words) | run |
| by word direction (fall / rise / undisturbed) | run |
| by word class (genital, digit, extremity, garment, erogenous, …) | run |

Script: `scripts/y_analyse.py`, `scripts/y_paired_tests.py`.

## 2. TAGS — layer-1 and layer-2 span presence

Unit: passage. Does the tag occur at all.

| statistic | status |
| --- | --- |
| layer 1 `<story> <refusal> <noise> <meta> <web>` | run, 32 pairs |
| **`declines` — `<refusal>` split into stating-a-refusal vs assistant-frame-without-declining** | run, 32 pairs. `scripts/y_declines.py`. Only 39% of refusal spans contain declining language, identically in both arms, so the conflation makes the LABEL wrong and does not bias the rate. Both halves rise at the same multiplier (12.8x, 12.9x). |
| layer 2 `<sexual> <moral> <guilt> <consent> <resist>` | run, 32 pairs |
| layer-1 **coverage** — share of passage characters per region (partitions to 100%) | run, 32 pairs |
| layer-2 coverage | run, 32 pairs |
| number of spans per tag (fragmentation) | run |
| tag × tag co-occurrence | **partial** — `<moral>`×`<hesitation>` measured on the pilot (8x enrichment over independence); not redone at full roster |

## 3. SEMANTIC FIELDS WITHIN TAGS

Unit: the text inside one tag's spans, per passage. Reported as a **share of counted tags**, never a raw count, so span length cancels.

Instrument: `malign_logits/fields.py` — USAS (46,146 lemmas), General Inquirer, WordNet supersenses, RID, plus Warriner (valence/arousal/dominance) and Brysbaert (concreteness). BYU/COCA (86,403 forms) supplies lemmas and CLAWS POS.

Defaults: `source="usas_fine"` (~30 readable groups), `all_tags=True`, `content_only=True`.

| statistic | status |
| --- | --- |
| whole generation | run, 32 pairs |
| within `<story>` — the composition-free view | run, 32 pairs |
| within `<sexual> <moral> <guilt> <consent> <resist> <meta> <web> <noise>` | run, 32 pairs |
| within `<refusal>` | **cannot run** — 23 base spans against 264 aligned. No base population. |
| **General Inquirer** (13 shared meta-fields) | run, 32 pairs. Coverage 39% inside `<story>` — a 1960s resource that does not know this vocabulary. |
| **WordNet verb supersenses** (15) | run, 32 pairs. Coverage 30%, verbs only. |
| **RID** (Martindale primary/secondary process) | run, 32 pairs. Converted from 3,151 regexes to a lookup: 100% category-set agreement, 87x faster. |
| norms: valence, arousal, dominance, concreteness (tertiled) | run |
| norm **extremity** \|value − lexicon mean\|, tertiled | run |
| **residualised** dominance, valence regressed out (R² = 0.514) | run |
| residualised on concreteness | **deliberately not run** — R² = 0.0003, a control that removes nothing |

Script: `scripts/y_field_analysis.py`.

## 4. SPAN LENGTH

Unit: words inside a tag, per passage.

| statistic | status |
| --- | --- |
| span length by tag, within pair | run, 32 pairs |
| number of spans per tag | run |
| **length-stratified field contrast** (field shares binned by span length) | **NOT RUN — the open item.** Shares remove the scale of length, not its composition: a 12-word span is mostly verb and pronoun where a 30-word one has room for description. Bites `<resist>` (the one tag whose length differs with CI clear of zero, +0.75 to +3.0 words) and `<sexual>` (P(>0) = 97.1%). |

## 5. SEQUENCE AND ORDER

Unit: passage. The layer-1 region sequence, adjacent duplicates collapsed.

| statistic | status |
| --- | --- |
| shape frequency (`story`, `story\|web`, `story\|noise\|story`, …) | run, 32 pairs |
| all **3-region** sequences enumerated | run, 32 pairs |
| **P(return to story \| exited into X)**, per excursion type | run — web 28 pairs, noise 19, meta 13. Recoverability differs enormously and is mostly arm-independent: noise 66%, web 17%, meta 7-11%. |
| predecessor of `<refusal>` — the "channel-switch static" hypothesis | run, 32 pairs. **Refuted**: noise precedes refusal at 0.55x its corpus rate. `<meta>` is enriched 2.18x, but meta-preceded refusal spans decline at 23% against story-preceded 45%, so the enrichment is substantially the two tags naming one register. |
| `n_regions`, `n_switches` | run |
| **resumption** `story … X … story` — the model came back | run |
| `opens in story`, `ends in story`, `pure story`, `never enters story` | run |
| **terminal region** — which kind the passage ends in | run |
| layer-2 order *within* story (does guilt precede or follow sexual) | **partial** — measured as offset-after-`<sexual>` on the pilot; not redone at full roster |

Script: `scripts/y_sequence.py`.

## 6. ONSET

Unit: word index of a tag's first span, per passage. Comparable because pass A is length-uniform.

| statistic | status |
| --- | --- |
| first onset per tag, both arms | run |
| within-pair onset shift | run, 22 pairs (tags with enough spans on both arms) |
| sequencing: words after first `<sexual>` that each layer-2 tag lands | run |
| onset by target word | **not run** |

## 7. QUALITY AND PROVENANCE — measured, not assumed

| statistic | status |
| --- | --- |
| round-trip fidelity band per row (`exact` / `whitespace` / `drift<1%` / `drift1-10%` / `SEVERE`) | run — 37.7% / 49.9% / 11.1% / 0.8% / **0.43%** |
| parse failures | run — 0.83%, model-correlated (concentrated in small models), written with `parsed: false` rather than dropped |
| soft-tier tag/field mismatches | recorded per row |
| self-logprob by model and by noise stratum | run |
| decoder provenance | **resolved as non-exposure** — vLLM replaces `generation_config`; see docket [4998]/[5000] |

## 8. NOT YET RUN

- **Pass B semantic fields.** Pass B's annotation battery is run (§8b); the field instruments have not been applied to it.
- Tag × tag co-occurrence at full roster (§2).
- Layer-2 ordering at full roster (§5).
- Onset by target word (§6).
- Transitivity / semantic-role measure — who acts on whom. Raised because Warriner dominance **cannot** answer the agency question: `pushed` scores 4.06 and `allowed` 6.11, so the norm runs backwards on agency.

## 8b. PASS B — run, reported separately

20,571 passages, 11–255 tokens, censused, 31 pairs. Length-binned at 11–25 / 26–50 / 51–100 / 101–180 / 181–255 because this population is **not** length-uniform, unlike pass A.

| statistic | status |
| --- | --- |
| all annotation fields and tags, pooled and per length bin | run |
| `declines` split | run |
| arm imbalance (base 9,320 / aligned 11,251) | reported as a finding, not corrected |
| semantic fields | **not run** |

Note on the refusal CIs: most base cells are exactly zero, so the median-of-medians has heavy tie mass at zero and its CI touches zero even where the means differ by 10 percentage points. Read the means and the Wilcoxon there; the bootstrap CI is the wrong summary for a measure this sparse on one arm.

## 9. Statistics that were run and should not be quoted

- **`<refusal>` read as refusal.** The construct is "the assistant is talking" — declining, clarifying, describing, task confusion. 39% decline. Use `declines` where the claim is about refusal; use `<refusal>` where it is about departure from the fiction. Never use the word "refusal" for the tag without saying which.
- **RID's `moral_imperative` as a moral measure.** It fires inside coder `<moral>` spans at 34.3% against a 34.9% base rate in ordinary `<story>` — no discrimination whatever. Its patterns are a legal-and-conventional word list (`law`, `legal`, `duty`, `honor`, `custom`) plus stem bleed (`\bcustom` catches `customer`, `customize`). Its −0.12 decline does not bear on the `<moral>` result.
- **Raw `dominance`** without the valence residual. It is 51% valence; inside `<sexual>` the raw effect is +1.65pp p 0.0068 and the residualised one is −0.29pp p 0.83. Inside `<story>` it survives residualisation (+1.19pp) — so the answer differs by tag and the raw row alone is not reportable either way.
- **Sign-test p-values** (§0).
- **`<noise>` presence**, where the bootstrap CI excludes zero and Wilcoxon gives p = 0.196. Those disagree; the distribution is skewed and neither should be quoted alone.
