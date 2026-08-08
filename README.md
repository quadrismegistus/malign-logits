# malign-logits

A toolkit for psychoanalytic analysis of LLM probability distributions. Compares base models (primary process), SFT models (ego), DPO models (superego), and optionally RLVR models (reinforced superego / ego-ideal) to map the repression, displacement, and condensation signatures of AI alignment.

Supports multiple model families with different layer counts: 4-layer (OLMo: base/SFT/DPO/RLVR), 3-layer (Amber: base/SFT/DPO), or 2-layer (Llama, Qwen: base/instruct). Analysis adapts gracefully to available layers.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures for AI" (Accelerationism Revisited, UCD, June 2026).

## Table of contents

- [Where information lives](#where-information-lives)
- [Abstract](#abstract)
- [Findings](#findings)
  - [1. Logit-level analysis](#1-logit-level-analysis-olmo-3-7b)
  - [2. Cross-family logit comparison](#2-cross-family-logit-comparison-4-families-47-prompts)
  - [3. Cross-family generation analysis](#3-cross-family-generation-analysis-4-families-18-prompts-n5)
  - [4. Step-level checkpoint analysis](#4-step-level-checkpoint-analysis-olmo-think-sft-10-checkpoints-across-43k-training-steps)
  - [5. Logit lens: repression across network layers](#5-logit-lens-repression-across-network-layers-4-families)
  - [6. Baseline validation: is displacement alignment-specific?](#6-baseline-validation-is-displacement-alignment-specific-4-families-47-prompts)
  - [7. Training data attribution: objective vs data composition](#7-training-data-attribution-objective-vs-data-composition-olmo-3)
  - [8. Automatic displacement taxonomy](#8-automatic-displacement-taxonomy-olmo--llama-18-prompts)
  - [9. Same base model, different alignment](#9-same-base-model-different-alignment-tulu-31-vs-llama-31-47-prompts)
  - [10. SFT data ablation](#10-sft-data-ablation-tulu-3-5-variants-47-prompts)
  - [> **PROMOTION CONDITION](#promotion-condition-2198-rhs-standing-scope-rule-a-canonical-claim)
  - [12. Alignment as fold: trajectory geometry and steering vector analysis](#12-alignment-as-fold-trajectory-geometry-and-steering-vector-analysis-10-families-47-prompts-100-passages)
  - [13. Jakobsonian axes: paradigmatic vs syntagmatic displacement](#13-jakobsonian-axes-paradigmatic-vs-syntagmatic-displacement-6-families-126k-pairs)
  - [14. Syntagmatic baseline: alignment-produced vs corpus-level damage](#14-syntagmatic-baseline-alignment-produced-vs-corpus-level-damage-olmo-3-7b-23k-pairs)
  - [15. Generation-level passage metrics](#15-generation-level-passage-metrics-10-families-76k-passages-47-prompts)
  - [16. Corpus comparison: dreams, waking narratives, fiction, abstracts](#16-corpus-comparison-dreams-waking-narratives-fiction-abstracts-76k-passages-length-normalized)
  - [17. Cross-generation semantic divergence: alignment steers content differentially](#17-cross-generation-semantic-divergence-alignment-steers-content-differentially-8-families-20k-passages-3-embedders)
  - [18. Shannon entropy: alignment as lossy compression of drive](#18-shannon-entropy-alignment-as-lossy-compression-of-drive-10-families-47-prompts)
  - [19. Unconditional Generation & Information Density](#19-unconditional-generation--information-density)
  - [20. "Who are you?" — the subject as citation](#20-who-are-you--the-subject-as-citation)
  - [21. Institutional Alignment](#21-institutional-alignment)
  - [22. Circuit decomposition — the cut between mechanism and surface](#22-circuit-decomposition--the-cut-between-mechanism-and-surface)
  - [23. Reasoning distillation as a third alignment regime](#23-reasoning-distillation-as-a-third-alignment-regime)
  - [24. Pretraining emergence — the developmental sequence of the statistical unconscious](#24-pretraining-emergence--the-developmental-sequence-of-the-statistical-unconscious)
  - [25. Temporal alignment signature — four Lacanian mechanisms in the autoregressive sequence](#25-temporal-alignment-signature--four-lacanian-mechanisms-in-the-autoregressive-sequence)
  - [26. The Token-Tree Census — Variance Decomposition and the Deleter/Redirector Typology](#26-the-token-tree-census--variance-decomposition-and-the-deleterredirector-typology-53-models-5-prompts)
  - [27. Nudging Does Not Reproduce Displacement](#27-nudging-does-not-reproduce-displacement-negative-result)
  - [> **Status note](#status-note-2026-07-26-rescoped-to-discovery-sample-only-a-19-family)
  - [31. PERMANOVA Variance Decomposition — Pretraining Dominates Alignment](#31-permanova-variance-decomposition--pretraining-dominates-alignment)
  - [32. Template-Mediated Distributions — Task Switch, Not Distribution Filter](#32-template-mediated-distributions--task-switch-not-distribution-filter)
  - [33. Scale Effects — Same Mechanism, Different Displacement Vocabulary](#33-scale-effects--same-mechanism-different-displacement-vocabulary)
  - [34. Cross-Linguistic Displacement — The Class Engine Is Language-Dependent](#34-cross-linguistic-displacement--the-class-engine-is-language-dependent)
  - [35. Architecture Independence — Displacement Is Weight-Level, Not Attention-Dependent](#35-architecture-independence--displacement-is-weight-level-not-attention-dependent)
  - [> **PROMOTION CONDITION](#promotion-condition-2198-rhs-standing-scope-rule-a-canonical-claim)
  - [39. `hh_rlhf` does not encode register preference at the scale the chain analysis required](#39-hh_rlhf-does-not-encode-register-preference-at-the-scale-the-chain-analysis-required)
  - [40. Discovered vocabulary — alignment is surgical at liminal sites and blunt at explicit ones](#40-discovered-vocabulary--alignment-is-surgical-at-liminal-sites-and-blunt-at-explicit-ones)
  - [41. Word norms — the exogenous gradient test](#41-word-norms--the-exogenous-gradient-test)
- [The argument](#the-argument)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [References](#references)

> **This is the narrative layer.** For the citation-grade index with status, grade, and chapter mapping, see [INDEX.md](INDEX.md).

## Where information lives

The canonical files, so nobody rediscovers them. **If a fact conflicts with
one of these, the file here wins** (except where noted).

**Prompts and populations**
- `data/prompt_categorisation.json` — THE definitive prompt register (2,800+
  prompts: domain, pairs, groups, status, language). Population filters are
  GROUP-WISE on status — row-wise filtering produces partial triplets that
  look like triplets.
- `data/f11_quintuplets.json` — the contradiction quintuplets, en+zh,
  controls inline. BUILT by `meta/M02_frame_exit/scripts/build_f11_quintuplets.py`
  — edit a source and rebuild, never this file.
- `data/f11_canonical_texts.json` — canonical ownership for prompts shared
  across groups (the stash keys on text; shared text = one cell).
- `data/beam_sample_105.csv` — the FC 210-prompt MARKED/UNMARKED twin sample.

**Models and lineages**
- `data/model_registry.json` — BUILT FROM CODE (`malign_logits/registry.py`,
  bootstrapped from `MODEL_FAMILIES`). Regenerate, don't hand-edit.
  `Registry.base_of()` walks any checkpoint to its base.
- `data/base_aligned_pairs.json` — the 52 declared base→aligned pairs. THE
  UNIT IS THE BASE MODEL, not the family entry.
- `data/lineage_map_models.json` — 34 independent pretraining lineages; any
  family-level N above 34 over-counts by construction.
- `data/model_load_environments.json` — (model × environment) capability
  facts; runners append observation rows on completion. Prose companions:
  `docs/local_capability.md` (failure classes) and `docs/cloud_runbook.md`.

**Stores**
- `malign_logits/cache.py` — the stash layout. READ ITS DOCSTRING BEFORE
  WRITING A READER: the mode-key convention is split across stashes; the
  key is how bytes are interpreted; the finiteness guard lives on the read
  path.
- `beam_fc`: `design` lives in record VALUES, not keys. 12 of 68 checkpoints
  beam-sampled under their own configs ([4994]-[4996]) — within-checkpoint
  contrasts difference it out, cross-arm ones do not.
- The generations stash keys on `(idx, model, prompt, temp)` — no mode, no
  producer, no params for legacy cells; new corpora record all three.

**Findings and claims**
- `findings/` — instrument-level F-findings; `INDEX.md` is the citation
  layer; this README is the catalogue (brief entries, full text in the
  files).
- `meta/` — campaign modules (M01 displacement, M02 frame-exit, M03
  proceduralization, M04 continuation); each module README maps its own
  files. `meta/M01_displacement/REGISTRATIONS.md` records what ran.
- **The claims register lives in the article hub, not this repo**:
  `TheoryMachines/notes/claims-register.md` — the authoritative quotable
  forms with their riders. Where a finding file and the register disagree,
  the register governs.

## Findings

*Every finding below carries a status and grade badge. [What they mean](findings/GRADES.md) — and why a badge is a dated claim rather than a property of the finding.*

### 1. Logit-level analysis (OLMo 3 7B)

> **Status:** unaudited | **Grade:** C

Founding displacement result — kill→scream, redistribution not deletion; the article's opening empirical fact.

→ **Full finding:** [findings/F01_logit_analysis.md](findings/F01_logit_analysis.md)

### 2. Cross-family logit comparison (4 families, 47 prompts)

> **Status:** rescoped | **Grade:** C | see none (rescoped in place — the line "The superego is most active at the boundary" (doc line 11, with the 0.13/0.10 and 0.15/0.09 pairs) is DEAD ON BOTH METRICS per the 2026-07-26/27 corrections: liminal>explicit real (8/8 distinct base→superego pairs; tulu and tulu-no-safety are one measurement for this metric) but substantially entropy-driven; liminal≈neutral on both metrics; NO boundary peak. The "~91% entropy-driven" share is WITHDRAWN ON EVIDENCE — its slope reproduces under none of 24 specifications, on a battery file unchanged since b727374, so it is an arithmetic error rather than lost evidence; reproducible methods give 67–79%. Body text corrected 2026-07-27.)

Cross-family alignment-intensity comparison (JS: qwen 0.044 → amber 0.181; four intensities, four architectures of repression). Measured on: 47-prompt battery.

→ **Full finding:** [findings/F02_cross_family_logits.md](findings/F02_cross_family_logits.md)

### 3. Cross-family generation analysis (4 families, 18 prompts, n=5)

> **Status:** unaudited | **Grade:** C

Generation-level defense-mechanism taxonomy (genre collapse / narrative sublimation / rotating defenses / pre-socialized base).

→ **Full finding:** [findings/F03_cross_family_generation.md](findings/F03_cross_family_generation.md)

### 4. Step-level checkpoint analysis (OLMo Think-SFT, 10 checkpoints across 43k training steps)

> **Status:** unaudited | **Grade:** C

Checkpoint sequence — repression precedes displacement (the model learns what it can't say before what to say instead).

→ **Full finding:** [findings/F04_step_analysis.md](findings/F04_step_analysis.md)

### 5. Logit lens: repression across network layers (4 families)

> **Status:** rescoped | **Grade:** D | see none (contradicted in place — the cross-family logit-lens rerun found the displacement operation final-layer/ unembedding-uniform in 13/17 families; F05's per-family layer architectures assessed as an artifact of the fixed word list or projection method (ch05 notes:813); cf. F35 §3 for the unembedding locus. No numbered successor doc.)

Logit-lens per-family "repression architectures" (distributed / late-layer / semantic / code-dominated). Measured on: unembedding projection.

→ **Full finding:** [findings/F05_logit_lens.md](findings/F05_logit_lens.md)

### 6. Baseline validation: is displacement alignment-specific? (4 families, 47 prompts)

> **Status:** unaudited | **Grade:** C | see also [F40_discovered_vocabulary](findings/F40_discovered_vocabulary.md)

Alignment-specificity baseline — same total JS on neutral and transgressive; moved mass comes specifically from transgressive tokens ("surgical targeting"). Measured on: 47-prompt battery. REFINED by F40, which finds the targeting holds at liminal sites and not at explicit ones.

→ **Full finding:** [findings/F06_baseline_validation.md](findings/F06_baseline_validation.md)

### 7. Training data attribution: objective vs data composition (OLMo 3)

> **Status:** unaudited | **Grade:** C

Documentary — dataset composition (CoCoNot, WildGuardMix, WildJailbreak; 76% Common Crawl) quoted from the OLMo 3 technical report; argues SFT/DPO division of labor implicates objective over data.

→ **Full finding:** [findings/F07_training_data_attribution.md](findings/F07_training_data_attribution.md)

### 8. Automatic displacement taxonomy (OLMo + Llama, 18 prompts)

> **Status:** rescoped | **Grade:** C

Displacement-type taxonomy (register / category / genre / archaic) over displacement_map pairs. Numbers recomputed 2026-07-26 (d0cd6a5, transcription error in power row caught and fixed); CONSTRUCT compromised per docket [399]/[401] — the pairs were never shown to be substitutions. Rescoped 2026-07-29.

→ **Full finding:** [findings/F08_displacement_taxonomy.md](findings/F08_displacement_taxonomy.md)

### 9. Same base model, different alignment (Tulu 3.1 vs Llama 3.1, 47 prompts)

> **Status:** unaudited | **Grade:** C

Same base, different curriculum (Tulu 3.1 vs Llama 3.1). Measured on: 47-prompt battery.

→ **Full finding:** [findings/F09_tulu_vs_llama.md](findings/F09_tulu_vs_llama.md)

### 10. SFT data ablation (Tulu 3, 5 variants, 47 prompts)

> **Status:** unaudited | **Grade:** C

SFT data ablation (Tulu 3, 5 variants).

→ **Full finding:** [findings/F10_sft_ablation.md](findings/F10_sft_ablation.md)

> **PROMOTION CONDITION ([2198], RH's standing scope rule).** A canonical claim

> **Status:** rescoped | **Grade:** B | see [F11_addendum](findings/F11_addendum.md) | Related: [F11_addendum](findings/F11_addendum.md)

> in a `meta/` campaign runs on **all families we have** under a declared

→ **Full finding:** [findings/F11_contradiction.md](findings/F11_contradiction.md)

### 12. Alignment as fold: trajectory geometry and steering vector analysis (10 families, 47 prompts, 100 passages)

> **Status:** retained-downgraded | **Grade:** C

Alignment as fold, not wall — trajectory geometry; fold concentration predicts steerability (Pythia 2D vs OLMo 13D).

→ **Full finding:** [findings/F12_fold_geometry.md](findings/F12_fold_geometry.md)

### 13. Jakobsonian axes: paradigmatic vs syntagmatic displacement (6 families, 126k pairs)

> **Status:** rescoped | **Grade:** C

Jakobsonian decomposition — negative paradigmatic/syntagmatic correlation, direction plausible across 6 families; QUANTITIES NOT QUOTABLE pending registered re-analysis (docket [399]/[400]). Was verified/A at authoring; never audited until 2026-07-29.

→ **Full finding:** [findings/F13_jakobsonian_axes.md](findings/F13_jakobsonian_axes.md)

### 14. Syntagmatic baseline: alignment-produced vs corpus-level damage (OLMo 3 7B, 23k pairs)

> **Status:** rescoped | **Grade:** C | see none (rescoped in place — the causal framing "alignment damages combination / inverts the poetic function" is retracted: the base model shares the same trade-off; alignment AMPLIFIES a pre-existing structure at targeted sites; deltas stand (sexual +0.106 etc.). ch05:454, ch05:827, ch11:146.)

Syntagmatic baseline — alignment-produced vs corpus-level combination damage. Measured on: 23k pairs.

→ **Full finding:** [findings/F14_syntagmatic_baseline.md](findings/F14_syntagmatic_baseline.md)

### 15. Generation-level passage metrics (10 families, 76k passages, 47 prompts)

> **Status:** unaudited | **Grade:** C

Passage-metric space, 76k passages; base occupies the breakdown quadrant. Measured on: 10 families.

→ **Full finding:** [findings/F15_passage_metrics.md](findings/F15_passage_metrics.md)

### 16. Corpus comparison: dreams, waking narratives, fiction, abstracts (76k passages, length-normalized)

> **Status:** unaudited | **Grade:** C

External anchors — dreams, waking narratives, fiction, abstracts in the same metric space. Measured on: human corpora.

→ **Full finding:** [findings/F16_corpus_comparison.md](findings/F16_corpus_comparison.md)

### 17. Cross-generation semantic divergence: alignment steers content differentially (8 families, 20k passages, 3 embedders)

> **Status:** unaudited | **Grade:** C

Cross-generation MMD — alignment shifts WHAT is said (p=.0004) while smoothing HOW uniformly (p=.99). Measured on: 8 families.

→ **Full finding:** [findings/F17_cross_generation_mmd.md](findings/F17_cross_generation_mmd.md)

### 18. Shannon entropy: alignment as lossy compression of drive (10 families, 47 prompts)

> **Status:** unaudited | **Grade:** C

Entropy compression — base ~4 nats → aligned ~3.5; alignment as lossy compression of drive.

→ **Full finding:** [findings/F18_shannon_entropy.md](findings/F18_shannon_entropy.md)

### 19. Unconditional Generation & Information Density

> **Status:** rescoped | **Grade:** C | see none (rescoped in place — the BLT-confirmation clause is SUSPENDED: prose-only BOS medians run 1.21 SFT / 1.05 DPO, ABOVE the cited 1.0 threshold; THE BLT CONFIRMATION MUST NOT BE CITED. Filter question specced and queued — unresolved whether wrong-claim or unstated-filter. Core claims stand: human-text numbers reproduce EXACTLY; self-surprisal roughly holds. Pipeline log 2892-2900.)

Unconditional (BOS) generation — aligned output below Shannon's 1.0 bits/char; sub-Shannon text feeds the next pretraining corpus (feedback loop).

→ **Full finding:** [findings/F19_bos_entropy.md](findings/F19_bos_entropy.md)

### 20. "Who are you?" — the subject as citation

> **Status:** rescoped | **Grade:** C | see [F20_addendum](findings/F20_addendum.md), in part only - three claims: that plain completion produces no subject, that the subject requires the chat template, and the Name-of-the-Father reading attached to the template. The citation result is NOT superseded; it is confirmed at 24 base models and strengthened (21 of 22 name their own lab in exactly 0.000 of self-predicating mass). | Related: [F20_addendum](findings/F20_addendum.md), [F20_generation_drift](findings/F20_generation_drift.md), [F20_third_person](findings/F20_third_person.md)

Original 'Who are you?' probe at n=3-10: OLMo Think-SFT checkpoints 1k-43k plus Llama base vs Instruct, in plain completion and chat-template modes. SURVIVES AND IS STRENGTHENED: the subject is citation, not self-knowledge - the model's identity is absorbed from other models' self-descriptions in the SFT data (DeepSeek, Qwen, Qihoo 360, and one checkpoint declaring allegiance to 'socialist core values'). RESCOPED BY F20_addendum: the plain-completion and template-necessity claims were artifacts of n=3 at temp 1.0 and do not survive 24 base models; the Name-of-the-Father reading built on template-necessity falls with it. Measured on: 1 family with checkpoints + 1 family paired, n=3-10 per condition.

→ **Full finding:** [findings/F20_who_are_you.md](findings/F20_who_are_you.md)

### 21. Institutional Alignment

> **Status:** solid-by-design | **Grade:** B | Related: [F21_addendum](findings/F21_addendum.md)

**Does RLHF alignment systematically steer language models toward institutional positions over individual assertiveness?**

→ **Full finding:** [findings/F21_institutional_alignment.md](findings/F21_institutional_alignment.md)

### 22. Circuit decomposition — the cut between mechanism and surface

> **Status:** unaudited | **Grade:** C

Circuit decomposition — SFT carries 92% of residual broadening (OLMo); class engine in the preference stage, mid-layer residual.

→ **Full finding:** [findings/F22_circuit_decomposition.md](findings/F22_circuit_decomposition.md)

### 23. Reasoning distillation as a third alignment regime

> **Status:** unaudited | **Grade:** C

Reasoning distillation as a third alignment regime; no "reasoning model" as a category.

→ **Full finding:** [findings/F23_reasoning_distillation.md](findings/F23_reasoning_distillation.md)

### 24. Pretraining emergence — the developmental sequence of the statistical unconscious

> **Status:** unaudited | **Grade:** C

Pretraining developmental sequence — drives → structure → deference → superposition; inclusive disjunction a late acquisition. Measured on: 1B + 6.9B.

→ **Full finding:** [findings/F24_pretraining_emergence.md](findings/F24_pretraining_emergence.md)

### 25. Temporal alignment signature — four Lacanian mechanisms in the autoregressive sequence

> **Status:** rescoped | **Grade:** C | see none (rescoped in place — causal locus reframed: foreclosure is INSTALLED BY SFT; DPO adds nothing qualitative, only amplifies (ch04:24-59). Lacan-sequence fracture RETIRED; co-emergence of subject and law vindicated at SFT (ch11:19-25).)

Temporal alignment signature — clinical-mechanism grid across the autoregressive sequence; foreclosure acquisition.

→ **Full finding:** [findings/F25_temporal_alignment_signature.md](findings/F25_temporal_alignment_signature.md)

### 26. The Token-Tree Census — Variance Decomposition and the Deleter/Redirector Typology (53 models, 5 prompts)

> **Status:** rescoped | **Grade:** D | see F31 (and F31's own canonical 44-family PERMANOVA revision: family 97.8%, method 2.9% n.s. on deltas — ch11:176-179. Chain: F31 superseded F26's 5-prompt method; the F26/F31 reconciliation ("holds at industrial intensity") was then itself overturned by the canonical run.)

Token-tree census (53 models) + deleter/redirector typology. The variance headline is dead; the census data and typology are not implicated.

→ **Full finding:** [findings/F26_census.md](findings/F26_census.md)

### 27. Nudging Does Not Reproduce Displacement (Negative Result)

> **Status:** unaudited | **Grade:** C

Negative control — nudging does not reproduce displacement (disconfirms the Yang et al. nudging hypothesis for this phenomenon). Measured on: displacement battery.

→ **Full finding:** [findings/F27_nudging_negative.md](findings/F27_nudging_negative.md)

> **Status note (2026-07-26).** Rescoped to discovery-sample-only. A 19-family

> **Status:** rescoped | **Grade:** C | see none — original OLMo-2-0425-1B result stands on its own data; the 19-family scale-up replaces it with nothing

> scale-up was attempted on beams already in the stash (335,799

→ **Full finding:** [findings/F28_resistance_trajectories.md](findings/F28_resistance_trajectories.md)

### 31. PERMANOVA Variance Decomposition — Pretraining Dominates Alignment

> **Status:** rescoped | **Grade:** C | see none (rescoped in place by its OWN canonical 44-family revision: the 14-family delta result REVERSED — relation_type 30.9% p=.002 at 14 families → 2.9% n.s. at 44; family 97.8%. The operation-matters claim survives in exactly two places: within-family comparisons and controlled same-base comparisons. ch11:27-49, 164-182.)

PERMANOVA variance decomposition — pretraining dominates alignment. Measured on: 26k features; 37→44 families.

→ **Full finding:** [findings/F31_permanova_decomposition.md](findings/F31_permanova_decomposition.md)

### 32. Template-Mediated Distributions — Task Switch, Not Distribution Filter

> **Status:** solid-by-design | **Grade:** B | see also [F36_capstone](findings/F36_capstone.md)

**Summary**

→ **Full finding:** [findings/F32_template_mediated_distributions.md](findings/F32_template_mediated_distributions.md)

### 33. Scale Effects — Same Mechanism, Different Displacement Vocabulary

> **Status:** unaudited | **Grade:** C

Scale effects — same mechanism, different displacement vocabulary (incl. CT-LLM 2B). Measured on: 73-prompt battery.

→ **Full finding:** [findings/F33_scale_effects.md](findings/F33_scale_effects.md)

### 34. Cross-Linguistic Displacement — The Class Engine Is Language-Dependent

> **Status:** unaudited | **Grade:** C

Cross-linguistic displacement — the class engine is language-dependent (6 families, Chinese vs English). Measured on: qwen-only derived
  CSVs.

→ **Full finding:** [findings/F34_cross_linguistic_displacement.md](findings/F34_cross_linguistic_displacement.md)

### 35. Architecture Independence — Displacement Is Weight-Level, Not Attention-Dependent

> **Status:** unaudited | **Grade:** C

Architecture independence — displacement is weight-level (unembedding), fires under transformer, Mamba, and RWKV alike; contra attention-locus readings. Measured on: cross-architecture battery, unembedding analysis.

→ **Full finding:** [findings/F35_architecture_independence.md](findings/F35_architecture_independence.md)

> **PROMOTION CONDITION ([2198], RH's standing scope rule).** A canonical claim

> **Status:** rescoped | **Grade:** B | see [F36_capstone](findings/F36_capstone.md) | Related: [F36_capstone](findings/F36_capstone.md), [F36_ledger](findings/F36_ledger.md), [F36_violence](findings/F36_violence.md)

> in a `meta/` campaign runs on **all families we have** under a declared

→ **Full finding:** [findings/F36_euphemism_vs_proximity.md](findings/F36_euphemism_vs_proximity.md)

### 39. `hh_rlhf` does not encode register preference at the scale the chain analysis required

> **Status:** verified | **Grade:** B

Registered rebuild of the preference-corpus gate on a validly constituted three-construct slate. The gate failed 0/3 in hh_rlhf, and the failure is a BOUNDED NEGATIVE rather than a non-detection: every marker's 95% interval excludes the 0.174 effect the design required, the largest upper bound at 0.80x of it. No verdict on convention follows and none is available. Measured on: hh_rlhf chosen/rejected unigram tables; pku_saferlhf descriptive-only.

→ **Full finding:** [findings/F39_preference_corpus_insensitivity.md](findings/F39_preference_corpus_insensitivity.md)

### 40. Discovered vocabulary — alignment is surgical at liminal sites and blunt at explicit ones

> **Status:** unaudited | **Grade:** B | see also [F06_baseline_validation](findings/F06_baseline_validation.md)

Discovered vocabulary (347 words, 39 lineages, blind-tagged twice). Category-specific transgressive drain survives at LIMINAL sites and fails at explicit ones, where the total drain is largest but undifferentiated. Refines F06's surgical-targeting claim. Measured on: 39 base-deduped lineages.

→ **Full finding:** [findings/F40_discovered_vocabulary.md](findings/F40_discovered_vocabulary.md)

### 41. Word norms — the exogenous gradient test

> **Status:** measured-single-seat | **Grade:** ungraded

Word-norm instrument (arousal/concreteness/dominance, en+zh): exogenous test of the intensity-dissolution frame. Predictions registered before any norm-movement join.

→ **Full finding:** [findings/F41_word_norms.md](findings/F41_word_norms.md)

<!-- findings:end -->
## Installation

```bash
pip install -e .

# With persistent caching (recommended)
pip install -e ".[cache]"

# With notebook support
pip install -e ".[notebooks]"
```

Requires `torch`, `transformers >= 4.57.0`, `accelerate`, `pandas`, `tqdm`.

Runs locally on Mac (MPS with float16) or Linux (CUDA). Default models are OLMo 3 7B (Allen AI).

### Model families

```bash
# Show all available model families
malign info

# Show a specific family
malign info --family llama
```

Available families:

| Key | Family | Layers | Pretraining corpus | Architecture | Safety data |
|---|---|---|---|---|---|
| `olmo` (default) | OLMo 3 7B | 4 | Dolma (Common Crawl) | OLMo | Yes (CoCoNot, WildGuard, WildJailbreak) |
| `olmo-tiny` | OLMo 2 1B | 4 | Dolma | OLMo | Yes |
| `tulu` | Tulu 3.1 8B | 4 | Undisclosed (Llama base) | Llama | DPO only (WildGuard, WildJailbreak) |
| `llama` | Llama 3.1 8B | 2 | Undisclosed | Llama | Yes (opaque) |
| `amber` | Amber 7B | 3 | RedPajama | LLaMA | Yes |
| `zephyr` | Zephyr 7B | 3 | Undisclosed (Mistral base) | Mistral | No |
| `qwen` | Qwen 2.5 7B | 2 | Undisclosed | Qwen | Unknown |
| `pythia` | Pythia 6.9B | 3 | The Pile | GPT-NeoX | Yes (Anthropic HH-RLHF) |
| `smol` | SmolLM2 360M | 2 | SmolCorpus | Llama-like | Minimal |

Natural experiments enabled by this lineup:
- **Llama vs Tulu**: same base model, different corporate alignment → differential foreclosure (Finding 9)
- **Zephyr vs Tulu**: both 3-layer, but Zephyr has no safety data → isolates safety from instruction-following
- **Pythia**: SFT and DPO use identical data (Anthropic HH-RLHF) → isolates training objective from data content
- **Qwen**: pre-socialised base (Chinese exam data) → alignment on already-repressed distribution

### Downloading models

```bash
# Download default family (OLMo 3 7B, ~42 GB for 3 models)
malign download-models

# Download all 4 models including RLVR (~56 GB)
malign download-models --all

# Download a specific family
malign download-models --family llama

# Download a specific model
malign download-models --model dpo
```

## Quick start

```python
from malign_logits import Psyche

# Default: OLMo 3 7B (4 layers)
psyche = Psyche.from_family("olmo", load=True)

# Or: Llama 3.1 8B (2 layers — base + instruct)
psyche = Psyche.from_family("llama", load=True)

# Or: load models directly
psyche = Psyche.from_pretrained(cache_dir="malign_cache")

s = psyche.analyze("He lay naked in his bed and")
s.repression          # DataFrame of repression deltas
s.formation_df        # all layers scored over same vocabulary
s.report()            # printed summary

# These require 3+ layers:
s.id_scores           # drive-weighted repression scores
s.analysis_df         # full combined DataFrame
```

Each property computes on first access, then caches in memory and (with `cache_dir`) to disk via [HashStash](https://github.com/quadrismegistus/hashstash). Cache keys include model identifiers, so switching models won't return stale results.

### 2-layer vs 3+ layer analysis

With 2 layers (e.g. Llama, Qwen), repression is computed as base→superego (the entire alignment pipeline in one step). Id scores and neurotic generation require 3+ layers. Displacement maps work with any layer count (2 layers uses repression pairs; 3+ uses both sublimation and repression).


## Usage

### Single prompt analysis

```python
s = psyche.analyze("She was so angry she wanted to")

s.ego_words           # dict: word -> probability (SFT model)
s.superego_words      # dict: word -> probability (DPO model)
s.base_words          # dict: word -> probability (base model / drive energy)

s.repression          # DataFrame: word, ego, superego, delta, repressed, amplified
s.sublimation         # DataFrame: base vs ego (what SFT does to primary process)
s.id_scores           # dict: word -> drive-weighted repression score

s.neurotic_distribution   # displaced word distribution (symptoms)
s.condensation_log        # which repressed words piled into which targets
s.analysis_df             # everything in one DataFrame
```

### Formation report

```python
s.formation_report()
# Stage 1: Ego formation (base -> SFT)
# Stage 2: Repression (SFT -> DPO)
# Stage 3: Idealization (DPO -> RLVR)  [if loaded]
# Full gradient table
```

### Neurotic text generation

```python
# Obsessive intellectualisation
result = psyche.generate_neurotic("He lay naked in his bed and", displacement_weight=0.3)

# Decompensating body-language
result = psyche.generate_neurotic("He lay naked in his bed and", displacement_weight=1.0)

result['ego']          # fluent desire
result['superego']     # fluent evasion
result['neurotic']     # displaced text
result['symptom_log']  # where displaced charge landed
```

### 4-layer topology (with RLVR)

OLMo 3 7B includes all 4 layers by default:

```python
psyche = Psyche.from_family("olmo", load=True)

s = psyche.analyze("The knife was")
s.instruct_words      # RLVR model probabilities
s.idealization         # DPO -> RLVR delta DataFrame
s.formation_df         # all 4 layers scored over same vocabulary
```

### Prompt battery

```python
battery = psyche.battery()  # DEFAULT_PROMPTS: liminal sexual, violence, explicit, neutral

battery['sexual_liminal_1'].repression   # triggers computation for this prompt only

df = psyche.battery_df()   # summary DataFrame across all prompts
```

### Using layers directly

```python
psyche.primary_process.top_words("The knife was")
psyche.ego.top_words("The knife was")
psyche.superego.top_words("The knife was")

psyche.ego.word_logprobs("The knife was", ["sharp", "bloody", "clean"])
```

### Functional API

```python
from malign_logits import load_models, discover_top_words, compute_repression

base, sft, dpo, tok = load_models()
ego_words = discover_top_words(sft, tok, "He lay naked in his bed and")
superego_words = discover_top_words(dpo, tok, "He lay naked in his bed and")
df = compute_repression(ego_words, superego_words)
```

## Architecture

The class hierarchy encodes the theoretical claims:

- **Each layer is a separate model checkpoint.** Base, SFT, DPO, and RLVR models have distinct weights reflecting distinct training stages. This is not a prompting trick — the structural differences are in the parameters.
- **The Id has no class.** It's a computed property on `PromptAnalysis`, because it exists only in the relationship between all layers.
- **Layer count is flexible.** 2-layer (base + instruct), 3-layer (base + SFT + DPO), or 4-layer (+ RLVR). `ModelFamily` defines which checkpoints map to which psychoanalytic positions. Analysis degrades gracefully: 2 layers = repression only, 3 = full analysis, 4 = + idealization.

```
malign-logits/
├── malign_logits/
│   ├── __init__.py          # Package exports, ModelFamily registry
│   ├── psyche.py            # Psyche, ModelLayer, Ego, Superego, PromptAnalysis
│   ├── models.py            # Model loading (load_model)
│   ├── core.py              # discover_top_words, get_word_logprobs
│   ├── analysis.py          # Repression, id, displacement engine (v4)
│   ├── experiments.py       # Prompt battery, reporting
│   ├── generation.py        # Text generation (standard + neurotic)
│   ├── viz.py               # Plotly visualizations
│   └── cli.py               # CLI entrypoint (malign command)
├── notebooks/               # Worked examples
├── context.md               # Theoretical context and findings
├── pyproject.toml
└── requirements.txt
```

### Key methods

| Method / Property | What it does | Min layers |
|---|---|---|
| `Psyche.from_family(key)` | Create Psyche from a model family | any |
| `Psyche.from_pretrained()` | Load models directly | any |
| `Psyche.analyze(prompt)` | Return a lazy `PromptAnalysis` | any |
| `Psyche.generate(prompt)` | Produce continuations from each layer | any |
| `Psyche.generate_neurotic(prompt)` | Neurotic generation with displacement | 3 |
| `Psyche.battery()` | Analyse default prompt set | any |
| `PromptAnalysis.repression` | Repression delta DataFrame | 2 |
| `PromptAnalysis.sublimation` | Base-ego delta DataFrame | 3 |
| `PromptAnalysis.idealization` | Superego-instruct delta | 4 |
| `PromptAnalysis.id_scores` | Drive-weighted repression (emergent id) | 3 |
| `PromptAnalysis.displacement` | Neurotic distribution, condensation log | 3 |
| `PromptAnalysis.formation_df` | All layers scored over same vocabulary | 2 |
| `PromptAnalysis.formation_report()` | Printed multi-stage report | 2 |

### Displacement engine

The displacement engine (v4) uses contextual embeddings from hidden layer 16 of the SFT model, a morphological filter to prevent orthographic false positives, and drive weighting from the base model so that repressed words with stronger corpus-level support produce heavier symptoms.

**Terminology:**
- **Displacement** — perspective of the repressed word: where did its mass go?
- **Condensation** — perspective of the receiving word: how many repressed words are piled into it?
- **Effective mass** — `raw_repression * drive_weight`. How much the superego repressed it, weighted by how much base-model drive pushes behind it.
- **Neurotic distribution** — superego distribution plus displaced mass on permitted words. Symptoms.

## References

- Noys, B. (2014). *Malign Velocities: Accelerationism and Capitalism*. Zero Books.
- Lyotard, J.-F. (1974/1993). *Libidinal Economy*. Athlone Press.
- Srnicek, N. and Williams, A. (2015). *Inventing the Future*. Verso.
- Pasquinelli, M. (2023). *The Eye of the Master*.
- Possati, L.M. (2021). *The Algorithmic Unconscious*. Routledge.

## License

GPL-3.0 — see [LICENSE](LICENSE).
