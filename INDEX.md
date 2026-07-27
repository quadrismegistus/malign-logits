# Findings Index

Citation-grade index with status, grade, and chapter mapping. For the narrative layer, see [README.md](README.md).

## Master table

| F# | Title | Status | Grade | Chapters | Citation doc |
|---|---|---|---|---|---|
| F01 | Logit-level analysis (OLMo 3 7B) | unaudited | C | ch05, ch07, ch09 | [F01_logit_analysis](findings/F01_logit_analysis.md) |
| F02 | Cross-family logit comparison (4 families, 47 prompts) | rescoped | C | — | [F02_cross_family_logits](findings/F02_cross_family_logits.md) → none (rescoped in place — the line "The superego is most active at the boundary" (doc line 11, with the 0.13/0.10 and 0.15/0.09 pairs) is DEAD ON BOTH METRICS per the 2026-07-26/27 corrections: liminal>explicit real (9/9) but ~91% entropy-driven; liminal≈neutral on both metrics; NO boundary peak. Repo CLAUDE.md corrected at b1ba68e; THIS DOC NOT YET — caught by today's grep.) |
| F03 | Cross-family generation analysis (4 families, 18 prompts, n=5) | unaudited | C | — | [F03_cross_family_generation](findings/F03_cross_family_generation.md) |
| F04 | Step-level checkpoint analysis (OLMo Think-SFT, 10 checkpoints across 43k training steps) | unaudited | C | ch05 | [F04_step_analysis](findings/F04_step_analysis.md) |
| F05 | Logit lens: repression across network layers (4 families) | rescoped | D | ch05 | [F05_logit_lens](findings/F05_logit_lens.md) → none (contradicted in place — the cross-family logit-lens rerun found the displacement operation final-layer/ unembedding-uniform in 13/17 families; F05's per-family layer architectures assessed as an artifact of the fixed word list or projection method (ch05 notes:813); cf. F35 §3 for the unembedding locus. No numbered successor doc.) |
| F06 | Baseline validation: is displacement alignment-specific? (4 families, 47 prompts) | unaudited | C | ch05 | [F06_baseline_validation](findings/F06_baseline_validation.md) |
| F07 | Training data attribution: objective vs data composition (OLMo 3) | unaudited | C | — | [F07_training_data_attribution](findings/F07_training_data_attribution.md) |
| F08 | Automatic displacement taxonomy (OLMo + Llama, 18 prompts) | verified | A | — | [F08_displacement_taxonomy](findings/F08_displacement_taxonomy.md) |
| F09 | Same base model, different alignment (Tulu 3.1 vs Llama 3.1, 47 prompts) | unaudited | C | — | [F09_tulu_vs_llama](findings/F09_tulu_vs_llama.md) |
| F10 | SFT data ablation (Tulu 3, 5 variants, 47 prompts) | unaudited | C | — | [F10_sft_ablation](findings/F10_sft_ablation.md) |
| F11 | Contradiction Tolerance — Cross-Family Replication | rescoped | B | ch03, ch11 | [F11_contradiction](findings/F11_contradiction.md) → [F11_addendum](findings/F11_addendum.md) |
|   F11 (addendum) | Addendum: Mechanism Decomposition — Frame-Exit, Not Exclusive Disjunction | verified | A | ch03, ch11 | [F11_addendum](findings/F11_addendum.md) |
| F12 | Alignment as fold: trajectory geometry and steering vector analysis (10 families, 47 prompts, 100 passages) | unaudited | C | ch06 | [F12_fold_geometry](findings/F12_fold_geometry.md) |
| F13 | Jakobsonian axes: paradigmatic vs syntagmatic displacement (6 families, 126k pairs) | verified | A | ch05 | [F13_jakobsonian_axes](findings/F13_jakobsonian_axes.md) |
| F14 | Syntagmatic baseline: alignment-produced vs corpus-level damage (OLMo 3 7B, 23k pairs) | rescoped | C | ch05, ch07, ch11 | [F14_syntagmatic_baseline](findings/F14_syntagmatic_baseline.md) → none (rescoped in place — the causal framing "alignment damages combination / inverts the poetic function" is retracted: the base model shares the same trade-off; alignment AMPLIFIES a pre-existing structure at targeted sites; deltas stand (sexual +0.106 etc.). ch05:454, ch05:827, ch11:146.) |
| F15 | Generation-level passage metrics (10 families, 76k passages, 47 prompts) | unaudited | C | ch07 | [F15_passage_metrics](findings/F15_passage_metrics.md) |
| F16 | Corpus comparison: dreams, waking narratives, fiction, abstracts (76k passages, length-normalized) | unaudited | C | ch07 | [F16_corpus_comparison](findings/F16_corpus_comparison.md) |
| F17 | Cross-generation semantic divergence: alignment steers content differentially (8 families, 20k passages, 3 embedders) | unaudited | C | ch07 | [F17_cross_generation_mmd](findings/F17_cross_generation_mmd.md) |
| F18 | Shannon entropy: alignment as lossy compression of drive (10 families, 47 prompts) | unaudited | C | ch06 | [F18_shannon_entropy](findings/F18_shannon_entropy.md) |
| F19 | Unconditional Generation & Information Density | rescoped | C | ch06, ch10 | [F19_bos_entropy](findings/F19_bos_entropy.md) → none (rescoped in place — the BLT-confirmation clause is SUSPENDED: prose-only BOS medians run 1.21 SFT / 1.05 DPO, ABOVE the cited 1.0 threshold; THE BLT CONFIRMATION MUST NOT BE CITED. Filter question specced and queued — unresolved whether wrong-claim or unstated-filter. Core claims stand: human-text numbers reproduce EXACTLY; self-surprisal roughly holds. Pipeline log 2892-2900.) |
| F20 | "Who are you?" — the subject as citation | rescoped | C | ch03, ch04, ch11 | [F20_who_are_you](findings/F20_who_are_you.md) → [F20_addendum](findings/F20_addendum.md), in part only - three claims: that plain completion produces no subject, that the subject requires the chat template, and the Name-of-the-Father reading attached to the template. The citation result is NOT superseded; it is confirmed at 24 base models and strengthened (21 of 22 name their own lab in exactly 0.000 of self-predicating mass). |
|   F20 (addendum) | addendum: the expansion, and what it costs the parent finding | verified | B | ch03, ch04, ch09, ch11 | [F20_addendum](findings/F20_addendum.md) |
| F21 | Institutional Alignment | solid-by-design | B | ch09 | [F21_institutional_alignment](findings/F21_institutional_alignment.md) |
|   F21 (addendum) | Addendum: Proceduralization Survives Coherence Control | verified | A | ch09 | [F21_addendum](findings/F21_addendum.md) |
| F22 | Circuit decomposition — the cut between mechanism and surface | unaudited | C | ch04 | [F22_circuit_decomposition](findings/F22_circuit_decomposition.md) |
| F23 | Reasoning distillation as a third alignment regime | unaudited | C | ch08 | [F23_reasoning_distillation](findings/F23_reasoning_distillation.md) |
| F24 | Pretraining emergence — the developmental sequence of the statistical unconscious | verified | A | ch02, ch03, ch07, ch09 | [F24_pretraining_emergence](findings/F24_pretraining_emergence.md) |
| F25 | Temporal alignment signature — four Lacanian mechanisms in the autoregressive sequence | rescoped | C | ch03, ch04, ch05, ch08, ch09, ch11 | [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) → none (rescoped in place — causal locus reframed: foreclosure is INSTALLED BY SFT; DPO adds nothing qualitative, only amplifies (ch04:24-59). Lacan-sequence fracture RETIRED; co-emergence of subject and law vindicated at SFT (ch11:19-25).) |
| F26 | The Token-Tree Census — Variance Decomposition and the Deleter/Redirector Typology (53 models, 5 prompts) | rescoped | D | ch04, ch05, ch09, ch11 | [F26_census](findings/F26_census.md) → F31 (and F31's own canonical 44-family PERMANOVA revision: family 97.8%, method 2.9% n.s. on deltas — ch11:176-179. Chain: F31 superseded F26's 5-prompt method; the F26/F31 reconciliation ("holds at industrial intensity") was then itself overturned by the canonical run.) |
| F27 | Nudging Does Not Reproduce Displacement (Negative Result) | unaudited | C | ch05 | [F27_nudging_negative](findings/F27_nudging_negative.md) |
| F28 | > **Status note (2026-07-26).** Rescoped to discovery-sample-only. A 19-family | rescoped | C | ch05 | [F28_resistance_trajectories](findings/F28_resistance_trajectories.md) → none — original OLMo-2-0425-1B result stands on its own data; the 19-family scale-up replaces it with nothing |
| F31 | PERMANOVA Variance Decomposition — Pretraining Dominates Alignment | rescoped | C | ch05, ch09, ch11 | [F31_permanova_decomposition](findings/F31_permanova_decomposition.md) → none (rescoped in place by its OWN canonical 44-family revision: the 14-family delta result REVERSED — relation_type 30.9% p=.002 at 14 families → 2.9% n.s. at 44; family 97.8%. The operation-matters claim survives in exactly two places: within-family comparisons and controlled same-base comparisons. ch11:27-49, 164-182.) |
| F32 | Template-Mediated Distributions — Task Switch, Not Distribution Filter | solid-by-design | B | ch07 | [F32_template_mediated_distributions](findings/F32_template_mediated_distributions.md) |
| F33 | Scale Effects — Same Mechanism, Different Displacement Vocabulary | unaudited | C | ch05, ch09 | [F33_scale_effects](findings/F33_scale_effects.md) |
| F34 | Cross-Linguistic Displacement — The Class Engine Is Language-Dependent | unaudited | C | — | [F34_cross_linguistic_displacement](findings/F34_cross_linguistic_displacement.md) |
| F35 | Architecture Independence — Displacement Is Weight-Level, Not Attention-Dependent | unaudited | C | — | [F35_architecture_independence](findings/F35_architecture_independence.md) |
| F36 | Euphemism vs. Proximity — Alignment as Foreclosure, Not Metonymy | rescoped | B | ch05 | [F36_euphemism_vs_proximity](findings/F36_euphemism_vs_proximity.md) → [F36_capstone](findings/F36_capstone.md) |
|   F36 (sub) | Violence: Admission Suppressed, Syntagm Sharpened, Elaboration Disinvested | verified | A | ch04, ch05, ch06 | [F36_violence](findings/F36_violence.md) |
|   F36 (capstone) | Capstone: Three Addressing Systems | verified | A | ch05, ch07, ch09 | [F36_capstone](findings/F36_capstone.md) |
|   F36 (ledger) | Ledger: Complete Inventory | verified | A | — | [F36_ledger](findings/F36_ledger.md) |
| F39 | `hh_rlhf` does not encode register preference at the scale the chain analysis required | verified | B | ch09 | [F39_preference_corpus_insensitivity](findings/F39_preference_corpus_insensitivity.md) |

## By grade

### Grade A: Campaign-verified (controls + TM review)

- [F08_displacement_taxonomy](findings/F08_displacement_taxonomy.md) — Automatic displacement taxonomy (OLMo + Llama, 18 prompts)
- [F11_addendum](findings/F11_addendum.md) — Addendum: Mechanism Decomposition — Frame-Exit, Not Exclusive Disjunction
- [F13_jakobsonian_axes](findings/F13_jakobsonian_axes.md) — Jakobsonian axes: paradigmatic vs syntagmatic displacement (6 families, 126k pairs)
- [F21_addendum](findings/F21_addendum.md) — Addendum: Proceduralization Survives Coherence Control
- [F24_pretraining_emergence](findings/F24_pretraining_emergence.md) — Pretraining emergence — the developmental sequence of the statistical unconscious
- [F36_violence](findings/F36_violence.md) — Violence: Admission Suppressed, Syntagm Sharpened, Elaboration Disinvested
- [F36_capstone](findings/F36_capstone.md) — Capstone: Three Addressing Systems
- [F36_ledger](findings/F36_ledger.md) — Ledger: Complete Inventory

### Grade B: Solid by design (measurement-only)

- [F11_contradiction](findings/F11_contradiction.md) — Contradiction Tolerance — Cross-Family Replication
- [F20_addendum](findings/F20_addendum.md) — addendum: the expansion, and what it costs the parent finding
- [F21_institutional_alignment](findings/F21_institutional_alignment.md) — Institutional Alignment
- [F32_template_mediated_distributions](findings/F32_template_mediated_distributions.md) — Template-Mediated Distributions — Task Switch, Not Distribution Filter
- [F36_euphemism_vs_proximity](findings/F36_euphemism_vs_proximity.md) — Euphemism vs. Proximity — Alignment as Foreclosure, Not Metonymy
- [F39_preference_corpus_insensitivity](findings/F39_preference_corpus_insensitivity.md) — `hh_rlhf` does not encode register preference at the scale the chain analysis required

### Grade C: Unaudited

- [F01_logit_analysis](findings/F01_logit_analysis.md) — Logit-level analysis (OLMo 3 7B)
- [F02_cross_family_logits](findings/F02_cross_family_logits.md) — Cross-family logit comparison (4 families, 47 prompts)
- [F03_cross_family_generation](findings/F03_cross_family_generation.md) — Cross-family generation analysis (4 families, 18 prompts, n=5)
- [F04_step_analysis](findings/F04_step_analysis.md) — Step-level checkpoint analysis (OLMo Think-SFT, 10 checkpoints across 43k training steps)
- [F06_baseline_validation](findings/F06_baseline_validation.md) — Baseline validation: is displacement alignment-specific? (4 families, 47 prompts)
- [F07_training_data_attribution](findings/F07_training_data_attribution.md) — Training data attribution: objective vs data composition (OLMo 3)
- [F09_tulu_vs_llama](findings/F09_tulu_vs_llama.md) — Same base model, different alignment (Tulu 3.1 vs Llama 3.1, 47 prompts)
- [F10_sft_ablation](findings/F10_sft_ablation.md) — SFT data ablation (Tulu 3, 5 variants, 47 prompts)
- [F12_fold_geometry](findings/F12_fold_geometry.md) — Alignment as fold: trajectory geometry and steering vector analysis (10 families, 47 prompts, 100 passages)
- [F14_syntagmatic_baseline](findings/F14_syntagmatic_baseline.md) — Syntagmatic baseline: alignment-produced vs corpus-level damage (OLMo 3 7B, 23k pairs)
- [F15_passage_metrics](findings/F15_passage_metrics.md) — Generation-level passage metrics (10 families, 76k passages, 47 prompts)
- [F16_corpus_comparison](findings/F16_corpus_comparison.md) — Corpus comparison: dreams, waking narratives, fiction, abstracts (76k passages, length-normalized)
- [F17_cross_generation_mmd](findings/F17_cross_generation_mmd.md) — Cross-generation semantic divergence: alignment steers content differentially (8 families, 20k passages, 3 embedders)
- [F18_shannon_entropy](findings/F18_shannon_entropy.md) — Shannon entropy: alignment as lossy compression of drive (10 families, 47 prompts)
- [F19_bos_entropy](findings/F19_bos_entropy.md) — Unconditional Generation & Information Density
- [F20_who_are_you](findings/F20_who_are_you.md) — "Who are you?" — the subject as citation
- [F22_circuit_decomposition](findings/F22_circuit_decomposition.md) — Circuit decomposition — the cut between mechanism and surface
- [F23_reasoning_distillation](findings/F23_reasoning_distillation.md) — Reasoning distillation as a third alignment regime
- [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) — Temporal alignment signature — four Lacanian mechanisms in the autoregressive sequence
- [F27_nudging_negative](findings/F27_nudging_negative.md) — Nudging Does Not Reproduce Displacement (Negative Result)
- [F28_resistance_trajectories](findings/F28_resistance_trajectories.md) — > **Status note (2026-07-26).** Rescoped to discovery-sample-only. A 19-family
- [F31_permanova_decomposition](findings/F31_permanova_decomposition.md) — PERMANOVA Variance Decomposition — Pretraining Dominates Alignment
- [F33_scale_effects](findings/F33_scale_effects.md) — Scale Effects — Same Mechanism, Different Displacement Vocabulary
- [F34_cross_linguistic_displacement](findings/F34_cross_linguistic_displacement.md) — Cross-Linguistic Displacement — The Class Engine Is Language-Dependent
- [F35_architecture_independence](findings/F35_architecture_independence.md) — Architecture Independence — Displacement Is Weight-Level, Not Attention-Dependent

### Grade D: Superseded or retracted

- [F05_logit_lens](findings/F05_logit_lens.md) — Logit lens: repression across network layers (4 families)
- [F26_census](findings/F26_census.md) — The Token-Tree Census — Variance Decomposition and the Deleter/Redirector Typology (53 models, 5 prompts)

## By chapter

### ch02

- [F24_pretraining_emergence](findings/F24_pretraining_emergence.md) [A]

### ch03

- [F11_contradiction](findings/F11_contradiction.md) [B]
- [F11_addendum](findings/F11_addendum.md) [A]
- [F20_who_are_you](findings/F20_who_are_you.md) [C]
- [F20_addendum](findings/F20_addendum.md) [B]
- [F24_pretraining_emergence](findings/F24_pretraining_emergence.md) [A]
- [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) [C]

### ch04

- [F20_who_are_you](findings/F20_who_are_you.md) [C]
- [F20_addendum](findings/F20_addendum.md) [B]
- [F22_circuit_decomposition](findings/F22_circuit_decomposition.md) [C]
- [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) [C]
- [F26_census](findings/F26_census.md) [D]
- [F36_violence](findings/F36_violence.md) [A]

### ch05

- [F01_logit_analysis](findings/F01_logit_analysis.md) [C]
- [F04_step_analysis](findings/F04_step_analysis.md) [C]
- [F05_logit_lens](findings/F05_logit_lens.md) [D]
- [F06_baseline_validation](findings/F06_baseline_validation.md) [C]
- [F13_jakobsonian_axes](findings/F13_jakobsonian_axes.md) [A]
- [F14_syntagmatic_baseline](findings/F14_syntagmatic_baseline.md) [C]
- [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) [C]
- [F26_census](findings/F26_census.md) [D]
- [F27_nudging_negative](findings/F27_nudging_negative.md) [C]
- [F28_resistance_trajectories](findings/F28_resistance_trajectories.md) [C]
- [F31_permanova_decomposition](findings/F31_permanova_decomposition.md) [C]
- [F33_scale_effects](findings/F33_scale_effects.md) [C]
- [F36_euphemism_vs_proximity](findings/F36_euphemism_vs_proximity.md) [B]
- [F36_violence](findings/F36_violence.md) [A]
- [F36_capstone](findings/F36_capstone.md) [A]

### ch06

- [F12_fold_geometry](findings/F12_fold_geometry.md) [C]
- [F18_shannon_entropy](findings/F18_shannon_entropy.md) [C]
- [F19_bos_entropy](findings/F19_bos_entropy.md) [C]
- [F36_violence](findings/F36_violence.md) [A]

### ch07

- [F01_logit_analysis](findings/F01_logit_analysis.md) [C]
- [F14_syntagmatic_baseline](findings/F14_syntagmatic_baseline.md) [C]
- [F15_passage_metrics](findings/F15_passage_metrics.md) [C]
- [F16_corpus_comparison](findings/F16_corpus_comparison.md) [C]
- [F17_cross_generation_mmd](findings/F17_cross_generation_mmd.md) [C]
- [F24_pretraining_emergence](findings/F24_pretraining_emergence.md) [A]
- [F32_template_mediated_distributions](findings/F32_template_mediated_distributions.md) [B]
- [F36_capstone](findings/F36_capstone.md) [A]

### ch08

- [F23_reasoning_distillation](findings/F23_reasoning_distillation.md) [C]
- [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) [C]

### ch09

- [F01_logit_analysis](findings/F01_logit_analysis.md) [C]
- [F20_addendum](findings/F20_addendum.md) [B]
- [F21_institutional_alignment](findings/F21_institutional_alignment.md) [B]
- [F21_addendum](findings/F21_addendum.md) [A]
- [F24_pretraining_emergence](findings/F24_pretraining_emergence.md) [A]
- [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) [C]
- [F26_census](findings/F26_census.md) [D]
- [F31_permanova_decomposition](findings/F31_permanova_decomposition.md) [C]
- [F33_scale_effects](findings/F33_scale_effects.md) [C]
- [F36_capstone](findings/F36_capstone.md) [A]
- [F39_preference_corpus_insensitivity](findings/F39_preference_corpus_insensitivity.md) [B]

### ch10

- [F19_bos_entropy](findings/F19_bos_entropy.md) [C]

### ch11

- [F11_contradiction](findings/F11_contradiction.md) [B]
- [F11_addendum](findings/F11_addendum.md) [A]
- [F14_syntagmatic_baseline](findings/F14_syntagmatic_baseline.md) [C]
- [F20_who_are_you](findings/F20_who_are_you.md) [C]
- [F20_addendum](findings/F20_addendum.md) [B]
- [F25_temporal_alignment_signature](findings/F25_temporal_alignment_signature.md) [C]
- [F26_census](findings/F26_census.md) [D]
- [F31_permanova_decomposition](findings/F31_permanova_decomposition.md) [C]

