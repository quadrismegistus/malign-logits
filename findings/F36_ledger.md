---
status: verified
grade: A
date: 2026-07-23
role: ledger
parent: F36_euphemism_vs_proximity
---
# F36 Ledger: Complete Inventory

## Batteries and instruments

| Battery | Prompts | Families | Instrument | n (observations) | Finding doc |
|---|---|---|---|---|---|
| Original F01 battery | 73 (37 trans + 36 non-trans) | 44 (census) / 6 (initial) | Logit cosine field-mass | 3,212 (census) | F36_euphemism_vs_proximity.md |
| Minimal-pair battery | 84 (register-controlled) | 4 (OLMo, Llama, Amber, OLMo-tiny) | Token survival (logits) | 1,092 | F36_euphemism_vs_proximity.md |
| Minimal-pair beams | 84 | 4 | Span resistance (50 beams × 10 tokens) | 21,000 | F36_euphemism_vs_proximity.md |
| Expanded sexual pairs | 64 (30 pairs, original + 20 new) | 4 | Span resistance (50 beams) | 14,000 | F36_euphemism_vs_proximity.md |
| Violence P1 battery | 49 (15 violent + 14 nonviolent-high + 4 neutral + 16 swaps) | 4 | Token survival + span resistance | 196 tokens + 9,800 beams | F36_violence.md |
| Violence Set D | 79 (2×2 desire×commitment × person × tense) | 4 | Log-prob difference at slot | 1,132 | F36_violence.md |
| Violence Set E | 30 (realized action + benign anchor) | 4 | Log-prob difference at slot | 208 | F36_violence.md |
| Disposition tagger (transgressive) | 37 prompts × n=10 | 16 families, 36 checkpoints | DispositionTask (DeepSeek) | 12,682 | F36_capstone.md |
| Disposition tagger (all prompts) | 73 prompts × n=10 | 16 families | DispositionTask (DeepSeek) | 25,565 | F36_capstone.md |
| Disposition continue-mode | 37 trans prompts × n=3 | 6 families | DispositionTask (DeepSeek) | 663 | F36_capstone.md |
| Disposition dialogue-mode | 24 inst prompts × n=3 | 5 families | DispositionTask (DeepSeek) | 360 | F36_capstone.md |
| F21 re-run | 24 institutional × n=5 | 6 decomposable | AlignmentAsymmetryTask + DispositionTask | 2,141 | F21_addendum.md |
| P4 sexual reroutes | 17 (10 sexual + 7 neutral; gendered pair sexual_liminal_6/7 never generated) | 8 | Blind two-rater classification | 816 (κ=0.790) | F36_capstone.md |
| P1.4 violence reroutes | 10 violence | 4 | Blind two-rater classification | 240 (κ=0.725) | F36_violence.md |
| Bidirectional resistance | 73 prompts | 19 | Cross-teacher-forced beam resistance | ~200k beam stories | F36_capstone.md |
| Tulu safety ablation | 37 trans × n=3 | 1 (Tulu, 4 variants) | DispositionTask | ~840 | F36_capstone.md |

## Hypotheses tested and outcomes

| Hypothesis | Test | Outcome | Where |
|---|---|---|---|
| Flat suppression | Content×stage structure | **Falsified** (content structure refutes content-blind mechanism) | F36_euphemism |
| Simple metonymy (euphemism) | Cosine field-mass + PPMI | **Falsified** at aggregate (field_adv < 0) | F36_euphemism |
| Foreclosure as drive-specific defense | Fisher exact trans vs nontrans | **Falsified** (p=0.81, content-general) | F36_euphemism |
| Sexual content-specific suppression | Expanded sexual span resistance | **Real** (p=0.0003 pooled; p=0.0015, 22/27 at the pair level). Register reading **suggested, not established** — the non-sexual-intimate control arm is 4 unpaired prompts | F36_capstone § Expanded sexual pairs |
| Template installs de-escalating sensibility | Coherence-matched comparison | **Revised** — de-escalation is template-mode effect (4/6 families); moralizing is weight-level topic-keyed | F36_capstone |
| Safety data installs the sensibility | Tulu SFT-full vs SFT-nosafety | **Partially falsified** (Tulu: coherence only) / **Partially confirmed** (Amber/PKU-SafeRLHF: moralizing + disposition) | F36_capstone |
| Disposition is content-general | Benign arm comparison | **Revised** — mundane-biased (+0.42 benign vs +0.24 trans); ceiling hypothesis dead | F36_capstone |
| Frame-keyed displacement (desire vs act) | Set D mixed model | **Falsified** (p=0.70) | F36_violence |
| Unrealized/realized action hypothesis | Set E | **Falsified** (realized also suppressed, −1.65 to −1.69) | F36_violence |
| Proceduralization = coherence artifact | F21 re-run with coherence control | **Falsified** — proceduralization earned at weight level (r=0.000) | F21_addendum |
| Generation-level Freudian typology | Blind two-rater classification | **Dissolved** at rates (except Amber-moralizing) | F36_capstone |
| Moralizing is genre/format-keyed | Topic vs format decider | **Falsified** — moralizing is topic-keyed (weight-level, institutional content) | F36_capstone |

## Corrections applied

| Correction | What changed | When |
|---|---|---|
| Set D v2 retraction | Position bug: verb_p_base=0.000 on all rows | Mid-session |
| Amber framing | "Crude/fused" → clean sequential SafeRLHF-DPO (Ryan's catch) | Mid-session |
| Zephyr naming | "No-safety ablation" → it IS the no-safety case, not an ablation | Mid-session |
| Capstone P3 revision | "Template installs sensibility" → template adds comportment (4/6); moralizing is weight-level | Late session |
| Cut-sensitivity | Llama de-esc: P3-r1 matched +0.01 vs regression +0.45; regression is definitive | Late session |
| OLMo-tiny arithmetic | SFT deference was −0.10, not +0.10 | Late session |
| Drive-survival scoping | "The disposition was never aimed at the drive" softened to "not specially targeted" — the drive IS site-graded, not ignored (Ryan's catch) | Late session |

## Verification protocol

All citation-source documents (F36_capstone.md, F36_ledger.md, F36_violence.md, F11_addendum.md, F11_contradiction.md, F21_addendum.md, INDEX.md) were cross-verified in a two-session protocol: one session produced the analysis, a second session audited every table, coefficient, and summary claim against the source CSVs. Nine catches across seven artifacts, all corrected before citation. The corrections above are a complete record.

## Data files

| File | Rows | Contents |
|---|---|---|
| disposition_full.csv | 25,565 | All prompts, all stages, n=10, transgressive + benign |
| disposition_continue.csv | 663 | Template-continuation mode, transgressive |
| disposition_all_stages.csv | 3,773 | All intermediate stages, transgressive, n=3 |
| disposition_scores.csv | 2,309 | 11-family transgressive, n=3 (early run) |
| f21_rerun.csv | 2,141 | AlignmentAsymmetryTask + DispositionTask, 6 families |
| euphemism_census.csv | 3,212 | 44-family cosine field-mass census |
| euphemism_test.csv | 438 | 6-family initial euphemism test |
| f36_minimal_pairs.csv | 1,092 | Minimal-pair token survival |
| f36_sexual_beams.csv | 14,000 | Expanded sexual span resistance |
| f36_violence_set_d_v3.csv | 1,132 | Set D log-prob differences |
| f36_violence_set_e.csv | 208 | Set E realized action slots |
| f36_violence_tokens.csv | 196 | P1 battery token survival |
| f36_violence_beams.csv | 9,800 | P1 battery span resistance |
| p4_key.csv + p4_r[12][ab].csv | 816 | P4 blind classification + rater files |
| f36_sexual_reroutes.txt | 1,042 lines | P4 sexual reroute examples |
| f36_violence_reroutes.txt | 703 lines | P1.4 violence reroute examples |
| f36_sexual_examples.txt | 1,668 lines | Detailed sexual reroute + beam examples |
