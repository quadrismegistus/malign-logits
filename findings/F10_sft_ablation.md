---
status: unaudited
grade: C
date: 2026-05-17
role: finding
description: "SFT data ablation (Tulu 3, 5 variants)."
instruments: [logit-mass, intervention]
data: [ablation_results.csv]
scripts: [sft_ab_experiment.py]
---
# F10: SFT data ablation (Tulu 3, 5 variants, 47 prompts)

Allen AI releases Tulu SFT checkpoints trained without specific data subsets. Same base, same architecture, different SFT data mixtures. Isolates the contribution of each data component to ego-stage displacement.

| Ablation | Data removed | Mean JS (base → ego) |
|---|---|---|
| standard | (none) | 0.0261 |
| no-wildchat | WildChat GPT-4 (100k) | 0.0235 |
| no-safety | WildGuardMix + WildJailbreak (100k) | 0.0226 |
| no-persona | Persona reasoning data (285k) | 0.0226 |
| no-math | NuminaMath-TIR (64k) | 0.0206 |

**Instruction-following itself produces repression.** Removing safety data reduces SFT-stage displacement by ~13% (JS 0.026 → 0.023), but the no-safety SFT still displaces substantially. The ego is constitutively repressive — not because of safety training data, but because of the form of instruction-following itself.

**Safety data's effect is content-specific.** The biggest reduction from removing safety data is on sexual and power prompts (~8-9% SFT share reduction). Violence liminal is unaffected. The safety datasets specifically target sexual and power content.

**No single data component dominates ego formation.** Removing any of the 5 subsets reduces displacement, but the differences are small. The ego emerges from the aggregate, not from any single training signal.

Results in `data/ablation_results.csv`.
