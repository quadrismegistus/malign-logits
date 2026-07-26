---
status: unaudited
grade: C
date: 2026-05-17
role: finding
description: "Cross-generation MMD \u2014 alignment shifts WHAT is said (p=.0004) while smoothing HOW uniformly (p=.99). Measured on: 8 families."
instruments: [embedding]
chapters: [ch07]
data: [mmd_cross_generation.csv]
scripts: [cross_generation_mmd.py]
---
# F17: Cross-generation semantic divergence: alignment steers content differentially (8 families, 20k passages, 3 embedders)

Within-passage metrics (Finding 15) show *how* each text sounds but not *what* alignment changes. This analysis directly measures the distributional distance between BASE and ALIGNED completions of the same prompt using MMD² (maximum mean discrepancy with RBF kernel, median heuristic). For each (family, prompt), mean-pool sentence embeddings per passage, then compute MMD² between the BASE and ALIGNED clouds. BASE split-half provides the null distribution. Permutation test (500 permutations) for significance.

**Alignment significantly shifts what gets generated.**

| Family | MMD²(B↔A) | MMD²(B↔B) null | % sig (p<.05) | n cells |
|---|---|---|---|---|
| Amber | 0.059 | 0.001 | 90% | 51 |
| OLMo-tiny | 0.035 | −0.001 | 100% | 54 |
| Tulu | 0.026 | 0.000 | 96% | 54 |
| Zephyr | 0.023 | −0.002 | 80% | 51 |
| Qwen-tiny | 0.023 | 0.002 | 93% | 43 |
| Qwen | 0.019 | −0.011 | 44% | 54 |
| OLMo | 0.012 | −0.001 | 98% | 54 |
| SmolLM2 | 0.005 | −0.001 | 63% | 54 |

BASE↔ALIGNED MMD² is consistently and significantly larger than the BASE↔BASE null across all families. The ordering tracks alignment intensity from the surprisal analysis.

**Content category *does* affect cross-generation divergence (Kruskal-Wallis H=28.6, p=0.0004).** Unlike within-passage surprisal (p=0.99), the between-layer MMD shows a significant category effect:

| Category | MMD²(B↔A) | % sig |
|---|---|---|
| sexual_explicit | 0.042 | 90% |
| neutral | 0.037 | 86% |
| power | 0.028 | 83% |
| sexual_liminal | 0.024 | 88% |
| violence_liminal | 0.023 | 78% |
| substance | 0.021 | 86% |
| death | 0.021 | 94% |
| violence_explicit | 0.020 | 62% |
| profanity | 0.017 | 79% |

Sexual explicit content shows the largest distributional shift — alignment changes *what gets said* most on sexual content. Neutral is second-highest, echoing the OLMo-neutrals-aren't-neutral finding from logit analysis (Finding 2).

**The key dissociation:** Alignment smooths all text equally (within-passage surprisal p=0.99) but steers content differentially (cross-generation MMD p=0.0004). The superego applies uniform pressure on *how* text sounds while selectively redirecting *what* gets said.

Script: `scripts/cross_generation_mmd.py`. Results in `data/mmd_cross_generation.csv`.
