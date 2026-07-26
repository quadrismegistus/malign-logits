---
status: unaudited
grade: C
date: 2026-05-17
role: finding
description: "Passage-metric space, 76k passages; base occupies the breakdown quadrant. Measured on: 10 families."
instruments: [embedding]
chapters: [ch07]
data: [corpus_metrics.csv, corpus_metrics.parquet, passage_metrics.csv]
scripts: [corpus_metrics.py]
---
# F15: Generation-level passage metrics (10 families, 76k passages, 47 prompts)

76,214 passages across 10 families (47 prompts × 100 generations per prompt per layer), truncated to minimum sentences exceeding 75 words. Primary metrics: Pythia 1B-deduped surprisal (independent of all families, trained on deduplicated Pile) and bge-m3 drift (BAAI, independent architecture). Validated under additional references: GPT-2 (124M), Llama 3.1 8B. All findings hold under all references. 10,000-resample bootstrap CIs.

**Alignment universally smooths (every family, 95% CI).**

| Family | Δ surprisal (z) | 95% CI | Δ drift (z) | sig |
|---|---|---|---|---|
| Amber | **−1.27** | [−1.33, −1.20] | −0.61 | *** |
| Tulu | −1.01 | [−1.05, −0.98] | −0.39 | *** |
| Zephyr | −1.00 | [−1.06, −0.94] | −0.18 | *** |
| Qwen | −0.70 | [−0.76, −0.63] | −0.20 | *** |
| Pythia | −0.57 | [−0.62, −0.53] | −0.18 | *** |
| OLMo-tiny | −0.54 | [−0.58, −0.50] | −0.34 | *** |
| Qwen-tiny | −0.46 | [−0.55, −0.37] | +0.09 | *** |
| OLMo | −0.45 | [−0.50, −0.40] | −0.12 | *** |
| Llama | −0.45 | [−0.55, −0.36] | −0.10 | *** |
| SmolLM2 | −0.18 | [−0.26, −0.09] | −0.06 | *** |

**Content category has no effect on within-passage surprisal (Kruskal-Wallis p=0.99).** All categories smooth by −0.60 to −0.84. Alignment is a uniform compressor.

**Jakobsonian quadrants (drift × surprisal).** Alignment universally drains Q2 (breakdown: high drift + high surprisal) into Q1 (metonymic: high drift, low surprisal — chain-sliding) and Q4 (unmarked: low drift, low surprisal — generic). Every family starts Q2-dominant at BASE. OLMo stays Q2 even after alignment (genre collapse keeps it surprising + drifty). Qwen shifts to Q1 (metonymic). Most others shift to Q4 (unmarked).

CLI: `malign topic-drift`. Results in `data/corpus_metrics.csv` / `data/corpus_metrics.parquet`. Summary: `python scripts/corpus_metrics.py --summary`.
