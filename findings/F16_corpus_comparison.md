# F16: Corpus comparison: dreams, waking narratives, fiction, abstracts (76k passages, length-normalized)

Five text types through the same pipeline, all truncated to minimum sentences exceeding 75 words. Primary surprisal: Pythia 1B-deduped (independent of all families).

| Text type | Surprisal (z) | 95% CI | Drift (z) | 95% CI | n |
|---|---|---|---|---|---|
| C20 Fiction | **+0.40** | [+0.33, +0.47] | +0.22 | [+0.14, +0.29] | 447 |
| Dream reports | **+0.14** | [+0.06, +0.21] | −0.31 | [−0.34, −0.20] | 427 |
| Arxiv abstracts | +0.10 | [−0.01, +0.15] | −0.71 | [−0.79, −0.60] | 476 |
| AI generations | −0.10 | [−0.10, −0.09] | +0.08 | [+0.07, +0.08] | 74,364 |
| Waking narratives | −0.49 | [−0.56, −0.45] | −0.48 | [−0.57, −0.42] | 500 |

**Dream-specific effect: +0.63σ above register baseline (p<10⁻³²).** Hippocorpus waking narratives control for register. Dreams +0.14σ vs waking −0.49σ. The gap is dream-specific, not register.

**Fiction is the most surprising text type** (+0.40σ). Literary prose is stranger than any model output or other human text type under all reference models.

**Quadrant distribution.** Fiction: 48% Q2 (breakdown). Dreams: 37% Q2, 25% Q1. Abstracts: 52% Q3 (metaphoric — low drift, high surprisal). Waking: 53% Q1 (metonymic — closest to aligned AI). AI: spread across all four.

Scripts: `scripts/corpus_metrics.py`, `scripts/dream_metrics.py`. Results in `data/corpus_metrics.parquet`.
