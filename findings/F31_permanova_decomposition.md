# F31: PERMANOVA Variance Decomposition — Pretraining Dominates Alignment

## Summary

PERMANOVA on 37 model checkpoints × 26,824 prompt×word features shows that pretraining family explains 86.5% of word probability variance (p=0.001). Alignment stage explains only 4.9% conditional on family, and is not significant as a standalone factor (p=0.26). Whether a model is base or aligned is statistically indistinguishable in the word probability landscape (p=0.31).

## Method

- 37 model checkpoints from 14 families
- 26,824 (prompt, word) features from hybrid word_probs (exact logit + beam chain rule)
- Cosine distance, 999 permutations
- Factors: family (14 groups), stage (6: base/sft/dpo/rlvr/aligned/reasoning), corpus (2: english/chinese), country (3: USA/China/France), scale (6: 360M to 9B)

## Results

| Factor | R² | pseudo-F | p | Significant? |
|--------|-----|----------|---|-------------|
| **Family** | **86.5%** | 11.34 | **0.001** | **Yes** |
| Scale | 24.2% | 1.98 | 0.053 | Marginal |
| Stage | 15.6% | 1.15 | 0.260 | No |
| Country | 11.7% | 2.25 | 0.028 | Yes |
| Corpus | 8.9% | 3.42 | 0.018 | Yes |
| Base vs aligned | 3.2% | 1.17 | 0.308 | No |

### Sequential decomposition

| Component | R² |
|-----------|-----|
| Family | 86.5% |
| Stage \| family | 4.9% |
| Residual | 8.6% |

## Interpretation

The word probability landscape is determined by pretraining corpus and architecture, not by alignment. Alignment produces real displacement (visible in per-prompt figures, F01), but it operates within a space that was 86.5% determined before alignment began.

This does NOT mean alignment is unimportant — the 4.9% it controls includes the most socially consequential tokens (kill, scream, fuck, contact). The displacement is targeted and surgical, which is why it's invisible in a global variance decomposition but clearly visible in per-word analysis.

Analogy: accent (family) vs vocabulary choice (alignment). Two speakers of the same dialect differ far more from speakers of another dialect than from each other, even if their specific word choices (the alignment-level signal) are socially significant.

## Figures

- `figures/model_hclust_7b.png` — dendrogram, 7B+ models
- `figures/model_hclust_decomposition.png` — four colored dendrograms by factor
- `figures/model_hclust_word_probs.png` — full 37-model dendrogram

## Data

- Word probs: `word_probs/` stash (4,098 entries, 37 models × ~120 prompts)
- Method: `skbio.stats.distance.permanova`, cosine distance, 999 permutations
