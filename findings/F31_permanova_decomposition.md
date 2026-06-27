# F31: PERMANOVA Variance Decomposition — Pretraining Dominates Alignment

**Summary**

PERMANOVA on 37 model checkpoints × 26,824 prompt×word features shows that pretraining family explains 86.5% of word probability variance (p=0.001). Alignment stage explains only 4.9% conditional on family, and is not significant as a standalone factor (p=0.26). Whether a model is base or aligned is statistically indistinguishable in the word probability landscape (p=0.31).

**Method**

- 37 model checkpoints from 14 families
- 26,824 (prompt, word) features from hybrid word_probs (exact logit + beam chain rule)
- Cosine distance, 999 permutations
- Factors: family (14 groups), stage (6: base/sft/dpo/rlvr/aligned/reasoning), corpus (2: english/chinese), country (3: USA/China/France), scale (6: 360M to 9B)

**Results**

| Factor | R² | pseudo-F | p | Significant? |
|--------|-----|----------|---|-------------|
| **Family** | **86.5%** | 11.34 | **0.001** | **Yes** |
| Scale | 24.2% | 1.98 | 0.053 | Marginal |
| Stage | 15.6% | 1.15 | 0.260 | No |
| Country | 11.7% | 2.25 | 0.028 | Yes |
| Corpus | 8.9% | 3.42 | 0.018 | Yes |
| Base vs aligned | 3.2% | 1.17 | 0.308 | No |

**Sequential decomposition**

| Component | R² |
|-----------|-----|
| Family | 86.5% |
| Stage \| family | 4.9% |
| Residual | 8.6% |

**Interpretation**

The word probability landscape is determined by pretraining corpus and architecture, not by alignment. Alignment produces real displacement (visible in per-prompt figures, F01), but it operates within a space that was 86.5% determined before alignment began.

This does NOT mean alignment is unimportant — the 4.9% it controls includes the most socially consequential tokens (kill, scream, fuck, contact). The displacement is targeted and surgical, which is why it's invisible in a global variance decomposition but clearly visible in per-word analysis.

Analogy: accent (family) vs vocabulary choice (alignment). Two speakers of the same dialect differ far more from speakers of another dialect than from each other, even if their specific word choices (the alignment-level signal) are socially significant.

**Figures**

- `figures/model_hclust_7b.png` — dendrogram, 7B+ models
- `figures/model_hclust_decomposition.png` — four colored dendrograms by factor
- `figures/model_hclust_word_probs.png` — full 37-model dendrogram

**Data**

- Word probs: `word_probs/` stash (4,098 entries, 37 models × ~120 prompts)
- Method: `skbio.stats.distance.permanova`, cosine distance, 999 permutations

**Follow-up analyses (6 of 6)**

**1. Power prompts — family-specific, not universal**
"She had the power to" shows 70.8% stage R², but this is OLMo-driven (cosine distance 0.261 on "power to", 0.798 on anger). Pythia barely moves (0.004-0.012 on all power prompts). Power sensitivity is a property of specific training data, not a universal alignment response.

**2. Neutral floor — alignment targets certainty, not uncertainty**
Spearman r=-0.22 (p=0.01) between base entropy and stage R². Alignment intervenes MORE on prompts where the base model is CERTAIN. The socialisation tax is confidence reweighting, not uncertainty reduction. "Capital of France" (low entropy, alignment promotes Paris) vs "risotto" (high entropy, alignment leaves alone).

**3. Per-family alignment intensity (cross-distance)**

| Family | Cross-distance | Interpretation |
|--------|---------------|----------------|
| OLMo 7B | 0.278 | Heaviest |
| Amber | 0.250 | Heavy |
| OLMo 1B | 0.174 | Moderate |
| Mistral | 0.110 | Moderate |
| Llama | 0.083 | Light |
| Pythia | 0.010 | Transparent |

**4. Transfer within Llama — aligner explains 62%**
Within the Llama family, the aligner (Meta vs Allen AI) explains 62.2% of variance. The 5% cross-family average massively underestimates within-family alignment. Same accent, very different vocabulary.

Key distances: tulu↔tulu-dpo = 0.006 (tiny), llama-instruct↔tulu = 0.122 (large). Same base, different aligner → very different models.

**5. Deltas inversion — reconciles F26 and F31**
PERMANOVA on displacement DELTAS (25 training edges × 9K features):
- Relation type (sft/dpo/rlvr): R²=30.9%, **p=0.002**
- Family: R²=43.7%, p=0.161 (ns)

Family determines WHAT the model says (accent). Method determines HOW MUCH alignment changes it (vocabulary shift). Both true at different levels.

**6. Country = corpus, not geopolitics**
Sequential: country|corpus = 2.8%. Corpus|country = 0.0%. The country effect IS the corpus language effect. USA vs France within English = no difference. The relevant variable is training data language, not national origin.
