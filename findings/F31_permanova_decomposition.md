# F31: PERMANOVA Variance Decomposition — Pretraining Dominates Alignment

**Summary**

PERMANOVA on 112 model checkpoints from 44 families shows that pretraining family explains 78.9% of word probability variance (p=0.001). Alignment stage explains only 4.9% and is not significant (p=0.067). Whether a model is base or aligned explains just 2.4% (p=0.044, marginal). Replicated from original 14-family finding to full 44-family census with consistent results.

**Method**

- 112 model checkpoints from 44 families (updated from original 37 checkpoints / 14 families)
- 670 non-zero-variance features from 19 prompts × 63 target words
- Cosine distance, 999 permutations

**Results (44-family replication)**

| Factor | R² | pseudo-F | p | Significant? |
|--------|-----|----------|---|-------------|
| **Family** | **78.9%** | 5.91 | **0.001** | **Yes** |
| Stage | 4.9% | 1.87 | 0.067 | No |
| Base vs aligned | 2.4% | 2.69 | 0.044 | Marginal |

**Original results (14 families, 37 models)**

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
| Family | 78.9% (was 86.5%) |
| Stage \| family | 4.9% (unchanged) |
| Residual | 16.2% (was 8.6%) |

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

**3. Per-family alignment intensity (cosine distance base→aligned)**

At 44 families, intensity spans 250×. Top and bottom 5:

| Family | Cross-distance | Interpretation |
|--------|---------------|----------------|
| DeepSeek 7B | 0.745 | Heaviest (by far) |
| Falcon-H1 7B | 0.512 | Very heavy |
| OLMo 32B | 0.463 | Heavy |
| Amber | 0.353 | Heavy |
| OLMo 7B | 0.274 | Moderate-heavy |
| ... | ... | ... |
| TinyLlama | 0.030 | Near-transparent |
| Yi | 0.027 | Near-transparent |
| Pythia | 0.007 | Transparent |
| Archangel (all 4) | 0.003 | Transparent |

**4. Transfer within Llama — aligner explains 62%**
Within the Llama family, the aligner (Meta vs Allen AI) explains 62.2% of variance. The 5% cross-family average massively underestimates within-family alignment. Same accent, very different vocabulary.

Key distances: tulu↔tulu-dpo = 0.006 (tiny), llama-instruct↔tulu = 0.122 (large). Same base, different aligner → very different models.

**5. Deltas — family determines even the change**

*Original (14 families, 25 edges):* Relation type (sft/dpo/rlvr) R²=30.9% (p=0.002), family R²=43.7% (ns). Interpretation: method determines how much alignment changes.

*Updated (44 families, 68 edges):* Family R²=97.8% (p=0.001), relation type R²=2.9% (ns). The original finding was underpowered. At census scale, even the *displacement pattern* — not just the starting point — is family-specific. The sft/dpo distinction visible at 14 families was driven by a few dominant families (OLMo's heavy SFT), not a universal property of the training step type.

Archangel provides the cleanest test: same base + SFT, four different preference methods (DPO/KTO/PPO/SLIC), all producing near-identical deltas (cosine distance 0.003). The alignment method doesn't matter; the training data does.

**6. Country = corpus, not geopolitics**
Sequential: country|corpus = 2.8%. Corpus|country = 0.0%. The country effect IS the corpus language effect. USA vs France within English = no difference. The relevant variable is training data language, not national origin.
