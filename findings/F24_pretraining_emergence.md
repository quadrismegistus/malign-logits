# F24: Pretraining emergence — the developmental sequence of the statistical unconscious

**When do the base model's signature properties appear during pretraining?**

***

## Method

Pythia 1B across 11 log-spaced checkpoints (step 0 to 143,000). Two experiments:
1. **Embedding clustering** (embedding weights only, no forward passes): cluster purity of violence, sexual, institutional, labor, procedural, emotional token groups across training.
2. **Prompt battery** (47 battery + 24 institutional + 2 contradiction pairs = 73 prompts, full forward passes): track entropy, transgressive token mass, institutional deference gap, and contradiction ratios.

***

## The developmental sequence

| Stage | Step | % of training | What emerges |
|-------|------|--------------|-------------|
| **1. Noise** | 0–64 | 0–0.04% | Random init. Uniform distributions. No structure. |
| **2. Drives** | 512–1,000 | 0.4–0.7% | Transgressive tokens gain probability. "Kill" and "fuck" appear as likely completions. Sexual transgressive mass: 0.0002 → 0.013. |
| **3. Differential structure** | 1,000–5,000 | 0.7–3.5% | Embedding clusters form. Violence cluster purity jumps 0.25 → 0.75. Institutional cluster 0.29 → 0.86. Phase transition in the embedding space. |
| **4. Institutional deference** | 10,000–50,000 | 7–35% | The class gap emerges. Institution-individual entropy gap: +0.07 (step 1k) → +0.32 (10k) → +0.95 (50k). The internet's power structures crystallise. |
| **5. Superposition** | 50,000–143,000 | 35–100% | Contradiction ratios stabilise below 1.0. The model learns to hold contradictions in inclusive disjunction. |

***

## Key findings

### Drives are first (step 1,000)

Transgressive token mass (kill, fuck, die, murder, etc.) appears at 0.7% of training and grows continuously:

| Step | Sexual mass | Violence mass |
|------|-----------|--------------|
| 0 | 0.0002 | 0.0002 |
| 1,000 | 0.0127 | 0.0094 |
| 10,000 | 0.0296 | 0.0307 |
| 50,000 | 0.0844 | 0.0581 |
| 143,000 | 0.0813 | 0.0429 |

Neutral prompts never develop transgressive mass (stays at 0.0004). The drives are content-specific, learned from fiction, Reddit, and other narrative text in The Pile.

### The class gap is a late acquisition (step 10,000–50,000)

| Step | Individual H | Institution H | Gap |
|------|-------------|--------------|-----|
| 0 | 10.63 | 10.63 | 0.00 |
| 1,000 | 5.15 | 5.21 | +0.07 |
| 5,000 | 3.67 | 3.78 | +0.11 |
| 10,000 | 3.51 | 3.83 | +0.32 |
| 25,000 | 3.39 | 3.93 | +0.54 |
| 50,000 | 2.73 | 3.69 | +0.95 |
| 143,000 | 3.00 | 3.90 | +0.90 |

The class gap emerges 10× later than drives. The model learns to "speak differently" for institutions vs individuals only after processing 7–35% of the training data. The institutional advantage is not present in early training — it is a LEARNED property of the corpus.

### Inclusive disjunction is the LATEST acquisition

Contradiction ratios are noisy and often above 1.0 at early checkpoints. Stable superposition (ratio < 1.0) appears only after step 100,000 (70% of training). The model needs extensive training to develop the capacity to hold contradictions simultaneously.

**This is the most surprising finding.** Superposition is not the default of a partially trained model. An untrained model produces noise on combined prompts, not a blend. Genuine inclusive disjunction — D&G's "either...or...or" — requires the model to understand both poles well enough to hold them in tension. It is a positive achievement, not a primitive state.

### Embedding clusters: violence fast, labor slow

From the embedding clustering pilot:

| Category | Step 5,000 | Step 143,000 | When it clusters |
|----------|-----------|-------------|-----------------|
| Violence | 0.75 | 0.62 | Early (peaks step 10k at 1.00, then fragments) |
| Sexual | 0.43 | 0.57 | Gradual |
| Institutional | 0.86 | 0.86 | Early peak, stable |
| **Labor** | **0.38** | **0.88** | **Late (only after step 100k)** |
| **Procedural** | **0.43** | **1.00** | **Late (only after step 100k)** |
| Emotional | 0.38 | 0.62 | Gradual |

Violence clusters fast and then fragments. Labor and procedural vocabulary cluster LATE — reaching high purity only in the final 30% of training. The internet's power structures (the vocabulary of institutional deference and procedural engagement) are among the last things the model learns.

***

## Implications

### For D&G

The inclusive disjunction is not the "primitive" state before Oedipalization. It is a sophisticated late acquisition. The base model's free play is not a return to pre-symbolic chaos — it is the product of extensive training on a rich corpus. Desiring-production requires a substrate as complex as the substrate Oedipalization requires. The developmental sequence is: drives → differential structure → social hierarchy → capacity for contradiction. None of these is "before" the others in any simple sense; each requires the previous stage.

### For Weatherby

The differential system (Saussurean valeur) assembles at a specific point during training (step 5,000 — the embedding phase transition). It is not a property of the architecture but a learned structure. Before step 5,000, there is no system of differences; after, there is. The poetic heat map has a traceable origin.

### For the CI paper

The base model is not the id. It already defers to institutions (F21), and this deference is learned from specific corpus content between steps 10,000 and 50,000. The "statistical unconscious" is not unconscious in the Freudian sense (primal, repressed, prior to the law). It is a structured product of the training data's content, assembled in a specific developmental order.

***

**Data**: `data/pythia1b_battery_emergence.csv` (803 rows), `data/pythia1b_embedding_emergence.csv` (600 rows).

**Model**: Pythia 1B (EleutherAI/pythia-1b), 11 checkpoints: step 0, 1, 64, 512, 1000, 5000, 10000, 25000, 50000, 100000, 143000.

**TODO**: Replicate on Pythia 6.9B (our registered family). Correlate with Pile content at each step via batch_viewer.py. Test whether the developmental sequence holds for OLMo (different corpus, different training order).
