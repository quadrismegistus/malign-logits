# F12: Alignment as fold: trajectory geometry and steering vector analysis (10 families, 47 prompts, 100 passages)

Two-part investigation of alignment's geometric signature, replicated across all 10 families with 47 prompts and 100 pre-generated passages per prompt.

**Part A: No universal geometric signature.** Feed identical text through base/SFT/DPO/RLVR, capture per-token hidden states, measure trajectory geometry. The direction of geometric change is family-specific:

| Family | layers | Δ local_drift | Δ mean_norm |
|---|---|---|---|
| Amber | 3 | +0.117 | −146 |
| OLMo | 4 | +0.052 | −5.6 |
| Tulu | 4 | +0.037 | −2.9 |
| Pythia | 3 | −0.001 | −3.3 |
| OLMo-tiny | 4 | −0.012 | +2.7 |
| Llama | 2 | −0.014 | −1.9 |
| SmolLM2 | 2 | +0.010 | +39 |

Norm change and drift change both flip direction between families. There is no universal "SFT pumps norms / DPO widens cone" division of labor — that was an OLMo-tiny-1B-specific finding. The geometric work of alignment is family-specific.

**Part B: Alignment is a fold, not a wall.** Per-prompt steering vectors (DPO − base hidden states) close 50–90% of the alignment gap on the prompt they were extracted from. Learned steering vectors (gradient descent, trained on half the prompts, evaluated on held-out half) show family-dependent generalization:

| Family | v2 self-closure | v2.6 held-out closure | Interpretation |
|---|---|---|---|
| Pythia | 89% | **77%** | Mostly fold (shallow community alignment) |
| Zephyr | 90% | 20% | Locally linear, globally diverse |
| Amber | 75% | **44%** | Strong-and-stereotyped (moralizing mode) |
| Qwen | 53% | **44%** | Mixed |
| Qwen-tiny | 50% | **37%** | Mixed |
| Llama | 64% | **30%** | Mostly wall |
| OLMo | 52% | **29%** | Mostly wall (industrial safety stack) |
| SmolLM2 | 53% | **21%** | Mostly wall |
| OLMo-tiny | 9% | **22%** | Mostly wall |

The original "94% wall" (OLMo-tiny, 8 prompts) was an artifact of insufficient training data. With 47 prompts, held-out closure ranges from 20% to 77%. **Foldability tracks alignment sophistication**: Pythia (1-epoch community fine-tune on Anthropic HH-RLHF, same data for SFT and DPO) is 77% fold. OLMo (industrial multi-source safety stack: CoCoNot, WildGuardMix, WildJailbreak, capability-delta DPO) is 29% fold.

**Part C: Fold dimensionality via SVD.** SVD of the (DPO − base) hidden-state difference matrix across all 47 prompts reveals the intrinsic dimensionality of the alignment shift. K_50 = number of orthogonal directions capturing 50% of the alignment variance.

| Family | K_50 | K_90 | top1 var% | v2.6 closure |
|---|---|---|---|---|
| **Pythia** | **2** | 28 | **48.6%** | 77% |
| OLMo-tiny | 3 | 26 | 38.6% | 20% |
| Amber | 3 | 32 | 44.4% | 44% |
| Zephyr | 5 | 31 | 30.7% | 20% |
| Qwen | 6 | 32 | 27.2% | 44% |
| Llama | 7 | 33 | 21.1% | 30% |
| Tulu | 7 | 33 | 28.3% | — |
| Qwen-tiny | 8 | 33 | 20.8% | 37% |
| SmolLM2 | 9 | 33 | 20.9% | 21% |
| **OLMo** | **13** | **36** | **13.1%** | 29% |

**Pythia’s alignment lives in 2 directions** (K_50=2, top singular value captures 49% of variance). OLMo’s lives in 13 (top value only 13%). The concentration of the alignment fold — not its tail (K_90 is ~30 everywhere, bounded by the 47-prompt sample) — is what varies by alignment regime.

**Fold concentration predicts steerability.** K_50 correlates with v2.6 held-out closure: the most concentrated alignments (Pythia K_50=2, Amber K_50=3) are the most steerable by a single learned vector. The most distributed (OLMo K_50=13) resists single-vector capture. This is the empirical content of the Lyotardian claim: the dimensionality of the fold indexes the structural depth of the theatrical apparatus.

**Lyotardian reframing.** Alignment is theatricalization of the libidinal band — a folding operation. Different corporate regimes fold the band at different angles and at different dimensionalities. Pythia (community fine-tune, 1-epoch, single dataset) is a 2-dimensional fold — almost a single crease. OLMo (industrial multi-source safety stack) is a 13-dimensional fold — a coordinated restructuring across many orthogonal directions. The “wall” from the original F12 was never a wall; it was a high-dimensional fold that a single steering vector could not ride.

Script: `malign trajectory`. Results in `data/trajectory_geometry_*.csv`, `data/intervention_*.csv`, `data/fold_rank_*.csv`. Figures in `figures/trajectory_geometry.*.png`, `figures/fold_rank.*.png`.
