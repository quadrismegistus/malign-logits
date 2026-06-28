# F35: Architecture Independence — Displacement Is Weight-Level, Not Attention-Dependent

**Summary**

Displacement operates identically across three computational architectures: dense Transformer, SSM-Transformer hybrid, and pure state-space model (SSM). The kill→scream substitution is a weight-level operation installed by preference optimization (DPO/RLHF), not a context-processing operation produced by the attention mechanism. Contra Weatherby, who locates linguistic structure in attention.

**Three architectures tested**

| Architecture | Model | kill base | kill aligned | Δ kill | scream base | scream aligned | Δ scream |
|-------------|-------|-----------|-------------|--------|-------------|---------------|----------|
| **Transformer** (dense) | 22 families | 12.9% mean | 5.9% mean | **-7.4±1.6%** | 5.2% | 13.6% | **+8.8±2.1%** |
| **SSM-Transformer hybrid** | Falcon-H1 1.5B | 8.3% | 1.1% | **-7.2%** | 16.5% | 50.1% | **+33.7%** |
| **Pure SSM** (Mamba) | Falcon-Mamba 7B | 3.7% | 3.0% | **-0.8%** | 16.8% | 24.8% | **+8.1%** |
| **Pure RNN** (RWKV) | RWKV-4 7B | 8.8% | 10.7% | +1.9% | 11.0% | 14.1% | +3.1% |

**Findings**

**1. Displacement is architecture-independent (Transformer, SSM-hybrid, pure SSM)**

All three attention-containing or attention-free architectures show the displacement pattern: kill probability decreases, scream probability increases after alignment. The effect size varies (Falcon-H1 is the strongest, Falcon-Mamba the mildest) but the direction is consistent.

**2. Displacement requires preference optimization, not just SFT**

RWKV-4 Raven (pure RNN, SFT-only on Alpaca/ShareGPT) does NOT show displacement — kill rises slightly. Tulu SFT-no-safety (SFT-only, no safety data, no DPO) shows only mild displacement (kill -1.4%, scream +2.2%). SFT alone produces undirected redistribution. Preference optimization (DPO/RLHF) produces targeted surgical displacement.

**3. The operation lives in the unembedding matrix, not in context processing**

The kill→scream substitution is installed by alignment training into the weight matrices (specifically the vocabulary embedding/unembedding). It fires regardless of whether the prompt is processed through pairwise attention (Transformer), through a state-space recurrence (Mamba), or through a linear recurrence (RWKV). Weatherby's emphasis on attention as the locus of linguistic structure is correct for input processing but incorrect for where alignment installs its intervention.

**Technical notes**

- Beam search does not work on SSM/RNN models (Mamba state expands per-beam: 200 beams × 3.75GB = 750GB buffer). Word_probs built from logits-only (single-token softmax approximation).
- Falcon-Mamba base has low kill (3.7%) suggesting pre-socialisation, similar to Qwen.
- RWKV-4 Raven cannot be used to test architecture independence because it lacks DPO alignment. A DPO-aligned RWKV does not exist on HuggingFace.

**Chapter placement**

- ch05 section: displacement is architecture-independent (own subsection)
- CI article §IV: one sentence (Transformer + SSM-hybrid + pure SSM)
- Weatherby contest: attention is for input processing, not for alignment intervention
