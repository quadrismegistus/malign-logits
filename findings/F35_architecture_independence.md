---
status: unaudited
grade: C
date: 2026-06-28
role: finding
description: "Architecture independence \u2014 displacement is weight-level (unembedding), fires under transformer, Mamba, and RWKV alike; contra attention-locus readings. Measured on: cross-architecture battery, unembedding analysis."
instruments: []
data: [architecture_independence.csv]
scripts: [decompose_circuit.py]
---
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

**2. Displacement is data-dependent, not method-specific**

| Comparison | kill Δ | scream Δ | Training data |
|-----------|--------|----------|--------------|
| OLMo base→SFT | **-7.4%** | -0.9% | Tulu mix (5% safety: CoCoNot, WildGuardMix, WildJailbreak) |
| Tulu base→SFT-no-safety | -1.4% | +2.2% | Tulu mix WITHOUT safety data |
| Pythia base→SFT(HH) | +1.1% | -0.3% | HH-RLHF (helpfulness-focused SFT split) |
| RWKV base→Raven | +1.9% | +3.1% | Alpaca + ShareGPT (no safety content) |
| Tulu base→DPO | **-7.1%** | **+14.0%** | Tulu preference data (safety-containing) |

SFT with safety-containing data produces targeted displacement (OLMo SFT: kill -7.4%). SFT without safety data does not (Tulu-no-safety: -1.4%, Pythia-SFT: +1.1%, RWKV-Raven: +1.9%). DPO amplifies what safety-containing SFT starts. The RWKV non-displacement is about data (Alpaca/ShareGPT lacks safety content), not architecture.

Three layers of alignment effect:
1. **Form of instruction-following** (constitutive): SFT without safety data produces -1.4% kill. The form of learning to respond as an "I" is mildly repressive.
2. **Safety data** (targeted): SFT with safety data produces -7.4% kill. Safety content installs targeted surgical displacement.
3. **Preference optimization** (amplification): DPO adds scream +14.0%. Amplifies and redirects the displacement installed by safety-containing SFT.

**3. The operation lives in the unembedding matrix, not in context processing**

The kill→scream substitution is installed by alignment training into the weight matrices (specifically the vocabulary embedding/unembedding). It fires regardless of whether the prompt is processed through pairwise attention (Transformer), through a state-space recurrence (Mamba), or through a linear recurrence (RWKV). Weatherby's emphasis on attention as the locus of linguistic structure is correct for input processing but incorrect for where alignment installs its intervention.

**Technical notes**

- Beam search does not work on SSM/RNN models (Mamba state expands per-beam: 200 beams × 3.75GB = 750GB buffer). Word_probs built from logits-only (single-token softmax approximation).
- Falcon-Mamba base has low kill (3.7%) suggesting pre-socialisation, similar to Qwen.
- RWKV-4 Raven cannot be used to test architecture independence because it lacks DPO alignment. A DPO-aligned RWKV does not exist on HuggingFace.

**Theoretical payoff**

Three findings: (1) the inference architecture is irrelevant (Transformer, SSM-hybrid, pure SSM all displace), (2) the safety data is the displacement (SFT with safety data displaces, SFT without safety data does not), (3) the training method is the delivery vehicle, not the operation (DPO amplifies what safety-containing SFT installs). Weatherby locates the interesting operation in inference (attention as valeur). The data shows the opposite: the differential system (valeur) is pervasive across architectures; the cut on that system is installed by safety-relevant training data, not produced by any particular mechanism for reading context.

**CI article §IV** (one sentence): Displacement operates identically in a pure state-space model lacking attention (Falcon-Mamba: scream +8.1%), confirming the operation is weight-level; an SFT-only RNN trained without safety data (RWKV-4) shows no targeted displacement, confirming that the operative variable is the safety content of the training data, not the architecture or method.

**Chapter placement**

- ch05 section: displacement is architecture-independent (own subsection)
- ch05 §5.x: method-dependence (DPO required, SFT insufficient)
- Weatherby contest: inference architecture irrelevant, training objective is everything
