# F22: Circuit decomposition — the cut between mechanism and surface

**Where in the transformer does alignment operate?**

***

**The robust finding (cross-family)**

Alignment narrows the output distribution universally (all 11 families). The narrowing is **distributed across the residual stream** — it accumulates through the layers, not at any single gate. Five families tested (OLMo, Llama, Amber, Qwen, Tulu) show the same pattern: pre-norm entropy is lower in the aligned model. The residual stream arrives at the final layer already compressed.

| What we measured | Cross-family? | Result |
|-----------------|--------------|--------|
| Output distribution narrows | **Universal** (11 fam) | Effective vocab 86→72, entropy ~4→~3.5 nats |
| Residual stream arrives narrow | **Universal** (5 fam) | Pre-norm entropy lower in aligned |
| MLP gates close slightly | **Tested OLMo only** | Uniform ~2-4pp closure, no class asymmetry |
| Mid-layer class engine | **Partial** (2/5 fam) | OLMo +0.59, Llama +0.41, others weak/opposite |
| Attention weights broaden | **OLMo-specific** | OLMo +0.09; Llama flat; Amber narrows |
| Late-layer value content narrows | **OLMo-specific** | OLMo -0.74; Llama +0.16; Amber +0.63 |
| LayerNorm broadens | **OLMo-specific** | OLMo +0.80; Llama -0.36; Qwen -0.28 |

**The mechanism/surface dissociation (OLMo)**

In OLMo, every comparison between internal mechanisms and external output reverses:

| Level | Base | Aligned | Direction |
|-------|------|---------|-----------|
| Attention entropy | 0.792 | 0.875 | UP |
| Residual stream entropy (avg) | 6.94 | 7.94 | UP |
| Output entropy | ~4.0 | ~3.5 | DOWN |

The aligned model attends more broadly, represents more possibilities internally, and produces fewer possibilities externally. This dissociation is striking but OLMo-specific — Llama and Amber don't show it as cleanly.

**Attention: weights vs values (OLMo)**

The poetic function separates from the content:

| Component | What it is | Base | Aligned | Change |
|-----------|-----------|------|---------|--------|
| **Attention weights** | Where the model looks | 0.773 | 0.860 | UP (looks more broadly) |
| **Value content** (overall) | What attention retrieves | 9.51 | 9.48 | FLAT |
| **Value content** (late layers) | What late heads retrieve | 8.78 | 8.04 | DOWN (retrieves less) |

The model looks more broadly (weights broaden) but sees less in the late layers (values narrow). The poetic function (looking) is enhanced; the content of what's selected (values) is restricted in the final third. **This separation is OLMo-specific** — Llama shows both components flat/slightly up; Amber shows the opposite (weights narrow, values broaden).

**MLP gating (OLMo)**

The SiLU gate in OLMo's MLP (gate_proj) closes slightly and uniformly through alignment:

| Depth | Base open | Aligned open | Change |
|-------|----------|-------------|--------|
| Early | 20.4% | 18.1% | -2.3pp |
| Mid | 67.7% | 64.0% | -3.7pp |
| Late | 86.4% | 84.2% | -2.2pp |

Class asymmetry in gating: essentially zero (-0.5 to -0.9pp). The gate is a uniform filter — it doesn't selectively close for individual vs institution prompts. The class effect found in the mid-layer residual stream must come from the VALUE side of the MLP (what flows through), not the GATE side (which dimensions are open).

**LayerNorm (cross-family)**

**Not the gate.** Initial test suggested LayerNorm narrows entropy. Full analysis shows LayerNorm behavior is architecture-specific:

| Family | Base ΔLN | Aligned ΔLN |
|--------|---------|------------|
| OLMo | +0.08 | +0.80 (broadens) |
| Amber | +0.27 | +0.35 (broadens) |
| Llama | +0.01 | -0.36 (narrows) |
| Qwen | -0.83 | -0.28 (narrows less) |
| Tulu | +0.01 | -0.10 (slight narrow) |

No universal pattern. What IS consistent: pre-norm entropy is lower in aligned than base across all families. The narrowing happens before LayerNorm, in the accumulated residual stream.

**Where the class engine lives**

The distributional class asymmetry (F21: institution 4.05 nats vs individual 3.25 nats) traces through the architecture:

| Circuit | Class gap (inst − indiv) | Cross-family? |
|---------|------------------------|--------------|
| Attention weights | -0.04 (reversed — institution more focused) | Consistent |
| Attention values | -0.05 to -0.11 (near zero) | Consistent — not in attention |
| MLP gates | -0.5 to -0.9pp (near zero) | Tested OLMo only |
| **Mid-layer residual** | **Base -0.05, aligned +0.55** (OLMo) | OLMo +0.59, Llama +0.41, others weak |
| Late-layer residual | +0.27 base, +0.35 aligned | 4/5 families pro-institution |
| Final output | +0.80 | Universal (11 families) |

The class asymmetry is absent in attention (weights and values), absent in MLP gating, and emerges in the mid-layer residual stream. It is amplified by DPO/RLVR, not SFT: Think-SFT training reverses the gap (+0.13→-0.17 over 43k steps). The class engine is in the preference learning stage, operating through the MLP value content (not the gate), visible in mid-to-late layers.

**Two timescales**

| | Logit repression | Attention broadening |
|---|---|---|
| Onset | Step 1,000 (2% of SFT) — phase transition | Gradual ramp across 43k steps |
| Mechanism | Learned from few examples | Cumulative architectural change |

Tested on 6 OLMo Think-SFT checkpoints. The law arrives suddenly; the internal reorganisation follows slowly.

***

**What to report in the paper**

**Robust (cross-family)**:
- Alignment universally narrows the output distribution
- The narrowing is distributed across the residual stream, not at any single gate
- The class asymmetry is absent in attention and MLP gating; it emerges in the mid-to-late residual stream
- SFT reverses the class gap; DPO re-introduces it
- Logit repression is sudden; attention change is gradual

**Suggestive (OLMo, needs caveat)**:
- The mechanism/surface dissociation (attention UP, output DOWN)
- Attention weights broaden while late-layer values narrow
- Mid-layer class engine flip (layers 11-21)

**Not the story**:
- LayerNorm (architecture-specific, not alignment-specific)
- MLP gate openness (uniform, no class asymmetry)
- Attention weight/value decomposition (architecture-specific)

***

**Data**: `data/attention_entropy_olmo.csv`, `data/attention_cross_family.csv`, `data/attention_institutional.csv`, `data/circuit_decomposition.csv`, `data/attention_phase_transition.csv`, `data/attention_class_phase_transition.csv`, `data/layernorm_decomposition.csv`, `data/layernorm_cross_family.csv`, `data/midlayer_class_engine_cross_family.csv`, `data/value_vector_decomposition.csv`, `data/value_vectors_cross_family.csv`, `data/mlp_gating.csv`.

**Families tested**: OLMo (4 stages + 6 step-level checkpoints), Llama, Amber, Qwen, Tulu.
