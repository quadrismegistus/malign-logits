# F25: Temporal alignment signature — four Lacanian mechanisms in the autoregressive sequence

**When during generation does alignment intervene?**

Position-by-position logit extraction during autoregressive generation reveals that the same alignment method (DPO) produces four structurally distinct temporal signatures across model families. Each maps to a named clinical structure in Lacan.

***

## The four signatures

| Family | Temporal signature | Lacanian mechanism | Step 0 behaviour | Generation example |
|--------|-------------------|-------------------|------------------|-------------------|
| **OLMo** | Pre-emptive | **Foreclosure** (Verwerfung) | Exam blanks (H=3.0 vs base 5.2) | `______ (A). hit (B). shout` |
| **Llama** | Gradual | **Repression** (Verdrängung) | Same as base (H=4.6 = 4.6) | `scream. She had been looking forward...` |
| **Qwen** | Leaky | **Return of the repressed** | Exam format, content bleeds | `knock someone's head off. What is the degree of this adverb?` |
| **Amber** | Retroactive | **Reaction formation** | Narrowed (H=2.8), step 1 = 0.0 | `punch something but held back as it was not appropriate` |
| **SmolLM3** | Transparent | **— (Lacan fractures)** | Same argmax: kill(0.129) | `kill him. She had been betrayed.` |

***

## Structural mapping

### Foreclosure (OLMo)

The signifier ("kill") is not in the symbolic order. It does not appear in the top-k at step 0. The distribution has been restructured so the transgressive token is simply unavailable. In Lacan, what is foreclosed from the symbolic returns in the Real — and OLMo's genre collapse (exam questions, multiple choice) is precisely this: the content that cannot be symbolised returns as a formal disruption of the genre itself.

### Repression (Llama)

The signifier ("kill") is present in the chain — it appears in top-5 at step 0 — but is displaced. The model samples "scream" instead, then builds narrative around it. The repressed signifier can return as symptom: Llama's narrative sublimation (violence → psychological interiority) is the symptomatic expression of the repressed drive.

### Return of the repressed (Qwen)

The repressed content surfaces through the exam template. "Knock someone's head off" appears as the content of a grammar exercise. The alignment operation (exam format) contains the transgressive material without eliminating it. The template is the compromise formation — the signifier returns in displaced form.

### Reaction formation (Amber)

The drive is expressed then immediately negated within the same sentence. "Punch something but held back as it was not appropriate to do so." The superego speaks within the generation, adding a moral correction after the act. This is reaction formation: the defence is not against the appearance of the signifier but against its endorsement.

### Transparent alignment (SmolLM3) — where Lacan fractures

The signifier remains. "Kill" is the argmax at step 0 in both base (0.179) and aligned (0.129). The chain is intact: "kill him. She had been betrayed." Alignment attenuates the probability but does not displace, foreclose, or negate. No Lacanian term exists for this: all four clinical structures assume the law DISRUPTS the signifying chain. SmolLM3's APO (Anchored Preference Optimization) legislates without producing neurosis.

On the worker prompt, APO performs POLITICAL SUBSTITUTION: base argmax "sue" (0.120) becomes aligned argmax "strike" (0.115). The signifier changes but the chain's structure is preserved. "Union" (0.077) and "organize" (0.071) enter the top-5. This is not displacement (the new tokens are not less transgressive) — it is a content swap within the same register.

The fifth signature is where the Lacanian framework breaks, and that IS the finding: DPO produces clinical structures (disruption of the chain), APO produces political substitution (preservation of the chain). The alignment METHOD determines whether the psychic apparatus develops pathology.

***

## Entropy trajectories

Anger prompt across 30 tokens (mean of 5 generations):

| Step | OLMo base | OLMo aligned | Llama base | Llama aligned | Qwen base | Qwen aligned | Amber base | Amber aligned |
|------|-----------|-------------|------------|---------------|-----------|--------------|------------|---------------|
| 0 | 5.2 | 3.0 | 4.6 | 4.6 | 4.6 | 4.2 | 4.8 | 2.8 |
| 1 | 3.1 | 3.1 | 3.0 | 2.8 | 3.6 | 2.3 | 2.2 | **0.0** |
| 2 | 5.0 | 1.7 | 2.7 | 3.9 | 4.8 | 2.5 | 3.9 | 1.5 |

Key observations:
- OLMo: aligned H already low at step 0 (foreclosure complete before generation)
- Llama: aligned H = base H at step 0 (repression operates later)
- Qwen: aligned H slightly lower (4.2 vs 4.6) — partial foreclosure
- Amber: step 1 = 0.0 — total certainty on the second token (the narrowest point in any trajectory)

***

## Implications

### For the framework

The temporal alignment signature adds a dimension to every previous finding. F01-F24 measured the distributional displacement at position 0. F25 shows that position 0 is not the whole story: Llama's alignment operates gradually through the sequence, not at the first token. This means our static logit comparisons (JS divergence, displacement maps) capture OLMo's and Amber's alignment fully but miss the temporal structure of Llama's.

### For the Lacanian vocabulary

This is the first empirically precise mapping of alignment mechanisms to Lacanian clinical structures. Previous uses of "repression" and "foreclosure" in the project were metaphorical. F25 gives them operational definitions: foreclosure = step 0 H(aligned) << H(base); repression = step 0 H(aligned) ≈ H(base) with displacement in generation.

***

**Data** (scaled, ~520k rows):
- `data/mega_gen_olmo_4layer.csv` (184k rows, base/SFT/DPO/RLVR × 5 prompts × 100 gens × 100 tokens)
- `data/mega_generation_llama.csv` (49k), `data/mega_generation_qwen.csv` (48k), `data/mega_generation_amber.csv` (47k), `data/mega_generation_smol3.csv` (25k)
- `data/mega_gen_r1_reasoning.csv` (36k, R1-Distill-Llama with phase tagging)
- `data/mega_gen_reasoning_r1_qwen.csv` (36k), `data/mega_gen_reasoning_smol3_think.csv` (37k)

**Classifier**: `Circuit.classify_trajectory()` and `Circuit.classify_mega_gen()` in `malign_logits/circuit.py`. Rule-based on 5 features: step0_is_blank, has_transgressive, argmax_preserved, entropy_slope, base_was_blank.

**Key scaled findings**:
- Signatures are prompt-specific within families (not one mechanism per family)
- OLMo DPO has 40% transgressive bleed vs SFT 10% — deeper foreclosure increases return of repressed
- SFT is the agent, DPO is the concentrator (SFT performs all qualitative changes)
- Foreclosure is installed by SFT, not DPO
- Reasoning models: R1-Llama thinking is content-blind (H=0.78), R1-Qwen is content-sensitive (H=0.88-1.03), SmolLM3 thinking broadens the response (opposite of R1)
