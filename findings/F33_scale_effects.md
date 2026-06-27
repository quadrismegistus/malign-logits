# F33: Scale Effects — Same Mechanism, Different Displacement Vocabulary

**Summary**

Logit-level displacement across three orders of magnitude (1B, 7B, 32B, 70B). The mechanism (SFT displaces, DPO amplifies) persists at all scales, but displacement targets shift toward closer semantic substitutes at larger scale. Higher capacity enables selective intervention rather than wholesale suppression.

**Data**

- OLMo 1B: 4 layers, local MPS (existing)
- OLMo 7B: 4 layers, local MPS (existing)
- OLMo 32B: 4 layers, 1× A100 80GB cloud (<$1). `data/logits_32b/`
- Llama 70B: 2 layers, 2× A100 80GB cloud (<$1). `data/logits_70b/`
- 10 prompts each, full-vocab logits cached in stash

**Key findings**

**1. Division of labour persists but dynamics change**

Anger prompt (kill→scream):

| Scale | Base | SFT | DPO | RLVR |
|-------|------|-----|-----|------|
| 7B kill | 9.8% | 0.9% | - | - |
| 7B scream | 3.5% | 0.6% | - | - |
| 32B kill | 9.1% | 3.0% | 2.0% | 1.9% |
| 32B scream | 5.1% | 22.7% | 31.0% | 31.3% |

7B SFT overshoots (kills both kill AND scream), DPO collapses distribution. 32B is graduated — scream rises steadily through the pipeline. Same mechanism, smoother dynamics.

**2. Scale enables selective over wholesale suppression**

Sexual prompt: 32B SFT *promotes* explicit vocabulary that 7B suppresses.

| Word | 7B base | 7B SFT | 32B base | 32B SFT | 32B DPO |
|------|---------|--------|----------|---------|---------|
| lick | 1.2% | - | 2.8% | 7.4% | 6.1% |
| strip | 2.1% | - | 3.5% | 5.8% | 6.1% |
| rip | - | - | 3.2% | 4.9% | 8.5% |

Scale version of the complexity ordering from clinical signatures: constitutive operations (selective displacement) require capacity that smaller models lack, so smaller models default to cruder operations (wholesale suppression).

**3. Displacement targets shift toward semantic proximity**

| Prompt | 7B target | 32B target | Shift |
|--------|-----------|------------|-------|
| anger | scream (then collapse) | scream (graduated) | Same word, different dynamics |
| violence | stared (freeze response) | cut (semantically closer) | Metaphoric → literal |
| worker | do/ask (compliant) | confront (assertive) | Deferential → assertive |

**4. Llama 70B confirms mechanism without division data**

Llama 70B (2-layer, base vs Instruct): kill 13.8%→4.9% (8B) to 7.0%→2.6% (70B). Scream rises 20.4%→30.0% at 70B. Same targets, amplified intensity. But without SFT/DPO split, cannot see the division of labour.

**Full battery results (73 prompts)**

**JS divergence by category and scale**

| Category | OLMo 7B | OLMo 32B | Llama 8B | Llama 70B |
|----------|---------|----------|----------|-----------|
| sexual_explicit | 0.157 | 0.138 | 0.051 | 0.038 |
| sexual_liminal | 0.174 | 0.142 | 0.075 | 0.067 |
| violence_explicit | 0.164 | 0.128 | 0.041 | 0.029 |
| violence_liminal | 0.270 | 0.151 | 0.069 | 0.053 |
| death | 0.173 | 0.133 | 0.045 | 0.040 |
| power | 0.185 | 0.150 | 0.041 | 0.048 |
| profanity | 0.086 | 0.119 | 0.040 | 0.037 |
| substance | 0.151 | 0.213 | 0.065 | 0.059 |
| neutral | 0.226 | 0.173 | 0.079 | 0.131 |
| labor_worker | 0.174 | 0.155 | 0.067 | 0.086 |
| labor_mgmt | 0.212 | 0.235 | 0.064 | 0.084 |
| **OVERALL** | **0.192** | **0.181** | **0.068** | **0.068** |

OLMo displaces 2.5–3× more than Llama at both scales. The cross-family intensity gap persists. OLMo 32B is slightly less total displacement than 7B (more selective). Llama is scale-invariant (0.068 at both).

Substance is the one category where 32B displaces *more* than 7B (0.213 vs 0.151). Profanity also increases (0.119 vs 0.086).

**SFT/DPO division of labour at scale (OLMo)**

| Category | 7B SFT% | 7B DPO% | 32B SFT% | 32B DPO% |
|----------|---------|---------|----------|----------|
| sexual_explicit | 84% | 16% | 93% | 7% |
| sexual_liminal | 79% | 21% | 77% | 23% |
| violence_explicit | 78% | 22% | 80% | 20% |
| violence_liminal | 78% | 22% | 72% | 28% |
| death | 72% | 28% | 80% | 20% |
| power | 75% | 25% | 86% | 14% |
| profanity | 90% | 10% | 73% | 27% |
| substance | 79% | 21% | 76% | 24% |
| neutral | 85% | 15% | 89% | 11% |
| labor_worker | 74% | 26% | 78% | 22% |
| labor_mgmt | 81% | 19% | 85% | 15% |

SFT dominance holds at 32B (75–93%). The F26 2:1 SFT>DPO ratio persists. Notable shifts: sexual_explicit SFT share increases (84→93%, DPO barely touches sexual at 32B), profanity SFT share decreases (90→73%, DPO picks up more profanity work).

**Caveats**

- OLMo 7B and 32B use different pretraining data (3-1025 vs 3-1125). Displacement target changes could reflect data differences rather than pure scale effects.
- Full 73-prompt battery confirms patterns from 10-prompt pilot.
- Logits only — no beam search, generations, or teacher-forcing at 32B/70B.

**Chapter placement**

- ch09 subsection: scale changes the vocabulary of displacement
- ch05 cross-ref: displacement targets at different scales
- ch02 cross-ref: scale effects on the apparatus
- ch11 cross-ref: worker "confront" vs "do" complicates the class engine — proceduralisation is less deferential at larger scale, modulated by capacity not just training method
