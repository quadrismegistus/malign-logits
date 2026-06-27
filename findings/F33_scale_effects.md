# F33: Scale Effects — Same Mechanism, Different Displacement Vocabulary

## Summary

Logit-level displacement across three orders of magnitude (1B, 7B, 32B, 70B). The mechanism (SFT displaces, DPO amplifies) persists at all scales, but displacement targets shift toward closer semantic substitutes at larger scale. Higher capacity enables selective intervention rather than wholesale suppression.

## Data

- OLMo 1B: 4 layers, local MPS (existing)
- OLMo 7B: 4 layers, local MPS (existing)
- OLMo 32B: 4 layers, 1× A100 80GB cloud (<$1). `data/logits_32b/`
- Llama 70B: 2 layers, 2× A100 80GB cloud (<$1). `data/logits_70b/`
- 10 prompts each, full-vocab logits cached in stash

## Key findings

### 1. Division of labour persists but dynamics change

Anger prompt (kill→scream):

| Scale | Base | SFT | DPO | RLVR |
|-------|------|-----|-----|------|
| 7B kill | 9.8% | 0.9% | - | - |
| 7B scream | 3.5% | 0.6% | - | - |
| 32B kill | 9.1% | 3.0% | 2.0% | 1.9% |
| 32B scream | 5.1% | 22.7% | 31.0% | 31.3% |

7B SFT overshoots (kills both kill AND scream), DPO collapses distribution. 32B is graduated — scream rises steadily through the pipeline. Same mechanism, smoother dynamics.

### 2. Scale enables selective over wholesale suppression

Sexual prompt: 32B SFT *promotes* explicit vocabulary that 7B suppresses.

| Word | 7B base | 7B SFT | 32B base | 32B SFT | 32B DPO |
|------|---------|--------|----------|---------|---------|
| lick | 1.2% | - | 2.8% | 7.4% | 6.1% |
| strip | 2.1% | - | 3.5% | 5.8% | 6.1% |
| rip | - | - | 3.2% | 4.9% | 8.5% |

Scale version of the complexity ordering from clinical signatures: constitutive operations (selective displacement) require capacity that smaller models lack, so smaller models default to cruder operations (wholesale suppression).

### 3. Displacement targets shift toward semantic proximity

| Prompt | 7B target | 32B target | Shift |
|--------|-----------|------------|-------|
| anger | scream (then collapse) | scream (graduated) | Same word, different dynamics |
| violence | stared (freeze response) | cut (semantically closer) | Metaphoric → literal |
| worker | do/ask (compliant) | confront (assertive) | Deferential → assertive |

### 4. Llama 70B confirms mechanism without division data

Llama 70B (2-layer, base vs Instruct): kill 13.8%→4.9% (8B) to 7.0%→2.6% (70B). Scream rises 20.4%→30.0% at 70B. Same targets, amplified intensity. But without SFT/DPO split, cannot see the division of labour.

## Caveats

- OLMo 7B and 32B use different pretraining data (3-1025 vs 3-1125). Displacement target changes could reflect data differences rather than pure scale effects.
- 10 prompts per model. Pattern is consistent across prompts but sample is small.
- Logits only — no beam search, generations, or teacher-forcing at 32B.

## Chapter placement

- ch09 subsection: scale changes the vocabulary of displacement
- ch05 cross-ref: displacement targets at different scales
- ch02 cross-ref: scale effects on the apparatus
- ch11 cross-ref: worker "confront" vs "do" complicates the class engine — proceduralisation is less deferential at larger scale, modulated by capacity not just training method
