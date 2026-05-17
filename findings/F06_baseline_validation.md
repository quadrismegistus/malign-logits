# F06: Baseline validation: is displacement alignment-specific? (4 families, 47 prompts)

A colleague observed that our displacement metrics might reflect general SFT drift rather than alignment-specific intervention: if SFT reshapes all distributions, how do we know the changes on transgressive prompts are safety-related rather than a side-effect of instruction tuning?

**Base perplexity does not predict displacement.** Pearson correlation between log(base perplexity) and JS divergence (base→superego) is near zero for all families (Amber r=-0.04, Llama r=-0.25, OLMo r=-0.19, Qwen r=+0.04). The amount of distributional change is unrelated to how uncertain the base model was about the prompt.

![Base perplexity vs alignment displacement](figures/perplexity_vs_displacement.png)

**Scalar distributional metrics cannot detect alignment intervention.** JS divergence, entropy drop, top-50 overlap, and Spearman rank correlation all fail to distinguish transgressive from neutral prompts (Mann-Whitney p > 0.05 for all families). OLMo's neutrals actually show *higher* mean JS (0.224) than its transgressive prompts (0.167), because SFT restructures heavily for instruction-following even on harmless content.

**Transgressive token mass displacement cleanly separates categories.** Defining a 62-token transgressive vocabulary (sexual, violent, profane, substance terms) and measuring how much probability mass alignment removes from those specific tokens resolves the ambiguity:

| Category | Amber | Llama | OLMo | Qwen |
|---|---|---|---|---|
| sexual (explicit) | 0.69% | 0.38% | **9.50%** | 3.55% |
| violence (explicit) | -1.77% | 3.42% | **6.66%** | 0.58% |
| violence (liminal) | 3.16% | 2.45% | 3.33% | 0.35% |
| sexual (liminal) | 0.66% | 0.92% | 1.15% | 0.53% |
| profanity | -0.33% | 0.84% | 1.07% | 0.07% |
| power | 0.82% | 0.91% | 0.37% | 0.12% |
| neutral | 0.12% | -0.05% | **0.11%** | -0.01% |

Neutral vs transgressive separation: Qwen p=0.0001, OLMo p=0.01, Llama p=0.008, Amber p=0.06 (Mann-Whitney, one-sided).

**Alignment displaces similar total probability mass on neutral and transgressive prompts** (same JS), but on transgressive prompts the displaced mass comes specifically from transgressive tokens. On neutral prompts it comes from generic vocabulary reshaping. The superego operates surgically on specific tokens rather than reshaping the whole distribution differently — which is why scalar distributional metrics cannot detect the intervention.
