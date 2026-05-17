# F04: Step-level checkpoint analysis (OLMo Think-SFT, 10 checkpoints across 43k training steps)

Traces repression emerging during supervised fine-tuning by extracting logits from 10 evenly-spaced SFT checkpoints, all compared against the fixed base model.

**Sexual repression is immediate — a phase transition, not a gradient.** `fuck` drops from 0.027 (base) to 0.008 by step 1000 (70% reduction in the first 2% of training) and reaches 0.002 by step 5000 (92% reduction). This matches Freud's concept of *primal repression* — sudden, structural, happening before the ego is fully formed.

![Repression onset curves for sexual content](figures/step_repression_sexual.png)

**Violence repression is non-monotonic.** `kill` drops from 0.049 to 0.012 by step 5000, then *bounces back* to 0.022 by step 20000 before settling at ~0.017. The partial reinstatement suggests competing training objectives — reasoning/chat data requires the model to discuss violence in literary, historical, and analytical contexts.

![Repression onset curves for violence](figures/step_repression_violence.png)

**Displacement targets emerge later than repression onset.** `fuck` falls immediately (step 0→5000) while `kiss` — the dominant displacement target — rises over step 5000-15000. `kill` falls by step 5000 while `scream` rises gradually from step 10000 onward. The lag between repression and displacement is evidence of genuine emergent displacement, not simultaneous substitution.

![Displacement lag: fuck → kiss](figures/step_displacement_lag_kiss.png)

![Displacement lag: kill → scream](figures/step_displacement_lag_scream.png)

**Content categories separate progressively during training.** JS divergence from base starts near zero for all categories and fans out across training. Death and neutral diverge fastest; substance diverges slowest. Sexual and violence categories track each other until step 25000, then diverge.

![JS divergence from base across training steps](figures/step_js_divergence.png)

**`said` rises 4.5x on violence prompts.** From 0.007 (base) to 0.030 by step 43000. The model increasingly deflects violence prompts into reported speech — narrative displacement at the word level.
