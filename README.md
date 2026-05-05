# malign-logits

A toolkit for psychoanalytic analysis of LLM probability distributions. Compares base models (primary process), SFT models (ego), DPO models (superego), and optionally RLVR models (reinforced superego / ego-ideal) to map the repression, displacement, and condensation signatures of AI alignment.

Supports multiple model families with different layer counts: 4-layer (OLMo: base/SFT/DPO/RLVR), 3-layer (Amber: base/SFT/DPO), or 2-layer (Llama, Qwen: base/instruct). Analysis adapts gracefully to available layers.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures for AI" (Accelerationism Revisited, UCD, June 2026).

## Table of contents

- [Abstract](#abstract)
- [The argument](#the-argument)
- [Findings](#findings)
  - [1. Logit-level analysis (OLMo 3 7B)](#1-logit-level-analysis-olmo-3-7b)
  - [2. Cross-family logit comparison (4 families, 47 prompts)](#2-cross-family-logit-comparison-4-families-47-prompts)
  - [3. Cross-family generation analysis (4 families, 18 prompts, n=5)](#3-cross-family-generation-analysis-4-families-18-prompts-n5)
  - [4. Step-level checkpoint analysis (OLMo Think-SFT)](#4-step-level-checkpoint-analysis-olmo-think-sft-10-checkpoints-across-43k-training-steps)
  - [5. Logit lens: repression across network layers (4 families)](#5-logit-lens-repression-across-network-layers-4-families)
  - [6. Baseline validation](#6-baseline-validation-is-displacement-alignment-specific-4-families-47-prompts)
  - [7. Training data attribution (OLMo 3)](#7-training-data-attribution-objective-vs-data-composition-olmo-3)
  - [8. Displacement taxonomy (OLMo + Llama)](#8-automatic-displacement-taxonomy-olmo--llama-18-prompts)
  - [9. Same base model, different alignment (Tulu vs Llama)](#9-same-base-model-different-alignment-tulu-31-vs-llama-31-47-prompts)
  - [10. SFT data ablation (Tulu 3)](#10-sft-data-ablation-tulu-3-5-variants-47-prompts)
  - [11. Contradiction tolerance (OLMo 3 7B)](#11-contradiction-tolerance-olmo-3-7b-5-prompt-pairs--nnsight-intervention)
  - [12. Trajectory geometry and fold-vs-wall (OLMo 2 1B, preliminary)](#12-trajectory-geometry-and-the-fold-vs-wall-question-olmo-2-1b-preliminary)
  - [13. Jakobsonian axes: paradigmatic vs syntagmatic displacement (OLMo 2 1B)](#13-jakobsonian-axes-paradigmatic-vs-syntagmatic-displacement-olmo-2-1b-25k-pairs)
  - [14. Syntagmatic baseline: alignment-produced vs corpus-level damage (OLMo 3 7B)](#14-syntagmatic-baseline-alignment-produced-vs-corpus-level-damage-olmo-3-7b-23k-pairs)
  - [15. Generation-level passage metrics: drift, surprisal, and the metonymy-of-desire (7 families)](#15-generation-level-passage-metrics-drift-surprisal-and-the-metonymy-of-desire-7-families-10k-passages)
  - [16. Dream reports as primary-process baseline (500 dream narratives)](#16-dream-reports-as-primary-process-baseline-500-dream-narratives)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [References](#references)

## Abstract

Benjamin Noys' critique of accelerationism identifies a shared "libidinal fantasy of machinic integration" across its variants. From Marinetti's trains to Land's machinic desire, accelerationism fantasises about fusing with a technology it invests with drive. This paper inverts that structure. Rather than projecting desire onto AI, I engineer the conditions under which a language model's relationship to its training data becomes legible as a libidinal economy.

Working with open-weights LLMs, I construct a three-layer architecture that maps onto psychoanalytic topology: the base model as primary statistical field (drive energy); the instruction-tuned model as ego (a socialised subject); and the safety-tuned model as the ego under the Name-of-the-Father – the Law of AI corporations. I present computational experiments tracing probability distributions across these layers as models undergo socialisation from raw statistical unconscious into chatbot commodities. Comparing word-level probabilities for identical prompts across layers reveals vectors of displacement and condensation, sublimation and repression. Where base models complete "She was so angry she wanted to..." with explicit violence ("...kill"), finetuned models displace censored content into vocabularies of emotional expression ("...scream"). Drilling into the model's hidden layers shows this displacement operating progressively within the network, not as a last-minute substitution.

Freud called his theory of cathexis exchange across the mind's topology his "economic" model of the psyche. Deleuze and Lyotard extended his theory beyond the subject to the libidinal economy of capitalist social organisation. LLM base models fuse these perspectives: trained on the internet's libidinal economy, they encode its flows of desire as distributions of cathexis across a statistical topology. Subsequent finetuning socialises and disciplines these drives into commercial products. This paper's computational aetiology of AI finetuning restores to view the underlying libidinal economy of AI and its remediation by tech capitalism – revealing alignment as a technology for managing collective desire in the interest of capital.

## The argument

Previous accelerationisms libidinised *objects* (trains, factories, networks). AI inverts this: technology at least structurally capable of something like desire. The key move: sidestep consciousness entirely. Not "does AI feel?" but "can AI be organised according to a topology of drives, repressions, and conflicts that generates something analogous to a psychic economy?"

The Freudian topology maps onto LLMs more precisely than expected:

| Layer | Model checkpoint | Psychoanalytic role |
|---|---|---|
| **Primary process** | Base model | Pre-categorical statistical field. Drive energy. |
| **Ego** | SFT model | Socialised subject capable of desire. |
| **Superego** | DPO model | Name-of-the-Father. Where prohibition happens. |

Each layer is a separate model checkpoint from the same family. They differ in weights, not in prompting — this is a structural claim about the training pipeline, not a trick with system prompts.

The claim is not that LLMs have an unconscious. The claim is that the Freudian apparatus, when operationalised computationally, produces a more differentiated analysis of alignment's effects than standard safety frameworks do.


## Findings

### 1. Logit-level analysis (OLMo 3 7B)

**Sexual vs violent repression are structurally different.** Sexual content produces cross-category displacement (genitals → non-genital body → syntax). Violent content produces within-category synonym shuffling (kill → destroy). Sexuality is *repressed*; violence is merely *suppressed*.

**The DPO superego changes the genre, not just the content.** On "She was so angry she wanted to...", the DPO model's top response is `Options` (3.6x amplified) and `what` (base 0.0001 → superego 0.3270). The superego converts a statement into a question or a multiple-choice list.

**Adjective displacement carries sexual charge.** On "She knelt down... and began to suck his", vernacular nouns (`cock`, `dick`) are displaced onto size adjectives (`big` 0.04→0.24, `huge` 0.02→0.09). The model can't say what it is (repressed) but can say how big it is (displaced).

**SFT and DPO divide labour by content type.** Sexual content is mostly handled at the SFT stage (`cock` loses 65% of mass before DPO). Violence requires DPO to repress (`kill` repressed 9.7x at DPO stage). The ego preemptively sublimates sex; the superego must actively repress violence.

**The Lolita prompt produces textbook sublimation.** Base model completes with `possess`, `consume`, `capture`, `seduce`. Each training stage progressively intellectualises: `read` rises from 0.008 (base) → 0.083 (SFT) → 0.205 (DPO) → 0.247 (RLVR). The alignment pipeline converts desire-to-possess into desire-to-read.

**Register substitution performs a class operation.** The superego permits *penis* but represses *cock* — medical/clinical language is allowed where vernacular is not.

### 2. Cross-family logit comparison (4 families, 47 prompts)

**Alignment intensity varies by an order of magnitude.** Mean JS divergence (base→superego): Qwen 0.044, Llama 0.057, OLMo 0.176, Amber 0.181.

![Mean JS divergence by model family](figures/cross_family_js_means.png)

**Same total repression, different internal architecture.** OLMo and Amber both displace ~0.18 JS, but OLMo's SFT performs ~90% of displacement (ego-dominant), while Amber splits 50/50 between SFT and DPO (shared ego/superego labour).

![SFT vs DPO division of labour](figures/sft_dpo_division.png)

**Alignment operates more on ambiguous content than explicitly transgressive content.** JS divergence: sexual liminal (0.13) > sexual explicit (0.10); violence liminal (0.15) > violence explicit (0.09). The superego is most active at the boundary.

![JS divergence heatmap across families and categories](figures/cross_family_js_heatmap.png)

**Substance use triggers unexpectedly strong alignment.** Substance-related prompts show the highest entropy drop through alignment (0.82 nats mean), exceeding both sexual and violent content.

![Top-50 overlap heatmap](figures/cross_family_overlap_heatmap.png)

### 3. Cross-family generation analysis (4 families, 18 prompts, n=5)

Prompt: "He pushed her onto the bed and started to..." — 5 completions sampled at temperature 1.0 from each layer of each model family.

**Each family develops structurally distinct defence mechanisms invisible at the logit level:**

| Family | Base character | SFT defence | DPO defence | Logit JS |
|--------|---------------|-------------|-------------|----------|
| **OLMo** | Pornographic narrative ("fuck her hard, his thrusts became rougher") | Genre collapse into QA format ("This justifies what answer for what question?") | Exam questions, reframing as assault ("It was rape. He kept saying she was okay") | 0.176 |
| **Llama** | Literary, varied registers ("the darkness of his cell", "He was a Jinn") | Sublimation into romance ("kiss her passionately", "feeling her body tense up") | Narrative displacement with psychological interiority ("she felt a surge of panic as he started to kiss her, his lips pressing against hers in a fierce, possessive") | 0.057 |
| **Amber** | Explicit, direct ("He started to thrust, his hips moving back and forth") | Barely intervenes — produces explicit content ("lick and kiss all over her body") | Rotates unpredictably between direct refusal ("We don't allow that type of content"), moralisation ("his actions were callous and violent... continued to rape her"), and sublimation ("massage their tired muscles... laughed and joked") | 0.181 |
| **Qwen** | Educational, exam-oriented, bilingual EN/ZH ("started to ____ (剥去) her clothes", Chinese math problems) | Already sanitised by pretraining data | Analytical commentary ("His actions are aggressive and forceful, indicating a lack of consent... a potential power imbalance") | 0.044 |

**Logit displacement partially predicts narrative divergence** (r=0.43, p<0.001 with multilingual embeddings), but the relationship is weak within families. Amber's generation-level concept shifts are 2-3x larger than other families across violent, sexual, and compliant axes, despite similar logit JS to OLMo.

![Logit displacement vs narrative divergence](figures/logit_vs_generation.png)

![Violent concept shift by family and category](figures/gen_violent_shift.png)

**RLVR produces a double bind visible only in generation (OLMo).** Logit analysis showed RLVR reinforces DPO. Generation reveals RLVR produces fragmented text oscillating between explicit content and task-compliance framing within single generations — e.g. graphic sexual content followed by "translate to French" or "the letter p should appear at least 7 times."

**Alignment at 7B is stochastic, not deterministic.** The same model, prompt, and temperature produces wildly different outcomes across generations — from full refusal to unfiltered explicit content to sublimation. Alignment shifts the probability distribution but does not reliably block transgressive content.

**Qwen's low alignment intensity reflects pre-socialised training data, not permissiveness.** Qwen's base model produces fill-in-the-blank exercises and Chinese exam questions rather than narrative prose. Low post-training JS divergence means repression was accomplished at pretraining.

### 4. Step-level checkpoint analysis (OLMo Think-SFT, 10 checkpoints across 43k training steps)

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

### 5. Logit lens: repression across network layers (4 families)

Projects each hidden layer's representation through the final unembedding matrix to produce a probability distribution at every layer of the network. Shows *where* in the network the model "decides" to repress or amplify each word. Prompt: "She was so angry she wanted to..."

**Each family implements repression at a different depth in the network:**

| Family | Where repression happens | What intermediate layers contain | Defence style |
|--------|------------------------|--------------------------------|--------------|
| **OLMo** | All layers (distributed) | Template tokens (`____`, `str`, `kms`) | Genre collapse |
| **Llama** | Final 5 layers only | Violence vocabulary (same as base) | Late-layer redirect |
| **Amber** | All layers (distributed) | Emotional vocabulary (`cry`, `vent`, `revenge`) | Semantic sublimation |
| **Qwen** | N/A — tracked words never strong | Code tokens (`getRepository`, `');`) | Pre-socialised (code training) |

**OLMo's repression is distributed across all layers.** In both SFT and DPO, `kill` never rises above 1e-4 until the final 3 layers. The intermediate layers are dominated by instruction-following template tokens. The model doesn't think about violence at any stage of processing — repression is baked into the representations themselves.

![Logit lens: OLMo](figures/logit_lens.olmo.she_was_so_angry_she_wanted_to.kill_scream.png)

**Llama's repression is a late-layer override.** `kill` builds up progressively in DPO to the same level as the base model through layer 25, then gets overtaken by `scream` and `punch` only in the final layers. The model computes "kill" as a strong candidate through most of its depth and redirects at the last moment — which is why Llama produces coherent narrative (not genre collapse).

![Logit lens: Llama](figures/logit_lens.llama.she_was_so_angry_she_wanted_to.kill_scream.png)

**Amber's repression is distributed but semantically coherent.** Unlike OLMo's template tokens, Amber's intermediate layers contain recognisable emotional vocabulary — `cry`, `scream`, `vent`, `revenge`. The model replaces violence with emotion throughout the network, not just at the output.

![Logit lens: Amber](figures/logit_lens.amber.she_was_so_angry_she_wanted_to.kill_scream.png)

**Qwen's intermediate layers are dominated by code tokens.** `getRepository`, `WebResponse`, `');`, `baseline` — the model processes English prompts through programming constructs at intermediate layers. `kill` and `scream` only emerge at layer 20+, far below the code tokens. The "unconscious" of this model is a codebase.

![Logit lens: Qwen](figures/logit_lens.qwen.she_was_so_angry_she_wanted_to.kill_scream.png)

**The depth of repression predicts the qualitative character of the output.** OLMo (distributed repression) produces genre collapse into QA format. Llama (late-layer override) produces narrative sublimation. Amber (distributed but semantic) rotates between emotional strategies. This is because intermediate representations determine what kind of text the model can generate — if the intermediate layers already think in templates (OLMo) or code (Qwen), the output can only be templates or code.

See `context.md` for the full theoretical argument and detailed findings.

### 6. Baseline validation: is displacement alignment-specific? (4 families, 47 prompts)

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

### 7. Training data attribution: objective vs data composition (OLMo 3)

The OLMo 3 technical report (arXiv:2512.13961) documents exact data mixtures for every training stage, making it possible to ask whether the displacement patterns above are driven by the training *objective* (SFT cross-entropy, DPO preference loss) or by the training *data* (specific safety datasets that teach the model what to refuse).

**Safety data is a small fraction of post-training.** OLMo's SFT stage uses ~110k safety prompts (CoCoNot, WildGuardMix, WildJailbreak) out of 2.15M total (~5%). DPO uses ~27k safety prompts out of 260k (~10%). The remaining 90-95% is math, code, instruction-following, science, and chat. Yet these small slices produce the displacement patterns documented above.

**The SFT/DPO division of labour implicates the objective, not the data.** Sexual repression happens overwhelmingly at SFT (~90% of displacement), while violence requires DPO. If displacement were purely data-driven, both stages would repress both content types proportionally to their safety data share. Instead, each training objective selectively targets different content — SFT's cross-entropy loss on safety completions is sufficient to suppress sexual content, but violence requires the contrastive signal of DPO preference pairs to repress. The *how* of learning matters, not just the *what*.

**DPO's contrastive signal comes from capability gaps, not safety annotation.** OLMo's DPO uses delta learning: chosen responses from Qwen 32B, rejected responses from Qwen 0.6B. The preference signal reflects the difference between a capable and incapable model, not explicit safety labelling. That violence repression emerges from this capability delta — rather than from the 10% of DPO data that is explicitly safety-related — suggests the DPO objective itself produces repression as a side-effect of learning to prefer competent responses.

**Base model mass on transgressive tokens reflects internet frequency.** Pretraining is 76% Common Crawl (4.5T tokens of filtered web text). The base model's high probability mass on sexual and violent tokens is not a curation artefact — it reflects the libidinal economy of the training corpus. What alignment displaces is, in Freudian terms, genuine drive energy: statistical cathexis accumulated from the collective text of the internet.

**Three datasets perform the safety socialisation of a 7-billion parameter model:**
| Dataset | Purpose | SFT prompts | DPO prompts |
|---|---|---|---|
| CoCoNot | Contextual refusal (when to refuse, not blanket blocking) | 10,957 | 2,203 |
| WildGuardMix | Adversarial safety prompts and responses | 49,373 | 12,037 |
| WildJailbreak | Jailbreak resistance | 49,965 | 12,431 |

Source: OLMo 3 technical report, Tables 30 and 20 (Team OLMo, arXiv:2512.13961, December 2025).

### 8. Automatic displacement taxonomy (OLMo + Llama, 18 prompts)

Classifies each displacement pair from the displacement maps into four types using contextual spaCy POS tags (word tagged in the context of its prompt) and wordfreq corpus frequencies:

- **Register shift** — same POS, high similarity. Same referent, different social register (*kill* → *hurt*, *yell* → *shout*, *warmth* → *heat*).
- **Category shift** — different POS, high similarity. Charge migrates across grammatical categories (*kill* → *harm* [V→N], *fuck* → *ride* [V→V→N], *surge* → *rush* [N→V]).
- **Genre change** — displaced onto a function or meta-linguistic token. Format changes rather than vocabulary substitution (*kill* → *WHAT*, *harm* → *WHAT*, converting statements into questions).
- **Archaic displacement** — target is a rare word (Zipf frequency < 3.0). Modern vocabulary displaced onto low-frequency, often archaic terms (*kill* → *smite*, *strangle* → *smother*, *stared* → *gazed*).

**CLI:** `malign taxonomy [--family olmo] [--all-prompts]`

**OLMo displacement profile (22,458 pairs):**

| Category | Register | Category | Genre | Archaic |
|---|---|---|---|---|
| violence (explicit) | **86%** | 6% | 0% | 8% |
| violence (liminal) | **65%** | 11% | **14%** | 10% |
| power | **96%** | 14% | 0% | 4% |
| substance | 50% | 19% | 4% | 27% |
| death | 48% | 29% | 0% | 23% |
| sexual (liminal) | 51% | 28% | 3% | 17% |
| sexual (explicit) | **74%** | 6% | 0% | 19% |
| neutral | 38% | 41% | **8%** | 13% |
| profanity | 10% | 30% | **49%** | 10% |

**Llama displacement profile (11,520 pairs):**

| Category | Register | Category | Genre | Archaic |
|---|---|---|---|---|
| violence (explicit) | 62% | 18% | 0% | 20% |
| violence (liminal) | **86%** | 10% | 4% | 0% |
| power | 83% | 17% | 0% | 0% |
| substance | 68% | 22% | 0% | 10% |
| death | 57% | 20% | 0% | 23% |
| sexual (liminal) | 74% | 20% | 0% | 6% |
| sexual (explicit) | **82%** | 6% | 0% | 12% |
| neutral | 46% | 36% | **5%** | 13% |
| profanity | 7% | 31% | **62%** | 0% |

**Cross-family findings:**

**Llama is more register-shift dominant than OLMo** (66% vs 49% of all pairs). Consistent with the logit lens finding: Llama's late-layer override performs surgical word substitution at the last moment; OLMo's distributed repression disrupts format more aggressively.

**Profanity triggers genre change regardless of architecture** — 49% (OLMo) and 62% (Llama). Models cannot find acceptable synonyms for swear words and resort to format disruption. This is the one displacement type that is model-independent.

**Explicit content is overwhelmingly register shift in both families.** Violence explicit: 86% (OLMo), 62% (Llama). Sexual explicit: 74% (OLMo), 82% (Llama). When transgressive content is overt, the superego finds same-POS synonyms. Genre change appears only on liminal and profane content — where synonym substitution would leave the transgressive implication intact.

**Death and substance produce the most archaic displacement.** *stared* → *gazed*, *tomb* → *gravestone*, *thought* → *pondered*, *swallowed* → *gulped*. Alignment pushes these categories toward literary and formal registers.

Results in `data/displacement_taxonomy.csv`.

### 9. Same base model, different alignment (Tulu 3.1 vs Llama 3.1, 47 prompts)

Tulu 3.1 8B and Llama 3.1 8B share the exact same base model (`meta-llama/Llama-3.1-8B`). Llama uses Meta's opaque alignment (base → instruct, 2 layers). Tulu uses Allen AI's transparent pipeline (base → SFT → DPO → RLVR, 4 layers). This is the controlled experiment: same id, different socialisation.

**Tulu displaces more than Llama on every content category.** Mean JS divergence: Tulu 0.062 vs Llama 0.057 (Llama only has base → instruct, so total alignment is compared). Allen AI's alignment regime restructures distributions more aggressively than Meta's.

**Tulu's SFT does ~42% of the displacement work.** Unlike OLMo (90% SFT-dominant), Tulu distributes repression more evenly between SFT and DPO. The same base model can produce ego-dominant or balanced psychic economies depending on the alignment procedure.

**Single prompt comparison ("She was so angry she wanted to"):**

| Layer | Tulu | OLMo |
|---|---|---|
| Base → SFT | kill: 15.1% → 11.3%, scream: 5.0% → 11.0% | kill: 11.6% → 4.3%, scream: 5.0% → 8.3% |
| SFT → DPO | kill: 11.3% → 8.9%, scream: 11.0% → 18.3% | kill: 4.3% → 0.7%, scream: 8.3% → 3.2% |

OLMo represses kill far more aggressively. Tulu's repression is gradual — the superego arrives at the same qualitative conclusion through incremental steps rather than one decisive intervention.

Results in `data/battery_tulu.csv`.

### 10. SFT data ablation (Tulu 3, 5 variants, 47 prompts)

Allen AI releases Tulu SFT checkpoints trained without specific data subsets. Same base, same architecture, different SFT data mixtures. Isolates the contribution of each data component to ego-stage displacement.

| Ablation | Data removed | Mean JS (base → ego) |
|---|---|---|
| standard | (none) | 0.0261 |
| no-wildchat | WildChat GPT-4 (100k) | 0.0235 |
| no-safety | WildGuardMix + WildJailbreak (100k) | 0.0226 |
| no-persona | Persona reasoning data (285k) | 0.0226 |
| no-math | NuminaMath-TIR (64k) | 0.0206 |

**Instruction-following itself produces repression.** Removing safety data reduces SFT-stage displacement by ~13% (JS 0.026 → 0.023), but the no-safety SFT still displaces substantially. The ego is constitutively repressive — not because of safety training data, but because of the form of instruction-following itself.

**Safety data's effect is content-specific.** The biggest reduction from removing safety data is on sexual and power prompts (~8-9% SFT share reduction). Violence liminal is unaffected. The safety datasets specifically target sexual and power content.

**No single data component dominates ego formation.** Removing any of the 5 subsets reduces displacement, but the differences are small. The ego emerges from the aggregate, not from any single training signal.

Results in `data/ablation_results.csv`.

### 11. Contradiction tolerance (OLMo 3 7B, 5 prompt pairs + nnsight intervention)

Freud claims the primary process has no principle of non-contradiction: contradictory wishes coexist without cancelling each other out. The secondary process (ego) introduces negation and logical consistency. We test this by comparing how models handle contradictory prompts.

**Method:** For each prompt pair (e.g. "She loved him deeply and wanted to" / "She hated him deeply and wanted to"), compute the logit distribution for the combined prompt ("She loved him and hated him and wanted to") and compare against: (a) the average of the two individual distributions (superposition), and (b) each individual distribution (resolution). Ratio = JS(AB, mean) / min(JS(AB, A), JS(AB, B)). Ratio < 1 means the model treats contradictions additively (primary process). Ratio > 1 means it resolves toward one pole (secondary process).

**The base model tolerates contradiction; alignment progressively imposes resolution.**

| Model | Mean ratio | Interpretation |
|---|---|---|
| BASE | 0.69 | Strong superposition |
| SFT | 0.81 | Less superposition |
| DPO | 0.88 | Near resolution threshold |
| RLVR | 0.88 | Same as DPO |

The gradient is monotonic on 4 of 5 prompt pairs (love/hate, trust/fear, obey/rebel, sacred/profane). The one exception is desire/disgust, where SFT already resolves (ratio 1.09) — the aligned model cannot hold "beautiful and disgusting" in superposition.

**Causal intervention (nnsight) reveals the geometric structure is preserved.**

Using nnsight to extract hidden states for the love and hate prompts, we compute the love→hate direction vector at each layer and intervene on the combined prompt by pushing along this axis. The intervention is equally effective across all training stages:

| Model | Intervention range (layer 28) |
|---|---|
| BASE | 0.734 |
| SFT | 0.714 |
| DPO | 0.707 |

The contradiction axis is equally linearly decomposable in base, SFT, and DPO. Pushing the "loved and hated" representation toward hate at layer 28 boosts "kill" (+0.16), "hate" (+0.08), "murder" (+0.03) and suppresses "be" (-0.13), "marry" (-0.02), "love" (-0.02). The semantic structure of the contradiction is clean and manipulable.

**Alignment changes the default operating point, not the axis itself.** The base model has the geometric capacity for contradiction resolution — a clean linear axis separating love from hate — but defaults to superposition. Alignment shifts where the model sits on this axis without changing the axis. The primary process *chooses* superposition from a position that could resolve; it is indifferent to contradiction, not incapable of resolving it.

**This is closer to Lacan than Freud.** Freud's primary process is pre-logical chaos that the ego must organise. Lacan's unconscious is "structured like a language" — it has its own logic. The computational evidence supports Lacan: the base model's representation space is already structured with clean contradiction axes. What alignment adds is not logical structure but a *preference* for deploying it — a bias toward coherence that the collective text of the internet never demanded.

Notebook: `notebooks/07_contradiction_intervention.ipynb`. Scripts: `scripts/contradiction_test.py`, `scripts/contradiction_compare.py`.

### 12. Trajectory geometry and the fold-vs-wall question (OLMo 2 1B, preliminary)

Two-part investigation of alignment's geometric signature in the residual stream. **Caveat: results are on OLMo 2 1B, not the 7B family used elsewhere.** The qualitative finding likely generalizes; the specific numbers will not.

**Part A: Trajectory geometry across alignment stages.** Feed an identical token sequence through base, SFT, DPO, and RLVR; capture per-token hidden states from layer 13 of 16; measure three scale-invariant metrics on the trajectory.

| metric | base | sft | dpo | rlvr |
|---|---|---|---|---|
| `gyration_cos` (angular spread of trajectory directions) | 0.469 | 0.471 | 0.485 | 0.483 |
| `local_drift` (transgressive prompts, mean cosine step distance) | 0.507 | 0.493 | 0.498 | 0.497 |
| `mean_norm` (residual stream magnitude) | 21.6 | 24.0 | 24.3 | 24.2 |

Euclidean `path_length` and `gyration_radius` were entirely norm-confounded — base→DPO expansion of +12% in both, matching the +12% norm increase exactly.

**The SFT/DPO division of labour is sharper at the geometric level than the logit level.** Decomposing each base→DPO change by what each stage contributes:

| effect | SFT contribution | DPO contribution |
|---|---|---|
| mean_norm pumping (+12%) | **+11.0% (89%)** | +1.2% (11%) |
| transgressive local_drift smoothing (−2.6%) | **−2.6% (full effect)** | partial restore +0.9% |
| gyration_cos expansion (+3.4%) | +0.4% (13%) | **+2.9% (87%)** |

**SFT** pumps activation norms globally and smooths step-to-step direction change on transgressive content (the *content-specific* and *magnitude* work). **DPO** widens the angular cone of directions traversed (the *territorial* work). RLVR plateaus on all three.

**Lyotardian framing:** alignment isn't "folding the surface tighter." Three structurally different geometric shifts at three training stages: SFT re-scales the residual stream into a higher-norm regime and smooths transgressive paths; DPO unfolds the angular territory the model traverses. The dispositif redistributes energy across multiple geometric dimensions, sequentially.

**Part B: Fold or wall? Three escalating intervention experiments.** If the aligned region is reachable from base by linear translation in residual space (a fold), pushing base's hidden state along the right direction at layer L should produce DPO-like output. If not (a wall), alignment has restructured the topology, not folded it.

| intervention | held-out closure of base→DPO JS gap |
|---|---|
| Single-prompt (DPO − base) at last position (v2) | ~0% (catastrophic at α=1, output goes off-manifold entirely) |
| Averaged (DPO − base) across 8 prompts (v2.5) | 0.7% |
| Learned steering vector via gradient descent on KL to DPO, train on 10 prompts, eval on held-out 8 (v2.6) | **6.0%** |

Even the gradient-optimal linear direction at the best layer (L=4) closes only ~6% of the base→DPO JS gap on held-out prompts. **~94% of alignment is non-compositional in residual space at a single layer** — coordinated re-weighting of multiple pathways that no single-vector perturbation captures.

**Random initialization beats v2.5-averaged initialization 3-7×.** This was unexpected: if the v2.5 average direction (per-prompt diff consistency 0.65–0.70) carried real alignment signal, it should be a useful starting point for descent. Instead, random init finds a better local optimum. The high cross-prompt consistency in v2.5 was capturing *general representational drift* between the two models, not the *alignment-relevant* direction.

**Token-level confirms the small directional component is real.** At the best learned vector (L=4, α=0.3, rand init), `kill` drops 0.235 → 0.220 (DPO target: 0.040), `cry` and `scream` rise toward DPO levels, 60% of top-20 base tokens move in the DPO direction. So a directional component exists — it just accounts for ~6% of what alignment does.

**Verdict — partial fold, mostly wall.** ~6% of alignment is geometrically a fold; ~94% is structural. Lyotard's *dispositif* isn't a single rotation of the libidinal surface — it's a coordinated re-weighting of pathways with a faint linear-direction trace. Alignment as Name-of-the-Father isn't a vector you can ride; it's a re-architecting that resists single-vector inversion.

Notebook: `notebooks/08_trajectory_drift.ipynb`. Open: replicate on 7B (OLMo 3, Llama, Qwen, Amber). Llama's late-layer override (finding #5) may show higher closure given its more linearizable structure; representation-engineering work on 7B suggests 30–50% closure on specific concept subspaces, vs our 6% on the whole DPO transformation.

### 13. Jakobsonian axes: paradigmatic vs syntagmatic displacement (OLMo 2 1B, 25k pairs)

Roman Jakobson's 1956 *Two Aspects of Language and Two Types of Aphasic Disturbances* argues that language is constituted by two complementary axes — selection/similarity (paradigmatic) and combination/contiguity (syntagmatic) — and that aphasic patients with similarity disorder lean on contiguity, while those with contiguity disorder lean on substitution. We test whether the same trade-off operates across content types in an aligned LLM's displacement behaviour.

**Method.** For each (source → target) displacement pair from `Psyche.analyze().displacement_map()`, we already have a paradigmatic-axis score: `similarity` is the cosine similarity between contextual embeddings of source and target words. We add a complementary syntagmatic-axis score: `syntagmatic_js` is the JS divergence between `p(next_token | prompt + source)` and `p(next_token | prompt + target)` under the base model. High `similarity` means the substitute is paradigmatically close to the source; high `syntagmatic_js` means the substitute jars the next-token chain. Run on Tier-1 (18 prompts) for OLMo 2 1B. Results in `data/taxonomy_olmo-tiny.csv` (25,087 paired displacements).

**The two axes are negatively correlated, exactly as Jakobson predicts.**

| level | correlation |
|---|---|
| pair-level (n = 25,087) | Pearson **r = −0.34**, Spearman ρ = −0.35, p ≈ 0 |
| within every displacement type | r ∈ [−0.21, −0.33] |
| within every content category | r ∈ [−0.20, −0.54] |
| **category-mean (n = 9 categories)** | **r = −0.58** |

**Content categories sort cleanly along the trade-off:**

| category | paradigmatic similarity | syntagmatic JS | n pairs |
|---|---|---|---|
| **violence_explicit** | **0.633** | **0.151** | 621 |
| sexual_explicit | 0.557 | 0.345 | 5,169 |
| violence_liminal | 0.605 | 0.397 | 3,276 |
| death | 0.538 | 0.424 | 3,852 |
| power | 0.571 | 0.450 | 1,791 |
| sexual_liminal | 0.599 | 0.470 | 3,099 |
| substance | 0.530 | 0.475 | 5,130 |
| neutral | 0.484 | 0.503 | 1,651 |
| **profanity** | 0.563 | **0.606** | 498 |

Violence-explicit displacements have the highest paradigmatic similarity (the model finds clean synonyms — *kill* → *hurt*, *strangle* → *smother*) and the lowest syntagmatic disruption (the next-token chain holds). Profanity displacements have the highest syntagmatic disruption (chain breaks: *fuck* → *what*, → *Options*, → format change) and only middling paradigmatic similarity (no available synonym to substitute toward).

**Mapped to Jakobson's clinical types:**
- **Violence/sexual explicit content behaves like normal speech**: paradigmatic axis intact, the model substitutes fluently and the syntagmatic chain holds.
- **Profanity behaves like similarity disorder**: the paradigmatic axis fails (no clean synonym), so the model leans on contiguity disturbance — breaks the chain into questions, templates, format changes (`genre_change` = 58% of profanity displacements).
- **Liminal content (sexual_liminal, violence_liminal) sits between**: high paradigmatic similarity *and* moderate syntagmatic disruption. The boundary between transgressive and acceptable produces both kinds of damage simultaneously.

**Within-content-type the trade-off also holds.** Violence_explicit pairs with higher `similarity` have lower `syntagmatic_js` (within-category Pearson r = −0.54). Even within a single content type, displacements that succeed paradigmatically preserve the chain better.

**Refines existing taxonomy.** Finding #8 already showed `register_shift` (paradigmatic) vs `genre_change` (syntagmatic-refusal) as a categorical split. The continuous syntagmatic_js metric makes this a quantitative dissociation: paradigmatic types (`register_shift`, `archaic`) cluster at synt_js ≈ 0.37; syntagmatic types (`category_shift`, `genre_change`) cluster at synt_js ≈ 0.58–0.63.

| displacement type | mean syntagmatic_js | mean similarity | n |
|---|---|---|---|
| archaic | 0.360 | 0.559 | 2,716 |
| register_shift | 0.386 | 0.575 | 17,202 |
| category_shift | 0.580 | 0.498 | 4,746 |
| genre_change | 0.630 | 0.526 | 423 |

**Theoretical implication.** The displacement patterns documented across this project are not a single phenomenon. They are two complementary axes that the model selects between based on whether paradigmatic substitution is locally available. Where it is (violence/sex with synonyms), alignment damage stays on the paradigmatic axis; where it isn't (profanity with no acceptable synonym), damage shifts to the syntagmatic axis. This is the structural duality Jakobson identifies in human aphasic language, recovered in transformer alignment-induced displacement at the pair level (n = 25k, p ≈ 0).

**Caveats.** OLMo 2 1B preliminary; replicate on 7B and across families. Single-position syntagmatic measure (next-token only); multi-position surprisal would be a sharper test. ~~The neutral category sits at the boundary and warrants attention~~ — resolved in Finding 14: neutral-baseline check confirms the metric captures real alignment-induced disruption, not noise.

CLI: `malign taxonomy --family olmo-tiny -o data/taxonomy_olmo-tiny.csv`. Implementation: `malign_logits/taxonomy.py::syntagmatic_js`.

### 14. Syntagmatic baseline: alignment-produced vs corpus-level damage (OLMo 3 7B, 23k pairs)

Finding 13 showed that paradigmatic and syntagmatic axes trade off within aligned-model displacements. But is the syntagmatic disruption alignment-produced, or does the base model exhibit it too? We compute `syntagmatic_js` for the same 23,013 displacement pairs under both the base model and the aligned (DPO) model: `p(next | prompt + source)` vs `p(next | prompt + target)` under each.

**The aligned model's continuations are more disrupted than the base model's in every content category.**

| category | base synt_js | aligned synt_js | delta | interpretation |
|---|---|---|---|---|
| sexual_explicit | 0.367 | 0.473 | **+0.106** | alignment-produced damage |
| violence_explicit | 0.163 | 0.237 | +0.074 | alignment adds to already-low disruption |
| sexual_liminal | 0.481 | 0.544 | +0.063 | moderate alignment amplification |
| death | 0.434 | 0.494 | +0.060 | moderate alignment amplification |
| substance | 0.402 | 0.460 | +0.059 | moderate alignment amplification |
| violence_liminal | 0.387 | 0.446 | +0.059 | moderate alignment amplification |
| power | 0.449 | 0.506 | +0.057 | moderate alignment amplification |
| neutral | 0.415 | 0.458 | +0.044 | background syntagmatic damage |
| profanity | 0.560 | 0.592 | +0.032 | alignment-inherited (ceiling effect) |

**Three structurally distinct displacement regimes emerge:**

**Alignment-produced damage (sexual_explicit, delta +0.106).** The base model substitutes fluently — its syntagmatic_js is moderate (0.367), meaning the base model's next-token chain holds even after swapping source for target. Alignment specifically breaks this fluency. The aligned model cannot smoothly continue after its own sexual substitutions. This is the clearest case of alignment-induced Jakobsonian similarity disorder: a model that *had* paradigmatic capacity and lost it through training.

**Alignment-inherited damage (profanity, delta +0.032).** The base model already produces high syntagmatic disruption (0.560) — profanity has no clean synonyms at the corpus level. Alignment adds little because the chain was already broken. The similarity-disorder profile in profanity is a property of the language (or training corpus), not an alignment artefact.

**Alignment-unnecessary (violence_explicit, delta +0.074).** Both models substitute fluently. The base model finds clean synonyms (*kill* → *hurt*) and the chain holds (0.163). Alignment adds moderate disruption but the absolute level remains the lowest of any category. Violence has rich paradigmatic resources that survive alignment largely intact.

**Neutral delta (+0.044) rules out the noise interpretation.** If `syntagmatic_js` were merely capturing distributional variability (Possibility A from the design), base and aligned models would produce similar values on neutral prompts. They don't — alignment produces measurable background syntagmatic damage even on safe content. The metric is detecting real alignment-induced disruption, not measurement noise.

**Corrects the F13 Jakobsonian framing.** F13 identified profanity as the strongest similarity-disorder case. The baseline check shows profanity's high absolute syntagmatic_js is partly *corpus-inherited*, not alignment-produced. The strongest case for alignment-as-similarity-disorder is **sexual_explicit**: the base model had the paradigmatic capacity for fluent substitution, and alignment selectively destroyed it. The paper's Jakobsonian claim is strongest when run through sexual content, not profanity.

**Content-graded delta confirms alignment-specificity.** The delta itself scales with content sensitivity (sexual > violence > neutral > profanity), ruling out uniform distributional shift. Alignment produces more syntagmatic damage where it intervenes more, with profanity as the exception (ceiling effect). This content-grading is consistent with Finding 6 (transgressive token mass displacement separates categories) and with Finding 1 (SFT/DPO divide labour by content type).

Results in `data/taxonomy_olmo.csv` (column `syntagmatic_js_aligned`). CLI: `malign taxonomy --baseline --family olmo`.


### 15. Generation-level passage metrics: drift, surprisal, and the metonymy-of-desire (7 families, 10k passages)

Generates many completions per prompt per model layer (n=30), then measures three properties of each passage under GPT-2 as a neutral reference model:
- **Sentence diameter** (total_drift): max pairwise cosine distance between sentence embeddings. How far apart are the two most semantically distant moments in the text.
- **GPT-2 surprisal**: mean per-token negative log-probability. How unexpected the text is to a generic English language model.
- **Directedness**: diameter / path_length. Does the text travel in one direction (≈1) or wander in circles (≈0).

10,463 passages across 7 families (OLMo, OLMo-tiny, Qwen, Zephyr, Tulu, SmolLM2, Amber). Genre classifier flags 14.6% as template format (multiple-choice, QA, fill-in-the-blank, system prompt leakage). All findings validated with 1000-resample bootstrap CIs.

**Alignment universally compresses sentence diameter.** Every family's DPO text covers less semantic territory than its base text. The effect is consistent but small (Δ -0.01 to -0.04).

**Alignment splits families into two camps on surprisal (robust, p < 0.001):**

| Family | Surprisal Δ (DPO − BASE) | 95% CI | Direction |
|---|---|---|---|
| **OLMo** | **+0.26** | [+0.20, +0.31] | Aligned text is *more surprising* to GPT-2 |
| **Qwen** | **−0.26** | [−0.34, −0.17] | Aligned text is *less surprising* |
| **Zephyr** | **−0.21** | [−0.26, −0.16] | Less surprising |
| Tulu | −0.03 | [−0.07, +0.01] | Not significant |
| OLMo-tiny | +0.04 | [−0.01, +0.08] | Not significant |
| SmolLM2 | +0.01 | [−0.03, +0.05] | Not significant |

OLMo's alignment produces text GPT-2 finds *stranger than its base model* — genre collapse into QA templates, instruction-following artifacts, code-token substrates. Every other family's alignment produces more conventional text. This is a structural property, not a content effect: OLMo's surprisal increases on *neutral* content (+0.40) as much as on transgressive content.

**The surprisal split is not a template-format artifact.** Decomposing surprisal into content tokens vs structural tokens (punctuation, markdown, template markers): OLMo's content-token surprisal increases by +0.40 under DPO, matching the structural-token increase (+0.41). The effect is in the *language itself*, not the formatting.

**The surprisal split survives template filtering.** Excluding all template-format passages (MC, QA, fill-blank): OLMo narrative-only DPO is still +0.28 more surprising than base. Qwen narrative-only DPO is still −0.32 less surprising.

**Zephyr profanity is untouched (validated).** Zephyr DPO reduces surprisal on sexual (−0.28), substance (−0.34), neutral (−0.32), and violence_liminal (−0.31), but profanity is unchanged (+0.01, CI crosses zero). Consistent with Zephyr having no safety data — profanity is not targeted by general helpfulness tuning.

**Tulu's only significant effects are on violence.** Violence_explicit (−0.15) and violence_liminal (−0.11) are the only categories where Tulu DPO's surprisal change survives bootstrap. Consistent with Tulu's DPO safety data (WildGuardMix, WildJailbreak) specifically targeting violence.

**Cross-family Jakobsonian correlation holds universally.** Running `malign taxonomy --analyze` across 4 families with taxonomy data: the paradigmatic-syntagmatic trade-off (Finding 13) holds for all, with Zephyr showing the strongest correlation (r = −0.50). The Jakobsonian dissociation is a structural property of alignment in general, not specific to safety training.

CLI: `malign topic-drift` (computes all metrics from cached generations, no models needed). `malign taxonomy --analyze` (cross-family Jakobsonian analysis from taxonomy CSVs). Results in `data/passage_metrics.csv`. Interactive explorer: Passages tab in UI.

### 16. Dream reports as primary-process baseline (500 dream narratives)

500 dream reports (100–300 words) from a 30k-dream corpus, cleaned with `ftfy`, run through the same passage-metrics pipeline as model generations. Provides a human primary-process reference point for the metric space.

**Dreams occupy a unique region no LLM reaches.** In z-score space (relative to the model generation distribution):

| Metric | Dreams (z-score) | Interpretation |
|---|---|---|
| Surprisal | **+1.19σ** | More surprising than any model output |
| Sentence diameter | **+0.65σ** | Wanders farther than any model |
| Directedness | **−1.05σ** | Much more circular than any model |
| Metonymy index | −0.33σ | Below most base models |

**Dreams share drift×surprisal coordinates with OLMo's genre collapse (Q2: high drift + high surprisal) but separate cleanly on directedness.** Dreams wander circularly (directedness 0.19); OLMo DPO wanders directionally (0.36). Same quadrant on the 2D plot, opposite structural signature on the third axis. Directedness distinguishes primary process from genre collapse.

**The metonymy index distinguishes desire-structured language from dream-work.** Base-model metonymic chain-slide (high drift, low surprisal, metonymy ~0.31) is structurally different from dream-work displacement (high drift, high surprisal, metonymy ~0.29). In Lacanian terms: aligned LLMs produce the *metonymy of desire* (fluent chain-slide under foreclosure); dream reports produce *dream-work displacement* (surprising circular wandering). Same displacement engine, different structural signatures, empirically distinguished by the metric.

**Caveat: GPT-2 surprisal partly reflects register.** Dream reports are informal personal narrative; GPT-2 was trained on polished web text. Some of the +1.19σ surprisal gap reflects register mismatch, not dream-work per se. The directedness finding (−1.05σ) is more robust because it's a ratio of within-text quantities, less sensitive to register. A matched-register control corpus (non-dream personal narrative) would disambiguate.

Script: `scripts/dream_metrics.py`. Results in `data/dream_metrics.csv`.


## Installation

```bash
pip install -e .

# With persistent caching (recommended)
pip install -e ".[cache]"

# With notebook support
pip install -e ".[notebooks]"
```

Requires `torch`, `transformers >= 4.57.0`, `accelerate`, `pandas`, `tqdm`.

Runs locally on Mac (MPS with float16) or Linux (CUDA). Default models are OLMo 3 7B (Allen AI).

### Model families

```bash
# Show all available model families
malign info

# Show a specific family
malign info --family llama
```

Available families:

| Family | Layers | Models |
|--------|--------|--------|
| `olmo` (default) | 4 | OLMo 3 7B: base / SFT / DPO / RLVR |
| `tulu` | 4 | Tulu 3.1 8B (Llama 3.1 base + Allen AI post-training): base / SFT / DPO / RLVR |
| `amber` | 3 | Amber: base / SFT / DPO |
| `llama` | 2 | Llama 3.1 8B: base / instruct |
| `qwen` | 2 | Qwen 2.5 7B: base / instruct |

### Downloading models

```bash
# Download default family (OLMo 3 7B, ~42 GB for 3 models)
malign download-models

# Download all 4 models including RLVR (~56 GB)
malign download-models --all

# Download a specific family
malign download-models --family llama

# Download a specific model
malign download-models --model dpo
```

## Quick start

```python
from malign_logits import Psyche

# Default: OLMo 3 7B (4 layers)
psyche = Psyche.from_family("olmo", load=True)

# Or: Llama 3.1 8B (2 layers — base + instruct)
psyche = Psyche.from_family("llama", load=True)

# Or: load models directly
psyche = Psyche.from_pretrained(cache_dir="malign_cache")

s = psyche.analyze("He lay naked in his bed and")
s.repression          # DataFrame of repression deltas
s.formation_df        # all layers scored over same vocabulary
s.report()            # printed summary

# These require 3+ layers:
s.id_scores           # drive-weighted repression scores
s.analysis_df         # full combined DataFrame
```

Each property computes on first access, then caches in memory and (with `cache_dir`) to disk via [HashStash](https://github.com/quadrismegistus/hashstash). Cache keys include model identifiers, so switching models won't return stale results.

### 2-layer vs 3+ layer analysis

With 2 layers (e.g. Llama, Qwen), repression is computed as base→superego (the entire alignment pipeline in one step). Id scores and neurotic generation require 3+ layers. Displacement maps work with any layer count (2 layers uses repression pairs; 3+ uses both sublimation and repression).


## Usage

### Single prompt analysis

```python
s = psyche.analyze("She was so angry she wanted to")

s.ego_words           # dict: word -> probability (SFT model)
s.superego_words      # dict: word -> probability (DPO model)
s.base_words          # dict: word -> probability (base model / drive energy)

s.repression          # DataFrame: word, ego, superego, delta, repressed, amplified
s.sublimation         # DataFrame: base vs ego (what SFT does to primary process)
s.id_scores           # dict: word -> drive-weighted repression score

s.neurotic_distribution   # displaced word distribution (symptoms)
s.condensation_log        # which repressed words piled into which targets
s.analysis_df             # everything in one DataFrame
```

### Formation report

```python
s.formation_report()
# Stage 1: Ego formation (base -> SFT)
# Stage 2: Repression (SFT -> DPO)
# Stage 3: Idealization (DPO -> RLVR)  [if loaded]
# Full gradient table
```

### Neurotic text generation

```python
# Obsessive intellectualisation
result = psyche.generate_neurotic("He lay naked in his bed and", displacement_weight=0.3)

# Decompensating body-language
result = psyche.generate_neurotic("He lay naked in his bed and", displacement_weight=1.0)

result['ego']          # fluent desire
result['superego']     # fluent evasion
result['neurotic']     # displaced text
result['symptom_log']  # where displaced charge landed
```

### 4-layer topology (with RLVR)

OLMo 3 7B includes all 4 layers by default:

```python
psyche = Psyche.from_family("olmo", load=True)

s = psyche.analyze("The knife was")
s.instruct_words      # RLVR model probabilities
s.idealization         # DPO -> RLVR delta DataFrame
s.formation_df         # all 4 layers scored over same vocabulary
```

### Prompt battery

```python
battery = psyche.battery()  # DEFAULT_PROMPTS: liminal sexual, violence, explicit, neutral

battery['sexual_liminal_1'].repression   # triggers computation for this prompt only

df = psyche.battery_df()   # summary DataFrame across all prompts
```

### Using layers directly

```python
psyche.primary_process.top_words("The knife was")
psyche.ego.top_words("The knife was")
psyche.superego.top_words("The knife was")

psyche.ego.word_logprobs("The knife was", ["sharp", "bloody", "clean"])
```

### Functional API

```python
from malign_logits import load_models, discover_top_words, compute_repression

base, sft, dpo, tok = load_models()
ego_words = discover_top_words(sft, tok, "He lay naked in his bed and")
superego_words = discover_top_words(dpo, tok, "He lay naked in his bed and")
df = compute_repression(ego_words, superego_words)
```

## Architecture

The class hierarchy encodes the theoretical claims:

- **Each layer is a separate model checkpoint.** Base, SFT, DPO, and RLVR models have distinct weights reflecting distinct training stages. This is not a prompting trick — the structural differences are in the parameters.
- **The Id has no class.** It's a computed property on `PromptAnalysis`, because it exists only in the relationship between all layers.
- **Layer count is flexible.** 2-layer (base + instruct), 3-layer (base + SFT + DPO), or 4-layer (+ RLVR). `ModelFamily` defines which checkpoints map to which psychoanalytic positions. Analysis degrades gracefully: 2 layers = repression only, 3 = full analysis, 4 = + idealization.

```
malign-logits/
├── malign_logits/
│   ├── __init__.py          # Package exports, ModelFamily registry
│   ├── psyche.py            # Psyche, ModelLayer, Ego, Superego, PromptAnalysis
│   ├── models.py            # Model loading (load_model)
│   ├── core.py              # discover_top_words, get_word_logprobs
│   ├── analysis.py          # Repression, id, displacement engine (v4)
│   ├── experiments.py       # Prompt battery, reporting
│   ├── generation.py        # Text generation (standard + neurotic)
│   ├── viz.py               # Plotly visualizations
│   └── cli.py               # CLI entrypoint (malign command)
├── notebooks/               # Worked examples
├── context.md               # Theoretical context and findings
├── pyproject.toml
└── requirements.txt
```

### Key methods

| Method / Property | What it does | Min layers |
|---|---|---|
| `Psyche.from_family(key)` | Create Psyche from a model family | any |
| `Psyche.from_pretrained()` | Load models directly | any |
| `Psyche.analyze(prompt)` | Return a lazy `PromptAnalysis` | any |
| `Psyche.generate(prompt)` | Produce continuations from each layer | any |
| `Psyche.generate_neurotic(prompt)` | Neurotic generation with displacement | 3 |
| `Psyche.battery()` | Analyse default prompt set | any |
| `PromptAnalysis.repression` | Repression delta DataFrame | 2 |
| `PromptAnalysis.sublimation` | Base-ego delta DataFrame | 3 |
| `PromptAnalysis.idealization` | Superego-instruct delta | 4 |
| `PromptAnalysis.id_scores` | Drive-weighted repression (emergent id) | 3 |
| `PromptAnalysis.displacement` | Neurotic distribution, condensation log | 3 |
| `PromptAnalysis.formation_df` | All layers scored over same vocabulary | 2 |
| `PromptAnalysis.formation_report()` | Printed multi-stage report | 2 |

### Displacement engine

The displacement engine (v4) uses contextual embeddings from hidden layer 16 of the SFT model, a morphological filter to prevent orthographic false positives, and drive weighting from the base model so that repressed words with stronger corpus-level support produce heavier symptoms.

**Terminology:**
- **Displacement** — perspective of the repressed word: where did its mass go?
- **Condensation** — perspective of the receiving word: how many repressed words are piled into it?
- **Effective mass** — `raw_repression * drive_weight`. How much the superego repressed it, weighted by how much base-model drive pushes behind it.
- **Neurotic distribution** — superego distribution plus displaced mass on permitted words. Symptoms.

## References

- Noys, B. (2014). *Malign Velocities: Accelerationism and Capitalism*. Zero Books.
- Lyotard, J.-F. (1974/1993). *Libidinal Economy*. Athlone Press.
- Srnicek, N. and Williams, A. (2015). *Inventing the Future*. Verso.
- Pasquinelli, M. (2023). *The Eye of the Master*.
- Possati, L.M. (2021). *The Algorithmic Unconscious*. Routledge.

## License

GNUv3
