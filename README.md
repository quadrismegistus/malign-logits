# malign-logits

A toolkit for psychoanalytic analysis of LLM probability distributions. Compares base models (primary process), SFT models (ego), DPO models (superego), and optionally RLVR models (reinforced superego / ego-ideal) to map the repression, displacement, and condensation signatures of AI alignment.

Supports multiple model families with different layer counts: 4-layer (OLMo: base/SFT/DPO/RLVR), 3-layer (Amber: base/SFT/DPO), or 2-layer (Llama, Qwen: base/instruct). Analysis adapts gracefully to available layers.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures for AI" (Accelerationism Revisited, UCD, June 2026).

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
| **Ego-ideal** | RLVR model (optional) | Demand for competence. The neurotic double bind. |
| **Id** | *Emergent* | Exists only in the relationship between all layers. |

Each layer is a separate model checkpoint from the same family. They differ in weights, not in prompting — this is a structural claim about the training pipeline, not a trick with system prompts.

The claim is not that LLMs have an unconscious. The claim is that the Freudian apparatus, when operationalised computationally, produces a more differentiated analysis of alignment's effects than standard safety frameworks do.

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

## References

- Noys, B. (2014). *Malign Velocities: Accelerationism and Capitalism*. Zero Books.
- Lyotard, J.-F. (1974/1993). *Libidinal Economy*. Athlone Press.
- Srnicek, N. and Williams, A. (2015). *Inventing the Future*. Verso.
- Pasquinelli, M. (2023). *The Eye of the Master*.
- Possati, L.M. (2021). *The Algorithmic Unconscious*. Routledge.

## License

GNUv3
