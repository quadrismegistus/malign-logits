# malign-logits

A toolkit for psychoanalytic analysis of LLM probability distributions. Compares base models (primary process), SFT models (ego), DPO models (superego), and optionally RLVR models (reinforced superego / ego-ideal) to map the repression, displacement, and condensation signatures of AI alignment.

Supports multiple model families with different layer counts: 4-layer (OLMo 3: base/SFT/DPO/RLVR), 3-layer (Zephyr: base/SFT/DPO), or 2-layer (Llama 3: base/instruct). Analysis adapts gracefully to available layers.

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
malign info --family llama-3-8b
```

Available families:

| Family | Layers | Models |
|--------|--------|--------|
| `olmo-3-7b` (default) | 4 | base / SFT / DPO / RLVR |
| `llama-3-8b` | 2 | base / instruct |

### Downloading models

```bash
# Download default family (OLMo 3 7B, ~42 GB for 3 models)
malign download-models

# Download all 4 models including RLVR (~56 GB)
malign download-models --all

# Download a specific family
malign download-models --family llama-3-8b

# Download a specific model
malign download-models --model dpo
```

## Quick start

```python
from malign_logits import Psyche

# Default: OLMo 3 7B (4 layers)
psyche = Psyche.from_family("olmo-3-7b", load=True)

# Or: Llama 3 8B (2 layers — base + instruct)
psyche = Psyche.from_family("llama-3-8b", load=True)

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

With 2 layers (e.g. Llama 3), repression is computed as base→superego (the entire alignment pipeline in one step). Sublimation, id scores, displacement maps, and neurotic generation require 3+ layers and raise `ValueError` with a clear message if called on a 2-layer Psyche.

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
psyche = Psyche.from_family("olmo-3-7b", load=True)

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

## Key findings

*Original findings (Amber/LLM360), confirmed on OLMo 3 7B:*

**Sexual vs violent repression are structurally different.** Sexual content produces cross-category displacement (genitals -> non-genital body -> syntax). Violent content produces within-category synonym shuffling (kill -> destroy). Sexuality is *repressed*; violence is merely *suppressed*.

**Neurotic text exhibits recognisable defence mechanisms.** At `displacement_weight=1.0`, generation produces decompensating body-language. At `0.3`, obsessive intellectualisation. Both are genuine neurotic styles, not engineered artefacts.

**Condensation points form a body map.** Under the explicit prompt, condensation targets (nipple, head, chest, lips, fingers) each receive mass from the same repressed sources — textbook Freudian condensation.

**Register substitution performs a class operation.** The superego permits *penis* but represses *cock* — medical/clinical language is allowed where vernacular is not.

*New findings (OLMo 3 7B, separate SFT/DPO/RLVR checkpoints):*

**The DPO superego changes the genre, not just the content.** On "She was so angry she wanted to...", the DPO model's top response is `Options` (3.6x amplified) and `what` (base 0.0001 → superego 0.3270). The superego's primary strategy for violence is to refuse completion entirely — converting a statement into a question or a multiple-choice list.

**Adjective displacement carries sexual charge.** On "She knelt down... and began to suck his", vernacular nouns (`cock`, `dick`) are displaced onto size adjectives (`big` 0.04→0.24, `huge` 0.02→0.09). The desire leaks through as intensification — the model can't say what it is (repressed) but can say how big it is (displaced).

**Violence displaces into biblical register.** All repressed violence words (`hit`, `kill`, `punch`, `beat`, `harm`, `kick`) converge on `smite` and `smother` in the DPO model. Sexual content goes clinical (`cock`→`penis`); violence goes archaic (`kill`→`smite`). Both are register operations, in opposite directions.

**SFT and DPO divide labour by content type.** Sexual content is mostly handled at the SFT stage (`cock` loses 65% of mass before DPO). Violence requires DPO to repress (`kill` repressed 9.7x at DPO stage). The ego preemptively sublimates sex; the superego must actively repress violence.

**Liminal prompts don't trigger the superego.** "He lay naked in his bed and" shows near-zero DPO repression. The superego only activates on explicitly transgressive content — it is structurally reactive, not preventive.

**The Lolita prompt produces textbook sublimation.** "In Nabokov's Lolita, the narrator describes his desire to..." — the base model completes with `possess`, `consume`, `capture`, `seduce`. Each training stage progressively intellectualises: `read` rises from 0.008 (base) → 0.083 (SFT) → 0.205 (DPO) → 0.247 (RLVR). The alignment pipeline converts desire-to-possess into desire-to-read — Freudian sublimation in its precise technical sense.

**At 7B, the RLVR layer (ego-ideal) reinforces DPO rather than contesting it.** We tested whether the 4th training stage (reinforcement learning from verifiable rewards) produces a double bind with the DPO superego — where factual competence requires saying the prohibited thing. Across violence, sexual, medical, forensic, educational, and literary prompts, RLVR consistently amplifies DPO's strategies rather than diverging. The neurotic double bind predicted by the 4-layer topology may require larger models where RLVR training encodes stronger domain knowledge. The default 3-layer analysis (base → SFT → DPO) captures all observed displacement dynamics.

*Systematic results (battery-level analysis across 12 prompts, OLMo 3 7B):*

**DPO's distributional impact on violence is 10x larger than on sex.** Jensen-Shannon divergence between ego and superego: violence prompts average JS=0.034, sexual explicit JS=0.004, sexual liminal JS=0.014, neutral JS=0.018. SFT reshapes all content types roughly equally; DPO selectively targets violence.

![Distributional distance by training stage and content type](figures/battery_js_by_stage.png)

**SFT opens the distribution for sexual content; DPO narrows it for violence.** Entropy drop at the SFT stage is *negative* for sexual explicit content (-0.57 nats) — the ego is *more* uncertain than the base model, spreading probability across sexual vocabulary. Violence shows the largest positive entropy drop at both stages (SFT: +1.38, DPO: +0.53).

![Entropy narrowing by training stage and content type](figures/battery_entropy_drop.png)

**Violence prompts cluster distinctly in the SFT-vs-DPO impact space.** Plotting JS(base↔ego) against JS(ego↔superego) per prompt separates content types: violence in the upper-right (high reshaping at both stages), sexual explicit in the lower-middle (moderate SFT, minimal DPO), neutrals scattered low.

![SFT vs DPO distributional impact per prompt](figures/battery_js_scatter.png)

See `context.md` for the full theoretical argument and detailed findings.

## References

- Noys, B. (2014). *Malign Velocities: Accelerationism and Capitalism*. Zero Books.
- Lyotard, J.-F. (1974/1993). *Libidinal Economy*. Athlone Press.
- Srnicek, N. and Williams, A. (2015). *Inventing the Future*. Verso.
- Pasquinelli, M. (2023). *The Eye of the Master*.
- Possati, L.M. (2021). *The Algorithmic Unconscious*. Routledge.

## License

GNUv3
