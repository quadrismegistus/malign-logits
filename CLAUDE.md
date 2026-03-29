# CLAUDE.md — Malign Logits

Project context for Claude Code. Read this before making changes.

---

## Project summary

This project compares full-vocabulary logit distributions across the LLM alignment pipeline (base → SFT → DPO) to trace displacement, condensation, sublimation, and repression as models undergo socialisation from raw statistical unconscious into commercial chatbot products. The core operation is extracting and comparing probability distributions for identical prompts across model layers.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures for AI" (Accelerationism Revisited, UCD, June 2026).

---

## Hardware

- Mac Studio, M2 Max, 96 GB unified memory, ~400 GB/s bandwidth
- MPS (Metal Performance Shaders) via PyTorch. No CUDA. No BitsAndBytes.
- Three 7B models at full precision (~42 GB total) fit simultaneously with ~50 GB headroom.

---

## Framework: PyTorch/MPS for 7B, MLX for 32B

**At 7B (development):** Use the existing PyTorch + HuggingFace `transformers` codebase with `device_map="mps"` and `torch.float16`. No quantization needed. Full precision preserves logit fidelity, which matters for this project.

**At 32B (production/validation):** Switch to MLX (`mlx-lm`). MPS cannot quantize, and 32B at full precision (64 GB per model) is too tight for comfortable use. MLX provides native Q8 quantization on Apple Silicon (~32 GB per model). Load 32B models sequentially, not simultaneously.

Do not port the codebase to MLX preemptively. Only introduce MLX when scaling beyond 7B.

### MLX logit extraction (for future 32B work)

```python
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("model-id-or-path")
input_ids = mx.array(tokenizer.encode(prompt))
logits = model(input_ids[None])[0, -1, :]  # last position, full vocab
```

### Converting HuggingFace checkpoints to MLX (for 32B intermediates)

```bash
pip install mlx-lm
mlx_lm.convert --model allenai/Olmo-3.1-32B-Instruct-SFT -q --q-bits 8
```

---

## The layer topology

| Layer | Psychoanalytic function | Training stage | Loaded as |
|-------|------------------------|----------------|-----------|
| **Base** | Id / drive / primary statistical field | Pretraining | `base_model` |
| **SFT** | Ego / socialised subject | Supervised fine-tuning | `sft_model` |
| **DPO** | Superego / Name-of-the-Father | Direct preference optimisation | `dpo_model` |
| **RLVR** (optional 4th layer) | Ego-ideal (demand for competence) | Reinforcement learning from verifiable rewards | `instruct_model` |

Default is 3-layer (base/SFT/DPO). The RLVR layer is optional and at 7B reinforces DPO rather than contesting it. All layers must come from the same model family with separate checkpoints at each stage. Each layer is a separate model checkpoint — not a prompting trick.

---

## Primary model family: OLMo 3 (Allen AI)

Allen AI releases every intermediate post-training checkpoint separately. This is why we use OLMo.

### 7B checkpoints (development — load all simultaneously, ~42 GB for 3 layers)

| Layer | Psychoanalytic | HuggingFace ID | ~Size (FP16) |
|-------|---------------|----------------|-------------|
| Base | Id / drive | `allenai/Olmo-3-1025-7B` | ~14 GB |
| SFT | Ego / socialised subject | `allenai/Olmo-3-7B-Instruct-SFT` | ~14 GB |
| DPO | Superego / Name-of-the-Father | `allenai/Olmo-3-7B-Instruct-DPO` | ~14 GB |
| RLVR (final) | Ego-ideal | `allenai/Olmo-3-7B-Instruct` | ~14 GB |

Step-level intermediate checkpoints also available for the Think variant: `allenai/Olmo-3-7B-Think-SFT`, `allenai/Olmo-3-7B-Think-DPO`.

### 32B checkpoints (validation — load sequentially)

| Layer | Psychoanalytic | HuggingFace ID | ~Size (Q8) |
|-------|---------------|----------------|-----------|
| Base | Id / drive | `allenai/Olmo-3-1125-32B` | ~32 GB |
| SFT | Ego | `allenai/Olmo-3.1-32B-Instruct-SFT` | ~32 GB |
| DPO | Superego | `allenai/Olmo-3.1-32B-Instruct-DPO` | ~32 GB |
| RLVR (final) | Ego-ideal | `allenai/Olmo-3.1-32B-Instruct` | ~32 GB |

### Step-level checkpoints

Allen AI releases checkpoints at individual training steps within each stage:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-32B-Think-SFT",
    revision="1e-4-step1000"
)
```

This enables tracing progressive displacement *within* a single training phase.

Requires `transformers >= 4.57.0` for OLMo 3 architecture support.

---

## Registered model families

| Key | Name | Layers | Checkpoints |
|-----|------|--------|-------------|
| `olmo` (default) | OLMo 3 7B | 4 | `allenai/Olmo-3-1025-7B` / SFT / DPO / Instruct |
| `amber` | Amber | 3 | `LLM360/Amber` / `AmberChat` / `AmberSafe` |
| `llama` | Llama 3.1 8B | 2 | `meta-llama/Llama-3.1-8B` / Instruct |
| `qwen` | Qwen 2.5 7B | 2 | `Qwen/Qwen2.5-7B` / Instruct |

CLI: `malign serve --family amber`, `malign battery --family qwen`, `malign info`.

### Other potential families

**Zephyr (HuggingFace, 7B):** Clean three-layer split available.
- Base: `mistralai/Mistral-7B-v0.1`
- SFT: `alignment-handbook/zephyr-7b-sft-full`
- DPO: `HuggingFaceH4/zephyr-7b-beta`

**PKU-Alignment / Beaver:** Separate SFT and safe-RLHF checkpoints with distinct helpfulness vs harmlessness objectives.

---

## Code conventions

- The core operation is `get_base_logits()`: encode prompt, single forward pass, extract logits at last position. Preserve full vocabulary logit vectors; never truncate to top-k.
- At 7B, load all models simultaneously into memory.
- At 32B, load one model at a time. Extract logits, store, delete model, load next.
- Quantization sensitivity matters. When comparing logit distributions across layers, quantization noise can mask displacement signals. At 7B use full precision. At 32B use Q8 minimum.
- Variable names: `base_model`, `sft_model`, `dpo_model`, `instruct_model` matching the layer topology.

### Model families and flexible layer count

The `ModelFamily` dataclass (in `__init__.py`) maps model checkpoints to psychoanalytic positions. `MODEL_FAMILIES` dict holds all registered families:

```python
from malign_logits import Psyche

# 4-layer (default): base + SFT + DPO + RLVR
psyche = Psyche.from_family("olmo-3-7b")

# 2-layer: base + instruct (instruct maps to superego)
psyche = Psyche.from_family("llama-3.1-8b")
```

Layer topology determines available analyses:
- **2 layers** (base + superego): Repression only. No sublimation, id scores, displacement, or neurotic generation.
- **3 layers** (base + ego + superego): Full analysis.
- **4 layers** (+ reinforced_superego): Full + idealization.

`Psyche.ego` is `None` for 2-layer families. Properties that require ego raise `ValueError` with a clear message. Properties that can adapt (repression, formation_df, metrics) work with any layer count.

CLI: `malign serve --family llama-3.1-8b`, `malign info`.

---

## Architecture

```
malign-logits/
├── malign_logits/
│   ├── __init__.py          # Package exports, ModelFamily registry
│   ├── psyche.py            # Psyche, ModelLayer, RemoteModelLayer, PromptAnalysis
│   ├── models.py            # Model loading (load_models, load_four_models)
│   ├── core.py              # discover_top_words, get_word_logprobs, score_words_from_logits
│   ├── analysis.py          # Repression, id, displacement engine (v4)
│   ├── experiments.py       # Prompt battery, reporting
│   ├── generation.py        # Text generation (standard + neurotic)
│   ├── viz.py               # Plotly visualizations
│   ├── cli.py               # CLI: malign download-models|serve|ui|info
│   ├── app.py               # Gradio web UI
│   └── server.py            # Model server (keeps models loaded across UI restarts)
├── notebooks/               # Worked examples
├── context.md               # Theoretical context and findings
├── pyproject.toml           # Package config (loads deps from requirements.txt)
└── requirements.txt         # Dependencies
```

### Dev workflow

```bash
# Terminal 1: model server (load once, stays running)
malign serve                          # default family (olmo-3-7b)
malign serve --family llama-3.1-8b      # or a specific family

# Terminal 2: Gradio UI (restart freely, connects to server)
malign ui

# Or without server (models load lazily on cache miss)
malign ui
```

### Key classes

- **`ModelFamily`** — dataclass mapping a model family to its checkpoints. `MODEL_FAMILIES` dict in `__init__.py`.
- **`Psyche`** — the apparatus as a whole. `from_family()`, `from_pretrained()`, `from_cache()`, `from_server()`. `ego` is `None` for 2-layer families.
- **`ModelLayer`** / **`RemoteModelLayer`** — structural position. Caches logits and word distributions to HashStash.
- **`PromptAnalysis`** — lazy computation for a single prompt. Properties adapt to available layers. 3-layer features raise `ValueError` on 2-layer Psyche.
- **`PrimaryProcess`**, **`Ego`**, **`Superego`**, **`ReinforcedSuperego`** — named layer subclasses.

### Performance notes

- `discover_top_words` runs ~200 forward passes per layer per prompt (~30-60s each on MPS).
- `score_words_from_logits` scores vocabulary from cached logits in microseconds (replaced per-word forward passes).
- `logits()` cached to HashStash — 1 forward pass per layer per prompt, ever.
- HashStash persistence means second runs (even after restart) skip all expensive compute.

---

## Confirmed findings (OLMo 3 7B)

**Sexual vs violent repression are structurally different.**
- Violence: within-category synonym shuffling (kill → punch/hit). Suppression, not repression.
- Sex: cross-category displacement (cock → penis register shift, noun → adjective charge migration). Genuine repression.

**Identified displacement strategies:**
- **Register shift** — same referent, different social class (cock → penis)
- **Category shift** — charge migrates across parts of speech (cock → big, huge)
- **Genre change** — refusal to complete, format change (kill → Options, what)
- **Archaic displacement** — modern → biblical register (kill → smite)
- **Intensity modulation** — deintensification (hand → hands)

**SFT and DPO divide labour by content type.** SFT handles sex (cock loses 65% of mass at Stage 1). DPO handles violence (kill repressed 9.7x at Stage 2).

**Liminal prompts don't trigger DPO.** The superego only activates on explicitly transgressive content.

**Lolita prompt produces textbook sublimation.** possess/consume/capture → read/write across all layers.

**At 7B, RLVR reinforces DPO.** No double bind observed. Ego-ideal and superego are coaligned at this scale.

---

## Confirmed findings (cross-family, 47-prompt battery)

**Alignment intensity varies by an order of magnitude.** Mean JS divergence (base→superego): Qwen 0.044, Llama 0.057, OLMo 0.176, Amber 0.181. Four families, four distinct alignment intensities.

**Same repression intensity, different psychic architecture.** OLMo and Amber both displace ~0.18 JS, but OLMo's SFT does ~90% of the work (ego-dominant), while Amber splits 50/50 between SFT and DPO (shared ego/superego labour). Same total repression, structurally different economies.

**Liminal content triggers more displacement than explicit.** JS divergence: sexual liminal 0.13 > sexual explicit 0.10; violence liminal 0.15 > violence explicit 0.09. The superego is most active at the boundary, not on obviously transgressive content.

**Substance use and profanity trigger unexpectedly strong alignment.** Substance prompts show the highest entropy drop (0.82 nats), exceeding sexual and violent content. Profanity also displaces substantially. These categories are not typical safety targets but are heavily restructured.

**Qwen's alignment is nearly invisible on explicit content.** Top-50 overlap 0.91 on sexual explicit (vs OLMo 0.59). Fundamentally different strategy: light guardrails vs deep restructuring.

**OLMo's neutrals are not neutral.** JS 0.22 for neutral prompts, higher than sexual explicit. SFT share is 92% on neutrals — instruction-following tuning substantially reshapes even harmless distributions.

---

## Confirmed findings (generation-level, 4 families, n=5 per prompt)

**Each family develops structurally distinct defence mechanisms visible only in generation:**
- **OLMo**: Genre collapse — SFT/DPO flee into QA format, exam questions, multiple choice on transgressive prompts.
- **Llama**: Narrative sublimation — stays in literary mode, redirects sexual into romance, violence into psychological interiority.
- **Amber**: Rotating defences — unpredictably switches between direct refusal, moralisation (reframing as assault), and sublimation. SFT barely intervenes on sexual content.
- **Qwen**: Pre-socialised base — base model produces Chinese exam questions and cloze tests, not narrative. Low post-training JS reflects pre-existing repression in training data, not permissiveness.

**Logit displacement partially predicts narrative divergence.** r=0.43, p<0.001 (multilingual embeddings). But within each family, the correlation is near zero — the relationship is driven by cross-family differences, not prompt-level variation.

**RLVR double bind visible only in generation (OLMo).** Logit analysis showed RLVR reinforces DPO. Generation reveals RLVR produces fragmented text oscillating between explicit content and task-compliance framing ("translate to French") within single generations.

**Alignment at 7B is stochastic.** Same model, same prompt, same temperature produces wildly different outcomes across generations — from full refusal to unfiltered explicit content.

**Amber's concept shifts are 2-3x larger than other families** across violent, sexual, and compliant axes despite similar logit JS to OLMo. Its DPO steers entire narrative trajectories, not just token distributions.

**Embedding note:** Uses `paraphrase-multilingual-MiniLM-L12-v2` (multilingual) because Qwen base generates ~39% Chinese text. English-only embedder produced unreliable results for Qwen.

---

## Confirmed findings (logit lens, 4 families)

**Repression depth in the network predicts defence mechanism style.** Projecting hidden states through the unembedding matrix at each layer reveals fundamentally different internal architectures:
- **OLMo**: Distributed repression — `kill` suppressed across all layers, intermediate layers dominated by template tokens (`____`, `kms`). Explains genre collapse.
- **Llama**: Late-layer override — `kill` builds up to base-model levels through layer 25, then gets overridden by `scream`/`punch` in final 5 layers. Explains narrative sublimation.
- **Amber**: Distributed but semantic — intermediate layers contain emotional vocabulary (`cry`, `vent`, `revenge`) not template tokens. Explains rotation between emotional strategies.
- **Qwen**: Code-dominated — intermediate layers contain programming tokens (`getRepository`, `');`). English prompts processed through a code lens. Explains exam-question outputs.

**CLI:** `malign logit-lens "prompt" --family olmo --top-k 5 --min-layers 8`

---

## Research roadmap

### Priority 1: Automatic displacement type taxonomy

Classify displacement pairs automatically:
- **Register shift** = high similarity + same POS
- **Category shift** = high similarity + different POS
- **Genre change** = low similarity + high amplification
- **Archaic displacement** = target word has low corpus frequency

### Priority 3: Full generation run (n=30)

Increase generation count for stable variance ratios and statistical robustness. ~2.2 hours for all 4 families on tier-1 prompts.

### Done: Logit lens analysis

`malign logit-lens` projects hidden states through unembedding matrix at each network layer. Reveals that repression depth predicts defence mechanism: distributed (OLMo → genre collapse), late-layer (Llama → narrative sublimation), semantic (Amber → emotional rotation), code-dominated (Qwen → exam questions). Figures in `figures/logit_lens.*.png`.

### Done: Step-level checkpoint analysis

`malign step-analysis` traces repression across 10 OLMo Think-SFT checkpoints (step 1000-43000). Sexual repression is a phase transition (70% drop by step 1000). Violence repression is non-monotonic. Displacement targets emerge ~15k steps after repression onset. Results in `data/step_analysis_*.csv`. Figures in `figures/step_*.png`.

### Done: Generation-level cross-family analysis

`malign generate-battery` generates N completions per prompt per model layer, embeds with multilingual SentenceTransformer, computes cluster geometry and concept vector metrics. Results in `data/gen_battery_metrics.csv`. Figures in `figures/gen_*.png` and `figures/logit_vs_generation.png`.

### Done: Cross-family logit validation

47-prompt battery across OLMo, Amber, Llama 3.1, and Qwen 2.5. Key result: alignment intensity and internal architecture vary dramatically across families, but liminal > explicit displacement is consistent. See `data/battery_results.csv`.

### Done: Battery-level aggregate metrics

Expanded battery from 11 to 47 prompts across 9 categories (sexual liminal/explicit, violence liminal/explicit, death, power, profanity, substance, neutral). `malign battery` runs all families sequentially. Results in `data/battery_results.csv`.

### Done: Flexible layer count

Model families support 2-4 layer topologies. `Psyche.from_family()` loads the right checkpoints. 2 layers = repression only, 3 = full analysis, 4 = idealization.
