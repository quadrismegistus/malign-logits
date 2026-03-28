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

## Other usable model families

**Zephyr (HuggingFace, 7B):** Clean three-layer split available.
- Base: `mistralai/Mistral-7B-v0.1`
- SFT: `alignment-handbook/zephyr-7b-sft-full`
- DPO: `HuggingFaceH4/zephyr-7b-beta`

**PKU-Alignment / Beaver:** Separate SFT and safe-RLHF checkpoints with distinct helpfulness vs harmlessness objectives.

---

## Code conventions

- The core operation is `get_base_logits()`: encode prompt, single forward pass, extract logits at last position. Preserve full vocabulary logit vectors; never truncate to top-k.
- At 7B, load all three default models simultaneously into memory (~42 GB total).
- At 32B, load one model at a time. Extract logits, store, delete model, load next.
- Quantization sensitivity matters. When comparing logit distributions across layers, quantization noise can mask displacement signals. At 7B use full precision. At 32B use Q8 minimum.
- Variable names: `base_model`, `sft_model`, `dpo_model`, `instruct_model` matching the layer topology.

---

## Architecture

```
malign-logits/
├── malign_logits/
│   ├── __init__.py          # Package exports, model constants
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
malign serve

# Terminal 2: Gradio UI (restart freely, connects to server)
malign ui

# Or without server (models load lazily on cache miss)
malign ui
```

### Key classes

- **`Psyche`** — the apparatus as a whole. `from_pretrained()`, `from_cache()`, `from_server()`.
- **`ModelLayer`** / **`RemoteModelLayer`** — structural position. Caches logits and word distributions to HashStash.
- **`PromptAnalysis`** — lazy computation for a single prompt. Properties: `base_words`, `ego_words`, `superego_words`, `repression`, `sublimation`, `formation_df`, `displacement_map()`, etc.
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

## Research roadmap

### Priority 1: Battery-level aggregate metrics (for paper)

Run full prompt battery across content categories and compute:
- **Total repression mass** by content type (sexual liminal, sexual explicit, violence, neutral)
- **Mean displacement distance** in embedding space per category
- **Condensation entropy** — does displaced mass scatter or concentrate?
- **Trajectory distribution** — % of words that are decline/rise/V/peak per category
- **SFT/DPO division of labour** — for each word, what fraction of total displacement happens at Stage 1 vs Stage 2? Plot distribution by content category.

Goal: turn individual prompt observations into statistical claims. "Sexual vocabulary is repressed N× more than violent vocabulary, but violent vocabulary displaces M% further in semantic space."

### Priority 2: Automatic displacement type taxonomy

Classify displacement pairs automatically:
- **Register shift** = high similarity + same POS
- **Category shift** = high similarity + different POS
- **Genre change** = low similarity + high amplification
- **Archaic displacement** = target word has low corpus frequency

Quantify which strategies each training stage (SFT vs DPO) prefers for which content types. This becomes a key figure in the paper.

### Priority 3: Step-level checkpoint analysis

Use Allen AI's step-level checkpoints to trace displacement emerging *during* training:
- Track a word's probability at every checkpoint within the SFT or DPO stage
- Watch repression emerge progressively (e.g. `cock` at DPO step 0, 500, 1000, 2000...)
- Shows displacement is a training dynamic, not a final-state artifact

### Priority 4: Cross-family validation

Run the same battery on Zephyr (Mistral base → SFT → DPO). If the violence/sex structural difference replicates across OLMo and Zephyr, it's a property of the training *method*, not the model family.

### Future: Flexible layer count

Support 2-4 layer topologies (e.g. Llama base + instruct = 2 layers). Graceful degradation: 2 layers = sublimation only, 3 = full analysis, 4 = idealization. See memory file for design plan.
