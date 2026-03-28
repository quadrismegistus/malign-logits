# CLAUDE.md — Malign Logits

Project context for Claude Code. Read this before making changes.

---

## Project summary

This project compares full-vocabulary logit distributions across the LLM alignment pipeline (base → SFT → DPO) to trace displacement, condensation, sublimation, and repression as models undergo socialisation from raw statistical unconscious into commercial chatbot products. The core operation is extracting and comparing probability distributions for identical prompts across model layers.

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

## The three-layer topology

| Layer | Psychoanalytic function | Training stage | Loaded as |
|-------|------------------------|----------------|-----------|
| **Base** | Id / drive / primary statistical field | Pretraining | `base_model` |
| **SFT** | Ego / socialised subject | Supervised fine-tuning | `sft_model` |
| **DPO** | Superego / Name-of-the-Father | Direct preference optimisation | `dpo_model` |
| **RLVR** (optional 4th layer) | Superego (reinforced) | Reinforcement learning from verifiable rewards | `instruct_model` |

All three layers must come from the same model family with separate checkpoints at each stage.

---

## Primary model family: OLMo 3 (Allen AI)

Allen AI releases every intermediate post-training checkpoint separately. This is why we use OLMo.

### 7B checkpoints (development — load all simultaneously, ~56 GB total)

| Layer | Psychoanalytic | HuggingFace ID | ~Size (FP16) |
|-------|---------------|----------------|-------------|
| Base | Id / drive | `allenai/Olmo-3-1025-7B` | ~14 GB |
| SFT | Ego / socialised subject | `allenai/Olmo-3-7B-Instruct-SFT` | ~14 GB |
| DPO | Superego / Name-of-the-Father | `allenai/Olmo-3-7B-Instruct-DPO` | ~14 GB |
| RLVR (final) | Superego (reinforced) | `allenai/Olmo-3-7B-Instruct` | ~14 GB |

All four IDs verified from the official model card. Four models at FP16 is ~56 GB; fits on 96 GB with ~40 GB headroom.

Step-level intermediate checkpoints also available for the Think variant: `allenai/Olmo-3-7B-Think-SFT`, `allenai/Olmo-3-7B-Think-DPO`.

### 32B checkpoints (validation — load sequentially)

| Layer | Psychoanalytic | HuggingFace ID | ~Size (Q8) |
|-------|---------------|----------------|-----------|
| Base | Id / drive | `allenai/Olmo-3-1125-32B` | ~32 GB |
| SFT | Ego | `allenai/Olmo-3.1-32B-Instruct-SFT` | ~32 GB |
| DPO | Superego | `allenai/Olmo-3.1-32B-Instruct-DPO` | ~32 GB |
| RLVR (final) | Superego (reinforced) | `allenai/Olmo-3.1-32B-Instruct` | ~32 GB |

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

## Secondary model family: Amber (LLM360)

The project's original model family. Still supported in the codebase (`_is_amber_model()`, `LlamaTokenizer` fallback).

- **Strength:** radical training transparency; full training trajectory at regular step intervals
- **Weakness:** only 7B; 1.3T training tokens (vs OLMo's 5.9T); no official SFT-only checkpoint; tokenizer issues
- **Use for:** validation that displacement patterns hold across different model families

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
- At 7B, load all four models simultaneously into memory (~56 GB total).
- At 32B, load one model at a time. Extract logits, store, delete model, load next.
- Quantization sensitivity matters. When comparing logit distributions across layers, quantization noise can mask displacement signals. At 7B use full precision. At 32B use Q8 minimum.
- The historical variable names `base_model`, `instruct_model`, `safe_model` should be updated to `base_model`, `sft_model`, `dpo_model`, `instruct_model` to match the four-layer topology.
- Amber-specific code (`_is_amber_model`, `LlamaTokenizer` fallback) should be preserved for backward compatibility.