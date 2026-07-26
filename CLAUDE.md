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
- Package manager: **uv**. Install dependencies with `uv pip install <package>`, not `pip` or `pip3`.

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

**47 families / 107 unique checkpoints** are registered in `MODEL_FAMILIES` (`__init__.py`) — the single source of truth. Do NOT rely on any table in a doc; run `malign info` for the live list, or see `docs/model_candidates.md` for provenance notes. Coverage spans: the OLMo 3 pipeline (default `olmo`, 4 layers), OLMo variants (think, tiny, 32B), Tulu + 4 SFT-ablation families, Llama (+70B, Dolphin, R1-distill), Qwen (+Qwen3, think), Amber, Zephyr, Pythia, SmolLM2/3, Falcon (+Mamba/H1 SSM variants), RWKV, OLMoE (MoE), archangel method variants (DPO/PPO/KTO/SLiC), Chinese families (MAP-Neo, CT-LLM, InternLM2, Yi, Baichuan, GLM4, MiniCPM), and scale ladders (Falcon3 1B→10B).

Common keys: `olmo` (default), `olmo-tiny` (dev/low-RAM), `smol` / `qwen-tiny` (2-layer test fixtures), `amber`, `llama`, `qwen`, `zephyr`, `pythia`, `tulu`.

CLI: `malign serve --family amber`, `malign battery --family qwen`, `malign info`.

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
psyche = Psyche.from_family("olmo")

# 2-layer: base + instruct (instruct maps to superego)
psyche = Psyche.from_family("llama")
```

Layer topology determines available analyses:
- **2 layers** (base + superego): Repression only. No sublimation, id scores, displacement, or neurotic generation.
- **3 layers** (base + ego + superego): Full analysis.
- **4 layers** (+ reinforced_superego): Full + idealization.

`Psyche.ego` is `None` for 2-layer families. Properties that require ego raise `ValueError` with a clear message. Properties that can adapt (repression, formation_df, metrics) work with any layer count.

CLI: `malign serve --family llama`, `malign info`.

---

## Architecture

```
malign-logits/
├── malign_logits/
│   ├── __init__.py          # Package exports, ModelFamily registry (MODEL_FAMILIES)
│   ├── psyche.py            # Psyche, ModelLayer, RemoteModelLayer, PromptAnalysis
│   ├── models.py            # Model loading (load_model)
│   ├── core.py              # get_base_logits, score_words_from_logits, beam_word_probs, hybrid_word_probs
│   ├── analysis.py          # Repression, id, displacement engine (v4)
│   ├── metrics.py           # Pure-numpy metrics (JS, mode decomposition, resistance)
│   ├── probe.py             # Probe: token-tree exploration + teacher-forced annotation (F26)
│   ├── deep_probe.py        # DeepDive: full tensor collection to parquet
│   ├── beam.py              # Beam storylines + teacher-forced beam annotation (F27/F28)
│   ├── registry.py          # Registry: model-centric view, NICKNAMES, typed relations
│   ├── profile.py           # CircuitProfile per-family summaries
│   ├── trajectory.py        # Hidden-state geometry, steering vectors, fold analysis (F12)
│   ├── circuit.py           # Circuit class (F25 temporal signatures, reasoning branches, modes)
│   ├── cache.py             # CacheManager + open_stash — ALL stash access goes through here
│   ├── experiments.py       # Prompt battery definitions (DEFAULT/TIER1/INSTITUTIONAL_PROMPTS)
│   ├── generation.py        # Text generation (standard + neurotic)
│   ├── embedding.py         # Generation + embedding pipeline, surprisal (run_generate_battery)
│   ├── taxonomy.py          # Displacement taxonomy + syntagmatic_js
│   ├── graphdb.py           # ArangoDB graph of trees/annotations (probe ingest/census)
│   ├── vecdb.py             # DuckDB vector store (hidden-state search)
│   ├── ablation.py          # Tulu SFT data-mixture ablation
│   ├── battery.py           # Multi-family prompt battery driver
│   ├── logit_lens.py        # Logit-lens CSV+figure driver
│   ├── step_analysis.py     # Step-level checkpoint analysis
│   ├── api_generate.py      # Frontier-API generation (DeepSeek etc.)
│   ├── cloud.py             # vast.ai orchestration (malign cloud ...)
│   ├── produce.py           # produce-all driver
│   ├── viz.py               # Plotly visualizations
│   ├── viz_sankey.py        # Beam/displacement Sankey figures
│   ├── displacement_graph.py, training_graph.py, vocab.py  # graph/vocab utilities
│   ├── cli.py               # CLI router (delegates to analysis modules)
│   ├── app.py               # LEGACY Gradio UI (superseded by Svelte data explorer)
│   ├── server.py            # Model/data server; serves Svelte UI from ui_dist/
│   └── ui_dist/             # Built Svelte data explorer (source in ui/)
├── ui/                      # SvelteKit data-explorer source
├── notebooks/               # Worked examples
├── findings/                # F01–F35 individual finding files (README is built from these)
├── context.md               # HISTORICAL (March 2026) theory notes — gitignored, architecture obsolete
├── pyproject.toml           # Package config (loads deps from requirements.txt)
└── requirements.txt         # Dependencies
```

### Dev workflow

```bash
# Terminal 1: model/data server (load once, stays running; binds 127.0.0.1)
malign serve                     # default family (olmo)
malign serve --family llama      # or a specific family
malign serve --data-only         # cached data only, no model loading

# Terminal 2: open the Svelte data explorer (served by malign serve)
malign ui                        # opens browser; --dev runs Vite from ui/
```

### CLI subcommands

`malign <cmd>` — run `malign <cmd> --help` for flags. (Source of truth: `cli.py`.)

| Command | Purpose | Finding |
|---|---|---|
| `info` | Print registered families + config | — |
| `serve` / `ui` | Model/data server + Svelte explorer | — |
| `precompute` | Fast logit caching (1 fwd pass/layer×prompt) | F01 |
| `battery` | Cross-family logit battery | F02, F06, F09 |
| `taxonomy` | Displacement taxonomy; `--analyze`, `--baseline` | F08, F13, F14 |
| `logit-lens` | Hidden states → unembedding per layer | F05 |
| `step-analysis` | Repression across training-step checkpoints | F04 |
| `trajectory` | Hidden-state geometry, steering vectors, fold | F12 |
| `generate-battery` | Generate + embed + cluster metrics | F03 |
| `topic-drift` / `surprisal` / `embed` | Passage metrics on cached generations | F15, F18 |
| `bos-generate` | Unconditional (BOS) generation | F19 |
| `ingest` | Load human corpora into the generation cache | F16 |
| `api-generate` | Frontier-API generation (DeepSeek etc.) | F21 |
| `vllm-generate` | Batched vLLM generation (`--tp` for tensor-parallel) | — |
| `probe` | Token trees: `batch`, `status`, `merge`, `ingest`, `census`, `download`, `cloud-status` | F26, F27, F28 |
| `deep-probe` | Full tensor collection to parquet | — |
| `circuit` | Reasoning-model circuit / temporal signatures | F22, F23, F25 |
| `ablation` | Tulu SFT data-mixture ablation | F10 |
| `cloud` | vast.ai orchestration (`launch`, `run`, ...) | — |
| `produce-all` | Run the full production pipeline | — |
| `download-models` | Pre-fetch checkpoints | — |

### Key classes

- **`ModelFamily`** — dataclass mapping a model family to its checkpoints. `MODEL_FAMILIES` dict in `__init__.py`.
- **`Psyche`** — the apparatus as a whole. `from_family()`, `from_pretrained()`, `from_cache()`, `from_server()`. `ego` is `None` for 2-layer families.
- **`ModelLayer`** / **`RemoteModelLayer`** — structural position. Caches logits and word distributions to HashStash.
- **`PromptAnalysis`** — lazy computation for a single prompt. Properties adapt to available layers. 3-layer features raise `ValueError` on 2-layer Psyche.
- **`PrimaryProcess`**, **`Ego`**, **`Superego`**, **`ReinforcedSuperego`** — named layer subclasses.
- **`Circuit`** — extended topology for reasoning models. Nodes are distributions, edges are alignment operations. Supports reasoning branches, mode switching (raw/chat/think), cross-circuit comparison. See below.

### Circuit class (`circuit.py`)

Maps to the book's TOC diagram. Unlike Psyche's linear base→SFT→DPO, Circuit supports:
- **Reasoning branches**: `load_reasoning()` adds distilled reasoning models (R1-Distill-Llama, R1-Distill-Qwen)
- **Mode switching**: `Mode.RAW`, `Mode.CHAT`, `Mode.THINK` — same weights, different templates
- **Cross-circuit comparison**: `convergence()` measures whether different families converge after alignment

```python
from malign_logits import Circuit, Mode
circuit = Circuit.from_config(main='olmo', reasoning='r1-llama')
circuit.compare('base', 'dpo', prompt)     # JS, displacement at an edge
circuit.branch_compare(prompt)             # base vs aligned vs reasoning
```

**F25 classifier**: `Circuit.classify_trajectory()` classifies generation-level temporal signatures:
- **foreclosure**: step 0 argmax is blank (exam template)
- **repression**: argmax changed from base (displaced)
- **return_of_repressed**: blank template but transgressive tokens in top-5
- **reaction_formation**: entropy slope < -0.15 (narrowing trajectory)
- **transparent**: argmax preserved from base (APO signature)

`classify_mega_gen()` and `signature_summary()` for batch analysis. `mega_generate()` extracts position-level entropy/top-k during autoregressive generation, cached to lmdb.

### Performance notes

- `discover_top_words` runs ~200 forward passes per layer per prompt (~30-60s each on MPS).
- `score_words_from_logits` scores vocabulary from cached logits in microseconds (replaced per-word forward passes).
- `logits()` cached to HashStash — 1 forward pass per layer per prompt, ever.
- HashStash persistence means second runs (even after restart) skip all expensive compute.

### Cache access (CacheManager)

**Always use `CacheManager` from `malign_logits/cache.py`, never raw HashStash.** CacheManager uses the absolute `PATH_DATA_RAW` path, so stashes resolve to the project's `data/raw/cache/` directory. Raw HashStash with a relative path resolves to `~/.cache/hashstash/...` — wrong location.

```python
from malign_logits.cache import get_cache
cm = get_cache()
```

**BOS prompts vary by tokenizer.** When accessing unconditional (BOS) generations, the prompt key is the model's BOS token, not an empty string:

| BOS token | Families |
|-----------|----------|
| `'<\|endoftext\|>'` | OLMo, OLMo-tiny, Pythia, Qwen, SmolLM |
| `'<s>'` | Llama, Amber, Zephyr (Mistral-based) |
| `''` | Human text (dreams, fiction, etc.) — no model prompt |

```python
cm.count_generations('LLM360/Amber', '<s>')          # → 100
cm.count_generations('allenai/Olmo-3-1025-7B', '<|endoftext|>')  # → 100
cm.count_generations('LLM360/Amber', '')              # → 0 (wrong!)
```

**Never open the same lmdb environment twice in one process.** If a notebook already has a HashStash open on `data/raw/cache/generations`, creating a CacheManager (which opens its own HashStash on the same path) will fail or corrupt reads. Use one or the other, not both.

**Generation keys:** `{'model': str, 'prompt': str, 'temp': float, 'idx': int}`. The `model` value is the full HuggingFace ID (e.g. `'allenai/Olmo-3-1025-7B'`), not the family key.

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

**Liminal content displaces more than explicit — but this is ~91% an entropy effect, and "boundary" overstates it.** CORRECTED 2026-07-26. Recomputed across 9 families (amber, llama, olmo, olmo-tiny, qwen, smol, tulu, tulu-no-safety, zephyr), paired, family as unit:

- **liminal − explicit JS = +0.0271**, CI [+0.0102, +0.0440], t=+3.15, **9/9 families positive**. Direction is solid.
- **liminal − neutral JS = +0.0097**, CI [−0.0153, +0.0347], 6/9. **Not significant.** Liminal is indistinguishable from neutral, so there is no boundary *peak* — explicit sits low and everything else is flat.
- Within-family JS-vs-entropy slope is +0.0187/nat; the liminal−explicit entropy gap of 1.315 nats predicts +0.0246 against +0.0271 observed. **Residual +0.0026.** The effect is almost entirely explained by liminal sites being higher-entropy than explicit ones.

The previous version of this claim ("sexual liminal 0.13 > sexual explicit 0.10; violence liminal 0.15 > violence explicit 0.09; the superego is most active at the boundary") cited numbers matching no surviving file — `data/battery_results.csv` was later overwritten by a single-family (zephyr) run, so the booked figures had no live source. Per-family batteries in `data/battery_*.csv` are the real basis.

Independently, lacan's freed-mass metric on the same prompts finds **no family-level liminal/explicit difference** (20/37) and explicit *below* neutral (26/37, p=0.020). Both metrics agree liminal ≈ neutral and that the raw effect is entropy-driven; they differ on the residual sign. The boundary claim is **METRIC-QUALIFIED** pending a registered metric-comparison study.

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

## Confirmed findings (baseline validation, 4 families)

**Prompted by a colleague's observation** that the displacement metrics might reflect general SFT drift rather than alignment-specific intervention: if SFT reshapes all distributions, how do we know the changes on transgressive prompts are safety-related rather than a side-effect of instruction tuning?

**Base perplexity does not predict displacement.** Pearson correlation between log(base perplexity) and JS divergence (base→superego) is near zero for all families: Amber r=-0.04, Llama r=-0.25, OLMo r=-0.19, Qwen r=+0.04. The amount of distributional change is unrelated to how uncertain the base model was about the prompt. The confound does not hold.

**Scalar metrics do not separate transgressive from neutral prompts.** JS divergence, entropy drop, top-50 overlap, and Spearman rank correlation all fail to distinguish categories (Mann-Whitney p > 0.05 for all families). OLMo's neutrals actually show *higher* mean JS (0.224) than its transgressive prompts (0.167), because SFT restructures heavily for instruction-following even on harmless content.

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

Neutral vs transgressive separation: Qwen p=0.0001, OLMo p=0.01, Llama p=0.008, Amber p=0.06 (Mann-Whitney, one-sided). Amber is marginal because its tokenizer only matches 14 of 62 transgressive tokens.

**Interpretation:** Alignment displaces similar *total* probability mass on neutral and transgressive prompts (same JS), but on transgressive prompts the displaced mass comes specifically from transgressive tokens. On neutral prompts it comes from generic vocabulary reshaping. The superego operates surgically on specific tokens rather than reshaping the whole distribution differently. This is why scalar distributional metrics (JS, entropy, rank correlation) cannot detect alignment intervention — the signal is in *which* tokens move, not *how much* moves.

Results in `data/transgressive_mass.csv`. Figure in `figures/perplexity_vs_displacement.png`. New metrics in `battery_results.csv`: `perplexity_base/ego/superego/instruct`, `rank_corr_base_superego/base_ego/ego_superego/superego_instruct`.

---

## Confirmed findings (training data attribution, OLMo 3)

**The OLMo 3 report (arXiv:2512.13961) documents exact data mixtures for each training stage.** Safety data is ~5% of SFT (110k of 2.15M prompts) and ~10% of DPO (27k of 260k), sourced from CoCoNot, WildGuardMix, and WildJailbreak.

**SFT/DPO division of labour implicates the training objective, not the data.** If displacement were data-driven, both stages would repress proportionally. Instead SFT handles sex and DPO handles violence — the *how* of learning matters. DPO uses delta learning (Qwen 32B chosen vs Qwen 0.6B rejected), so violence repression emerges from a capability gap, not explicit safety annotation.

**Base model transgressive mass reflects internet frequency.** Pretraining is 76% Common Crawl (4.5T tokens). High probability on sexual/violent tokens is genuine corpus-level drive energy, not curation artefact.

---

## Research roadmap

### Done: F19 unconditional generation & BLT byte-level analysis

`malign bos-generate` produces 100 completions per layer from BOS token across all 10 families. BLT 1B (`itazap/blt-1b-hf`) scores all text at byte level for true character-level bits/char. Aligned BOS prose has lower median information density than all human text types. `data/jakobson.parquet` = 143k passages (141k AI + 2k human) with drift, surprisal, BLT bits/char, genre, quadrants. Notebook: `notebooks/F19_bos_entropy.ipynb`. Finding: `findings/F19_bos_entropy.md`.

### Done: Findings refactor & F01 notebook

`findings/F01-F19_*.md` — individual finding files. `scripts/build_readme.py build` regenerates README with auto-TOC. `notebooks/F01_logit_analysis.ipynb` — JS divergence, entropy, SFT/DPO division, word trajectories across all 10 families from cached logits.

### Done: Unified CLI for cached data

- `malign surprisal [--ref MODEL] [--self]` — ref + self surprisal for all cached generations + human corpora
- `malign embed [--embedder MODEL]` — sentence embeddings (bge-m3) for all cached data
- `malign ingest {dreams,waking,fiction,abstracts,all}` — ingest human corpora from original CSVs into generation cache
- `malign precompute --logits-only` — fast logit caching (1 fwd pass per layer×prompt), all families by default
- `malign bos-generate [--prompt TEXT] [--family FAM] --n N` — unconditional generation

### Done: Full generation run (n=100, cloud) — COMPLETE

All 10 families × 47 prompts × 100 generations. 141k+ passages cached.

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

### Done: Displacement taxonomy

`malign taxonomy` classifies displacement pairs into register shift, category shift, genre change, and archaic displacement using contextual spaCy POS tagging (word tagged in prompt context) and wordfreq. Cross-family: Llama is more register-shift dominant (66%) than OLMo (49%), consistent with late-layer override vs distributed repression. Profanity is 49-62% genre change in both — model-independent. Explicit content is overwhelmingly register shift; genre change only appears on liminal/profane content. Results in `data/displacement_taxonomy.csv`.

### Done: Syntagmatic baseline check (aligned vs base model)

Computing `syntagmatic_js` under both the base model and the aligned (DPO) model on the same 23k displacement pairs reveals that alignment produces measurable syntagmatic damage — the aligned model's continuations are *more* disrupted by its own substitutions than the base model's are. Delta is positive for every content category: sexual_explicit +0.106, violence_explicit +0.074, neutral +0.044, profanity +0.032.

Three structurally distinct cases emerge:
- **Alignment-produced damage** (sexual_explicit): base model substitutes fluently (synt_js 0.37); alignment specifically breaks the chain (+0.106). The strongest case for Jakobsonian similarity disorder as an alignment-produced impairment.
- **Alignment-inherited damage** (profanity): base model already struggles (synt_js 0.56); alignment adds little (+0.032). The chain was already broken at the corpus level — profanity has no clean synonyms in any model.
- **Alignment-unnecessary** (violence_explicit): both models substitute fluently (synt_js 0.16/0.24). Paradigmatic resources exist and alignment barely disrupts them.

Neutral delta (+0.044) rules out the noise interpretation — alignment produces background syntagmatic damage even on safe content. Results in `data/taxonomy_olmo.csv` (`syntagmatic_js_aligned` column). CLI: `malign taxonomy --baseline --family olmo`.

### Done: Generation passage metrics (10 families, 76k passages, Pythia 1B + bge-m3)

76,214 passages in definitive `data/corpus_metrics.parquet`. Primary metrics: Pythia 1B-deduped surprisal (independent of all families) + bge-m3 drift (BAAI, 1024d). GPT-2 and MiniLM as validation. Alignment universally smooths (all families p<0.001). Category has no effect on within-passage surprisal (p=0.99). Jakobsonian quadrants: alignment drains Q2 (breakdown) → Q1 (metonymic) / Q4 (unmarked). Summary: `python scripts/corpus_metrics.py --summary` saves to `data/corpus_metrics.md`.

### Done: Cross-generation MMD (10 families, 76k passages)

`scripts/cross_generation_mmd.py` — MMD² between BASE and ALIGNED completion clouds. Category effect p=0.0004 (sexual_explicit highest). The how/what dissociation: uniform within-passage smoothing but differential content steering.

### Done: Corpus comparison (dreams, waking, fiction, abstracts)

Under Pythia 1B: Fiction (+0.40σ) > Dreams (+0.14σ) > Abstracts (+0.10σ) > AI (-0.10σ) > Waking (-0.49σ). Dream-specific +0.63σ above register baseline (p<10⁻³²).

### Done: Shannon entropy & self-surprisal (F18)

Logit-level entropy: alignment reduces from ~4 to ~3 nats. Self-surprisal: base models at Shannon's English rate (~1.0 bits/char), alignment compresses 9/10 families below. Self-vs-reference gap widens = "private language." Amber anomaly: safety model more surprised by own output. Category: liminal loses most entropy (r=-0.84, p=0.004). Data: `data/shannon_entropy.csv`, `data/self_surprisal.csv`. Notebook: `notebooks/10_shannon.ipynb`.

### Done: Fold dimensionality (F12 revised; numbers corrected 2026-07-05)

Alignment is fold not wall. v2.6 held-out closure (corrected — the original evaluation leaked training prompts): 61% (Pythia) to 4% (OLMo, Llama); train closure 37–92% shows the gap is generalization, not capacity. SVD K_50: 2 (Pythia) to 13 (OLMo). Foldability tracks alignment sophistication (ρ ≈ −0.75 vs K_50). Data: `data/fold_rank_summary.csv`, `data/intervention_*.csv`, `data/trajectory_geometry_*.csv`. Re-derivation: `scripts/rederive_f12_heldout.py`.

### Done: vLLM generation pipeline

`malign vllm-generate --prompts all --n 100` — batched generation 50-100x faster than HF. `scripts/vllm_generate.py`. Cloud: `malign cloud run malign vllm-generate ...`. Docker: `vllm/vllm-openai:latest`.

### Done: Cache redesign

`malign_logits/cache.py` — CacheManager with lmdb engine, dict keys, typed methods. 6 stashes in `data/raw/cache/`. Old pairtree stashes migrated via `scripts/migrate_stash.py`. 29 tests, GitHub Actions CI.

### Done: Cross-family Jakobsonian analysis

`malign taxonomy --analyze` reads all taxonomy CSVs and computes paradigmatic-syntagmatic correlations per family. Trade-off holds for all 6 families tested (126k pairs): r = −0.34 (olmo-tiny) to −0.53 (llama). The Jakobsonian dissociation is structural, not safety-training-specific. (See F13 for current numbers.)

### Done: Flexible layer count

Model families support 2-4 layer topologies. `Psyche.from_family()` loads the right checkpoints. 2 layers = repression only, 3 = full analysis, 4 = idealization.
