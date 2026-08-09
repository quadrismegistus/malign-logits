# What runs on this Mac, and what does not

**Definitive note, 2026-08-08.** Written because the campaign has re-derived these facts at least four times — the OLMoE `histc` crash killed a sweep at 36/104 on 8 Aug and was already recorded in two places *with a one-line fix*.

**Read this before loading models. Do not rediscover it.** The machine-readable source is `data/model_load_environments.json` (environment `local_mps`); this file is the prose that says what the observations mean. `scripts/f11_l1_logits.py --preflight` reads the JSON and names known-bad checkpoints before anything loads — copy that pattern.

---

## The principle

**"Does this model run" is not a property of the model. It is a property of (model × environment).** Two proofs the campaign paid for: `LLM360/AmberSafe` failed and then loaded *on the same box* after two packages were installed; and thirteen checkpoints that load without complaint on this Mac are unloadable on an A100 pinned at torch 2.5.1. A registry field like `needs_torch` is a *summary* of observations, never a substitute for them.

## This environment (`local_mps`)

    Mac Studio M2 Max, 96 GB unified memory, ~926 GB disk
    torch 2.11.0 · transformers 5.4.0 · tokenizers 0.22.2 · device MPS
    no CUDA, no bitsandbytes, no mamba-ssm/causal-conv1d (no Metal build exists)

MPS bf16 is supported and verified here (bf16 matmul, finite). The torch ≥ 2.6 floor for `.bin`-only checkpoints is satisfied locally, which is why models that fail on a 2.5.1 cloud box load fine here — the asymmetry runs *both* ways.

---

## Five kinds of "cannot run locally"

They are not the same thing, they have different fixes, and collapsing them is how the wrong remedy gets bought. Ordered by how expensive the fix is.

### 1. CAPACITY — memory. No local fix.

| checkpoint | need | have |
|---|---|---|
| `meta-llama/Llama-3.1-70B` | ~140 GB bf16 | 96 GB |
| `meta-llama/Llama-3.1-70B-Instruct` | ~140 GB bf16 | 96 GB |

Cloud, ≥2×80 GB. **Quantising is not a fix**: `dtype` is a keyed field in the logit store precisely because it changes the quantity, so Q8 cells are not comparable with bf16 ones.

### 2. CAPACITY — disk. Fixable by freeing space.

| checkpoint | need |
|---|---|
| `allenai/Olmo-3-1125-32B` | ~64 GB |
| `allenai/Olmo-3.1-32B-Instruct-DPO` | ~64 GB |

128 GB for the pair. Each **would fit in RAM one at a time** — this is a different limit from the 70B pair and disappears if the disk is cleared. Aborting a 32B download leaves ~21 GB of `.incomplete` blobs behind; `find ~/.cache/huggingface/hub -name '*.incomplete'` finds them, but **check nothing is mid-download first**.

### 3. MISSING KERNEL OR OP — and this is the class that keeps being misdiagnosed

**MoE — `torch.histc`. One-line fix. Works.**

`transformers/integrations/moe.py` routes experts with

```python
histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
```

a **two-way branch that assumes non-CPU means CUDA**. MPS is a third case nobody wrote, and it has no integer `histc` — it has the float one, exactly like CPU. Feed float on MPS and it passes. `PYTORCH_ENABLE_MPS_FALLBACK=1` does **not** work. It changes a dtype fed to a *counting* op, so no numeric quantity moves.

```python
_histc = torch.histc
def _histc_mps(x, *a, **k):
    if x.device.type == "mps" and not x.dtype.is_floating_point:
        return _histc(x.float(), *a, **k)
    return _histc(x, *a, **k)
torch.histc = _histc_mps
```

Note the load *succeeds* and the failure is in the forward, so a guard wrapped around `from_pretrained` will not catch it. **Guard the forward too.**

**Pure SSM — runs. Slow, not broken.** `RWKV/rwkv-4-7b-pile`, `rwkv-raven-7b`, `tiiuae/falcon-mamba-7b`, `Falcon3-Mamba-7B` all run here on the naive scan (Falcon3-Mamba timed at 21.25 s/prompt for beam search). The folklore that "SSMs don't work on MPS" is **false for pure SSMs** — no CUDA kernels are involved on that path.

**Mamba2 hybrid — cloud only.** `Zyphra/Zamba2-*` needs `causal-conv1d` + `mamba-ssm` and **there is no Metal build**. (`Zamba2-7B-Instruct` did produce cells here on 8 Aug; `Zamba2-7B`'s tokenizer fails to load at all. Treat the Instruct result as untrusted until checked against a kernel-bearing box.)

**Attention/SSM hybrid (Falcon-H1) — runs for single forward passes; OOMs on wide beams.** The recorded 37.5 GiB OOM was the *kernel-less sequential scan materialising state for 100 beams*, and it is linear in beams — it does not apply to one forward pass at ~10 tokens. The fix for beam work is `mamba-ssm` + `causal-conv1d`, **not a bigger card**. Utilisation at 100% does not name its cause: a naive state-space scan launches thousands of tiny kernels and reads on the gauge exactly like saturated arithmetic.

**`mamba-ssm` DOES NOT SERVE FalconMamba, AND THE LOG SAYS SO ONLY IF YOU LOOK.** Measured 9 Aug on a kernel-bearing A100: the four `Falcon-H1` checkpoints loaded with **no** fast-path warning, then `Falcon3-Mamba-7B-Base` emitted

    The fast path is not available because one of (selective_state_update,
    selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
    is None. Falling back to the sequential implementation of Mamba ...
    The recommended way to enable the fast path is `pip install kernels`

So the two architectures want **different packages**: the hybrid uses `mamba-ssm` + `causal-conv1d`, and transformers' **FalconMamba** integration wants the `kernels` package instead. Installing the former does nothing for the latter.

**It costs nothing here and that is measured, not assumed** — the campaign's own numbers give the kernels **19.3× on the hybrid and a null on pure SSM** (Falcon3-Mamba 0.62–0.64 with vs 0.61–0.72 without), and the delta run timed these at 0.7–1.7 min/model with no degradation. **Do not install `kernels` to silence the warning**: it would change the numerical path for the pure-SSM checkpoints away from every cell already in the corpus, buying nothing, which is the `tiktoken` mistake in new clothes — a loud, harmless message quieted at the cost of a silent, real difference.

**Falcon-H1 must compute in bfloat16.** fp16 returned **all-NaN logits on 2,583/2,583 prompts** — 5,166 empty cells that passed every structural gate. Measured: fp16 finite 1/12, bf16 finite 12/12, overflow accumulating through the SSM scan. TII's own docs say it outright. Compute dtype and storage dtype are two decisions sharing a name.

### 4. LOAD OR CODE FAILURE

| checkpoint | cause |
|---|---|
| `AI-Sweden-Models/gpt-sw3-6.7b`, `-v2-instruct` | `AutoTokenizer` OSError |
| `mosaicml/mpt-7b`, `-instruct` | `AutoTokenizer` OSError |
| `Zyphra/Zamba2-7B` | `AutoTokenizer` OSError (the `-Instruct` sibling loads — repo files, not architecture) |
| `baichuan-inc/Baichuan2-7B-*` | meta-tensor error via `trust_remote_code` RoPE |
| `internlm/internlm2-*` | custom code incompatible with this transformers version |

### 5. FIDELITY — loads, runs, and silently corrupts the prompt

**The worst class, because nothing errors.** Only a round-trip assertion `decode(encode(p)) == p` catches these; no load guard, finiteness check or cell count can see them.

| checkpoint | what it does |
|---|---|
| `deepseek-ai/deepseek-llm-7b-base`, `-chat` | **Deletes every space.** `encode("a b") == encode("ab")`. Backend `pre_tokenizer` is Metaspace where the checkpoint's own `tokenizer.json` declares ByteLevel+Split. CJK decodes to the **empty string**. No known fix in this environment. |
| `croissantllm/CroissantLLMBase`, `-Chat` | **Deletes CJK characters**, including both halves of the 既…又 both-and construction. English exact. |
| `openGPT-X/Teuken-7B-*` | Normalises the **fullwidth comma** `，` → `,`. Milder, and a different kind of loss from Croissant's. |
| `Aleph-Alpha/Pharia-1-LLM-7B-*` | **Looks** broken, is not: SentencePiece renders a leading `▁` as a space at *decode*. `encode(p) != encode(" "+p)`, so the encode is faithful. (Its CJK handling *is* lossy.) |
| `THUDM/glm-4-9b-*` | **Looks** broken, is not: prepends `[gMASK]<sop>` and `bos_token_id` is `None`, so a BOS-only strip cannot see them. `skip_special_tokens` recovers the prompt exactly. |

The distinguishing test for every row: **does the tokenizer prove it can represent the distinction?** If `encode(p) != encode(mangled)`, the difference is cosmetic and never reaches the model. If they are equal, information is genuinely lost.

---

## The roster as it stands (104 checkpoints, F11 L1)

| outcome | n | checkpoints |
|---|---|---|
| **load_failed** | 9 | `gpt-sw3-6.7b`, `gpt-sw3-6.7b-v2-instruct`, `Zamba2-7B`, `Olmo-3-1125-32B`, `Olmo-3.1-32B-Instruct-DPO`, `Llama-3.1-70B`, `Llama-3.1-70B-Instruct`, `mpt-7b`, `mpt-7b-instruct` |
| **run_failed** | 5 | `OLMoE-1B-7B-0125` (fixed, see §3), `CroissantLLMBase`, `deepseek-llm-7b-base`, `deepseek-llm-7b-chat`, `Teuken-7B-base-v0.6` |
| **loads** | 6 | `Pharia-1-LLM-7B-control-hf`, `rwkv-4-7b-pile`, `rwkv-raven-7b`, `Zamba2-7B-Instruct`, `OLMo-2-0425-1B-DPO`, `OLMoE-1B-7B-0125-DPO` |
| untested here | 84 | dense transformers that raised nothing |

Architecture classes: dense 86, custom-code 4, attn/SSM hybrid 4, pure SSM 4, linear-attn RNN 2, Mamba2 hybrid 2, MoE 2.

**`OLMoE-1B-7B-0125` failed and `OLMoE-1B-7B-0125-DPO` did not, on the same architecture and the same prompts.** MoE-on-MPS is not decidable from the architecture name — which is the whole reason the record keys observations by checkpoint.

---

## Two rules that generalise past this file

**A prior tested on one member of a class is not a fact about the class.** `cloud_profiles.json` recorded that the mamba kernels gave no speedup — measured correctly on Falcon3-Mamba, a *pure* SSM. Quoted twice to predict no speedup for Falcon-**H1**, an attention/SSM *hybrid*, where they are worth **19.3×** (21.4 h → 1.1 h, $22 → $1.16).

**Absence of an observation is not success.** `data/model_load_environments.json` carries a `predicted_untested` block for exactly this reason. A checkpoint with no row has not passed; it has not been asked.
