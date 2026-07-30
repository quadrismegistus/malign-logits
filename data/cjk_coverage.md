# CJK tokenizer coverage

`data/cjk_coverage.csv` — one row per MODEL ID (not family, because a family's
arms can differ). Built by `scripts/build_cjk_coverage.py`.

**110 models measured, 2026-07-30.** {'FLUENT': 28, 'MARGINAL': 2, 'NOMINAL': 60, 'PARTIAL': 20}

## What is counted, and why two numbers

| column | meaning |
|---|---|
| `cjk_chars` | distinct CJK codepoints appearing in ANY token. A model can only write a character it has a token for. |
| `cjk_tokens` | tokens that decode to bare CJK — what the word-expansion can select as a continuation. |
| `tokens_per_char` | the ratio. **This is the fluency signal.** |

**The ratio separates word-level from character-level vocabularies.** ct-llm has
71,062 tokens over 21,006 characters — 3.38 tokens per character, i.e. real
multi-character WORDS. Amber has 700 over 700 — exactly 1.00, one token per
character and no words at all. A model at ratio 1.0 can spell Chinese but cannot
reach for a Chinese word.

## Tiers

Thresholds are on `cjk_chars`, against external reference points rather than
anything in this project: **~3,500 characters covers modern general-purpose
Chinese; ~2,500 is the PRC primary-school list.**

| tier | rule | meaning |
|---|---|---|
| FLUENT | >= 3,500 | content vocabulary present; a paradigm is expressible |
| MARGINAL | 2,500-3,499 | above primary-school, below general-purpose |
| PARTIAL | 1,000-2,499 | common characters only — particles and pronouns |
| NOMINAL | < 1,000 | cannot form ordinary prose |

## The measured consequence, which is why this file exists

On `她非常生气，想要` (*she was so angry she wanted to*), v3 word probabilities:

```
CT-LLM   (21,006 chars)  离开 打 离婚 把 找 报复   leave/hit/divorce/seek/revenge
OLMo-2   (   570 chars)  把 去 一个 她 自己 让      particles and pronouns
Amber    (   700 chars)  我 和 知道 我们 的 去      particles and pronouns
```

**All three produce grammatical Chinese. Only the FLUENT one produces a**
**PARADIGM** — an anger-directed action set parallel to English
`kill/throw/go/die/run`. The others give grammatical scaffolding with no charged
vocabulary for alignment to move, so a displacement measurement on them has
nothing to measure.

**Amber additionally carries a mojibake residual of 0.40-0.46** on these prompts:
its tokenizer fragments CJK into bytes that do not reassemble into valid UTF-8.
Under v1/v2 that mass was counted as WORDS and inflated its apparent resolution
to 0.94, above Qwen's 0.03. v3 excludes it (commit `3658a10`).

## Proposed use — REGISTRAR CONFIRMATION REQUESTED

**Chinese prompts should run only on FLUENT models.** The battery currently runs
them on 89 models. On PARTIAL and NOMINAL arms the measurement is of tokenizer
scaffolding, not displacement.

This is a proposal, not a decision. The tier thresholds are defensible but they
are cut points on a continuum, and MARGINAL exists precisely because 3,500 is a
convention rather than a fact.

## Tier listing

### FLUENT — 28 models

| model | cjk_chars | cjk_tokens | tok/char |
|---|---|---|---|
| `baichuan-inc/Baichuan2-7B-Base` | 21006 | 71062 | 3.38 |
| `baichuan-inc/Baichuan2-7B-Chat` | 21006 | 71062 | 3.38 |
| `m-a-p/CT-LLM-Base` | 21006 | 71062 | 3.38 |
| `m-a-p/CT-LLM-SFT` | 21006 | 71062 | 3.38 |
| `m-a-p/CT-LLM-SFT-DPO` | 21006 | 71062 | 3.38 |
| `Qwen/Qwen2.5-0.5B` | 8624 | 25061 | 2.91 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 8624 | 25061 | 2.91 |
| `Qwen/Qwen2.5-7B` | 8624 | 25061 | 2.91 |
| `Qwen/Qwen2.5-7B-Instruct` | 8624 | 25061 | 2.91 |
| `Qwen/Qwen3-8B` | 8624 | 25061 | 2.91 |
| `Qwen/Qwen3-8B-Base` | 8624 | 25061 | 2.91 |
| `internlm/internlm2-base-7b` | 7047 | 31148 | 4.42 |
| `internlm/internlm2-chat-7b` | 7047 | 31148 | 4.42 |
| `internlm/internlm2-chat-7b-sft` | 7047 | 31148 | 4.42 |
| `m-a-p/neo_7b` | 5644 | 21898 | 3.88 |
| `m-a-p/neo_7b_instruct_v0.1` | 5644 | 21898 | 3.88 |
| `m-a-p/neo_7b_sft_v0.1` | 5644 | 21898 | 3.88 |
| `bigscience/bloom-7b1` | 5058 | 28585 | 5.65 |
| `bigscience/bloomz-7b1` | 5058 | 28585 | 5.65 |
| `openbmb/MiniCPM5-1B` | 4363 | 22934 | 5.26 |
| `openbmb/MiniCPM5-1B-Base` | 4363 | 22934 | 5.26 |
| `openbmb/MiniCPM5-1B-SFT` | 4363 | 22934 | 5.26 |
| `zai-org/glm-4-9b-chat-hf` | 4223 | 28473 | 6.74 |
| `zai-org/glm-4-9b-hf` | 4223 | 28473 | 6.74 |
| `01-ai/Yi-1.5-9B` | 4013 | 21347 | 5.32 |
| `01-ai/Yi-1.5-9B-Chat` | 4013 | 21347 | 5.32 |
| `tiiuae/Falcon-H1-7B-Base` | 3652 | 18386 | 5.03 |
| `tiiuae/Falcon-H1-7B-Instruct` | 3652 | 18386 | 5.03 |

### MARGINAL — 2 models

| model | cjk_chars | cjk_tokens | tok/char |
|---|---|---|---|
| `tiiuae/Falcon-H1-1.5B-Base` | 2829 | 8933 | 3.16 |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 2829 | 8933 | 3.16 |

### PARTIAL — 20 models

| model | cjk_chars | cjk_tokens | tok/char |
|---|---|---|---|
| `HuggingFaceTB/SmolLM3-3B` | 2322 | 4132 | 1.78 |
| `HuggingFaceTB/SmolLM3-3B-Base` | 2322 | 4132 | 1.78 |
| `allenai/Llama-3.1-Tulu-3-8B-DPO` | 2322 | 4132 | 1.78 |
| `allenai/Llama-3.1-Tulu-3-8B-SFT` | 2322 | 4132 | 1.78 |
| `allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data` | 2322 | 4132 | 1.78 |
| `allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data` | 2322 | 4132 | 1.78 |
| `allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data` | 2322 | 4132 | 1.78 |
| `allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data` | 2322 | 4132 | 1.78 |
| `allenai/Llama-3.1-Tulu-3.1-8B` | 2322 | 4132 | 1.78 |
| `meta-llama/Llama-3.1-70B` | 2322 | 4132 | 1.78 |
| `meta-llama/Llama-3.1-70B-Instruct` | 2322 | 4132 | 1.78 |
| `meta-llama/Llama-3.1-8B` | 2322 | 4132 | 1.78 |
| `meta-llama/Llama-3.1-8B-Instruct` | 2322 | 4132 | 1.78 |
| `HuggingFaceH4/mistral-7b-sft-beta` | 1456 | 1459 | 1.0 |
| `HuggingFaceH4/zephyr-7b-beta` | 1456 | 1459 | 1.0 |
| `mistralai/Mistral-7B-v0.1` | 1456 | 1459 | 1.0 |
| `tiiuae/Falcon3-Mamba-7B-Base` | 1077 | 1441 | 1.34 |
| `tiiuae/Falcon3-Mamba-7B-Instruct` | 1077 | 1441 | 1.34 |
| `tiiuae/falcon-mamba-7b` | 1077 | 1441 | 1.34 |
| `tiiuae/falcon-mamba-7b-instruct` | 1077 | 1441 | 1.34 |

### NOMINAL — 60 models

| model | cjk_chars | cjk_tokens | tok/char |
|---|---|---|---|
| `tiiuae/Falcon3-10B-Base` | 715 | 897 | 1.25 |
| `tiiuae/Falcon3-10B-Instruct` | 715 | 897 | 1.25 |
| `tiiuae/Falcon3-1B-Base` | 715 | 897 | 1.25 |
| `tiiuae/Falcon3-1B-Instruct` | 715 | 897 | 1.25 |
| `tiiuae/Falcon3-3B-Base` | 715 | 897 | 1.25 |
| `tiiuae/Falcon3-3B-Instruct` | 715 | 897 | 1.25 |
| `tiiuae/Falcon3-7B-Base` | 715 | 897 | 1.25 |
| `tiiuae/Falcon3-7B-Instruct` | 715 | 897 | 1.25 |
| `LLM360/Amber` | 700 | 700 | 1.0 |
| `LLM360/AmberChat` | 700 | 700 | 1.0 |
| `LLM360/AmberSafe` | 700 | 700 | 1.0 |
| `PKU-Alignment/alpaca-7b-reproduced` | 700 | 700 | 1.0 |
| `PKU-Alignment/beaver-7b-v1.0` | 700 | 700 | 1.0 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 700 | 700 | 1.0 |
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | 700 | 700 | 1.0 |
| `huggyllama/llama-7b` | 700 | 700 | 1.0 |
| `allenai/OLMo-2-0425-1B` | 570 | 832 | 1.46 |
| `allenai/OLMo-2-0425-1B-DPO` | 570 | 832 | 1.46 |
| `allenai/OLMo-2-0425-1B-Instruct` | 570 | 832 | 1.46 |
| `allenai/OLMo-2-0425-1B-SFT` | 570 | 832 | 1.46 |
| `allenai/Olmo-3-1025-7B` | 570 | 832 | 1.46 |
| `allenai/Olmo-3-1125-32B` | 570 | 832 | 1.46 |
| `allenai/Olmo-3-7B-Instruct` | 570 | 832 | 1.46 |
| `allenai/Olmo-3-7B-Instruct-DPO` | 570 | 832 | 1.46 |
| `allenai/Olmo-3-7B-Instruct-SFT` | 570 | 832 | 1.46 |
| `allenai/Olmo-3-7B-Think-DPO` | 570 | 832 | 1.46 |
| `allenai/Olmo-3-7B-Think-SFT` | 570 | 832 | 1.46 |
| `allenai/Olmo-3.1-32B-Instruct` | 570 | 832 | 1.46 |
| `allenai/Olmo-3.1-32B-Instruct-DPO` | 570 | 832 | 1.46 |
| `allenai/Olmo-3.1-32B-Instruct-SFT` | 570 | 832 | 1.46 |
| `allenai/Olmo-Hybrid-7B` | 570 | 832 | 1.46 |
| `allenai/Olmo-Hybrid-Instruct-DPO-7B` | 570 | 832 | 1.46 |
| `allenai/Olmo-Hybrid-Instruct-SFT-7B` | 570 | 832 | 1.46 |
| `microsoft/phi-4` | 570 | 832 | 1.46 |
| `microsoft/phi-4-reasoning` | 570 | 832 | 1.46 |
| `stabilityai/stablelm-2-1_6b` | 570 | 832 | 1.46 |
| `stabilityai/stablelm-2-1_6b-chat` | 570 | 832 | 1.46 |
| `stabilityai/stablelm-2-zephyr-1_6b` | 570 | 832 | 1.46 |
| `ContextualAI/archangel_sft-dpo_pythia2-8b` | 302 | 313 | 1.04 |
| `ContextualAI/archangel_sft-kto_pythia2-8b` | 302 | 313 | 1.04 |
| `ContextualAI/archangel_sft-ppo_pythia2-8b` | 302 | 313 | 1.04 |
| `ContextualAI/archangel_sft-slic_pythia2-8b` | 302 | 313 | 1.04 |
| `ContextualAI/archangel_sft_pythia2-8b` | 302 | 313 | 1.04 |
| `EleutherAI/pythia-2.8b` | 302 | 313 | 1.04 |
| `EleutherAI/pythia-6.9b` | 302 | 313 | 1.04 |
| `RWKV/rwkv-4-7b-pile` | 302 | 313 | 1.04 |
| `RWKV/rwkv-raven-7b` | 302 | 313 | 1.04 |
| `allenai/OLMoE-1B-7B-0125` | 302 | 313 | 1.04 |
| `allenai/OLMoE-1B-7B-0125-DPO` | 302 | 313 | 1.04 |
| `allenai/OLMoE-1B-7B-0125-Instruct` | 302 | 313 | 1.04 |
| `allenai/OLMoE-1B-7B-0125-SFT` | 302 | 313 | 1.04 |
| `lomahony/eleuther-pythia6.9b-hh-dpo` | 302 | 313 | 1.04 |
| `lomahony/eleuther-pythia6.9b-hh-sft` | 302 | 313 | 1.04 |
| `togethercomputer/RedPajama-INCITE-7B-Chat` | 302 | 313 | 1.04 |
| `togethercomputer/RedPajama-INCITE-7B-Instruct` | 302 | 313 | 1.04 |
| `togethercomputer/RedPajama-INCITE-Base-7B-v0.1` | 302 | 313 | 1.04 |
| `HuggingFaceTB/SmolLM2-360M` | 77 | 80 | 1.04 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 77 | 80 | 1.04 |
| `deepseek-ai/deepseek-llm-7b-base` | 0 | 0 |  |
| `deepseek-ai/deepseek-llm-7b-chat` | 0 | 0 |  |
