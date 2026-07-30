# CJK tokenizer coverage

`data/cjk_coverage.csv` — one row per MODEL ID. Rebuilt by
`scripts/build_cjk_coverage.py`.

**110 models, 2026-07-30.** {'FLUENT': 28, 'MARGINAL': 4, 'NOMINAL': 58, 'PARTIAL': 20}
 · **drops_cjk: 0**

## MEASURED THROUGH THE LOADER TABLE, and the first build was not

The first build called `AutoTokenizer` directly and recorded
**deepseek-llm-7b at 0 CJK characters**. That was the loading bug, not the
model: transformers v5 (#45488) installs a SentencePiece Metaspace
pre-tokenizer over the ByteLevel one the repo declares, deleting whitespace and
dropping CJK entirely, with `unk_token: null` so nothing raises. Through
`twp_cloud.load_tokenizer` the same model measures **3,429 characters /**
**18,006 tokens**.

**A file that describes tokenizers must load them the way the runner loads**
**them, or it documents a different object.** The same correction was applied
to preconditions 5 and 6.

## Columns

| column | meaning |
|---|---|
| `cjk_chars` | distinct CJK codepoints reachable. Sets the tier. |
| `cjk_tokens` | tokens decoding to bare CJK. |
| `tokens_per_char` | **the fluency signal** — words vs spellings. |
| `drops_cjk` | tokenizer discards CJK input entirely. Separate axis from tier. |

## Tiers

Cut on external reference points: **~3,500 characters covers modern
general-purpose Chinese; ~2,500 is the PRC primary-school list.**

| tier | rule |
|---|---|
| FLUENT | >= 3,500 |
| MARGINAL | 2,500–3,499 |
| PARTIAL | 1,000–2,499 |
| NOMINAL | < 1,000 |

## The two metrics can disagree, and deepseek is the case

```
ct-llm    21,006 chars   71,062 tokens   3.38/char   broad and deep
deepseek   3,429 chars   18,006 tokens   5.25/char   NARROW AND DEEPEST
amber        700 chars      700 tokens   1.00/char   narrow and flat
```

**deepseek has the roster's highest tokens-per-character** — a word-level
Chinese vocabulary — while sitting **71 characters below the FLUENT cut**.
`cjk_chars` measures breadth of glyph coverage; `tokens_per_char` measures
whether the vocabulary holds words or spellings. MARGINAL exists for exactly
this: a cut point on a continuum, surfacing the case rather than hiding it.

## Chinese scope

**FLUENT + MARGINAL, excluding `drops_cjk` — 32 arms** (was 30 before the
deepseek correction). Chinese prompts do not run on PARTIAL or NOMINAL arms:
a displacement measurement requires a paradigm to displace, and those arms
return particles and pronouns.

### FLUENT — 28

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

### MARGINAL — 4

| model | cjk_chars | cjk_tokens | tok/char |
|---|---|---|---|
| `deepseek-ai/deepseek-llm-7b-base` | 3429 | 18006 | 5.25 |
| `deepseek-ai/deepseek-llm-7b-chat` | 3429 | 18006 | 5.25 |
| `tiiuae/Falcon-H1-1.5B-Base` | 2829 | 8933 | 3.16 |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 2829 | 8933 | 3.16 |

### PARTIAL — 20

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

### NOMINAL — 58

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
