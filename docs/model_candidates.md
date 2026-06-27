# Model Candidates — HuggingFace Survey (2026-06-27)

Comprehensive survey of model families with at least 2 training stages available as separate HuggingFace checkpoints. Organized by what they would add to the project.

## Already registered (20 families)

olmo, olmo-think, olmo-tiny, olmo-32b, olmoe, amber, zephyr, pythia, tulu, map-neo, ct-llm, internlm2, smol, smol3, qwen-tiny, qwen, qwen3, llama, llama-70b, deepseek-7b

## Priority 1 — Architecturally distinct

| Family | Arch | Stages | Size | HF IDs |
|--------|------|--------|------|--------|
| **OLMo Hybrid 7B** | SSM-Transformer hybrid | 3 (base/SFT/DPO) | 7B | `allenai/Olmo-Hybrid-7B`, `Olmo-Hybrid-Instruct-SFT-7B`, `Olmo-Hybrid-Instruct-DPO-7B` |
| **Falcon-Mamba 7B** | Pure SSM (Mamba) | 2 (base/instruct) | 7B | `tiiuae/falcon-mamba-7b`, `falcon-mamba-7b-instruct` |
| **Falcon-H1 7B** | SSM-Transformer hybrid | 2 (base/instruct) | 7B | `tiiuae/Falcon-H1-7B-Base`, `Falcon-H1-7B-Instruct` |
| **OLMoE 1B-7B** | MoE (64 experts, 8 active) | 4 (base/SFT/DPO/RLVR) | 7B total / 1B active | Already registered, pipeline running |
| **DeepSeek MoE 16B** | MoE (64 experts, 6 active) | 2 (base/chat) | 16B total | `deepseek-ai/deepseek-moe-16b-base`, `deepseek-moe-16b-chat` |

## Priority 2 — Methodologically distinct

| Family | Method contrast | Stages | Size | HF IDs |
|--------|----------------|--------|------|--------|
| **Archangel** | DPO vs KTO vs PPO vs SLIC on same SFT | 4+ variants | 2.8B-30B | `ContextualAI/archangel_sft_pythia2.8b` → `archangel_sft-dpo_*`, `sft-kto_*`, `sft-ppo_*`, `sft-slic_*` |
| **RLHFlow Iterative DPO** | Progressive DPO (iter1→iter3→final) | 5 | 8B | `RLHFlow/LLaMA3-SFT-v2` → `Llama3-v2-iterative-DPO-iter{1,2,3}`, `final` |
| **Tulu 2 13B** | Has PPO stage (dropped in Tulu 3) | 4 | 13B | `allenai/tulu-2-13b`, `tulu-2-dpo-13b`, `tulu-v2.5-ppo-13b-*` |
| **Janus 7B** | DPO vs ORPO from same SFT | 3+ | 7B | `kaist-ai/janus-7b`, `janus-dpo-7b`, `janus-orpo-7b` |

## Priority 3 — Chinese/multilingual

| Family | Language | Stages | Size | HF IDs |
|--------|----------|--------|------|--------|
| **GLM-4 9B** | Chinese+English | 2 + DPO variant | 9B | `zai-org/glm-4-9b-hf`, `glm-4-9b-chat-hf`, `LongReward-glm4-9b-DPO` |
| **MiniCPM5 1B** | Chinese (Tsinghua) | 3 (base/SFT/final) | 1B | `openbmb/MiniCPM5-1B-Base`, `MiniCPM5-1B-SFT`, `MiniCPM5-1B` |
| **Baichuan2 7B** | Chinese+English | 2 + pretraining ckpts | 7B | `baichuan-inc/Baichuan2-7B-Base`, `Baichuan2-7B-Chat` |
| **Qwen3.5 9B** | Chinese+English | 2 | 9B | `Qwen/Qwen3.5-9B-Base`, `Qwen3.5-9B` |
| **Yi 1.5 9B** | Chinese+English | 2 | 9B | `01-ai/Yi-1.5-9B`, `Yi-1.5-9B-Chat` |
| **Sailor2** | Southeast Asian | 3 (pre/SFT/chat) | 1B-20B | `sail/Sailor2-{1,3,8,14,20}B-Pre/SFT/Chat` |

## Priority 4 — Scale/coverage

| Family | Stages | Sizes | HF IDs |
|--------|--------|-------|--------|
| **Gemma 2/3** | 2 (base/instruct) | 1B-27B (9 variants) | `google/gemma-{2,3}-{size}-{pt,it}` |
| **Falcon3** | 2 (base/instruct) | 1B-10B | `tiiuae/Falcon3-{1,3,7,10}B-Base/Instruct` |
| **OLMo-2 13B** | 5+ (SFT/DPO/RLVR1/RLVR2) | 13B | `allenai/OLMo-2-1124-13B-*` |
| **StableLM 2** | 3 (base/chat-SFT/zephyr-DPO) | 1.6B, 12B | `stabilityai/stablelm-2-{1_6b,12b}*` |
| **RedPajama** | 3 (base/instruct-SFT/chat-RLHF) | 3B, 7B | `togethercomputer/RedPajama-INCITE-*` |

## Excluded

- **InternLM2**: Custom modeling code incompatible with current transformers (see TODO.md)
- **Phi-3/3.5/4-mini**: Instruct-only, no base models
- **CohereForAI (Aya/Command R)**: No base models released
- **MosaicML MPT**: Repos removed from HuggingFace
- **Cerebras-GPT**: Base-only, no aligned variants
- **DeepSeek V2/V3/R1**: Only final aligned model released (no SFT/DPO intermediates)
