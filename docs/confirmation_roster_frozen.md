# Confirmation census — frozen roster characterization

**FROZEN 2026-07-26, before any join.** The prereg requires this table to exist before results do; the join is where results begin to exist. Source: `data/confirmation_roster_frozen.csv`.


## Confirmation-bearing: 11

Prereg minimum is 10, target 15–20. **The floor was reached; the target was not** — see Reachability below for why.


| # | family | stratum | backend | base | aligned | language |
|---|---|---|---|---|---|---|
| 1 | `bloom` | core | vllm | `bloom-7b1` | `bloomz-7b1` | multi(46) |
| 2 | `falcon3-7b` | core | vllm | `Falcon3-7B-Base` | `Falcon3-7B-Instruct` | en |
| 3 | `glm4` | core | vllm | `glm-4-9b-hf` | `glm-4-9b-chat-hf` | zh/en |
| 4 | `map-neo` | core | vllm | `neo_7b` | `neo_7b_instruct_v0.1` | zh/en |
| 5 | `qwen3` | core | vllm | `Qwen3-8B-Base` | `Qwen3-8B` | zh/en/multi |
| 6 | `redpajama` | core | vllm | `RedPajama-INCITE-Base-7B-v0.1` | `RedPajama-INCITE-7B-Chat` | en |
| 7 | `yi` | core | vllm | `Yi-1.5-9B` | `Yi-1.5-9B-Chat` | zh/en |
| 8 | `ct-llm` | small | mps | `CT-LLM-Base` | `CT-LLM-SFT-DPO` | zh/en |
| 9 | `minicpm` | small | mps | `MiniCPM5-1B-Base` | `MiniCPM5-1B` | en |
| 10 | `stablelm` | small | mps | `stablelm-2-1_6b` | `stablelm-2-zephyr-1_6b` | en |
| 11 | `tinyllama` | small | mps | `TinyLlama-1.1B-intermediate-step-1431k-3T` | `TinyLlama-1.1B-Chat-v1.0` | en |

## Exploratory, generated but NOT confirmation-bearing: 7

Reclassified from the roster's original `stratum` column to match prereg roster rulings 1 and 2, which supersede it. The original value is retained in `roster_stratum_original`.


| family | why exploratory | roster said |
|---|---|---|
| `falcon-h1-1.5b` | ruling 2: non-transformer | exploratory |
| `falcon-h1-7b` | ruling 2: non-transformer | exploratory |
| `falcon-mamba` | ruling 2: non-transformer | core |
| `falcon3-10b` | ruling 1: Falcon3 scale series | exploratory |
| `falcon3-1b` | ruling 1: Falcon3 scale series | small |
| `falcon3-3b` | ruling 1: Falcon3 scale series | small |
| `falcon3-mamba` | ruling 2: non-transformer | exploratory |

## Unreachable: 3

**This is part of the selection story, not an appendix.** The held-out set skews recent partly because the old is unrunnable: vendor remote code drifts out of compatibility with current runtimes while the weights remain perfectly intact. Two of the three approved downloads meant to take this census from 10 to 13 failed this way.


| family | cause |
|---|---|
| `baichuan` | 3 paths, 3 reasons: aligned weights never local (CONFIG-ONLY); BaichuanForCausalLM removed from vLLM after 0.23; remote code indexes past_key_values[0][0], a DynamicCache under transformers 5.4 |
| `internlm2` | remote-code InternLM2ForCausalLM.forward() signature missing intermediate_tensors under vLLM 0.26 |
| `rwkv` | base weights incomplete, 0/17 shards; dropped pre-generation (prereg ruling 2) |

## Counts

- Roster rows: 21
- Confirmation-bearing: **11**
- Exploratory (generated): 7
- Unreachable: 3

- Generations: 21,584 total — 4,544 MPS, 17,040 vLLM

- Cross-backend sanity pair: TinyLlama, 1,136 cloud draws in `data/tinyllama_cloud_sanity.jsonl`, never written to any stash

