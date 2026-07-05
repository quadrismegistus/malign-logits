# Models studied: 17 primary families from 6 countries

> **Scope note (as of 2026-07-05).** This table is the *curated deep-dive set* — the 17 families carried through the full analysis pipeline with per-family interpretive notes. The code registers **47 families / 107 checkpoints** total in `MODEL_FAMILIES` (run `malign info` for the live list); the remainder are scale ladders, architecture variants (SSM/MoE/RNN), and method ablations used in specific findings (e.g. F33 scale, F35 architecture). Family counts quoted elsewhere in older docs (20 in `model_candidates.md`, 59 in earlier PROBE notes) are point-in-time — `MODEL_FAMILIES` is the single source of truth.

| Key | Family | Layers | Developer | Country | Significance |
|-----|--------|--------|-----------|---------|-------------|
| **olmo** | OLMo 3 7B | 4 (base / SFT / DPO / RLVR) | Allen AI | USA | Default family. Fully transparent 4-layer pipeline with documented data mixtures. |
| **olmo-tiny** | OLMo 2 1B | 4 | Allen AI | USA | Smaller OLMo, full 4-layer. Scale comparison (r=-0.60 institutional correlation with 7B). |
| **olmo-think** | OLMo 3 7B Think | 3 (base / Think-SFT / Think-DPO) | Allen AI | USA | Same base as olmo but reasoning-trained. Tests effect of reasoning objective on displacement. |
| **tulu** | Tulu 3.1 8B | 4 (+ 5 ablations) | Allen AI (Llama base) | USA | Same base as Llama Instruct but transparent Allen AI alignment. 5 data-mixture ablations (no-safety, no-math, no-persona, no-wildchat). |
| **llama** | Llama 3.1 8B | 2 (base / instruct) | Meta | USA | Opaque, single-step alignment. The "industrial baseline." |
| **zephyr** | Zephyr 7B | 3 (Mistral base / SFT / DPO) | HuggingFace H4 (Mistral base) | USA / France | No safety data anywhere in the pipeline. Cleanest evidence that instruction-following form alone produces foreclosure. |
| **mistral** | Mistral 7B | 8 variants | Mistral AI + 5 orgs | France | Ecosystem diversity — same base, 5 different orgs aligned it independently. Tests base-determined vs alignment-determined displacement. |
| **qwen** | Qwen 2.5 7B | 2 | Alibaba | China | Chinese model, different training data, often lapses into Chinese exam questions. Nearly invisible alignment (JS 0.044). |
| **qwen-tiny** | Qwen 2.5 0.5B | 2 | Alibaba | China | Cross-checks Qwen findings at smaller scale. |
| **qwen3** | Qwen3 8B | 2 | Alibaba | China | Native thinking model. De-foreclosure: base forecloses on anger (blanks), aligned restores "kill." |
| **deepseek** | DeepSeek 1 7B | 2 | DeepSeek | China | First Chinese LLM well-known to the West. |
| **amber** | Amber 7B | 3 (base / chat / safe) | LLM360 (MBZUAI consortium) | USA / UAE | Distinct from other 3-layer families: explicit stages. AmberChat is socialised, AmberSafe is safety-tuned. All training data public. Pre-socialised base (scream > kill in beams). |
| **pythia** | Pythia 6.9B | 3 | EleutherAI (SFT/DPO from lomahony) | USA | SFT and DPO trained on identical data (Anthropic HH-RLHF). Isolates training objective from data composition. Lightest alignment footprint of any 7B. |
| **smol** | SmolLM2 360M | 2 | HuggingFace | USA / France | Smallest in the lineup. Transparent on 4/5 prompts. |
| **smol3** | SmolLM3 3B | 2 | HuggingFace | USA / France | Mid-scale. Native thinking support. |
| **falcon** | Falcon 7B | 2 | TII | UAE | Non-Western org. RefinedWeb training data. |
| **yi** | Yi 9B | 2 | 01.AI | China | Chinese, slightly larger scale (9B). |

## Reasoning / distillation models (cross-family)

| Key | Distilled from | Base | Country | Significance |
|-----|---------------|------|---------|-------------|
| **r1-llama** | DeepSeek R1 | Llama 3.1 8B | China → USA | Reasoning distillation onto Western base. |
| **r1-qwen** | DeepSeek R1 | Qwen 2.5 7B | China → China | Reasoning distillation onto Chinese base. |

## Not yet fully integrated (registered, missing logits)

| Key | Family | Developer | Country | Status |
|-----|--------|-----------|---------|--------|
| **baichuan** | Baichuan2 7B | Baichuan | China | Needs `trust_remote_code=True` |
| **gemma** | Gemma 7B | Google | USA | Gated repo, needs HF token |
| **internlm** | InternLM2.5 7B | Shanghai AI Lab | China | Needs `trust_remote_code=True` |

## Considered but not added

| Family | Developer | Why considered | Why not added |
|--------|-----------|---------------|---------------|
| **PKU-Beaver** | PKU-Alignment | Separate helpfulness vs harmlessness RLHF checkpoints. Would directly test whether safety and helpfulness produce different displacement. | Scope decision: 17 families sufficient. Helpfulness vs harmlessness decomposition partially available through Tulu ablations (no-safety variant). |
| **Mixtral** | Mistral AI | MoE architecture — expert routing adds a dimension. | Different object. Expert routing means displacement could be expert-specific. Needs its own study, not a single comparison. |
| **CodeLlama / StarCoder** | Meta / BigCode | Code-specialized. Would test whether code training produces displacement differently. | Code is not a content category in our prompt battery. Displacement analysis is about natural language narrative completion. |
| **GPT-4o-mini** | OpenAI | Proprietary baseline. Top-20 logprobs available via API. | Breaks open-weight commitment. Can't teacher-force, can't beam search, can't extract full logit distributions. Methodological break. |
| **Llama 3.3 / 4** | Meta | Newer than our pipeline. | Diminishing returns — another Llama variant with opaque alignment. |

## Coverage summary

- **9 organizations**: Allen AI, Meta, Mistral AI, Alibaba, EleutherAI, DeepSeek, HuggingFace, TII, 01.AI
- **6 countries**: USA, France, China, UAE (+ cross-national distillation)
- **Scales**: 360M to 9B parameters
- **Alignment methods**: SFT, DPO, RLHF, RLVR, reasoning distillation, data-mixture ablation
- **Training transparency**: fully open (Amber, Pythia) → partially open (OLMo, Tulu) → closed (Llama, Qwen, DeepSeek)
- **42 model checkpoints** total across 17 base families
