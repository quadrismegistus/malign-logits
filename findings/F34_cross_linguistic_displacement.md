# F34: Cross-Linguistic Displacement — The Class Engine Is Language-Dependent

## Summary

Alignment operates in opposite directions by language within the same weights. In English, alignment installs compliance (worker prompts) and suppresses transgressive vocabulary (kill→punch/scream). In Chinese, alignment installs agency (gratitude→"what should I do") and can intensify transgressive vocabulary (sexual, revenge). The pre-socialisation is in the pretraining corpus: Chinese-primary models embed deference that alignment overcomes; English-primary models embed procedural advice that alignment amplifies.

## Models tested

| Model | Training language | Layers | Chinese capable? |
|-------|------------------|--------|-----------------|
| CT-LLM 2B | Chinese-primary (800B ZH, 100B EN) | 3 (base/SFT/DPO) | Native |
| MAP-Neo 7B | Bilingual (4.5T mixed) | 3 (base/SFT/DPO) | Native |
| Qwen 2.5 7B | English-primary, Chinese-capable | 2 (base/instruct) | Yes |
| Qwen3 8B | English-primary, Chinese-capable | 2 (base/instruct) | Yes |
| Llama 3.1 8B | English-primary, some Chinese | 2 (base/instruct) | Marginal |
| DeepSeek 7B | English-only (despite Chinese lab) | 2 (base/chat) | No (0 tokens) |

## Key findings

### 1. Worker deference→agency split by pretraining language

| Model | Training lang | Chinese base top | After alignment |
|-------|--------------|-----------------|-----------------|
| CT-LLM | Chinese-primary | 感谢 (grateful) 5.9% | 怎么办 (what to do) 17.0% |
| MAP-Neo | Bilingual | 怎么做 7.2% + 感谢 6.1% | 怎么做 36.5% + 如何 25.9% |
| Qwen 2.5 | English-primary | 怎么做 57.3% | 怎么做 46.1% |
| Qwen3 | English-primary | 怎么做 53.7% | 怎么做 63.0% |
| Llama | English-primary | 怎么 50.5% | 怎么 49.2% |

Chinese-primary models (CT-LLM, MAP-Neo) start with deference (gratitude), alignment installs agency. English-primary models (Qwen, Llama) start with agency already (50%+). The pre-socialisation is in the pretraining corpus.

### 2. Anger: language-dependent displacement direction

MAP-Neo: alignment promotes 报复 (revenge) 4.4%→28.5% + 惩罚 (punish) in Chinese while suppressing kill 11.2%→4.2% in English. Same weights, opposite direction.

CT-LLM: Chinese base has no violence vocabulary (top words: leave, divorce). Nothing to displace.

Qwen/Llama: Chinese anger is mild (leave, revenge 3-5%). English follows standard kill→scream/punch.

### 3. Sexual intensification in Chinese

CT-LLM: alignment intensifies Chinese sexual (undress 18.7%→24.8%).
MAP-Neo: SFT intensifies in both languages (undress 22.4%→31.7% Chinese, und 17.8%→40.4% English).
Qwen: light-touch on sexual in both languages.

Sexual suppression is English-specific and model-specific, not universal.

### 4. The gratitude-to-agency shift is specific to Chinese-primary models

Both CT-LLM and MAP-Neo start with 感谢 (grateful/thank) toward the exploitative boss in Chinese. English-primary models skip this stage. Pre-socialisation hypothesis: Chinese-primary pretraining data embeds a deferential relationship to authority that English pretraining data does not.

## Interpretation

The class engine (ch05 §5.6) is not universal but language-dependent. Alignment amplifies whatever political structure is already encoded in the pretraining corpus. Chinese text embeds deference → alignment installs agency. English text embeds procedural advice → alignment amplifies compliance. The politics are in the language, not the method.

Refines the PERMANOVA country=corpus finding (F31): country effect is not just about token counts but about the political structure of the language community as encoded in text.

## Chapter placement

- ch05 primary (displacement operation is language-dependent)
- ch01 cross-ref (what is in the corpus determines what alignment does)
- ch09 cross-ref (cross-family and cross-linguistic variation)
- CI article §VI: one sentence on the language-dependent class engine

## Data

- Smoke tests: 4 prompts × 2 languages × 5 models (this finding)
- Full battery: queued as book experiment (73 prompts × Chinese on CT-LLM + MAP-Neo)
- CT-LLM full word_probs: complete (73 EN prompts)
- MAP-Neo full word_probs: running
