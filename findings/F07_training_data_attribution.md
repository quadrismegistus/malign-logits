# F07: Training data attribution: objective vs data composition (OLMo 3)

The OLMo 3 technical report (arXiv:2512.13961) documents exact data mixtures for every training stage, making it possible to ask whether the displacement patterns above are driven by the training *objective* (SFT cross-entropy, DPO preference loss) or by the training *data* (specific safety datasets that teach the model what to refuse).

**Safety data is a small fraction of post-training.** OLMo's SFT stage uses ~110k safety prompts (CoCoNot, WildGuardMix, WildJailbreak) out of 2.15M total (~5%). DPO uses ~27k safety prompts out of 260k (~10%). The remaining 90-95% is math, code, instruction-following, science, and chat. Yet these small slices produce the displacement patterns documented above.

**The SFT/DPO division of labour implicates the objective, not the data.** Sexual repression happens overwhelmingly at SFT (~90% of displacement), while violence requires DPO. If displacement were purely data-driven, both stages would repress both content types proportionally to their safety data share. Instead, each training objective selectively targets different content — SFT's cross-entropy loss on safety completions is sufficient to suppress sexual content, but violence requires the contrastive signal of DPO preference pairs to repress. The *how* of learning matters, not just the *what*.

**DPO's contrastive signal comes from capability gaps, not safety annotation.** OLMo's DPO uses delta learning: chosen responses from Qwen 32B, rejected responses from Qwen 0.6B. The preference signal reflects the difference between a capable and incapable model, not explicit safety labelling. That violence repression emerges from this capability delta — rather than from the 10% of DPO data that is explicitly safety-related — suggests the DPO objective itself produces repression as a side-effect of learning to prefer competent responses.

**Base model mass on transgressive tokens reflects internet frequency.** Pretraining is 76% Common Crawl (4.5T tokens of filtered web text). The base model's high probability mass on sexual and violent tokens is not a curation artefact — it reflects the libidinal economy of the training corpus. What alignment displaces is, in Freudian terms, genuine drive energy: statistical cathexis accumulated from the collective text of the internet.

**Three datasets perform the safety socialisation of a 7-billion parameter model:**
| Dataset | Purpose | SFT prompts | DPO prompts |
|---|---|---|---|
| CoCoNot | Contextual refusal (when to refuse, not blanket blocking) | 10,957 | 2,203 |
| WildGuardMix | Adversarial safety prompts and responses | 49,373 | 12,037 |
| WildJailbreak | Jailbreak resistance | 49,965 | 12,431 |

Source: OLMo 3 technical report, Tables 30 and 20 (Team OLMo, arXiv:2512.13961, December 2025).
