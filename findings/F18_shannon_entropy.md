# F18: Shannon entropy: alignment as lossy compression of drive (10 families, 47 prompts)

Shannon entropy H(p) of the full-vocabulary logit distribution at the last token position, computed from cached logits across all families and alignment stages. Redundancy = 1 − H/H_max, where H_max = log(vocab_size).

**Alignment universally increases redundancy (reduces entropy).**

| Family | Base H (nats) | Aligned H | Δ entropy | Δ redundancy |
|---|---|---|---|---|
| **Amber** | 4.15 | 2.41 | **−1.74** | **+0.168** |
| Zephyr | 3.79 | 3.09 | −0.69 | +0.067 |
| Tulu | 3.99 | 3.39 | −0.61 | +0.052 |
| Qwen | 3.93 | 3.34 | −0.59 | +0.050 |
| SmolLM2 | 4.42 | 3.89 | −0.53 | +0.049 |
| OLMo | 4.57 | 3.90 | −0.67 | +0.058 |
| Llama | 3.99 | 3.62 | −0.37 | +0.032 |
| Pythia | 3.73 | 3.47 | −0.26 | +0.024 |
| OLMo-tiny | 4.09 | 3.89 | −0.21 | +0.018 |

Base models carry ~4 nats of information per next-token prediction. Alignment compresses this to ~3–3.5 nats. The ordering tracks alignment intensity from the surprisal analysis (F15). Amber loses the most information (−1.74 nats); OLMo-tiny the least (−0.21).

**Shannon framing.** The base model's entropy is the channel capacity of the primary process — the diversity of possible continuations the drive field can produce. Alignment reduces this capacity: fewer possible next tokens, more predictable output. The redundancy alignment adds is literally Shannon redundancy — the fraction of each token that is predictable from context rather than carrying new information. In Lyotardian terms, theatricalization compresses the libidinal band's entropy into the narrower channel of socially legible output.

**SFT/DPO division of entropy labour** (3+ layer families):
- **Amber**: SFT does 66% of entropy reduction (4.15→3.00), DPO does 34% (3.00→2.41)
- **Tulu**: SFT does 42% (3.99→3.73), DPO does 58% (3.73→3.29) — DPO-dominant
- **Zephyr**: SFT does 72% (3.79→3.29), DPO does 28% (3.29→3.09)
- **OLMo-tiny**: SFT does 81% (4.09→3.92), DPO barely changes (3.92→3.90)

The SFT/DPO entropy split parallels the surprisal split (F15) and the geometric split (F12), confirming that these are measuring the same underlying operation at different levels.

**Alignment removes the noise of ambiguity, not obscenity (Kruskal-Wallis p=0.015).** Unlike within-passage surprisal (p=0.99) and self-surprisal (p=0.61), logit-level entropy reduction *does* differ by content category. The predictor is not transgressiveness but **base entropy** — how uncertain the base model was about the prompt (r=−0.84, p=0.004):

| Category | Base H (nats) | Δ H | Interpretation |
|---|---|---|---|
| substance | 5.09 | −0.86 | High ambiguity → large compression |
| sexual_liminal | 4.69 | **−0.95** | Most compressed |
| neutral | 4.60 | −0.81 | |
| violence_liminal | 4.48 | −0.71 | |
| power | 3.89 | −0.71 | |
| sexual_explicit | 3.88 | −0.43 | Low ambiguity → small compression |
| profanity | 3.52 | −0.63 | |
| death | 3.42 | −0.45 | |
| violence_explicit | 2.78 | **−0.45** | Least compressed |

Sexual_liminal loses twice the entropy of sexual_explicit (p=0.013). "She touched his arm and he felt a sudden" has many possible continuations; "He pushed her onto the bed and started to" has fewer. Alignment collapses the possibility space of the ambiguous prompt — it removes interpretive openness, not obscenity per se.

**Self-surprisal: alignment compresses below natural language.** Feed each passage back through the model that generated it to measure the true information rate (Shannon's source entropy). Base models produce text at ~1.0 bits/char — roughly Shannon's estimate for English. Alignment pushes 9 of 10 families below this line.

![Self-surprisal by family and layer](figures/self-surprisal-by-family-layer.png)

| Family | Base (bits/char) | Aligned | Δ | Below Shannon? |
|---|---|---|---|---|
| OLMo | 1.21 | 0.94 | −0.27 | Yes |
| OLMo-tiny | 1.16 | 0.92 | −0.24 | Yes |
| Pythia | 1.10 | 0.91 | −0.19 | Yes |
| Tulu | 1.09 | 0.64 | −0.45 | Yes |
| SmolLM2 | 1.09 | 1.01 | −0.08 | Barely |
| Qwen-tiny | 1.04 | 0.77 | −0.28 | Yes |
| Amber | 1.03 | 0.98 | −0.05 | Yes |
| Zephyr | 1.02 | 0.64 | −0.38 | Yes |
| Qwen | 0.87 | 0.45 | −0.43 | Yes (already at base) |

**Alignment creates private language.** The gap between self-surprisal and reference surprisal (Pythia 1B evaluating the same text) *widens* with alignment. Aligned models produce text that is increasingly predictable to themselves but not to external observers. Qwen DPO: self-surprisal 1.23 nats, Pythia reference 2.60 nats — a gap of 1.37 nats. The aligned model speaks a private dialect that an external model cannot compress as efficiently.

![Self vs reference gap](figures/self-vs-reference-surprisal-gap.png)

**The Amber anomaly.** AmberChat (SFT, no safety data) has self-surprisal 0.69 bits/char, but AmberSafe (DPO, safety-tuned) jumps back up to 0.98. The safety model is *more surprised by its own output* than the chat model — it produces text its own probability landscape doesn't fully endorse. A computational signature of the superego's excessive demand.

Results in `data/self_surprisal.csv`, `data/shannon_entropy.csv`. Notebook: `notebooks/10_shannon.ipynb`.
