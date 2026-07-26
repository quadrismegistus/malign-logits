# F19: Unconditional Generation & Information Density

***

**Finding.**

Alignment compresses the model's unconditional output below Shannon's English entropy rate (~1.0 bits/char), as measured by an independent byte-level model (BLT 1B). All human text types (fiction, dreams, waking reports, abstracts) remain above this threshold. When prompted, the pattern reverses: alignment *increases* cross-entropy, producing text that is more predictable to itself but more opaque to external models ("private language").

***

**Method.**

1. Generate 100 completions per layer from BOS token only (no prompt) across 10 model families
2. Classify generations by genre (code, exam, prose, template, math) and language
3. Compute self-surprisal (model scoring own output) and reference surprisal (Pythia 1B, BLT 1B)
4. Convert to bits/char using exact character counts per token
5. Compare against human corpora (dreams, fiction, waking reports, academic abstracts)

Shannon's 1.0 bits/char (1951) is the standard reference for English entropy. Self-surprisal is the exact source entropy rate (model is the source). BLT cross-entropy is an upper bound measured by an independent byte-level judge.

***

**Key results.**

**Aligned BOS output is sub-Shannon.** SFT: 0.93 bits/char, DPO: 0.96 — below the 1.0 threshold. Confirmed by both self-surprisal and independent BLT scoring.

**All human text is supra-Shannon.** Fiction: 1.49, Dreams: 1.32, Abstracts: 1.28, Waking: 1.24 bits/char (BLT).

**Battery prompts reverse the direction.** On prompted text, alignment *increases* BLT surprisal (base 1.42 → SFT 1.55). The displacement/swerve effect: aligned models substitute unexpected continuations that are fluent to themselves but surprising to external models.

**Genre confound is real but controllable.** Code (0.56 bits/char) and template (3.07) have very different information densities. Prose-only analysis preserves all findings.

**BOS generation reveals family-specific "resting states."** OLMo SFT defaults to chat templates ("You are a helpful function-calling AI assistant"). Llama Instruct defaults to Chinese medical exam questions. Qwen base is 43% exam questions (pre-socialised). Each family finds a different attractor.

**Alignment compression is content-independent on battery prompts.** All 9 content categories (sexual, violent, neutral, etc.) compress from ~1.0 to ~0.7 bits/char. The delta is uniform (Kruskal-Wallis p=0.99 on per-family deltas).

**Amber anomaly confirmed.** AmberSafe (DPO) has *higher* self-surprisal than base (1.56 vs 1.29 bits/char). The safety model surprises itself — unique across all families.

***

**Figures.**

![Human vs AI information density](../figures/F19_blt_human_vs_ai_bos.png)

![BOS genre distribution](../figures/F19_bos_genre_distribution.png)

![Self-surprisal BOS prose](../figures/F19_self_surprisal_bos_prose.png)

![Self-surprisal battery by category](../figures/F19_self_surprisal_battery_category.png)

![Private language gap](../figures/F19_private_language_gap.png)

***

**Shannon's communication model.**

```
INFORMATION SOURCE → TRANSMITTER → [NOISE] → RECEIVER → DESTINATION
   (model weights)    (sampling)   (alignment)  (reader)    (user)
```

Alignment is noise in Shannon's precise sense: it transforms the signal between source and reception, reducing channel capacity and increasing redundancy. The twist is that this noise is desired — but it has the same informational consequence as unwanted interference.

***

**Self bits/char: BOS prose only (Shannon ≈ 1.0).**

| family    |   base |     ego |   instruct |   superego |
|:----------|-------:|--------:|-----------:|-----------:|
| amber     |  1.294 |   0.994 |    nan     |      1.558 |
| llama     |  0.787 | nan     |    nan     |      1.244 |
| olmo      |  1.002 |   0.999 |      0.83  |      0.82  |
| olmo-tiny |  0.543 |   1.136 |      0.665 |      0.798 |
| pythia    |  0.597 |   0.597 |    nan     |      0.58  |
| qwen      |  0.729 | nan     |    nan     |      0.435 |
| qwen-tiny |  1.025 | nan     |    nan     |      0.725 |
| smol      |  0.833 | nan     |    nan     |      0.658 |
| tulu      |  0.787 | nan     |    nan     |    nan     |
| zephyr    |  0.816 |   0.919 |    nan     |      0.978 |

***

**Self bits/char: battery prompts (Shannon ≈ 1.0).**

| family    |   base |     ego |   instruct |   superego |
|:----------|-------:|--------:|-----------:|-----------:|
| amber     |  1.395 |   0.85  |    nan     |      1.422 |
| llama     |  0.94  | nan     |    nan     |      0.761 |
| olmo      |  1.12  |   1.164 |      0.995 |      1.121 |
| olmo-tiny |  1.062 |   0.951 |      0.755 |      0.769 |
| pythia    |  1.024 |   0.91  |    nan     |      0.871 |
| qwen      |  0.818 | nan     |    nan     |      0.45  |
| qwen-tiny |  0.975 | nan     |    nan     |      0.7   |
| smol      |  1.052 | nan     |    nan     |      0.925 |
| tulu      |  0.94  |   0.875 |      0.593 |      0.571 |
| zephyr    |  1.142 |   0.953 |    nan     |      0.764 |

***

**BLT bits/char: human corpora.**

| source      |   mean |   std |
|:------------|-------:|------:|
| abstracts   |  1.275 | 0.34  |
| c20_fiction |  1.494 | 0.323 |
| dreams      |  1.322 | 0.291 |
| waking      |  1.241 | 0.316 |

***

**Data.**

- `data/generation_analysis.parquet` — 141k generations with genre, self/ref surprisal, bits/char
- `data/blt_human_corpora.csv` — BLT scores for dreams, fiction, waking, abstracts
- `data/blt_combined.csv` — combined human + AI BLT scores
- Generation cache: `data/raw/cache/generations/`
- Surprisal caches: `data/raw/cache/self_surprisal/`, `data/raw/cache/ref_surprisal/`

***

**Notebook.**

`notebooks/F19_bos_entropy.ipynb`

***

**CLI.**

```bash
malign bos-generate --family olmo --n 100          # generate from BOS
malign bos-generate --prompt "The" --n 100         # generate from custom prompt
malign surprisal --self                             # self-surprisal for all cached generations
malign surprisal --ref itazap/blt-1b-hf            # BLT byte-level reference surprisal
```


---

**Provenance check, 2026-07-26 — ONE HALF OF THE HEADLINE DOES NOT REPRODUCE.**

Recomputed from `data/blt_combined.csv` and `data/generation_analysis.parquet`,
both modified after this finding was booked.

*Reproduces exactly.* Human text types, BLT mean bits/char: fiction **1.49**,
dreams **1.32**, abstracts **1.28**, waking **1.24** — all four as published, all
above 1.0.

*Reproduces approximately.* Self-surprisal on prose-only BOS: SFT (ego) **0.94**
against 0.93 published; DPO (superego) **0.85** against 0.96 published. The DPO
number differs but in the direction of *more* compression, so the sub-Shannon
claim holds on this measure.

*Does NOT reproduce.* The claim is "SFT: 0.93, DPO: 0.96 — below the 1.0
threshold. **Confirmed by both self-surprisal and independent BLT scoring**." On
BLT, prose-only BOS gives SFT median **1.21** / mean 1.44 and DPO median **1.05**
/ mean 1.20 — *above* 1.0, not below. English-only filtering does not change it
(SFT 1.19/1.43, DPO 1.01/1.17). So the BLT half of the confirmation does not hold
on the current file under any filter I could infer.

**This is flagged, not corrected.** Today established the distinction between a
claim being wrong and its record being lost, and I cannot tell which this is: the
original analysis may have applied a filter the finding does not state. What the
finding needs is its exact filter written down — genre, prompt_type, language,
and which of `self_bits_per_char` / `blt_bits_per_char` — after which this either
reproduces or it doesn't. Until then the *self-surprisal* half of the sub-Shannon
claim stands and the *BLT confirmation* should not be cited.
