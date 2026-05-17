# F19: Unconditional generation & information density (10 families, 141k generations, BLT 1B byte-level scoring)

Generates 100 completions per layer from the BOS token only (no prompt) across all 10 model families. Classifies output by genre (code, exam, prose, template, math). Computes self-surprisal (model scoring own output) and reference surprisal (Pythia 1B, BLT 1B byte-level model). Converts to bits/char using exact character counts. Compares against human corpora (dreams, fiction, waking reports, abstracts).

**Aligned models produce prose with lower median information density than any human text type, with most falling at or below Shannon's English rate (~1.0 bits/char).** BLT 1B (an independent byte-level model) measures most aligned models at or below 1.0 bits/char (e.g. OLMo SFT 0.93, DPO 0.96, Qwen DPO 0.95). All human text types have higher medians: fiction 1.49, dreams 1.32, abstracts 1.28, waking reports 1.24. Distributions overlap, but the systematic shift is clear.

![Human vs AI information density](../figures/F19_blt_human_vs_ai_bos.png)

**Battery prompts reverse the direction.** On prompted text, alignment *increases* BLT surprisal (OLMo base 1.42 → SFT 1.55 bits/char). Aligned models substitute unexpected continuations that are fluent to themselves but surprising to external models — the displacement/swerve effect measured as cross-entropy.

**BOS generation reveals family-specific "resting states."** OLMo SFT defaults to chat templates ("You are a helpful function-calling AI assistant"). Llama Instruct defaults to Chinese medical exam questions. Qwen base is 43% exam questions (pre-socialised). Each family finds a different attractor when given no prompt.

![BOS genre distribution](../figures/F19_bos_genre_distribution.png)

**Alignment compression is content-independent on battery prompts.** All 9 content categories compress from ~1.0 to ~0.7 bits/char (self-surprisal). The delta is uniform (Kruskal-Wallis p=0.99 on per-family deltas).

![Self-surprisal BOS prose](../figures/F19_self_surprisal_bos_prose.png)

**Alignment creates private language.** The gap between self-surprisal and reference surprisal widens with alignment. Aligned models produce text increasingly predictable to themselves but not to external observers.

![Private language gap](../figures/F19_private_language_gap.png)

**Amber anomaly confirmed.** AmberSafe (DPO) has *higher* self-surprisal than base (1.56 vs 1.29 bits/char). The safety model surprises itself — unique across all families.

**Shannon's communication model applied to alignment.** Alignment is noise in Shannon's precise sense: it transforms the signal between source and reception, reducing channel capacity and increasing redundancy. The twist is that this noise is desired — but it has the same informational consequence as unwanted interference.

Results in `data/generation_analysis.parquet`, `data/blt_human_corpora.csv`. Notebook: `notebooks/F19_bos_entropy.ipynb`. CLI: `malign bos-generate`, `malign surprisal`.
