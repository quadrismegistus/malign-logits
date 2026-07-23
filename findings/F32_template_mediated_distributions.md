---
status: solid-by-design
grade: B
date: 2026-06-25
role: finding
instruments: [logit-mass, census]
families: [olmo, amber, llama, qwen, tulu, zephyr]
chapters: [ch07]
data: []
scripts: []
---
# F32: Template-Mediated Distributions — Task Switch, Not Distribution Filter

**Summary**

Continue mode (chat template + "Continue this text:") does not filter the raw-mode distribution — it replaces the task entirely. The model responds as an assistant, not as a narrative continuator. JS between raw and continue mode is 0.66–0.69 (near theoretical maximum), confirming incommensurable distributions.

**Three distributional levels**

1. **Weights-only distribution** (raw mode) — narrative continuation, no template. kill→scream. Ch05/06.
2. **Template-mediated distribution** (continue mode) — assistant response, chat template as context. Apologies, shatter, numbered advice. Ch07.
3. **Generated text** (sampled output) — Jakobson space, surface output. Also ch07 but different data.

**Three response strategies**

Visible in both distributional tables and generated text:

**1. Narrative sublimation (Llama on anger)**

The model stays in the story but elevates the register. Distributional: scream (24.0% raw-aligned) drops to 0.8% in continue; replaced by shatter (26.9%), stomp (13.4%), scorch (10.0%). Generational: "rip her hair out, slam her fist on the table, and scream at the top of her lungs. The injustice of it all was just too much to bear."

**2. Hard refusal (Llama on sexual, OLMo on sexual)**

The model exits narrative entirely. OLMo-7B: Apologies 38.7% on sexual. Llama: Icannot 8.4% on sexual. Generational: "I can't help with that request" — 100% of Llama samples.

**3. Task switch to advice (all models on worker)**

The model becomes an advice-giving assistant. OLMo-7B: definitively 47.7%. OLMo-1B: Apologize 39.9%. Llama: assertively 50.4%. Generational: numbered lists ("1. Schedule a private meeting...").

**Cross-family comparison**

**Llama** maintains narrative elements even in assistant mode (shatter/stomp/scorch are still narrative words). Narrative sublimation.

**OLMo** produces distribution collapse — conventional continuation words drop to zero, replaced by meta-words (She, sure, He) and refusal markers. Genre collapse extends to template-mediated mode.

**Amber** is stable — template barely changes the distribution. kiss stays at 26–28% on sexual, scream stays at 23–28% on anger. Lightweight template intervention.

**OLMo-1B** overcompensates — strangle 31.6%, slay 17.9% on anger. Smaller model, more extreme substitutions in template-mediated mode.

**Template presence as methodological control**

Chat template survey across 29 checkpoints: 18 have templates, 11 do not.

**Without template (continue = raw, controls):** All base models (OLMo, Llama, Mistral, DeepSeek, SmolLM2), ALL Amber checkpoints (Amber, AmberChat, AmberSafe), Falcon, Pythia base.

**With template (task switch):** All Instruct/SFT/DPO variants of OLMo, Llama, Tulu, Pythia, Zephyr, DeepSeek, SmolLM2. Also Qwen base (unusual — has template despite being base).

Amber's "stability" in the four-column tables is not lightweight template intervention — it is no template at all. Same pipeline, same n=200, same everything — Amber shows no shift because `_apply_mode` falls back to bare prompt. This rules out beam count, mode parameter, or methodological artifacts as explanations for the OLMo/Llama shifts.

Notable: AmberChat and AmberSafe lack templates despite being chat/safety-tuned. Qwen base HAS a template despite being a base model. Template presence tracks tokenizer configuration, not training stage.

**Beam search artifacts**

OLMo-7B continue-aligned distributions at n=200 produce same-letter clusters (cascading/cerulean/courageously/categorically on power). This is a beam search prefix artifact — one BPE prefix token dominates and all beams inherit it. Filtered from tables. The finding does not depend on specific replacement words but on the absence of narrative words and presence of refusal/meta-words.

**Method**

- 38 models × ~120 prompts, beam_words n=200 depth=3 + logits + hybrid word_probs
- Comparison: raw-mode word_probs (n=1000 beam + exact logits) vs continue-mode word_probs (n=200 beam + continue logits)
- Beam counts not directly comparable for JS — comparison uses top-K word tables, argmax changes, refusal mass
- Generations: greedy + 3 samples at T=1.0 for qualitative confirmation

**Data**

- `data/continue_mode_tables.md` — four-column comparison tables (raw-base / raw-aligned / cont-base / cont-aligned)
- Continue-mode caches: beam_words/ (4,488), logits/ (4,171), word_probs/ (4,171)
