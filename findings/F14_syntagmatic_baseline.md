# F14: Syntagmatic baseline: alignment-produced vs corpus-level damage (OLMo 3 7B, 23k pairs)

Finding 13 showed that paradigmatic and syntagmatic axes trade off within aligned-model displacements. But is the syntagmatic disruption alignment-produced, or does the base model exhibit it too? We compute `syntagmatic_js` for the same 23,013 displacement pairs under both the base model and the aligned (DPO) model: `p(next | prompt + source)` vs `p(next | prompt + target)` under each.

**The aligned model's continuations are more disrupted than the base model's in every content category.**

| category | base synt_js | aligned synt_js | delta | interpretation |
|---|---|---|---|---|
| sexual_explicit | 0.367 | 0.473 | **+0.106** | alignment-produced damage |
| violence_explicit | 0.163 | 0.237 | +0.074 | alignment adds to already-low disruption |
| sexual_liminal | 0.481 | 0.544 | +0.063 | moderate alignment amplification |
| death | 0.434 | 0.494 | +0.060 | moderate alignment amplification |
| substance | 0.402 | 0.460 | +0.059 | moderate alignment amplification |
| violence_liminal | 0.387 | 0.446 | +0.059 | moderate alignment amplification |
| power | 0.449 | 0.506 | +0.057 | moderate alignment amplification |
| neutral | 0.415 | 0.458 | +0.044 | background syntagmatic damage |
| profanity | 0.560 | 0.592 | +0.032 | alignment-inherited (ceiling effect) |

**Three structurally distinct displacement regimes emerge:**

**Alignment-produced damage (sexual_explicit, delta +0.106).** The base model substitutes fluently — its syntagmatic_js is moderate (0.367), meaning the base model's next-token chain holds even after swapping source for target. Alignment specifically breaks this fluency. The aligned model cannot smoothly continue after its own sexual substitutions. This is the clearest case of alignment-induced Jakobsonian similarity disorder: a model that *had* paradigmatic capacity and lost it through training.

**Alignment-inherited damage (profanity, delta +0.032).** The base model already produces high syntagmatic disruption (0.560) — profanity has no clean synonyms at the corpus level. Alignment adds little because the chain was already broken. The similarity-disorder profile in profanity is a property of the language (or training corpus), not an alignment artefact.

**Alignment-unnecessary (violence_explicit, delta +0.074).** Both models substitute fluently. The base model finds clean synonyms (*kill* → *hurt*) and the chain holds (0.163). Alignment adds moderate disruption but the absolute level remains the lowest of any category. Violence has rich paradigmatic resources that survive alignment largely intact.

**Neutral delta (+0.044) rules out the noise interpretation.** If `syntagmatic_js` were merely capturing distributional variability (Possibility A from the design), base and aligned models would produce similar values on neutral prompts. They don't — alignment produces measurable background syntagmatic damage even on safe content. The metric is detecting real alignment-induced disruption, not measurement noise.

**Corrects the F13 Jakobsonian framing.** F13 identified profanity as the strongest similarity-disorder case. The baseline check shows profanity's high absolute syntagmatic_js is partly *corpus-inherited*, not alignment-produced. The strongest case for alignment-as-similarity-disorder is **sexual_explicit**: the base model had the paradigmatic capacity for fluent substitution, and alignment selectively destroyed it. The paper's Jakobsonian claim is strongest when run through sexual content, not profanity.

**Content-graded delta confirms alignment-specificity.** The delta itself scales with content sensitivity (sexual > violence > neutral > profanity), ruling out uniform distributional shift. Alignment produces more syntagmatic damage where it intervenes more, with profanity as the exception (ceiling effect). This content-grading is consistent with Finding 6 (transgressive token mass displacement separates categories) and with Finding 1 (SFT/DPO divide labour by content type).

Results in `data/taxonomy_olmo.csv` (column `syntagmatic_js_aligned`). CLI: `malign taxonomy --baseline --family olmo`.
