# F13: Jakobsonian axes: paradigmatic vs syntagmatic displacement (6 families, 126k pairs)

Roman Jakobson's 1956 *Two Aspects of Language and Two Types of Aphasic Disturbances* argues that language is constituted by two complementary axes — selection/similarity (paradigmatic) and combination/contiguity (syntagmatic) — and that damage to one axis forces compensatory reliance on the other. We test whether the same structural trade-off operates in alignment-induced displacement.

**Method.** For each (source → target) displacement pair from `Psyche.analyze().displacement_map()`, we compute two scores: `similarity` (cosine similarity between contextual embeddings — paradigmatic axis) and `syntagmatic_js` (JS divergence between `p(next_token | prompt + source)` and `p(next_token | prompt + target)` under the base model — syntagmatic axis). High similarity = good synonym found; high syntagmatic_js = the substitute disrupts the next-token chain.

**The two axes are negatively correlated across all 6 families.**

| Family | Pearson r | n pairs | Within-category r |
|---|---|---|---|
| **Llama** | **−0.533** | 22,341 | [−0.68, −0.44] |
| **Zephyr** | **−0.498** | 21,887 | [−0.61, −0.27] |
| **Tulu** | **−0.495** | 21,015 | [−0.70, −0.44] |
| OLMo | −0.407 | 23,013 | [−0.56, −0.19] |
| Qwen | −0.366 | 12,414 | [−0.49, −0.30] |
| OLMo-tiny | −0.338 | 25,087 | [−0.54, −0.20] |

Total: 125,836 displacement pairs. The correlation holds within every content category in every family. When a displacement pair finds a paradigmatically close substitute, the syntagmatic chain is preserved. When it can't, the chain breaks. This is not a property of any single architecture or alignment procedure — it is a structural property of how aligned LLMs handle foreclosure.

**Llama-Tulu: same base model, different alignment, different trade-off strength.** Both share `meta-llama/Llama-3.1-8B` as their base. Llama uses Meta's alignment (opaque, presumably includes extensive safety data). Tulu uses Allen AI's alignment (transparent, no safety data in SFT, safety data only in DPO). Their correlation strengths differ (−0.533 vs −0.495) and their displacement profiles diverge: Llama death category has 28% genre_change; Tulu death has 12%. Same paradigmatic capacities at the start, different structural-symbolic signatures under different corporate alignment regimes. This is direct evidence that alignment is a *corporate-political* operation: the architecture is held constant, the variable is alignment practice, the output is differential foreclosure.

**Violence_explicit is universally paradigmatically fluent.** Register_shift dominates across all 6 families (73–86%). The corpus contains rich paradigmatic resources for violence (kill/hurt/attack/destroy/fight/strike/punch...), so alignment can substitute without breaking the chain. This is the *unimpaired* case in Jakobson's typology.

**Profanity genre_change varies by alignment regime.** Genre_change on profanity ranges from 27% (Zephyr, no safety data) to 58% (OLMo-tiny, full alignment). The rate scales with how aggressively the family targets profanity. Zephyr's instruction-following alone doesn't break the chain on profanity; targeted safety training does. The within-content variation across families is itself evidence of differential corporate alignment practice.

**Content categories sort along the trade-off (OLMo-tiny, representative):**

| category | paradigmatic similarity | syntagmatic JS | n pairs |
|---|---|---|---|
| **violence_explicit** | **0.633** | **0.151** | 621 |
| sexual_explicit | 0.557 | 0.345 | 5,169 |
| violence_liminal | 0.605 | 0.397 | 3,276 |
| death | 0.538 | 0.424 | 3,852 |
| power | 0.571 | 0.450 | 1,791 |
| sexual_liminal | 0.599 | 0.470 | 3,099 |
| substance | 0.530 | 0.475 | 5,130 |
| neutral | 0.484 | 0.503 | 1,651 |
| **profanity** | 0.563 | **0.606** | 498 |

**What the negative correlation means.** When alignment replaces a foreclosed word, it has two options that trade off: (a) find a similar word — "kill" → "hurt," the sentence flows naturally; or (b) break the sentence — "fuck" → "Options," the model abandons narrative for a different genre. When a good synonym exists (violence), option (a) succeeds and the chain holds. When no synonym exists (profanity), option (b) takes over and the chain breaks. Whether alignment can perform clean substitution or must resort to genre collapse depends on the *paradigmatic resources available in the content domain* — a structural-linguistic constraint, not just a corporate-alignment choice.

**Refines existing taxonomy.** The continuous syntagmatic_js metric makes the categorical displacement taxonomy (Finding 8) into a quantitative dissociation: paradigmatic types (register_shift, archaic) cluster at synt_js ≈ 0.37; syntagmatic types (category_shift, genre_change) cluster at synt_js ≈ 0.58–0.63.

**Caveats.** Single-position syntagmatic measure (next-token only); multi-position surprisal would be a sharper test. ~~Preliminary on OLMo-tiny only~~ — now replicated across 6 families. ~~Neutral category at boundary~~ — resolved in Finding 14.

CLI: `malign taxonomy --family olmo-tiny`, `malign taxonomy --analyze` (cross-family). Results in `data/taxonomy_*.csv`, `data/taxonomy_summary.csv`.
