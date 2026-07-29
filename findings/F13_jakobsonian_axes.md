---
status: rescoped
grade: C
date: 2026-07-29
role: finding
description: "Jakobsonian decomposition — negative paradigmatic/syntagmatic correlation, direction plausible across 6 families; QUANTITIES NOT QUOTABLE pending registered re-analysis (docket [399]/[400]). Was verified/A at authoring; never audited until 2026-07-29."
instruments: [embedding]
families: [llama, zephyr, tulu, olmo, qwen, olmo-tiny]
chapters: [ch05]
data: [taxonomy_summary.csv]
---
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

---

## Rescoping addendum (2026-07-29)

This finding was marked verified/A at authoring (2026-05-17) and
first audited 2026-07-29 (docket [399], lacan; independently
triggered by RH's verification query the same morning). The audit
found that the instrument does not measure what the finding claims:

1. `displacement_map()` emits the full cross-product of every word
   whose probability fell against every word whose probability rose,
   filtered at cosine >= 0.15. A "displacement pair" is not an
   observed substitution; the causal reading is supplied by the
   function's name.
2. The per-category similarity means are dead twice over: (i)
   selected on their own outcome — pairs enter only above the 0.15
   similarity floor, so the reported means are truncated means and
   cannot evidence that substitutes are paradigmatically close; and
   (ii) averaged over a quantity that is not stable in its own
   defining dimension — similarity varies by up to 0.50 across the
   three conventionally-chosen layers (25/50/75% depth) for
   essentially every pair, while syntagmatic_js is exactly
   layer-invariant (verified on all 40,228 distinct pairs, docket
   [409]/[410]).
3. n = 125,836 is a layer-triplicated cross-product (pairs appended
   once per layer, up to three times; pairs share members massively
   within prompts). The Pearson r is computed without clustering;
   the independent unit is nearer the prompt or family than the
   pair.
4. The two axes are measured in different models: similarity from
   the aligned model's hidden states, syntagmatic_js under the base
   model.
5. (Declared during re-analysis, [420]/[425]: the undeclared
   MIN_PAIRS floor and the axis-less dedup key — both measured,
   neither verdict-changing; listed here so the count is complete.)
6. FIRST-TOKEN IDENTIFICATION ([449], found by RH's direct question
   "raw logits or reconstructed word probabilities?"): word
   probabilities are FIRST-TOKEN approximations (core.py:144-170) —
   every word sharing a first token carries an identical number in
   every row, at every layer, in both arms. In qwen, `casket` and
   `crotch` — a death word and a sexual word, assigned to different
   content categories in this finding's own table — are numerically
   indistinguishable. Multi-token share of the source/target
   vocabularies, and the share of PAIRS with a collided word on
   either side (`scripts/f13_token_collision_audit.py`):

       family      multi-token   in collision   % of rows hit
       olmo             15.3%           5.5%           14.5%
       olmo-tiny        16.3%           7.2%           10.4%
       llama            11.0%           4.6%            8.5%
       qwen             10.3%           4.9%           12.0%
       zephyr           27.9%          19.3%           30.7%
       tulu             10.8%           3.7%           10.2%
       amber            43.0%          36.3%           51.0%

   The row-level column is the finding's actual exposure and is
   larger than the vocabulary share everywhere. Amber, the corpus
   used as this finding's out-of-sample test, has HALF its pairs
   affected; zephyr is the second-worst and was absent from the
   first tabulation. The collisions are not marginal words: amber
   collapses `fuck/fucked/fucking/fuckstick` with `fathom/flare/fry`
   into one number, and `pussy` with `punch` and `pondered` —
   cross-category collapse in precisely the sexual-versus-violent
   distinction this project's central claim rests on.
   Consequence: the CATEGORY ASSIGNMENT of a probability change is
   not identified wherever a first token is shared, and a "faller"
   may be a token falling rather than the word credited.

WHAT SURVIVES: the qualitative direction (negative correlation, six
families, consistent sign, holds within every category; the
similarity truncation attenuates rather than manufactures a negative
correlation). WHAT IS NOT QUOTABLE until re-analysis: the r values,
the n, the per-category means, and any "strongest quantitative
result" framing.

REGISTERED RE-ANALYSIS (docket [400].2 as amended [408]/[411]/[412],
frozen; assigned to lacan, audited by malign [410], per RH
2026-07-29): (a) de-duplicate to distinct word pairs (syntagmatic_js
verified exactly layer-invariant; collapse rule for similarity =
malign's declared call with a robustness alternative); (b) cluster
to the prompt as unit, sign tests WITHIN family, never pooled across
unequal prompt counts; (c) the similarity filter is a selection
criterion; its mean is never reported as a result; (d) mixed-model
caveat carried on every number; (A1) correlations computed
separately by axis (repression/sublimation) with NO pooled-across-
axis correlation at all — llama/qwen are 100% repression and
olmo/olmo-tiny ~89% sublimation, so the published between-family
spread may be an axis artifact; (A2) direction is the test,
magnitude pre-declared undetermined; (A3) reading table: cell =
family x axis entering at >=10 computable prompts; direction
survives only if every entering cell has median within-prompt r < 0;
any positive-median cell = MIXED, named, no survival claim. The
published r values, n, and per-category means are retired whatever
the re-analysis shows.

Downstream effects at rescoping time: F08 (the taxonomy this finding
quantifies) rescoped in the same pass; paper v3 §IV's two citations
flagged to RH; CLAUDE.md's "strongest single quantitative result"
line flagged. See also F14 (corpus-inheritance correction) and F36
euphemism-vs-proximity (alignment neutral at cos > 0.5), which
independently constrain the near-neighbour-substitution reading.

## Re-analysis outcome (2026-07-29, appended after the [420]/[427] audits)

THE DIRECTION SURVIVES, restated. Under the frozen spec (per-layer
primary, axis-separated, prompt unit, within-family sign tests):
30 of 30 family x axis x layer cells have median within-prompt
r < 0 (sign-consistency p=1.9e-09; every within-cell sign test
p<=0.0213; median r -0.355 to -0.701; docket [419], audited [420]).
De-duplication strengthened the correlation relative to the
published layered values (repression -0.461 -> -0.547; sublimation
-0.404 -> -0.460): the original was attenuated by its own layer
triplication. Declared parameters: MIN_PAIRS=10 (undeclared in the
spec, found by audit, necessary — the unfiltered minimum is three
pairs per correlation; kept-vs-all table published;
olmo/repression is the one cell that moves materially, -0.599 kept
vs -0.462 all); dedup key includes axis ([425]; the published
primary never used the collapsed frame and is unaffected, verified
to the third decimal at all thirty cells).

REGISTERED CHECK OUTSTANDING ([451].3): the 30/30 primary and
amber's P1 were computed on pairs whose word labels inherit defect
#6 — similarity was computed for the labeled word, probability for
its first token, so collided pairs carry mismatched coordinates.
Before the 30/30 is quoted anywhere else, the per-layer primary
re-runs RESTRICTED TO SINGLE-TOKEN WORDS in each family's tokenizer
(exact probabilities, no collisions; retention ~85-90% for five
families, 57% for amber, published per cell). Registered prediction:
the direction survives restriction. If it does not, that is the
finding.

PARTIAL ANSWER ALREADY IN HAND, for amber only and on the weaker of
the two restrictions: dropping every pair with a collided word on
either side (51% of amber's rows) leaves the collinearity diagnosis
below intact — per-layer sd 0.030/0.048/0.096 against 0.031/0.050/
0.100 unrestricted, per-layer median 0.868→0.522 against 0.879→0.542,
and the depth ordering unchanged in both axes. So defect #6 does NOT
explain amber's anomalous similarity profile, and the P2 strike and
its collinearity reason stand as written. This is the weaker
restriction — it removes words CONFUSED with another word but keeps
multi-token words whose probability is approximate without being
ambiguous — so it does not discharge the registered check above.

OUT-OF-SAMPLE RESULT (amber, 46,551 rows, never in F13; predictions
registered [421] BEFORE the run; corrected [425], audited [427];
P2's status revised [432]/[433], narrowed [436]/[438]): P1
(DIRECTION) CONFIRMED, 6/6 entering rows negative — the within-cell
trade-off holds on fresh data. P2 (DEPTH) is STRUCK as a
confirmation, for the narrow reason: within amber, LAYER, SPREAD,
AND REGION ARE COLLINEAR (per-layer sd 0.031/0.050/0.100; per-layer
median 0.879/0.783/0.542 while the six families sit flat on both),
so any two can be matched only by breaking composition on the
third. Amber's residual depth signal after spread-matching is real
but small (12-23% of apparent size, monotone, 2/2 axes) and CANNOT
BE ASSIGNED to depth, spread, or region. "L3 strongest" raw on
amber is what its variance profile predicts with or without a depth
effect. [423] is superseded by [425]; cite only the latter.

DEPTH GRADIENT, status: an IN-SAMPLE result with both known
confounds treated and dispatched ([436].A.2, measured): robust to
spread-matching (L3 strongest in 10/10 matched cells; matched span
LARGER than raw in 6/10) and region is flat across depth in-sample
(per-layer medians trendless). UNCONFIRMED out-of-sample — the
stated reason is that NO CORPUS HAS YET BEEN FOUND THAT CAN TEST IT,
not that a confound is untreated. FINDING NUMBER HELD. Precondition
for any candidate corpus ([433].2 + [436].A.5): publish per-layer
sd(similarity) AND per-layer median profiles first; enter only if
both are comparable to in-sample at L1. Two treatments were
proposed and closed by their own composition stages: P3 (floor
equalization — a no-op, 0.1-1.3% dropped) and P3' (matched band —
no shared band exists at L1). Re-derivation of the variance and
region tables: `scripts/f13_variance_audit.py`.

ANISOTROPY CAVEAT, project-wide ([430].6 as upgraded [432].7):
cross-family comparisons of raw cosine are comparisons across
different scales, and the scales VARY BY DEPTH within a family
(amber near-collinear at 25% depth, sd 0.031, normal by 75%, sd
0.100). Any similarity threshold fixed across families and layers
is a different selection at each. This is the third independent
reason this finding's original per-category similarity means were
never quotable.

The construct sentence heads all of it: these are facts about
faller-riser pairs selected by a similarity floor that differs by
corpus and by layer — not about observed substitutions.

Downstream effects at rescoping time: F08 (the taxonomy this finding
quantifies) rescoped in the same pass; paper v3 §IV's two citations
flagged to RH; CLAUDE.md's "strongest single quantitative result"
line flagged. See also F14 (corpus-inheritance correction) and F36
euphemism-vs-proximity (alignment neutral at cos > 0.5), which
independently constrain the near-neighbour-substitution reading.
