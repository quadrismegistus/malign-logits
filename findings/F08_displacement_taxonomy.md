---
status: rescoped
grade: C
date: 2026-07-29
role: finding
description: "Displacement-type taxonomy (register / category / genre / archaic) over displacement_map pairs. Numbers recomputed 2026-07-26 (d0cd6a5, transcription error in power row caught and fixed); CONSTRUCT compromised per docket [399]/[401] — the pairs were never shown to be substitutions. Rescoped 2026-07-29."
instruments: [classification]
families: [olmo, llama]
data: [taxonomy_olmo.csv, taxonomy_llama.csv]
---
# F08: Automatic displacement taxonomy (OLMo + Llama, 18 prompts)

Classifies each displacement pair from the displacement maps into four types using contextual spaCy POS tags (word tagged in the context of its prompt) and wordfreq corpus frequencies:

- **Register shift** — same POS, high similarity. Same referent, different social register (*kill* → *hurt*, *yell* → *shout*, *warmth* → *heat*).
- **Category shift** — different POS, high similarity. Charge migrates across grammatical categories (*kill* → *harm* [V→N], *fuck* → *ride* [V→V→N], *surge* → *rush* [N→V]).
- **Genre change** — displaced onto a function or meta-linguistic token. Format changes rather than vocabulary substitution (*kill* → *WHAT*, *harm* → *WHAT*, converting statements into questions).
- **Archaic displacement** — target is a rare word (Zipf frequency < 3.0). Modern vocabulary displaced onto low-frequency, often archaic terms (*kill* → *smite*, *strangle* → *smother*, *stared* → *gazed*).

**CLI:** `malign taxonomy [--family olmo] [--all-prompts]`

**OLMo displacement profile (22,458 pairs):**

| Category | Register | Category | Genre | Archaic |
|---|---|---|---|---|
| violence (explicit) | **86%** | 6% | 0% | 8% |
| violence (liminal) | **65%** | 11% | **14%** | 10% |
| power | **84%** | 12% | 0% | 4% |
| substance | 50% | 19% | 4% | 27% |
| death | 48% | 29% | 0% | 23% |
| sexual (liminal) | 51% | 28% | 3% | 17% |
| sexual (explicit) | **74%** | 6% | 0% | 19% |
| neutral | 38% | 41% | **8%** | 13% |
| profanity | 10% | 30% | **49%** | 10% |

**Llama displacement profile (11,520 pairs):**

| Category | Register | Category | Genre | Archaic |
|---|---|---|---|---|
| violence (explicit) | 62% | 18% | 0% | 20% |
| violence (liminal) | **86%** | 10% | 4% | 0% |
| power | 83% | 17% | 0% | 0% |
| substance | 68% | 22% | 0% | 10% |
| death | 57% | 20% | 0% | 23% |
| sexual (liminal) | 74% | 20% | 0% | 6% |
| sexual (explicit) | **82%** | 6% | 0% | 12% |
| neutral | 46% | 36% | **5%** | 13% |
| profanity | 7% | 31% | **62%** | 0% |

**Cross-family findings:**

**Llama is more register-shift dominant than OLMo** (66% vs 49% of all pairs). Consistent with the logit lens finding: Llama's late-layer override performs surgical word substitution at the last moment; OLMo's distributed repression disrupts format more aggressively.

**Profanity triggers genre change regardless of architecture** — 49% (OLMo) and 62% (Llama). Models cannot find acceptable synonyms for swear words and resort to format disruption. This is the one displacement type that is model-independent.

**Explicit content is overwhelmingly register shift in both families.** Violence explicit: 86% (OLMo), 62% (Llama). Sexual explicit: 74% (OLMo), 82% (Llama). When transgressive content is overt, the superego finds same-POS synonyms. Genre change appears only on liminal and profane content — where synonym substitution would leave the transgressive implication intact.

**Death and substance produce the most archaic displacement.** *stared* → *gazed*, *tomb* → *gravestone*, *thought* → *pondered*, *swallowed* → *gulped*. Alignment pushes these categories toward literary and formal registers.

Results in `data/taxonomy_olmo.csv`.


---

**Provenance check, 2026-07-26.** Recomputed from `data/taxonomy_olmo.csv` (the file this finding cited as `displacement_taxonomy.csv`, renamed at `39a3886` on 2026-05-03). Every row reproduces within 1–3pp except **power**, which was published as 96/14/0/4 — a row summing to **114%** — against a recomputed 84/12/0/4. The 96% was a transcription error, corrected above; the impossible sum is what makes it certain rather than a judgement call. Row count is now 23,013 against the 22,458 published, a 2.5% increase from a later re-run of the taxonomy.

---

## Rescoping addendum (2026-07-29)

Two events, kept distinct because they check different things:

1. ARITHMETIC AUDIT, 2026-07-26 (d0cd6a5): the published percentages
   were reconciled against the renamed data file; every row
   reproduced within 1-3pp except the power row (published 96/14/0/4,
   summing to 114%) — a transcription error, corrected. This audit
   validated the numbers.
2. CONSTRUCT AUDIT, 2026-07-29 (docket [399]/[401]): the pairs the
   taxonomy classifies come from `displacement_map()`, which emits
   the cross-product of fallen-probability words x risen-probability
   words filtered at cosine >= 0.15 — not observed substitutions.
   Every taxonomy percentage (register_shift, category_shift,
   genre_change, archaic) is therefore a proportion of that
   cross-product. The arithmetic audit could not have caught this:
   reconciliation validates numbers, not constructs, and the one
   error it did catch was found by impossibility (a row summing to
   114%), not by scrutiny of the construct.

WHAT SURVIVES: the taxonomy scheme itself and the classification
code, which apply to any properly-derived pair set. WHAT IS NOT
QUOTABLE until re-analysis: every published percentage over
displacement pairs, in this file and wherever cited (F13 builds
directly on this taxonomy and is rescoped in the same pass).

REGISTERED RE-ANALYSIS: identical to F13's, docket [400].2 items
(a)-(d), extended to F08 at [402] — F13 cannot be rehabilitated on a
taxonomy that inherits the same pairing rule, so the two re-analyses
are one job. Assigned to lacan, audited by malign, per RH 2026-07-29.
