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
