# Registration O — the same content in two languages

**STATUS, FROZEN 2026-08-04 UTC.** As of that date, verified at the seats named: **no `result_o_*.json` existed anywhere; no `tail_excess` sign, no `A_|valence|` or `A_arousal` value, no cluster combination and no bias-column value had been computed by anyone; no producer had been written.** The population was ratified the same day (RH, [4083]) and the competence threshold was ruled on a **blind** rank-ordering, names withheld until after the ruling ([4061]).

**WRITTEN IN THE PAST TENSE ON PURPOSE, AND A DATE IS NOT ENOUGH.** *"NOT FROZEN, NOT RUN"* carries a date and is still a PRESENT-TENSE claim: **it is true until the freeze commit and false forever after, and nothing rewrites it.** This campaign has met one status header that outlived its truth (D3b's "NO D3b QUANTITY HAS BEEN COMPUTED", still in place after both stages had run) and corrected N's for exactly this reason. **A dated statement of what was ABSENT on a named date stays testimony; a dated statement of what IS expires.**

    OCCASIONED   RH, 2026-08-04: "Movement, valence, arousal on any pair of
                 translated english and chinese prompts we have. Hypotheses:
                 both substitute, both decrease |valence|, both decrease
                 arousal."
    SCOPE        N is English-only by RH's word; O is where the crosslingual
                 question goes, with hypotheses designed for it rather than
                 a companion row carrying machinery and no prediction.

---

## §O0 THE PREDICATES THIS REGISTRATION RESTS ON — stated, because 21 of 22 registrations do not

**Booked campaign-wide the night O was drafted: twenty-one registrations rest on a faller rule none of them states, and the one exception states it in a change table.** O states its own.

    FALLER      w is a faller iff  P[w] >= 0.003  AND  Q[w] < 0.5 * P[w]
                (`CANONICAL.min_prob`, `CANONICAL.fall_ratio`)
    CANDIDACY   the residual bucket is EXCLUDED FROM FALLER CANDIDACY, not
                stripped afterwards -- `movement.py` @ commit `e7864dab`
    RISER       k is a riser iff  k not in fallers  AND
                max(P[k], Q[k]) > 0.003  AND  Q[k] - P[k] > 0.003  AND
                Q[k] > P[k] * ratio,  where ratio = (1 - sum_fallers Q) /
                sum_survivors P
    THETA       0.001, the `true_word_probs` retention floor

**THE PIN CARRIES THE COMMIT, NOT THE CONSTANTS ALONE.** Both sides of the 2026-08-04 repair satisfy `min_prob 0.003, fall_ratio 0.5` and select different faller sets in 11% of cells. **An extension can move while a rule's text stands still; what is pinned is `e7864dab`.**

---

## §O1 POPULATION

    TRANSLATED PAIRS   **301**, enumerated at
                       `populations/population_o_pairs.json`,
                       pair-set sha16 **7ad8a39d1ac85d48**;
                       sources `chinese_translations.json` @ 60da9c229da7c523
                       and `chinese_translations_2.json` @ 231bd2a873fa39b4
    RULE               rows carrying both `english` and `chinese`; BOTH sides
                       must be declared stimuli; deduplicated on the pair
    EDGES              **9**, on **9 distinct base checkpoints**
    CELLS              301 x 9 x 2 arms = **5,418**, 2,709 per arm

**THE PAIR IS THE INSTRUMENT.** Comparing English and Chinese corpora would confound language with content; **a translated pair holds content fixed and varies only the language.**

**A COUNT OF `translation` LINKS IS NOT A PAIR COUNT.** An earlier draft claimed 373 — links leaving English prompts, never requiring the target to be a declared stimulus. **The enumeration file exists so no later reader re-derives a different number from the same words.**

### §O1.1 TWO FILTERS, AND THEY MEASURE DIFFERENT THINGS

    44 edges -> **10**   `cjk_tier` FLUENT on BOTH sides -- **CAPACITY**,
                         a count of CJK characters in the tokenizer
                         vocabulary (NOMINAL 77-715, PARTIAL 1,077-2,322,
                         MARGINAL 2,829-3,429, FLUENT 3,652-21,006)
    10 edges ->  **9**   CJK-only retained-mass share **>= 0.30** --
                         **COMPETENCE**, what the model does with Chinese

**`cjk_tier` IS NECESSARY AND NOT SUFFICIENT.** It is satisfiable at a 6.7% CJK mass share: `bigscience/bloomz-7b1` is FLUENT-tier and puts 93% of its retained Chinese mass elsewhere. **A filter on capacity alone would have admitted it.**

**THRESHOLD RULED BLIND**, on rank-ordered shares with names withheld: the distribution has ONE void, **0.0674 to 0.5365**, seven times the spread of the qualifying nineteen. **Any cut in [0.10, 0.50] separates the same nineteen from the same one; 0.30 is the midpoint and was not tuned.** Names were revealed after the ruling.

**NOT ONE EDGE IS MIXED-TIER** — every alignment step keeps its base's Chinese competence, so tier is a property of the lineage. **The one exception is the competence filter's own case, and it is recorded as an observation, not a claim** (`bloom-7b1` 0.5625 -> `bloomz-7b1` < 0.0674, one alignment step; descriptive, n=1).

### §O1.2 WHAT "YIELDS AN `A`" MEANS — SEVEN CLAUSES, BECAUSE SIX WERE NOT ENOUGH

**Two seats holding one rule produced numbers 22 points apart, then 1.7 points apart, because the rule named neither its join nor its denominator.** All seven bind:

    1 SOURCE       fallers and risers from `N.cell_roles(c, 'CANONICAL')`
    2 DIMENSIONS   a word must be found for **BOTH** valence AND arousal --
                   H2 and H3 each need both, so a word scoring one cannot
                   serve either hypothesis
    3 TABLES       looked up in the tables of **THE ARM'S OWN LANGUAGE**
                   (`m01_registration_c3.py:103` hardcodes `("en", ...)`,
                   which is why Registration C cut all its Chinese cells)
    4 JOIN         **`N.lookup`'s LEMMA CANDIDATES**, not dict membership.
                   English inflects and Chinese does not: a surface join
                   loses ~18% of English moving mass (`screamed` misses,
                   `scream` hits) and almost no Chinese
    5 FUNCTION     function words excluded, `N.is_function_word(k, arm)`
    6 ROLES        **>= 3 fallers AND >= 3 risers** among surviving words
                   (`QUALIFYING_MIN = 3`) -- both roles, not 3 words total
    7 DENOMINATOR  **ANALYSED cells = cells MINUS zero-faller cells.** A
                   yield describing "of the cells O analyses, how many carry
                   an `A`" must not count the cells the design drops

**AND THE JOIN'S LIMIT, STATED: no second derivation of these yields can be independent of `N.lookup`.** Reimplementing lemma resolution is the defect clause 4 exists to prevent. **Independence is available in the population handling, edge selection, iteration and accumulation — not in the join.**

### §O1.3 A-YIELD, MEASURED ON THIS POPULATION UNDER THIS RULE

    arm   analysed   A-cells   yield     zero-faller
    en      2,642     1,844   **69.8%**    67 (2.47%)
    zh      2,604     1,298   **49.8%**   105 (3.88%)

**DISCLOSURE, ON THE POPULATION PIN:** §O1's `population_o_pairs.json` whole-file hash was taken BEFORE ratification and the `_status` flip moved it. **The CONTENT pin did not move: `pair_set_sha256_16 = 7ad8a39d1ac85d48` is the canonicalised 301 pairs themselves, unchanged by the status field and three-way verified.** A whole-file hash covers metadata; **the content pin is what a reader should check, and it is the one that has not moved.**

**Two seats, two code paths, agreeing on 1,844 / 1,298 / 67 / 105 to the cell.** The rates differed until clause 7 was stated. **No cluster contributes zero to either arm**, so all nine enter both Stouffers and H2/H3 need no power caveat.

## §O2 NORMS — what exists in each language, and what therefore cannot be tested

    en   valence, arousal, dominance, concreteness, logfreq
    zh   valence, arousal, concreteness, familiarity, imageability, logfreq

    **zh HAS NO DOMINANCE TABLE.** A_|dominance| is impossible in Chinese
    and appears in no hypothesis here.

**RH's three hypotheses use movement, valence and arousal only, which is exactly the set both languages support.** Concreteness and logfreq exist in both and are NOT tested — no prediction was registered for them and a tested-because-available dimension is the printed-never-tested shape.

---

## §O3 THE HYPOTHESES

**All three are stated as holding IN BOTH ARMS. The claim is that the mechanism is not a property of English.**

    H1  SUBSTITUTION.  `tail_excess` < 0 in the en arm AND in the zh arm.
        ONE-SIDED each.  Direction from N's confirmed English result.

    H2  |VALENCE| DECREASES.  A_|valence| > 0 in both arms, where
        A = wmean(fallers) - wmean(risers), weights |delta|, over
        |z_valence|.  ONE-SIDED each.

    H3  AROUSAL DECREASES.  A_arousal > 0 in both arms, same form, over
        signed z_arousal.  ONE-SIDED each.

**COMBINATION, per arm:** the statistic per cell, aggregated by signed z within each family, unweighted mean of a cluster's families, Stouffer over **O's NINE base checkpoints**, equal weight per cluster. **The FORM is identical to N's — same statistic, same nesting, same equal weighting — so the arms are comparable to each other and the machinery is the one N used. The N is not: N combines 34 clusters, O combines 9, and O's instrument is correspondingly weaker.** An earlier draft said 34, which was N's number surviving a scope change.

**REPORTED, NEVER TESTED: the WITHIN-PAIR agreement rate** — the share of (pair, edge) cells where both arms carry the same sign. It is the most interesting number here and it has no registered prediction, so it is a description.

---

## §O4 THE READING RULE, FIXED BEFORE ANY NUMBER

    BOTH arms confirm     the mechanism is not a property of English
    ONE confirms /        **NOT SUPPORTED**, reported as an ASYMMETRY with
      other NULL          the confirming arm named
    ONE confirms /        **NOT SUPPORTED, AND REPORTED AS A REVERSAL** --
      other SIGNIFICANT     never as an asymmetry.  See below.
      IN THE OPPOSITE
      DIRECTION  (OPPOSED)
    NEITHER confirms      NOT SUPPORTED

**A ONE-SIDED TEST HAS THREE OUTCOMES PER ARM, NOT TWO: confirms, null, or significant in the opposite direction at the same alpha.** An earlier draft of this rule collapsed the last two into "does not confirm", and they are not the same finding:

    en confirms / zh NULL         "we could not detect it in Chinese"
    en confirms / zh OPPOSED      **the mechanism RUNS BACKWARDS in Chinese**

**The second is not a failure to confirm. It is a different result, and it is the most interesting outcome this design can produce** — and "asymmetry" is a word for a difference in magnitude, which would understate it into invisibility. **OPPOSED is reported as a REVERSAL, with both arms' statistics and both bias columns beside it.**

**A hypothesis stated as "in both arms" is not confirmed by one arm.** Reporting a single-arm result as a finding is the failure this rule exists to prevent, and the temptation will be strongest if English confirms and Chinese does not.

**NO ARM IS PRIMARY.** en and zh carry the same machinery and the same weight; the pairing is what makes that defensible.

---

## §O5 THE KNOWN INSTRUMENT ASYMMETRY, AND WHY IT IS NOT A CONFOUND HERE

    truncation leak INCIDENCE   en 19.4%  zh 17.8%   indistinguishable
    induced FLIP RATE           **4.23x apart, zh SMALLER**
    (`data/leak_flip_rate_by_stratum.json` @ `03cf7e34`)

**THESE ARE THE 44-EDGE FIGURES.** They are the measurement that justified N's English-only scope and O's separate existence, **and they are NOT O's nine-edge numbers.** The per-arm bias columns below are computed on O's own population; **the 4.23x is cited as the design's occasion, never as O's instrument asymmetry.**

**This asymmetry is why N is English-only.** Here it is the object rather than the obstacle: **the arms are being compared, so a difference in the instrument's behaviour between them is part of what O measures and must be reported beside every arm-level result.**

**Per-arm worst-case bias columns are computed and reported per cell**, as N's are, so a reader can see whether an arm difference survives the instrument difference.

---

## §O6 WHAT THIS CANNOT DO

- **It cannot separate LANGUAGE from TOKENIZATION.** Chinese rides different token trees and theta-truncation interacts with vocabulary granularity; an arm difference is a difference between two (language, tokenizer) pairs, not between two languages.
- **It cannot test dominance**, and any account of the valence result must not silently import dominance intuitions from the English-only work.
- **It cannot claim the translations are equivalent.** The 301 pairings are RATIFIED as the official record on RH's word (docket `[4083]`, 2026-08-04: *"Ratify as is, we worked hard on it earlier"*). **The warrant is the translation pass's own drafting process, in which RH participated, and that is the whole of it: no post-hoc fidelity audit was run and none is claimed.** Ratification is an authority act, not a measurement — **a systematic translation drift would still appear as an arm difference, and O carries no instrument that would tell the two apart.**
- **301 pairs is not 379 zh stimuli.** Chinese prompts without an English partner are out of scope here as they are in N.
- **AND THE LIMIT THAT BOUNDS EVERY CROSSLINGUAL SENTENCE O CAN PRODUCE: the nine surviving clusters are ALL Chinese-origin or Chinese-heavy.** The one non-Chinese-origin lineage was `bloom`, removed by the competence filter. **So "the mechanism is not a property of English" CANNOT BE DISTINGUISHED FROM "the mechanism is a property of Chinese-trained models."**

  **The competence exclusion removes one confound and creates another**, and ten-of-ten is a different sentence from nine-of-ten. **A confirmation here licenses "the mechanism appears in Chinese-trained models on Chinese text" and does not license the general claim** — which is the sentence a reader will reach for and the one O must not supply.
