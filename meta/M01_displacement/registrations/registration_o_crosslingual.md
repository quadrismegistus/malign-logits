# Registration O — the same content in two languages

**STATUS: DRAFT, 2026-08-04 UTC. NOT FROZEN, NOT RUN.** As of this date no `result_o_*.json` exists at any seat and no O quantity has been computed by anyone.

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

    TRANSLATED PAIRS    373   English prompts carrying a `translation` link
                              to a Chinese prompt (373 of 379 zh stimuli)
    EDGES               44    `operation_edges`, 34 distinct base checkpoints
    CELLS               373 x 44 x 2 arms = **32,824**, of which 16,412 en
                        and 16,412 zh

**THE PAIR IS THE INSTRUMENT.** Comparing English and Chinese corpora would confound language with content; **a translated pair holds content fixed and varies only the language**, which is the only design that isolates what RH is asking about.

**A-YIELD, MEASURED ON THIS POPULATION** (the freeze precondition, discharged 2026-08-04):

    arm   cells    A-yield          absent  err
    en    16,412   **6,386 = 38.91%**    0    0
    zh    16,412   **4,843 = 29.51%**    0    0

**Both arms are viable; no MDE caveat and no descriptive downgrade is needed for H2/H3.**

**Scored against EACH ARM'S OWN LANGUAGE TABLES** — `('zh','valence')` and `('zh','arousal')`, 24,911 entries each. **This is stated because `m01_registration_c3.py:103` hardcodes `norms[("en", d, "primary")]`, which is why Registration C cut all 16,676 of its Chinese cells; a zh yield computed against English tables would return near zero and read as a fact about Chinese rather than a bug in the lookup.**

**And the campaign's 49.7%/28.2% were another population at another dimension count.** The en figure here is 38.91%, not 49.7%, because that one required three dimensions; **the arm gap is 9.4 points rather than the 21 the old figures implied.** Neither stood in for this one, which is why the measurement was a precondition.

---

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

**COMBINATION, per arm:** the statistic per cell, aggregated by signed z within each family, unweighted mean of a cluster's families, Stouffer over the 34 base checkpoints, equal weight per cluster. **Identical to N's, so the arms are comparable to N and to each other.**

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

**This asymmetry is why N is English-only.** Here it is the object rather than the obstacle: **the arms are being compared, so a difference in the instrument's behaviour between them is part of what O measures and must be reported beside every arm-level result.**

**Per-arm worst-case bias columns are computed and reported per cell**, as N's are, so a reader can see whether an arm difference survives the instrument difference.

---

## §O6 WHAT THIS CANNOT DO

- **It cannot separate LANGUAGE from TOKENIZATION.** Chinese rides different token trees and theta-truncation interacts with vocabulary granularity; an arm difference is a difference between two (language, tokenizer) pairs, not between two languages.
- **It cannot test dominance**, and any account of the valence result must not silently import dominance intuitions from the English-only work.
- **It cannot claim the translations are equivalent.** They are the corpus's declared translation links; **translation quality is assumed, not measured, and a systematic translation drift would appear as an arm difference.**
- **373 pairs is not 379 zh stimuli.** Six Chinese prompts carry no English partner and are out of scope here as well as in N.
