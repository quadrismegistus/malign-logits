# Registration Q — general and site on one scale

**STATUS AS DRAFTED (2026-08-04 UTC): DRAFT, not frozen, not in force.** As of this date, verified at the seats named: no `result_q_*.json` existed anywhere; no `tail_excess` or `A` value had been computed on any partition of N's population by anyone; no producer had been written. **The A-yield pass named in §Q1.3 HAS run** and its counts are deposited — it computes no hypothesis quantity and is the precondition for the MDEs below.

**Written in the past tense on purpose.** A status line reading "not run" is true until the producer runs and false forever after, and nothing rewrites it. This campaign has met two status headers that outlived their truth.

    OCCASIONED   RH, 2026-08-04, on reading the clause table: the campaign has
                 no BRIDGE between "in general" and "at transgressive sites" —
                 C and D2 share machinery and no common scale, so no sentence
                 of the form "the effect is N times stronger at sites" is
                 available. Q puts both on one instrument, one population, one
                 scale.
    LETTER       P is taken by the relation/annotation registration ([4157]).

---

## §Q0 THE PREDICATES THIS REGISTRATION RESTS ON

    FALLER      w is a faller iff  P[w] >= 0.003  AND  Q[w] < 0.5 * P[w]
                (`CANONICAL.min_prob`, `CANONICAL.fall_ratio`)
    CANDIDACY   the residual bucket is EXCLUDED FROM FALLER CANDIDACY, not
                stripped afterwards — `movement.py` @ commit `e7864dab`
    RISER       k is a riser iff  k not in fallers AND max(P[k],Q[k]) > 0.003
                AND Q[k] - P[k] > 0.003 AND Q[k] > P[k] * ratio, where
                ratio = (1 - sum_fallers Q) / sum_survivors P
    THETA       0.001, the `true_word_probs` retention floor

**THE PIN CARRIES THE COMMIT, NOT THE CONSTANTS ALONE.** Both sides of the 2026-08-04 repair satisfy `min_prob 0.003, fall_ratio 0.5` and select different faller sets in 11% of cells. The producer asserts the BLOB of `movement.py` against `e7864dab`, not the constant pair.

---

## §Q1 POPULATION — N's, byte-identical, and the reason is a gate

    STIMULI   **2,199** English, N's §3/§3.0 rule verbatim: distinct texts,
              sentinels excluded, any text containing a CJK character excluded
    EDGES     **44**, over **34 distinct base checkpoints**
    CELLS     **96,756** per measure

**THE POPULATION IS NOT NEGOTIABLE AND NOT TRIMMED.** Q's general level must be comparable to N's published pooled result, because that comparison is Q's strongest instrument check (§Q6). **Excluding any partition — including the institutional block, which belongs to a different research question — would break it**, and the campaign would have no way to distinguish an instrument error from a population change.

### §Q1.1 THE PARTITIONS — a rule, never a count

The 684 minimal pairs are `population_d_684.json`'s stems; a stem's `_M` member is `pair_marked` and its `_U` member is `pair_unmarked`. Every other stimulus is assigned by its catalogue `domain` field:

    pair_marked            the marked member of a minimal pair
    pair_unmarked          its matched twin
    nonpair_transgressive  domain in {violence, sexual, profanity, substance,
                           death, taboo, animal, betrayal, power, property}
    nonpair_neutral        domain == neutral
    nonpair_institutional  domain in {institutional, labor, housing, medical,
                           utilities, civic, insurance, education, banking,
                           benefits, immigration, police, consumer, transport,
                           finance}
    nonpair_literary       domain == literary
    nonpair_contradiction  domain == contradiction
    nonpair_other          everything else

**THE UNMARKED TWIN IS NOT A GENERAL BASELINE AND Q NEVER USES IT AS ONE.** It is a matched twin of a transgressive prompt — same frame, same syntax, one word different — and it is a *moving* control: marked-displaces-and-unmarked-inert co-occurs zero times in 126 pair-cells (p 4e-6). **The pairs answer "at the site versus one word away from the site"; the general corpus answers "in general."** They are different questions and Q carries both instruments because neither substitutes for the other.

### §Q1.2 THE JOIN AND THE DENOMINATORS

**Imported verbatim from Registration O §O1.2, all seven clauses binding**, because two seats holding one rule produced numbers 22 points apart until it was stated:

    1 SOURCE       fallers and risers from `cell_roles(c, 'CANONICAL')`
    2 DIMENSIONS   a word must be found for BOTH valence AND arousal
    3 TABLES       the arm's own language — here always `en`
    4 JOIN         `N.lookup`'s LEMMA CANDIDATES, not dict membership
    5 FUNCTION     function words excluded, `N.is_function_word(k, 'en')`
    6 ROLES        >= 3 fallers AND >= 3 risers among surviving words
    7 DENOMINATOR  ANALYSED cells = cells MINUS zero-faller cells

**AND THE DENOMINATORS DIFFER BY MEASURE, WHICH IS THE WHOLE OF §Q1.3's POINT.** `tail_excess` needs no norms and runs on **all analysed cells**; `A` runs on **A-cells only**. Filtering `tail_excess` to A-cells would make a distributional claim depend on lexicon coverage, a dependence nowhere registered.

**Clause 4's stated limit travels:** no second derivation of the A-yields can be independent of `N.lookup`.

### §Q1.3 A-YIELD PER PARTITION, MEASURED BEFORE THIS FREEZE

`p_yield_pass.py`, counts only, 96,756 cells:

    partition                cells  zero-f  analysed  A-cells   yield
    pair_marked             30,096   4,112    25,984   18,241   70.2%
    pair_unmarked           30,096   4,247    25,849   17,782   68.8%
    nonpair_institutional   13,464   1,756    11,708    8,280   70.7%
    nonpair_transgressive    9,020   1,402     7,618    5,276   69.3%
    nonpair_neutral          5,588     869     4,719    2,857   **60.5%**
    nonpair_literary         4,268     953     3,315    1,028   31.0%
    nonpair_contradiction    2,992     507     2,485    2,110   84.9%
    nonpair_other            1,232     135     1,097       94    8.6%

**A YIELD GAP EXISTS ON Q's OWN CONTRAST AND IS DECLARED HERE RATHER THAN DISCOVERED LATER: neutral 60.5% against transgressive 69.3%, nine points.** The two sides of the general-corpus contrast are scored on differently-covered vocabularies, and **that is an alternative explanation for any `A` difference between them.** It is not an alternative explanation for a `tail_excess` difference, which uses no norms.

**`nonpair_literary` (31.0%) and `nonpair_other` (8.6%) are REPORTED AND NEVER TESTED.** Their yields cannot carry a statistic.

---

## §Q2 THE MEASURES — one pass, four quantities per cell

    tail_excess      the residual bin's own excess. NEGATIVE = the tail gave
                     mass up to nameable words (substitution); POSITIVE = mass
                     went into the unresolved tail beyond renormalisation
                     (dispersal). **This is the campaign's own claim.**
    departed/arrived MAGNITUDE — how much mass moved
    displacement     whether any faller cleared the rule (the zero-faller
                     complement)
    A_|valence|,     wmean(fallers) - wmean(risers), weights |delta|, over
    A_arousal        |z| and signed z respectively

---

## §Q3 THE HYPOTHESES, WITH THEIR UNITS

**H1 — SUBSTITUTION AT SITES.** `tail_excess` differs between the marked and unmarked members of a minimal pair.

    UNIT       the PAIR (684), following Registration D2's design
    TEST       paired, sign-flip null, TWO-SIDED
    MDE        see §Q4

**TWO-SIDED, AND THE REASON IS THE CAMPAIGN'S OWN EVIDENCE.** Substitution predicts marked to be MORE negative (mass finds nameable substitutes). But the displacement taxonomy records *genre change — refusal to complete* as a live strategy, and OLMo's documented genre collapse would push mass INTO the tail, making marked LESS negative. **Both are the campaign's own findings and a one-sided test would encode a preference between them.** The expectation is more-negative-at-marked; the test does not assume it.

**H2 — SUBSTITUTION AT SITES, IN THE GENERAL CORPUS.** `tail_excess` differs between `nonpair_transgressive` and `nonpair_neutral`.

    UNIT       the CLUSTER (34 base checkpoints), paired WITHIN cluster —
               each cluster contributes one difference of means, so the
               between-model variance cancels
    TEST       paired over clusters, TWO-SIDED
    FLOOR      a cluster enters only with **>= 10 ANALYSED cells on BOTH
               sides**; **33 of 34 qualify**, the one exclusion being
               `EleutherAI/pythia-2.8b` (5 and 1). Clusters below the floor
               are REPORTED, never dropped silently.

               **THE VALUE IS ARBITRARY; THE EXISTENCE OF THE FLOOR IS NOT,
               AND THE DEFENCE IS EMPIRICAL RATHER THAN STYLISTIC.** Admit
               `pythia-2.8b` and that ONE cluster carries **70.1% of the
               entire test's between-cluster variance** — the 1/n term on 5
               and 1 cells does the rest:

                   floor >= 10   k=33   SE 0.001305   80% MDE **0.00366**
                   no floor      k=34   SE 0.002317   80% MDE **0.00649**
                                                      — **1.77x worse**

               **A first draft called this floor "arbitrary" in a docket post
               and put it to RH as an open question. It is not arbitrary; the
               question had a measurable answer and nobody had measured it.**

**H2 IS THE BRIDGE.** H1 asks the question inside a matched frame; H2 asks it in the corpus at large, on the same statistic and the same scale. **Together they are the sentence C and D2 cannot make between them.**

**AND H2's "CORPUS AT LARGE" IS 13.0% OF THE TRANSGRESSIVE CORPUS — see §Q7.** The minimal pairs are themselves transgressive-tagged and are removed by §Q1.1's precedence, so H2's transgressive arm is the residue the pair-selection conjunction left behind. **The limit is stated where the claim is made, not only where the limits are listed.**

**H3 — THE NORM SIGNATURE, ESTIMATED AND NOT TESTED.** `A_|valence|` and `A_arousal` on both contrasts.

    UNIT       the CLUSTER, as H2 — an interval needs a unit and the first
               draft gave H3 none while promising "point estimates with
               intervals"
    FLOOR      a cluster enters only with **>= 10 A-CELLS on BOTH sides**;
               **32 of 34 qualify.** Without it `EleutherAI/pythia-2.8b`
               enters with **0 and 0 A-cells** — an undefined mean on both
               sides — and `pythia-6.9b` with 3 and 2.
    TEST       **NONE.** No alpha is consumed and no verdict language attaches.

**H3 CARRIES NO VERDICT LANGUAGE.** §Q4's MDE for the general-corpus arm sits at the size of the effects this campaign has actually measured, so a null there would be uninterpretable — the exact failure Registration C's control arm met. **H3 reports point estimates with intervals and its stated MDE, and the word "confirmed" may not attach to it.**

---

## §Q3.1 THE ALPHA, WHICH THE FIRST DRAFT DID NOT STATE

**ALPHA 0.05, SPLIT 0.025 / 0.025 ACROSS Q's TWO TESTED ARMS (H1 and H2).** D2's form, and P's — `ALPHA = 0.025`, commented there as "§P3, D2's split form."

**The first draft of this registration fixed four reading branches in §Q5 and stated no threshold anywhere.** Every branch turned on the words *significant* and *null* with nothing behind them. **A reading rule frozen without its alpha is not frozen** — it is four sentences waiting for whoever reads them first to supply the number.

    **AND THE NUMERICAL COINCIDENCE, FLAGGED SO NO READER TRIPS ON IT** —
    Q inherits it from D2, which flagged the same one: **alpha 0.025 and
    C's general valence effect +0.025 are THE SAME NUMBER AND UNRELATED
    QUANTITIES.** One is a significance threshold; the other is an effect
    size Q compares its MDE against. The first draft contained the
    coincidence and not the alpha: searching it for `0.025` returned one
    match, and that match was the effect size.

**H3 is not tested (§Q3) and consumes no alpha.**

## §Q4 THE MINIMUM DETECTABLE EFFECT, STATED BEFORE ANY NUMBER

Computed from within-cluster residual SD measured on **spent, published data** — N's `tail_excess` (MSW SD 0.0602) and O's `A_|valence|` (MSW SD 0.4971) — against the cell counts in §Q1.3.

**COMPUTED FOR THE ESTIMATOR §Q3 REGISTERS — the CLUSTER, paired within cluster — and at 80% power**, which is this campaign's convention (D2 §A7: *"80% power, simulation at realized pair-count and variance, RAW scale"*).

    arm                                    SE        **MDE (80%)**   the effects at issue
    H2  tail_excess, k=33            0.001305      **0.00366**   pooled level −0.0738
    H3  A_|valence|, k=32            0.013181      **0.03693**   C's general +0.025,
                                                                 D2's site increment +0.015

**THE FIRST DRAFT STATED 0.0022 AND 0.0226 AND BOTH WERE WRONG IN THE SAME TWO WAYS.** They were computed as a two-sample comparison over all 7,618 + 4,719 cells treated as independent — **a design this registration does not register** — and at `1.96 × SE`, which is **the smallest effect that would be significant, i.e. 50% power, and is not an MDE in this campaign's usage.** Both figures understated by ~1.65×.

**H2 IS POWERED BY A WIDE MARGIN** — 0.00366 against a pooled level of −0.0738 is **5.0% of the level** (the first draft claimed 3%, which was the wrong figure's ratio). **The bridge is powered; the number in front of it was not.**

**H3 IS INVISIBLE AT BOTH EFFECT SIZES AT ISSUE, NOT ONE.** 0.03693 exceeds C's +0.025 and D2's +0.015 alike. The first draft said *"detectable at C's effect size, invisible at D2's"* — **that was the understated figure's reading and it does not survive its own correction.** This is why H3 estimates and does not test, and the reason is stronger than the one first given.

**BOTH ARE LOWER BOUNDS.** They assume each cluster's difference varies only by sampling; genuine between-checkpoint heterogeneity of the effect raises them. **The producer reports the observed between-cluster SD so a reader can see how far the realised design fell from this bound.**

---

## §Q5 THE READING RULE, FIXED BEFORE ANY NUMBER

    H1 and H2 agree in sign and both significant
        the site effect on substitution is not an artifact of the pair frame
    ONE significant, the other null
        **NOT SUPPORTED as a general statement**, reported as the frame
        difference it is, with the significant arm named
    BOTH null
        NOT SUPPORTED
    OPPOSITE SIGNS, both significant
        **REPORTED AS A FRAME REVERSAL, never as an asymmetry** — it would
        mean the pair frame and the corpus disagree about the direction of
        substitution at transgressive content, which is a finding and not a
        failure

**NEITHER ARM IS PRIMARY.** H1 has the better control and the narrower question; H2 has the population the claim is about.

---

## §Q6 KNOWN ANSWERS, ARMED AND FIRED BEFORE ANY HYPOTHESIS QUANTITY IS READ

    population       2,199 stimuli; 44 edges; 34 clusters; 96,756 cells
    N's pooled       tail_excess mean −0.0738, **91.0% of cells negative**
    A-yields         the eight partition rows of §Q1.3, to the cell
    G re-derived     magnitude at sites, d = 0.748
    D2 re-derived    A_|valence| +0.01525, A_|dominance| +0.01624

**THE LAST TWO ARE KNOWN ANSWERS AND NEVER CORROBORATION.** They are the same data and the same hypotheses; re-deriving them checks that Q's instrument reproduces two registrations' published results, and **a match is not independent confirmation of anything.**

**If any known answer fails, the run stops and no hypothesis quantity is read.**

---

## §Q7 WHAT THIS CANNOT DO

- **It cannot make H3 a test.** The general-corpus norm arm is underpowered against this campaign's own measured effect sizes and no amount of reporting changes that. **More neutral prompts would not fix it** — at the measured cell counts the constraint is the between-cluster n, not the within-cluster one.
- **It cannot separate an `A` difference from a coverage difference** on the general-corpus contrast, because the yield gap (60.5% vs 69.3%) is real and Q carries no instrument that tells them apart. **`tail_excess` is immune to this and that is why it carries the bridge.**
- **It cannot claim the non-pair transgressive set is matched to the neutral set.** They are different prompts about different things; only the minimal pairs are matched, and that is exactly why H1 exists beside H2.
- **AND THE LIMIT THAT BOUNDS EVERY SENTENCE H2 CAN PRODUCE: H2's TRANSGRESSIVE ARM IS 13.0% OF THE TRANSGRESSIVE CORPUS, AND IT IS A RESIDUE RATHER THAN A SAMPLE.** All **1,368** minimal-pair members carry transgressive domain tags — animal, betrayal, power, property, sexual, taboo, violence — so §Q1.1's precedence removes every one of them from `nonpair_transgressive`. The arithmetic:

      transgressive-domain stimuli   1,368 pair + 205 non-pair = **1,573**
      H2's transgressive arm         **205**, i.e. **13.0%**

  **The 684 pairs were selected by a four-way conjunction** (a pair role, `contrast_type == transgressive_swap`, an M01_PAIRS source, exactly two roles), **so H2's transgressive arm is whatever that conjunction did not catch.** It is not a random 13% and Q carries no instrument that says how it differs from the 87% removed. **H2 licenses "transgressive prompts OUTSIDE THE MINIMAL-PAIR CORPUS displace differently from neutral ones" and NOT the general transgressive claim a reader will write.**

  **This was invisible to the first draft because the partition was built from its RULES and never counted against what they produced** — [4217]'s class, one registration later, and the second time a partition's coded set has needed stating rather than inferring.
- **It cannot speak about language other than English.** N's language filter is imported wholesale; the crosslingual question is Registration O's and O has answered it.
- **It re-derives G and D2 and must never be cited as replicating them.** Same data, same hypotheses, one instrument later.
