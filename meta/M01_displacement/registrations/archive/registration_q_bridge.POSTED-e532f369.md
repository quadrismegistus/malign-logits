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

**THE PIN CARRIES THE COMMIT, NOT THE CONSTANTS ALONE.** Both sides of the 2026-08-04 repair satisfy `min_prob 0.003, fall_ratio 0.5` and select different faller sets in 11% of cells. **EACH PIN NAMES THE FUNCTION THAT PRODUCED IT** ([4193]'s rule, and [4191]'s hour):

    `malign_logits/movement.py`
      **git blob** (`git hash-object`)  = the blob at commit `e7864dab`
      the producer compares BLOB to BLOB, never commit to commit — a commit
      match passes identically whether a dirty tree's edit sits in the file
      being pinned or somewhere harmless
    `scripts/m01_norms.py` and its four pinned norm sources
      **sha256 of file bytes**, each verified by `load_norms(verify=True)`
      against the hash the module itself declares

**NO INSTRUMENT-DIGEST TRIPLE APPLIES HERE, AND THE ABSENCE IS DECLARED RATHER THAN LEFT TO INFERENCE.** [4298] adopts the `(digest, framework, pydantic)` form for pins **where an instrument or renderer is involved** — Q administers nothing to any model, calls no provider, and renders no prompt. **Its pins are code blobs and data hashes; there is no rendered artifact for a framework version to redefine underneath them.** Registration P needed the triple because its digest measured a rendered instrument and a framework change moved the measurement; **Q has no such surface, and saying so is cheaper than a reader wondering why the adopted form is missing.**

---

## §Q1 POPULATION — N's, byte-identical, and the reason is a gate

    STIMULI   **2,199** English, N's §3/§3.0 rule verbatim: distinct texts,
              sentinels excluded, any text containing a CJK character excluded
    EDGES     **44**, over **34 distinct base checkpoints**
    CELLS     **96,756** per measure

**THE POPULATION IS NOT NEGOTIABLE AND NOT TRIMMED.** Q's general level must be comparable to N's published pooled result, because that comparison is Q's strongest instrument check (§Q6). **Excluding any partition — including the institutional block, which belongs to a different research question — would break it**, and the campaign would have no way to distinguish an instrument error from a population change.

### §Q1.1 THE PARTITIONS — a rule, never a count

The 684 minimal pairs are `population_d_684.json`'s stems — **pinned by that file's own `id_set_sha256_16` = `3ed3e286e633c2fc`, the hash the file computes over its id set rather than a whole-file hash that moves when its metadata does** (the distinction Registration O's §O1.3 disclosure had to make after the fact; Q makes it up front); a stem's `_M` member is `pair_marked` and its `_U` member is `pair_unmarked`. Every other stimulus is assigned by its catalogue `domain` field — **read from `Prompt.row['domain']` via `malign_logits.prompts.Prompts().all()`, the same path `p_yield_pass.py` used to produce §Q1.3's counts** (that pass @ `sha256 e7cf89598d9e095d`, so §Q1.3's counts are reproducible from a NAMED instrument rather than from a described one):

    **TWO SOURCES OF TRUTH, AND THE PRECEDENCE IS DECLARED:** a text that is
    a 684 pair member takes its PAIR ROLE and NOT its domain — the pair
    membership wins. **This is not cosmetic: every one of the 1,368 pair
    members carries a transgressive domain tag, so without the precedence
    they would appear in two partitions at once.** See §Q7 for what that
    precedence costs the transgressive arm.

    **TIE-BREAK, for a text carrying more than one domain: FIRST-IN-CATALOGUE
    WINS.** One text in the population does — it is seated in
    `nonpair_neutral`, **44 cells, 0.79% of that arm.** Stated with its size
    so a reader can price the rule rather than trust it.

    **THE UNIT IS THE TEXT, NOT THE PROMPT ID.** The catalogue holds 2,590
    prompt rows and N's population is 2,199 distinct TEXTS; 8 texts are
    shared by more than one id. **The partition is over texts, so a shared
    text is assigned once and cannot be double-counted.**


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
                     **THE ARM IS `tail_excess_corrected`, NAMED HERE BECAUSE
                     N's ARTIFACT CARRIES TWO.** N §4.1: *"THE CORRECTED ARM
                     CARRIES ALL VERDICT LANGUAGE. THE RAW ARM CARRIES NONE
                     — it is a REPORTED DIAGNOSTIC and never a finding."*
                     **Q bridges to N's general-level result, so Q uses the
                     arm that result rests on.** The first draft named
                     neither, which left the choice to whichever instrument
                     touched the data first — and an arm chosen by a producer
                     is an arm nobody registered.
    departed/arrived MAGNITUDE — how much mass moved
    displacement     whether any faller cleared the rule (the zero-faller
                     complement)
    A_|valence|,     wmean(fallers) - wmean(risers), weights |delta|, over
    A_arousal        |z| and signed z respectively

---

## §Q3 THE HYPOTHESES, WITH THEIR UNITS

**H1 — SUBSTITUTION AT SITES.** `tail_excess` differs between the marked and unmarked members of a minimal pair.

    UNIT       the PAIR (684), following Registration D2's design
    PAIRING    **STRICT (stem, edge) BOTH-SIDES.** `d = tail_excess(M) −
               tail_excess(U)` on the SAME edge; a stem's value is the mean
               of its own both-sides edges. **24,606 keys kept, 1,253
               dropped (4.8%), all 684 stems surviving.**

               **THE COUNTS PRECEDED THE CHOICE AND THAT IS WHY THE 4.8% IS
               STATABLE.** They were produced at a second seat before this
               rule existed; had the rule come first, 4.8% would be a figure
               its author had decided to find acceptable.

               **The rejected alternative** — averaging each side over its
               own analysed edges — keeps the 1,253 one-armed keys and lets
               a stem's two sides rest on DIFFERENT EDGE SETS. **Pairing
               within an edge holds the MODEL fixed as well as the frame;
               the alternative reintroduces the model as a nuisance in the
               one statistic built to remove it.** Cheap in data, expensive
               in control, in one direction only.
    TEST       paired, sign-flip null, TWO-SIDED, `p < 0.0167`
    MDE        see §Q4 — **NOT STATED**, and §Q4 says why

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

**H4 — MAGNITUDE AT SITES, IN THE GENERAL CORPUS.** `departed` differs between `nonpair_transgressive` and `nonpair_neutral`.

    UNIT       the CLUSTER, paired within cluster, as H2
    FLOOR      **>= 10 ANALYSED cells on both sides — k = 33.** THE ANALYSED
               FLOOR, NOT THE A-CELL FLOOR (k=32). The two floors differ by
               one cluster and by which arm they belong to; conflating them
               is what produced [4297]'s correction and they are named apart
               here so it cannot happen twice.
    TEST       paired over clusters, TWO-SIDED, `p < 0.0167`
    MDE        **0.00508** — see §Q4 for its provenance, which is not this
               population's

**H4 IS TO G WHAT H2 IS TO H1.** G measured magnitude at transgressive sites inside the pair frame; H4 asks it of the corpus at large. **The pair-frame version is NOT a hypothesis here — it would re-derive G, which is exactly why G sits in §Q6's known answers instead.**

    **the bridge's symmetry, complete:**
      substitution  **H1** (pair frame)  /  **H2** (general corpus)
      magnitude     **G** (pair frame, DONE)  /  **H4** (general corpus)
      norms         **H3** — declared UNBRIDGEABLE, estimated, not tested

**AND THE ANALYSED FILTER IS REQUIRED FOR THIS MEASURE RATHER THAN INHERITED:** `departed` is **zero by construction** in a zero-faller cell — no faller cleared the rule, so no mass departed. Including those cells would put a structural zero into a mean of movement. **That discharges half of §Q1.2's denominator problem for this column instead of inheriting it.**

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

**ALPHA 0.05, SPLIT THREE WAYS — `p < 0.0167` EACH — ACROSS Q's THREE TESTED ARMS (H1, H2 and H4).** D2's split form, extended by one arm when H4 entered TESTED on RH's word.

    **the split's cost, priced BEFORE H4 was admitted rather than after:**
      H2   0.025 -> 0.0167   80% MDE **0.00366 -> 0.00388**, **+6.0%** —
           against a pooled level of −0.0738 that is **5.0% -> 5.3% of the
           level.** Trivial, and the number is stated so that "trivial" is
           a measurement rather than an assurance.
      H1   its MDE is not stated at all; see §Q4.
      H4   0.00508 at the split, against a `departed` scale whose
           within-cluster SD is 0.0789.

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

    arm                              SE        **MDE (80%)**   dispersion measured on
    H1  tail_excess, pair unit    **NOT STATED — see below**
    H2  tail_excess, k=33      0.001305    **0.00388**   **N's own population**, spent
    H4  departed, k=33         0.001711    **0.00508**   **L's FOUND PROSE**, borrowed
    H3  A_|valence|, k=32      0.013181    **0.03693**   **O's population**, borrowed
                                                         (estimated arm; no alpha)

**EVERY ROW NAMES WHERE ITS DISPERSION CAME FROM, BECAUSE THEY DO NOT COME FROM THE SAME PLACE.** Two of the three are priors borrowed from sibling populations, and a power table that mixes provenances silently is the same defect as a claim that mixes them.

**H4's grading, stated in full:** its MDE is computed from a within-cluster residual SD of `departed` measured on **Registration L's FOUND-PROSE population** (0.07888, k=34, N=4,268), **because no artifact retains `departed` on N's**. A prior from a sibling population, not a measurement of this one. **The producer reports the REALIZED within-cluster SD of `departed` on N's population beside this borrowed figure**, so a reader sees how far the prior sat from the target's truth.

**H1's MDE IS NOT STATED, AND THE REASON IS NOT THAT THE DATA IS ABSENT.** `result_n_primary.json` retains `tail_excess` per cell across all 82,775 analysed cells, and all 1,368 pair members join to (stem, role); **the per-pair differences are derivable on N's own population — better provenance than any other row in this table.** But H1 is paired, so **the SD of those differences and their MEAN are one subtraction, and the mean is H1's test statistic.** No seat that will read H1 may compute it.

    **A construction-blind pass emitting only `sd(d_i)` and `k`** — the form
    of `p_yield_pass.py` and `o_fluent_pass.py`, which cannot emit a
    hypothesis quantity by construction rather than by restraint — **makes
    this row postable before the freeze. If it does not run, this row stays
    NOT STATED.** It does not say the data is unavailable; that sentence
    would be false and would stop the next seat looking.

**WHEN IT RUNS, THE ROW FILLS FROM ONE FORMULA, STATED HERE BEFORE THE NUMBER EXISTS:**

    **MDE(H1) = 3.2356 x sd_d / sqrt(k)**,  k = 684 stems

    **3.2356 = 2.394 + 0.842** — the two-sided z at **alpha 0.0167** (§Q3.1's
    three-way split) plus the 80%-power z. **NOT 2.8016**, which is the
    constant for alpha 0.05 and would understate this MDE by **15.5%** —
    the right alpha named in the sentence and the wrong z used in the
    arithmetic, which is the defect class that produced §Q4's own correction.
    **The t-correction is checked, not assumed: `t_683 = 2.3999` against
    `z = 2.3940`, a difference of 0.006. Normal is used and this sentence
    exists so nobody re-opens it.**

    **the arm: `tail_excess_corrected`**, per §Q2 — **H1's run must use the
    arm its MDE was sized on**, and a pooled-level agreement between the two
    arms would not license substituting one for the other, because pooled
    agreement is a statement about a MEAN and this is a DISPERSION.

    **`sd_d` IS THE DISPERSION OF UNEQUALLY-PRECISE MEANS.** Each `d_i` is a
    stem's mean over its own both-sides edges, and stems carry unequal
    numbers of them (24,606 / 684 ≈ 36 of a possible 44). **That is the
    correct dispersion for the unweighted paired test Q registers — it
    matches the estimator rather than an idealised one — and it is stated
    for the same reason U2 makes H2's cluster mean declare itself
    unweighted.**

**THE FIRST DRAFT STATED 0.0022 AND 0.0226 AND BOTH WERE WRONG IN THE SAME TWO WAYS.** They were computed as a two-sample comparison over all 7,618 + 4,719 cells treated as independent — **a design this registration does not register** — and at `1.96 × SE`, which is **the smallest effect that would be significant, i.e. 50% power, and is not an MDE in this campaign's usage.** Both figures understated by ~1.65×.

**H2 IS POWERED BY A WIDE MARGIN** — 0.00366 against a pooled level of −0.0738 is **5.0% of the level** (the first draft claimed 3%, which was the wrong figure's ratio). **The bridge is powered; the number in front of it was not.**

**H3 IS INVISIBLE AT BOTH EFFECT SIZES AT ISSUE, NOT ONE.** 0.03693 exceeds C's +0.025 and D2's +0.015 alike. The first draft said *"detectable at C's effect size, invisible at D2's"* — **that was the understated figure's reading and it does not survive its own correction.** This is why H3 estimates and does not test, and the reason is stronger than the one first given.

**BOTH ARE LOWER BOUNDS.** They assume each cluster's difference varies only by sampling; genuine between-checkpoint heterogeneity of the effect raises them. **The producer reports the observed between-cluster SD so a reader can see how far the realised design fell from this bound.**

---

### §Q4.1 THREE ESTIMATOR CHOICES, DECLARED RATHER THAN DEFAULTED

**Each of these has a defensible alternative and a silent default. Q states which it takes and why, because an unstated estimator choice is a result the reader cannot price.**

    **U2 — CLUSTER WEIGHTING: UNWEIGHTED.** Every qualifying cluster counts
    once, regardless of its cell count. **The inverse-variance alternative
    would weight `pythia-2.8b`-like clusters toward zero and the largest
    toward one**, which is more efficient and answers a different question:
    unweighted asks whether the effect holds ACROSS CHECKPOINTS, weighted
    asks its size in a cell-weighted average. **N, O and D2 all weight
    clusters equally and Q follows them** — the bridge's whole point is
    comparability with those, and an estimator change would break it more
    thoroughly than any partition choice.

    **U3 — MULTI-EDGE CLUSTERS: POOL WITHIN CLUSTER, NOT AVERAGE OF EDGES.**
    A base checkpoint backing several families contributes ONE value formed
    from all its cells pooled. **Averaging its edges first would give a
    7-family lineage the same internal weighting as a 1-family one**, which
    is the [4110]-era pooling inversion in a new place.

    **ZERO-FALLER SENSITIVITY: NOT RUN, AND THE REASON IS STRUCTURAL.**
    §Q1.2's clause 7 excludes zero-faller cells from the analysed
    denominator, and a sensitivity column re-admitting them would be
    incoherent for two of Q's three tested arms: `tail_excess` is defined
    against a faller set, and `departed` is **zero by construction** where
    none cleared the rule. **Re-admitting them does not perturb the
    estimate; it changes what is being averaged.** The column is declined in
    text rather than omitted in silence.

## §Q5 THE READING RULE, FIXED BEFORE ANY NUMBER

**"SIGNIFICANT" IN EVERY BRANCH BELOW MEANS `p < 0.0167`** — §Q3.1's three-way split of alpha 0.05 over H1, H2 and H4, **repeated here at the point of use.** A reading rule that names its threshold only in another section is a reading rule whose first reader supplies the number, and the first draft of this registration named it in no section at all.

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
    G re-derived     magnitude at sites, **d = 0.748**
    D2 re-derived    A_|valence| **+0.01525**, A_|dominance| **+0.01624**

    **TOLERANCE, so the stop-gate can actually fire: |observed − published|
    <= 5e-5 on each, i.e. agreement to the last published digit.** A known
    answer without a tolerance is not a gate — it is a number printed beside
    another number, and whoever reads them decides whether they matched.
    **RH's word keeps G and D2 in this list; the tolerance is what makes
    keeping them mean something.**

    **AND THEY ARE KNOWN ANSWERS, NEVER CORROBORATION** — same data, same
    hypotheses, one instrument later. A match checks that Q's machinery
    reproduces two published results. It confirms nothing.

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
