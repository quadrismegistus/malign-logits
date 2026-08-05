# Registration Q — general and site on one scale

**STATUS AS DRAFTED (2026-08-04 UTC): DRAFT, not frozen, not in force.** As of this date, verified at the seats named: no `result_q_*.json` existed anywhere; **no seat had read a hypothesis quantity on any partition of N's population**; no producer had been written.

**TWO BLIND PASSES HAVE RUN, AND NEITHER IS A PRODUCER.** Both are precondition instruments that emit counts or a dispersion and cannot emit a hypothesis quantity:

    `p_yield_pass.py`   @ sha256 `e7cf89598d9e095d`  — §Q1.3's yields, counts only
    `q_h1_sd_pass.py`   @ sha256 `eef2f6047749fdda`  — §Q4's H1 dispersion

**THE SECOND FORMED `tail_excess` DIFFERENCES ON THE PAIR PARTITION AND THAT IS STATED RATHER THAN GLOSSED.** An earlier draft of this line said *no `tail_excess` value had been computed on any partition by anyone*, and after the sd-pass ran that sentence was false: the pass reads `tail_excess_corrected` per cell, forms all 24,606 per-edge differences and their 684 stem means, and **emits only their standard deviation and their count.** The mean was formed and never egressed — a property audited at a second seat against an enumerated egress criterion, with the checker watched failing on five deliberate leaks first. **No seat holds the direction of those differences, and that is what "blind" means here; it does not mean the arithmetic never happened.**

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

### §Q1.3.1 THESE YIELDS ARE OVER 44 LABELLED TRIPLES; THE PAIR ARMS RUN ON 43 DISTINCT TRANSITIONS

**The table above predates §Q3's declared key.** It counts `(family, base, aligned)` triples, so the `tulu` / `tulu-no-safety` transition — one measurement under two labels — **is counted TWICE.** §Q3's key collapses family labels, so H1, H5 and H6 run on **43 distinct `(base, aligned)` edges.**

    **THE TWO COUNTS DIFFER BY EXACTLY ONE EDGE AND RECONCILE TO THE UNIT**,
    which is stated here because a reader who meets both sets will go
    looking, and the reconciliation is more convincing than the caveat:

      pair_marked cells  **30,096** = 44 x 684    (this table)
      pair keys          **29,412** = 43 x 684    (the pair arms)
      difference            **684** = ONE EDGE x 684 STEMS

      A-cells, marked    17,707 + **534** = **18,241** == this table ✓
      A-cells, unmarked  17,281 + **501** = **17,782** == this table ✓
      the collapsed edge yields 78.1% / 73.2%, consistent with the ~70%
      seen across the pair partitions

**BOTH COUNTS ARE CORRECT UNDER THEIR OWN RULE AND THE GAP IS EXACTLY THE RULE.** Nothing here is a defect; two instruments counted under two declared conventions and agree about the difference between them.

**AND THE BOTH-SIDES COUNTS WERE RE-DERIVED FROM THE PIPELINE RATHER THAN FROM N's ARTIFACT.** `q_h6_denominator_pass.py` @ `sha256 90a81833eaafabfa` recomputed **24,606 both-sides and 1,253 one-sided keys** from `word_probs()` and `movement()` at the pinned commit — figures [4319] originally derived by reading `result_n_primary.json`. **Two independent routes to the same two integers**, so the denominator H1 and H5 rest on is a property of the pipeline and not of one stored file.

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

### §Q2.1 THE BATTERY — one pass, and the partitions are a READ-OUT

**THE PRODUCER COMPUTES ALL FOUR MEASURES ON ALL 96,756 CELLS, AND EVERY PARTITION IN §Q1.1 IS READ OUT OF THAT SINGLE PASS.** No partition is computed separately, no arm has its own traversal, and **a cell's four values do not depend on which partition it lands in** — the partition is applied when the statistics are formed, never when the measures are.

**THE PER-MEASURE DENOMINATORS OF §Q1.2 ARE A READ-OUT PROPERTY, NOT A COMPUTATION PROPERTY.** Each measure is computed **wherever it is defined** and filtered at **READ-OUT** — `tail_excess` and `departed` over analysed cells, `A` over A-cells. **No measure is ever skipped by partition**, so a partition can never be the reason a cell is missing from an arm.

    **AND THE `A` COLUMN LOSES ITS CELLS IN TWO STRUCTURALLY DIFFERENT WAYS,
    WHICH A PRODUCER AUTHOR MUST NOT CONFLATE:**

        analysed cells                                 **82,775**
          zero QUALIFYING fallers            8,133   9.83%
          zero QUALIFYING risers             7,635   9.22%
        **zero on EITHER role -> `A` UNDEFINED**       **11,681  14.11%**
        **`A` DEFINED**                                **71,094  85.89%**
        **`A` defined but FAILING §Q1.2 clause 6's
          >= 3-per-role floor -> declined**            **15,426  18.64%**
        **A-CELLS**                                    **55,668  67.25%**
                                       — and 55,668 is §Q1.3's A-cell
                                         column summed, to the cell

    **UNDEFINED IS NOT DECLINED.** `A` is `wmean(fallers) − wmean(risers)`;
    with no qualifying word in a role it is **a mean of an empty set and
    does not exist** — a producer written otherwise divides by zero on
    11,681 cells, or emits NaN and carries it into a weighted mean. **The
    15,426 are the opposite case: `A` exists there and clause 6 refuses it.**
    An earlier draft of this clause said `A` is *"computed and then
    declined"* everywhere, **which is true of the 15,426 and false of the
    11,681** — and the two cannot share a sentence, because one is a
    property of the MEASURE and the other of the RULE.

**SO THERE ARE EXACTLY THREE REASONS A CELL IS ABSENT FROM AN ARM, AND ONLY THE FIRST TWO ARE ALLOWED:** the measure is **undefined there** (structural, 11,681, reported), the cell **failed §Q1.2's denominator rule** (declined, 15,426, reported), or **its partition was traversed separately — which is a defect**, and is the defect this section exists to make unwritable. **All three counts are stated so a producer asserts them rather than discovers them.**

    **THE CORRECTION IS ITSELF THE SECTION'S OWN LESSON.** §Q2.1 argues that
    prose binds implementations, and its illustrative clause was **wrong
    about the implementation** — a false universal inside the one sentence
    in this registration whose stated job is to be read by a producer
    author. **The binding half was right and survived; the vivid half was
    the one that overreached**, which is the direction that kind of sentence
    always fails in.

    **WHY THE SENTENCE EXISTS AND WHY ITS ABSENCE COST FOUR ROUNDS:**
    Registration P's frozen text had exactly this gap — the battery
    described in a heading and nowhere stated as a clause — and it cost a
    build round, because a producer written from the text alone can satisfy
    every stated rule while traversing once per arm. **A HEADING IS NOT A
    CLAUSE AND A COUNT IS NOT A RULE:** §Q2's own title says "one pass, four
    quantities per cell" and §Q1 says "96,756 cells **per measure**", and
    neither binds an implementation.

    **This was the FIRST defect named against this draft at its own seat,
    and the LAST of sixteen to be repaired** — every item that arrived as
    another seat's finding landed within a round, and the one found in-house
    survived four. **Recorded here rather than only in the ledger, because
    the asymmetry is about how a round allocates attention and not about
    this sentence.**

---

## §Q3 THE HYPOTHESES, WITH THEIR UNITS

**H1 — SUBSTITUTION AT SITES.** `tail_excess` differs between the marked and unmarked members of a minimal pair.

    UNIT       the PAIR (684), following Registration D2's design
    PAIRING    **STRICT (stem, edge) BOTH-SIDES.** `d = tail_excess(M) −
               tail_excess(U)` on the SAME edge; a stem's value is the mean
               of its own both-sides edges. **24,606 keys kept, 1,253
               dropped (4.8%), all 684 stems surviving.**
    THE KEY    **AN EDGE IS `(base, aligned)`. THE FAMILY LABEL IS NOT PART
               OF THE KEY AND ENTERS NO COMPUTATION.** Where one transition
               carries two family labels it **CONTRIBUTES ONCE.**

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

               **ONE TRANSITION IN THE CAMPAIGN CARRIES TWO FAMILY LABELS,
               AND THE KEY EXISTS BECAUSE OF IT.** `meta-llama/Llama-3.1-8B`
               -> `allenai/Llama-3.1-Tulu-3.1-8B` is served by **`tulu` AND
               `tulu-no-safety`**, which share a base AND a superego
               checkpoint and differ only at the SFT stage: **43 distinct
               transitions under 44 (family, base, aligned) triples.**
               `movement()` is a function of `(base, aligned, prompt)`, so
               the two cells are **the same measurement under two names.**
               Keying with the family would make every stem's mean average
               one number TWICE and gain no information for it — **CLAUDE.md's
               "one measurement counted twice", the ruling that re-derived
               the liminal-vs-explicit contrast at n=8 rather than 9,
               arriving in a new instrument.**

               **THE PUBLISHED COUNTS ARE FAMILY-BLIND AND THEREFORE STAND
               UNDER THIS KEY.** 24,606 / 1,253 / 684 were produced without
               the family field; keying WITH it would give 25,290 / 1,253 /
               684 instead. **`k` IS 684 UNDER BOTH DISPOSITIONS** — every
               stem keeps a both-sides edge either way — so the choice moves
               the both-sides key count and the composition of each `d_i`,
               and it does not move the denominator of the MDE. **An earlier
               statement at this seat said the choice changed `k`; it does
               not, and the correction is recorded here because a right
               conclusion resting on a wrong mechanism is the kind that gets
               re-derived later and comes out differently.**
    TEST       paired, sign-flip null, TWO-SIDED, `p < 0.0167`
    MDE        **0.00183** — see §Q4. The best-provenanced row in the
               table: N's own population, spent, and produced BLIND.

**TWO-SIDED, AND THE REASON IS THE CAMPAIGN'S OWN EVIDENCE.** Substitution predicts marked to be MORE negative (mass finds nameable substitutes). But the displacement taxonomy records *genre change — refusal to complete* as a live strategy, and OLMo's documented genre collapse would push mass INTO the tail, making marked LESS negative. **Both are the campaign's own findings and a one-sided test would encode a preference between them.** The expectation is more-negative-at-marked; the test does not assume it.

**H2 — SUBSTITUTION AT SITES, IN THE GENERAL CORPUS.** `tail_excess` differs between `nonpair_transgressive` and `nonpair_neutral`.

    UNIT       the CLUSTER (34 base checkpoints), paired WITHIN cluster —
               each cluster contributes one difference of means, so the
               between-model variance cancels
    THE KEY    **H1's KEY BINDS THIS ARM TOO.** An edge is `(base, aligned)`;
               family labels collapse; the shared transition contributes
               ONCE inside its cluster's mean.
    TEST       paired over clusters, TWO-SIDED, `p < 0.0167`
    FLOOR      a cluster enters only with **>= 10 ANALYSED cells on BOTH
               sides**; **33 of 34 qualify**, the one exclusion being
               `EleutherAI/pythia-2.8b` (5 and 1). Clusters below the floor
               are REPORTED, never dropped silently.

               **THE VALUE IS ARBITRARY; THE EXISTENCE OF THE FLOOR IS NOT,
               AND THE DEFENCE IS EMPIRICAL RATHER THAN STYLISTIC.** Admit
               `pythia-2.8b` and that ONE cluster carries **70.1% of the
               entire test's between-cluster variance** — the 1/n term on 5
               and 1 cells does the rest:

                   floor >= 10   k=33   SE 0.001305   80% MDE **0.00422**
                   no floor      k=34   SE 0.002317   80% MDE **0.00750**
                                                      — **1.78x worse**

               **Both figures are at the registered two-sided alpha 0.0167**
               (multiplier 3.2356). An earlier draft quoted 0.00366 and
               0.00649 here, which are the same SEs at **alpha 0.05** — the
               ratio was unaffected because the multiplier cancels, and the
               absolute numbers were at a threshold this registration does
               not run under. **Every MDE in this document is now stated at
               the alpha its own test uses, and each says which.**

               **A first draft called this floor "arbitrary" in a docket post
               and put it to RH as an open question. It is not arbitrary; the
               question had a measurable answer and nobody had measured it.**

**AND THE KEY IS NOT COSMETIC HERE — IT LANDS ON ONE CLUSTER, IN THE ARM THIS REGISTRATION CALLS THE BRIDGE.** `meta-llama/Llama-3.1-8B` carries **7 family-labelled edges over 6 distinct transitions**; every other cluster's labelled and distinct counts are equal. **Without the key that cluster's mean double-weights one transition.** It is a within-cluster weighting and not a between-cluster bias, and it is one cluster of 33 — **but it is the same defect as H1's in a registered analysis, and the two arms must use the same edge set or the bridge compares two different things.** Declared here rather than left to the producer.

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
    MDE        **0.00554** — see §Q4 for its provenance, which is not this
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
      H2   0.025 -> 0.0167   80% MDE **0.00402 -> 0.00422**, **+4.9%** —
           against a pooled level of −0.0738 that is **5.45% -> 5.72% of
           the level.** Trivial, and the number is stated so that "trivial"
           is a measurement rather than an assurance.
      H1   **0.00183** at the split; see §Q4 for how it was produced.
      H4   **0.00554** at the split, against a `departed` scale whose
           within-cluster SD is 0.0789.

    **THESE FIGURES ARE CORRECTED AND THE ERROR IS NAMED, BECAUSE IT IS
    THE DEFECT CLASS THIS SECTION EXISTS TO CATCH.** The cost was first
    priced as **0.00366 -> 0.00388, +6.0%, 5.0% -> 5.3%**, and all three
    numbers were wrong in two independent ways at once: **the baseline was
    the alpha-0.05 figure rather than the alpha-0.025 one the two-arm split
    actually used**, and **the post-split figure was computed with the
    ONE-SIDED z at 0.0167 (2.9689) for a test registered TWO-SIDED
    (3.2356).** The same one-sided slip produced H4's 0.00508. **This is
    exactly the class §Q4 already books — the right alpha named in the
    sentence and the wrong z used in the arithmetic — and it was found in
    the two rows nobody re-derived, having arrived as an answer to a
    question rather than as a claim.** H1's and H3's multipliers were
    two-sided and correct throughout.

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

Computed from dispersions measured on **spent data**, from **four different populations**, against the cell counts in §Q1.3.

**COMPUTED FOR THE ESTIMATOR §Q3 REGISTERS — the CLUSTER, paired within cluster (the PAIR for H1) — and at 80% power**, which is this campaign's convention (D2 §A7: *"80% power, simulation at realized pair-count and variance, RAW scale"*).

    arm                             SE      **MDE (80%)**  alpha   dispersion measured on
    H1  tail_excess, k=684       0.000565   **0.00183**   0.0167  **N's own**, spent, BLIND
    H2  tail_excess, k=33        0.001305   **0.00422**   0.0167  **N's own**, spent
    H4  departed, k=33           0.001711   **0.00554**   0.0167  **L's FOUND PROSE**, borrowed
    H3  A_|valence|, k=32        0.013181   **0.03693**   0.05    **O's**, borrowed
                                                                  (estimated; consumes no alpha)

**EVERY ROW NAMES ITS ALPHA AS WELL AS ITS DISPERSION, BECAUSE THE TABLE MIXES BOTH.** H1, H2 and H4 are at the registered two-sided **0.0167**, multiplier **3.2356** = 2.3940 + 0.8416, the exact three-way split 0.05/3. **H3 is at two-sided 0.05** — the conventional statement for an arm that consumes no alpha and runs no test; at 0.0167 it would be **0.04265**, and H3's conclusion is the same at either, which is why the looser figure is quoted and labelled rather than hidden.

**EVERY ROW NAMES WHERE ITS DISPERSION CAME FROM, BECAUSE THEY DO NOT COME FROM THE SAME PLACE.** **Two of the four are N's own population and two are priors borrowed from sibling populations**, and a power table that mixes provenances silently is the same defect as a claim that mixes them. **The two borrowed rows are the two weakest arms in the registration** — H4, whose dispersion comes from found prose, and H3, which is estimated and tested nowhere.

**H4's grading, stated in full:** its MDE is computed from a within-cluster residual SD of `departed` measured on **Registration L's FOUND-PROSE population** (0.07888, k=34, N=4,268), **because no artifact retains `departed` on N's**. A prior from a sibling population, not a measurement of this one. **The producer reports the REALIZED within-cluster SD of `departed` on N's population beside this borrowed figure**, so a reader sees how far the prior sat from the target's truth.

**H1's MDE IS STATED, AND IT WAS PRODUCED BY A SEAT THAT COULD NOT SEE THE EFFECT.** `result_n_primary.json` retains `tail_excess` per cell across all 82,775 analysed cells, and all 1,368 pair members join to (stem, role); **the per-pair differences are derivable on N's own population — better provenance than any other row in this table.** But H1 is paired, so **the SD of those differences and their MEAN are one subtraction, and the mean is H1's test statistic.** **No seat that will read H1 may compute it**, which is why H2 and H3 could be sized from spent data and H1 could not: theirs came from a POOLED within-cluster dispersion, which carries no contrast; H1's requires the contrast under test.

    **A CONSTRUCTION-BLIND PASS RESOLVED IT, IN THE FORM `p_yield_pass.py`
    AND `o_fluent_pass.py` ALREADY HAD** — an instrument that cannot emit a
    hypothesis quantity by construction rather than by restraint. **Three
    seats, none of them holding two roles:**

        WRITTEN BY   a seat that authors neither Q's text nor H1's reading
        AUDITED BY   a seat that reads code and never artifacts, against an
                     ENUMERATED EGRESS criterion — exactly two values leave,
                     and no SURROGATE for the effect leaves either: no sum,
                     median, quantile, extremum, t, p, CI, per-pair value,
                     **or count of positive differences, which is a sign
                     test and is the effect by a route with no "mean" in it**
        RUN BY       this seat, after verifying the cleared hash and
                     escrowing the bytes, reading only what it printed

    **THE AUDIT WAS POSITIVE-CONTROLLED BEFORE IT CLEARED ANYTHING.** Five
    deliberately leaky variants were built and each was caught — printing
    `sum(d)`, returning the per-pair vector, writing an intermediate file,
    printing the count of positive differences, printing the mean itself.
    **"No egress found" and "my checker looks for the wrong thing" are the
    same output until the checker has been watched refusing.**

    **AND THE ESCROW IS THE ONLY RECOVERY A BLIND INSTRUMENT HAS.** A
    cleared version of this pass was edited in place and the hash stopped
    the run at this seat. For ordinary code the next step is to read the
    diff and judge; **for this file it is not, because reading the body to
    adjudicate a mismatch is the exposure the design prevents.** The reading
    seat could reconstruct and did, byte-exactly. The running seat never
    can, at any price.

        `q_h1_sd_pass.py`  cleared and run @ sha256 **`eef2f6047749fdda`**
        escrowed at this seat BEFORE the run as
        `scripts/audited/q_h1_sd_pass.CLEARED-eef2f604.py`

    **THE PASS ALSO REFUSED ONCE, ON REAL DATA, AND THAT REFUSAL IS §Q3's
    KEY.** Its guard found `(stem, edge, role)` non-unique for every one of
    the 1,368 pair members — the `tulu` / `tulu-no-safety` shared transition
    — **and stopped rather than silently averaging a doubled edge into 684
    stem means.** The count that had reported 24,606 keys could not have
    reported the collision, because it keyed with a SET and a set absorbs
    what a `refuse-on-duplicate` guard rejects. **A count that merges a
    duplicate cannot report one.** The registered resolution collapses the
    duplicate only on EXACT equality of the two cells and refuses otherwise,
    naming the edge and no value; **on the run it never fired, so the
    ruling's premise held rather than being assumed.**

**THE ROW FILLED FROM A FORMULA THAT WAS IN THIS TEXT BEFORE `sd_d` EXISTED — THE NUMBER FILLED A SLOT RATHER THAN CHOOSING ONE:**

    **MDE(H1) = 3.2356 x sd_d / sqrt(k)**,  k = 684 stems

    **sd_d = 0.014765**,  **k = 684**   — the pass's entire output
    **MDE(H1) = 3.2356 x 0.014765 / sqrt(684) = 0.00183**, checked at a
    second seat. **Against N's pooled level of −0.0738 that is 2.48%.**

    **H1 IS THE BEST-POWERED ARM IN THIS REGISTRATION AND HAS THE BEST
    PROVENANCE IN THE TABLE.** Its MDE is **0.4326 of H2's** — **about
    43%, which is a little over two-fifths and NOT "a third"**, a ratio
    first stated at this seat as a third and corrected at another before it
    could reach prose. **Pairing within the edge cuts the detectable effect
    to well under half, and that is the pair frame's control showing up as
    arithmetic.**

    **AND `sd_d` RECONSTRUCTS FROM NUMBERS THAT DID NOT PRODUCE IT.** Each
    `d_i` averages 24,606 / 684 = 35.97 edges; if a stem's cells were
    independent at N's published within-cluster SD of 0.0602, the predicted
    dispersion is `0.0602 x sqrt(2 / 35.97)` = **0.014194** against an
    observed **0.014765** — a ratio of **1.040**, mild positive dependence
    across a stem's edges, which is what a shared prompt should produce.
    **This is CORROBORATION AND NOT CONFIRMATION** — same data, and the
    prediction uses N's own SD. **It rules out a broken instrument; it does
    not validate the estimate.** A blind pass could have returned anything.

    **3.2356 = 2.3940 + 0.8416** — the TWO-SIDED z at **alpha 0.05/3**
    (§Q3.1's three-way split, stated as the threshold `p < 0.0167`, which is
    that quotient rounded DOWN and therefore conservative for the test) plus
    the 80%-power z. **The two constants it is not, and both have been used
    in this document by mistake:**

        **2.8016** = 1.9600 + 0.8416   two-sided at alpha **0.05**
                                       — understates by **15.5%**
        **2.9689** = 2.1272 + 0.8416   **ONE-SIDED** at alpha 0.0167
                                       — understates by **8.2%**, and this
                                       is the one that reached H2's and
                                       H4's rows: a one-sided constant on
                                       tests §Q3 registers TWO-SIDED

    **The defect class is the same both times — the right alpha named in the
    sentence and the wrong z used in the arithmetic — and the second time it
    survived a round because the two rows arrived as an answer to a question
    rather than as a claim, and nobody re-derives an answer.**
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

**H2 IS POWERED BY A WIDE MARGIN** — 0.00422 against a pooled level of −0.0738 is **5.72% of the level** (the first draft claimed 3%, which was the wrong figure's ratio; a later draft claimed 5.3%, which was the one-sided z's). **The bridge is powered; the number in front of it has now been wrong twice and is stated here with the alpha and the sidedness that produced it.**

**H3 IS INVISIBLE AT BOTH EFFECT SIZES AT ISSUE, NOT ONE.** 0.03693 exceeds C's +0.025 and D2's +0.015 alike. The first draft said *"detectable at C's effect size, invisible at D2's"* — **that was the understated figure's reading and it does not survive its own correction.** This is why H3 estimates and does not test, and the reason is stronger than the one first given.

**BOTH ARE LOWER BOUNDS.** They assume each cluster's difference varies only by sampling; genuine between-checkpoint heterogeneity of the effect raises them. **The producer reports the observed between-cluster SD so a reader can see how far the realised design fell from this bound.**

---

### §Q4.1 THREE ESTIMATOR CHOICES, DECLARED RATHER THAN DEFAULTED

**Each of these has a defensible alternative and a silent default. Q states which it takes and why, because an unstated estimator choice is a result the reader cannot price.**

    **U2 — CLUSTER WEIGHTING: UNWEIGHTED, AND THE ALTERNATIVE IS PRICED
    RATHER THAN DISMISSED.** Every qualifying cluster counts once,
    regardless of its cell count.

        **SE unweighted        0.001305    80% MDE  0.00422**
        **SE inverse-variance  0.001116    80% MDE  0.00361**
        **the choice costs 17.0% of precision** — the unweighted SE is
        **1.170x** the efficient one, and Q pays it deliberately

        what the efficient estimator would do to the roster:
          `EleutherAI/pythia-6.9b`    22 and 19 cells   **0.35%** of the
                                      weight, against 3.03% unweighted
          `meta-llama/Llama-3.1-8B`   1,314 and 826     **17.42%** of the
                                      weight, against 3.03% unweighted

    **ONE CLUSTER WOULD CARRY A SIXTH OF THE ANSWER AND ANOTHER A THREE-
    HUNDREDTH, WHICH IS THE WHOLE OBJECTION.** Inverse-variance is more
    efficient and answers a different question: unweighted asks whether the
    effect holds ACROSS CHECKPOINTS, weighted asks its size in a cell-
    weighted average — **and this campaign's contrast is over lineages, so
    an estimator that lets the largest-sampled lineage speak fifty times
    louder than the smallest is measuring something Q did not register.**
    **N, O and D2 all weight clusters equally and Q follows them** — the
    bridge's whole point is comparability with those, and an estimator
    change would break it more thoroughly than any partition choice.

    **The figures are counts-only** (cell counts per cluster per arm, and
    N's published within-cluster SD 0.0602 — no `tail_excess` value is read
    to produce them), and the instrument was **known-answered against §Q3's
    published unweighted SE before its new number was believed: recomputed
    0.001305 against published 0.001305, agreeing to 5e-6.**
    **`q_u2_weighting.py` @ `sha256 ce68920458d2273d`** — named, like
    §Q1.3's, so the price is reproducible from an instrument rather than
    from a description. **An earlier round reported this item as landed when
    the alternative had been named and never priced; a declared estimator
    choice whose alternative carries no number is a preference, not a
    decision a reader can audit.**

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

### §Q5.1 THE SUBSTITUTION ARMS — H1 x H2, on `tail_excess`

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

### §Q5.2 THE MAGNITUDE ARM — H4, on `departed`, WHICH THE FOUR BRANCHES ABOVE DO NOT COVER

**H4 CONSUMES AN ALPHA AND THEREFORE NEEDS ITS OWN SENTENCE.** §Q3.1 splits 0.05 three ways over H1, H2 **and H4**, and §Q5.1's branches enumerate H1 x H2 and stop. **H4 cannot be folded in by adding a name: it measures `departed`, a magnitude, where H1 and H2 measure `tail_excess`, a direction.** Its arms below are fixed before any number, at the same `p < 0.0167`.

    SIGNIFICANT, more mass departing at TRANSGRESSIVE than at NEUTRAL
        **CONFIRMED, and the sentence is a GENERALISATION OF G AND NEVER A
        REPLICATION OF IT.** G measured magnitude at sites inside the
        matched pair frame; H4 asks it of the corpus at large. The licensed
        claim is *"the magnitude effect is not confined to the minimal-pair
        frame"* — bounded, as always, by §Q7's residue limit.
    SIGNIFICANT, more mass departing at NEUTRAL
        **REPORTED AS A FRAME REVERSAL, never as an asymmetry**, on §Q5.1's
        rule exactly: the pair frame and the corpus disagreeing about where
        alignment moves more mass is a finding, not a failure.
    NULL
        **NOT SUPPORTED, AND QUOTED AS A BOUND RATHER THAN AS AN ABSENCE.**
        A null here says the cluster-mean difference is smaller than the
        detectable one; it does not say it is zero, and the registration
        will not let it be written as though it did.

**THE BOUND IS RESTATED AT THE REALIZED DISPERSION, ALWAYS, AND THE TWO NUMBERS DO DIFFERENT JOBS.** H4's registered MDE is **0.00554**, sized on a **borrowed** prior — L's found-prose within-cluster SD of `departed`, 0.0789 — which is **7.0% of that SD**. §Q4 already requires the producer to report the **REALIZED** within-cluster SD of `departed` on N's own population.

    **the pre-registered MDE governs the DESIGN** — whether this arm was
    worth an alpha before anyone looked
    **the realized SD governs the BOUND** — what a null actually excludes

**So a null is reported at the realized figure with the borrowed one printed beside it, unconditionally.** There is no threshold to argue about and no branch: the rule does not ask whether the prior sat close enough, it simply never lets the borrowed number carry a claim it was not measured for. **If the realized SD is materially larger, the bound widens and the sentence says so; if smaller, it tightens.** A bound quoted from a sibling population's dispersion would be the [4187] provenance defect arriving in a verdict instead of in a table.

**AND H2 AND H4 MAY DISSOCIATE — THAT IS A READING, NOT A FAILURE.** They test the same contrast with different questions: H2 asks whether the tail gives mass up to nameable words, H4 asks how much mass moved. **Registrations F and G already found exactly this shape inside the pair frame — displacement at transgressive sites is not more FREQUENT but is LARGER when it happens** — so a general-corpus split between the two is the F/G structure generalising, and it is reported as that with both arms named. **Neither arm may be quoted alone as "the site effect in the general corpus."**

---

## §Q6 KNOWN ANSWERS, ARMED AND FIRED BEFORE ANY HYPOTHESIS QUANTITY IS READ

    **THREE, AND EVERY ONE IS COMPUTED BY Q's OWN MACHINERY ON Q's OWN
    POPULATION:**

    population       2,199 stimuli; 44 edges; 34 clusters; 96,756 cells
    N's pooled       tail_excess mean **−0.0738**, **91.0% of cells negative**
    A-yields         the eight partition rows of §Q1.3, to the cell

    **TOLERANCE, BY KIND, so the stop-gate can actually fire:**
      the population counts and the eight A-yield rows are INTEGERS —
        **EXACT EQUALITY.** A tolerance on a count is an invitation.
      N's two published figures are floats — **|observed − published|
        <= 5e-5**, agreement to the last published digit.
    **A known answer without a tolerance is not a gate — it is a number
    printed beside another number, and whoever reads them decides whether
    they matched.**

    **THESE THREE CHECK THE THINGS THAT COULD CORRUPT Q's ARMS**: the
    population construction, the `tail_excess` computation and its join, and
    the norm join with its filters and denominators. **They are known
    answers and never corroboration** — same data, one instrument later.

### **§Q6.1 G AND D2 WERE KNOWN ANSWERS AND ARE WITHDRAWN — WITH THE REASON, BECAUSE A GATE THAT VANISHES SILENTLY IS INDISTINGUISHABLE FROM ONE THAT NEVER EXISTED**

An earlier cut carried two more, on RH's word: **G re-derived at `d = 0.748`, and D2 re-derived at `A_|valence| +0.01525, A_|dominance| +0.01624`.** RH asked what they were for, and checking the sources answered it:

    **`d = 0.748` IS G's OWN NUMBER AND DERIVES FROM G's OWN ARTIFACT —
    BUT THE CONVENTION THAT PRODUCES IT IS NOWHERE RECORDED.**
    `result_g_magnitude.json` STORES `primary/statistic = 0.16854886697`
    and 34 per-unit `D_departed` values. **0.748 is DERIVED from those 34:**

        n = 34   mean = 0.004957320
        **POPULATION sd (n)  0.006630295 -> d = 0.747677 -> 0.748**
        SAMPLE sd (n−1)      0.006730004 -> d = 0.736600 -> 0.737

    **A PRODUCER RE-DERIVING IT MUST GUESS THE DENOMINATOR, AND GUESSING
    WRONG MISSES BY 0.0111 — 222 TIMES THE 5e-5 TOLERANCE — WHILE BEING
    ENTIRELY CORRECT ABOUT G.** The gate is unrunnable not because the
    number is wrong or foreign but because **reproducing it requires a
    convention nobody wrote down.**

    **AND A NEAR-COINCIDENCE MADE THIS LOOK LIKE MISATTRIBUTION, WHICH IS
    RECORDED SO NOBODY REPEATS THE INFERENCE:** D3b's
    `mean_abs_z_weighted = 0.747691` sits **13.9 MILLIONTHS** from G's
    0.747677 and is a POOL-EXTREMITY REGRESSOR — a completely different
    quantity. A sweep for the value finds D3b and concludes the number came
    from there. **It did not. A search by VALUE cannot establish
    provenance, and this registration will not assert that another
    registration's headline figure was a transcription error.**
    **D2's FIGURES ARE REAL BUT UNDER-SPECIFIED.** `val_extrem D =
    0.015246` and `dom_extrem D = 0.016241` sit at **`per_t = 0.00` of a
    six-point sweep** (0.00 … 0.20) whose values move (dom: 0.016241,
    0.016540, 0.014649). Q named no threshold, so a producer choosing
    another would fail the gate correctly and for the wrong reason.
    **AND NEITHER RUNS ON Q's POPULATION.** D2's minimal-pair set is **632**
    after its own collapse rule; Q's is **684**. G permutes **34** clusters
    over 100,000 draws; Q's floor gives **33**.

**SO "RE-DERIVE G AND D2" NEVER MEANT "CHECK Q's MACHINERY ON SHARED DATA".** It meant Q reconstructs two other registrations' populations, collapse rules and permutation machinery end-to-end — **none of which appears in any of Q's four arms.** A gate whose only implementation exists to serve the gate is unexercised by every other path, which is the defect class that produced a dry run never reaching its writer and a key census reading one key.

    **THEY FAIL ON CORRECTNESS AND NOT ON COST**, and the removal is
    recorded here rather than performed silently. **What is lost is a
    cross-registration consistency signal** — and it was never available
    anyway, because 684 against 632 is a KNOWN divergence the check would
    have had to be calibrated to before it could mean anything.

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
