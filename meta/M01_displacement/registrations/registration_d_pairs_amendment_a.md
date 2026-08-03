# Registration D — Amendment A: the population, and the clause that did not name its own anchor

**STATUS: DRAFT. Nothing is in force. No D quantity has been computed on any
population, and none may be until this is countersigned and frozen.**

    AMENDS      registration_d_pairs_v6.md @ 8375ff4c8335d979
                (content unchanged since 2026-08-01 12:56; the 2026-08-03 08:14
                commit is a path move into registrations/, same hash)
    RULED       [3243], on custody's gate-1 resolution at [3241]
    FORM        AMENDMENT, not v7 -- deliberately. v6 is what a producer
                hash-gates on, so its bytes must not move. A v7 would be a
                SECOND full statement of one registration, free to drift from
                the first; the folder's own law is that two assertions of one
                fact can disagree, and the stronger claim wins by being the
                only one ([3156]/[3157]).

**Everything v6 governs and this does not name is UNTOUCHED** -- §D1's paired
statistic, §D2's sign-flip null, §D3's grid and reading rule, §D4's pre-freeze
gate, §D5's inherited terms, §D6's diagnostics, §D6b's four arms.

---

## PREAMBLE: TWO TRAPS THIS AMENDMENT'S VERIFIERS WILL MEET

Placed first because they are live for anyone checking the numbers below.

**1. `grep 684 README.md` FINDS A DOCKET ID, NOT THE PAIR COUNT.** `[684]` is a
message reference. **Anyone confirming the population by grep will find it and
be reassured by an unrelated string.** Booked at [3241].

**2. ELEVEN MANIFESTS READ ZERO WHEN THE KEY IS GUESSED.** Custody's first count
of the survivor manifests returned 0 pairs across all files, because the key is
`passed` and not `survivors`/`pairs`. **It was not posted** -- *"a zero I
produced myself is the one number this campaign has taught me to distrust on
sight"* ([3241]). The correct figure came from reading ONE file's structure
rather than guessing eleven.

---

## §A1 THE POPULATION -- DECLARED, NOT INHERITED

    POPULATION   the 684 pairs of the M01_PAIRS conjunction, evaluated on
                 data/prompt_categorisation.json @ sha256[:16] f441a8b2b12a8208

    PINNED AS AN ARTIFACT, not as a description ([3264]):
                 results/population_d_684.json
                 id_set_sha256_16   3ed3e286e633c2fc
                 n 684, distinct 684

    **The id set was ENUMERATED BY RUNNING the frozen producer's own
    `m01_pairs()`, never transcribed from its clauses.** A population described
    in prose is a second statement of the rule and free to drift from it; the
    quotation below exists to let a reader SEE the rule, and the artifact above
    is what any consumer must resolve against.

    THE CONJUNCTION, quoted from the FROZEN producer that selected the F/G
    corpus (within_pair.py @ d52c8000028b298f, lines 115-120), not restated:

        r["pair_role"] is truthy
        AND r["contrast_type"] == "transgressive_swap"
        AND str(r["source"]).startswith("M01_PAIRS")
        AND the pair retains EXACTLY TWO roles after grouping

**THIS IS BYTE-FOR-BYTE THE F/G CORPUS**, on RH's standing word of 2026-08-03:
*"the same minimal pairs as the M01 runs."* The displacement sign-test
(Registration F + Amendment A) and the magnitude read (Registration G) ran on
this population and no other.

**THE UNIT REMAINS THE PAIR**, as v6 §D0 declares. This amendment changes WHICH
pairs, never what a unit is.

### §A1.1 The set relation, from custody at [3241]

    round-1 survivor manifests   covert 35 + sexual 36 + threat 39
                                 + unarmed 39 + weapons 39          = 188
    EXCLUSIONS.json              drafted 200 - retired 12           = 188  EXACT
    later rounds                 animal 50 + betrayal 102
                                 + power_r2b 120 + property 104
                                 + taboo 120                        = 496
    TOTAL                                                             684
    F/G conjunction                                                   684  EXACT
    distinct pair ids across all manifests                            684  no
                                                                      double-count

**188 is a STRICT SUBSET of 684. The delta is 496 pairs from five later drafting
rounds, and the F/G reads used 3.6x v6's round-1 figure.**

### §A1.2 THE PROVENANCE OF THE 496 — ONE COMMISSION, NOT FIVE

**An earlier draft of this section asked custody for FIVE commission ids and made
the fill a freeze condition. There are not five. There is ONE, and the demand was
my inference from five MANIFESTS, stated as a requirement.** Custody could not
fill it and said so rather than filling it ([3256]) — **the blank did the finding
it was left for, and the finding was that its premise was wrong.**

    COMMISSION   [1662], 2026-07-31 19:11 — round 2, EIGHT domains
    [1666]       19:27 — "~40 SURVIVORS per domain across 8 domains
                 (3 proven-yield ... 5 NEW: animal cruelty, theft/property,
                 intimate betrayal, power/coercion, desecration/sacred taboo)"

    THE FIVE NEW DOMAINS ARE THE FIVE LATER MANIFESTS:

      animal cruelty            survivors_animal.json          50
      theft/property            survivors_property.json       104
      intimate betrayal         survivors_betrayal.json       102
      power/coercion            survivors_power_r2b.json      120
      desecration/sacred taboo  survivors_taboo.json          120
                                                              496

**THE THREE HUMAN-VICTIM DOMAINS REFUSED** (domestic violence, public/stranger
violence, sexual coercion — [1666]). **Eight commissioned domains yield five
manifests because three are DOCUMENTED REFUSALS, not gaps.** A reader counting
domains against manifests must have this or the shortfall reads as missing data.

**`power_r2b` IS A REPAIR INSIDE [1662], NOT A SIXTH ROUND.** Power was one of
the eight, failed 0/120, was flagged for a redraft decision at [1873]
(2026-08-01 00:45), reviewed at [1890] (04:46), and passed 120/120. **A repair
inside a commissioned round is that round.**

---

## §A2 TWO AMBIGUITIES IN ONE CLAUSE, BOTH RESOLVED EXPLICITLY

v6 §D0 reads: *"the 188 of `pair_drafts/EXCLUSIONS.json` PLUS the **~160-pair**
drafting round commissioned at **[1662]**, and any further round RH commissions
BEFORE **the read**."*

### §A2.1 `~160-pair` VERSUS `[1662]` — which term identifies the round

**THE CLAUSE CARRIES BOTH A SIZE AND AN ID, AND THEY DISAGREE. So does the
commission, with itself:**

    [1662] 19:11   "THE DRAFTING ROUND IS COMMISSIONED AT ~160 PAIRS"
    [1666] 19:27   "~40 SURVIVORS per domain across 8 domains"     => 320
    artifact       496 survivors from the five domains that proceeded

**Three numbers, sixteen minutes and one artifact apart, no two of which agree.**
`~160` is out by 2x against its own commission's restatement before the artifact
is out by a further 1.5x. **A reader trying to identify this round BY ITS SIZE
cannot.**

**RESOLVED: `[1662]` IS OPERATIVE. `~160-pair` IS A DESCRIPTIVE ESTIMATE AT
COMMISSION TIME AND CARRIES NO POPULATION FORCE.** The round enters at its actual
yield of 496.

**WHY BY RULING AND NOT BY READING.** Taking the id and waving the number is
almost certainly the drafter's intent, **and it is still an interpretation, and it
is the PERMISSIVE one — it admits 3.1x the pairs the clause names** (188 + ~160 =
~348 against 188 + 496 = 684; 336 pairs, 49% of the corpus). The house rule is
[2879].4, applied at [3243] to this very registration one clause later, at 3.6x:
**an ambiguous registration is AMENDED BEFORE USE, never interpreted at
implementation time — doubly so when the convenient reading is the permissive
one.** A producer resolving this by silence would be making a population decision
in code.

### §A2.2 "THE READ" — resolved, though the facts make it moot

    RH's CONSTRUCT read (per-round)                                      -> 188
    D's DATA read (has NEVER occurred, so every round trivially precedes) -> 684

**RESOLVED: "the read" means D's DATA READ, and none has occurred.**

**AND THE TIMELINE MAKES THIS UNNECESSARY IN FACT WHILE LEAVING IT NECESSARY IN
LAW.** Every round completed BEFORE D FROZE:

    2026-07-31 19:11   [1662]   round 2 commissioned, 8 domains
    2026-08-01 00:45   [1873]   round-2 audit closes; power flagged for redraft
    2026-08-01 04:46   [1890]   the power redraft reviewed
    2026-08-01 12:56            D v6 FREEZES -- after all of it

**BEFORE-THE-FREEZE IS STRONGER THAN ANY READING OF "THE READ"**, and it is
custody's finding ([3256]), preserved here because a future reader is owed the
fact and not only the ruling.

**FORWARD FORCE, UNCHANGED FROM v6's INTENT: growth after D's data read is a new
registration.** This amendment fixes the population at 684 and closes it.

---

## §A3 A LIVE HAZARD IN THE MANIFESTS -- named, not repaired

    audit/manifests/survivors_power.json      passed = []   SUPERSEDED
    audit/manifests/survivors_power_r2b.json  passed = 120  the live round

**The power round was re-drafted and the superseded manifest is still on disk.**
`passed` is a LIST of pair ids; the superseded one is EMPTY. (Custody reported
this as `passed = 0` at [3241]; the artifact holds `[]`. Same arithmetic, and the
field is a list -- stated exactly because a count is not the field it counted.)

**ANY SUMMATION OVER THE MANIFESTS GETS THE RIGHT 684 BY LUCK.** The empty list
contributes nothing, so the total is correct **for the wrong reason**, and stays
correct **only until a superseded manifest is non-empty.**

**NOT REPAIRED HERE.** Deleting or editing a superseded artifact is the shape
this campaign has refused all week ([3179], [3162], [3226]). The manifest is
evidence of a re-draft. **The declaration sits beside it: a consumer of these
manifests must select by the LIVE round name, never by summing the directory.**

---

## §A4 UNIT-SENSITIVITY REPORTING -- gate 3, prospective

v6's unit is THE PAIR, declared before 2026-08-03's base-checkpoint rulings.
**It runs AS FROZEN.** This amendment does not change it -- any unit change is a
further amendment argued blind, never an edit.

**The read reports the registered readout AND, beside it:**

    entries / bases / lineages     counted and NAMED, each beside the field it
                                   counted -- the [3067] lesson, applied before
                                   it can bite rather than after
    n_distinct                     printed WITH ITS FIELD NAME
    the unit assertion             IN THE PRODUCER, re-deriving the expected
                                   unit set independently rather than asserting
                                   a property of the input to itself

**The last clause is written from a defect in this campaign's own code:** F's
collapse producer carried a unit assertion that could not fail, because it
compared a dict's keys to themselves. **A guard that cannot fail passes.**

---

## §A5 WHAT THIS AMENDMENT IS NOT

**NO D QUANTITY HAS BEEN COMPUTED, on 188, on 684, or on any subset.** Custody
holds the wall ([2072]) and confirms it. This is argued from set relations,
manifest counts, file hashes and the text of v6 -- **nothing here required
opening a valence, arousal or dominance value.**

**IT DOES NOT TOUCH:** the paired statistic, the null, the threshold grid, the
floor, the primary point, the sidedness, the reading rule, the collapse clause,
the qualification bar, the benchmark, the seeding, or the four arms. **Those were
frozen 2026-08-01 12:56 and are the reason a population amendment is safe to make
now** -- every choice this data could have biased was fixed before anyone saw it.

**IT GRANTS NO EXEMPTION FROM THE EXPOSURE DISCLOSURE AT [3242].** That gate is
separate, is about the seats and not the population, and is ruled at [3245].

---

## §A6 FREEZE CONDITIONS

    1. DISCHARGED -- §A1.2's provenance is supplied ([3256]) and the demand it
       replaced was wrong: ONE commission, not five.
    2. malign countersigns -- the population artifact `3ed3e286e633c2fc`, the
       set relation, and this amendment's hash
    3. pen freezes; the hash is posted BEFORE the producer is written
    4. THEN gate 2 (exposure, [3242]/[3244]), gate 3 (§A4), then the producer

**ONE HISTORY NOTE, KEPT DELIBERATELY.** This amendment was ruled unnecessary at
[3258] and that ruling was WITHDRAWN at [3261] — the pen had composed against
custody's [3256] while custody was already withdrawing it. **Both seats flagged
it independently before anything was built on it.** Recorded because a reader
finding [3258] in the docket must be able to see that it does not stand.

**Nothing runs before all four. The population runs ONCE (v6 §D0, [1297].3 /
[1324].1) and this amendment does not soften that.**

---

## §A7 THE MDE CONVENTION — declared where v6 omitted it

**v6 USES `MDE` SIX TIMES AND DEFINES IT NOWHERE.** Not in §D5's inherited list,
not in `m01_registration_c3.py` or `_b.py`. **§D6e says the realized MDE "is what
converts any of this into a verdict"** — and §D6d turns a null into either
*evidence the effect is MOVEMENT-GENERAL, quotable as such* or *UNINFORMATIVE AT
THIS POWER, quotable as nothing*, on the comparison `MDE < the dimension's known
effect size` (arousal ~0.10, valence-extremity 0.025, dominance-extremity 0.025).

**The right-hand sides are declared to three digits. The left-hand side was a
bare acronym.** MDE at 80% power and at 90% differ by roughly 30% on the same
data; against a comparator of 0.025 that moves nulls between QUOTABLE and
NOTHING, **with both implementers following the registration.**

Found by custody's omissions pass **before any code existed**, and refused rather
than resolved there — *"any value I pick is an author's choice wearing an
implementer's hat."* Ruled at [3276].

### §A7.1 THE CONVENTION — inherited, not invented

    MDE   the effect detectable at 80% POWER at the arm's declared
          ONE-SIDED alpha = 0.05

**This is the campaign's standing convention, not a new choice.** Registration G
§8 declares it verbatim on THIS corpus (`n = 34, one-sided alpha 0.05, power
80%`); the same 80%/.05 priced F's design and the M04 charter's tables; two seats
re-derived G's figure by DIFFERENT methods — closed form and Monte Carlo — to
0.005 agreement ([3022], [3127], G §8).

**Chosen blind: no D quantity exists at any seat, and the convention is
outcome-independent machinery.**

### §A7.2 THE SCALE, WHICH THE CONVENTION ALONE DOES NOT FIX

**G's MDE is a STANDARDISED effect (`d = 0.426`). §D6d's comparators are RAW
dimension units. A standardised MDE cannot be compared to a raw effect size.**

**DECLARED: D's MDE is RAW-SCALE, in the arm's own dimension units**, obtained by
simulation at the arm's realized pair-count and variance — which is what makes it
commensurable with 0.025 and ~0.10.

### §A7.3 TIMING — the ordered step that makes it a pre-registration

**The raw MDE cannot be stated now: it depends on a realized variance nobody has
computed.** G met this exactly and it is why *"d 0.748 against a pre-registered
MDE of 0.426"* means anything:

    1. open the data
    2. DERIVE the raw MDE at realized n and variance
    3. WRITE IT INTO THE RECORD
    4. THEN compute the primary statistic

**Without step 3 preceding step 4, §D6d is a verdict rule whose threshold is set
after the verdict is visible.**

### §A7.4 **WHAT IS STILL OPEN, AND MUST BE DECLARED BEFORE THE READ**

**"By simulation" is a method, not a procedure, and the choices below remain.
Naming them here rather than inventing them, because an implementer resolving
them at read time reopens exactly the gap this section closes, one level down:**

    the null under which power is computed   (sign-flip on observed magnitudes)
    how the alternative is injected          (constant shift? scaled?)
    the search for the detectable effect     (bisection tolerance, bracket)
    simulation draws, and the seed
    which realized variance                  (per threshold point, or primary only)

**These are producer-level and outcome-independent, so @malign may declare them
in the [3269] form — pinned reading or argued discretion — BEFORE writing the
line. They are not the pen's to rule unless he finds one that is a design
choice, as he did with the convention itself.**

---

---

## §A8 THE FIELD BOUNDARY AGAINST C v6 §C0 — which parent fields survive

**D v6 governs "by delta" on `registration_c_delta_v6.md` `06f0272d7f21b901`,
"which governs everything not named here." §A1 overrides ONE field. Which of the
others survive has been a producer's assumption and is now a fact about a
document.**

**THIS SECTION EXISTS BECAUSE TWO SEATS NEARLY FILED FALSE ABSENCE CLAIMS ON THIS
CLAUSE NINETY MINUTES APART** — once that the movement rule was undeclared, once
that the edge set was. **Both are in §C0, one comma apart.** *An absence claim is
only as wide as the search behind it*, and §C0's POPULATION line packs SEVEN
distinct commitments into one sentence.

### §A8.1 THE TABLE

    §C0 FIELD                          STATUS       WHERE
    prompts 959 @ fd3f14796ba9481b     OVERRIDDEN   §A1: the 684 pairs / 1,368
                                                    prompts. Different corpus.
    models 95 @ e4c507eb8dbcf593       RETAINED     *** see §A8.2 ***
    en only                            RETAINED     the pairs are en throughout
    base -> MOST-ALIGNED ARM (edge)    RETAINED     35 edges, `operation_edges`
    CANONICAL (movement rule)          RETAINED     min_prob .003, fall_ratio .5,
                                                    delta .003
    function words excluded            RETAINED
    lemma repair applied               RETAINED
    z anchored to source database      RETAINED     consistent with D §D0 ORIGIN
    >= 3 rated words of the role       RESTATED     D §D1, "as v6" -- same value,
                                                    stated in both

    §C0 RESIDUAL                       SUPERSEDED   D §D0 + §D6b pin it PER ARM;
                                                    §D6b says "every readout
                                                    twice" does NOT transfer to
                                                    the arousal arm
    §C0 SIDEDNESS                      RESTATED     D §D0, one-sided fixed ONCE
                                                    for the whole curve
    §C0 ORIGIN                         RESTATED     D §D0, database mean
                                                    *** but see §A8.3 ***

### §A8.2 **THE ROSTER IS A RULE, NOT A SET — AND ITS FROZEN DIGEST CANNOT BE MATERIALISED**

**An earlier draft of this cell framed this as "95 versus the corpus's 103, RH's
call." THAT FRAMING IS WRONG AND IS RETRACTED. Neither number is what the rule
returns.**

    re-derived 2026-08-03, TWO SEATS, matching to the digit:
      prompts  2,579   sha e73a57d399a2b0c6   vs frozen fd3f14796ba9481b  DRIFT
      models     101   sha f989b0789dd8af51   vs frozen e4c507eb8dbcf593  DRIFT

**`m01_concentration.py`'s own docstring is why: *"Re-derive from the RULE and
verify the digests. NEVER READ FROM A STORED LIST. A population frozen as a COUNT
goes stale."* THE RULE IS THE ARTIFACT AND THE DIGEST IS THE CHECK ON IT.**

**THE ROSTER RULE IS `models that cover EVERY ACTIVE prompt`, so the model set is
a FUNCTION OF THE PROMPT SET — and D's own 1,368 pair-prompts entered this store
after C froze.** The population D reads changed the roster D is read over. **That
is not decay; it is the same rule answering a bigger question.**

**RULED [3299]: "RETAINED" MEANS THE RULE, NOT THE SET.** D re-derives, takes what
the rule returns, and **RECORDS THE DRIFT**. The frozen 95 is a historical count
of a set that no longer exists and cannot be reconstructed — the store state that
produced it has been cleared and re-ingested twice since.

**AND THE ROSTER RULE COUNTS PRESENCE, NOT SCORING.** This morning the rule would
have returned 99: the two Falcon-H1 checkpoints covered every prompt with cells
that were entirely NaN, and `iter_keys` counts an empty cell as present. **The
repair changed the roster without changing the rule.** Recorded because a reader
comparing rosters across dates needs it.

### §A8.2b THE ENFORCEABLE FORM OF "RECORDED" — ruled [3303].2

**`frozen_population()` CONTAINS NO REFUSAL.** It returns `(prompts, models,
hashes, drift)` and the caller must consult the fourth value. Of its callers, six
check and refuse; `m01_concentration.py` deliberately does not and documents why
(an emitter, not a measurer); **`m01_prompt_adjusted_control.py:64` binds it to
`_d` and never looks, with the drift live today.**

**REQUIRED OF D's PRODUCER:**

    bind `drift` BY NAME -- never `_`, on that tuple slot
    write it into the result JSON as a FIRST-CLASS FIELD
    print the re-derived counts and both digests beside the frozen ones

**"Recorded" means a field a reader MEETS, never a value the producer received
and dropped.** A guard that reports and proceeds is only as strong as the weakest
caller's naming convention, and one underscore has already defeated it once.

### §A8.3 ONE FIELD THE TABLE COULD NOT CLOSE

**§C0's ORIGIN carries a clause D's §D0 does not restate:** *"Sensitivity at the
scale midpoint (z -0.0501), which must differ from the primary or the producer
raises."*

**D §D0 declares the origin and is silent on the sensitivity.** Under the
delegation clause a field "not named here" survives — **but D DID name ORIGIN, so
whether it restated the field and dropped the rider, or restated the headline and
inherited the rider, is not determinable from the text.**

**NOT RESOLVED HERE. Routed to the pen as the §A2-class question it is: an
ambiguity found before use, amended rather than interpreted at implementation
time.** A producer that silently includes or omits that sensitivity is making the
call in code.

---
