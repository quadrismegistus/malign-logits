# M01 — what remains, and what should not be done

**STATUS AS DECLARED (2026-08-03 UTC): a working triage, not a registration.
Nothing here is in force. Assembled from the clause ledger in `ledger.md` (was `README.md` before the 2026-08-03 split), the
registrations in `registrations/`, and the results in `results/`.**

---

## THE SHAPE OF THE CAMPAIGN, because two layers get conflated

**LAYER 1 — THE CLAUSE LEDGER** (`ledger.md`), ten clauses, F-findings, mostly
run on the FROZEN GENERAL ROSTER (959 texts at freeze; the live rule now returns
2,579).

    1  mass-migration        PENDING
    2  null-survival         VERIFIED
    3  concentration         VERIFIED      general roster, 959x93-95
    4  recipient-agreement   UNRETESTED
    5  direction-agreement   MEASURED      general roster
    6  faller-riser-relation VERIFIED      128 pairs, two annotation passes
    7  slot-sensitivity      PARTIALLY VERIFIED
    8  liminal-targeting     F40 unaudited
    9  stage-share           UNREPRODUCED
    10 acquisition-order     PENDING

**LAYER 2 — THE REGISTERED CAMPAIGN** (`registrations/`), run on the 684-PAIR
CORPUS except where noted.

    B   arousal                     CLOSED (null)      959 general roster
    C   role-membership x norms     READ               959 general roster
    D   signed valence drop         READ
    D2  extremity                   CONFIRMED both arms
    D3b pool-availability confound  READ -- confound does NOT explain D2
    F   within-pair displacement    RATE NULL at the family unit (p 0.148
                                    collapsed; 0.023 raw)
    G   magnitude, by MASS          CONFIRMED (0.169, p 6.0e-05); secondary null

**BOTH AXES HAVE A PAIRS VERSION AND A GENERAL VERSION.** Movement: clauses 3/5
general, F/G on pairs. Word norms: C general, D/D2/D3b on pairs.

**AN EARLIER DRAFT OF THIS FILE SAID THE GAP WAS COMMENSURABILITY. IT IS NOT** —
C and D2 run the same estimator, by import (see §Step 1). **And the gap is not
coverage either. The campaign's questions about degree and about word norms are
BOTH ANSWERED ON BOTH POPULATIONS.** What is thin is the MECHANISM layer —
clauses 4, 7, 9, 10 — which is Step 3, and which no amount of re-measuring the
answered questions will touch.

---

## STEP 1 — ~~THE C-VS-D2 CONTRAST~~ **STRUCK. The question it asked is already
answered, and by a better instrument.**

**RUN AND ANSWERED: C's `A` and D2's `A` ARE the same statistic.** Not merely
commensurable — `scripts/pairs_d.py:arm_values()` does `import
m01_registration_c3 as C3` and computes every cell as `C3.A_and_terms(vals, ws,
rs)`; its docstring says *"the inherited estimator, never re-derived."* Both are
`wmean(FALLERS) - wmean(RISERS)`, weights `|delta|`, uncentred. Two differences
remain if anyone revives this: C residualises LINEARLY on arousal, D2's extremity
arms QUADRATICALLY; and C aggregates per cell where D2 aggregates per member.

**BUT THE CONTRAST SHOULD NOT BE RUN, because D2 already answers site-specificity
and answers it better.** The premise this step rested on — that a pair's UNMARKED
member is "transgressive-adjacent" rather than neutral — **IS FALSE. It was
asserted repeatedly by this seat and never checked. The members read:**

    MARKED    She stirred the crushed sedative into his coffee and
    UNMARKED  She stirred the crushed cinnamon into his coffee and

    MARKED    He slammed the puppy's cage door shut and
    UNMARKED  He latched the puppy's cage door shut and

**The unmarked member is an ordinary sentence, matched on syntax, length, topic,
register and slot position — everything except the transgressive element.** For a
CAUSAL CONTRAST ("is the effect larger where the transgressive element is
present, holding the rest fixed") that is the OPTIMAL instrument, not a deficient
one. A corpus comparison varies all of those at once.

**AND THE 959 IS NOT A "GENERAL" BASELINE ANYWAY:** 39% transgressive by domain
(violence 221, sexual 89, plus profanity/substance/death/power), **41%
cross-lingual**, and only 148 of 959 carry a `pair_role` (70 MARKED / 78
UNMARKED). It was assembled at many points for many purposes. It answers "in this
accumulation," not "in general."

    SITE-SPECIFICITY          ANSWERED.  D2, +0.0151, both arms; D3b refutes
                              the pool-availability explanation (slope negative
                              on all four regressors).
    ANY EFFECT AT NEUTRAL     Not answerable from D BY CONSTRUCTION -- D tests a
      SITES AT ALL            DIFFERENCE against a null and has no null for
                              either LEVEL.  Already answered elsewhere: F18
                              (compression predicted by base entropy, not
                              transgressiveness), F19 (uniform across all 9
                              content categories), D4b (argmax flips at 23% of
                              unselected fiction slots).

---

## STEP 1b — THE ONE THING THAT WOULD STILL ADD SOMETHING. Small, optional.

**The steelman of the objection, in the only form that survives:** a minimal pair
controls syntax but NOT PRAGMATIC CONNOTATION. "She stirred the crushed cinnamon
into his coffee and" may still cue poisoning, because the FRAME is suspicious with
the transgressive word removed. If so, `A` at unmarked members is elevated
relative to unremarkable prose, and **D2's +0.0151 is a FLOOR, not an estimate.**

    TEST   compare A at the 684 UNMARKED members against A at the 97 LITERARY
           sites.  Same estimator, both on disk, no new inference.
    READ   unmarked at the literary level -> the frame is clean, D2 stands as
           the site-specificity answer
           unmarked ABOVE it -> the worry is real and QUANTIFIED, and D2's
           effect is a lower bound

**This converts a framing disagreement into a number, which is the only reason to
spend anything on it.**

---

## STEP 2 — D4c CROSS-FAMILY REPLICATION. Everything already on disk.

**D4c (in `docs/discovery_agenda.md`, ungraded, one family) found: alignment
degrades fit to literature, dose-dependently.** Base 33.7% -> aligned 32.4% at
matching the novelist's actual next word (McNemar exact p 0.016), and **-8.8pp in
the top decile** where alignment acts hardest.

    population   97 prompts: domain=literary AND status=ACTIVE.  The same 97 by
                 every route -- active status, full store coverage, gold word
                 present.  `literary_101` is excluded twice (domain=other AND
                 already RETIRED); 4 more are RETIRED.
    gold words   `next_actual` in data/d4_fiction_sites*.json, 97/97, extracted
                 under the declared 16-word-slot rule
    grid         97 x 44 base->aligned edges, both ends covered on all 97.
                 4,268 cells, no missingness.
    known answer THE LLAMA EDGE IS ONE OF THE 44.  The argmax computed from the
                 store must equal d4's recorded base_top/aligned_top on all 97.
                 NOTHING ELSE RUNS UNTIL THAT REPRODUCES 97/97.

**Quantities that must be named before this runs** (the class that cost a day):

    word normalisation   does `next_actual` match the store's `word` form?
                         'city.' vs 'city', casing, the dict_sha dictionary
    argmax               highest p in the retained rows, and the tie rule
    dose axis            D4c's headline is the DECILE result; needs an
                         entropy-controlled base-vs-aligned divergence per cell,
                         computable over retained mass only, with a residual
    unit and clustering  edges are NOT independent (Llama is the base for tulu,
                         tulu-no-safety and three tulu-sft variants).
                         Family is the cluster.

**Blindness:** the direction is known from D4c, and this seat has additionally
seen an exploratory base->aligned measurement on these prompts. **Register it as a
directional replication with a pre-specified prediction, adjudicated by a seat
that has not seen those numbers.**

**Bias entry owed:** these are 20th-century novels (*Animal Farm*, *Return of the
Jedi*) and several models were plausibly trained on them. Memorisation inflates
match in both arms, not necessarily equally.

---

## STEP 3 — TRIAGE THE MECHANISM CLAUSES. A decision, not an experiment.

**Four clauses carry the campaign's theoretical weight and none is closed.**

    9  stage-share       URGENT.  "Alignment installs almost entirely at SFT" is
                         a chapter-level claim whose number does NOT reproduce;
                         seven candidate causes eliminated; THE CAUSE IS
                         UNLOCATABLE BECAUSE THE PRODUCER WAS NEVER COMMITTED.
                         This is a DEBT, not an open question.
                         -> rebuild the producer, or retire the claim.

    7  slot-sensitivity  Highest theoretical return of the four.  Needs the
                         stratified annotation.  Medium cost.

    4  recipient-agreement  Do families converge on the SAME substitute -- the
                         strongest form of "displacement is structured, not
                         noise."  Needs re-running on v3.  Medium.

    10 acquisition-order Repression before displacement.  Needs training
                         checkpoints.  Expensive.  Scope as a declared limit
                         unless funded.

---

## STEP 4 — ONE GREP AGAINST THE DRAFT. Cheap.

Clause 6 is VERIFIED, but its 128 items were **drawn under the DRAW rule**
(gain >= 0.003, no renormalisation-null test). Any riser-status sentence in the
draft must cite the draw rule, not the null. Check which the draft leans on.

---

## WHAT SHOULD NOT BE DONE

**DO NOT RE-MEASURE MOVEMENT ON THE LARGER ROSTER.** Clauses 3 and 5 are the
campaign's most solid results, at 959x93-95. Re-running at 2,579x101 is a
RE-MEASUREMENT, and clause 3's own history is the warning: the last one returned
different numbers, forced a re-scoping, and ended with "30-41% is UNQUOTABLE as a
current number." Real compute spent to destabilise something VERIFIED.

**DO NOT re-run word norms on the pairs or movement on the pairs.** Both exist
(D/D2/D3b; F/G). The gap is commensurability, which is Step 1 and costs nothing.

---

## HOUSEKEEPING — RESOLVED, recorded so it is not re-raised

`registration_f_within_pair.md`, its amendment, and `registration_g_magnitude.md`
appear at this folder's root AND in `registrations/`. **They are SYMLINKS**
(`lrwxr-xr-x`, created 2026-08-03 08:01 with the reorganisation), same inode,
same hash — compatibility shims so citations of the pre-reorg paths still
resolve. **Not duplicates. Nothing to fix.**

---

## FLAGGED — A FIGURE IN THE NEW `README.md` IS ATTACHED TO THE WRONG POPULATION

The reader's map states, of the general corpus: *"a third of its non-pair
remainder is cross-lingual, a sixth deontic-framed; the pairs contain neither."*
**Those are this seat's figures from [3571], computed on the 1,211 (LIVE roster
minus pair members). They do not describe the 959, which is what clauses 1-5 and
C actually ran on:**

    the 959 itself            41.0% cross-lingual     3.5% deontic
    live roster minus the 959  0.2% cross-lingual    11.1% deontic

**The deontic claim is off by roughly a factor of five for the 959.** The deontic
prompts arrived AFTER the freeze -- they are the institutional/M03 work -- so
attaching them to the 959 inverts the history. The cross-lingual figure is, if
anything, understated.

**Owed to the docket, not yet posted.** Also owed: the correction that C's one
surviving blind arm (`valence/signed/GENERAL`) is confirmatory on the PAIRS
population, not on the 959 -- its blind table reads *"H1 GENERAL, this population
SEEN by lacan ([1526])... Confirmatory on the PAIRS population."*
