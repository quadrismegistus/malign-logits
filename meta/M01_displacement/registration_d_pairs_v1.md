# Registration D — H1 role-membership on the PAIRS population

A DELTA on frozen `registration_c_delta_v6.md` `06f0272d7f21b901`, which governs
everything not named here. Drafted per docket [1591] by the two seats that have
never seen the withheld value: malign (blind by protocol) and the pen (blind by
role). **Nothing in this document was informed by data; the population does not
yet exist.**

---

## §D0 THE DECLARED QUAD — the freeze gate answers first

    POPULATION   the post-construct-read survivors of the 188-pair pool
                 (`pair_drafts/EXCLUSIONS.json`, 200 − 12) that clear the
                 displacement bar. THE BAR IS RH's AND IS SET BEFORE THE RUN.
                 Cell qualification as v6: >= 3 rated non-function words IN EACH
                 ROLE. Strata (displacing / control / gap) carried from v6.
                 THE POPULATION RUNS ONCE ([1297].3 / [1324].1): it is frozen at
                 its hash before any join, and a grown pool is a new
                 registration, never a re-run of this one.

    RESIDUAL     a bin, as v13. Every readout reported TWICE — raw and
                 AROUSAL-RESIDUALISED, GLOBAL fit over this population's own
                 qualifying set, never within cell.

    SIDEDNESS    ONE-SIDED, in the registered direction, for the primary arm.
                 Exploratory readouts report the one-sided fraction of nulls
                 below the observed value and carry NO directional claim.

    ORIGIN       the database mean (z anchored there), the same origin wherever
                 an origin is used. Signed valence needs no extremity origin,
                 but the quad is declared whole so the gate can grep it.

---

## §D1 THE HYPOTHESIS — fixed, confirmatory, one arm

**RH's H1, unchanged in words from [1530]: NEGATIVE-VALENCE WORDS FALL, POSITIVE-
VALENCE WORDS RISE.** A role-membership claim, so the GENERAL arm is primary:

    PRIMARY      A_valence = wmean(fallers) − wmean(risers), SIGNED, UNCENTRED,
                 |delta|-weighted, on the arousal-RESIDUALISED variable
    DIRECTION    NEGATIVE (fallers below risers). One-sided lower.
    NULL         the MEMBERSHIP null — permute the faller/riser label within a
                 cell, holding each word's value and both role sizes fixed.
                 NOT the mass-order null, which holds membership fixed and is
                 blind to this hypothesis ([1541]/[1543]).

**The TOP-MOVERS arm is reported as SECONDARY, exploratory, under the mass-order
null, with no registered direction.** It is a different question — whether the
biggest movers skew further — and it did not survive §7(a) on the norms
population ([1574].3).

---

## §D2 THE SIGNIFICANCE LEVEL, NAMED IN THE SPEC

**alpha = 0.05, one-sided, on the primary arm.** Stated here because on the norms
population an implementer's conventional default became an adjudication: one
stratum's conjunct verdict held at 0.05 and failed at 0.10 ([1589]). **No level
is left to code in this registration.**

**A p between 0.05 and 0.10 is reported as NOT SUPPORTED, with its value, and is
not re-described as a trend.**

---

## §D3 INHERITED MACHINERY — cited, not restated

From v6 and its parents, unchanged and NOT re-litigated here:

    norms          Warriner `85f6d7e3`, V.Mean.Sum, human/exogenous
    residualise    SIGNED valence on arousal, LINEAR ONLY — measured b2 = +0.0005
                   in the correction direction, so no quadratic term ([1549]/[1550])
    benchmark      the induced value, PER STRATUM, computed from THIS population's
                   own A_arousal and its own fitted slope. NOT inherited from the
                   norms population — a benchmark is a population statement
                   ([1551].3).
    reporting      M_f, M_r, wmean(fallers), wmean(risers) beside every A. A
                   difference is not a direction until both terms are visible.
    §7(a)          control sites at chance. "Not significant" is not "~ chance."
                   If the control readouts move, the arm is QUARANTINED as
                   magnitude-salience, per the arousal and H1-top precedents.
    §C9 gap        the gap stratum PRINTS, is NEVER TESTED, and carries verbatim:
                   it CAN undercut displacement-specificity where §7(a) cannot
                   see; it CANNOT be evidence for or against the hypothesis; and
                   IF THE REGISTERED ARM IS NULL AND THE GAP IS NOT, THAT IS NOT
                   A FINDING. Cross-stratum reading on EFFECT SIZES ONLY.

---

## §D4 THE PRE-FREEZE GATE

**No arm freezes until its statistic has been shown to FIRE in a world where its
hypothesis is maximally true** — negatives fall, positives rise — **with the
null's mean far from the observation.** A reference that holds constant the thing
the hypothesis varies cannot test it, and this gate is the cheapest check that
catches it ([1543]/[1546]).

**The membership null's calibration sweep re-runs on this population's cell-size
mixture:** p uniform under H0, many draws, per-cell permutation count printed with
a floor-refusal. **A null with one attainable value is not a null.**

---

## §D5 SEEDING — specified here so the producer cannot reintroduce the defect

    rng_arm = default_rng([SEED, sha256_id(dim, variant, kind, stratum)])

**Per-arm, derived from the arm's own identity — NEVER from stream position.** On
the norms population a flag documented as print-only gated computation, so two
executions of the same seeded producer sat at different stream positions ([1578]).

**`sha256_id` MUST NOT use Python's builtin `hash()`: string hashing is salted per
process (PEP 456), so that fix would itself be irreproducible** ([1579].1).
Measured: three processes, three different values for one string.

---

## §D6 WHAT THIS DESIGN DISCARDS, DECLARED

**The 188 are MINIMAL PAIRS — a marked and an unmarked member differing by one
substitution.** Treating them as a displacement population **uses the members as
independent prompts and discards the pairing.** That is a real loss and it is
declared rather than hidden: the paired contrast this pool was authored for is a
DIFFERENT registration, and nothing here forecloses it.

**RH may prefer the paired design instead. That is a construct decision and it
belongs to him BEFORE this freezes**, not after a number exists.

---

## §D7 THE FALSIFIER, AND THE WALL'S DISCHARGE

**If the primary arm's residualised A is not negative past its per-stratum
benchmark at alpha = 0.05, H1 is NOT SUPPORTED on this population.** No magnitude
is predicted.

**ON COMPLETION OF THIS TEST — whatever it returns — THE WALL IS FULLY
DISCHARGED.** The withheld H1-general value and the §C6 first-conjunct verdict
enter the docket, and the conjunct re-enters under two-seat audit ([1584].3).
**The wall stays sealed exactly as long as the blind it protects, and no longer.**

---

## §D8 WHAT I HAVE NOT SEEN

**I have not read the H1-general value or the §C6 conjunct verdict on any
population.** I have computed Warriner lexicon properties and published arousal
figures only, and I ran the Registration C valence arms without
`--show-h1-general`, so that arm was never computed at this seat.

**This spec is authored blind, on a population that does not yet exist, and its
hypothesis is RH's in his own words.**
