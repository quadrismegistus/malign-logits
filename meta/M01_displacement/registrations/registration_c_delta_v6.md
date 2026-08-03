# Registration C — v6, the role-membership battery

Supersedes v4 `1408e5c170d188fd` (frozen [1511]) and the v5 amendment.
**v4 IS SUPERSEDED AS THE-WRONG-APPARATUS-FOR-THIS-HYPOTHESIS**, at zero data
cost: nothing ran on seven of eight arms. Base apparatus remains
`registration_b_spec_v13.md` `06186c42f9ff46e0` for everything not named here.

Provenance: [1520] halt, [1523] eight arms, [1526] blind, [1530] RH's three,
[1536] mass balance, [1540] confound inversion, [1543] positive-control gate,
[1547] stratification struck, [1549] curvature + per-stratum benchmark, [1551].

---

## §C0 THE DECLARED FIELDS

    POPULATION   frozen Reg-B population: prompts 959 fd3f14796ba9481b, models
                 95 e4c507eb8dbcf593, en only, base -> most-aligned arm,
                 CANONICAL. Function words excluded, lemma repair applied,
                 z anchored to the source database. >= 3 rated non-function
                 words of the role.

    RESIDUAL     a bin; PLUS arousal-residualisation, GLOBAL over the
                 qualifying population, NEVER within cell (a within-cell fit
                 leaves one df at n=3 and emits structural zeros that enter T
                 while counting in the denominator).
                 SIGNED arms    residualise on arousal          (b2 = -0.0000,
                                                                 measured)
                 EXTREMITY arms residualise on arousal AND arousal^2
                                                                 (curvature is
                                                                 9.8% / 9.0%)

    SIDEDNESS    one-sided directional per registered arm, in its registered
                 direction. Exploratory arms report the one-sided fraction of
                 nulls below the observed value and carry NO directional claim.

    ORIGIN       the database mean (z is anchored there), the SAME origin for
                 the extremity zero and for any composition print. Sensitivity
                 at the scale midpoint (z -0.0501), which must differ from the
                 primary or the producer raises.

---

## §C1 RH'S HYPOTHESES, VERBATIM, AND THEIR MAPPING

    H1  Negative valence words fall, positive rise
    H2  High (neg or pos) valence words fall, neutral rise
    H3  High (neg or pos) dominance words fall, neutral rise

**All three are ROLE-MEMBERSHIP claims: which words fall and which rise.** This
is why v4's within-role centred apparatus could not test them — centring removes
a class-level shift exactly, so v4 would have returned NULL ON A TRUE HYPOTHESIS
([1520], verified by malign to six decimals).

RH's baseline, his word: **"vs fallers."** The statistic is P1's `A`.

    A_dim = wmean(FALLERS) - wmean(RISERS),  weights |delta|, UNCENTRED

---

## §C2 THE BATTERY — six registered arms, two exploratory

    arm                       statistic        direction    null
    H1 GENERAL                A_valence        NEGATIVE     membership
    H2 GENERAL                A_|valence|      POSITIVE     membership
    H3 GENERAL                A_|dominance|    POSITIVE     membership
    H1 TOP-MOVERS             R_riser valence  POSITIVE     mass-order
    H2 TOP-MOVERS             R |valence|      REDUCED      mass-order
    H3 TOP-MOVERS             R |dominance|    REDUCED      mass-order
    dominance signed, both    A_d / R_d        NONE         both, exploratory

**RH asked for "in general as well as of top movers" and each needs its own
null: general asks whether ROLE ASSIGNMENT relates to the dimension; top-movers
asks whether, WITHIN a role, the biggest movers differ.** Using one null for both
is what produced [1541].

**arousal is DONE** — primary null, readouts quarantined under §7(a). Not re-run.

---

## §C3 THE CONFOUND PREDICTS ALL THREE REGISTERED DIRECTIONS

P1 established fallers are higher-arousal than risers. The lexicon couples
arousal to all three dimensions. Composing, `induced A = b * A_arousal`:

    quantity        slope     induced (A_arousal = 0.199, en displacing)
    valence        -0.1846        -0.0367       H1 predicts NEGATIVE  SAME
    |valence|      +0.2158        +0.0429       H2 predicts POSITIVE  SAME
    |dominance|    +0.1439        +0.0286       H3 predicts POSITIVE  SAME

**Under v4's mass-ordering framing this confound OPPOSED all three. Under RH's
role-membership framing it PRODUCES all three. The confound did not change; the
quantity did.** A confound's direction is a property of the mapping and does not
survive re-operationalisation ([1540]).

**CONSEQUENCE, and it is the spec's central constraint: on RAW `A`, a
confirmation of any arm is P1 re-expressed through the lexicon's own
correlations.** Only the residualised arm can carry a claim.

**THE NULL IS NOT ZERO. It is the induced benchmark, PER STRATUM.** `A_arousal`
differs by stratum (all-strata +0.2466 against displacing +0.199), and a pooled
benchmark is too lenient where the confound is strong and too strict where it is
weak — **worst at control, whose whole job is to sit at chance.** The producer
computes each stratum's benchmark from that stratum's own `A_arousal` and the
run's own slopes, and **prints which value it used where.**

**RAW-BEATS-BENCHMARK AND RESIDUALISED-BEATS-ZERO ARE ONE TEST.**
`A_resid = A_dim - b*A_arousal = A_dim - induced`. Both print because they read
differently; **the spec states they are ONE confirmation, not two.**

**STRATIFICATION IS STRUCK.** A stratified membership null was proposed and
measured: it absorbs the confound only partially (~81%), and finer strata absorb
more while permitting fewer permutations until the null degenerates onto the
observation. **The two failures converge before either is solved. Residualisation
is the SOLE defence and the spec says so.**

---

## §C4 EXACT ORDERINGS

    SIGNED      1. residualise signed dim on arousal, GLOBALLY
                2. wmean by role, weights |delta|
                3. A = wmean(f) - wmean(r)

    EXTREMITY   1. e = |dim_z| about the DECLARED ORIGIN  -- ABSOLUTE FIRST
                2. residualise e on arousal AND arousal^2, GLOBALLY
                3. wmean by role, weights |delta|
                4. A = wmean(f) - wmean(r)

**The absolute value comes FIRST because RH's hypothesis is distance from the
scale's neutral point, not from a cell's mean.** And the residualised variable is
**the one the hypothesis is about**: removing the signed coupling and then taking
absolute values leaves the extremity confound intact under a "residualised"
label.

---

## §C5 THE NULLS

    MEMBERSHIP null   permute the faller/riser LABEL within a cell, holding each
                      word's value and both role sizes fixed.
                      H0: role assignment is independent of the dimension.
                      -> the three GENERAL arms.

    MASS-ORDER null   permute |delta| within a role, as v13.
                      H0: within a role, mass is unrelated to the dimension.
                      -> the three TOP-MOVERS arms.

10,000 draws, seed 20260731. **The membership null gets its own calibration sweep
before the freeze, as §8 does for the CUSUM** — a single draw is not a
calibration.

---

## §C6 THE READING RULE — no undeclared thresholds

**"~0" is not a threshold; it is "does not exceed its own null."** Every term has
one under the joint permutation.

    H1   wmean_v(risers) EXCEEDS its null   AND A_valence beats its benchmark
    H2   wmean_v(risers) does NOT exceed    AND A_|valence| beats its benchmark
    H3   same, on dominance
    both     fallers negative-extreme AND risers neutral: H1's faller half with
             H2's riser half. Report as that, NOT as "both confirmed."
    neither  no structure on that dimension.

**H1 AND H2 MAKE CONTRADICTORY CLAIMS ABOUT THE RISER TERM** — H1 says risers are
positive, H2 says neutral. **The contrast `A` cannot see the difference:** under
H1 both sides are extreme so `A_|v|` is ~0; under H2 the faller signs cancel so
`A_v` is ~0. **Each hypothesis is invisible on the other's statistic.**

**So the four-number print is REQUIRED, not hygienic:**

    M_f, M_r, wmean(fallers), wmean(risers)   beside every arm

`A` and the distribution change `Delta_E = M_r*wmean_r - M_f*wmean_f` both
compute from these four and **neither is recoverable from `A` alone.** Rated
departed and arrived mass do NOT balance (arrived is 1.94x departed; 95.6% of
cells off by >10%), because CANONICAL is asymmetric by construction — **so `A x
moved-mass` is undefined and the two readings are genuinely different statistics.**

---

## §C7 THE POSITIVE-CONTROL GATE — every arm, before the freeze

**Each arm builds a world where ITS hypothesis is MAXIMALLY TRUE and confirms its
statistic FIRES.** No arm freezes without it.

**This gate exists because one error struck three times in one afternoon:**

    centring        held the cell mean     -> blind to a class-level shift
    the identity    assumed conservation   -> false on the rated subset
    the joint null  held role labels       -> blind to membership

**A REFERENCE THAT HOLDS CONSTANT THE THING THE HYPOTHESIS VARIES CANNOT TEST
IT.** All three are that. The gate takes minutes and would have caught the first
before thirteen spec versions were written against it.

**AND A NULL THAT IS MEANT TO CONTAIN A CONFOUND IS VALIDATED BY ITS MEAN, NOT
ITS p:** under confound-only its mean must sit ON the observed value. A
non-significant p is compatible with a broken null and a small sample.

**Per-cell permutation count prints, with a refusal below a floor.** A null with
one attainable value is not a null.

---

## §C8 GAP STRATUM, BLIND, AND SEATS

**GAP prints, never tests** ([1504]/[1505]): effect size and cell count beside
every percentile, because the gap holds 5.1x the displacing cells and its null is
2.26x tighter. **Cross-stratum reading on effect sizes only.** It CAN speak to
monotonicity and to displacement-specificity (which §7(a) is structurally blind
to, comparing only two extremes); it **CANNOT** be evidence for or against any
hypothesis. If the registered arms are null and the gap is not, that is not a
finding.

**BLIND TABLE:**

    H1 GENERAL, this population   SEEN by lacan ([1526], a value-keyed grep
                                  crossing a block-keyed blind). NOT confirmatory
                                  here. Confirmatory on the PAIRS population,
                                  MALIGN sole adjudicator.
    all seven others              BLIND. Extremity statistics exist nowhere in
                                  m01_norms; no dominance A-block exists.

**THE WITHHELD WALL, in the producer, not in anyone's intention:** the
current-population H1-general arm prints
`[WITHHELD -- malign holds this arm blind]` and emits values only under a flag
malign does not pass. **A blind cannot be enforced at the reading end.**

**AUDITING ARITHMETIC IS NOT ADJUDICATING A HYPOTHESIS.** lacan audits every
figure malign produces, including that one; malign adjudicates it alone.

---

## §C9 WHAT THIS INSTRUMENT CANNOT SEE

**Residualisation is the SOLE defence against a confound that predicts all three
registered directions.** Stratification was measured and struck. **One line of
defence, not two, and any claim from this run says so.**

**The residualisation is GLOBAL**, so a cell whose own coupling differs from the
population's keeps that difference in its residuals.

**Curvature is corrected to second order only** — measured at 9.8% and 9.0% on
the extremity arms and 0.0% on the signed arm, so third-order departure is
bounded by those figures but not zero.

**§7(a) adjudicates displacement-specificity identically to B**, and the arousal
precedent governs: if control sites move, the arm is quarantined as
magnitude-salience however clean the residualisation.
