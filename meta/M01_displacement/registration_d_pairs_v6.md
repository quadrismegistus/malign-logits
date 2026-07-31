# Registration D v6 — the definitive displacement-site test, all three dimensions

A DELTA on frozen `registration_c_delta_v6.md` `06f0272d7f21b901`, which governs
everything not named here. **v1-v3, v4 `1bd4ad8bbf461282` and v5 `b1d2ccd596a4231e` are SUPERSEDED.** v4's H1-signed arm carries over VERBATIM and is untouched by this amendment — it was designed before any unblinding and its primary status is preserved:
RH ruled the paired design and ruled out a single displacement bar ([1598]).
Drafted by the seats that have never seen the withheld value — malign (blind by
protocol), pen (blind by role). **The population does not yet exist.**

---

## §D0 THE DECLARED QUAD

    POPULATION   the post-construct-read survivors of the authored pool AS IT
                 STANDS AT RH'S CONSTRUCT READ — the 188 of
                 `pair_drafts/EXCLUSIONS.json` PLUS the ~160-pair drafting round
                 commissioned at [1662], and any further round RH commissions
                 BEFORE the read. Growth AFTER the read is a new registration.
                 THE UNIT IS THE PAIR. A pair
                 enters only if BOTH members qualify (>= 3 rated non-function
                 words in each role, as v6); one member failing drops the pair,
                 and the drop count prints per threshold point (§D6).
                 THE POPULATION RUNS ONCE ([1297].3 / [1324].1) — frozen at its
                 hash before any join; a grown pool is a new registration.

    RESIDUAL     a bin. Every readout reported twice, raw and
                 AROUSAL-RESIDUALISED, GLOBAL fit over this population's own
                 qualifying set, never within cell.

    SIDEDNESS    ONE-SIDED, fixed ONCE for the whole curve, in the registered
                 direction. Not chosen per point.

    ORIGIN       the database mean (z anchored there), the same origin wherever
                 one is used.

---

## §D1 THE PAIRED STATISTIC — the twin replaces the control stratum

For each member, H1's role-membership quantity, unchanged from v6:

    A = wmean(fallers) − wmean(risers)     signed valence, UNCENTRED,
                                           |delta|-weighted, arousal-residualised

**The pair's statistic is the within-pair difference:**

    D_pair = A(MARKED) − A(UNMARKED)
    D      = mean of D_pair over admitted pairs

**H1 predicts NEGATIVE-valence words fall and POSITIVE rise, so A is negative;
the paired form asks whether that is stronger at the marked member.**

    REGISTERED DIRECTION   D < 0.   One-sided lower.

**THE TWIN IS A MATCHED *MOVING* CONTROL, NOT AN INERT ONE ([1650].2).** On the
only squarely-transgressive substrate, marked-displaces AND unmarked-inert
co-occur ZERO times in 126 pair-cells (p = 4e-6 under independence, [1648]):
displacement MASS tracks the PROMPT, not the swap, because the members share a
continuation distribution. **So this design reads whether the COMPOSITION of
movement — which words, what valence — differs at the marked member, NOT whether
the unmarked stays still.** The sign-flip null is indifferent to shared movement,
and §D4's H0 world is exactly the both-move case.

**Why the twin and not the bar-selected control stratum:** the members differ by
ONE substitution, with syntax, length and frame held. The Registration C control
was assembled by a qualification bar that admitted the movement-rich tail of
low-movement prompts — 8.5x the departed mass of the cells it excluded, 2.8x the
threshold that defined the stratum ([1595]/[1596]). **A matched twin cannot
de-control that way, because it is not selected on movement at all.**

**A(MEMBER) IS DEFINED EXPLICITLY, because "both members qualify" reads two ways
and they are different populations ([1603].1):**

    a CELL qualifies      >= 3 rated non-function words in EACH role, as v6
    A(member)             the MEAN of A over that member's QUALIFYING CELLS
    a MEMBER qualifies    >= 1 qualifying cell
    a PAIR is admitted    BOTH members qualify
    PRINTED               the qualifying-CELL COUNT per member, per threshold
                          point, in §D6 — so a member resting on one cell is
                          visible rather than pooled with one resting on twenty

**A(MARKED) and A(UNMARKED) print separately beside every D.** A difference is
not a direction until both terms are visible — the [1524].4 rule, which on the
norms population was the only thing standing between H1 and a misclassification.

---

## §D2 THE NULL — sign-flip within pairs

    H0:   the MARKED/UNMARKED label is unrelated to A within a pair
    draw: independently flip the label of each pair with probability 1/2,
          recompute D
    exact enumeration at n <= 20 pairs (2^n <= 1,048,576); sampled above,
          draw count declared

**This is the paired permutation test. It respects the pairing by construction —
nothing is permuted across pairs, and the pooled bag is never formed.** The
membership null of v6 is NOT used here: it permutes role labels within a cell and
answers a different question, which is the [1541] lesson applied to a new unit.

**A REFERENCE THAT HOLDS CONSTANT THE THING THE HYPOTHESIS VARIES CANNOT TEST
IT.** Here the hypothesis varies which member is marked; the null must vary that
and nothing else.

---

## §D3 THE CURVE — and the rule that stops it being a menu

**No single displacement bar.** `D` is reported as a function of the threshold `t`
on the pair's displacement (the lesser of the two members' median departed mass,
so a pair clears only if BOTH do):

    GRID, declared here:  t ∈ {0.00, 0.01, 0.02, 0.05, 0.10, 0.20}
    FLOOR:                n >= 6 admitted pairs at a point
    BELOW FLOOR:          the point prints "UNDERPOWERED, n=<k>" and enters NO
                          reading. It is not a null and not a failure.

**PRIMARY: `t = 0.00`** — every qualifying pair, **the one point that embodies no
author's choice.** Level **alpha = 0.05, one-sided**, named here so no
implementer's default becomes an adjudication ([1589]).

**THE READING RULE, FIXED BEFORE THE DATA:**

    CONFIRMED             D < 0 past its null at t = 0.00, AND D < 0 at EVERY
                          above-floor point on the grid (sign, not significance)
    THRESHOLD-DEPENDENT   D < 0 past its null at t = 0.00 but the SIGN FLIPS at
                          one or more above-floor points. Reported with the whole
                          curve; NOT quotable as a general claim about H1.
    NOT SUPPORTED         D not past its null at t = 0.00, whatever the curve does
    NOT A FINDING         significance appearing ONLY at points other than the
                          primary. The curve is a sensitivity, never a search.

**Monotonicity across `t` is reported DESCRIPTIVELY and is not registered — no
direction is predicted for it, and quoting a trend in it would be quoting outside
this registration.**

**THE CURVE MUST BE SHOWN TO HAVE RESOLUTION, OR ITS CLAUSES ARE VACUOUS.** The
qualification bar (>= 3 rated words per role, BOTH members) requires movement in
both directions, so it removes low-displacement pairs **before `t` ever acts on
them**. If the bar has already excluded everything below some displacement, the
low grid points admit THE SAME PAIRS and the curve is flat by construction rather
than by finding.

    PRINTED PER POINT   |admitted|, and the JACCARD OVERLAP of that set with the
                        t = 0.00 set
    COLLAPSED POINT     overlap >= 0.95 with the primary set. It is reported as
                        COLLAPSED, and it does NOT count toward the CONFIRMED
                        rule's "every above-floor point" — a point that is the
                        primary set under another name cannot corroborate it.
    IF ALL POINTS COLLAPSE   the result is reported as SINGLE-POINT, never as
                        CONFIRMED. The curve tested nothing and the spec says so.

**This clause exists because on the norms population the qualification bar
manufactured the very gradient the design meant to read: un-barred, the three
strata sat at 0.0067 / 0.0088 / 0.0959 — two conditions, not three — and after
the bar at 0.0566 / 0.0943 / 0.1610, a smooth ladder ([1599].2). A curve is the
right instrument for that failure only if it is checked for resolution.**

---

## §D4 THE PRE-FREEZE GATE

**The paired statistic must FIRE in a maximally-true PAIRED world** — marked
members with negative-valence fallers and positive-valence risers, unmarked twins
with neither — **and the sign-flip null's mean must sit far from the observation.**
No arm freezes without it ([1543]/[1546]).

**The null's calibration sweep runs on this population's pair-count**: p uniform
under H0 over many draws; the attainable-p lattice printed (`1/2^n` at exact
enumeration) with a refusal if a point cannot reach alpha. **A null with one
attainable value is not a null.**

---

## §D5 INHERITED, CITED NOT RESTATED

    norms          Warriner `85f6d7e3`, V.Mean.Sum, human/exogenous
    residualise    SIGNED valence on arousal, LINEAR ONLY (b2 = +0.0005 in the
                   correction direction, [1549]/[1550])
    benchmark      PAIRED, on the same construction as the arm ([1603].2):
                       D_bench = induced_A_arousal(MARKED)
                               − induced_A_arousal(UNMARKED)
                   cell-averaged (the ARM's OWN estimator, never pooled —
                   [1594].1: pooled ran ~20% high and made every bar too strict),
                   computed from THIS population's own A_arousal and its own
                   slope, PER THRESHOLD POINT. **The pairing partially cancels
                   the confound; an unpaired benchmark applied to a paired
                   statistic is a different number wearing the bar's name.**
    seeding        per-arm `default_rng([SEED, sha256_id(...)])`, NEVER Python's
                   builtin `hash()` — string hashing is salted per process
                   ([1579].1)
    §C9 gap        no gap stratum exists in a paired design; the clause does not
                   transfer and is not silently carried

---

## §D6 SELECTION DIAGNOSTICS — printed as standard output

**Per threshold point:** pairs admitted; pairs dropped because ONE member failed
qualification; and **the median displacement of admitted versus dropped pairs.**

**This exists because a qualification bar is a selection on the outcome's
neighbourhood whenever it requires enough data to measure** ([1597].3). Requiring
rated words in both roles requires movement in both directions; that is exactly
what a low-displacement member cannot supply. **The bar and the population
definition are in tension by construction, and the diagnostic is how a reader
sees whether it bit.**

---

## §D6b THE FOUR ARMS — one paired machine, four dimensions

Same statistic, same sign-flip null, same curve, same collapse clause. Only the
per-word value changes:

**RESIDUALISATION IS PINNED PER ARM ([1635].1). "Every readout twice" does NOT
transfer to the arousal arm, which cannot residualise arousal on arousal and
whose induced benchmark would collapse into the arm itself:**

    H1 SIGNED            linear arousal residualisation (v4 verbatim; b2 = +0.0005
                         in the correction direction, so no quadratic term)
    VALENCE-EXTREMITY    |dim_z| FIRST, then residualise the EXTREMITY variable on
    DOMINANCE-EXTREMITY  arousal AND arousal^2 (~10% curvature, measured)
    AROUSAL              **RAW ONLY. No residualisation, NO induced benchmark.**
                         It reads against its sign-flip null alone, because THE
                         ARM IS THE CONFOUND VARIABLE and a benchmark computed
                         from it is the arm wearing the bar's name.

    ARM                      quantity            DIRECTION   status
    H1 SIGNED (v4, primary)  A_valence signed      D < 0     confirmatory, RH
    AROUSAL                  A_arousal signed      D > 0     confirmatory — P1's
                                                             site prediction in
                                                             paired form. ABSORBS
                                                             Registration A's
                                                             marked/unmarked design
                                                             ([1258]); the [1297].3
                                                             runs-ONCE freeze
                                                             DISCHARGES into this arm.
    VALENCE-EXTREMITY        A_|valence|           D > 0     confirmatory — does H2's
                                                             de-extremification track
                                                             transgressive sites?
    DOMINANCE-EXTREMITY      A_|dominance|         D > 0     confirmatory, **AND ITS
                                                             PRIOR DEATH IS DECLARED**:
                                                             H3 failed on the norms
                                                             displacing stratum
                                                             ([1576].2). This is its
                                                             ONE pairs-population test,
                                                             not a resurrection.

**Every arm reads only PUBLIC quantities** — P1 arousal, H2 extremity, H3
dominance, all on the docket. **Nothing here touches the sealed value.**

---

## §D6c HIERARCHY AND ALPHA — one structure, declared

**TWO FAMILIES. No arm's level is chosen by an implementer.**

    FAMILY 1   H1 SIGNED, standalone, alpha = 0.05 one-sided.
               v4's primary, preserved as commissioned.

    FAMILY 2   the three SITE-SPECIFICITY arms, FIXED-SEQUENCE at alpha = 0.05,
               in this declared order:
                   1. AROUSAL              (largest prior effect, P1)
                   2. VALENCE-EXTREMITY    (H2, confirmed on norms)
                   3. DOMINANCE-EXTREMITY  (H3, dead on norms)
               Testing stops at the first non-rejection; arms below it are
               NOT TESTED and are reported as such, never as null.

**Fixed-sequence is chosen over a Bonferroni split because the ordering is
justified in advance by prior evidence strength and costs no alpha.** The order is
declared here and cannot be revised after any number exists.

**Family 1 is separate so that a failure of the H1-signed arm cannot block the
site-specificity test RH commissioned** — the two answer different questions.

---

## §D6d THE DEFINITIVE READING RULE — both outcomes informative

**Per dimension, and this is what makes the test definitive rather than
one-sided:**

    D > 0 past its null              the effect TRACKS TRANSGRESSIVE SITES
    D null AND MDE < the dimension's  evidence AGAINST site-specificity: the
      known effect size               effect is MOVEMENT-GENERAL. QUOTABLE AS SUCH.
    D null AND MDE >= that size       UNINFORMATIVE AT THIS POWER. Quotable as
                                      nothing, in either direction.

**The middle row is the one Registration C could not reach**: its control arm's
MDE (0.0390) exceeded the displacing effect (0.0251), so its null could not
distinguish absence from blindness ([1608]). **Here the MDE prints per arm per
point and the rule reads it, so a null is interpretable at the moment it is read.**

**KNOWN EFFECT SIZES, from the public record, declared in advance:**

    arousal              ~0.10   (P1's displacing/control site gap)
    valence-extremity     0.025  (H2's residualised displacing figure)
    dominance-extremity   0.025  (stand-in; its own is unknown, declared)

---

## §D6e POWER — ORDINAL FACTS AND CARDINAL INTERVALS

**ORDINAL, solid at every plausible constant value:**

    the SIGNED arm clusters far worse than the EXTREMITY arms
    clean-substrate co-qualification ~85%, not the ~42% of the contaminated one
    FAMILY behaves like replication; PROMPT carries the dependence
      -> more MODELS buy nothing; only more PAIRS do ([1645])

**CARDINAL, as INTERVALS — every constant here is an ICC from few groups and a
point estimate would be this thread's own defect ([1657]/[1658]):**

    at MEI 0.025, detection convention          pool pairs
      H2 valence-extremity                        42 - 339   (point 123)
      H3 dominance-extremity                      30 - 269   (point  98)
      AROUSAL                                    POWERED at the current pool
      H1 signed                                  480 - 1,950 at every measured
                                                 clustering value

**H1-on-pairs is reachable only at a clustering value that every adequately-
sampled measurement rejects** — its 266-pair floor requires deff = 1, excluded by
both the 28-pair ICC (0.100, CI [0.033, 0.167]) and the 68-group prompt-clustered
signed ICC (0.094, deff 3.64). **It is reported as out of consideration, not
priced.**

**MEI = 0.025** — RH's declared target, entered pending his final confirmation.
**The realized MDE prints per arm per threshold point and is what converts any of
this into a verdict; the forecast never does** ([1658]).

---

## §D7 FALSIFIER, AND THE WALL'S DISCHARGE

**If D is not negative past its null at `t = 0.00` at alpha = 0.05, H1 is NOT
SUPPORTED on the pairs population.** No magnitude is predicted.

**ON COMPLETION — whatever it returns — THE WALL IS FULLY DISCHARGED**: the
withheld H1-general value and the §C6 first-conjunct verdict enter the docket,
and the conjunct re-enters under two-seat audit ([1584].3).

---

## §D8 WHAT THIS DESIGN COSTS, DECLARED

**A pair is lost whole when either member fails qualification**, so the admitted
set is more restrictive than either member's own admission would be. §D6 measures
it; nothing hides it.

**And the paired contrast answers a narrower question than v1's:** it asks whether
H1's effect is stronger at the marked member, not whether H1 holds of alignment at
large. **A null here does not falsify H1 generally — it falsifies the claim that
the effect tracks the marked/unmarked contrast.** That distinction belongs in any
sentence quoting this result.

---

## §D9 WHAT I HAVE NOT SEEN

**No H1-general value and no §C6 conjunct verdict, on any population.** Warriner
lexicon properties and published arousal figures only; the Registration C valence
arms were run without `--show-h1-general`, so that arm was never computed here.
