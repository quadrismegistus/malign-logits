# Registration C — the valence/dominance delta on frozen v13 (v4, CONSOLIDATED)

A DELTA, not a new spec. Frozen `registration_b_spec_v13.md` `06186c42f9ff46e0`
governs everything not named here.

**v1 `3912f1dc9e9eb7a7` ([1488]), v2 `56099e877bc54126` ([1489]) and v3
`a365e03edd0491e6` ([1500], sight-checked [1501], origin ruled [1502]) are
SUPERSEDED, kept as record. v4 = v3 + §C9 only; nothing else moved.** v3 consolidates [1486], [1487], [1490], [1491],
[1492], [1493], [1495] per [1491].5. **NOT FROZEN — RH is still adding
hypotheses and each is free while the blind holds.**

---

## §C0 THE DECLARED TRIPLE — the [1479] gate, answered per dimension

    POPULATION   the frozen Registration B population, unchanged: prompts 959
                 fd3f14796ba9481b, models 95 e4c507eb8dbcf593, en only, base ->
                 most-aligned arm, CANONICAL. Function words excluded, lemma
                 repair applied, z anchored to the source database. Qualifying
                 = >= 3 rated non-function words of the role.

    RESIDUAL     a bin, as v13. PLUS: every readout reported TWICE, raw and
                 AROUSAL-RESIDUALISED. Three specifics, per [1487]:

                 GLOBAL, NEVER WITHIN CELL. One regression over ALL rated
                 non-function words in the qualifying population; residuals
                 per word. A within-cell fit leaves ONE degree of freedom at
                 n=3 (328 displacing cells, 7%; 995 at n<=5, 21%) and emits
                 structural zeros that enter T while counting in the
                 denominator -- silent dilution, the [1435].1 class.

                 THE RESIDUALISED VARIABLE IS THE ONE THE HYPOTHESIS IS
                 ABOUT. For signed hypotheses, regress the signed dimension on
                 arousal. For EXTREMITY hypotheses, regress the EXTREMITY
                 VARIABLE on arousal -- see §C4. Removing the signed coupling
                 and then taking absolute values leaves the extremity confound
                 intact while carrying the label "residualised" ([1492].3).

                 IDENTITY CHECK PRINTS FOR EVERY ARM. E[s|multiset]=1 is a
                 property of the permutation, so residualised values -- fixed
                 per word before any permutation -- preserve exactness. The
                 null median of T MUST centre on the NUMBER OF QUALIFYING
                 CELL-ROLES. On residualised arms it doubles as the detector
                 for a mis-fitted or mis-ordered residualisation, which shows
                 as a null median BELOW the cell count ([1487].3).

    ORIGIN       THE DATABASE MEAN, and it is the SAME origin in all three of
                 its roles ([1494], [1497]). v13 already anchors z there, so
                 |dim_z(w)| IS deviation-from-database-mean; this states it
                 rather than leaving it implicit. The origin is load-bearing
                 three times over and was undeclared in all three:

                   1. it sets the EXTREMITY readout's zero -- what "|dim|" means
                   2. it sets the H1/H2 DIVIDING LINE -- the two hypotheses are
                      IDENTICAL below it and OPPOSED above it (§C6)
                   3. it sets the COMPOSITION diagnostic that decides whether
                      the H1/H2 comparison had power at all

                 On the lexicon the three candidate origins (corpus mean 5.064,
                 scale midpoint 5.000, median 5.200) move the discriminating
                 share by 6.1 points, 54.2% / 55.8% / 49.7% -- and the median
                 origin flips it BELOW HALF. On the smaller, mass-selected
                 qualifying population the spread will differ again.

                 SENSITIVITY, printed: the composition at the database mean AND
                 at the scale midpoint. Two pre-declared numbers, and they tell
                 a reader whether the comparison's power was robust to the
                 convention or an artefact of it ([1497].4).

    SIDEDNESS    ONE-SIDED UPPER for T (max|C| is a magnitude). ONE-SIDED
                 DIRECTIONAL for each readout carrying a registered direction,
                 in that direction. Exploratory readouts report the one-sided
                 fraction of nulls below the observed value and carry NO
                 directional claim.

---

## §C1 WHAT CHANGES — ONE PARAMETER, THREE COLUMNS

The dimension column, from the SAME hash-pinned Warriner file (`85f6d7e3`,
`V.Mean.Sum` / `A.Mean.Sum` / `D.Mean.Sum`). **Not the Martinez LLM set.**

**Unchanged:** M_cell, one joint permutation null at seed 20260731, 10,000
draws, T, both directional readouts, §7(a) controls, |delta| ranking on both
arms, the reopen bar, guard-(a), §8 and its precondition, every counter and
bucket. **No new cloud run; a local re-join.**

---

## §C2 THE DECLARED BATTERY — four hypotheses, three shapes, all pre-data

    arousal      SIGNED, done. Primary NULL (T p=0.1018); readouts QUARANTINED
                 under §7(a). Not re-run, not re-read.

    valence H1   SIGNED UP. CONFIRMATORY, RH. High-mass RISERS rate HIGHER
                 (more positive) valence than the riser tail. Faller arm
                 reported with NO registered direction.

    valence H2   EXTREMITY DOWN. CONFIRMATORY, RH. mean |centred valence| of
                 the high-mass movers REDUCED against their role tail.
                 COMPETING with H1, not a second shot at it -- see §C5.

    dominance    EXTREMITY DOWN. CONFIRMATORY, RH. De-extremification: high-mass
                 movers are dominance-MODERATE, toward neither high nor low.

**INDEPENDENCE, recorded honestly for the multiplicity ([1493].3):**

    corr(valence, dominance)        +0.7166   SIGNED-signed
    corr(|valence|, |dominance|)    +0.4834   EXTREMITY-extremity, r^2 = 0.23

**Dominance-extremity is independent of valence H1 (signed), correlating only
`-0.19` with it. It is NOT independent of valence H2, sharing ~23% of variance
through the extremity channel.** So the battery is three distinct shapes, of
which **shapes 2 and 3 are partially one measurement** and must not be reported
as mutual corroboration without that qualifier.

---

## §C3 THE CONFOUND IS MEASURED, AND IT OPPOSES EVERY REGISTERED DIRECTION

Registration B established -- quarantined as non-displacement-specific, but real
about the data -- that **the biggest movers of either role are the higher-arousal
words of their role.** The frozen lexicon fixes what that implies. Warriner
table alone, no movement data joined:

    SIGNED, by arousal decile          decile 1 (z -1.523)   decile 10 (z +1.955)
      mean valence                         +0.230                -0.320
      mean dominance                       +0.159                -0.367
      at arousal z >= +1.0 (n=2244):   valence -0.3147   dominance -0.3289

    EXTREMITY, by arousal
      corr(|valence - mean|,   arousal)    +0.3601
      corr(|dominance - mean|, arousal)    +0.2389
      mean |dominance| at arousal z >= +1.0   1.045   vs   0.636 at z <= -1.0

**THE SIGNED U IS ASYMMETRIC. The extremes do not cancel; they net NEGATIVE at
-0.31** ([1484], superseding the cancellation reading of [1481].2/[1482].2).

**Every registered direction is therefore opposed by the confound:**

    H1  predicts valence POSITIVE      confound predicts NEGATIVE (-0.31)
    H2  predicts |valence| REDUCED     confound predicts INCREASED (+0.360)
    DOM predicts |dominance| REDUCED   confound predicts INCREASED (+0.239)

**All three are hard tests. A confirmed result must overcome an opposing prior
of substantial size, which is what makes any of them new information.**

**AND THE CONVERSE, which is the risk: all three share ONE confound.** A failure
of the arousal residualisation produces CORRELATED failures across all three,
not independent evidence. **§C0's identity check on the residualised arms is
what certifies otherwise, and no unifying reading is available unless it passes.**

---

## §C4 EXACT ORDERINGS — so two implementations cannot diverge

    SIGNED (valence H1)
      1. residualise signed valence on arousal, GLOBALLY
      2. centre within cell-role
      3. CUSUM / readouts, direction POSITIVE

    EXTREMITY (valence H2, dominance)
      1. e_w = |dim_z(w)|          ABSOLUTE VALUE FIRST, about the DECLARED
                                   ORIGIN of §C0 (the database mean)
      2. residualise e on arousal, GLOBALLY      <- the EXTREMITY variable
      3. centre the residual within cell-role
      4. CUSUM / readouts, direction REDUCED

**Why the absolute value must come first ([1492].2):** centring before it
computes `|d_w - mean_cell(d)|`, distance from the CELL's mean -- a DISPERSION
measure. RH's hypothesis is distance from the SCALE's neutral point. **These come
apart whenever a cell's own mean is off-neutral, which is most cells: a cell whose
words are uniformly high-dominance has small dispersion and large extremity, and
the wrong order scores it de-extremified when it is not.**

The centring at step 3 is machinery, not the hypothesis -- the CUSUM path must
return to zero -- and it is applied to the extremity variable, not to the raw
dimension.

---

## §C5 THE PRE-COMMITTED OUTCOME MAP — fixed now so it is not chosen later

**Per hypothesis, raw x residualised ([1486].3):**

    raw POSITIVE   resid POSITIVE   ->  the hypothesis, clean
    raw NULL       resid POSITIVE   ->  the hypothesis, MASKED by the coupling
    raw NEGATIVE   resid NULL       ->  the arousal confound; no claim
    raw NULL       resid NULL       ->  no structure; the hypothesis DIES

**ONLY both-null falsifies.** [1480].1's original falsifier fired on rows two and
three and would have killed a true hypothesis sitting on an opposing confound.

**Across H1 and H2 ([1491].3), no shared credit:**

    signed shift positive, controls flat        ->  H1
    extremity reduced, controls flat            ->  H2
    both                                        ->  shift-and-narrow
    neither                                     ->  no valence structure

**H2 confirmed is a confirmation of de-extremification, NOT a rescue of
positivity. The honest report is: two competing shapes were pre-registered and
the data selected X.**

**AND ONE FURTHER ROW, per §C6:** if the positive-valence share of the qualifying
population is small, **"the data selected H2" is NOT AVAILABLE** -- H2 would win
on cells where H1 predicts the same thing. The report is then *both shapes are
consistent with the data and this design did not separate them.*

**§7(a) adjudicates displacement-specificity for each, identically to B, and the
arousal precedent is the adjudicator: if control sites move, the readout is
quarantined as magnitude-salience however clean the residualisation.**

---

## §C6 WHERE H1 AND H2 SEPARATE — a power statement, registered

    valence vs ORIGIN     H1 (signed UP)   H2 (|v| DOWN)        contributes?
    BELOW the origin      predicts UP      predicts UP          NOTHING, identical
    ABOVE the origin      predicts UP      predicts DOWN        ALL of it

**A clean dichotomy, not a tendency ([1497].1): the two hypotheses are IDENTICAL
below the origin and OPPOSED above it. The whole model comparison lives on one
side of a line, and the line is §C0's declared origin.**

**The model comparison rests entirely on the POSITIVE-valence portion of the
qualifying population.** This is a transgressive-displacement corpus and the
moving words at displacing sites are plausibly skewed negative, in which case the
aggregate readouts are dominated by the words on which H1 and H2 AGREE and the
comparison is underpowered **by composition, not by sample size.**

**The composition cannot be measured now -- it is a join of valence to movement
data and the blind covers it. It PRINTS AT RUN TIME:** the share of rated
non-function words above and below neutral, and the same for the high-mass words
specifically. That is an output of the run, so the blind holds.

**NOT stratifying the readout by the word's own valence sign.** Valence is the
measured quantity; selecting cells by it and then measuring it is
selection-on-outcome. **The composition print is descriptive of the population;
a stratified readout would be a different and invalid test.**

---

## §C7 WHAT PRINTS THAT DID NOT PRINT BEFORE

    (i)    corr(dim, log P) per dimension, measured on THIS population against
           the 0.15 bar. NO arousal figure inherited: not +0.08 (v13 §2, looser
           population), not +0.071 (the B run), and NOT +0.026, which is
           corr(arousal, logFREQ), a different quantity ([1482].1, struck).
    (ii)   V/A/D pairwise WITHIN the qualifying population, SIGNED AND
           EXTREMITY. The 13,915-word lexicon is the wrong denominator
           ([1481].3). For dominance the confound print is |dominance| BY
           AROUSAL -- the signed table does not speak to the registered
           extremity direction ([1493].4b).
    (iii)  the valence composition of §C6.
    (iv)   the identity check per arm, per §C0.
    (v)    the §5 third curve's verdict COMPUTED, not asserted. In the B run it
           printed a conditional whose antecedent it never evaluated, and the
           antecedent FAILED (log-P quintile 5 = +0.108). [1472].3 deferred the
           fix to the producer's next touch; this is that touch.

---

## §C8 WHAT THIS INSTRUMENT CANNOT SEE

**Residualisation is GLOBAL, so it removes the population-level linear relation
and not any cell-specific one.** Correct here -- the within-cell alternative
destroys small cells -- but a cell whose own coupling differs from the
population's keeps that difference in its residuals. A trade, not a free win.

**Residualising removes the LINEAR component only.** The valence-arousal relation
is U-shaped on |valence|; a linear residual leaves the quadratic part standing.
**A residualised-positive result is evidence, not proof, that the shift is not
the U.**

**A mean readout cannot distinguish "no effect" from "symmetric polarisation."**
The extremity readouts mitigate partially: extremity and mean are two statistics,
not a decomposition.

**Nothing here speaks to displacement-specificity.** §7(a) does, and it has
already quarantined one dimension on this exact apparatus.


---

## §C9 THE GAP STRATUM — PRINTED, NEVER TESTED

`collect()` already assigns `stratum = "gap"` and builds those rows; `main()`
iterates only `("displacing", "control")`, so **24,038 qualifying cell-roles —
81% of the population — are computed and discarded.** v4 prints them.

    stratum       cells    vs displacing    null SE vs displacing
    gap          24,038        5.11x            0.44x   (2.26x TIGHTER)
    displacing    4,700        1.00x            1.00x
    control         934        0.20x            2.24x

**CONDITION ONE — COMPARE EFFECT SIZES, NEVER PERCENTILES ([1505].2).** The gap
arm holds 5.1x the displacing cells and 25.7x the control cells, so its null is
correspondingly tighter. **An effect HALF the size of the displacing one prints a
MORE EXTREME percentile in the gap.** Every stratum row therefore prints its
EFFECT SIZE and its CELL COUNT beside its percentile, on the same line, so the
asymmetry is visible where the numbers are. **Cross-stratum reading is on effect
sizes only.**

**CONDITION TWO — WHAT THE GAP CAN AND CANNOT SUPPORT, fixed before the number
exists ([1505].3).** "Exploratory" is an adjective and adjectives do not survive
a striking number; a printed percentile is a test in everything but name.

    CAN      support or undercut MONOTONICITY across gap -> control ->
             displacing, read on effect sizes
    CAN      undercut displacement-specificity: if gap ~ displacing, the
             "displacing-specific" framing fails in a way §7(a) CANNOT SEE,
             because §7(a) compares two extremes and a confound present at
             both but absent in the middle passes it
    CANNOT   serve as evidence FOR or AGAINST H1, H2 or the dominance
             hypothesis. IF THE REGISTERED ARMS ARE NULL AND THE GAP IS NOT,
             THAT IS NOT A FINDING — it is an unregistered look, and the
             honest report is that the registered test was null.

**The third line is written now because it is the line nobody will want to write
afterwards.**

**OUT OF SCOPE, unchanged:** the gap is not tested, carries no direction, enters
no multiplicity record, and changes no row of §C5. Registration B is not
reopened. A continuous-displacement design is a different apparatus, not a delta.
