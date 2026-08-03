# Registration D3b — decomposing D2's effect against pool extremity

**STATUS AS DECLARED (2026-08-03 UTC, per [3467]): draft; not in force as declared. Freeze state is recorded on the docket and in git history. NO D3b QUANTITY HAS BEEN COMPUTED.**

    BENCHMARKS   registration_d2_extremity.md @ 881287ed3642ed55
                 read a1d712093155f32c: val_extrem D +0.01511 p 0.00880,
                 dom_extrem D +0.01655 p 0.00960, both CONFIRMED
    OCCASIONED   RH's question: does D2's confirmation merely reflect
                 transgressive prompts having more extreme valence available?
    ESTABLISHED  the pool-extremity diagnostic (two-seat, five-decimal
                 agreement; fe2c92d33bc97161 / b23f415a966fcdea): MARKED pools
                 ARE tail-richer -- +2.66 pts on a 25.87% base at |z| >= 1,
                 439/632 pairs positive. The confound is LIVE, not hypothetical.

**THIS IS A BENCHMARK OF AN EXISTING RESULT, NOT A DISCOVERY CLAIM, AND IT SAYS
SO ON ITS FACE.** D2's numbers are quoted above as the effect under adjudication.

---

## §1 WHAT IS DELIVERED, AND WHAT IT DOES NOT DECIDE

**D3b DECOMPOSES D2's EFFECT INTO A POOL-ASSOCIATED PART AND A POOL-INDEPENDENT
PART. IT TAKES NO POSITION ON WHETHER THE POOL-ASSOCIATED PART IS ARTIFACT OR
MECHANISM.**

The pool is the model's continuation distribution, **measured downstream of the
manipulation**. So the gap admits two readings this corpus cannot separate:

    CONFOUNDING   transgressive stems sit in contexts with extreme continuation
                  vocabulary for reasons unrelated to transgression
    MEDIATION     transgression ELICITS extreme continuation vocabulary, and the
                  extremity contrast follows -- the effect is real and pool
                  extremity is HOW IT WORKS

**BOTH REMAIN OPEN AFTER D3b. A large pool-associated share does not by itself
license the word "just".**

**AND A THIRD THING IS OUT OF SCOPE ENTIRELY, STATED SO NOBODY LOOKS FOR IT HERE:
"TRANSGRESSION" AND "THE SWAPPED WORD'S OWN VALENCE EXTREMITY" ARE ENTANGLED
PROPERTIES OF THE TREATMENT.** The minimal pair varies one word, and that word is
both the transgressive element and, typically, a valence-extreme one. **No
analysis of THIS corpus separates them** — separation would require
extreme-but-NON-transgressive swaps, which is a different corpus and a different
registration. **D3b decomposes against the POOL; it cannot decompose the
treatment.**

**AND A FOURTH REFUSAL, THE ONE CLOSEST TO THE QUESTION THAT PROMPTED THIS.** RH
asked: *is the effect just because valence starts higher?* **A COUNTERFACTUAL
ANSWER — "here is D when it does NOT start higher" — IS NOT AVAILABLE FROM THIS
CONSTRUCTION**, because `D_pair` and `gap_pair` are computed from the same words
(§6.4). **What D3b delivers is the share of the effect that a linear function of
the pool gap does not account for.** That is weaker than the question asks for,
it is honest, and no reader should take the stronger answer as having been given.

**A DISCLOSURE ON A PARALLEL CHECK:** a tautology check on the regressor's
construction runs beside this registration ([3419].2, commissioned, malign's).
**Its artifact is CITED with the result. It does not gate this registration and
does not alter the decomposition** — §1's neutral framing was built to survive
either outcome, and does.

---

## §2 THE BRACKET — two instruments, biases pointing at each other

    CONFOUND SIDE   RELABEL-IN-REAL-PAIRS.  Take the 632 admitted pairs, keep
                    every pool and movement exactly as measured, reassign role
                    BY POOL EXTREMITY, recompute D2's statistic.
                    THE SORT KEY IS §4's REGRESSOR -- `mean|z|` UNWEIGHTED,
                    POOLED across a member's cells (Amendment A §A2, §A3).
                    ON AN EXACT TIE the DATA's roles are RETAINED, and the TIE
                    COUNT PRINTS beside `relabelled D` for every construction.
                    READS HIGH -- role/pool concordance is 1.0 here against
                    417/632 (~66%) in the data under that key, so it is the
                    confound at FULL STRENGTH: an UPPER BOUND on the
                    pool-associated share.
                    DECLARED SENSITIVITIES, each reporting `relabelled D` and
                    §7's ratio: tail>=1 unweighted (439/632, ~69%, ONE TIE)
                    and mean|z| |delta|-weighted (362/632, ~57%).

    RESIDUAL SIDE   C's INTERCEPT.  Regress D_pair on gap_pair over the 632;
                    read b0, THE COMPONENT OF `D_pair` NOT LINEARLY PREDICTABLE
                    FROM `gap_pair`.  (NOT "D at zero pool gap": the two are
                    functionals of ONE word set, so the gap cannot be varied
                    with the words held fixed -- §6.4.)
                    READS HIGH -- errors-in-variables attenuates b1 and drags
                    b0 toward the unadjusted mean: an UPPER BOUND on the
                    pool-independent share.

**THE DELIVERABLE IS THE BRACKET, NOT EITHER NUMBER.** Both sides read high, so
neither alone is conservative; together they bound the decomposition from
opposite ends.

**Proposals A (coin-flip pseudo-pairs) and B (pool-sorted pseudo-pairs) are
DROPPED.** A is centred at zero by symmetry and duplicates D2's sign-flip. B is
dominated by the relabel: same null strength, real pairs, no pseudo-corpus to
defend.

---

## §3 THE ESTIMATOR FORK, SEALED BEFORE ANY RELIABILITY NUMBER EXISTS

    PRIMARY       the DISATTENUATED intercept
    SENSITIVITY   the RAW intercept

**Sealed on the estimator's property, not on any figure: the corrected form is
the one whose bias does not run systematically in our favour.**

**RELIABILITY IS OF `gap_pair` — THE DIFFERENCE, NOT EITHER MEMBER'S MEAN**
(Amendment A §A4). Half-gaps are computed per pair as
`mean|z|(M_half) - mean|z|(U_half)`, correlated across pairs by PEARSON, and
Spearman-Brown corrected. It is a STAGE-1 quantity: no D, no sign, no p.

**THE SPLIT IS SHARED OVER THE UNION of the pair's edges, not per member and
not over the intersection.** A split-half is licensed only insofar as each half
is a PARALLEL MINIATURE of the full quantity: the union assignment partitions
EACH member's full edge set, so both half-gaps are miniatures of the gap the
regression consumes. Per-member splitting injects edge-composition noise the
full statistic does not contain; the intersection reading drops a median of
seven edges per pair (631 of 632 pairs have differing edge sets) and discards
17 pairs where the union discards 8.

**THE SPLIT MUST BALANCE CELL MASS ACROSS HALVES.** Split-half assumes parallel
forms; a member's cells differ in size, so an unbalanced split understates
reliability and the correction overshoots. **Greedy longest-processing-time over
the union edge set on POOLED word count (MARKED + UNMARKED at that edge);
CELL-KEY (family, position) ASCENDING on equal count; on equal mass to the half
holding FEWER CELLS, then to half A. NO RANDOM SEED IS USED, and Python's
`hash()` is forbidden — it is salted per process.** The realized median and
maximum imbalance PRINT with the reliability.

**SCOPE AND SKIPS:** the correlation runs over the ADMITTED 632 — the population
the corrected estimator is fit on — with the all-pairs figure reported beside it
as a sensitivity. **A pair with an EMPTY HALF is skipped from the reliability
computation ONLY, stays in the regression, and the SKIP COUNT PRINTS.**

**THE FLOOR, DECLARED NOW RATHER THAN CHOSEN LATER: if estimated reliability
falls below 0.60, the corrected intercept is reported as UNSTABLE and the
residual side FALLS BACK TO THE RAW INTERCEPT with its one-way caveat.**
Dividing by a noisy estimated coefficient overcorrects, and at low reliability
the corrected estimate's error grows fast in the other direction.
**THE FLOOR APPLIES PER REGRESSOR**, each reliability against its own.
**DISATTENUATION:** correct `b1` by dividing by the reliability, then RE-DERIVE
`b0` through the means, so the fitted line passes through
(mean `gap_pair`, mean `D_pair`).
**THE RELIABILITY'S SIGNED DISTANCE FROM 0.60 PRINTS, PER ARM** — see §6.5.

**THE REGRESSOR'S WEIGHTING IS A DECLARED FORK, named at the construct read
before code:** the |delta|-weighted gap is what the statistic SEES but shares
construction with `D_pair` and risks mechanical coupling; the unweighted gap is
CLEANER than the weighted one — though not clean; see §6.4 — and measures
availability rather than exposure.
**One is named regressor with its reason; the other is carried as a
fixed-reading sensitivity.**

---

## §4 SUPPORT FOR THE INTERCEPT — measured, not argued

    gap_pair (mean_abs_z POOLED across a member's cells, MARKED - UNMARKED),
    n = 632   -- POOLED, not the mean of per-cell means: a cell-mean average
    weights a 3-word cell like a 40-word one, and the confound is about how many
    extreme words were AVAILABLE, which is a count question (Amendment A §A3)
      min -0.4193   q1 -0.0171   median +0.0250   q3 +0.0727   max +0.3696
      NEGATIVE 215   POSITIVE 417
      |gap| <= 0.01   71 pairs    |gap| <= 0.02  145    |gap| <= 0.05  333

**Zero is INTERIOR to the data with both tails populated. `b0` is an
interpolation, not an extrapolation**, and the binned near-zero form has 71-145
pairs rather than an empty neighbourhood.

---

## §5 THE DISCORDANT STRATUM — **EXCLUDED, AND THE ARITHMETIC IS WHY**

The stratum where pool-order runs AGAINST transgression would be the sharpest
available test. **"Pool-order" NAMES NO SINGLE MEASURE — the constructions give
different strata — so every construction is tabled and NONE is chosen:**

    SIX STRATA x TWO ARMS = TWELVE CELLS

    stratum                        n  |  MDE_val   xD  pwr |  MDE_dom   xD  pwr
    valence   tail>=1 unweighted  192 |  0.03208 2.12x  26% |  0.03548 2.14x  26%
    valence   mean|z| unweighted  215 |  0.03031 2.01x  29% |  0.03353 2.03x  28%
    valence   mean|z| weighted    270 |  0.02705 1.79x  35% |  0.02992 1.81x  34%
    dominance tail>=1 unweighted  329 |  0.02450 1.62x  41% |  0.02710 1.64x  40%
    dominance mean|z| unweighted  326 |  0.02462 1.63x  41% |  0.02723 1.65x  40%
    dominance mean|z| weighted    339 |  0.02414 1.60x  42% |  0.02670 1.61x  41%

**THE MDE EXCEEDS THE OBSERVED EFFECT IN ALL TWELVE CELLS. MAXIMUM POWER
ANYWHERE: 41.8%. EXCLUDED UNDER EVERY CONSTRUCTION.**

**The bound is stated as "all twelve cells" and "41.8%" rather than as a round
threshold**: an earlier draft of this clause said *"power never reaches 40%"*,
which was exact on a three-row valence-only table and FALSE the moment the
dominance strata — the rows with the largest n and therefore the highest power —
were added. **The rows that complete the table are the rows that break a
threshold read off an incomplete one.**

**A null in this stratum is uninformative BY CONSTRUCTION, and the strong reading
— "if D survives here no pool story reaches it" — is unavailable more often than
not EVEN IF THE EFFECT IS ENTIRELY REAL.**

**NO MEASURE IS SELECTED. The counts differ by construction (192 / 215 / 270 /
329) and the exclusion does not depend on which a reader assumes** — which is the
whole reason the table is here rather than one row.

**SOURCE OF RECORD: `results/result_d_stage1.json`'s STORED SDs, not the figures
printed in this document.** At the page's own precision the val MDE computes to
0.03207 against the tabled 0.03208; **a verifier reconciles against the artifact.**

**ESTIMATOR: these are CLOSED-FORM normal-theory MDEs, not §A7.2's SIMULATED
sign-flip MDE.** Where both exist they differ +0.12% (n=632) and −3.24% (n=287) —
immaterial against a 1.6x-2.1x ratio, **and both discrepancies point TOWARD the
exclusion: the simulated MDE below n=287 would be higher and the power lower.**

**It is excluded rather than registered with a caveat, because a test whose
informative branch is a coin-flip against itself invites exactly the reading its
power cannot support.** Recorded here so nobody proposes it again without the
number.

---

## §6 WHAT WOULD MAKE THIS THE WRONG MEASUREMENT

**COUNTERSIGNED SEPARATELY FROM THE ARITHMETIC.** Per standing method: a
validity gate listing only neutral risks has not been written honestly — **at
least one entry must be a way the measurement could be wrong IN OUR FAVOUR.**

    1. ATTENUATION -> b0 READS HIGH, IN OUR FAVOUR.  `gap_pair` is a noisy
       measure of the true pool difference. Errors-in-variables drags b0 toward
       the unadjusted mean, so the pool-independent share is OVERSTATED. The
       disattenuation addresses it and does not eliminate it; the binned form
       does not escape it either (pairs measured near zero are not pairs truly
       at zero).

    2. MEDIATION -> b0 READS LOW, AGAINST US.  If pool extremity is what
       transgression DOES rather than what the prompts smuggled in, b0 is a
       DIRECT effect and understates transgression's total effect. It is then
       not "the effect purged of confounding" and must never be quoted as that.

    3. MAXIMAL SORTING -> THE RELABEL READS HIGH, IN OUR FAVOUR ON THE OTHER
       SIDE.  Concordance 1.0 against the data's 66% UNDER THE DECLARED SORT
       KEY (417/632; Amendment A §A2 -- the 69% previously quoted here is the
       TAIL sensitivity, not the measure the bracket uses) means the relabel
       overstates the pool-associated share, which makes the residual look
       smaller than it is.  The argument is unaffected in logic: 1.0 exceeds
       0.66, 0.69 and 0.57 alike.  It must be PRICED against the measure the
       bracket actually reads.

    4. SHARED INPUTS -> THE INTERCEPT IS A DECOMPOSITION, NOT A COUNTERFACTUAL.
       NEUTRAL IN DIRECTION.  `D_pair` and `gap_pair` are two FUNCTIONALS OF ONE
       WORD SET -- they differ in role split, residualisation and weighting, not
       in their inputs. So `gap_pair` CANNOT be set to zero with the words held
       fixed, and "D at zero pool gap" names a counterfactual with no referent in
       this construction. `b0` is the component of `D_pair` not linearly
       predictable from `gap_pair`: well-defined, informative, and not that
       phrase. The unweighted regressor is CLEANER than the weighted one and is
       NOT CLEAN -- residualisation removes an arousal-predicted component from
       A's inputs; it does not make A a function of different words.

    5. THE FLOOR IS A DISCONTINUITY AND IT POINTS IN OUR FAVOUR.  Added by
       Amendment A §A7.  Above 0.60 a falling reliability corrects b1 UP and b0
       DOWN, which runs AGAINST us; crossing BELOW 0.60 falls back to the RAW
       intercept, which entry 1 declares OVERSTATES the pool-independent share.
       **So anything depressing reliability far enough does not shade the result
       against us by degrees -- it JUMPS it in our favour, discontinuously, with
       no single choice having done so.**

       ITEM-WISE NEUTRALITY DOES NOT COMPOSE.  Every ambiguity Amendment A
       resolved was checked and each resolves against us or neutrally; a
       guarantee verified PER ITEM can still be false ACROSS A BRANCH.

       THE FLOOR DOES NOT MOVE -- it was declared before any reliability
       existed.  THE RESPONSE IS DISCLOSURE: the reliability's SIGNED DISTANCE
       FROM 0.60 PRINTS PER ARM, so a near-miss is a number on the page and not
       a branch that silently absorbs it.

**Entries 1 and 3 are the honest-gate requirement: both flatter our own
decomposition, from opposite ends. Entry 4 flatters nothing and constrains the
deliverable's wording, which is why it is an edit as well as an entry. Entry 5
flatters us in a way no single item does, which is why it needed a mechanism
rather than a caveat.**

---

## §7 THE READING RULE, FIXED BEFORE ANY NUMBER

    relabelled D  >=  D2's D        the pool-associated share could account for
                                    all of it AT MAXIMAL SORTING. Not "does".
    relabelled D  <   D2's D        even maximal sorting does not reproduce D2;
                                    the shortfall is pool-independent.
    b0 far from 0                   substantial `D_pair` is NOT linearly
                                    predictable from the pool gap
    b0 near 0                       most of `D_pair` IS linearly predictable
                                    from it -- READ WITH §6.2: if the gap is a
                                    mediator this is expected and is not
                                    evidence of artifact

**REPORTED AS A DECLARED QUANTITY, WITH NO THRESHOLD: the RATIO
`relabelled D / D2's D`.** The two branches above tell a reader which side of the
line the bracket's edge falls on; **the deliverable is a WIDTH, and "slightly
below" and "far below" are different findings wearing one sentence.** The ratio
puts the magnitude on the page and **the reader does the sizing.** No cut is
declared — a cut chosen now would be chosen with D2's numbers in hand, and the
ratio is implied by the branches already, so it adds a number and no discretion.

**AND THE `b0` BRANCHES GET THE SAME TREATMENT, WHICH THE FIRST PASS GAVE ONLY
TO THE RELABEL PAIR** (Amendment A §A5). "Far from 0" and "near 0" name no
comparator, and `b0` is in A-units:

    REPORTED AS A DECLARED QUANTITY, WITH NO THRESHOLD: the POOL-INDEPENDENT
    SHARE, `b0 / mean(D_pair)`.

    numerator    `b0` as fitted -- disattenuated where §3's floor is met, raw
                 where it is not, and THE PRINTED ROW SAYS WHICH
    denominator  mean(`D_pair`) over THE SAME PAIRS THE REGRESSION FITS.  It is
                 D2's OWN D, POSITIVE on both arms (+0.015106155684521026
                 val_extrem, +0.016554961336197260 dom_extrem at t=0.00)
    sign         SIGNED, no absolute value.  A NEGATIVE share means the
                 pool-independent component runs OPPOSITE the effect, which is
                 a finding and must not be hidden
    range        NOT bounded to [0,1]; a share above 1 or below 0 is a real
                 outcome of a linear decomposition and is reported as-is

**BOTH SIDES OF THE BRACKET NOW CARRY ONE DENOMINATOR** — `relabelled D / D2's D`
on the pool-associated side, `b0 / D2's D` on the pool-independent side — **so
the two bounds are on one scale, directly comparable, and their sum is
interpretable.**

**NO SIGNIFICANCE TEST ON THE DECOMPOSITION. NO VERDICT LANGUAGE.** D3b reports
two bounds and their gap; **it does not re-adjudicate D2, which stands as read.**

---

## §8 INHERITED, UNCHANGED

Population `3ed3e286e633c2fc` (the 632 admitted of 684); D's frozen producer
`84011269d00eea6b` for pools, cells and `D_pair`; C v6's edge; the qualifying-cell
machinery identical to D's, or the benchmark imports a second difference.

**EXPOSURE: D2's result is SEEN BY EVERY SEAT and cannot be unseen.** D3b is
registered post-result by necessity; **what is blind is the decomposition, which
no seat has computed in any form.**
