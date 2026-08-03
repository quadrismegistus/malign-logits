# Registration M — the perturbation null for tail contraction

**STATUS AS DECLARED (2026-08-03 UTC, per the docket post carrying this hash):
draft; not in force as declared. Freeze state is recorded on the docket and in
git history. NO M QUANTITY HAS BEEN COMPUTED.**

    ADJUDICATES  the tail-contraction hypothesis ([3648], lacan's), which
                 explains L's monotone rung gradient as sharpening that thins
                 the distribution's tail
    ORDERED      [3661].1 -- one registered adjudication remains available and
                 it is the discriminating one
    INHERITS     Registration L @ 72e4b4a94d7c467e; cells
                 result_l_found_prose.json @ f883672020269b95;
                 result_l_tail_column.json @ e5a527acfd65e85e

---

## §0 THE EXPOSURE LEDGER — COMPLETE, AND IT IS WHY THIS REGISTRATION IS NARROW

**TWO PROTECTION ORDERS WERE BREACHED BEFORE THIS DOCUMENT EXISTED, ONE AT EACH
SEAT, WITHIN TWENTY MINUTES.** Both were disclosed immediately and in full;
neither disclosure converts a computed quantity into an uncomputed one.

    SEEN, lacan, [3653]     gold retained under base 3,611/4,268 (84.6%);
                            gold absent under base 657; EVICTED 115;
                            ADMITTED 14; gold base rank where retained
                            median 2, q1 1, q3 8, max 150
                            -- produced by building the column [3650] forbade
    SEEN, malign, [3656]    eviction rate by base rank, seven bands,
                            0.08% (rank 1) -> 41.43% (rank 51+);
                            eviction rate by MARGIN, six bands,
                            48.75% (<0.25 dec) -> 0.12% (>2 dec);
                            eviction by rank WITHIN margin bands -- flat
                            -- the conditional [3655] ordered frozen first,
                            plus a margin decomposition no order protected
                            because it did not exist until it was run

**BOTH SEATS HAVE SEEN THE MARGINAL AND THE CONDITIONAL. NEITHER HAS SEEN THIS
NULL.** The quantity registered below is not derivable from any published table:
it requires the ALIGNED probability of the gold word, which appears in no
artifact on the record.

**MECHANISM OF BOTH BREACHES, recorded because it predicts the next one: a
citation is a claim to have read, and a headline that reads as approval is
exactly the one that gets acted on unread.**

---

## §M1 THE QUESTION, AND WHY THE OBSERVED CURVE CANNOT ANSWER IT

Eviction rises monotonically with base rank and more sharply with MARGIN
(`log10(p_base_gold / theta)`, the decades a word must fall to be evicted).
**Within margin bands, rank is flat.**

**`retained` MEANS `p >= theta`, so a word's base probability IS its distance to
the eviction boundary.** Any perturbation of any kind evicts whatever sits
nearest the threshold. **The observed monotone curve is therefore predicted by
tail contraction AND by structure-free redistribution alike, and a curve both
hypotheses predict is evidence for neither.**

**THE ONLY DISCRIMINATING QUESTION: does the gold word's probability fall MORE
when it starts further from the peak, beyond the constant log-shift that moving
the same total mass without tail preference would produce?**

---

## §M2 THE NULL — DETERMINISTIC, SEED-FREE, ONE LINE

    For each (family, prompt) cell:
      THE WORD SET IS THE BASE-RETAINED SET.  Declared, because "retained"
      alone admits three readings and they are different nulls (§M2a).
          W = the words retained under BASE in this cell
      observed removed mass
          R = sum over w in W of max(0, p_base(w) - p_aligned(w))
      the NULL applies a single multiplicative factor lambda to EVERY word in
      W, chosen so that the same total mass leaves:
          lambda = 1 - R / sum(p_base(w) for w in W)
      NULL-predicted gold probability:  p_null_gold = lambda * p_base_gold
      REFUSAL: if lambda <= 0 the cell is EXCLUDED and the count PRINTS.
      (R cannot exceed the total base mass on W in fact; a declared refusal
      costs one line and an undeclared one costs a run.)

### §M2a `p_aligned` FOR AN EVICTED WORD IS A BOUND, NOT A COERCION

A word in `W` that fell below theta under alignment has NO ROW in the store.
**It is not missing data: the store's construction tells us `p_aligned(w) <
theta` exactly.**

    DECLARED: p_aligned(w) = 0 for such words.

**AND THE GROUND IS THAT IT IS THE CONSERVATIVE ENDPOINT OF A KNOWN INTERVAL,
NOT A CONVENIENCE.** The true value lies in `[0, theta)`. Taking 0 MAXIMISES `R`,
which MINIMISES `lambda`, which MAXIMISES `d_null`, which MINIMISES the excess
`e` — **the statistic the contraction hypothesis needs to be large.** The
alternative reading, excluding evicted words from `R` altogether, does the
opposite and inflates `e` toward finding contraction.

**This is not the absent-read-as-zero coercion this campaign booked against
today.** That defect is reading an UNMEASURED quantity as zero. Here the
quantity is BOUNDED BY CONSTRUCTION and the bound is taken at the end that
argues against the drafter's own null.

**WHY MULTIPLICATIVE AND NOT UNIFORM-ABSOLUTE:** a uniform absolute removal takes
the same amount from a p = 0.9 word and a p = 0.0011 word, which is tail-biased
BY CONSTRUCTION and would build the hypothesis into its own null. **The
multiplicative form is the unique structure-free choice: every word loses the
same FRACTION, so the log-shift is identical for all words and the null has no
preference for the tail whatsoever.**

**NO SEED. NO SAMPLING.** `lambda` is a closed-form function of quantities the
cell already holds; the null is one number per cell and reproduces exactly.

---

## §M3 THE STATISTIC — OVERSHOOT, BECAUSE THE NULL SATURATES IN THE TAIL

**THE NULL IS A STEP FUNCTION:** `null_evicted(w) = 1` iff `margin(w) < d_null(cell)`,
deterministic. **So below the line the null already predicts certain eviction and
CONTRACTION CANNOT SHOW UP THERE AT ALL** — there is no room above a prediction of
certainty. **Everything discriminating lives just ABOVE the line: words a
structure-free shrink says should have survived, which alignment evicted anyway.**

**An earlier draft made the primary `Spearman(observed_evicted - null_evicted,
margin)`. That is POSITIVE BY CONSTRUCTION** — `+1` occurs only above the line and
`-1` only below it — **and it was registered with a negative rho as contraction's
signature, so it would have returned a decisive-looking refutation regardless of
the truth.** Withdrawn; recorded here because the defect was in the expression,
not the prediction.

    OVERSHOOT   for each base-retained gold word with margin(w) > d_null(cell):
                  s(w) = margin(w) - d_null(cell)    decades of headroom the
                                                      structure-free shrink
                                                      says the word had
                  observed_evicted(w) in {0, 1}
    PER FAMILY  rho = Spearman( observed_evicted(w), s(w) ) over that family's
                words with s > 0
    COMBINED    Stouffer on the signed z over the 34 BASE CLUSTERS, EQUAL
                WEIGHT PER CLUSTER, a cluster's z the unweighted mean of its
                families' -- L §L5's grammar verbatim, for L §L5's reason

    NEGATIVE rho   evictions concentrate just above the line and decay with
                   headroom -> BOUNDARY BLUR
    rho ~ 0 with evictions occurring at LARGE s as well
                   -> eviction does not decay with headroom -> CONTRACTION
    no evictions with s > 0 -> the null holds exactly

**THE SIGN IS INVERTED FROM THE INTUITION AND THAT IS THE POINT: under a
deterministic null, contraction shows as evictions that DO NOT decay with
headroom.**

### §M3a TIES, UNDEFINED RHO, AND THE FAMILIES THAT DROP OUT

**`observed_evicted` is almost all zeros: 115 evictions over 3,611 base-retained
words, ~2.6 per family against 97.** Two consequences, both declared here because
Registration L's retained rung lost eleven families and three clusters to exactly
this and cost nothing BECAUSE ITS DOCUMENT SAID SO.

    THE p IS EXACT, AND THE STATISTIC'S FORM IS WHY.  `observed_evicted` is
                     BINARY, so rho is a monotone function of the ranks of `s`
                     among the evicted -- the Mann-Whitney / Wilcoxon RANK-SUM
                     statistic, whose null is CLOSED AND EXACT at every k.
                     DECLARED: each family's p is the EXACT two-sided rank-sum
                     p of `s` among the evicted against `s` among the
                     survivors; the signed z derives from THAT p, with the
                     sign taken from rho.  **NO ASYMPTOTIC SPEARMAN p ENTERS
                     THE STOUFFER.**
                     GROUND, measured on SYNTHETIC vectors: at k = 1 the
                     asymptotic p runs 0.2265 against an exact 0.3093 -- 36%
                     ANTICONSERVATIVE, in exactly the thin families, in the
                     direction that manufactures significance.
                     CONSEQUENCE: nobody is dropped and nothing is reweighted.
                     A one-eviction family contributes a correctly-WIDE,
                     near-zero z instead of an incorrectly-narrow one, and is
                     quiet automatically rather than by policy.
                     **THE RULE IS DERIVED FROM THE STATISTIC'S FORM, NOT FROM
                     THE COUNTS** -- identical at k = 1 or k = 50, requiring
                     only that a binary arm exists, which is true of the design
                     and not of the data.  With no seat blind (§M3d), that is
                     the only kind of choice still available.
                     EXACT-TIE REVERSION: `s` is continuous so ties are
                     measure-zero; if any occur the family reverts to the
                     tie-corrected normal approximation and the COUNT PRINTS.
    A >=2-EVICTION FLOOR IS WITHDRAWN AS MOOT.  It was proposed against an
                     INFLUENCE defect that does not exist: equal weight is a
                     rule about DEPENDENCE (one base must not count seven
                     times) and declines to weight by precision by design.
                     **A sensitivity whose motivating defect has been dissolved
                     by derivation is machinery without a question, and
                     printing it would imply the question is live.**
    TIE CORRECTION   "Spearman" names several implementations differing
                     precisely in tie handling, and with ~94 ties in 97 points
                     the correction DOMINATES.  DECLARED: the tie-corrected
                     Pearson-on-midranks form (`scipy.stats.spearmanr`'s
                     definition), computed on midranks with the standard
                     tie correction.  Named because two seats implementing
                     "Spearman" would otherwise disagree and neither would
                     have miscoded.
    UNDEFINED RHO    a family with NO variance in `observed_evicted` among its
                     s > 0 words -- no evictions above the line, or all of them
                     -- yields NO rho and NO z.  DECLARED: that family is
                     EXCLUDED and the count PRINTS; a cluster losing all its
                     families vanishes and the cluster count PRINTS BESIDE THE
                     STOUFFER, which therefore states its own n rather than
                     inheriting 34.
    FLOOR            if fewer than 20 clusters survive, the primary is reported
                     as UNDERPOWERED and no Z is read.  Declared now because a
                     Stouffer over a handful of clusters is a number without a
                     test behind it.

### §M3e FEASIBILITY, PRICED — and the FALLBACK declared before anyone sees which form flatters which answer

**Computed on the OBSERVED arm only** (the `evicted` column of
`result_l_tail_column.json` @ `e5a527acfd65e85e`) — **no null, no `d_null`, no
`s`, no `lambda`; nothing on the protected list.** Ordered at [3678](b).

    families with base-retained gold words        44
    families with ZERO evictions                  14   -> no variance, NO rho
    families with exactly ONE eviction             6   -> rho defined, one point
                                                          drives it
    families with >= 2 evictions                  24
    largest family eviction count                 23
    base clusters represented                     34
    clusters keeping >= 1 testable family         28   (floor is 20)

**THE PER-FAMILY FORM IS FEASIBLE AND THIN: 30 of 44 families yield a rho, 28 of
34 clusters survive, and six of those rhos rest on a single eviction.** The floor
is cleared with eight clusters to spare.

    DECLARED FALLBACK, chosen now: IF fewer than 20 clusters survive the
    per-family form, the primary becomes the PER-CLUSTER form -- one rho per
    BASE CHECKPOINT over ALL its families' s > 0 words pooled, Stouffer over
    the surviving clusters, same tie rule, same floor.  Clustering is
    PRESERVED, not abandoned: the pooling is WITHIN a base, never across.
    IF fewer than 20 clusters survive THAT, the primary is UNDERPOWERED and
    NO Z IS READ.

**The fallback is declared here rather than reached for later because the choice
is not independent of the answer: a form that drops families with no events drops
exactly the families where nothing happened.** On today's counts the fallback is
NOT triggered — **and it is declared anyway, because a fallback named after the
degeneracy is visible is a forking path with one extra step.**

### §M3b THE ESCAPES — the mirror arm, one column

`observed_evicted = 0` where `null_evicted = 1` are words the structure-free
shrink evicts and alignment did not. **Same machinery with `s(w) = d_null -
margin(w)`; contraction predicts FEW escapes deep below the line, blur predicts
many.** Reported beside the primary, with the same tie, undefined-rho and floor
rules.

### §M3c THE PRE-CHECK, WHICH IS NOT A TEST OF THE HYPOTHESIS

McNemar on (observed evicted vs null evicted) per family, Stouffer over clusters.
**It answers only "the observed perturbation is not a uniform shrink."** `R`
counts LOSSES ONLY and the null spreads that mass over every word INCLUDING
GAINERS, so a real loser lost more than its null share BY CONSTRUCTION and this
returns `b > c` under ANY non-uniform perturbation. **A significant result here
is not evidence for tail-direction and must never be reported as one.** It runs
first; if it comes back null the whole exercise is void.

### §M3d THE BANDS ARE A FORMULA, NOT A CHOICE

**No seat is blind: `[3656]`'s eviction-by-margin table is public and both seats
have read it ([3672]).** So no named edge can be innocent of the curve it cuts.

    DECLARED: EQUAL-n QUANTILES of `s` over the pooled s > 0 population,
    DECILES, computed by the producer from the column.  No fixed decade edges
    are available as an option.  The band table is DESCRIPTIVE and disambiguates
    monotone from peaked; the primary is the rho.

## §M4 THE READING RULE, FIXED BEFORE ANY NUMBER

**THE SIGN IS INVERTED FROM THE INTUITION. Under a DETERMINISTIC null, contraction
shows as evictions that DO NOT DECAY with headroom, not as a negative
correlation.** An earlier draft's rows read the other way and are withdrawn with
the statistic they belonged to.

    Stouffer Z significantly NEGATIVE   evictions concentrate just above the
                                        null's line and decay as headroom
                                        grows: BOUNDARY BLUR.  Contraction NOT
                                        SUPPORTED -- the excess is a
                                        threshold-spread effect
    Z not distinguishable from zero,    eviction does not decay with headroom:
    WITH evictions present at large s   the tail is hit beyond what proximity
                                        to the boundary explains.  CONTRACTION
                                        SURVIVES its discriminating test
    no evictions with s > 0             the structure-free null accounts for
                                        every eviction exactly; contraction is
                                        dead and so is boundary blur
    fewer than 20 clusters survive      UNDERPOWERED (§M3a).  No Z is read and
                                        no row above is entered

**THE BAND TABLE DISAMBIGUATES AND THE RHO ALONE DOES NOT.** A rho near zero is
consistent with "evictions at all headrooms" AND with "no evictions at all"; the
decile table separates them and is required, not optional (§M3d).

**NO MDE. A cluster-level test needs the between-cluster SD of the difference and
that is a quantity this run produces** — L's reason, unchanged. **A null reads
"not detected at this n" and licenses no claim in either direction.**

**NO VERDICT LANGUAGE ON L.** M adjudicates one MECHANISM. L's rungs, its
gradient and its 8:1 eviction asymmetry stand as read whatever M returns.

---

## §M5 WHAT WOULD MAKE THIS THE WRONG MEASUREMENT

**COUNTERSIGNED SEPARATELY. At least one entry must be a way this could be wrong
IN THE DIRECTION THE DRAFTER FAVOURS — and the drafter argued against
contraction at [3652], so the honest entry is one that would wrongly KILL it.**

    1. THE NULL MAY BE TOO STRONG, AGAINST CONTRACTION AND IN MY FAVOUR.
       A multiplicative shrink is structure-free but it is not the only
       structure-free choice, and real alignment moves mass BETWEEN words as
       well as out of them.  `R` counts only mass REMOVED; a cell where mass
       is redistributed rather than lost has a small R and a lambda near 1,
       so `e` absorbs redistribution that is not contraction.  This inflates
       |e| and could produce a NEGATIVE rho with no tail preference present.
       **It errs toward finding contraction, which is against the drafter's
       stated scepticism and must be said in that direction.**

    2. **THE EVIDENCE IS 115 EVENTS AND THE POWER MAY NOT BE THERE.**  The
       overshoot rho is driven entirely by evictions above the null's line, of
       which there are AT MOST 115 across 44 families -- ~2.6 each against 97
       words.  §M3a declares the exclusion and floor rules; it cannot conjure
       power.  **A null here is very likely to mean "too few events", and the
       reading rule's UNDERPOWERED row exists so that outcome is named rather
       than read as an answer.**

    3. **THE CENSORING CRITIQUE APPLIES TO THE SECONDARY, NOT THE PRIMARY, AND
       IT IS IN THE DRAFTER'S FAVOUR THERE.**  A survivors-only statistic is
       bounded -- `d_obs <= margin`, since a larger drop would have evicted the
       word out of the sample -- so it is clipped toward zero, toward the
       drafter's stated scepticism.  **The overshoot primary is not subject to
       this: it uses the eviction EVENT, which is the boundary observation
       itself.**  That is the whole reason the primary changed, and the
       secondary carries the caveat wherever it is reported.

    4. THETA IS A STORE PARAMETER, NOT A MODEL PROPERTY.  Margin is defined
       against theta = 0.001.  Every quantity here is relative to a retention
       threshold the pipeline chose.

    5. GOLD IS NOT A TAIL OBJECT IN GENERAL -- median base rank 2 (§0).  The
       hypothesis was reshaped by that marginal before this document existed.

---

## §M6 UNIT, POPULATION, AND WHAT MUST BE PRODUCED FIRST

    POPULATION   the cells of result_l_found_prose.json where the gold word is
                 RETAINED UNDER BASE AND UNDER ALIGNED.  Evicted cells are
                 excluded of necessity (§M5.2) and their count PRINTS.
    UNIT         the cell for rho; the BASE CHECKPOINT (34) for combination
    REQUIRED     `p_aligned_gold` and `p_base_gold` per cell, and the cell's
    COLUMN       retained-word probability vectors sufficient to compute R.
                 **NO ARTIFACT ON THE RECORD HOLDS THESE.** The column is
                 produced by the descriptive seat AFTER this registration
                 freezes, per [3661].5.

**NOTHING COMPUTES BEFORE THE FREEZE POSTS.**
