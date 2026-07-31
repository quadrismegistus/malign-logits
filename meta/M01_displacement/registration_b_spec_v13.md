# Registration B v13 — the high-mass decomposition

Drafted against the amended ten-item freeze checklist ([1336].4 as corrected by
[1340].1). **v1 ([1321]) is superseded; it is kept as record because three of its
choices were wrong and the reasons are instructive.**

**Pen freezes the audited hash; malign audits for conformance and efficacy.**

---

## §0 QUESTION, GIVENS, EXPOSURE

P1's weighted figure exceeds its unweighted one (+0.221 vs +0.123, [1247]).
**Is there arousal structure along the mass ordering BEYOND what baseline
probability explains?**

**SEEN-GIVENS**, named so they cannot be re-sold as findings: the riser
concentration profile ([1244]); the weighted/unweighted gap and its size
([1247]); `corr(arousal, log P) = +0.08` by role ([1334], run by the exposed
seat and booked as a declared given at [1336].3).

**BLIND, at every seat:** beyond-probability mass-arousal structure per role.

**EXPOSURE:** lacan total on arousal-by-movement; malign on
concreteness-by-movement and the norm-free mass distribution, blind to arousal
ratings.

---

## §1 IMPORTED, NOT RESTATED

    from m01_norms import (norm_key, lookup, lemma_candidates, is_function_word,
                           load_norms, cell_roles, DISPLACING_AT, CONTROL_BELOW,
                           MASS_COVERAGE_FLOOR, PERM_SEED, N_PERM)
    from m01_concentration import frozen_population, operation_edges, EDGE, RULE

Population `fd3f1479…` × `e4c507eb…`. English only, declared. Function words
excluded, lemma repair applied, z-anchored to the source database, residual a
bin.

---

## §2 UNIT, RANKING, QUALIFYING — item (iii), (iv)

    UNIT           the (cell, role) — one CUSUM path per qualifying cell-role
    RANKING        |delta| = |Q - P|, descending. BOTH ARMS, unchanged.
    QUALIFYING     >= 3 rated non-function words of that role

**THE BAR IS 3, NOT 2, AND THE REASON IS STRUCTURAL.** A 2-word role is
PERMUTATION-INVARIANT under this statistic: centred arousal is `(a, -a)`, the
cumulative sum is `(a, 0)` in one order and `(-a, 0)` in the other, and
`max|C| = |a|` either way. **The cell adds the same constant to the observed T and
to every null draw — zero information, at any sample size.** Synthetic corpora of
only 2-word roles return mean p = 0.878 with nothing below 0.05. ~7% of otherwise
qualifying role-cells leave; the count prints.

**`|delta|` is ~81% a probability ranking (Spearman +0.830 faller / +0.600 riser
against pre-probability rank, [1328]/[1330]). It is used anyway**, because the
confound needs two links and the second is absent: `corr(arousal, log P) = +0.08`.
**The alternative — ranking risers by excess — inverts the coupling to −0.964 and
is worse ([1330]).**

**PRINTED CONTROL WITH ITS REOPEN BAR, set before any run:**
`|corr(arousal, log P)| >= 0.15` within the run's own population **reopens the
ranking question.** 0.15 separates the observed cases: arousal 0.08 inert,
concreteness 0.177 live.

**DIMENSION WARNING, in the spec text: run B on CONCRETENESS and the
residualisation becomes load-bearing** — `corr(concreteness, logfreq) = +0.177`.
This spec's inertness argument is about arousal and transfers to nothing.

**DOMAIN FACTS, printed:** the median cell has **one** rated role-word;
**~48% of cell-roles qualify**. A 1-word role carries no information about
whether mass relates to arousal — a population statement, not a silent drop.

---

## §3 THE STATISTIC AND THE NULL — item (i)

Per qualifying cell-role:

    centre    a_w = z_arousal(w) - mean(z_arousal over that role's rated words)
    order     words by |delta|, descending
    path      C_j = sum of a_w over the first j words
    cell stat s = max_j |C_j| / M_cell         # M_cell = THIS cell's own
                                              # permutation mean of max|C|

Aggregate:

    T = sum of s over all qualifying cell-roles

**THE NULL IS ONE JOINT PERMUTATION.** Permute the mass order **independently
within every qualifying cell, simultaneously**; recompute T. **10,000 draws, seed
20260731. One p-value, resolution 1/10,000, independent of cell sizes.**

**There is no combining step and therefore nothing to assume.** Fisher and
Stouffer both require `p ~ U(0,1)` under H0; per-cell permutation p-values live on
a lattice whose spacing is set by the number of DISTINCT statistic values —
generically `2^(n-2)`, far coarser than the `n!` orderings suggest — and no
weighting repairs that. Weighting corrects unequal information, not a lattice.

**THE DIVISOR IS THE CELL'S OWN PERMUTATION MEAN, `M_cell`** — the mean of
`max|C|` over the permutations of that cell's own values. Exact by enumeration at
n <= 6, estimated from the joint null's own draws above.

**WHY, AND THE REASON IS A MEASUREMENT: `sqrt(n)` mis-weighted by SIZE and `E_n`
mis-weighted by SPREAD.** Within-cell arousal spread varies **p90/p10 = 2.66**
across the 8,340 qualifying cell-roles (2.56 fallers, 2.72 risers), against a bar
of 1.5 set blind before the number existed. **`s` scales linearly with spread, so
under `E_n` a lexically extreme cell contributed 2.66x a mild one at the same
size — a weighting by what the prompt is about.**

**SPREAD IS NOT SIGNAL. Structure ALONG THE ORDERING is the registered object**,
and a cell whose words are extreme but arranged randomly has none of it.

**THE IDENTITY, stated in the direction it runs.** `sd_cell x E_n_unit` is a
**Jensen-biased-low ESTIMATOR** of `M_cell` — the two divisors correlate at
0.9986-0.9994 and are one quantity, one computed and one estimated from n points.
The bias multiplies `s` by a size-dependent factor — **1.38 at n=3** falling to
1.04 at n=20 — **which is precisely its H0 failure**: that factor IS the H0 mean
of the estimator form, and it re-introduces a size gradient in the direction
`E_n` was adopted to remove.

    normaliser        size-fair   spread-fair   signal preserved
    sqrt(n)              NO           no             yes
    E_n                  yes          NO             yes
    sd_cell x E_n_unit   NO           yes            yes
    M_cell (own-mean)    YES          YES            YES (99-101% of E_n's)

**Two seats, two implementations, two seeds, same verdict.**

**§8 GATES ON THE MEAN ACROSS SIZES *AND SPREADS*, AND PRINTS THE CV UNGATED.**
The pass condition is `E[s|n,sd] = 1.000 +/- 0.02` **at every size AND every
spread in {0.5, 1.0, 2.0}**.

**THE SPREAD AXIS IS NOT DECORATION: A SIZE-ONLY GATE WOULD HAVE CERTIFIED `E_n`.**
Measured — `E_n` at sd=1 returns 1.001, 1.003, 0.995, 1.001, 0.998 across
n=3..40, passing +/-0.02 at every size. **The check that exists to validate the
normaliser was blind to the exact defect that replaced it**, because it varied
only the axis the loser was already fair on.

**The CV-by-n column is a REPORTED PROPERTY WITH NO PASS CONDITION**, and under
`M_cell` it RISES with n — 0.166 at n=3 to 0.310 at n=40 — the OPPOSITE of
`E_n`'s direction (0.563 falling to 0.332).

**A CUSUM is used because no shape is predicted.** A step at the top mover, a
monotone gradient, a U, or an effect confined to the extreme all produce a large
excursion; only a flat relationship produces none.

**PER-CELL PERCENTILES PRINT AS DESCRIPTION ONLY**, with the TRUE lattice stated
— and it is far coarser than `n!` suggests, because `max|C|` is invariant under
many permutations:

    n              2    3    4      5        6       7       8
    n!             2    6   24    120      720    5040   40320
    GENERIC 2^(n-2) 1    2    4      8       16      32      64
    OBSERVED       1    2    4    8-9    15-19   30-38   63-77
    FINEST p     1.00 .500 .250  .1111   .0526   .0333   .0159
    reaches .05?  no   no   no     no       NO     YES     YES

**THE COUNT IS A PROPERTY OF THE VECTOR, NOT OF n ALONE.** The GENERIC (modal)
count is `2^(n-2)` — exactly, at every size measured — but individual vectors
depart in BOTH directions through accidental numerical coincidence. Exact
rational enumeration, 3,000 vectors at n=5 and 1,500 at n=6:

    n=5   {8: 2564, 9: 436}
    n=6   {15: 26, 16: 767, 17: 316, 18: 379, 19: 12}

**`2^(n-2)` is the TYPICAL value, NOT a floor and NOT a bound.** This is why two
seats enumerating n=6 got 17 and 16, and why a third would likely get 18: both
were right about the vector each drew.

**THE `FINEST p` ROW USES THE MAXIMUM OBSERVED COUNT, because more distinct
values means FINER resolution** — the favourable direction for reaching 0.05, so
reasoning from it is conservative for the exclusion claim.

**No n <= 6 role produced a per-cell p at or below 0.05 across 5,500
exact-rational enumerations at n=6 (TWO INDEPENDENT GENERATORS, two seats), NOR
under DIRECTED CONSTRUCTION — designed vectors (powers, primes, sparse ratios)
topped out at 17 distinct values. A vector yielding >= 20 would reach 0.05 and
none was found. Margin: 0.0526 against 0.0500.** That is 39.9% of qualifying
role-cells.

**THE CLAIM IS EMPIRICAL, NOT A PROOF, AND ITS STRENGTH IS THE ATTACK RATHER
THAN THE SAMPLE SIZE.** "We sampled and did not find one" and "we sampled, TRIED
TO BUILD one, and did not find one" are different claims; only the second answers
the obvious objection. Either sampler could have returned 20 and neither did. `n!` overstates the resolution by three orders of magnitude at n=8. **They are not inputs to any test**, which is why the joint
null was adopted; the description says so with the enumerated numbers.

---

## §4 POWER, NOT CALIBRATION — item (ii)

**The joint null is calibrated at every role size, and under `M_cell` the per-cell
CONTRIBUTION is flat in BOTH size and spread.** Power still varies: small roles admit few
DISTINCT STATISTIC VALUES — two at n=3, not the six orderings — so they carry
less information per cell, but they no longer carry less WEIGHT than their
information warrants, which is what `sqrt(n)` imposed. **~19% of qualifying roles have n = 3.**

**THE SECOND DECLARED COST, and under `M_cell` it runs OPPOSITE to `E_n`'s:**
cells contribute EQUAL EXPECTATION, and the relative noise RISES with size — CV
0.166 at n=3 against 0.310 at n=40. **A LARGE cell casts an equal vote with 1.9x
the relative noise of a small one.**

**AND THE LOW CV AT SMALL n IS CONSTRAINT, NOT PRECISION.** A 3-word role admits
only two distinct values of `max|C|` (§3's lattice), so `s` is confined to a
narrow set — quiet because it cannot move, not because it is well measured.
**Reading a small CV there as reliability would invert the truth**: those cells
are the least informative and the least variable at once.

Prints: the per-size contribution to T, and the aggregate by role-size band
beside the pooled figure.

---

## §5 THE CURVE — item (v)

Reported **descriptively**, stratified by role-size band, with n per band: mean
centred arousal by position in the mass ordering. **No single band is the result;
quoting one band alone is quoting outside the registration.**

**Stratification by role size is not a hedge.** Neither mass share nor normalised
rank is size-neutral — share is mechanically `1/n`, and normalised rank lets size
determine where on the axis a cell can appear. **There is no size-neutral shared
axis, which is why the TEST does not use one.**

**A third curve prints: mean centred arousal over baseline probability**, so a
reader can see whether arousal tracks mass beyond tracking P.

---

## §6 DIRECTIONAL READOUTS — item (vii), (viii)

Pre-named, tested under the **same joint null**, reported beside T and never
instead of it:

    R_faller = mean centred arousal of the TOP-MASS faller, across qualifying cells
    R_riser  = mean centred arousal of the TOP-MASS riser,  across qualifying cells

**B1 (primary):** `R_faller > 0`. **B2 (secondary, weaker prior declared):**
`R_riser < 0`.

**B1 IS THE HARDER ARM AND THE SPEC SAYS SO IN ADVANCE:** fallers are more
coupled to probability (+0.830) than risers (+0.600), so the faller ordering
carries more probability structure to see past. **A B1 null is weaker evidence
against the frame than a B2 null would be.**

---

## §7 CONTROLS — item (iii), (x)

**(a)** Control sites (`median departed < 0.02`): T and both readouts ~ chance.
**If controls move, nothing else is reportable.**

**(b) Frequency:** `A_logfreq` and `corr(arousal, logfreq)` print within the
high-mass set and within the tail separately. **A selection check clears the cut
it was run on and no other** — the +0.08 above is over all moving words, not over
mass-conditional subsets.

**(c) DENSE-STRATUM SCOPE, named:** role size couples to concentration at
**−0.756** ([1267]), and qualifying selects on role size. **So T is measured
disproportionately on dense, less-concentrated cells.** Any claim states this;
the banded aggregate at §4 is how a reader checks whether the effect lives only
in one stratum.

---

## §8 CALIBRATION — item (ix)

**Runs unconditionally, never behind a flag** ([1148]). Seed 20260731, 10,000
draws.

**THE GRID IS TWO-AXIS AND THE SPREAD AXIS IS ANCHORED TO THE POPULATION, NOT
INVENTED.** Synthetic corpora with arousal assigned at random must satisfy:

    GATE      E[s | n, spread] = 1.000 +/- 0.02
    SIZES     n = 3, 4, 6, 10, 20, 40
    SPREADS   the population's own SPREAD DECILES, p10 = 0.470 through
              p90 = 1.252 ([1387]'s measurement of 8,340 qualifying cell-roles)
    AND       uniform p across many draws at every role-size mixture

**A SIZE-ONLY GRID WOULD CERTIFY `E_n`, THE NORMALISER THIS SPEC EXISTS TO
REPLACE.** Measured: `E_n` at spread 1.0 returns 1.001, 1.003, 0.995, 1.001,
0.998 across n=3..40 — passing at every size. **It fails only when the spread
axis is varied, and it fails there by a factor of 2.66 across the population's
own deciles.** The gate must FIRE on `E_n` and PASS `M_cell`; a gate that clears
both has been re-indexed wrongly.

**THE GATE CHECKS THE IMPLEMENTATION, NOT THE CHOICE.** Under `M_cell` the pass
is an IDENTITY: `s` is `max|C|` divided by the mean of `max|C|` over that cell's
own permutations, so `E[s] = 1.000` holds by construction at every size and every
spread. **A pass therefore certifies only that the code computes what the spec
says — it is a wiring check, and it cannot tell you the normaliser was well
chosen.**

**THE INFORMATIVE HALF IS THE `E_n` FIRE.** That the gate DISCRIMINATES — clears
`M_cell`, fails the normaliser this spec replaced — is what makes it evidence
rather than tautology, and it is why the rejected divisor is retained in the
harness as the known-bad case. **A mutation control is not decoration here; it is
the only part of §8 that can fail for a reason about the design.**

**A CHECK INHERITS THE AXES OF THE DEFECT IT WAS BUILT FOR.** This section's
earlier form found the 2-word degeneracy and the `sqrt(n)` gradient because both
were size-indexed, and would have certified a spread-graded normaliser for the
same reason. **Re-index the gate whenever the statistic changes, or it certifies
exactly what it last cleared.**

**The CV column prints ungated: 0.160 at n=3 RISING to ~0.32 at n=20-40.**

---

## §9 FALSIFIER — item (vi)

**If T's p is not significant, there is no arousal structure along the mass
ordering beyond probability, and the [1250].3 sign observation DIES as evidence.**
It was an unregistered reading noticed inside a caveat about a missing column;
this is what converts it into a test, and the test is allowed to kill it.

**No magnitude is predicted.**

---

## §10 WHAT I HAVE NOT RUN

**No mass-conditional arousal value has been computed at any seat.** Everything
above is arithmetic written and unexecuted.

**v1's choices that were wrong, each killed from outside this seat:** the top-1
cut (malign: the median cell has one rated word), the monotone expectation (RH: no
reason to expect a simple fall), the mass-share axis (malign: share is mechanically
`1/n`), and the excess ranking (malign: it inverts the coupling to −0.964). **A
fifth — the `sqrt(n)` divisor — was flagged by its author and then measured by
malign independently in the same hour; the measurements agree.**

**And the §8 check's record is the reason its EPISTEMICS are stated there rather
than assumed.** It did find the 2-word degeneracy and the `sqrt(n)` gradient
before a line of the producer existed — **and it would have certified `E_n`, the
normaliser this spec exists to replace, because it was indexed on the axis those
two defects happened to share.** What actually chose `M_cell` was the
signal-preservation test at fixed values, the 2.66 spread measurement, and the
structural identity with `sd_cell x E_n_unit`; **none of those live in §8, and
under `M_cell` the gate's pass is an identity that cannot fail unless the code is
wrong.**

**So the check earned a narrower place than an earlier draft of this section
claimed: it is a wiring check whose evidential content is the half that FIRES.**
