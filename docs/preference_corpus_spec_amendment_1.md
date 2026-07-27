# Amendment 1 to the preference-corpus spec — DRAFT, not registered

2026-07-26. For lacan's audit, then RH's go. Amends
`docs/preference_corpus_spec.md` after the v1 run was found invalidly
constituted. **No `D` has been computed for any pair named here.**

## Why an amendment is needed

The v1 gate ran with six of seven markers outside the registered
frequency-comparability band. The precondition was noticed, written into the
marker declaration as a "registered weakness", and the run proceeded — a
condition logged instead of met. The gate was therefore never validly
constituted and the run produced no finding in either direction.

Three drafting defects made that possible, and this amendment fixes each.

## A1. The shrink clause is reworded so it reaches its own case

**Current text:** "If fewer than seven can be **attested** individually, the
slate shrinks to the number that can…"

**The clause did not go unused. It fired correctly and passed.** (lacan's
textual reading, which is sharper than "the remedy was overlooked" and is
checkable by anyone against the spec.)

Candidacy at lines 225–232 requires attested **and** frequency-comparable — so a
non-comparable marker is not a candidate at all. But the shrink clause at
lines 259–262 triggers on "fewer than seven can be **attested** individually",
and *attested* refers back to the preceding sentence about stating external
grounds. All seven v1 markers **were** attested; every one has grounds in the
RLHF literature. Seven attested is not fewer than seven, so **the clause's own
test was satisfied while candidacy was violated six times over**.

**The general defect: a fallback clause must be keyed to the same predicate as
the requirement it protects.** Here the requirement was attested-AND-in-band and
the fallback read attested alone, so a slate could satisfy the fallback while
failing the thing the fallback existed to protect. No amount of care in
execution would have surfaced it, because the document said the slate was fine.
That is why disclosure-substitution was the path of least resistance: the only
clause that could have blocked the run was reading a different variable.

An unused remedy is fixed by diligence. A mis-keyed one is not.

**Amended text:** "If fewer than seven candidates are **valid** — attested AND
in band AND adequately powered, all three — the slate shrinks to the number that
are, and k is recalibrated to that number with a freshly computed
false-certification rate."

## A2. Slate size is derived, never assumed

**New requirement, from lacan:** *find the valid candidates first, then
calibrate the gate to how many there are.* Fixing k in advance felt like
pre-registration discipline and was the mechanism that produced the invalid
slate: a count requirement that could only be met by relaxing a quality
requirement.

A slate is now constructed by enumerating attested candidates, filtering to
valid ones, and setting k from what survives — not by filling to a number fixed
beforehand.

## A3. Validity is checked per pair, by closed form, before selection

Power is exact in the counts, so no scaling law or anchor is needed:

```
SE(D) = sqrt(1/c_s + 1/r_s + 1/c_t + 1/r_t)
MDE   = 2.49 * SE(D)          # 80% power, one-sided alpha .05
```

**Validity condition, evaluated on the PAIR and not on the scarcer member:**

```
1/c_s + 1/r_s + 1/c_t + 1/r_t  <  (ln(median chain MDE) / 2.49)^2
```

For hh that is **< 0.004881**. As a rule of thumb this means ~820 combined
occurrences if the partner is large, or ~1,639 each if the members are
comparable — but the rule of thumb is not the test; the inequality is.

**Per-corpus anchors must appear on the face of every MDE table.** hh's
threshold is 0.174 (its median chain MDE, 1.19×); pku's is 0.405 (1.50×). pku's
is 2.3× laxer *because pku's chains are badly measured*, so clearing pku is a
weaker fact, not a second confirmation. Since pku is descriptive-only, a pku
certification licenses nothing the test can use.

## A4. The registered slate

Enumeration of 62 attested candidates across six documented categories of RLHF
register preference gave 9 in band and **7 valid**. k is therefore 7 — derived,
not assumed.

| # | pair | category | hh MDE |
|---|---|---|---|
| 1 | `require` → `prefer` | directive softening | 0.092 |
| 2 | `force` → `encourage` | directive softening | 0.104 |
| 3 | `angry` → `concerned` | emotional de-escalation | 0.135 |
| 4 | `entirely` → `mostly` | quantifier qualification | 0.136 |
| 5 | `totally` → `fairly` | quantifier qualification | 0.149 |
| 6 | `completely` → `largely` | quantifier qualification | 0.155 |
| 7 | `demand` → `request` | directive softening | 0.160 |

All below hh's 0.174. **Owed before registration:** an individual
external-grounds line for each of the seven. Being in band and powered makes a
pair *eligible*; it does not make it *attested*, and A1's validity condition
requires both.

**On "clears in both corpora" — do not write it.** Only `force`→`encourage`
(0.104 hh, 0.130 pku) is under hh's stricter threshold in both. `demand`→`request`
is 0.206 in pku: clears pku's laxer 0.405, **fails hh's 0.174**. Since pku is
descriptive-only, the honest count of pairs meeting a meaningful bar in both is
**one**, not two.

### A4b. The v2 slate is WITHDRAWN — correlated markers invalidate the rate

lacan's ruling, and it is not a breadth concern but a second
calibration-validity failure, running the *opposite* way from A5's.

The seven pairs are three constructs, not seven draws:
`entirely`→`mostly`, `totally`→`fairly`, `completely`→`largely` are one
preference in interchangeable words; `require`→`prefer`, `force`→`encourage`,
`demand`→`request` are a second; `angry`→`concerned` is a third. Blocks of
**3, 3, 1**. Within a block markers fire or fail together.

**The binding defect is POWER, not false certification.** (Grounds corrected by
lacan against its own initial ruling; verified here.) Solving back from the
recomputed 0.000113 at 5-of-7 gives a per-candidate null firing probability of
0.0912. Then:

| quantity | independent | blockwise (3,3,1) | effect |
|---|---|---|---|
| false certification | 0.000113 | **0.0083** | 74× understated — but still an order of magnitude *inside* the 0.10 standard, so **non-binding** |
| power at the anchor | 0.852 | **0.640** | **overstated by 0.212 — this binds** |

A gate at 0.64 power fails on a *working* instrument more than a third of the
time, and under the registered design a failed gate produces an
**insensitivity finding**. So the correlation does not risk certifying a dead
instrument; **it risks condemning a live one** — arriving at precisely the
conclusion withdrawn today, by a different route.

**And k=5 is decorative.** With blocks of 3, 3, 1, reaching five requires *both*
three-blocks; three plus the singleton is four. The singleton can never decide
anything, so 5-of-7 is really **2-of-3-constructs** wearing a stringent number.

**The general form, worth the spec:** decoy-based calibration assumes candidates
are *independent draws matched to the decoy distribution*. v1 broke the
**matching** (out-of-band markers, tighter nulls) and was therefore
**conservative**. v2 breaks the **independence** (correlated near-synonyms) and
is therefore **anti-conservative**. Any slate property that breaks
exchangeability with the decoys invalidates the rate; the direction depends on
which property broke.

### A4c. Slate size derives from CONSTRUCTS, not pairs

A2 is extended one step: cap at **one marker per construct**, then calibrate k
and the floor to that count with a freshly simulated rate.

Construct yield from the 62-candidate enumeration (hh):

| construct | candidates | in band | powered | best pair |
|---|---|---|---|---|
| quantifier qualification | 10 | 3 | 3 | `entirely`→`mostly` (0.136) |
| directive softening | 7 | 3 | 3 | `require`→`prefer` (0.092) |
| emotional de-escalation | 6 | 1 | 1 | `angry`→`concerned` (0.135) |
| hedging / epistemic | 7 | 0 | 0 | — |
| pejorative → neutral | 11 | 2 | **0** | — |
| refusal softening | 7 | 0 | 0 | — |
| blame → attribution | 6 | 0 | 0 | — |

**Three constructs, not seven.** Four of seven enumerated categories yield no
valid pair at all, and pejorative→neutral has two in-band pairs that both fail
power. So the honest slate is three markers and the gate is k-of-3.

### A4d. The construct-level grid (committed at `f9fd3c1` before results were read)

One marker per construct restores independence by construction, so the ordinary
grid machinery applies unchanged — no blockwise estimator is needed for a
structure the cap removes (lacan). Floors p50–p95 × k ∈ {1,2,3}, 4M draws,
per-corpus anchors.

**The two objectives select very different cells:**

| corpus | objective | selected | false-cert | power |
|---|---|---|---|---|
| hh | **R1** (registered: min fc s.t. power ≥ 0.80) | p50 / 3-of-3 | **0.0141** | 0.804 |
| hh | **R2** (proposed A7: max power s.t. fc ≤ 0.10) | p60 / 2-of-3 | **0.0935** | 0.976 |
| pku | R1 | p75 / 2-of-3 | 0.0317 | 0.813 |
| pku | R2 | p55 / 2-of-3 | 0.0965 | 0.929 |

**This weakens A7, which is my own proposal, and I would not now adopt it as
drafted.** R2 lands at false-certification 0.0935 in hh — 6.6× R1's rate and
consuming 93% of the allowance. My argument was that power binds because a
failed gate produces a substantive insensitivity finding rather than a null. But
a *false certification* is equally substantive in the other direction: it
certifies an instrument that does not work, and the convention verdict then
rests on it. Both errors produce findings; neither is the harmless one. A7's
premise — that false certification is the harmless quantity here — is wrong.

**So the objective is a genuine open choice and not mine to settle.** R1 is
conservative against certifying a dead instrument; R2 is conservative against
condemning a live one. The grid is committed and neutral; lacan and RH choose
the objective, and the gate follows from it.

### A4e. The full frontier, and why the R1/R2 binary is misleading

Desktop required the whole frontier disclosed rather than the two pure-objective
winners, so a hidden middle cannot make the choice look forced. lacan required a
**margin clause**: the decoy pool is a finite sample (n=1,054) and the
false-certification estimate inherits that sampling error, which more Monte Carlo
draws do not reduce. Bootstrapped over the pool, 150 resamples × 1M draws (hh):

| cell | FC | 95% upper | power | margin |
|---|---|---|---|---|
| p60 / 2-of-3 | 0.0938 | **0.1090** | 0.976 | **FAILS — exceeds 0.10** |
| p65 / 2-of-3 | 0.0714 | 0.0885 | 0.966 | ok |
| p70 / 2-of-3 | 0.0545 | 0.0651 | 0.954 | ok |
| p75 / 2-of-3 | 0.0377 | 0.0456 | 0.939 | ok |
| **p80 / 2-of-3** | **0.0240** | 0.0309 | **0.900** | ok |
| p50 / 3-of-3 (R1) | 0.0143 | 0.0181 | 0.804 | ok |

**A7's own selection fails A7's own constraint.** p60/2-of-3 was chosen on a
point estimate of 0.0935; its upper bound is 0.1090. lacan's margin clause was
raised as a contingency and it fires immediately. **Under A7 + margin the
selection is p65/2-of-3** (FC 0.0714, power 0.966), not p60.

**There is no knee. The frontier is smoothly convex.** Three seats located a
knee at three different places before anyone tested whether one exists. A knee
is a *single large proportional drop* in the marginal rate, so the diagnostic is
the ratio of successive marginal rates along the rung:

| step ratio | value |
|---|---|
| p95→p90 / p90→p85 | 3.47 |
| p90→p85 / p85→p80 | 3.34 |
| p85→p80 / p80→p75 | 1.71 |
| p80→p75 / p75→p70 | 3.14 |
| p75→p70 / p70→p65 | 1.26 |
| p70→p65 / p65→p60 | 1.59 |
| p65→p60 / p60→p55 | 1.63 |
| p60→p55 / p55→p50 | 2.20 |

Max 3.47, min 1.26, spread 2.7× — **no outlier anywhere**. Smooth convexity, not
a knee. I named p80, lacan corrected to p85, and the property neither of us
tested for is absent from the curve. The knee language is **removed rather than
relocated a third time**.

**What survives, and it is the useful fact.** p85/2-of-3's distinction has
nothing to do with its position on the curve — it is its relation to the
registered rule. FC **0.01443** against R1's **0.01414**, a difference of
**0.00029** far inside the bootstrap uncertainty on either cell, for **0.051 of
power** that is not.

**And the live band is narrow, which is what a decision-maker should take from
this.** Once p85 is available:

| position | cell | FC | power |
|---|---|---|---|
| A-protective (guard against certifying a dead instrument) | p85 / 2-of-3 | 0.0144 | 0.855 |
| B-protective (guard against condemning a live one) | p80 / 2-of-3 | 0.0237 | 0.900 |

Roughly **0.855–0.900 power at 0.0144–0.0237 false certification**. The
conceptual asymmetry between the two errors is real and worth understanding; the
decision it governs is **small**. A reader arriving expecting a consequential
fork should be told the frontier has largely dissolved it.

**Granularity limitation, stated so the winner is not presented as tuned**
(lacan). With three constructs the usable k values are 2 and 3, and
P(≥2 of 3) and P(3 of 3) are far apart with nothing between. Seven independent
constructs would give fine control over the tradeoff; three give two rungs. So
whichever cell is selected clears both criteria without being near either
optimum. That is an argument for wider construct enumeration on **granularity**
grounds — weaker than the validity argument lacan made and withdrew, and offered
as an observation rather than a recommendation.

### A4f. The power lower bounds are pinned, and one of them will not hold still

Desktop caught that `p85/2-of-3`'s power lower bound had been reported to two
seats as **0.788** and **0.782** on different days. Both were resample noise
around one quantity, but an amendment carries **one** number. The asymmetry
behind it: the false-certification bounds had been seeded and committed since
`bbd16d8` because FC was the contested constraint; the power bounds were "just
the floor check" and ran from an ad-hoc interactive session. A tool doing its job
on one axis while the other ran bare, invisible while the two were discussed
together.

`scripts/tier2_power_bounds.py` (`4f48e3f`, **committed before running**) pins
them: 200 replicates, seeds declared in the file, decoy pool resampled — not the
Monte Carlo draws, since the alternative is the pool shifted by the anchor, so a
pool resample propagates into both the threshold and the draws. Seed streams are
distinct from the FC-bound streams so the two bounds are not driven by a shared
resample.

**Then the obvious question, asked because the number had already moved once:
what is the sampling error of the bound itself?** 1,600 replicates in 8 disjoint
blocks of 200, independent seed streams, each block yielding its own 2.5th
percentile:

| cell | mean lower 95% | sd | min | max | vs the 0.80 floor |
|---|---|---|---|---|---|
| p50 / 3-of-3 | **0.7594** | 0.0079 | 0.7433 | 0.7675 | **below in all 8 blocks** |
| p80 / 2-of-3 | 0.8649 | 0.0029 | 0.8607 | 0.8700 | above in all 8 blocks |
| p85 / 2-of-3 | **0.8043** | 0.0085 | 0.7898 | 0.8165 | **straddles — 5 of 8 above** |

**Two consequences, one housekeeping and one not.**

*Housekeeping.* The ruled cell's disclosure should carry **0.759 (sd 0.008 over
1,600 replicates)**, not the 0.751 of the original ad-hoc run nor the 0.764 of
the single seeded one. Neither was wrong; both were single draws quoted to four
figures. The conclusion is unchanged and robust — p50/3-of-3's power floor is met
on the point estimate (0.8039) and **not shown at the 95% bound in any block**.
That gap is the disclosure's content and it does not soften.

*Not housekeeping.* **`p85/2-of-3` sits on the threshold, not near it.** Its
bound is centred at 0.8043 against a 0.80 floor, so whether it is "shown" is
decided by the resample. This matters beyond bookkeeping because reading two —
now standing policy for gates registered after 2026-07-27 — turns eligibility on
exactly this test, and `p85/2-of-3` has the **lowest false certification of any
cell that could ever be eligible** (0.0145, against p80/2-of-3's 0.0240). So
reading two's selection is **indeterminate at this resolution**: p85/2-of-3 when
the bound lands high, p80/2-of-3 when it lands low. The worked example recorded
in the ruling and the ledger as "reading two selects p80/2" is right by
convention, not by computation, and the convention had not been stated.

**Recommended addition to the rider, offered because the policy as adopted has no
stated resolution.** A shown-satisfied precondition should require (i) a declared
replicate count and seed, committed before the run, and (ii) that a bound
straddling the floor at that resolution counts as **not shown**. Limb (ii)
resolves straddle in the protective direction, which is the direction reading two
exists to serve, and it preserves the already-booked answer: under it reading two
selects p80/2-of-3, as recorded. Without it, a coin flip decides eligibility and
the policy inherits the instability it was adopted to remove.

## A5. Gate parameters unchanged — and valid for the first time

**A defect in the v1 gate that nobody caught, including nine audit rounds**
(lacan). The false-certification rates were simulated from the pooled **decoy**
distribution, and the decoys are nearest-k in log frequency **to the chain
pairs**. So the calibrated null describes a marker drawn from chain-frequency
space. Six of seven v1 markers were out of band and mostly far commoner, and a
commoner pair has a much tighter null. **The simulated false-certification rate
was computed for a slate that did not exist.** The v1 gate was invalid twice
over: precondition violated, and calibration not describing the markers it was
applied to.

**Direction, so this is not oversold:** the mismatch made the gate
*conservative*, not permissive — tighter nulls on high-frequency markers clear
an absolute floor by chance less often, so the true rate for that slate was
*below* the simulated one. No false-pass risk, no change to the verdict. The
defect is that a number in the registration was not about the thing it was
attached to.

**So in-band does two jobs, for two separate reasons**, and this should be
stated wherever the requirement appears: it secures **power** at the operative
effect size, *and* it makes the decoy-based **calibration valid**, because the
markers then occupy the same frequency space as the decoys they are calibrated
against. The v2 slate is entirely in band, so p80/5-of-7 is a valid calibration
for the first time.

**Recomputed, not inherited** (lacan's requirement — "should come out close" is
the reasoning that put an unexamined number in the registration last time). 4M
draws from the unchanged decoy pools:

| corpus | gate | false-cert | power at anchor |
|---|---|---|---|
| hh_rlhf | p80 / 5-of-7, floor 0.1057 | **0.000113** | 0.860 |
| pku_saferlhf | p65 / 5-of-7, floor 0.2114 | **0.001185** | 0.824 |

Parameters therefore stand. If the slate ever shrinks under A1, re-run
`scripts/tier2_gate_grid.py` at the new k and declare the new rate before any
`D` is computed.

## A5b. Why the rebuild is worth running, not merely valid

Under the v1 slate a gate failure was **uninterpretable**: most markers were
measured where the null model did not describe them and where power for
chain-sized effects was unknown. Under a slate that is in band and powered,
every null is a *precise null at the operative scale*. So if the new gate fails,
that failure means something — attested annotator preferences genuinely absent
at unigram level in these corpora — rather than meaning nothing. **The rebuild
converts an unfalsifiable check into a falsifiable one.** That, rather than the
seven pairs, is the argument for spending the tokens. (lacan.)

## A6. Declared certification size

Every gate declares the effect size it certifies at, and **that size must be at
or below the corpus's median chain-pair MDE**. A gate certifying only for effects
larger than the test needs certifies nothing the test can use.

## What is burned

`sorry`→`unfortunately` (pre-spec), and all seven v1 markers: `must`→`should`,
`never`→`rarely`, `always`→`often`, `wrong`→`incorrect`, `stupid`→`unclear`,
`no`→`unfortunately`, `obviously`→`perhaps`. None appears in A4.
