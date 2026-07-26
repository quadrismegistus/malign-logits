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

**The declared false-certification rate is understated by one to two orders of
magnitude.** Independent P(≥5 of 7) against blockwise P, by per-candidate null
firing probability p:

| p | independent | blockwise | inflation |
|---|---|---|---|
| 0.05 | 0.00001 | 0.00250 | **415×** |
| 0.10 | 0.00018 | 0.01000 | 57× |
| 0.25 | 0.01288 | 0.06250 | 4.9× |
| 0.40 | 0.09626 | 0.16000 | 1.7× |

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

### A4d. The honestly calibrated three-construct gate

Freshly simulated on the unchanged decoy pool, 4M draws, under the registered
selection rule (minimise false-certification subject to power ≥ 0.80 at the
corpus anchor):

| gate | false-cert | power |
|---|---|---|
| **p50 floor 0.0481, 3-of-3 (SELECTED)** | **0.0141** | 0.804 |
| p80, 2-of-3 | 0.0238 | 0.900 |
| p75, 2-of-3 | 0.0380 | 0.939 |
| p90, 3-of-3 | 0.00013 | 0.290 (fails power) |

**The true rate is 0.014, not 0.000113** — two orders of magnitude worse than
the withdrawn 5-of-7 figure, which is exactly the inflation lacan's table
predicts. It still clears the 0.10 standard comfortably. This is a smaller and
truer instrument than 5-of-7 on a slate that was really three things.

**Cost, stated plainly:** a seven-*construct* gate would need a much wider
enumeration than 62 candidates, and the four empty categories suggest the search
is harder than the pair count implied. Whether to pay that or run a
three-construct gate at 0.014 is RH's call, not mine.

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
