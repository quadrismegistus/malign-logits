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

Attestation is the external-grounds requirement; band-compliance and power are
separate requirements in the same bullet. On the literal text the v1 slate had
seven *attested* candidates, so the clause never triggered — the registered
remedy for exactly this situation was drafted in terms that did not reach it.

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

**Note on composition:** three of seven are quantifier qualification and three
are directive softening. A slate concentrated in two categories tests those
categories, not "register preference" in general. Whether that is acceptable is
lacan's call; the alternative is a wider enumeration to spread the categories,
at the cost of a longer search.

## A5. Gate parameters are unchanged

The registered selection rule chose the gate from the **decoy** distribution,
which this amendment does not touch. So hh remains **p80 floor / 5-of-7**
(threshold 0.1057) and pku **p65 / 5-of-7** (0.2114). Only the slate changes.

If the slate ever shrinks under A1, the grid in `scripts/tier2_gate_grid.py`
must be re-run at the new k and the resulting false-certification rate declared
before any `D` is computed.

## A6. Declared certification size

Every gate declares the effect size it certifies at, and **that size must be at
or below the corpus's median chain-pair MDE**. A gate certifying only for effects
larger than the test needs certifies nothing the test can use.

## What is burned

`sorry`→`unfortunately` (pre-spec), and all seven v1 markers: `must`→`should`,
`never`→`rarely`, `always`→`often`, `wrong`→`incorrect`, `stupid`→`unclear`,
`no`→`unfortunately`, `obviously`→`perhaps`. None appears in A4.
