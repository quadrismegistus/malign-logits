# Candidate slate v2 — enumeration result, NOT a registered slate

2026-07-26. Counts and MDE only. **No `D` has been computed for any pair below**,
so none is burned and all remain usable as blind controls.

## The question this answers

After the v1 gate was found invalidly constituted (six of seven markers outside
the frequency-comparability band), the live question was whether a compliant
slate can exist at all. lacan put it precisely: the yield rate is moot if the
attested literature is not deep enough to sustain it. A first enumeration of 34
candidates gave 3 powered pairs; seven would need ~78 on that rate.

## Result: the pool is deep enough

Enumeration extended to **62** attested candidate pairs across six documented
categories of RLHF register preference — hedging/epistemic, quantifier
qualification, pejorative→neutral, directive softening, refusal softening,
blame→attribution, emotional de-escalation.

| | count |
|---|---|
| enumerated | 62 |
| below the 20-occurrence floor | 5 |
| **in band** (881–7,926 combined, hh) | **9** (15%) |
| **in band AND powered** (MDE < 0.174) | **7** (78% of in-band) |

**Seven usable pairs, which is exactly the required slate size:**

| pair | hh MDE |
|---|---|
| `require` → `prefer` | 0.092 |
| `force` → `encourage` | 0.104 |
| `angry` → `concerned` | 0.135 |
| `entirely` → `mostly` | 0.136 |
| `totally` → `fairly` | 0.149 |
| `completely` → `largely` | 0.155 |
| `demand` → `request` | 0.160 |

All below the chain-relevant 0.174 (hh median chain-pair MDE, 1.19×). None is
burned. `stupid`→`unclear`, the one compliant marker from v1, is **not** on this
list — it was in band but underpowered at 0.219, and it is burned regardless.

## What this does not settle

- **These are hh MDEs**, and "clears in both corpora" is a phrase to avoid.
  pku's threshold is its own median chain-pair MDE (0.405, 1.50×) — 2.3× laxer
  than hh's 0.174, and laxer *precisely because pku's chains are badly
  measured*. So a pair can clear pku and still be inadequate by the standard
  that matters. `demand`→`request` is 0.206 in pku: clears pku's bar, **fails
  hh's**. Only `force`→`encourage` (0.104 hh, 0.130 pku) is under hh's stricter
  threshold in both corpora. Combined with pku being descriptive-only, a pku
  certification licenses nothing the test can use — so the honest count is
  **one** pair meeting a meaningful bar in both, not two. (lacan's catch.)
- **Attestation quality is not verified here.** Each pair needs its
  external-grounds line, individually, per the standing precondition. Being
  in-band and powered makes a pair *eligible*, not *attested*.
- **Nothing is registered.** A slate becomes registered through spec amendment,
  lacan's audit, and RH's go — in that order, and before any `D` is computed.

## lacan's design principle, which this vindicates

> Find the valid candidates first, then calibrate the gate to how many there are.
> Never fix the slate size and then go looking.

Fixing k=7 in advance felt like pre-registration discipline and was the
mechanism that produced an invalid slate: the count requirement could only be
met by relaxing the quality requirement, and the relaxation was disclosed rather
than blocking the run. Here the order was reversed — enumerate, filter, then
count — and it happened to yield exactly seven. Had it yielded three, the
correct response would have been a 2-of-3 or 3-of-3 gate with a freshly computed
false-certification rate, not a padded slate.
