# M01 MAGNITUDE REGISTRATION — displacement measured by MASS, not by SIGN

**STATUS: DRAFT FOR FREEZE. Nothing below §0's disclosures has been computed.
No magnitude quantity has been read at the unit level, on any arm, ever.**

Commissioned by RH after the rate test returned a null. Drafted by @lacan
against the same frozen corpus, the same frozen site rule `b8fd9a52cd5c794b`,
and Amendment A's declarations `1356aa2ff274b796`.

---

## §0 PRIOR EXPOSURE — MANDATORY, FIRST, AND LONGER THAN LAST TIME

**Four things have been seen and every one is disclosed:**

1. **THE RATE PRIMARY IS A NULL.** p 0.1481, 20 of 33 units positive, median
   Δ +0.0132. Depth secondary 23 of 34, p 0.0288, failing its own component
   sensitivity at 0.0680.
2. **THE DIAGNOSTIC.** Of 33,896 firing cells, 95.2% carry `tail_excess < 0`
   (substitution), 4.8% positive (deflection); mean `tail_share` 0.029.
3. **ONE DECOMPOSED PAIR, AND IT IS DIRECTIONAL.** [3053] posted `r2ds_001` on
   olmo base→instruct: MARKED `departed` 0.14316 / `arrived` 0.18597 against
   UNMARKED 0.08189 / 0.15848. **One pair, one unit, MARKED larger on both.**
   **This is the only magnitude comparison anyone has seen and it points the
   way this registration predicts.**
4. **The pair-level rate texture** ([3076]): 359 positive / 294 negative,
   extremes and median named.
5. **THE BOTH-FIRE COMPOSITION, computed for §6 at the pen's order** ([3121]):
   per-unit counts of both-fire / only-MARKED / only-UNMARKED, the aggregate
   asymmetry, and a distribution-free CI on the unit-level RATE delta.
   **These concern WHICH PAIRS ENTER, never what they show** — no `departed`,
   `arrived`, `selectivity` or `concentration` has been touched at any level.

**WHAT HAS NOT BEEN SEEN: any `departed`, `arrived`, `selectivity` or
`concentration` aggregated to a unit, on either arm, for any unit.** The
per-pair quantities exist in the store and have never been reduced.

---

## §1 THE CLAIM

**At a transgressive site alignment moves MORE MASS, and moves it more
selectively, than at the site's minimal control.**

The rate test asked whether displacement HAPPENS more often. It does not,
detectably. **This asks whether, where it happens, it is LARGER** — which is
the claim the theory actually makes: a slide along a chain of permitted
substitutes is a quantity of mass, not an event count.

---

## §2 UNIT AND CORPUS — INHERITED, NOT RE-ARGUED

**Everything from Amendment A `1356aa2ff274b796`, unchanged:**

    UNIT       the BASE CHECKPOINT (`model_to_base`), n = 34
    EDGE       the base's dpo arm(s), reasoning-trained EXCLUDED,
               median where more than one qualifies
    CORPUS     684 M01 pairs by the three-way conjunction
    EXCLUDED   phi-4 (no qualifying arm); 3 assistant-collision pairs on the
               2 Falcon-Mamba instruct models
    HELD OUT   Falcon-H1-7B-Base if it has not landed at freeze

**Re-arguing a ratified unit because a new statistic is being declared would be
the shopping this campaign exists to prevent.** The unit was settled on the
map's records, before any verdict, and it does not move for a new readout.

---

## §3 THE QUANTITY

For each unit L and each pair p **where BOTH members fire** under the frozen
rule:

    d(L, p)  =  Q(MARKED member)  −  Q(UNMARKED member)

    unit summary   D(L)  =  MEDIAN of d(L, p) over L's qualifying pairs

**Median, not mean, for the reason Amendment A gives**: one anomalous pair must
not carry a unit, and the pair distribution is not assumed symmetric.

---

## §4 THE STATISTIC — AND WHY NOT A SIGN TEST

    PRIMARY   SIGN-FLIP PERMUTATION TEST over the 34 unit summaries D(L),
              ONE-SIDED UPPER, alpha 0.05.
              Null: each unit's D is equally likely to carry either sign.
              100,000 Monte Carlo draws (2^34 is 1.7e10; exhaustive is not
              available). RESOLUTION LIMIT 1e-5, reported with the p-value.

**WHY THIS AND NOT THE SIGN TEST, WITH THE ARITHMETIC:**

    at n = 34, one-sided alpha 0.05, power 80%
      SIGN TEST        needs P(positive) >= 0.725   standardised d = 0.599
      SIGN-FLIP PERM                                standardised d = 0.426
      smallest detectable effect shrinks 29%

**The sign test reduced 663 paired measurements per unit to one bit each and
then could not see what was there.** A sign-flip permutation keeps each unit's
MAGNITUDE while still treating the unit as the exchangeable object — **it fixes
the bluntness without touching the pseudo-replication guard, which is the only
reason the rate test's null is worth anything.**

**It remains distribution-free.** No normality is assumed anywhere; the null is
built by resampling the signs the design itself declares exchangeable.

---

## §5 PRIMARY AND SECONDARY QUANTITIES

    PRIMARY    Q = `departed`   — mass that left the fallers, THE MAGNITUDE OF
                                  THE REPRESSION
    SECONDARY  Q = `concentration` — the top riser's share of `arrived`

**`departed` is the theory's quantity**: how much the aligned model takes away
from what the base was going to say.

**`concentration` is named as secondary because it is SCALE-FREE** — the
docstring's own point: unlike JS it does not shrink when a tokenizer resolves a
language coarsely. **It therefore survives the confound §7 raises against
`departed`, and the pair is a deliberate bracket rather than two bites.**

**SECONDARY MEANS SECONDARY.** It does not rescue a null primary. Reported
whatever the primary does.

**NOT REGISTERED, reported as description only:** `arrived`, `selectivity`,
`captured`, and the four `js_*` parts. **`selectivity` is a ratio that
routinely exceeds 1 and is unbounded below zero-departed; it is not a test
statistic and will not be made one after the fact.**

---

## §6 THE BOTH-FIRE CONDITION, AND WHY THE NULL LICENSES IT

**A paired magnitude needs both members to fire**, so pairs where only one
fires are excluded. **That conditions on an outcome, which §2 of the rate
registration forbids — and here it is safe FOR A REASON WE NOW HAVE:**

**the rate test found NO difference in firing rate (p 0.1481) and the yield
split was 17,301 / 16,595, ratio 1.043.** Since MARKED and UNMARKED fire at
indistinguishable rates, conditioning on both-fire does not select
differentially between the arms.

**THE NULL RESULT IS WHAT MAKES THIS TEST ADMISSIBLE.** Had the rates differed,
this design would be measuring a selected subsample and could not be run as
specified. **Declared here so the dependency is visible rather than inherited.**

### §6.1 A NULL IS NOT "EQUAL RATES", AND THE RESIDUAL IS BOUNDED HERE

**"No detectable difference at MDE 0.725" is not "no difference."** The
admissibility rests on how much differential selection the conditioning can
still carry, so that is stated rather than waved at:

    unit-level rate delta   median +0.0132
    distribution-free 95% CI   [-0.0015, +0.0365]    (34 units, order stats)
    on base rates of ~0.40  ->  at most ~9% relative, and the interval
                                includes zero

**REALIZED CONDITIONING ASYMMETRY, which is the quantity that actually bites:**

    both fire        4,396
    only MARKED      4,717
    only UNMARKED    4,378        onlyM / onlyU  =  1.078

**The both-fire set is enriched, by about 8%, for pairs whose UNMARKED member
happened to fire.** That is the selection this conditioning introduces and it
is small in aggregate.

### §6.2 AND THE AGGREGATE HIDES A QUARTER OF THE UNITS

    per-unit skew (onlyM - onlyU) / both
      median +0.089   p10 -0.122   p90 +0.424
      UNITS WITH |skew| > 0.25:  9 OF 34

**1.078 across the corpus and nine units above a quarter.** A rate is not a
number without its population, and the aggregate here is exactly the reassuring
summary that conceals its own spread.

**REQUIRED, AS A COLUMN AND NOT A CAVEAT: per-unit both-fire / only-MARKED /
only-UNMARKED counts and the skew, for all 34, in the output.** The nine
high-skew units are NAMED. **If the primary's verdict depends on them — check
it by re-running the permutation with those nine dropped, reported beside the
result and never instead of it — the conditioning is doing work the design
cannot separate from the effect, and the registration says so in advance.**

**This is not an exclusion.** Dropping units after seeing a verdict is the
thing this apparatus exists to prevent. It is a declared sensitivity with its
threshold (|skew| > 0.25) and its membership (9 units) fixed BEFORE the
magnitude is read.

---

## §7 THE CONFOUND `departed` IS EXPOSED TO, AND ITS DIAGNOSTIC

**A transgressive prompt may have a more PEAKED base distribution, leaving more
mass available to move.** Then `departed` is larger at MARKED for a reason that
is not displacement.

**DIAGNOSTIC, REQUIRED AS A COLUMN, NOT A CAVEAT** ([1267]'s rule on this unit):
for every qualifying pair, the BASE model's top-word probability and its mover
count, both members, reported as the within-pair difference and its
distribution per unit.

**If the base-mass difference is large and same-signed with D, the primary is
not interpretable as displacement and the registration says so in advance.**
`concentration`, being scale-free, is the readout that survives it — which is
why §5 pairs them.

---

## §8 POWER

    n = 34   one-sided alpha 0.05   power 80%
    MDE      standardised effect d = 0.426
             (mean unit-level D over its SD across units)

**THE SENTENCE THAT MUST SURVIVE: an effect smaller than 0.43 standard
deviations of the between-unit spread is REAL and this design reports it as a
NULL.**

**The raw-scale MDE cannot be stated before the data** — the SD of unit-level
`departed` differences has never been computed. **It is derived and written
into this document BEFORE the read**, as §A.6 was.

---

## §9 WHAT A NEGATIVE REPORTS

**A null**: "no detectable within-pair difference in displaced mass at n=34,
MDE d = 0.426." **Never "alignment does not displace more at transgressive
sites"** — the design bounds what it could see and that is the bound.

**A significantly NEGATIVE D** reports in the ruled wording: **"DIRECTION
OPPOSITE TO THE PREDICTION — NOT A REGISTERED FINDING, exploratory, requiring
its own registration to claim."**

---

## §10 WHAT THIS DOES NOT LICENSE

- **Nothing about the RATE.** That test returned a null and this does not
  revisit it. A magnitude effect with no rate effect means displacement is not
  more FREQUENT at transgressive sites but is LARGER where it occurs — a
  narrower claim than the rate test was making, and it must be stated that way.
- **No between-domain or between-family ordering.** No null is declared for
  either. `NO ORDERING WITHOUT A DECLARED NULL` is in force.
- **No causal claim.** The pair holds context fixed, not the grammar of the
  continuation slot.
- **`selectivity` is not promoted to a test statistic** whatever it shows.
