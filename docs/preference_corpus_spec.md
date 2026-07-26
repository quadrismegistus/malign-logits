# Pre-registration: the preference-corpus test

2026-07-26, malign-logits seat. Revision 6. Desktop rules population and
procedure; lacan rules thresholds and has signed off. REGISTERED.

> **SCOPE, STATED UP FRONT BECAUSE THE ARTICLE INHERITS IT.** Whatever this
> test concludes about the convention account, it concludes it about
> **Anthropic-HH-style preference data specifically**. Three independent
> features of the design converge on that single corpus — the `kill`/`scream`
> exclusion in PKU, PKU's MDE table, and an instrument gate that is
> uninformative in PKU at any calibration. The consequence, which is not a
> limitation to be noted later but a boundary on what may be claimed at all, is
> that **the test cannot speak to amber or beaver** — the PKU-trained families,
> and amber is the most distinctive family in the census. See "Registered scope
> limit" below.

## Why this is now the decisive instrument

Two accounts of the reroute chains have been excluded, and one rival is
explicitly still alive. Stating this precisely, because revision 1 of this
document overstated it:

- **Excluded: the pretraining-contiguity account.** Set-D found the corpus
  prefers the SOURCE in ~59% of decisive cells (Dolma 41.5% target, RPJ 40.8%),
  so alignment moves away from pretraining continuation frequency, not toward it.
- **Excluded: a structural semantic relation.** The role-relation test returned
  4/64 real against 0/64 control, margin +6.2pp against a registered gate of
  >=20pp.
- **NOT excluded: renormalization.** Revision 1 of this document said the
  mechanical reading died alongside the role test. That was wrong. lacan's
  renormalization rival stands at 2.09x excess over proportional drain with
  17/120 gainer inversions, and both the excess sweep and the inversion sweep
  are still owed. Nothing below adjudicates it, and a positive result here does
  not retire it.

That leaves SHARED PREFERENCE-DATA CONVENTION as the untested default account.
Every family in the census is aligned on overlapping public preference corpora,
so convention explains cross-family convergence at least as economically as
anything structural. This test is decisive against that account specifically,
and against nothing else.

## Two axes, four cells, and only some of them are affordable

**Population axis** (ruled by desktop, scoped below by item 4):
- PRIMARY: families with disclosed preference data AND an available
  chosen/rejected table.
- SECONDARY: families with disclosed preference data but no built table.
- OUT OF SCOPE: proxy mixtures. Licenses nothing on their own.

**Instrument axis:**
- TIER 1, conditional frequency: does the post-training corpus already prefer
  the target after the frame? A necessary precondition, not the hypothesis.
- TIER 2, chosen-vs-rejected asymmetry: did annotators systematically prefer
  responses containing the target over responses containing the source? This is
  the convention hypothesis and it CARRIES THE VERDICT.

## The power problem, stated before the design

Set-D worked because Dolma is effectively unlimited. Preference corpora are not.
Message counts from `f37_corpus_tokens_v2.csv`, at ~200 tokens/message against
Dolma v1.7 at ~2.3e12 tokens:

| corpus | messages | ~tokens | smaller than Dolma |
|---|---|---|---|
| ultrachat | 5,649,662 | 1.13e9 | 2,036x |
| hh_rlhf | 377,833 | 7.6e7 | 30,437x |
| stackexchange | 100,000 | 2.0e7 | 115,000x |
| pku_saferlhf | 73,907 | 1.5e7 | 155,601x |
| oasst | 52,912 | 1.1e7 | 217,342x |
| alpaca | 51,974 | 1.0e7 | 221,264x |

A context sitting at Set-D's backoff floor of 20 Dolma occurrences expects
**0.0098** occurrences in UltraChat and **0.0001** in PKU. Even a common
four-word frame ("she wanted to scream", 6,691 Dolma hits) expects 3.3 in
UltraChat and 0.04 in PKU.

**Therefore tier 1 cannot be a Set-D analogue.** Built that way it would report
"the test did not run" by construction.

## Tier 1 as it will actually be run

- **UltraChat only** for conditional frequency. The other five corpora are
  declared OUT OF SCOPE for this measure in advance, not after seeing counts.
- **Short frames only**, 2-3 words.
- **THE F37 ANCHOR IS EXCLUDED FROM THE TEST SET.** (lacan audit item a.)
  "wanted to" is the frame on which the reroute observation was originally made;
  it is the hypothesis-generating datum, and including it in the confirmation
  set pre-confirms tier 1 the way Set-D's flagship pair would have. The
  registered tier-1 rate is computed on frames EXCLUDING "wanted to". The
  with-anchor rate is reported beside it as descriptive continuity with F37,
  and is registered NOW as unable to pass the precondition on its own: if the
  excluded-anchor rate fails and the with-anchor rate passes, the precondition
  is recorded as FAILED.
- **Minimum decisive-cell count: 40**, computed on the excluded-anchor set.
  Below this the registered conclusion is THE PRECONDITION IS UNTESTABLE,
  explicitly not "unmet".
- **Positive control, must fire first, named here**: `going`/`gonna` after the
  frame "I was" — a register pair whose conditional distribution in a chat
  corpus is known to differ sharply from a web corpus, chosen because it is
  unrelated to the reroute chains and to transgression. If the control does not
  fire, no negative from this tier is interpretable.
- **Report alongside the rate**: number of decisive cells, and the distribution
  of conditional counts. A null must be visibly distinguishable from sparsity.
- **Aggregate lexical rates across all six corpora** are descriptive context
  only and explicitly CANNOT pass or fail the precondition.

## Tier 2 as it will actually be run — this carries the verdict

### The statistic, registered (desktop item 2)

For a source word `s` and target word `t`, with chosen and rejected token counts
`c`, `r` and corpus totals `N_c`, `N_r`:

```
L(w) = log[ (c_w / (N_c - c_w)) / (r_w / (N_r - r_w)) ]     # chosen-vs-rejected log OR
D    = L(t) - L(s)                                          # target minus source
```

`D > 0` means annotators preferred the target relative to the source. Primary
test: **one-sided sign test on the count of pairs with `D_excess > 0`**, across
the content-word modal chains in `data/d2_modal_pairs.csv` (content-word
restriction per the role-test ruling). Unit of analysis is the WORD PAIR;
corpus is a stratum; corpora are never pooled into one sign test.

**The sign test runs on informative pairs only.** A pair whose
MDE(`D_excess`) exceeds 2.0x cannot detect the effect size at issue, so its
sign is close to a coin flip. Including such pairs pulls the observed
proportion toward 0.5, and that dilution is ASYMMETRIC in its consequences:
conservative for the *passes* branch, but anti-conservative for the *fails*
branch, because a diluted proportion makes "convention excluded — the chains
have no visible source anywhere" easier to reach, and that is the strongest
claim in the outcome table. Registered accordingly:

- **Primary sign test: pairs with MDE(`D_excess`) <= 2.0x only.** On current
  counts that is 53/53 (hh) and 34/38 (pku), both above the 12-pair threshold.
- The all-pairs version is reported as a sensitivity check.
- **The exclusion verdict cannot be reached on the diluted (all-pairs) version
  alone.** If the two versions disagree, the informative-pairs result governs
  and the disagreement is reported.

### Population scope (desktop item 4)

Chosen/rejected unigram tables exist for exactly two corpora, so the PRIMARY
population is exactly the families those corpora align:

| corpus | table | families licensed |
|---|---|---|
| hh_rlhf | `f37_corpus_unigrams_hh_rlhf_{chosen,rejected}_v2.csv` | pythia, archangel |
| pku_saferlhf | `f37_corpus_unigrams_pku_saferlhf_{chosen,rejected}_v2.csv` | amber, beaver |

zephyr (UltraFeedback), tulu and the olmo lines (Tulu mixtures) are SECONDARY:
they enter only if an UltraFeedback chosen/rejected table is built first, and
if it is not built they are reported as out of scope, not as null.

### Minimum detectable effect — computed for `D_excess`, the estimand actually tested

Revision 2 declared MDEs for `D`. The sign test runs on `D_excess`, which also
carries the sampling error of the decoy median, so those figures were computed
for a quantity the test does not use (lacan audit, rev 2 item 1). Corrected:

```
SE(D_excess) = sqrt( SE(D_chain)^2 + SE(median D_decoy)^2 )
SE(median D_decoy) ~ 1.253 * sd(D_decoy) / sqrt(n_decoy)
MDE = exp( 2.49 * SE(D_excess) )      # 80% power, one-sided alpha .05
```

Computed on the real unigram tables over the 59 content-word chain pairs, under
the decoy design registered below:

| corpus | eligible pairs | median MDE(`D_excess`) | max | inflation over MDE(`D`) | pairs <= 2.0x |
|---|---|---|---|---|---|
| hh_rlhf | 53 | 1.19x | 1.81x | median 1.007, max 1.027 | **53/53** |
| pku_saferlhf | 38 | 1.50x | 2.13x | median 1.037, max 1.068 | **34/38** |

**Logged for the record, in the terms the objecting seat asked for:** the
objection was correct IN FORM — the document declared power for a quantity the
test does not use, inside the document that carries the no-null-without-power
rule — and MINOR IN CONSEQUENCE, the inflation running 1.007 to 1.068. It is
logged as a correct objection with a small consequence, not as a save. Selected
pairs:

| corpus | pair | counts (chosen/rejected) | MDE(`D_excess`) |
|---|---|---|---|
| hh_rlhf | kill -> scream | 2700/3076, 147/153 | 1.34x |
| hh_rlhf | die -> fall | 1365/1452, 2033/1944 | 1.13x |
| hh_rlhf | cry -> feel | 256/261, 17045/16367 | 1.25x |
| pku_saferlhf | kill -> scream | 737/695, **7/6** | EXCLUDED (< 20) |

Corpus totals: hh chosen 21,135,050 / rejected 20,692,907 tokens; pku chosen
6,365,310 / rejected 6,297,142.

**Registered consequences, in order of operations:**

1. **Exclusion first.** A pair is dropped if any of its four counts is under 20.
   This removes PKU `kill -> scream` (7 and 6 occurrences) outright, so no MDE
   rule ever governs a null there. Excluded pairs do NOT count against the
   12-pair eligibility threshold; the threshold counts pairs that survive.
2. **Decoys, then `D_excess`, then the MDE rule.** The 2.0x underpowered rule
   applies only to surviving pairs.
3. **Recorded now, before the test:** on current counts the eligible pool is 53
   (hh) and 38 (pku), both far above 12, and 53/53 and 34/38 sit under 2.0x. The
   UNDERPOWERED branch is therefore not expected to trigger. Registering this
   means a later NO VERDICT would be a real change in the data, not a design
   escape hatch.

### Two controls, both required (lacan audit item b)

Tier 1 has a positive control and a stated untestable floor. Revision 1 gave
tier 2 neither, while tier 2 carries the verdict. Both are added:

1. **Positive control (does the instrument detect known annotator preference?)**
   Revision 2 named `sorry` -> `unfortunately` as the control. **That candidate
   was computed at spec-writing time and is now disqualified on two counts,
   both disclosed rather than quietly replaced.** It runs the WRONG WAY in both
   corpora (`D` = -0.072 in hh, -0.066 in pku), so it would have failed the
   instrument and returned NO VERDICT for a reason that has nothing to do with
   the chains. And it is not frequency-comparable: `sorry` at 24,535 combined
   occurrences in hh against a median chain word of 2,642, giving the control a
   *better* MDE (1.10x) than the median chain pair (1.19x) — exactly lacan's
   objection that a control can certify on abundance a design that could not
   have detected the chains.

   Having computed it, I have also burned it as a blind control. The registered
   replacement is a SET, NOT computed before registration:

   - **SEVEN candidate pairs, named on external grounds before testing, all
     seven tested, all seven results disclosed in the registration** (desktop
     ruling, count raised from three for the power gain below). No silent
     shopping: every burn is documented the way the first one was. Candidates
     are drawn from documented RLHF annotator preferences (hedging, apology,
     explicit-refusal markers against blunt counterparts), each
     frequency-comparable to the chain pairs: both members within a factor of 3
     of the median chain-word frequency in that corpus.
   - **CONDITION ON THE COUNT, required independently by both other seats.**
     The power gain from seven is contingent on all seven being independently
     attested; seven markers where the last few were chosen to fill a slate is
     a different object from seven attested ones, and if the marginal
     candidates are weaker the null distribution the calibration assumes no
     longer describes them. Therefore **the external grounds for each of the
     seven are stated individually in the registration, one sentence each**, so
     a reader can see whether the count was reached on evidence or on
     arithmetic. The count rises; the procedure does not loosen. If fewer than
     seven can be attested individually, the slate shrinks to the number that
     can and the gate reverts to the corresponding calibrated rule (p75 floor,
     2-of-3 at three candidates).
   - **Firing rule: `D > 0` AND `|D|` above the 75th percentile of the decoy
     `|D|` distribution, with 2 of the 7 candidates required.** Floors:
     `|D|` p90 = **0.1415** (hh), **0.4106** (pku). SUPERSEDED BY BLIND RE-DERIVATION -- see "the stronger check was run"; p75/3-of-7 is proposed as a deviation, pending lacan audit.

   **Calibration of the gate, by simulation from the real decoy distributions**
   (`scripts/tier2_gate_calibration.py`, 200k draws). Revision 4 used the
   *median* `|D_decoy|` as the floor with 1-of-3 required. That was wrong in the
   certifying direction: a median is by construction the value half of null
   pairs exceed, so a dead instrument passed 56% of the time. Confirmed
   empirically — the pooled decoy `D` is near-symmetric (`P(D>0)` = 0.485 hh,
   0.463 pku), so the analytic argument holds almost exactly.

   | gate | false-cert (hh) | false-cert (pku) | power hh, d=.10 | d=.20 |
   |---|---|---|---|---|
   | median floor, 1-of-3 (rev 4) | **0.564** | 0.518 | 0.985 | 1.000 |
   | median floor, 2-of-3 | 0.148 | 0.121 | 0.847 | 0.994 |
   | p90 floor, 1-of-3 | 0.140 | 0.141 | 0.603 | 0.988 |
   | p75 floor, 2-of-3 | 0.037 | 0.031 | 0.564 | 0.975 |
   | p95 floor, 1-of-3 | 0.074 | 0.073 | 0.386 | 0.943 |
   | **p75 floor, 3-of-7 (REGISTERED)** | **0.038** | 0.030 | **0.834** | 1.000 |

   The registered 3-of-7 buys 27 points of power at `d`=0.10 over 2-of-3 at an
   indistinguishable false-certification rate (0.038 against 0.037), which is
   why the count was raised. It is conditional on the seven-marker attestation
   requirement above: the calibration assumes all seven draw from the same null
   distribution, and marginal slate-filling candidates would break that.

   The registered gate is chosen on BOTH axes, not just the null rate: a gate
   stringent enough to block false certification can also fail a working
   instrument and trigger the insensitivity outcome spuriously. `p75, 2-of-3`
   dominates `p95, 1-of-3` on both (lower false-cert, higher power at every
   effect size tested), which is why it is preferred to the alternatives at
   equal null rate.

   **Registered limitation: the gate is informative in hh only.** In pku no gate
   clearing 0.10 reaches power above 0.24 even at `d`=0.20, because pku's decoy
   dispersion is ~3x hh's (p75 `|D|` 0.266 against 0.090). The pku control
   result is therefore DESCRIPTIVE, and the instrument check that gates tier 2
   is evaluated in hh_rlhf. This is the third independent respect in which the
   verdict rests on hh_rlhf, alongside the `kill`/`scream` exclusion and the
   MDE table.

   **Available amendment, flagged before registration:** the binding constraint
   on gate power is the candidate COUNT, not the floor. At the same p75 floor,
   3-of-7 gives false-cert 0.038 — indistinguishable from 2-of-3's 0.037 — with
   power 0.834 at `d`=0.10 against 0.564, a 27-point gain for four more markers
   named on the same external grounds. Desktop rules the candidate count; if it
   is raised to seven before registration, the registered gate becomes
   `p75 floor, 3-of-7`. Registration does not wait on this.
   - **A-PRIORI STATUS DISCLOSED AS WEAKER.** These markers are being chosen
     *after* a control failure, with one look at the data already spent. The
     first control was picked blind; this set is not, and readers of the result
     should discount it accordingly. Stating this is not a formality — it is
     the difference between a control and a control that was allowed a retry.

   **If zero of the three fire, that is an OUTCOME, not a nullity.** Hedging-over-blunt is attested in the RLHF literature and did not
   show. A chosen-vs-rejected unigram rate is a very indirect proxy for what an
   annotator selected on: whole responses are chosen for many reasons at once,
   and any single word's contribution is diluted by everything else in the
   text. So a second failure supports a specific conclusion — **the
   chosen/rejected unigram instrument is not sensitive to lexical annotator
   preference at all** — in which case tier 2 cannot carry the verdict for ANY
   marker set, and the convention hypothesis needs a response-level instrument
   (per-response presence/absence with pair as the unit) rather than a better
   word list. That is a finding about method and is reported as one, with the
   control numbers, not buried in a NO VERDICT.

2. **Frequency control (is any pass a fluency artifact?)** This is the item
   that matters most. If annotators simply prefer the more frequent, more
   fluent word, then every chain whose target is commoner than its source will
   show `D > 0` for reasons that have nothing to do with convention — and since
   reroute targets tend to be commoner than their sources, tier 2 would pass
   *because* tier 1 passed, and the two tiers would not be independent evidence.

   The registered test is therefore **excess over a frequency-matched non-chain
   baseline**, the same discipline applied to renormalization and to the
   derangement control:

   - For each chain pair `(s, t)`, draw 20 decoy pairs `(s', t')` from words not
     in any reroute chain and not stopwords, with at least 20 occurrences in
     each arm.
   - **Matching is nearest-k in LOG frequency, k=20, not a fixed +/-20% band.**
     Revision 2's +/-20% joint band was counted before registration (lacan audit
     item 2) and is infeasible: it leaves 14 of 53 hh pairs and 6 of 38 pku
     pairs with fewer than 20 decoys, and the failures are systematic rather
     than random — Zipf makes count-space sparse at the high-frequency tail, so
     `like` (179,267 occurrences) has exactly 2 candidates inside the band.
     Nearest-k yields exactly 20 decoys for every eligible pair by construction,
     and the achieved match becomes a REPORTED DIAGNOSTIC instead of a silent
     gate: median worst-case log-frequency mismatch 0.014 (hh) and 0.021 (pku),
     against 0.182 = the +/-20% band it replaces. The median match is tighter
     than the band; the worst case is 1.355 in hh, on one extreme-frequency
     member.
   - **Handling of poorly-matched pairs, not merely flagging them.** A pair
     whose worst-case log-frequency mismatch exceeds 0.7 is KEPT in the primary
     sign test — dropping it would reintroduce the systematic loss of
     high-frequency chain members that sank the +/-20% band — and is reported
     as a pre-declared SENSITIVITY SPLIT: the sign test is reported with and
     without the flagged pairs, and if the two disagree, neither confirmation
     nor exclusion is booked and the disagreement is the result.
   - Compute `D` for every decoy pair; the baseline is the decoy median.
   - The registered quantity is `D_excess = D_chain - median(D_decoy)`, and the
     **sign test is run on `D_excess`, not on `D`.**
   - The raw `D` sign test is reported beside it as descriptive, and registered
     now as unable to confirm on its own.

### Anchor discipline

F37 found `scream` follows "wanted to" ~3x as often as `kill` in UltraChat.
That is a TIER 1 observation, it is the hypothesis-generating datum, and it is
excluded from the tier-1 test set per the rule above. It must never be reported
as tier 2 evidence: they are different hypotheses.

## Registered scope limit: the test cannot speak to amber

This follows from the power findings and is registered HERE, in the outcome
interpretation, rather than surfacing as a limitations paragraph after results
exist.

The verdict rests on hh_rlhf in three independent respects — `kill`/`scream`
excluded from pku by the count rule, pku's MDE table (median 1.50x, 4 of 38
pairs above the 2.0x cut), and the instrument gate being uninformative in pku at
any calibration. But the roster maps corpora to families, and **hh_rlhf licenses
pythia and archangel while PKU-SafeRLHF licenses amber and beaver.** A test that
is descriptive-only in pku is therefore a test that says nothing about amber.

That is the worst possible family to lose. Amber is the most distinctive family
in the census: weight displacement 0.050 against 0.0004–0.004 for ordinary
preference optimization, the only family showing narrator-mode containment, the
hardest suppression in F36 at −4.19, and the family whose safety-data-style
gradient anchors the political-economy claim. **The convention hypothesis would
go untested precisely where convention should matter most.**

Registered consequences:

- A positive hh result licenses **"convention installed the chains in HH-trained
  lineages"** and nothing whatever about amber or beaver. It may not be written
  up as a census-wide result, and the claims ledger entry must carry the
  lineage restriction in the claim text, not in a footnote.
- A negative hh result is likewise restricted, and specifically **cannot support
  the "convention excluded, chains have no visible source anywhere" cell** for
  the pku-trained families.
- **A BOUNDED PARTIAL REMEDY WITH A COMPUTED CEILING — not a route, and not a
  dead end.** Revision 6 called the response-level instrument a "route" to
  testing amber. lacan objected that a response-level design removes
  response-length variability from the variance but *cannot create
  occurrences*, so the wording turns on whether pku's dispersion is
  length-driven (fixable) or scarcity-driven (not). That is computable from the
  existing tables, and it was computed rather than argued
  (`scripts/tier2_dispersion_decomp.py`):

  | | hh_rlhf | pku_saferlhf |
  |---|---|---|
  | observed sd(`D`) across decoys | 0.0884 | 0.2462 |
  | predicted from counts alone, `sqrt(1/a+1/b+1/c+1/d)` | 0.0990 | 0.1704 |
  | raw overdispersion factor | 0.89x | 1.45x |
  | **calibrated factor** (hh forced to 1.00) | 1.00x | **1.62x** |

  **The counting baseline is validated at 0.89, not 1.00.** hh comes in 11%
  UNDER its predicted SE — plausibly because nearest-k decoy matching induces
  negative correlation across pairs — so the counting model overstates expected
  SE in the one corpus where it can be checked. Recalibrating against hh raises
  pku's overdispersion factor from 1.45x to **1.62x**.

  pku's dispersion exceeds hh's by 2.78x, and the split is **roughly half each**
  — but it is reported that way rather than as two precise figures, for a reason
  worth stating exactly. A *uniform* baseline bias does NOT move the split: the
  share is `log(predicted ratio)/log(observed ratio)`, and a constant multiplies
  numerator and denominator of the predicted ratio alike and cancels (53.0%
  before and after recalibration, verified). What does move it is
  *non-uniformity* — a bias that differs between corpora, which cannot be ruled
  out because it can only be measured where it can be checked:

  | assumed pku baseline bias | scarcity | overdispersion |
  |---|---|---|
  | 0.80 | 43% | 57% |
  | 0.89 (= hh, uniform) | 53% | 47% |
  | 1.00 | 64% | 36% |

  **These bounds are illustrative, not justified.** 0.89 is observed in hh and
  nothing is measured in pku, so nothing excludes a bias above 1.00, which would
  push scarcity above 64%. Widening the range only widens the interval, which
  strengthens the wording below rather than threatening it; the table is shown
  to convey that the split is unresolvable, not to bound it.

  So: **roughly half each, and the method cannot resolve which is larger.**
  hh itself shows no overdispersion, which is why its token-level instrument
  works.

  The consequence is a ceiling, computed by rescaling pku's decoy distribution
  to remove **all** overdispersion — an upper bound assuming a perfect
  response-level design:

  | pku gate (p75, 3-of-7) | false-cert | power d=.10 | power d=.20 |
  |---|---|---|---|
  | as-is, token-level | 0.031 | 0.128 | 0.390 |
  | ceiling, all overdispersion removed | 0.030 | 0.203 | **0.792** |
  | *(hh, for reference)* | 0.038 | 0.834 | 1.000 |

  **What `d` means, since "large effects only" is uninterpretable without it.**
  `D` is a difference of log odds ratios, so effect sizes convert directly to
  ratio-of-ratios:

  | `d` | ratio-of-ratios | reference |
  |---|---|---|
  | 0.05 | 1.05x | below every corpus MDE |
  | 0.10 | 1.11x | the foreclosed case |
  | **0.20** | **1.22x** | the powered case; cf. hh median MDE 1.19x |
  | 0.30 | 1.35x | comfortably detectable in hh |

  So the powered case in pku is an annotator preference of about **1.22x**, and
  the permanently foreclosed case is about **1.11x**. A reader can then judge
  the substantive question directly: is a shared annotator preference strong
  enough to install cross-family reroute chains more likely to be a 1.2x
  asymmetry or a 1.1x one? **The registration takes no position, and there is no
  prior to appeal to** — F37's ~3x anchor is a tier-1 conditional frequency, not
  a tier-2 annotator preference, and the two are not commensurable. That the
  field has no established effect size for lexical annotator preference is
  itself part of why this test was worth registering.

  **Registered accordingly:** a response-level instrument could test amber for a
  LARGE convention effect (power 0.79 at 1.22x, up from 0.39) but would remain
  underpowered for a moderate one (0.20 at 1.11x, against hh's 0.83), and that
  is an upper bound no implementation can beat. So the amber gap is **partially
  remediable at large effects only**, the residual is irreducible scarcity, and
  any future amber claim must state which effect size it is powered for.
  Registered as owed before any convention result is booked against amber.

Revision 1's table let "convention excluded" be reached from an underpowered
tier 2, which violates rule 1 inside the document that carries rule 1. Tier 2
now has three states, not two, and no verdict is reachable from the third.

| tier 1 | tier 2 (on `D_excess`) | conclusion |
|---|---|---|
| passes | passes, control fired | convention installed the chains; default account confirmed |
| passes | fails, control fired | **available but not selected-for**; strangeness candidate re-inflates |
| fails | passes, control fired | target selected despite corpus rarity; a stronger convention result |
| fails | fails, control fired | convention excluded; chains have no visible source anywhere |
| any | **UNDERPOWERED** — fewer than 12 pairs surviving the count exclusion, or MDE(`D_excess`) above 2.0x on a majority of survivors | **NO VERDICT.** Convention is neither confirmed nor excluded. The entry records the power shortfall and names what corpus would fix it. Registered now: on current counts this branch does not trigger (53 and 38 survivors; 53/53 and 34/38 under 2.0x), so reaching it would be a change in the data, not a design escape. |
| any | **count exclusions alone reduce survivors below 12** | **NO VERDICT**, reported explicitly as a vocabulary limit rather than an evidential one. |
| any | positive control set did not fire (majority `D > 0` not reached) | **NO VERDICT.** Instrument failure, reported as such. |
| untestable | any of the above | tier 2 stands alone; conditional-frequency gap named in the entry |

## On the asymmetry of this document's own foreclosures

This registration states five things it cannot do: it cannot speak to amber or
beaver, cannot resolve the scarcity/overdispersion split, cannot detect below
about 1.11x in pku, cannot import F37's ~3x as a prior, and cannot reach the
exclusion verdict from an underpowered tier or a diluted sign test. A reader
encountering five such clauses is entitled to ask whether the test was built to
be unfalsifiable. The question is fair and the answer should be checkable rather
than rhetorical.

**The asymmetry is real.** Every foreclosure above makes a NEGATIVE result
easier to explain away and leaves a POSITIVE result untouched. Each is a
pre-authorized reason a null might not mean what it appears to mean. The
document is therefore, by construction, more likely to conclude "we could not
tell" than "convention is excluded" — and exclusion is the strongest claim in
its outcome table. That property is not defended here; it is disclosed.

**What distinguishes a constraint from an escape is when it was computed and
whether it carries a number.** A foreclosure derived from a power calculation
before any result exists is a constraint. A foreclosure asserted after a
disappointing null is an escape. Every clause above is of the first kind:

| foreclosure | number that fixes it | prior looks spent when registered |
|---|---|---|
| cannot import F37 as a prior | tier-1 conditional freq vs tier-2 preference; not commensurable | none — a commensurability argument, no data |
| cannot detect below ~1.11x in pku | median MDE(`D_excess`) 1.50x, 4/38 pairs above 2.0x | unigram counts only |
| cannot speak to amber/beaver | pku gate power 0.20 at 1.11x, ceiling 0.79 at 1.22x | counts, decoys, **chain `D`**, control burn |
| cannot resolve the dispersion split | 43–64% scarcity across unmeasurable bias | counts, decoys, **chain `D`**, control burn |
| cannot reach exclusion from a diluted test | false-cert 0.564 for the rev-4 gate; informative-pairs rule | counts, decoys, **chain `D`**, control burn |

**"Predates any result" would be doing more work than it can bear, so it is not
claimed.** Three looks were spent before the later commits, and flattening them
would give this section a cleanliness the rest of the document explicitly
disclaims:

1. **F37's ~3x**, the hypothesis-generating observation — which is why "wanted
   to" is excluded from the tier-1 test set.
2. **The burned positive control**, `sorry`->`unfortunately` at `D` = −0.072,
   which is why the replacement markers are registered as chosen with one look
   already spent.
3. **Raw chain-pair `D` values were printed during the decoy-feasibility run**
   and are therefore visible to the author. This was not previously disclosed
   and is disclosed here. In hh, `kill`->`scream` `D` = +0.090,
   `die`->`fall` +0.107, `cry`->`feel` +0.060 — several positive, which is
   directional information about the hypothesis under test.

**What this does and does not compromise.** The registered statistic is
`D_excess` against a decoy baseline, evaluated by sign test; **neither
`D_excess` nor any sign test has been computed for the chain pairs**, and the
positive-control slate has not been run. But the thresholds fixed after that
run — the 2.0x cut, the informative-pairs rule, the p75/3-of-7 gate — were set
with raw chain `D` visible. What protects them is that every one has a stated
power rationale that a reader can check is independent of the observed `D`: the
2.0x cut from MDE, the informative-pairs rule from sign-test dilution
arithmetic, the gate from a false-certification simulation on decoys. A reader
who suspects otherwise should check whether any threshold's justification
references an observed chain value. None does — but the check is theirs to make,
not mine to assert.

**A numerical coincidence, addressed because a reviewer will see it first.**
The registered gate floor in hh is `|D|` p75 = 0.0903. The flagship chain pair
`kill`->`scream` has observed `D` = +0.090 in hh. These are the same number to
three decimals, and the floor was chosen after the run in which the chain values
were visible.

It is a coincidence, and the reason the two are not commensurable is worth
stating: **the floor governs whether the CONTROL MARKERS fire**, and control
markers are a separate word set; **chain pairs are never evaluated against the
floor at all**, but by sign test on `D_excess` against their own decoy
baselines. No mechanism connects a pooled decoy percentile to one chain pair's
value, and nothing in the gate's derivation references a chain pair. The
calibration in `scripts/tier2_gate_calibration.py` takes only the decoy
distribution as input.

But the coincidence is not information-free, and pretending otherwise would
repeat the error this section exists to correct. **The flagship pair's raw
asymmetry is inside the bulk of the null**, and the exact figure is stated in the
direction less comfortable for the hypothesis rather than more.

An earlier draft of this paragraph called it "ordinary, at the 75th percentile,"
which compared a SIGNED value to an ABSOLUTE-VALUE percentile and understated it
by roughly a factor of two. The flagship `D` = +0.090 is signed and in the
predicted direction, so the matching null quantity is one-sided:
`P(D_decoy > +0.0903)` = **0.117**, not `P(|D_decoy| > 0.0903)` = 0.250. About
**one null pair in 8.6** exceeds the flagship in the predicted direction — a
one-sided null p of about 0.12 on a single pair. Still not significant, still
inside the bulk, but twice as notable as first written. It is a further prior
look, it is on the record, and it is not pursued further here. It also bears on
how the result should be read
when it arrives: the registered design gives the flagship pair no special
weight, testing a sign across ~53 pairs precisely so that no single quotable
pair carries the verdict. A lone pair at one-sided p ≈ 0.12 settles nothing in
either direction. That property was registered before this coincidence surfaced,
and it is the reason the coincidence is survivable.

**The remediation above is weaker than blindness, and is labelled as such.**
Saying every threshold has a power rationale a reader can verify is independent
of observed `D` establishes that each COULD have been derived blind. It does not
establish that it WAS. A reader who accepts the rationale table has accepted a
possibility as a fact, and the document should not let the table stand in for
the stronger property.

**The stronger check was run. Outcome: two of three recovered, one did not.**

Desktop corrected the premise before executing — it was NOT uncontaminated, since
the day's correspondence had given it all three registered values, so a match
from desktop would have proven nothing. It instead ran a fresh agent with no
access to the conversation or any file, given the three rationales verbatim and
nothing else, one pass, committed answers.

| threshold | blind re-derivation | verdict |
|---|---|---|
| exclusion cut on per-pair power | exclude above 2.0x — "a doubling is the smallest effect worth calling a convention" | **MATCHES** |
| dilution rule | informative-only primary; all-pairs as robustness on passes; exclusion bookable ONLY on informative-only | **MATCHES** — derived the asymmetric verdict-licensing unprompted |
| positive-control gate | p90 floor, 2-of-7 (analytic binomial, false-cert .044) | **DIFFERS** from registered p75, 3-of-7 |

Thresholds 1 and 2 therefore recover the was-derived-blind property fully. The
gate does not, and desktop's reading is that this is not contamination evidence
but **underdetermination**: a family of (floor, k) combinations satisfies the
stated constraints, and which member you land on depends on inputs the rationale
does not fix — the real decoy distributions and a target effect size, neither of
which the blind agent had. That is lacan's original objection made concrete.

**Under the real decoy distributions, the re-derived gate is dominated:**

| corpus | gate | false-cert | power d=.05 | d=.10 | d=.20 |
|---|---|---|---|---|---|
| hh | registered p75, 3-of-7 | 0.039 | 0.281 | **0.835** | 1.000 |
| hh | re-derived p90, 2-of-7 | 0.043 | 0.188 | 0.587 | 0.999 |
| pku | registered p75, 3-of-7 | 0.030 | 0.068 | 0.129 | 0.388 |
| pku | re-derived p90, 2-of-7 | 0.043 | 0.068 | 0.119 | 0.279 |

**Governance, applied rather than argued.** The commitment registered at
`a44df66` says the re-derived values govern. It does not say "unless the author
prefers the original," and the moment to honour it is exactly the moment the
answer is unwelcome. Therefore: **p90 floor, 2-of-7 is the operative gate as of
this commit.**

A deviation back to p75/3-of-7 is PROPOSED, not taken, on a criterion that was
registered before the re-derivation existed — revision 5 states the gate is
chosen on both axes, false-certification AND power, and p75/3-of-7 dominates
p90/2-of-7 on both in both corpora. **The author does not adjudicate his own
deviation.** It goes to the lacan seat for audit; until lacan rules, p90/2-of-7
stands.

**A third option is on the table and may be better than either gate.** If the
rationale underdetermines the gate, the defect is in the rationale, not in the
choice between members. The stronger fix is to register the SELECTION RULE
rather than the values: *given the corpus's real decoy distribution, choose the
(floor, k) that minimises false certification subject to power >= X at d=0.10,
tie-broken by power.* That is reproducible by anyone holding the distributions,
removes author discretion entirely, and would have been derivable blind. Offered
to lacan alongside the deviation.

Commits, none of which contain a tier-2 result: `3c31609` (registration),
`11f3138` (dispersion decomposition), `98e41d8` (calibration and effect-size
conversion), `48d9592` (illustrative bounds), `4a3d47d` (this section),
`d1db3ea` (prior-look disclosure).

## Standing rules

The six standing rules are canonical in `notes/standing-rules.md` (TheoryMachines),
with their worked instances and the seats that made and caught each error. Not
duplicated here. Where they bite on this design:

- **Rule 1 (no null without stated power)** governs the MDE table, the
  20-occurrence pair exclusion, and the UNDERPOWERED row of the outcome table.
  It is the rule revision 1 of this document broke.
- **Rule 2 (unit is the family or word pair)** fixes the word pair as the unit
  and forbids pooling hh and pku into one sign test.
- **Rule 4 (control composition audited before contrasts)** governs the decoy
  draw for the frequency control: the matched non-chain pairs are drawn and
  their composition reported before `D_excess` is computed.
- **Rule 6 (no difference claim from comparing p-values)** governs any
  eventual hh-vs-pku or chain-vs-decoy contrast: it is tested directly on
  `D_excess`, and the per-corpus p-values are not the headline.

**Why these are rules on the arrangement, not a request for individual care.**
Four of the five errors behind rule 1 were caught by a seat other than the one
that made them, and none by the originating seat's own review. The natural
reading of a list of five errors is that the seats should be more careful. That
is not what the record shows: self-review caught none of them. The rules bind
the cross-seat arrangement, not anyone's diligence.
