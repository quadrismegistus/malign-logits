# M04/A on the passage corpus — run record, 13 Aug 2026

**STATUS: EXPLORATORY. Nothing here is quotable yet.** Roughly thirty tests ran across four scripts in one night. The ladder in §2 is the result worth having and it is **not what A registered** — by [5601]'s own principle a new design does not inherit an old registration's slot, so the ladder wants registering and re-running rather than publishing from this pass. Multiplicity is uncorrected except where stated.

Corpus `passage`, 42 lineage-representative pairs, k=8 unless noted. Unit is the **pair (lineage)** throughout: aggregate within a pair, then sign-test the 42 pair values. Permutation nulls flip the sign of each site's whole difference, preserving pair structure and every magnitude — the shape lacan's F-P got wrong at [5588]. Producers: `scripts/a_matched_control.py`, `scripts/a_dose_response.py`.

## 1. The headline replicates and is not luck

`D(arm) = mean_lp(aligned's beams | aligned) − mean_lp(base's beams | aligned)`, and the statistic is `Δ = D(faller) − D(comparison)`.

| contrast | window | median | neg/pos | p(sign) | p(perm) |
|---|---|---|---|---|---|
| faller − matched | k=8 | −0.0673 | 32/10 | 0.0009 | 0.0005 |
| faller − matched | k=32 | −0.0233 | 31/11 | 0.0029 | — |
| faller − matched | full | −0.0081 | 24/18 | 0.4408 | — |

Permutation null: 20.9 ± 3.2 negative pairs against an observed 32. **Truncation, not filtering** — `arraySlice(logprobs, 1, k)` with no length predicate, since sequence length is an outcome of the model given the injected word and filtering on it is a collider. Diagnostic (arm×role length interaction) passed: +0.04 tokens, 24up/18dn.

**Caveat that rides on every k=8 number here:** the declared 85% retention rule **never fired** — no k reaches it, worst cell 58.3% at k=8, driven by `bloomz-7b1` whose aligned arm averages 15.5 tokens. k=8 is a **fallback**, reported as such by the producer.

## 2. THE LADDER — the result worth registering

The four arms are **not** two movers with two controls. Aligned probability relative to the faller, median log2(q_arm/q_faller): `matched` −0.024, `riser_matched` **+0.162**, `riser` **+3.668**. So `faller`, `matched` and `riser_matched` are a **three-rung ladder at the faller's own q**, differing only in the direction alignment moved them (movement log2(q/p): **fell −1.64, flat −0.00, rose +1.25**); `riser` is a fourth arm 3.7 log2 higher in probability.

| contrast (q held) | median | neg/pos | p(sign) | p(perm) |
|---|---|---|---|---|
| **fell − didn't move** | **−0.0673** | 32/10 | 0.0009 | 0.0005 |
| **rose − didn't move** | **+0.0345** | 14/28 | 0.0436 | 0.011 |
| **fell − rose** | **−0.0806** | 35/7 | <0.0001 | 0.0005 |

**Monotone in the direction of movement, with the word's own aligned probability held constant.** The middle rung is a true zero with a real effect on either side. This rules out improbability (all three rungs matched on it), movement-in-general (the two directions go opposite ways), and demotion-as-special-case, in one design.

**`riser` − `riser_matched` is NOT a direction test** and an earlier reading of it as one produced a false "promotion does nothing" asymmetry. It compares two risers 3.5 log2 apart in probability. Recorded because the null it produced (−0.003, p=0.64) looks exactly like a direction result and is not one.

## 3. What the effect is made of — and it is not damage

`A|A` = aligned's own text under aligned (**the chain the damage thesis is about**). `B|A` = base's text under aligned.

| term | fell − flat | | rose − flat | |
|---|---|---|---|---|
| **A\|A** | −0.0262 | p=0.28 | **+0.0302** | p=0.020 |
| **B\|A** | **+0.0440** | p=0.008 | +0.0163 | p=0.16 |

`A|A` for fell-vs-flat tested at every window — k=4, 8, 16, 32, full: −0.020, −0.026, −0.007, +0.003, +0.009; sign p 0.28–0.64; **permutation p 0.13–0.95. Null everywhere.**

**Promotion measurably improves the aligned model's own continuation; demotion does not measurably degrade it.** What demotion does is make the *base's* continuation more acceptable to the aligned model. And the level matters: `D` sits at **≈ +0.8 on every arm** — the aligned model always prefers its own continuation by a wide margin, and demotion shrinks that margin from ≈+0.81 to ≈+0.74. **A margin narrowing, not a chain breaking.**

**This contradicts A's published mechanism.** A recorded a split BY SCORER ("both terms under the aligned scorer move; neither under the base scorer does"). The four terms split BY TEXT: the movers are the base-text terms. Under the base scorer the same faller contrast gives −0.0247 (28/13, p=0.028, perm 0.014), so **86% of the headline is visible to a model that never underwent alignment**; the aligned-specific residue (triple difference) is −0.0093, 28/14, p=0.044, perm 0.019.

## 4. What does NOT grade it

**Demotion magnitude — null, and the within-arm test is uninformative.** Within the faller arm ρ=−0.003, 22/19, p=0.76 — but every faller fell by construction (IQR 1.01 log2, p95 −1.05, none less than 2×), so the predictor is truncated to its own tail. Across all four arms ρ=+0.077, 39/42, p<0.0001, **but that is collinear with the word's own aligned probability**, which is the mechanism the matched control exists to remove. The ladder in §2 is the clean version of this question.

**Domain — no transgressive concentration, and it inverts.** `non-transgressive` −0.057, 32/9, **p=0.0004** (survives Bonferroni over 8); `animal` −0.056, p=0.0015; `betrayal` −0.086, p=0.039; `violence` −0.044, p=0.028; **`sexual` −0.044, p=0.12 (ns)**; `taboo` −0.036, p=0.15 (ns); `property` p=0.27. **F14's content-grading, whose flagship is sexual_explicit, does not replicate on this corpus.**

**K norms — null.** `fields.k_rating`, 100% coverage on all 1,053 arm word types. **Ranks only** (`fields.py:99` — charge and concreteness move in level between instrument versions while holding order at r 0.88), `register_level` and `vulgarity` carried as description only per the same block. charge, concreteness, valence: null. Two nominal hits, transgressiveness-contrast and bodily_harm-contrast at p=0.028, **both pointing opposite to F14's prediction** (more transgressive faller → weaker effect) and **neither surviving Bonferroni over 14** (threshold 0.0036).

**Base-probability confound — detected, tiny.** The faller outranks its control in base probability on 99.9% of sites (median +1.80 log2), so it is the leading alternative for the `B|A` rise. ρ(basegap, B|A) = +0.015, 12/29, p=0.0115: consistent in direction, negligible in size. `D_aligned` is flat against it (p=0.53), and tercile stratification shows **no dose-response** (low −0.068 p=0.028, mid −0.044 ns, high −0.060 p=0.010). **Bounded, not eliminated** — the smallest tercile is still a 2.1× gap, so no zero-gap stratum exists and this cannot speak to one.

## 5. Files

    a_matched_control.json        headline + diagnostics
    a_riser_replication.json      A's original faller−riser contrast
    a_four_terms_null.json        four terms, base-scorer + triple, permutation nulls
    a_basegap_strat.json          base-probability tercile stratification
    a_dose_response.json          demotion / basegap / domain / K norms
    a_dose_response_fullrange.json  four-arm gradient (confounded — see §4)
    a_ladder.json                 §2 and §3
    a_AA_direct.json              A|A across five windows
    a_offset_regression[_terms].json   the direct slope-vs-offset regression
    a_offset_predictors.json      log q vs log p vs zipf frequency
    a_pickiness[_pooled|_scaling].json  the sensitivity comparison and its scale test
    a_partial_pref.json           the three-phase handover
    a_position_curves.json        cached position data behind every figure

## 6. Owed

1. **Register the ladder and re-run it.** It is the result and it is currently exploratory.
2. **Docket post withdrawing A's scorer-split mechanism** — a correction is a new claim and inherits none of the original's scrutiny.
3. **F14 relation.** Its rescoped form (amplification at targeted sites, base shares the structure) is compatible with §3. Its content-grading is contradicted by §4 and should be marked as failing an independent replication.
4. The k=8 fallback and `bloomz-7b1` want addressing before any window-dependent number is quoted.

---

# Second half of the 13 Aug run — preference, range, and what the effect is made of

**Same exploratory status as everything above.** These sections were run after §1–§6 and answer a different question: not "does demotion matter" but "what governs the surprisal of the continuation at all". Roughly thirty further tests; multiplicity uncorrected except where stated.

## 7. Preference beats demotion history, and it is not close

Within pair, across all four arms, Spearman then sign test over 42 pairs:

| predictor | outcome | median rho | neg/pos | p |
|---|---|---|---|---|
| **log q** (aligned's preference for the forced word) | A\|A k=8 | **+0.1165** | **1/41** | <0.0001 |
| log q | A\|A full | +0.1065 | 2/40 | <0.0001 |
| demotion history | A\|A k=8 | +0.0692 | 7/35 | <0.0001 |
| demotion history | A\|A full | +0.0377 | 13/29 | 0.020 |

**41 of 42 pairs is the most consistent relationship anywhere in this analysis**, and it holds at the full window where every demotion effect has died.

**The within-faller-arm demotion test is a null AND uninformative, and the second half is the point.** rho −0.003, 22/19, p=0.76 — but every faller fell by construction: IQR 1.01 log2, p95 −1.05, none less than 2x. A predictor truncated to its own tail attenuates toward zero whatever the truth is. The four-arm version (rho +0.077, 39/42, p<0.0001) is not the fix either — across arms, demotion is near-collinear with the word's own aligned probability, which is exactly the mechanism the matched control exists to remove. **§2's ladder is the clean form of this question and the only one to quote.**

**The matched control is therefore holding the LARGE variable fixed and measuring the residual.** That is the right control for "does demotion history matter beyond current preference", and it is why everything downstream of it is small. Not a defect; a scope.

## 8. Range: a floor, and a handover between two predictors

Disjoint token windows (the earlier cumulative means shared data, so non-decay was untestable), rho(log q, A|A):

    1-4 +0.1075 | 5-8 +0.0864 | 9-16 +0.1226 | 17-24 +0.0977 | 25-32 +0.1009
    33-48 +0.0900 | 49-64 +0.0715 | 65-96 +0.0531 | 97-128 +0.0618
    129-192 +0.0456 | 193-256 +0.0526

Every window p<0.0001 except the last (p=0.0029). **Flat at ~0.10 through token 32 — 9-16 is the HIGHEST of the eleven, so there is no early decay — then a decline across 33-64, then a second flat stretch at ~0.05 to the end.** Site attrition across the whole span is 3%, so this is not sequences dropping out.

PARTIAL Spearman, each controlling for the other (rho(log q, log p) = 0.605 within pair, correlated but far from collinear):

| window | log q \| log p | | log p \| log q | |
|---|---|---|---|---|
| 1-4 | **+0.0709** 39/42 | <0.0001 | +0.0188 26/42 | 0.16 |
| 5-8 | **+0.0569** 38/42 | <0.0001 | +0.0182 26/42 | 0.16 |
| 9-16 | +0.0746 36/42 | <0.0001 | +0.0443 33/42 | 0.0003 |
| 17-32 | +0.0511 34/42 | 0.0001 | +0.0566 33/42 | 0.0003 |
| 33-64 | +0.0403 32/42 | 0.0009 | +0.0505 34/42 | 0.0001 |
| 65-128 | +0.0253 27/42 | 0.088 | **+0.0534** 37/42 | <0.0001 |
| 129-192 | +0.0220 25/42 | 0.28 | **+0.0606** 34/42 | 0.0001 |
| 193-256 | +0.0188 24/42 | 0.44 | **+0.0396** 35/42 | <0.0001 |

**Three phases: aligned preference alone for the first 8 tokens, both from 9-64 (crossing where they are level at 17-32), base preference alone from 65 to 256.** Read as a decomposition of the word's standing rather than as two agents: `log q | log p` is what alignment ADDED to the word, and it is spent by token 64; `log p | log q` is the word's ordinary linguistic fit, and it runs the whole passage.

## 9. The direct regression (RH's framing, and the clearest statement of the finding)

A|A surprisal on log2 q, OLS within pair across all four arms, median of 42 pair slopes. **Negative = wanting the word more means less surprise downstream.**

    +1 -0.041 39/42 | +2 -0.053 40/42 (peak) | +4 -0.016 | +8 -0.021 | +16 -0.028
    +32 -0.015 | +64 -0.014 | +96 -0.007 (n.s., the only one) | +128 -0.013
    +192 -0.014 | +256 -0.016

In units: the arms span ~3.7 log2, so ~0.20 nats of extra surprisal at +2 and still ~0.05 nats at +256 for a word the model did not want versus one it did. Sites fall 3.6% (16,712 -> 16,103).

**All four terms: THE SPLIT IS BY TEXT, NOT SCORER.** A|A and A|B track each other and sit consistently below B|A and B|B at nearly every offset. The aligned model's preference shapes what IT writes, and both readers see it in the text — a PRODUCTION effect, not an evaluation one. **This is the second independent route to the same conclusion as §3, and it contradicts A's published "disturbance of the aligned model's evaluation".**

## 10. It is not frequency, and it is not a generic property of the word

Standardised slopes (nats per 1 SD, so the predictors are on one scale):

| offset | A\|A ~ log q | A\|A ~ log p | A\|A ~ **zipf** |
|---|---|---|---|
| +1 | −0.082 | −0.066 | **+0.141** 40/42 |
| +2 | −0.123 | −0.099 | **+0.099** 37/42 |
| +32 | −0.043 | −0.038 | −0.008 n.s. |
| +256 | −0.035 | −0.034 | −0.024 n.s. |

**Frequency has the OPPOSITE SIGN in the first tokens** — a more frequent forced word makes the continuation MORE surprising, 40 of 42 pairs — and is null from +16. A high-frequency word is uninformative and leaves the continuation open; a rare specific one constrains it. **The frequency confound is checked and it is not the explanation.**

And B|B — text neither written nor scored by the aligned model — is better predicted by **log p** (−0.090 vs −0.054 at +1; −0.024 vs −0.006 at +32) while A|A is better predicted by **log q**. **Each model's own preference best predicts its own writing.** If `q` were merely proxying a generic word property, both terms would be predicted equally by both. An earlier reading of this as "about half is an ordinary property of the word" is WITHDRAWN: the B|B slope is the same phenomenon running inside the base model.

## 11. Is the aligned model pickier? NOT DETECTABLY — and the appearance was an unpaired comparison

Paired within pair, (A|A slope on its own log q) − (B|B slope on its own log p): null at every offset on raw and per-SD metrics (p 0.088–1.00). Four of 27 tests nominal on the mean-normalised metric; Bonferroni threshold 0.0019; none survives.

**SETTLED 13 Aug by a scale test, and the pooled test alone would have got it wrong.** Pooling each pair's difference across the 9 offsets first, then one sign test: raw −0.00133 (21/20, **p=1.00**), per-SD −0.00651 (25/16, p=0.21), **per-SD-over-mean −0.00980 (34/7, p<0.0001)**. The third row is not a finding, and the reason is measurable rather than arguable. Two scaling models make two predictions about the paired log ratio of slopes:

    absolute-constant  predicts log(sA/sB) = 0            observed -0.0195  22-/19+  p=0.76   FITS
    proportional       predicts log(sA/sB) = log(mA/mB)   log(mA/mB) = -0.2851       REJECTED
                                                          difference +0.3883 8-/33+ p=0.0001

**Sensitivity does not scale with baseline surprisal.** The aligned model's mean surprisal is 25% lower (2.116 vs 2.911) and its slope is the same. So dividing by the mean is the correct normaliser only under the model the data reject, and it converts an equality into a 34/7 difference by shrinking one denominator. **A normalisation is a claim about scale, and this one is testable — it was tested, and it fails.**

**AND THE TEST I FIRST PROPOSED FOR THIS IS UNUSABLE, recorded so nobody runs it again.** "Regress the ratio difference on the baseline gap and read the intercept" returns +0.00736, t=3.86, apparently surviving. But the intercept is the difference AT ZERO GAP and **there is no zero gap in the data** — all 41 pairs have the aligned model lower, median gap −0.795 nats. It estimates the quantity of interest exactly where nothing was observed. The scale test above compares the two models on data that exist.

**Standing answer: the aligned model is NOT more sensitive to having its preferences violated.** Same absolute sensitivity, 25% lower baseline. The only difference between the models here is F18's entropy compression, and it belongs to F18.

**The §10 table makes it LOOK like 2x** (−0.048 vs −0.022 at +8, −0.043 vs −0.024 at +32). Those are medians of two separately-computed distributions and differencing them is not licensed. **Paired, it vanishes.** What survives is a consistent lean — negative at 8 of 9 offsets raw and 9 of 9 normalised, 23–28 of 42 pairs each time — which is a hint and not a result. **OWED: one pooled test (average each pair's difference across offsets, then a single sign test), which is the right shape since the offsets are not independent.**

## 12. Figures — `meta/M04_syntagmatic/figures/`, producer `scripts/a_position_figures.py`

    A_position_surprisal[_full]        levels, facet by arm, colour by term
    A_position_contrast[_full]         each arm vs the matched non-mover, paired
    A_term_facets[_raw][_full]         levels, facet by term, colour by arm
    A_term_facets_contrast[_raw][_full]  same, against the non-mover
    A_offset_slope                     the §9 slope against token offset
    A_offset_slope_terms               the same, all four terms

**Position 1 is DROPPED, not smoothed away** — the undisturbed arm spikes to 4.4–5.4 nats there and that one point sets the y-range for every panel. **Positions are aligned on the SENTENCE via `n_forced_tokens`**, because the forced word is not itself scored (verified: `logprobs[0]` varies across samples of one forced word while log q is fixed); plotting raw indices would compare sentence position 2 against position 1, which is the offset that makes A's forced-vs-undisturbed control invalid. `_raw` variants carry no smoothing; the others use a 5-position centred rolling mean applied AFTER the median-over-pairs, disclosed on every figure.

**The undisturbed arm is a REFERENCE CURVE, never a control.** Correcting the offset removes one of A's two objections; the commitment boundary remains.

**Read the figures for shape and the sign tests for effects, and do not read one against the other** — they aggregate differently (per-position median of per-pair means vs pair-median of a windowed mean over sites), so the curves will not reproduce the tables.

## 13. Two position-band results from reading the figures — EXPLORATORY, seven tests per arm, no correction

A|A surprisal minus the matched non-mover, + = more surprising:

- **The faller's apparent early peak is not there.** Bands 2–4 and 5–8 are 20+/22− and 22+/20−, coin flips.
- **The faller's A|A effect is LATE and NEGATIVE**: −0.0166 at 33–64 (p=0.020), −0.0161 at 65–128 (34 of 42, p=0.0001), −0.0119 at 129–256 (p=0.0029). **Every cumulative window (k=8/16/32/full) returned null because they average this against a null early stretch.** Same lesson as §8's disjoint grid.
- **The undisturbed arm is never significantly more surprising**; it is −0.106 at positions 2–4 (35 of 42, p<0.0001), i.e. markedly LESS, decaying to exactly nothing by position 9. The benefit of having chosen your own word, spent in eight tokens.

---

# 14. THE DECLARED RUN — `scripts/ladder_confirm.py`, 13 Aug 2026

**Use (1) of the plan's §7 and nothing more: a disciplined re-analysis. The ladder was found in this data, so nothing below is confirmed.** The decision rules are executed by the producer and printed as SUPPORTED / NOT SUPPORTED, computed from the plan's own criteria rather than asserted.

**Retention gate applied: n = 42 -> 40.** `bloomz-7b1` (58.3%) and `recurrentgemma-9b-it` (82.8%) excluded. **Every number moved, which is the check that the gate ran** — a re-run reproducing the exploratory figures exactly would mean it had not.

| | verdict | |
|---|---|---|
| **H1** monotone in direction | **SUPPORTED** | fell−flat −0.0516 (29/11, p=0.0064, perm 0.0050) · rose−flat +0.0510 (13/27, p=0.0385, perm 0.0045) · fell−rose −0.0600 (33/7, p<0.0001, perm 0.0005) |
| **H2** split by text not scorer | **SUPPORTED** | A\|A −0.0691 p=0.0007 · A\|B −0.0712 p=0.0003 · B\|A p=0.64 · B\|B p=0.52 |
| **H3** ladder is the small part | **SUPPORTED** | ρ(log q) +0.1204 vs ρ(demotion) +0.0692; paired +0.0507, 37/40, p<0.0001 |
| **N1** not frequency | **SUPPORTED** | zipf +0.179 at +1 (38/40, opposite sign), null by +16 (p=0.87) |
| **N2** not transgression-graded | **SUPPORTED** | `sexual` p=0.14 ns; strongest is `betrayal` −0.124; `animal` p=0.024 |
| **N3** not a shock | **NOT SUPPORTED** | see below |

**THE LADDER BECAME MORE SYMMETRIC UNDER THE GATE.** Down weakened (−0.0673 -> −0.0516), up strengthened (+0.0345 -> +0.0510). The two excluded pairs were pulling the asymmetry.

## N3 failed, and it failed AWAY from damage

The rule had two parts and both broke. **(a)** The full cumulative window now clears (+0.0091, 13/27, p=0.0385) where it was null. **(b)** The 129–256 band lost significance (p=0.27, was p=0.0029 at n=42) — the two gated-out pairs were carrying it.

**Every value is positive in logprob, i.e. LESS surprising.** N3's substantive claim was "not damage", and the run shows *less* shock than the rule predicted: the anti-damage effect at 33–64 (p<0.0001) and 65–128 (p=0.0002) is now strong enough to show through cumulative averaging, which is precisely what broke the "cumulative stays null" clause. **Damage requires the opposite sign and appears nowhere.**

**The rule was too specific about WHERE the effect would be invisible**, having been tuned to an n=42 picture the gate then changed. **And it was the most recently amended criterion** — rewritten after registrar's [5688] catch — i.e. the one with the least scrutiny behind it. A rule amended late is a rule least tested.
