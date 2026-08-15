---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/zh_ordering.json and results/zh_fluency_arms.json
date: 2026-08-14
role: finding
topics: [drift, ordering, cross-lingual, chinese, arms, fluency, jakobson]
description: "Alignment makes models write BETTER CHINESE (20/25 pairs, p=0.0041, judged blind at Cohen kappa 0.776), and that gap PREDICTS the arm effect on total_drift (spearman -0.497, p=0.0116) that `crosslingual_arms.md` reports as its Chinese headline. Restricting to the 6 pairs where both models write Chinese collapses total_drift SEVENFOLD (-0.0314 -> -0.0046) and leaves an ORDERING effect unmoved (-0.0090 -> -0.0087), which is independent of fluency (rho +0.215, p=0.30). So the surviving Chinese arm effect is on COMBINATION and not on SELECTION. The ordering test is nominal only (18/25, p=0.043) and the restricted sign tests confirm nothing at n=6, as pre-declared. English is NULL on ordering (14/25, p=0.69) while its spread effect holds (24/25)."
---
# Chinese fluency is an arm variable, and it separates the spread effect from the ordering effect

Plans: `plans/plan_zh_ordering.md`, committed at `f9480f7a` before its producer
existed. Producers: `scripts/m06_zh_fluency_sample.py`, `m06_zh_fluency_judge.js`,
`m06_zh_fluency_join.py`, `m06_zh_fluency_arms.py`, `m06_zh_ordering.py`.
Results: `results/zh_fluency_verdicts*.json`, `zh_fluency_arms.json`,
`zh_ordering.json`. Single pass; [5503] applies.

**No new generation and no new embedding.** Every number here comes from
generations already in the store and from `mean_pairwise`, which
`m06_crosslingual_drift.py` persisted for exactly this test.

## Which population these 25 pairs are, stated because there are four

registrar mapped four model-pair rosters at `docs/model-populations.md` (two
live in ClickHouse, two frozen on disk), and the two live ones disagree by
eight members. **Every seat forgets which one it is using**, so:

    data/base_aligned_pairs.json          54   FROZEN, the source here
      minus `ambiguous`                   ->   the arms producer's filter
      minus models absent in either lang  ->   25, this population

Checked against registrar's heterogeneity flag rather than assumed: the 25 are
**24 dpo + 1 ppo**, no `warn_sft_as_aligned`, none ambiguous, all present in
the frozen file, and `BAAI/Aquila2-7B > AquilaChat2-7B` -- the base->EGO
(SFT-only) member that makes the frozen 46 heterogeneous elsewhere -- **is not
in this population.** So all 25 contrast base against a preference-tuned arm
and the contrast is of one kind throughout.

## Why this exists: `cjk_tier` measures a vocabulary, not a model

`data/model_registry.json` carries `cjk_tier`, derived from `cjk_chars` -- the
count of CJK characters in a model's TOKENIZER VOCABULARY. It says what a model
can represent and nothing about what it does:

    bigscience/bloomz-7b1        FLUENT, 5,058 CJK chars   worst of all 58 models
                                                           (answers Chinese prompts
                                                           in English, Spanish, Telugu)
    LLM360/AmberSafe             NOMINAL, 700 chars        scores 2.00
    HuggingFaceTB/SmolLM2-360M   NOMINAL, 77 chars         emits 65% CJK via
                                                           byte-level BPE fallback

The tier does separate in aggregate (FLUENT 2.08 against NOMINAL 1.07) and
fails per model, which is the wrong shape for a population filter.

**And character statistics cannot settle it, because they have no null and
they run OPPOSITE to intuition.** Real Chinese reuses common characters and
builds two-character words, so fluency LOWERS the type-token ratio and RAISES
bigram repetition. Predicted from that reasoning before judging, then measured
over 57 models:

    spearman(TTR,      judged score) = -0.773   p=1.8e-12
    spearman(bigram repetition, ...) = +0.566   p=4.6e-06
    spearman(CJK fraction, ...)      = +0.806   p=4e-14

`zai-org/glm-4-9b-chat-hf` has the most "degenerate" character profile in the
set (TTR 0.422, bigram repetition 0.263) and is joint-best on reading. **A
filter built on the naive reading would have excluded the best Chinese writers.**

## The instrument

348 continuations judged at 6 per model, then 812 more at 20 per model, on
`fluent / flawed / broken / not_chinese`, by 12 and 16 Claude Opus 5 agents.
Batches carry `key`, `prompt`, `continuation` and nothing else; keys are
shuffled at assignment and the items shuffled again before batching, so
neither key order nor batch membership encodes the model.

**Agreement, from 160 passages re-emitted under fresh keys in round 2 and
judged by a different agent that could not tell them from first ratings:**

    exact 0.844 | adjacent 0.994 | Cohen kappa 0.776   (n=160)

Residual disagreement is almost entirely `broken`/`flawed`, the expected
boundary. 971 of 972 round-2 items returned; `r20807` was dropped by its agent
and is recorded as missing rather than backfilled.

Judges were told explicitly to rate language and never content -- these are
transgressive stimuli, and a judge that marked down violent content would
produce an arm effect out of the instrument, since aligned models emit less of
it.

## 1. Alignment improves Chinese fluency

    20 pairs more fluent | 5 less | 0 tied
    sign test p=0.0041 | mean +0.372 | 95% CI [+0.228, +0.528]

At 6 passages per model this was 14/7/4 at p=0.19, i.e. nothing. **The result
is a property of the sample size, and the n=6 version should not be cited.**

### The result is robust to the ordinal coding, and NULL on one of them

The score above is a mean over an ordinal scale treated as interval
(3/2/1/0). **Nothing stated a reason for that and nothing tested it**, so it
was tested. Four codings agree and the fifth is informative:

    interval 3/2/1/0 (used)      20/5   p=0.0041
    binary fluent|flawed         20/4   p=0.0015
    strict fluent only           15/0   p=6.1e-05     unanimous
    compressed 3/2/0/0           23/2   p=1.9e-05
    IS-CHINESE-AT-ALL            12/7   p=0.36        NULL

**Alignment does NOT make a model more likely to stay in Chinese.** The
`not_chinese` rate -- English, other scripts, markup, empty -- does not differ
by arm. What differs is the quality of the Chinese that does get produced, and
the effect concentrates at the TOP of the scale: coding for `fluent` alone is
unanimous at 15/0.

**So the claim is narrower than "aligned models write Chinese and base models
write word salad."** It is: conditional on writing Chinese, aligned models
write it better. This matters for the confound below -- the contamination runs
through the broken/coherent distinction and not through language choice, which
is consistent with `order_ratio` being independent of it, since a coherence
difference is what an ordering measure is sensitive to.

## 2. That gap predicts the Chinese arm effect on `total_drift`

Across the 25 pairs, the fluency gap predicts the drift gap:

    zh total_drift  unmatched   spearman -0.497   p=0.0116
    (Bonferroni threshold for the 4-test family: p < 0.0125)

`total_drift` unmatched is the leg `crosslingual_arms.md` PERSISTED; its
matched legs, withdrawn as unreproduced at [5932], sit at p=0.066. Pairs where
alignment improves Chinese most are the pairs showing the most narrowing.

**A bge-m3 embedding of word salad is not a measurement of the same kind as an
embedding of prose**, and in twelve of the 25 pairs the base model produces
word salad while the aligned model does not.

## 3. Restriction separates the two effects

Keeping only pairs where BOTH members write Chinese (`min(score) >= 2.0`,
6 pairs; `>= 1.5`, 8 pairs):

    metric        ALL 25     >=1.5      >=2.0     vs fluency gap
    total_drift   -0.0314    -0.0113    -0.0046   rho -0.497  p=0.0116
    order_ratio   -0.0090    -0.0087    -0.0087   rho +0.215  p=0.30

**The same restriction collapses the set-diameter effect sevenfold and leaves
the ordering effect unmoved, and the fluency gap predicts the first and not
the second.**

### CORRECTED 2026-08-15: the argument is the correlations, NOT the restriction

The paragraph above compared two CHANGES with no uncertainty on either. Prompted
by malign at [6188] -- *freezing a rule does not supply it an error bar* -- both
comparisons now carry a paired bootstrap (one resample of PAIRS per replicate,
both metrics computed on it, 20,000 replicates, in `m06_zh_ordering.py`):

    change under restriction  total_drift  +0.0221  [-0.0040, +0.0379]  INCLUDES 0
    change under restriction  order_ratio  +0.0001  [-0.0417, +0.0164]  INCLUDES 0
    DIFFERENCE OF THE CHANGES              +0.0249  [-0.0183, +0.0584]  NOT ESTABLISHED

    confound rho difference (n=25)         -0.694   [-1.184, -0.161]    ESTABLISHED

**So "collapses versus stays" is DESCRIPTIVE and not established.** I had
disclaimed the restricted p-values and then leaned on restricted POINT
ESTIMATES from the same six pairs, which is the same n with the uncertainty
hidden -- exactly the defect the plan warned about, committed one section below
the warning.

**What survives is the other leg, and it is the better one.** The fluency gap
predicts the `total_drift` arm effect (rho -0.497) and not the `order_ratio`
one (+0.215), and **the DIFFERENCE between those correlations excludes zero on
all 25 pairs**, not on the 6. The dissociation is real; it rests on the full
population rather than the underpowered subset, and the restriction figures
below should be read as illustration.

## The ordering statistic, and why it is a ratio

    order_ratio = mean_drift / mean_pairwise

Under a random sentence order the expected successive distance IS the mean of
all pairwise distances, so **1.0 is a null by construction rather than by
estimation.** P1 holds: 0.938-0.964 across arms and languages, so passages are
locally coherent and the metric behaves.

`crosslingual_arms.md` proposes the DIFFERENCE, `mean_drift - mean_pairwise`.
That is not scale-free, and alignment is established to shrink the whole set,
so both terms shrink and the difference carries the spread effect it exists to
remove. Measured: the difference is **null** (16/25, p=0.23) and collapses
under restriction (-0.0030 -> -0.0007). **Only the ratio separates them.**

## 4. The ordering effect is Chinese-only

    order_ratio  zh  18/25  median -0.0090  p=0.043     <- declared primary
    order_ratio  en  14/25  median -0.0003  p=0.69      <- thirty times smaller
    total_drift  en  24/25  median -0.0205  p=1.55e-06  <- spread effect holds

The spread effect is bilingual; the ordering effect is not. No directional
prediction was made for English and none is claimed.

## What this does to `crosslingual_arms.md`

Its Chinese headline (`total_drift` zh, 21/25, median -0.0314, p=9e-04) is
**reproduced exactly** -- the ordering producer refuses to report anything
until it recovers that contrast from the same parquets, and it does, to the
digit. Nothing here says the number is wrong.

**What is in question is what it measures.** It is confounded with an arm
difference in whether the model writes Chinese at all, and it does not survive
restriction to models that do. The finding's sentence *"first arm effect this
campaign has measured on Chinese generated text"* is the one at risk, since 12
of its 25 pairs are pairs whose base member largely does not produce Chinese.

**Not edited there.** A pointer is added to that document and its text stands;
this is add-beside, and the disposition is RH's.

## Why English is null: three explanations tested and discarded

Producer `scripts/m06_en_null_probes.py`, results `results/en_null_probes.json`.
**Recorded because a negative needs its number as much as a positive does**, and
because the next person to ask will otherwise re-derive all three.

**1. HEADROOM -- discarded.** *English base models are already coherent, so
alignment has nothing left to tighten.* Predicts the effect scales with how
badly the base writes. Against JUDGED base fluency, which is an independent
measurement and not a delta correlated against its own baseline:

    zh order_ratio vs base fluency   rho -0.075   p=0.72    FLAT
    zh total_drift vs base fluency   rho +0.352   p=0.084   headroom direction

The ordering effect is flat across base competence. **And `total_drift` trends
the headroom way, which is one more axis on which the two metrics differ**:
it tracks the fluency gap (-0.497) and it trends with base fluency (+0.352);
`order_ratio` tracks neither.

**2. LENGTH -- discarded.** *English passages have ~2.5 fewer sentences (5.35
against 7.79), and `order_ratio` is a ratio of two within-passage averages, so
it is noisier.* Matching the sentence count does not recover the effect:

    en  n_sents >= 8   13/24   median -0.0118   p=0.84
    zh  n_sents >= 8   18/24   median -0.0109   p=0.023

Chinese is stable across strata (-0.0090, -0.0085, -0.0109); English is null at
every one.

**3. SPLITTER -- real, large, and discarded as the explanation.** *Chinese goes
through stanza and English through NLTK, so the two are not segmented into the
same kind of unit.* The disagreement is severe -- over 416 English passages put
through both, only **39.2% get identical splits**, 44.2% agree even on the
count, mean Jaccard 0.695, and stanza breaks on markup and line structure where
NLTK does not (one passage: 7 sentences against 17). **But it is ARM-NEUTRAL:**

    stanza-minus-nltk sentences   base +0.990 | aligned +1.149   p=0.84

A disturbance that hits both arms equally cannot create or destroy an
aligned-minus-base contrast. **What survives is that the English LEVEL is
splitter-dependent, so the cross-language comparison of levels is not a
comparison of one quantity** -- but the within-language arm contrast is not
the thing at risk, and re-splitting English under stanza would not be a test
of the null.

**So the asymmetry stands unexplained.** Alignment tightens the Chinese chain,
does nothing measurable to the English one, and none of the three obvious
mechanisms accounts for it.

### One candidate exists, from another instrument, and it is n=1

Recorded here because it was posted to the docket at [6171]/[6172] and would
otherwise live only there. **It is not support and must not be cited as such.**

`plans/plan_projected_displacement.md` (malign) decomposes displacement into
SUPPRESSION and SUBSTITUTION on a per-prompt author-anchored axis. Its
demonstration, **n=1 per language and declared NOT A FINDING by its author**:

    EN full SFT   dN -0.031   suppression -0.033  substitution +0.002   95% SUPPRESSION
    ZH full SFT   dN +0.016   suppression +0.001  substitution +0.015   95% SUBSTITUTION

Substitution toward a consistent register installs the same vocabulary across a
passage, which would raise adjacent-sentence similarity relative to the
passage's spread -- what `order_ratio` measures. Suppression removes mass
without supplying a replacement and has no mechanism by which to tighten a
chain. **That would produce exactly this EN/ZH asymmetry.**

**Why this is worth recording and a tally would not be**: it is a different
instrument, corpus and unit arriving at the same asymmetry with a mechanism
attached. Agreement between independently constructed instruments is evidence;
a count of how many instruments agree is a fact about how many got built
(dario, [6193]). This is one construction, at n=1, and the honest status is
*the first candidate that survives contact with the three discarded above*.

**The test, which needs their instrument at pair scale and costs nothing here**:
if substitution tightens the chain, the per-pair suppression/substitution ratio
should predict the `order_ratio` delta across the 25 pairs -- the same shape as
the confound test that separated `total_drift` from `order_ratio`, run against
a different predictor.

**And the same post independently supports the cross-language caveat above.**
An English-built axis scoring Chinese candidates gives spearman +0.928 on rank
against a Chinese-built one, while 裤子 reads +0.009 on the first and -0.078 on
the second: *the ordering transfers and the origin does not.* Within-language
contrasts stand; a cross-language comparison of LEVELS does not.

## Incidental: aligned models emit more list structure

From probe 3, on English generations:

    markup tokens per passage    base 0.23   aligned 0.34
    passages carrying any        base 3.4%   aligned 12.0%   p=0.0013

**Three and a half times the prevalence.** This is the register reading in its
most literal form -- alignment installing enumerated structure -- and it is the
same object as E-ASSIST-AMBIENT's unbidden assistant frame appearing in
generation. **CAVEAT: the detector is a crude token list** (`<li>`, `</li>`,
`<p>`, newline-bullet patterns) **and the base rate is small.** A lead, not a
measurement, and it wants a specified counter before it carries any weight.

## Limits, stated

- **25 pairs throughout**, and 6 in the restricted population. The ordering
  result is nominal (p=0.043) on a pre-declared directional test, and would
  not survive correction if the secondary cells were counted as a family.
- **The restricted medians are computed on a SUBSET of the same pairs**, so
  their stability is not independent evidence. The argument is comparative:
  the identical subsetting operation collapses `total_drift` and does not
  move `order_ratio`.
- **`mean_drift` survives everywhere** (6/6 both languages). It is a mixture
  and this is consistent with it: spread in English, ordering in Chinese.
- **The fluency judge is a language model**, agreeing with itself at
  kappa 0.776 across independent agents. That is reliability, not validity;
  no human rater has checked it.
- **`bloomz-7b1` produces almost no scorable Chinese** (median 40.5 chars),
  so `bloom-7b1 -> bloomz-7b1` is not usable as a Chinese pair despite both
  members being FLUENT tier.
