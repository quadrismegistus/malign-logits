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
the second.** That is the argument, and it is not the restricted p-values --
`plan_zh_ordering.md` declared in advance that a sign test over 6 pairs can
only return 6/6 (p=0.031) or nothing, and it returned 5/6 and 6/8.

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
