---
type: plan
date: 2026-08-14
role: plan
status: declared
topics: [drift, ordering, cross-lingual, chinese, arms, jakobson]
description: "The separable ordering test named as unrun in crosslingual_arms.md: does alignment shape the CHAIN or only the SET? Declared before computing, with directions, population and the scale-free statistic fixed in advance."
---

# Plan: the ordering test on Chinese generation

`findings/crosslingual_arms.md` names this test and does not run it:

> **The separable version, not run:** within-passage permutation holds the
> sentence set fixed BY CONSTRUCTION, and under a random order the expected
> successive distance is just the mean of all pairwise distances. So
> `mean successive - mean all-pairs` is a pure ORDERING measure with
> composition removed exactly. That is the test of whether alignment shapes
> the chain rather than the lexicon, and this instrument does not answer it.

No new compute is required. `m06_crosslingual_drift.py` persisted
`mean_pairwise` alongside `mean_drift` for exactly this purpose, and both are
in the committed per-passage parquets.

## THE STATISTIC IS A RATIO, NOT THE DIFFERENCE

The finding proposes `mean_drift - mean_pairwise`. **That subtraction is not
scale-free and this matters here**, because the campaign has already
established that alignment SHRINKS the whole sentence set in Chinese
(`total_drift`, a set diameter, 21/25 pairs). If every distance in a passage
shrinks, the successive distances and the all-pairs distances both shrink,
and so does their difference. The raw subtraction would therefore still carry
the spread effect it exists to remove.

    order_ratio = mean_drift / mean_pairwise

Under a RANDOM sentence order the expected successive distance IS the mean of
all pairwise distances, so **the null value is exactly 1.0** and it is a null
by construction rather than by estimation. Below 1: successive sentences are
closer than random pairs, i.e. the passage is locally coherent. Above 1: the
order is anti-coherent.

Both are reported. The RATIO is primary and the difference is secondary and
declared as such here, before either is computed.

## POPULATION

Two, both stated in advance, because the fluency result forces it:

  A. **All 25 arms pairs** — the population `crosslingual_arms.md` used.
  B. **Pairs where BOTH members write Chinese**, judged blind at
     `results/zh_fluency_verdicts*.json`, threshold `min(score) >= 2.0`,
     which is **6 pairs**, and `>= 1.5`, which is **8**.

B is the population that matters. `zh_fluency_arms.json` establishes that
alignment improves Chinese fluency (20/5/0, p=0.0041) and that the gap
predicts the `total_drift` arm effect (spearman -0.497, p=0.0116). A
semantic-geometry measurement over word salad is not the same measurement as
one over prose.

**MDE, stated before the run:** a sign test over 6 pairs needs **6 of 6** for
p=0.031 and cannot reach p<0.05 any other way; over 8 pairs it needs 8 of 8
(p=0.0078) or 7 of 8 (p=0.070). So population B can only ever return
"unanimous" or "nothing", and a non-unanimous result there is not evidence of
absence. This is a low-powered confirmatory test and is declared as one.

## PREDICTIONS

**P1, sanity, both arms and both languages.** `order_ratio < 1`. Generated
text should be locally coherent; successive sentences are nearer than random
pairs. If this fails the metric is not measuring what it is supposed to and
nothing below is interpretable.

**P2, directional, the test.** `aligned - base` on `order_ratio` is
**NEGATIVE**: alignment makes the chain MORE locally coherent. The reasoning
is that alignment rewards discourse scaffolding, connectives and enumerated
structure, all of which raise the similarity of adjacent sentences relative
to the passage as a whole. Both available readings of alignment point this
way, which is why the prediction is directional.

**P3, the discriminating one.** If P2 holds on population B while
`total_drift` does not (`zh_fluency_arms.json`: 4/6, median -0.0046, a
sevenfold collapse from -0.0314), then the surviving Chinese arm effect is on
COMBINATION and not on SELECTION -- the Jakobson axis the article's apparatus
turns on. If P2 fails on B while `mean_drift` holds, then `mean_drift`'s
survival was the spread effect leaking through the mixture and there is no
ordering claim.

**What would refute the whole thing:** `order_ratio` deltas that are null on
both populations, or that track the fluency gap as `total_drift` does. The
same confound test is therefore run on `order_ratio` and its result is
reported whatever it is.

## WHAT IS NOT CLAIMED

English is computed and reported for symmetry; no directional prediction is
made for it, and the language DiD is not part of this plan.

Producer: `scripts/m06_zh_ordering.py`. Committed BEFORE the producer exists.
