---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces
date: 2026-08-13
role: finding
topics: [drift, instrument, embedding, audit]
description: "An audit of the drift metric family, prompted by RH asking whether the quadrant names are earned. Four defects, each measured: total_drift is ORDER-INVARIANT (a dispersion measure, not a trajectory one); it is 92% NOISE at the passage level (ICC 0.082); DIRECTEDNESS IS SENTENCE COUNT (Spearman -0.923, R^2 0.795 against 1.68/n) and carries essentially no shape information; and the 75-word truncation analyses 46% of each generation while RETAINING FEWER passages than no truncation at all. mean_drift dominates total_drift on every property. The measure that would do what directedness was believed to do is the within-passage ordering contrast, n-controlled by construction."
---
# Drift metric audit: what these numbers can and cannot mean

Prompted by RH, 2026-08-13: *"Are you convinced by the quadrants and their
names?"*, then *"is this just an artefact of the types of measurement they
are?"*, then *"how is directedness measured? I never understood it."* Each
question found a defect. All four are measurements, not readings.

## 1. `total_drift` is ORDER-INVARIANT

`total_drift = 1 - min(pairwise cosine)` is the DIAMETER of the passage's
sentence set. Verified by permutation on three random 8-sentence passages:
identical to four decimals after shuffling, every time, while `mean_drift`
changed every time.

**So it measures SEMANTIC SPREAD, not trajectory.** No claim about how a
passage moves can rest on it, and in particular an order-invariant axis
cannot distinguish a metonymic chain from an undirected scatter of the same
diameter.

## 2. `total_drift` is 92% NOISE at the passage level

Variance decomposition on 9,501 cells with three samples each (within-cell =
same model, same prompt, different generation = pure sampling noise):

    metric           total SD   within SD   between SD    ICC
    mean_surprisal     0.7343      0.5822       0.4475   0.371
    total_drift        0.1397      0.1338       0.0401   0.082
    mean_drift         0.1191      0.1104       0.0448   0.141

`total_drift` is an EXTREME statistic -- a minimum over ~10 pairwise
similarities among ~5 sentences -- which is the noisiest thing constructible
from few units.

**Consequences.** (a) Any median split on it classifies passages nearly at
random, which is why the F15 quadrant flow shows no drift loading on Q1
(rho -0.05) even though dispersion moves at dz -0.98. (b) Cohen's d against
the passage SD divides by a denominator that is mostly measurement error, so
the surprisal:drift ratio is 4.4x on raw/total-SD, 2.1x on raw/between-SD,
and 1.3x on dz. **The apparent smallness of the drift effect is largely an
artefact of how noisily it is measured.** (c) Reliability is a property of
the unit you classify: Spearman-Brown gives 0.082 per passage, 0.211 per
3-sample cell, 0.988 per pair. Classify pairs, not passages.

## 3. DIRECTEDNESS IS SENTENCE COUNT

    directedness = total_drift / path_length
                 = diameter / (sum of successive steps)

Intended as a shape measure: how far the passage's extremes end up relative
to how far it travelled. But `path_length` grows with every sentence while
the diameter saturates, so the ratio falls as 1/n by construction. Measured
over all 76,214 rows of the F15/F16 artifact:

    Spearman(directedness, n_sentences)   -0.923
    fit directedness = 1.681 / n          R^2 = 0.795
    residual SD after removing 1.681/n    0.057  (raw SD 0.129)

    n_sents     3      4      5      6      7      8      9
    observed  0.611  0.442  0.345  0.282  0.239  0.209  0.185
    1.681/n   0.560  0.420  0.336  0.280  0.240  0.210  0.187

**Within a fixed sentence count every corpus converges** -- abstracts,
dreams, waking narratives, base AI and aligned AI all sit at 0.43-0.45 at
n=4 and 0.20-0.21 at n=8. The entire apparent ordering (abstracts most
"directed", diary entries most "wandering") is that abstracts have 3.7
sentences after truncation and diary entries have 6.1.

**This seat built a theoretical reading on it before checking** -- path shape
as secondary revision, and "alignment does not change directedness" -- and
both are void. The second is trivially true: base and aligned have identical
sentence counts (5.3 and 5.3).

## 4. The 75-word truncation analyses 46% of each generation

Generations run to a 256-token cap, median 183 words. The
fewest-sentences-exceeding-75-words rule leaves a median of 84 words and 5
sentences. Measured on 3,000 passages:

    floor        passages kept   median sents   pairwise sims
    75 words          86.1%            5             10
    100               84.8%            7             21
    150               78.7%           10             45
    none              95.1%           11             55

**No truncation keeps MORE passages** -- the word floor discards short
generations that have three or more sentences anyway -- and gives 5.5x the
pairwise comparisons. The rule is a CROSS-CORPUS length normalisation
inherited from F15/F16, where dreams, abstracts and fiction have wildly
different natural lengths. In a within-corpus arm contrast, where one token
cap already governs length, it normalises nothing and costs both data and
resolution. So defect 2 is substantially self-inflicted: five sentences were
analysed where eleven were available.

## What to use instead

**`mean_drift` over `total_drift`, on every property**: a mean rather than an
extreme (ICC 0.141 against 0.082), order-DEPENDENT rather than invariant, and
a larger effect on every standardisation (raw -0.051 against -0.023,
between-SD -1.13 against -0.57, dz -1.53 against -0.98).

**For path shape, the WITHIN-PASSAGE ORDERING CONTRAST, not directedness.**
Under a random ordering of a passage's own sentences the expected successive
distance is exactly the mean of all pairwise distances, so

    ordering = mean(successive distances) - mean(all pairwise distances)

is a pure sequence measure with composition AND sentence count held fixed by
construction -- the same sentences, the same n, only the order differs. It is
one extra scalar at encode time (`mean_pairwise`) and is wired into
`m06_crosslingual_drift.py`.

**And classify at the pair level, or not at all.** The quadrant framework
classified passages, which is where the drift axis is 92% noise; a per-pair
displacement in (drift, surprisal) needs no thresholds, is not
cohort-relative, and uses the continuous values.

## Fences

- Defects 1-3 are measured on the F15/F16 artifact (`corpus_metrics.parquet`,
  76,214 rows) and defect 2 on M06's own passage corpus; RH notes both
  `corpus_metrics.*` and the repo-level `findings/` are legacy, so this
  document records the instrument facts for M06's use rather than as a
  correction to those.
- The 1/n fit is descriptive: R^2 0.795 leaves real residual variance, so
  directedness is not ONLY sentence count -- it is dominated by it to the
  point where no cross-corpus comparison at differing n is interpretable.
- Exploratory throughout. These were computed answering questions in
  session, with no directions declared in advance.

## Addendum (2026-08-14): can directedness be normalised? No, and the reason is structural

RH asked whether directedness is merely too collinear with sentence count and
could be rescued by normalising. It can be made n-free OR shape-relevant, not
both, because `directedness = diameter / path_length` mixes an EXTENT measure
with a PATH measure and every normalisation either keeps the diameter or
cancels it.

Two identities, verified on the untruncated cross-lingual cells:

    path_length == (n_sents - 1) * mean_drift          corr 1.000000
    mean_pairwise / mean_drift == 1 - ordering/mean_drift   max diff 4.4e-16

**Route 1, divide out n.** `directedness * n_sents` does decorrelate:
rho(n_sents, .) goes -0.961 to -0.071. But it is then approximately
`total_drift / mean_drift`, so the diameter is still in the numerator with its
92% passage-level noise -- and in the arms contrast it FIRES HARD, +0.0775 zh
and +0.0742 en, 24 of 25 pairs each, p 1.6e-6. That is not a shape result.
Both extents fall under alignment and `mean_drift` falls about 2.4x faster:

    zh  total_drift -3.7%   mean_drift -9.9%   ratio +6.9%
    en  total_drift -4.3%   mean_drift -9.5%   ratio +5.7%

The ratio rises because of the differential decline of two quantities already
measured separately. **A reader given only the 24/25 and the p-value would
report that alignment makes passages more directed.** It does not; it makes them
smaller faster than it makes them narrower.

**Route 2, normalise against the reshuffle null.** The diameter is
order-invariant, so under a random reordering of a passage's own sentences it is
unchanged while expected path length becomes `(n-1) * mean_pairwise`. Then

    directedness_observed / directedness_reshuffled = mean_pairwise / mean_drift

and `total_drift` cancels exactly. This is the ORDERING measure in ratio form,
identical to the difference form to 4.4e-16, and it is NULL in the arms contrast
(see `crosslingual_arms.md`: four nulls, two languages by two truncation
regimes, each beside a positive control).

**Raw directedness is null anyway** in the same contrast: zh +0.0099, en
+0.0047, 15 up / 10 down, p 0.424 both. Retiring it costs nothing.

The general form, worth keeping: a ratio of two quantities that both move is not
a shape measure just because the units cancel. Check whether the normalisation
removes the nuisance or removes the construct.
