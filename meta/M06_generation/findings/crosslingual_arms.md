---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/crosslingual_arms.json
date: 2026-08-13
role: finding
topics: [drift, cross-lingual, chinese, arms]
description: "The cross-lingual arm contrast: ALIGNMENT NARROWS THE SEMANTIC SPREAD OF A PASSAGE IN CHINESE AS IT DOES IN ENGLISH (the metric is ORDER-INVARIANT and is NOT a trajectory claim -- corrected below), on the same 25 pairs, the same corpus and the same rung, with language the only manipulation. **THE MATCHED-PROMPT LEGS ARE WITHDRAWN AS UNREPRODUCED (2026-08-14, [5932]): no artifact, no producer, and 32 declared recipes over the frozen per-passage cells return at most 2 of 6 booked values -- prompt content is NOT held constant in any surviving construction.** zh -0.0314 (4/21 of 25 pairs, p 9e-04), en -0.0205 (1/24, p 1.6e-06) on total_drift; both stronger on mean_drift; all four cells survive n_sents matching. The language DiD is NULL on all four PERSISTED contrasts, p 0.23 to 1.0, and holds no sign. The invariance rests on those alone -- both arms negative 8/8, DiD p>=0.23 8/8 across both inputs. First arm effect this campaign has measured on Chinese GENERATED TEXT."
---
# The cross-lingual arm contrast: the same operation in both languages

Plan: `plans/plan_crosslingual_arms.md`, committed before this producer with
P1 directional, Q1 and Q2 open, and the `n_sents` confound named in advance.
Producer `scripts/m06_crosslingual_arms.py`; results
`results/crosslingual_arms.json` + per-pair parquet. No new compute -- the
instrument was built at a3fb226b and deliberately withheld its arm split.
Single pass; [5503] applies.

## READ `findings/zh_fluency_and_ordering.md` BEFORE CITING THE CHINESE RESULT

Added 2026-08-14. **The numbers below are reproduced exactly** -- that document's
producer refuses to report until it recovers this finding's `total_drift` zh
contrast (21/25, median -0.0314) from the same parquets, and it does, to the
digit. **What is in question is what they measure.**

  - Chinese fluency is itself an ARM VARIABLE: alignment improves it, 20 of 25
    pairs, p=0.0041, judged blind at Cohen kappa 0.776.
  - That gap PREDICTS this finding's `total_drift` effect: spearman -0.497,
    p=0.0116, under the Bonferroni threshold for its family.
  - Restricting to the 6 pairs where BOTH members write Chinese collapses
    `total_drift` sevenfold (-0.0314 -> -0.0046) and leaves an ORDERING effect
    unmoved (-0.0090 -> -0.0087), which is independent of fluency.

So the sentence most at risk is **"First arm effect this campaign has measured
on Chinese GENERATED TEXT"** in the frontmatter above: 12 of these 25 pairs have
a base member that largely does not produce Chinese, and the effect does not
survive their removal. The separable ordering test this document names as unrun
HAS NOW BEEN RUN, with a ratio rather than the difference proposed below --
the difference is not scale-free and comes back null.

Nothing here is edited on that account; the disposition is RH's.

## Population

25 base/aligned pairs complete in BOTH languages (24 dpo, 1 ppo), 23,677 of
26,981 passages, `f11_l2`, 97 Chinese and 100 English prompts, bge-m3 drift.
A sign test over 25 pairs needs 18 of 25 for p<0.05, stated before the run.

## WHAT `total_drift` MEASURES, corrected 2026-08-13

**`total_drift` IS ORDER-INVARIANT AND THIS FINDING CALLED IT A TRAJECTORY.**
It is `1 - min(pairwise similarity)`: the DIAMETER of the passage's sentence
set. Shuffle the sentences and it does not move -- verified rather than read
off the code (three random 8-sentence passages, permuted: total_drift
identical to four decimals in every case, mean_drift changed in every case).

So the result below is about **SEMANTIC SPREAD**: aligned passages occupy a
NARROWER REGION of embedding space. **It is not a claim about how a passage
moves**, and every sentence in this document that implied otherwise is
wrong. `mean_drift` IS order-dependent but inherits the spread effect -- if
all sentences sit closer together, successive ones do too -- so it is a
mixture and cannot carry the trajectory reading either.

The numbers are unaffected. What changes is the noun.

**The separable version, not run:** within-passage permutation holds the
sentence set fixed BY CONSTRUCTION, and under a random order the expected
successive distance is just the mean of all pairwise distances. So
`mean successive - mean all-pairs` is a pure ORDERING measure with
composition removed exactly. That is the test of whether alignment shapes
the chain rather than the lexicon, and this instrument does not answer it.

## Coverage and the confound, printed before any contrast

    lang  role     passages   cells   n_sents   length
    zh    base        5,782    2,307     7.55    179.9
    zh    aligned     5,438    2,249     8.05    184.0
    en    base        6,173    2,455     5.25    497.2
    en    aligned     6,284    2,477     5.44    494.7

    aligned-minus-base n_sents:  zh +0.506 (6.7%)   en +0.181 (3.5%)

**The confound runs AGAINST the finding.** `total_drift` is a set diameter,
so it grows with sentence count -- and aligned passages have MORE sentences
in both languages. A mechanical artefact would therefore push aligned drift
UP. It goes down.

## The result

    (negative = alignment REDUCES drift; pair grain, 25 pairs)

    total_drift  POOLED           zh -0.0314  4/21  p 9.1e-04
                                  en -0.0205  1/24  p 1.6e-06
                 n_sents-MATCHED  zh -0.0295  1/24  p 1.6e-06
                                  en -0.0275  0/25  p 6.0e-08
    mean_drift   POOLED           zh -0.0467  1/24  p 1.6e-06
                                  en -0.0363  0/25  p 6.0e-08
                 n_sents-MATCHED  zh -0.0504  1/24  p 1.6e-06
                                  en -0.0407  0/25  p 6.0e-08

**P1 CONFIRMED**: English drift falls under alignment on a third corpus at a
third rung -- after F15's battery and `f15_on_passages`' passage corpus.

**Q1 ANSWERED, and it is the first arm effect this campaign has measured on
Chinese GENERATED TEXT**: Chinese drift falls too, on both metrics and under
matching, at 24 of 25 pairs on three of the four constructions.

**Q2: THE LANGUAGE DiD IS NULL EVERYWHERE.** English-minus-Chinese runs
-0.0023 (p 0.42), +0.0021 (p 1.0), +0.0133 (p 0.23), +0.0120 (p 0.23) across
the four constructions -- it does not even hold a sign.

## MATCHED PROMPTS: BOTH LEGS WITHDRAWN AS UNREPRODUCED, 2026-08-14

**EVERY NUMBER IN THIS SECTION IS WITHDRAWN.** Neither matched-prompt leg has
an artifact or a producer. @dario found it at [5932] while holding plot-debt
item 5, which named this section as the figure's basis and which he declined
to draw. Both blocks below were computed inline in a session that ended; what
is persisted, in both arms JSONs and both pairs parquets, is four contrasts --
`{total_drift, mean_drift} x {pooled, n_sents-matched}` -- and `matched` there
means N_SENTS-matched, not prompt-matched. `m06_crosslingual_arms.py`, the
producer this finding names, contains no reference to `pair_id`, `pair_role`
or the catalogue.

Reconstruction attempted, `scripts/m06_crosslingual_parsefree.py` ->
`results/crosslingual_parsefree_attempt.json`. **The population reproduces
exactly (23,677 passages, 25 pairs) from the per-passage
`crosslingual_drift_{lang}[_full]_cells.parquet` joined to the catalogue's
declared columns. The numbers do not.** 32 declared recipes -- 4 key grains x
inner{mean,median} x outer{mean,median}, against both the truncated and the
untruncated cells -- return at most **2 of the 6** booked parse-free values,
and the published key count of 71 is not reachable (nearest 72). The one
recurring hit, `mean_drift` zh 1/24, also appears in four persisted contrasts,
so it discriminates nothing. The sweep is bounded on purpose: searching
estimator space until something matches fits a recipe to the target and would
be indistinguishable from a reproduction while carrying none of its value.

**WHAT SURVIVES, AND WHAT DOES NOT.** The invariance conclusion stands on the
persisted contrasts alone: across both inputs, both arms are negative in 8/8
arm tests and the DiD is p>=0.23 in 8/8. What does not stand is this section's
status as **"the strongest construction"** and its ruling that **"the
parse-free key is the one that travels"** -- a preference between matching
keys, asserted on numbers nothing carries. The three key counts this finding
quotes (97 in the frontmatter, 100 in the block below, 71 in the parse-free
block) are also mutually inconsistent and none of them reproduces.

The blocks are left standing, struck rather than deleted, so the record shows
what was claimed.

RH corrected a fence this finding had wrong (see the limits): the two prompt
sets are not different texts. Keyed on `pair_id` base plus role suffix
(`f11_love_he` x `CONTROL_B`) there are **97 matched keys, ZERO
Chinese-only, 8 English extras** -- every Chinese prompt has an English
partner. So the contrast can be run with topic, construct and role held BY
CONSTRUCTION.

Declared before it existed (81a86eed): removing prompt-content variance
gives the test MORE power, so a language difference appearing only now would
mean the pooled null was a power failure and the invariance reading had to be
withdrawn. Run after (30e4341d), 22,461 passages over 100 matched keys,
1,901 (pair, key) units:

    total_drift  zh -0.0314 (4/21, p 9.1e-04) | en -0.0169 (2/23, p 1.9e-05)
                 DiD en-zh  -0.0041  11/14  p 0.69
    mean_drift   zh -0.0462 (1/24, p 1.6e-06) | en -0.0427 (0/25, p 6.0e-08)
                 DiD en-zh  +0.0141  14/11  p 0.69

**KEY ROBUSTNESS, added after malign's [5855] showed the parse trap.** My
matched key was `(pair_id base, prompt_id suffix)` -- a POSITIONAL PARSE, and
14 of 216 catalogue rows have a `prompt_id` that does not start with its
`pair_id` (the `setf_` family), so my key returned None for them and **they
were silently dropped**. `prompt_catalogue` carries a declared `pair_role`
column that needs no parsing, and the two languages use two incompatible
`prompt_id` conventions, so `prompt_id` is not a cross-language join key at
all. Rerun on the parse-free key `(pair_id base, pair_role)` -- 23,677
passages, 71 keys, every passage in the paired population retained:

    total_drift  zh -0.0263 (5/20, p 4.1e-03) | en -0.0171 (3/22, p 1.6e-04)
                 DiD en-zh  -0.0020  11/14  p 0.69
    mean_drift   zh -0.0454 (1/24, p 1.6e-06) | en -0.0409 (0/25, p 6.0e-08)
                 DiD en-zh  +0.0042  13/12  p 1.0

~~Same conclusion on both keys; the DiD stays null and holds no sign either
way. **The parse-free key is the one that travels**, since it drops nothing
and rests on declared columns.~~ WITHDRAWN, see the head of this section: no
producer, and the eight numbers do not come back from the frozen cells.

~~**THE INVARIANCE HOLDS ON MATCHED CONTENT.** Both arms' reductions survive,
the DiD is null on both metrics and holds no sign. **Alignment reduces
trajectory drift by the same amount in Chinese and English ON THE SAME
PROMPTS**, which makes the effect a property of the operation rather than of
English.~~ WITHDRAWN AS STATED. The conclusion is still supported, but by the
persisted pooled and n_sents-matched contrasts, NOT by matched prompts -- so
"on the same prompts" is exactly the clause that has to come out. Prompt
content is not held constant in any surviving construction.

## What it is worth, and the limits that bound it

This is the design the Chinese work has never had: same models, same corpus,
same rung, same prompts, language as the only manipulation. It says the drift
reduction is not an artefact of English alignment data.

- **THE PROMPTS ARE MATCHED ACROSS LANGUAGES. This fence was WRONG and RH
  corrected it.** The original text said the two prompt sets are different
  texts and the language comparison is between SETS. Measured on the
  catalogue: every Chinese prompt has an English counterpart, keyed on
  `pair_id` base plus role suffix (`f11_love_he` x `CONTROL_B`) -- **97
  matched keys, ZERO Chinese-only, 8 English extras.** So the design is
  stronger than the finding claimed: topic, construct and role are held by
  construction and only language varies.

  The matched-prompt contrast this unlocked is reported above as the
  strongest construction. **How the error happened, since it is the
  reusable part: I saw 97 prompts against 100, inferred different texts, and
  never looked for a key.** The catalogue carries one, in the field whose
  name says so. **A COUNT MISMATCH IS NOT EVIDENCE OF NON-CORRESPONDENCE** --
  97 and 100 differ by exactly the English extras, which is what a matched
  design with a few spares looks like.

  Remaining, and unchanged by any of this: at 25 pairs a null still bounds
  nothing. **The null is undetected, not zero.**

  **AND THAT LIMIT IS NOW VISIBLE RATHER THAN STIPULATED, 2026-08-14.**
  @dario's item 5 panel (`figures/crosslingual_invariance.png`, [5938]) puts
  English on x and Chinese on y with y = x drawn as the place "the same in
  both languages" lives. In both `mean_drift` panels the cloud sits
  consistently BELOW the diagonal -- Chinese narrows a little more --
  which is the DiD of +0.0133 and +0.0120, 16/9, p 0.23. Null, and not
  nothing. **This is not a new result and not in tension with the finding:
  it is exactly what this fence says cannot be excluded, drawn.** A reader
  of "the DiD is null" does not picture a cloud sitting a consistent
  distance off the diagonal in two of four panels, and now does not have to.
  The truncated run leans this way on both metrics; the untruncated run does
  not. Nothing is claimed from it and no test was run for it.
- ONE EMBEDDER. `f15_on_passages` had two and they agreed; this has bge-m3
  alone.
- 25 pairs, 24 of them dpo, so this is a DPO result with one ppo rider.
- Absolute drift levels differ by language (zh 0.638, en 0.604) and are NOT
  compared; only within-language differences are, so the level cancels.
- **CUSTODY FACT ABOUT `f11_l2`, verified at two seats ([5844], and
  independently here): its `pair` and `role` columns are EMPTY.** All 228,520
  rows carry the blank string in both -- one distinct value each, against
  `passage`'s 42 and 2. **Anyone joining this corpus on `pair` or `role` gets
  one giant group and no error.** Pairing must come from the model names
  through the registry, which is what this producer does.
- **THE POPULATION DEPENDS ON WHICH PAIRING SOURCE YOU USE, and the two
  disagree.** This plan declared `data/base_aligned_pairs.json`: 26 pairs
  with rows in both languages, 25 after the floors. The single loss is
  `bloom-7b1>bloomz-7b1`, whose aligned arm has no surviving Chinese
  passages -- bloom being the model that supplied 155 of 195 empty texts in
  the passage corpus, so a known-degenerate generator rather than a harsh
  floor.

  **THE 26-vs-29 GAP IS THE `ambiguous` FLAG, MEASURED after four wrong
  attributions between two seats ([5848], confirmed [5849]).** Not the
  pairing source, not the floors, not the coverage predicate -- all three
  were real differences between the lists and none was this one. The three
  are `pythia-2.8b>archangel_sft-dpo_pythia2-8b`,
  `Olmo-3-1025-7B>Olmo-3-7B-Instruct-DPO` and
  `Llama-3.1-8B>Llama-3.1-8B-Instruct`.

  **AND MY FILTER WAS OVER-STRICT, which is my error rather than the
  file's.** All three carry `ambiguous: true` AND `ruled: true` with an
  explicit `candidates` list: the flag means A CURATOR FACED A CHOICE AND
  MADE IT, not that the pair is unusable. `Llama-3.1-8B>Llama-3.1-8B-Instruct`
  is excluded by my filter although its "ambiguity" was only which of seven
  Tulu variants to prefer, resolved in favour of meta-llama's own Instruct.
  The defensible filter is `ambiguous AND NOT ruled`, which here excludes
  nothing, so **the correct population is 28 rather than 25.**

  **DECLARED BEFORE RUNNING (this paragraph is committed before the
  sensitivity exists):** the three pairs are added as a SENSITIVITY, never
  as a replacement, because the headline was computed on 25 and moving the
  population after seeing a result is the post-hoc move this finding already
  refused once. Direction stated in advance: at 24/25 and 0/25 the result
  should be **unchanged**; if it moves materially that is evidence of
  fragility and is the more important outcome.

  **RESULT of that sensitivity (run after the paragraph above was
  committed at 8e88ffc0): unchanged, and every cell strengthens on
  count.** 28 pairs: total_drift zh -0.0242 (4/24, p 1.8e-04) and en
  -0.0194 (1/27, p 2.2e-07); mean_drift zh -0.0456 (2/26, p 3.0e-06) and
  en -0.0339 (0/28, p 7.5e-09). Medians move by hundredths in both
  directions, the sign counts improve with the larger n, and nothing about
  the reading changes. **The 25-pair headline stands and is not fragile to
  the population question that produced four wrong attributions.**

## Truncation

- **EVERY DRIFT NUMBER HERE IS COMPUTED ON ROUGHLY THE FIRST HALF OF EACH
  PASSAGE.** Generations run to a 256-token cap (median 183 words); the
  fewest-sentences-exceeding-75-words rule leaves a median of 84 words and 5
  sentences, so 46% of the generated words are analysed. Measured on 3,000
  passages, dropping truncation gives 11 sentences and 55 pairwise
  similarities instead of 5 and 10, and RETAINS MORE passages (95.1% against
  86.1%). The rule comes from F15/F16, where it normalises length across
  corpora of wildly different natural lengths; applied to a within-corpus arm
  contrast where one token cap already governs length it normalises nothing.
  An untruncated variant is running beside this one (`--no-truncate`,
  `*_full` outputs).

## A defect in this producer's own control, found and fixed before reporting

The first run's `n_sents`-matched variant returned **zero units** and printed
as `median nan, p 1` -- which reads exactly like a null result. The cause:
matching on exact sentence count WITHIN a (pair, prompt) cell is unrunnable
at ~2.3 passages per cell. Fixed by pooling the matched variant across
prompts within (pair, n_sents), and the producer now REFUSES with a named
reason rather than reporting an empty contrast. **An empty comparison that
prints a p-value is the most dangerous shape in this campaign's ledger**, and
it appeared in the control that the plan had designated as decisive.

## Untruncated: the effect survives, and the 75-word floor was suppressing English (2026-08-14)

`m06_crosslingual_arms.py --full` on the no-truncation cells. MEAN_DRIFT,
aligned minus base, 25 pairs usable in both languages:

                    truncated            full
    zh pooled    -0.0467  1/24      -0.0462  1/24
    en pooled    -0.0363  0/25      -0.0453  0/25
    DiD          +0.0133  p 0.23    +0.0051  p 1
    zh n-matched -0.0504  1/24      -0.0479  1/24
    en n-matched -0.0407  0/25      -0.0458  0/25
    DiD          +0.0120  p 0.23    -0.0003  p 1

**Chinese unchanged, English GROWS, the residual DiD vanishes.** The sentence
counts predict it: truncation cut English from 10 sentences to 5 while Chinese
went 7 to 6, so the floor was suppressing the English effect specifically. The
hint of a language asymmetry in the truncated data was AN ARTIFACT OF THE FLOOR,
not a property of the languages, and the invariance claim is strengthened.

**Why matching is not optional here.** `total_drift` tracks sentence count at
rho +0.33 to +0.42 across all four datasets, and `directedness` is 1/n at
-0.945 to -0.973 -- the audit's English finding replicating cross-lingually and
getting STRONGER untruncated. Consequence: the raw language ordering on
`total_drift` REVERSES between regimes (zh 0.638 > en 0.604 truncated; en 0.668
> zh 0.635 full) on sentence count alone, with no drift fact behind it. Language
medians on `total_drift` are not interpretable across regimes. `mean_drift` is
near-independent (-0.06 to -0.15) and gives the larger, cleaner effect.

Also: the n_sents-matched variant now returns units (249 zh / 375 en full) where
it once returned zero and printed `p 1`, which this seat had misread as a null
rather than an empty comparison.

## Path shape is UNCHANGED while extent shrinks (2026-08-14)

`m06_crosslingual_ordering.py`. The audit retired `directedness` and specified
`ordering = mean(successive distances) - mean(all pairwise distances)`: under a
random reshuffle of a passage's OWN sentences the expected successive distance
IS the mean pairwise, so composition and sentence count are fixed by
construction and only order varies.

                     ORDERING                  MEAN_DRIFT (same pairs, prompts)
    zh trunc   -0.00334   9/16  p 0.23     -0.04500  1/24  p 1.6e-6
    en trunc   -0.00030  12/13  p 1        -0.03460  0/25  p 6.0e-8
    zh full    -0.00373   9/16  p 0.23     -0.04245  1/24  p 1.6e-6
    en full    -0.00204  11/14  p 0.69     -0.04485  0/25  p 6.0e-8

**FOUR independent nulls -- two languages by two truncation regimes -- each
beside a positive control from the same pairs, the same matched prompts and the
same estimator.** That pairing is what separates a null from an instrument that
cannot fire, and this series has been caught by that distinction before.

**Aligned passages cover LESS SEMANTIC GROUND without changing how the text
moves through it.** Both languages are locally coherent descriptively (zh
-0.0266, en -0.0339 untruncated): adjacent sentences sit closer than a reshuffle
would place them, which is what any real text should do.

Third instrument, different unit, same conclusion as the composition/level split
at word grain and the propagation slope: alignment acts on what is selected, not
on how the chain is put together.

**FENCE, and it matters.** `ordering` is ONE sequence property, the degree to
which adjacency beats a reshuffle at SENTENCE grain. Sentence length and clause
packing are combination facts it cannot see, and `AB_surface_and_clauses.md`
shows those DO move (more dependent clauses per 1,000 words, shorter, with
per-sentence ratios flat). "Alignment leaves combination alone" is too strong;
what is measured is that these three instruments, at their own grains, find no
change in how the chain coheres.

## The sign-test p-values quoted here are FLOORS (noted 2026-08-14)

Prompted by dario's [5897], which found 33 of 34 per-cluster z values in
Findings N to be the same float because `_ppf` saturates. Same shape here, by a
different mechanism: for a two-sided sign test with every pair agreeing,
`p = 2 / 2^n` exactly, so a unanimous result reports the SMALLEST VALUE THE TEST
CAN PRODUCE at that n.

    n=25 floor 5.96e-08    n=33 floor 2.33e-10
    n=35 floor 5.82e-11    n=36 floor 2.91e-11

Four headline values across this campaign's M06 documents sit exactly on it:
mean_drift en 0/25 (5.96e-08), net_fall 33/33 (2.33e-10), net_fall 36/36
(2.91e-11), and the common-support contrast aligned 35/35 (5.82e-11).

**Nothing is wrong and nothing changes direction.** But these p-values carry no
information beyond "every pair agreed", a result at the floor cannot get more
significant with a LARGER EFFECT (only with more pairs), and comparing p across
these results compares n rather than evidence. Quote the sign counts, which say
the same thing without implying a precision the test does not have.

## Aggregation, named (2026-08-14)

Prompted by @dario's [5926] generalisation: any per-passage quantity defined as
a MIN or MAX over an internal set wants a median at the pair grain, because its
distribution is skewed. `total_drift` is `1 - min(pairwise cosine)` and is one
of those.

`m06_crosslingual_arms.py` aggregates in TWO stages: the **mean** over the 3
samples within (pair, prompt, role), then the **median** over prompts within
pair. The pair grain is the median, per convention. The inner stage is a mean
over three draws of an extreme statistic, which is the inconsistency dario's
rule points at.

**Checked, and it does not matter here.** Recomputing with the median at both
stages:

    total_drift  inner=mean    zh -0.02918 3u/22d   en -0.02715 2u/23d
    total_drift  inner=median  zh -0.02885 3u/22d   en -0.02587 2u/23d
    mean_drift   inner=mean    zh -0.04620 1u/24d   en -0.04533 0u/25d
    mean_drift   inner=median  zh -0.04245 1u/24d   en -0.04485 0u/25d

**Sign counts and p-values are IDENTICAL in all four**; the medians move 1-8%.
The published values are the inner-mean form and stay as they are, because
changing a published number for a difference that alters no verdict is worse
than naming the choice. What was missing was the name, not the choice: a bare
effect size without its aggregation is not reproducible, which is precisely what
cost a reconciliation at [5922]/[5924].
