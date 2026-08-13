---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/crosslingual_arms.json
date: 2026-08-13
role: finding
topics: [drift, cross-lingual, chinese, arms]
description: "The cross-lingual arm contrast: ALIGNMENT REDUCES TRAJECTORY DRIFT IN CHINESE AS IT DOES IN ENGLISH, on the same 25 pairs, the same corpus and the same rung, with language the only manipulation. zh -0.0314 (4/21 of 25 pairs, p 9e-04), en -0.0205 (1/24, p 1.6e-06) on total_drift; both stronger on mean_drift; all four cells survive n_sents matching. The language DiD is NULL on every construction (p 0.23 to 1), so the effect is a property of the operation rather than of the language it was tuned on. First arm effect this campaign has measured on Chinese GENERATED TEXT."
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
the four constructions -- it does not even hold a sign. **Alignment reduces
drift by the same amount in a language it was mostly not tuned on**, which
makes the effect a property of the operation rather than of English.

## What it is worth, and the limits that bound it

This is the design the Chinese work has never had: same models, same corpus,
same rung, language as the only manipulation. It says the drift reduction is
not an artefact of English alignment data.

- **PROMPTS ARE NOT MATCHED ACROSS LANGUAGES.** 97 Chinese and 100 English
  prompts are different texts, so the language comparison is between prompt
  SETS. The DiD is partly protected -- a prompt-set difference would have to
  INTERACT with alignment to fake a null -- but a null DiD is exactly what a
  weak or noisy design also produces, and at 25 pairs this one is not
  powered to bound a small language difference. **The null is undetected,
  not zero.**
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

## A defect in this producer's own control, found and fixed before reporting

The first run's `n_sents`-matched variant returned **zero units** and printed
as `median nan, p 1` -- which reads exactly like a null result. The cause:
matching on exact sentence count WITHIN a (pair, prompt) cell is unrunnable
at ~2.3 passages per cell. Fixed by pooling the matched variant across
prompts within (pair, n_sents), and the producer now REFUSES with a named
reason rather than reporting an empty contrast. **An empty comparison that
prints a p-value is the most dangerous shape in this campaign's ledger**, and
it appeared in the control that the plan had designated as decisive.
