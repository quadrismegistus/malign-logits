---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades
date: 2026-08-13
role: plan
topics: [drift, cross-lingual, chinese, arms]
description: "Plan: THE CROSS-LINGUAL ARM CONTRAST — does alignment reduce trajectory drift in Chinese as it does in English, measured on the SAME 25 pairs, the SAME corpus and the SAME rung with language as the only manipulation? English is directional (two prior corpora); Chinese is open; the language difference-in-differences is the question the design exists for. Declared before any arm number is computed on the instrument built at a3fb226b."
---
# Plan: the cross-lingual arm contrast

The instrument exists (`crosslingual_drift_{zh,en}_cells.parquet`,
a3fb226b) and deliberately computed NO arm contrast, because the base/aligned
split deserves declared directions rather than being computed the moment the
values appear. This is that plan.

**What makes it worth doing:** `f11_l2` carries both languages over one model
set at one rung, so the alignment effect can be measured in Chinese and
English **within the same pairs**, with language as the only thing that
varies. Our Chinese work has never had that -- it has always foundered on
roster mismatch (the 34-model wall).

## Population, and it is smaller than the English-only work

25 base/aligned pairs complete in BOTH languages (24 dpo, 1 ppo), from
`data/base_aligned_pairs.json` with ambiguous pairs excluded. 12,803 Chinese
and 14,178 English passages; 97 Chinese and 100 English prompts.

**POWER, stated before the run:** a sign test over 25 pairs needs **18 of 25**
in one direction for p < 0.05. That is a real constraint and the reason no
null here will be a bound.

## Unit and metric

Cell = (pair, role, lang, prompt) mean over its passages. Paired per
(pair, prompt) within language; pair median; sign test over the 25 pairs as
the conservative unit, cell grain reported beside it.

`total_drift` (1 - min pairwise sentence similarity) is primary, matching
F15's headline; `mean_drift` (mean successive step) is the sensitivity. Both
reported.

## Directions

  P1 (DIRECTIONAL, and it is a replication rather than a discovery):
     English drift falls under alignment on `f11_l2`. Grounds: the same
     direction on TWO prior corpora -- F15's battery and
     `f15_on_passages` (34/38 pairs, p 6e-07, embedder-independent). A
     third corpus at a different rung is the test of generality, not of
     existence.
  Q1 (OPEN, no direction): whether it holds in Chinese. Nothing in this
     campaign has measured a Chinese arm effect on generated text.
  Q2 (OPEN, and the reason this plan exists): the LANGUAGE DiD --
     (English reduction) minus (Chinese reduction), paired within pair.
     Larger in English would say the effect is partly a property of the
     language alignment was tuned on; equal would say it is a property of
     the operation.

## The confound that could produce Q2 on its own

**`total_drift` IS THE DIAMETER OF A SENTENCE SET, SO IT GROWS WITH THE
NUMBER OF SENTENCES.** More sentences means more pairs and a greater chance
of a distant one. If alignment changes sentence count differently in the two
languages, the DiD inherits that mechanically and has nothing to do with
trajectory.

Therefore, **before any contrast is printed**: `n_sents` by (language, role),
and per-arm cell survival by (language, role). And the contrast is reported
BOTH pooled and **matched on `n_sents`** (compare within sentence-count bins,
requiring both arms present in a bin). If the two disagree, the matched one
travels and the pooled one is named as confounded.

## Further fences, declared now

- **PROMPTS ARE NOT MATCHED ACROSS LANGUAGES.** 97 Chinese and 100 English
  prompts are different texts, so the language contrast is between prompt
  SETS and any topic or difficulty difference is confounded with language.
  The DiD is partly protected -- a prompt-set difference would have to
  INTERACT with alignment to fake it -- and that protection is partial, not
  total, and is not a substitute for a translated battery.
- Absolute drift differs by language (zh 0.638, en 0.604 median) and those
  levels are NOT compared; the DiD compares differences within language, so
  the level cancels by construction.
- bge-m3 places both languages in one space by design, which is what makes
  the metric the same operation in both; that is a property of the embedder
  and is asserted, not measured here.
- One embedder only. The English side of `f15_on_passages` had two and
  agreed; this run does not, and a bge-only result is what it is.

Producer: `scripts/m06_crosslingual_arms.py`. Results:
`results/crosslingual_arms.json` + per-pair parquet.
