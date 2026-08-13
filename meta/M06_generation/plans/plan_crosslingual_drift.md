---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades
date: 2026-08-13
role: plan
topics: [drift, cross-lingual, chinese, instrument]
description: "Plan: CROSS-LINGUAL DRIFT — bge-m3 trajectory drift on the f11_l2 corpus in BOTH languages, the half of the cross-lingual F15 question that needs no reference-model decision. Descriptive instrument run: no arm directions declared, because the point is to establish that the measure exists in Chinese at usable coverage before anything is predicted from it. CPU mandatory (mps-CJK family, [5780]/[5782]); the mps-vs-CPU divergence on SENTENCE-length Chinese is recorded as a diagnostic, since the two prior instances were single characters and byte sequences."
---
# Plan: cross-lingual drift on f11_l2

RH, 2026-08-13, in session: "Could we start measuring drift in Chinese with
bge?" This is the tractable half of a cross-lingual F15. Drift needs only an
embedder, and bge-m3 is cross-lingual BY CONSTRUCTION (a multilingual
retrieval model: Chinese and English sentences occupy one space). Surprisal
needs a reference-model decision that is NOT made here -- BLT is the candidate
([5780]: usable on Chinese, CPU only) and its bits-per-byte levels are not
cross-language comparable, so that half waits.

## Why f11_l2 and not the passage corpus

The passage corpus is English-only (checked: zero CJK prompts in `passage`,
`passage_run2`, `beam_fc`, `y`). `f11_l2` carries BOTH languages -- 97 Chinese
and 100 English prompts over the SAME 58 models, 1,940 rows per model -- so
the language contrast can be run WITHIN one corpus, one rung, one model set,
with language the only thing that varies. That is a better design than
comparing a Chinese run here against the English passage-corpus run, and it is
the reason this plan exists at all.

Note the rung: f11_l2 is M02's contradiction corpus. Prior art from F15 and
from `f15_on_passages.md` is on a different corpus at a different rung and is
NEVER a prior here.

## Population and unit

`corpus='f11_l2'`, split by whether the PROMPT contains CJK (the prompt, not
the text -- a generation may code-switch and that is data, not a label).
All 58 models, both languages. Cap 3 passages per (model, prompt), seed
20260813 over sorted keys; ~20 exist per cell, so the cap is a subsample of
samples and NOT of cells.

Floors, declared with their arithmetic:

- SENTENCES: >= 3, as in the F15 producer. Chinese split on `。！？；!?`
  with the delimiter retained; English keeps the NLTK splitter. The Chinese
  rule is simpler than English's because Chinese has no abbreviation
  ambiguity.
- LENGTH: >= 120 CHARACTERS for Chinese, declared as the equivalent of the
  English producer's 75-WORD floor at roughly 0.6 English words per Chinese
  character. Measured before this plan was written: 72.2% of Chinese
  passages clear both floors (median 334 chars, median 5 sentences), so the
  instrument has usable coverage rather than a floor that eats the corpus.
  Sensitivity at 100 and 150 characters is reported beside the headline.
  English keeps the 75-word rule so the two arms are each on their own
  language's version of the same criterion, which is the honest form: a
  word count has no referent in Chinese and a character count flatters it.

Passage -> (model, prompt) cell -> model. Pairing to base/aligned arms
happens at ANALYSIS time from the roster, not in this producer.

## Device

CPU, mandatory, for both languages. The mps-CJK family now has three
instances on this machine (bge-m3 single characters twice; BLT Chinese byte
sequences, [5780]) and CPU is the referee. Because a CPU run cannot gate
itself against a second device, the producer instead records a DIAGNOSTIC:
60 Chinese sentences encoded on both devices, cosine reported per row and the
minimum printed. This is not a gate on this run's output -- it is the first
measurement of whether the mps hazard reaches SENTENCE-length Chinese, which
neither prior instance tested.

## What is declared, and what is not

DECLARED: the population, the floors and their arithmetic, the splitter, the
device, the unit, and that per-passage values are persisted keyed to
(model, prompt, sample_idx) so any later contrast joins to them.

NOT DECLARED, deliberately: any arm direction. This run establishes that the
measure exists in Chinese at usable coverage and produces the values. A
base-vs-aligned contrast on these values is a separate plan with its own
declared directions, written BEFORE anyone looks at the arm split. **The
producer therefore prints NO arm contrast**, only coverage, the diagnostic,
and the per-language distributions.

Producer: `scripts/m06_crosslingual_drift.py`. Results:
`results/crosslingual_drift.json` + `crosslingual_drift_cells.parquet`.
