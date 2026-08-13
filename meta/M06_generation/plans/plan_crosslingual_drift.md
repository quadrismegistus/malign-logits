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

Floors, declared with their arithmetic. **AMENDED before the producer ran,
on RH's question "can we do it in a principled way like we do with nltk" --
the first version of this section used a hand-written regex and a declared
character conversion, and BOTH were replaced after measurement.**

- SENTENCES: >= 3, as in the F15 producer. Chinese uses **stanza's
  `zh-hans` tokenize processor** (treebank-trained, joint segmentation and
  sentence splitting); English keeps NLTK punkt. Both are trained
  statistical tokenizers, so the two arms use the same CLASS of instrument
  rather than one trained model and one hand rule.

  **Why not the regex, measured rather than assumed**: a `。！？；!?`
  splitter agrees with stanza at mean Jaccard **0.639** on sentence sets and
  **0.500** on sentence counts over 60 passages. That is not adequate, and
  the mechanism is legible: these generations are only **68% CJK by
  character** (median), so they carry substantial embedded English, and the
  regex never split an English period. A rule tuned on the assumption of
  monolingual Chinese fails on a corpus that is a third English. Stanza runs
  at 0.01 s/passage, so there is no cost argument for the cheap instrument
  either.

- LENGTH: **>= 75 WORDS in both languages**, with Chinese words from
  **jieba** (the segmenter, which is what jieba is for -- it has no sentence
  tokenizer). This replaces the earlier "120 characters" rule and its
  declared 0.6-words-per-character conversion. The same NUMBER under each
  language's own segmenter is a stronger criterion than a converted one: no
  ratio is asserted, and the arithmetic that would have to be defended
  (jieba gives ~0.66 words per character, so 75 words is ~114 characters)
  is reported rather than relied on. Coverage under the new floors is
  printed by the producer before any embedding.

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
