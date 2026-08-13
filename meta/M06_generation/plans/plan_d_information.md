---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [information, drift]
description: "Plan D: the existing instruments on the new corpus — BLT bits/char, Pythia-1B reference surprisal, bge-m3 drift. Descriptive letter, no registered directions; F19's sub-Shannon result is prior art on a different corpus and rung, never a prior."
---
# Plan D — the information instruments: entropy, surprisal, drift

Drafted 2026-08-13 by the registrar on RH's word. DESCRIPTIVE: no
registered directions. The one tempting inheritance — F19's aligned
sub-Shannon bits/char — was measured on UNCONDITIONAL (BOS) generation, a
different corpus at a different rung, and the module boundary says prior
art is never a prior. If the direction replicates here it is a
cross-corpus fact worth having; nothing is predicted.

## Instruments (all pre-existing; producers to be pointed, not built)

- `bits_per_char_blt` — BLT bits/char (`scripts/blt_combined.py` family;
  `blt_human_corpora.py` shows it already ran on the human/* stash
  entries, which plan E will want on the same scale).
- `surprisal_pythia1b_mean` — reference surprisal, cache method
  `ref_surprisal` (keyed ref/prompt/text).
- `drift_bge_m3` — embedding drift along the passage, cache method
  `sent_embeddings` (keyed embedder/prompt/text).

DEVICE ALERT inherited from [5699] before any D producer exists: bge-m3
on this machine's mps deterministically corrupts short-string embeddings
(measured on single CJK characters; 89% of that class, invisible to
re-audits down the same path). D's drift instrument therefore runs its
gate ON ITS OWN ROWS (never probe strings), encodes short strings on CPU
or behind a second-device gate, and inherits lacan's producer-level
device policy verbatim.

Extended per [5701] (malign's own-rows audit of sent_embeddings: clean —
0/60 below cos 0.99 — but BY ACCIDENT of its input distribution, not by
design): (1) the encode-unit lesson binds here — MEASURE THE LENGTH OF
WHAT WAS PASSED TO encode(), never the row (the producer encodes
SENTENCES, not passages; a row profile hides the at-risk unit); (2) this
plan's producer decision, as owner: the GUARD form is adopted, not the
pin — `embedding.py` remains shared and untouched; D's producer wraps it
with (a) CPU for any encode-unit under 12 characters, (b) mps otherwise
behind a second-device gate on its own rows that REFUSES TO WRITE on
mismatch — lacan's measured 46.6/sec CPU trade makes short-unit CPU
free at D's scale. Malign's boundary carried: the 60-row audit licenses
"the found class is absent", never "the store is verified".

Extended per [5780] (lacan's BLT measurement, third mps-CJK instance and
the first in a causal LM's logits, so the family generalises past
embedders): BLT (itazap/blt-1b-hf) on this machine's mps both
catastrophically fails Chinese passages (437/437 byte-units non-finite on
one of four) AND shifts survivors 1-2%; CPU returns zero non-finite units
on the same rows. Two rules adopted into this plan as owner: (1) ANY
cross-lingual `bits_per_char_blt` run computes on CPU — mps is not
trustworthy for CJK byte sequences on this box, CPU is the referee; an
English-only run does not inherit this (measured unaffected). (2) The
producer COUNTS NON-FINITE UNITS PER PASSAGE and reports them beside the
number — never a bare nanmean, which would return a plausible language
figure from the surviving passages and hide the failed one entirely (the
silent form; np.mean's whole-language NaN is the visible form and the
better default). Existence proof at n=4 passages; the rate is unmeasured
and the rule does not need it.

POS caution from [5632] stands: nothing here reaches for `fields._byu()`;
any POS need is served by the shared Stanza parses.

## Unit, strata

As plan A amended: undisturbed arm, (pair, prompt) cell, hardened stratum
primary, pair medians, per-pair denominators, per-arm exclusion rates.

## Sequencing

Pilot DEFERRED until the plan A/B full-run shards release the machine —
these instruments are model inference, not lexicon lookups. Pilot on the
pilot population first; the anti-conflation fence ([5670]) binds here
too: no drift or entropy number is read against lexical-diversity or
concentration results without a declared bridge.
