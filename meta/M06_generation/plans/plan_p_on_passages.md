---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [arm-signature, lexical, replication, amplification]
description: "Plan: P ON PASSAGES — the arm signature read from generated text on the passage corpus, and its replication against the next-word-probability instruments. Same prompts, same models as the logit grain, so the comparison isolates GRAIN: distribution vs sampled behaviour. Deliverables: a generation-side arm classifier (plateau form, grid declared), a generation-side per-word AUC vector, the cross-grain replication correlation, and the amplification map. Drafted by the lacan seat on RH's word; smoke test first, eyeball grade, nothing quoted."
---
# Plan: P on passages — does the arm signature survive the trip to the page?

RH's question (2026-08-13, in session): can we predict base/aligned from
their GENERATIONS, and do the AUCs there replicate the AUCs from next-word
probabilities? Drafted by the lacan seat; M06 is the module because the
substrate is the passage corpus and the strata are M06's own.

## Why this comparison is clean

The passage corpus's prompts are M01 sites and its 42 pairs are the
lineage representatives — the same objects the logit-side instruments run
on. Both grains can therefore be computed on IDENTICAL prompts and models,
and the comparison isolates exactly one thing: the next-word DISTRIBUTION
against 100-200 tokens of SAMPLED TEXT at temp 1.0. Policy against
behaviour, everything else held.

## Population

`gen_sequences`, `corpus='passage'`, **UNDISTURBED ARM ONLY for the
primary** (`forced_word=''`; the forced arms condition text on injected
words and are a secondary replication table, per plan A Amendment §1).
Strata: M06's declared screens, joined from `m06_text_flags.parquet` on
(pair, role, prompt_id, seq_idx) WITH the explosion assert. Primary
stratum: prose AND non-degenerate AND English (the hardened stratum).
**SmolLM2-360M is EXCLUDED from the stratified primary** — its flag rows
are ambiguous by key (two runs, disagreeing flags, [5707]) — and rides in
the pooled read only; n = 41 pairs primary. Everything is reported with
its stratum named.

## The instruments

Word rates per model: `fields.tokens()` over passage text (the campaign
tokenizer, matching M02's field analysis), counts per 1,000 tokens over
the model's pooled hardened-stratum undisturbed passages.

  I1  GENERATION-SIDE PER-WORD ARM AUC. For each word present in >= 20
      models, the AUC of its per-model rate over the arm label. High =
      aligned-side, matching `word_auc_en.tsv`'s orientation. Compositional
      caveat inherited: rates are shares of emitted tokens, so the
      observed-median / arm-flip-null / concentration references from
      `k_word_auc` are computed and reported the same way.

  I2  GENERATION-SIDE ARM CLASSIFIER. Columns = top-k words by pooled
      frequency, values = per-model rates, logistic with L2, leave-one-org-
      out, within-lineage arm-flip null. **The grid is declared here, and
      the PLATEAU is the quotable form, never a cell** (the [5720]-era
      lesson): k in {25, 50, 100, 200}.

  I3  CROSS-GRAIN REPLICATION. Spearman of I1 against (a) the canonical
      `word_auc_en.tsv` vector and (b) a logit-side per-word AUC recomputed
      ON THE PASSAGE PROMPTS ONLY (a `--prompts` variant of `k_word_auc`,
      suffixed output, nothing overwritten). (b) is the primary comparison:
      vectors are prompt-conditional at +0.711 even within grain, so (a)
      carries a known ceiling and is context, not the test.

  I4  THE AMPLIFICATION MAP (exploratory, descriptive, no directions).
      Per shared word: gen-side AUC minus logit-side AUC (same prompts).
      Positive = the page amplifies the word's arm signal beyond its
      distributional disposition (candidates: self-reinforcing register,
      formatting-adjacent vocabulary); negative = distributional signal
      that sampling rarely realises (candidates: tail-rank words). Field
      decomposition of the map via `k_field_poles` machinery.

## Declared directions

  P1 (directional, from M02's field replication on f11_l2 and the M06
      pilot's lexical carryover): I3(b) is POSITIVE and substantial —
      the signature survives to the page.
  Q1 (open): whether I2's plateau reaches the logit classifier's 0.90+.
      Lower is expected on power alone (41 pairs, ~190 undisturbed prompts
      per model vs 2,229); the comparison is reported against a
      same-population logit rerun, not against the headline plateau.
  Q2 (open): the amplification map's field structure. Exploratory.

## Fences, before any number

- ANTI-CONFLATION: usage-rate AUC and candidate-share AUC are different
  objects even when correlated; TTR/Gini clauses apply; no sentence reads
  one as the other.
- 41 pairs primary (SmolLM2 flag ambiguity), 42 pooled.
- Length enters through the per-1,000-token denominator; list-formatting
  and degeneracy through the stratum, not through word rates.
- One corpus, one prompt family: I3(a)'s ceiling is stated wherever (a)
  is shown.

## Stages

  SMOKE (this commit): 4 scout pairs (Amber, Olmo-3, Llama-3.1, gemma-2),
  undisturbed hardened stratum, pipeline end to end — flag join integrity
  (explosion assert), rate computation, a handful of I1 values for words
  with known logit-side positions, I3 direction on the smoke vocabulary.
  EYEBALL GRADE; nothing from the smoke is quoted, per M06 house style.

  FULL: all 41/42 pairs, all four instruments, results to
  `results/p_on_passages.json`; the `k_word_auc --prompts` variant run
  beside it. No new compute anywhere; everything reads CH and the flags.

Producer: `scripts/m06_p_on_passages.py`.
