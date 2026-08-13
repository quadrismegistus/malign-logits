---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades
date: 2026-08-13
role: plan
topics: [surprisal, drift, quadrants, jakobson, replication]
description: "Plan: F15 ON PASSAGES -- the surprisal x drift quadrant instrument reimplemented on the M06 passage corpus. F15's three claims become declared directional predictions at the pair grain (41 pairs vs F15's 10 families): alignment smooths surprisal, reduces drift, drains the breakdown quadrant into metonymic/unmarked. Committed instrument functions imported, not reimplemented; reference-model independence re-argued because Pythia is now IN the roster. Drafted by the lacan seat on RH's word; smoke first."
---
# Plan: F15 on passages -- does the quadrant flow survive the new corpus?

RH (2026-08-13, in session): "work on the surprisal x drift reimplementation
using new passage corpora." F15 (grade C, unaudited, 2026-05-17) measured
76,214 passages from 10 families on a 47-prompt battery; the M06 corpus is
41 matched pairs on the M01 sites at a forced-continuation rung. Different
corpus, different rung, ~4x the model pairs, with M06's strata and unit
discipline. This is the first entry on `meta/TODO.md`'s never-retried list
to get its retry.

## The instrument, imported not reimplemented

`malign_logits/embedding.py`, the committed F15 machinery:
`drift_metrics_from_embeddings` (successive-sentence cosine steps;
total_drift = 1 - min pairwise similarity), reference-model token surprisal
(`passage_surprisal*`), quadrants from the two. The producer imports these
functions; any adaptation is glue (fetch, strata, unit), never metric code.

One record discrepancy, stated now: the F15 finding text says passages were
truncated to the minimum sentences exceeding 75 words; the script signature
default is 100. This run declares min_words=75 per the finding text and
reports the truncation survival rate.

## Population

Undisturbed passages (`forced_word=''`), corpus='passage', hardened stratum
(non-degenerate AND English, per the M06 flags; prose screen still
deferred), SmolLM2 excluded ([5707]); 41 pairs. Truncation as above;
passages that cannot reach 75 words at a sentence boundary are dropped and
counted. SMOKE: 4 scout pairs (Amber, Olmo-3, Llama-3.1, gemma-2), at most
3 samples per (pair, role, prompt), eyeball grade, nothing quoted. FULL:
subsample size set from the smoke's measured throughput and DECLARED ON THE
DOCKET before the full run; stratified per (pair, role, prompt) with a
fixed seed over sorted keys (the [5744] rule).

## The two metrics and their references

SURPRISAL: reference = GPT-2 124M, primary. F15's primary was
Pythia-1B-deduped on an independence argument that has since expired --
Pythia is now two measured lineages of this roster, so a Pythia reference
scores its own family's generations with itself. GPT-2 is aligned by
nobody in the roster and was one of F15's validated references ("all
findings hold under all references"). Pythia-1B-deduped runs as SECONDARY
on the subsample WITH the pythia pairs excluded from that read.

DRIFT: sentence embeddings. Primary for the full population:
paraphrase-multilingual-MiniLM-L12-v2 (the module default; cheap).
Fidelity check: bge-m3 (F15's headline embedder) on a declared subsample;
the quotable claim requires sign agreement between the two. DEVICE: the
mps hazard is on record (single-CJK bge rows); this corpus is English
sentences, so mps is permitted ONLY behind the k_delta_embed-style gate --
60 rows (20 shortest + 40 random) re-encoded on CPU, cosine >= 0.999 or
the producer refuses to write. The gate runs on the store's own rows.

## Unit and tests

The passage is scored; the CELL is (pair, role): mean over its passages.
Contrasts are PAIRED PER PAIR across 41 pairs (sign test + median, means
labelled beside, the [5762] rule). F15's z-scored family deltas are not
reproduced as such; the pair grain is stricter and is the point.

Quadrant thresholds: pooled medians over BOTH arms' cell values in this
corpus, computed once, reported. Q1 metonymic (high drift, low surprisal);
Q2 breakdown (high, high); Q3 metaphoric (low drift, high surprisal); Q4
unmarked (low, low). Per (pair, role): share of passages per quadrant.

## Directions, declared before any number

  P1 (F15: all 10 families): aligned mean surprisal < base, per pair.
  P2 (F15: 9 of 10): aligned drift < base, per pair.
  P3 (F15's quadrant flow): base Q2 share > aligned Q2 share, per pair;
      the drained mass lands in Q1+Q4 rather than Q3.
  Q1 (open, no direction): whether the Q1 (metonymic) share RISES under
      alignment at the pair grain -- F15 saw it in one family (Qwen).
      Either answer is a finding.
  Q2 (open): whether the P-axis interiority score of a passage correlates
      with its quadrant assignment -- the bridge between this instrument
      and the P-on-passages series. Exploratory, stated so the two
      instruments are not conflated: axis score is COMPOSITION, drift is
      TRAJECTORY; no sentence reads one as the other.

## Fences

- Different corpus AND different rung from F15 (forced continuation on
  M01 sites vs plain generation on the 47-prompt battery): agreement
  extends F15, disagreement does not retract it -- it localises it.
- Truncation selects on length; the survival rate travels with every
  contrast, per arm (alignment changes length, so differential survival
  is itself a selection channel -- reported before any metric).
- Sharpness confound, inherited from P: mean surprisal under a REFERENCE
  model is not the generator's own sharpness, but low-entropy generators
  emit more predictable text; the two are entangled on the page. Stated,
  not solved here.
- Per-cell scores persisted to parquet (the [5760] habit).

Producer: `scripts/m06_f15_on_passages.py`. Results:
`results/f15_on_passages{_smoke,}.json` + per-cell parquet.
