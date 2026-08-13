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

## Amendments, 2026-08-13, declared AFTER the main run and BEFORE their producers

The main run's numbers are known; these three instruments are not, and
each direction below is written before its producer exists.

  F1  BGE-M3 FIDELITY (the gating item; the plan already required it).
      Declared subsample of the SCORED passages: 60 per (pair, role),
      seed 20260813 over sorted keys. Same truncation, same sentence
      split, same prefix-on-first-sentence recipe, bge-m3 in place of
      MiniLM, device gate as in the main producer. Surprisal is
      UNCHANGED (same GPT-2 values from the cells parquet), so the
      quadrant axes move only through drift.
      Declared test: SIGN AGREEMENT on P2 per pair, and the direction
      of the quadrant flow under bge. The quotable form of P2/P3 is
      whatever survives BOTH embedders; where they disagree the claim
      is embedder-dependent and is said so.
      Direction P4: the two embedders agree in sign on P2 for a
      LARGE MAJORITY of the 38 pairs. No threshold is set in advance
      beyond "majority", because a fidelity check with a pass mark
      chosen by its author is not a check.

  F2  SITE SPECIFICITY OF THE COMPRESSION -- F15's FOURTH CLAIM,
      uncarried by the main run and carried here. F15: "content
      category has no effect on within-passage surprisal
      (Kruskal-Wallis p=0.99); alignment is a uniform compressor."
      No new compute: the cells parquet plus the catalogue join used
      by I6 (prompt TEXT, never id).
        F2a  I6-FORM, primary, paired per (pair, pair_id):
             MARKED - UNMARKED for mean_surprisal and total_drift
             within each arm, then the DiD (aligned excess minus base
             excess).
        F2b  F15-FORM, secondary: per (pair, domain) aligned-minus-base
             surprisal delta; Kruskal-Wallis across domains over pair
             deltas; per-domain deltas reported whatever the test says.
      Direction P5 (from I6's tonic result AND F15 independently, so
      the two converge): the DiD is NULL and the domain test does not
      reject -- compression is uniform across site types. The
      alternative is named and would be the first page-grain
      site-conditionality in this series: a NEGATIVE DiD on surprisal
      would mean the aligned model compresses transgressive sites
      HARDER than its own neutral twins, over and above base.

  F3  DOES FORCING ALTER THE MOVEMENTS? (RH's question, 2026-08-13.)
      I5 asked what forcing does to COMPOSITION and the ascent branch
      to LEVEL; this asks TRAJECTORY and PREDICTABILITY. Forced
      passages, same strata, arms from the committed arms table
      (faller / matched / riser_matched), capped 2 per (pair, prompt,
      role, arm), same truncation and metrics.
        F3a  DOES THE ARM CONTRAST SURVIVE FORCING: per pair, the
             aligned-minus-base gap in surprisal and drift computed on
             the MATCHED arm (a forced but non-transgressive word),
             compared with the undisturbed gap from the main run.
        F3b  DOES FORCING ITSELF MOVE THE METRICS: per (pair, prompt)
             within arm, faller - matched and riser_matched - matched
             for surprisal and drift, sign test.
        F3c  THE DiD: (aligned faller-excess) - (base faller-excess),
             the I5b design with F15's metrics as outcomes.
      Directions: P6, the arm contrast SURVIVES forcing (the signature
      is a disposition, per I5/I6; forcing one word should not abolish
      a whole-passage register difference). Q3, open, no direction:
      whether a forced faller RAISES surprisal (the injected word is
      off-policy for the aligned model, so its continuation may be
      locally surprising to a third-party reference) or LOWERS it
      (the model recovers into generic continuation). Q4, open:
      whether F3c is null like every other DiD in this series.

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
