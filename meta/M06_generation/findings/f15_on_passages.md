---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/f15_on_passages.json
date: 2026-08-13
role: finding
topics: [surprisal, drift, quadrants, jakobson, replication]
description: "F15 on passages, single pass: all three F15 claims replicate at the pair grain -- alignment smooths reference-model surprisal (35/38 pairs, med -0.53 nats), reduces drift (34/38), and drains the breakdown quadrant (Q2 -0.21, 35/38) into unmarked (Q4 +0.22) AND metonymic (Q1 +0.14, 34/38) -- the Q1 gain F15 saw only in Qwen is general. Population 38 of 41 pairs; deepseek excluded by a CORPUS TEXT DEFECT found in the accounting (undetokenized text, reported to malign)."
---
# F15 on passages: the quadrant flow survives, and the metonymic gain is general

Plan: `plans/plan_f15_on_passages.md` (f2f5c804, committed before the
producer). Producer `scripts/m06_f15_on_passages.py`; results
`results/f15_on_passages.json` + per-passage parquet (35,230 rows).
Single pass; the [5503] rule applies. Metric code imported from the
committed F15 instrument (`malign_logits/embedding.py`), never
reimplemented. Reference GPT-2 (the only reference still independent of
the roster -- Pythia joined it); embedder MiniLM, mps behind the
second-device gate (min cos 1.0000 on the corpus's own rows). Subsample
declared on the docket before the run ([5765]): cap 3 per (pair, role,
prompt), seed over sorted keys.

## Population, and the accounting that found a corpus defect

38 pairs enter the paired contrasts. Named exclusions, each with its
mechanism:

- **deepseek**: 95% of its passage-corpus texts are spaceless with
  literal byte markers (mean 1.6 space-delimited words per 1,300
  chars), so 2,481 of 2,481 stratum-passing passages fail the 75-word
  rule. Reported to malign; ruled a generation-time tokenizer defect
  and FENCED, not repaired ([5776],
  `data/deepseek_passage_text_fence.json`): the word-initial spaces
  were never in the token ids (`['mu','ff','led','her',...]`, no
  `Ġher`), so no decode restores them and the ids themselves are
  intact. **My own inference here was wrong and is corrected in
  place**: I read the flags' english_nltkwords_share of 0.87 as
  evidence that spaced text once existed. It did not. Registrar
  recomputed the flags through their own producer path (416 rows, zero
  mismatches, [5777]): the screen's regex `[a-zA-Z']+` treats the
  byte markers as word boundaries, so marker-delimited stretches yield
  real words while concatenated stretches collapse to one blob. The
  0.87 is the honest output of that regex on these bytes. A regex is a
  tokenizer, and a screen built on one inherits its boundary
  conventions on exactly the pathological rows it exists to catch.
- **recurrentgemma**: 0.2-0.5% pass the degeneracy/English screens.
- **Teuken**: base arm 0% English.
- **bloomz**: aligned arm 7 surviving passages, below any floor.

Truncation survival differs by arm corpus-wide (base 0.836 vs aligned
0.805) -- the selection channel the plan fences; it travels with every
contrast below.

## The three declared directions, all confirmed (paired per pair, n=38)

    (medians travel; means beside)
    P1  mean_surprisal, aligned - base   med -0.526 (mean -0.541) nats  3/35   p=7e-8
    P2  total_drift                      med -0.023 (mean -0.032)       4/34   p=6e-7
    P3  quadrant shares, aligned - base:
        Q2 breakdown  (hi drift, hi surp)   med -0.211  3/35   p=7e-8
        Q4 unmarked   (lo, lo)              med +0.224  35/3   p=7e-8
        Q1 metonymic  (hi drift, lo surp)   med +0.137  34/4   p=6e-7
        Q3 metaphoric (lo drift, hi surp)   med -0.157  4/34   p=6e-7

F15's flow (Q2 drains into Q1 and Q4) replicates on a different corpus,
a different rung, and 4x the pairs. **And the plan's open question Q1
answers YES: the metonymic share rises under alignment in 34 of 38
pairs -- what F15 saw in one family (Qwen) is general.** Both
high-surprisal quadrants drain and both low-surprisal quadrants gain,
so the flow is carried by the surprisal axis with the drift axis moving
less: aligned text keeps crossing topics but becomes predictable while
doing so. Chain-sliding, at the pair grain.

## Open, declared -- and one F15 claim this redo did NOT carry

**GATING (nothing above is quotable as embedder-independent until it
runs): the bge-m3 fidelity subsample.** The plan requires sign
agreement between embedders for the quotable claim, and drift is not
only P2's outcome -- it is ONE OF THE TWO QUADRANT AXES, so P3 and the
metonymic result inherit the requirement. MiniLM alone is a single
instrument's read of trajectory.

**NOT CARRIED, and it is a real gap rather than a deferral: F15's
FOURTH claim.** F15 says "content category has no effect on
within-passage surprisal (Kruskal-Wallis p=0.99); alignment is a
uniform compressor." This redo tested three claims and never tested
that one. It is cheap now: the cells parquet carries `prompt_id`, and
prompts join to catalogue domain/pair_role exactly as I6 did. It is
also the claim with the most at stake after I6 -- uniform compression
is the surprisal-grain analogue of the TONIC result, and a failure
would be the first page-grain site-conditionality in the series.

Also open:

- Sharpness confound, testable rather than merely stated: the pair-wise
  surprisal delta against the same pair's top-1-mass/entropy delta from
  the twp side (P's nuisance block puts top-1 mass at AUC 0.889 for the
  arm). A partial correlation would say how much of P1 is generator
  sharpness rather than page predictability. Never run; the fence
  currently states the confound without bounding it.
- Pythia-1B-deduped secondary reference on the subsample, pythia pairs
  excluded. Not yet run.
- Prose-stratum rerun inherits from the M06 series.
- Plan Q2, the axis-quadrant bridge to the P series. Exploratory, not
  owed.
- Producer-layer reproduction: the aggregation layer is second-seated
  ([5772]/[5775], all six aggregates rebuilt from the cells); the fetch,
  strata, truncation and metric application remain single-pass.

## Fences

Different corpus AND rung from F15: agreement extends it, and this is
agreement. Sharpness confound stated in the plan: reference surprisal
entangles generator sharpness with text predictability on the page;
this instrument cannot separate them. Truncation selects on length per
arm (rates above). Quadrant thresholds are this corpus's own pooled
medians (drift 0.9393, surprisal 3.7278 nats -- from the JSON), not
F15's.
