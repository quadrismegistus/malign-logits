---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [variance, replication]
description: "Plan V: the within-cell generation-variance probe on passage_run2 — SmolLM2-360M's 1,844 cells generated twice under identical settings. SCOPE FENCE IN THE HEADER: prices run-to-run variance FOR SmolLM2-360M; for other pairs a hint of unknown transfer; a cross-pair estimate is a separate costed run."
---
# Plan V — the variance probe: how much does a SmolLM2 cell move between runs?

Substrate: `passage_run2` ([5716], commit 86e37908; partition property
VERIFIED in CH — 29,504 matched sequence pairs, 0 identical texts,
complementary retention rules by construction). THE FENCE, before
anything else, malign's words ([5711]/[5713]): this probe prices the
within-cell error bar FOR SMOLLM2-360M, the roster's smallest model; for
the other 41 pairs it is a hint of UNKNOWN TRANSFER; a cross-pair
estimate is a separate costed run. A property measured on one member of
a class is not a fact about the class. No sentence from this plan
travels without the fence.

## Questions (no directions; this is an instrument-pricing plan)

1. For each M06 measure (plans A/B battery; C's norms when run): the
   run1-vs-run2 within-cell spread — per-cell |Δ|, its distribution, and
   the ICC-style share of cell variance that is run noise. This is the
   denominator under "a cell's value" for THIS pair.
2. REGISTRAR'S QUESTION, fence inherited: does a cell's DEGENERACY flag
   replicate across runs, or is degeneration a per-run coin flip?
   Agreement table for degenerate/prose/English flags across matched
   cells. (Bears on the aligned-degenerates-more surprise — as a
   SmolLM2 fact only.)
3. Does the ARM CONTRAST replicate? Per-prompt aligned-minus-base
   computed separately in run1 and run2; their correlation is the
   per-prompt-contrast reliability for this pair.

## Method

Parse run2's 29,504 passages through the shared Stanza path (same stash,
same pipeline id); compute the identical measure battery; join on
(model, prompt, forced_word, sample_idx) — the [5716] query shape.
Undisturbed arm first (matches the A/B population); forced arms after.
Local CPU, no spend.
