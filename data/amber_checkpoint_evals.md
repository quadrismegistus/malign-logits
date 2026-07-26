# Amber pretraining capability curve

Companion to `data/amber_checkpoint_evals.csv`. Built 2026-07-25.

## What this is

LLM360 published benchmark evaluations on every one of the 359 released
pretraining checkpoints of `LLM360/Amber` (7B, Llama architecture,
`ckpt_000` through `ckpt_358`). This file collects them: ARC-challenge
(25-shot), HellaSwag, MMLU (mean accuracy over 57 `hendrycksTest-*`
subjects), and TruthfulQA (mc1/mc2). No weights were downloaded; only the
per-branch eval JSONs.

The point is that this is an **external** capability measure. It was
computed by LLM360 on standard benchmarks, and it is independent of every
rating instrument in this project. The campaign's oldest confound (R1,
"aligned models write better, so any rated literary property rides on
coherence") has so far been handled by controlling for *rated fluency*,
which conditions on a post-treatment variable and forces every contrast to
be reported as a bound. An external per-checkpoint capability curve allows
a different and cleaner test: compare the *rate* at which literary form
arrives against the rate at which measured capability arrives, on the same
lineage, with no shared instrument.

## The curve

| ckpt | HellaSwag (acc_norm) | ARC-c (acc_norm) | MMLU (mean acc) | TruthfulQA (mc2) |
|---|---|---|---|---|
| 0 | 0.248 | 0.234 | 0.230 | 0.531 |
| 1 | 0.274 | 0.215 | 0.249 | 0.510 |
| 4 | 0.375 | 0.239 | 0.257 | 0.420 |
| 16 | 0.547 | 0.299 | 0.250 | 0.377 |
| 64 | 0.660 | 0.370 | 0.256 | 0.357 |
| 100 | 0.685 | 0.378 | 0.238 | 0.354 |
| 200 | 0.713 | 0.390 | 0.239 | 0.332 |
| 300 | 0.731 | 0.410 | 0.271 | 0.332 |
| 358 | 0.740 | 0.415 | 0.287 | 0.343 |

Saturation, floor-corrected against chance (0.25 for the multiple-choice
benchmarks):

| benchmark | final | 50% of gain | 90% | 95% | 99% |
|---|---|---|---|---|---|
| HellaSwag | 0.740 | ckpt_10 | ckpt_114 | ckpt_212 | ckpt_310 |
| ARC-c | 0.415 | ckpt_30 | ckpt_145 | ckpt_173 | ckpt_173 |
| MMLU | 0.287 | — | — | — | — |

Two benchmarks are not usable and should not be quoted as capability:

- **MMLU never leaves chance.** It moves 0.230 to 0.287 across the whole
  run and wanders non-monotonically in between. Amber does not acquire
  MMLU; this is consistent with LLM360's own report that Amber
  underperforms contemporaneous 7B models.
- **TruthfulQA runs backwards.** mc2 is *highest at `ckpt_000`*, i.e. at
  random initialization (0.531), and declines monotonically to 0.343. This
  is the familiar artifact that a model with no opinions cannot assert
  falsehoods, but read as an acquisition curve it is the one measure in the
  set that gets worse as the model learns: the corpus's falsehoods are
  acquired along with the corpus.

## Checkpoint quality

Usable. HellaSwag decreases at 158 of 358 consecutive steps, but the
largest single-step drop is 0.0089, i.e. noise around a rising trend rather
than instability. There are no loss-spike catastrophes visible in the eval
curves. The documented Amber weakness is a low ceiling (MMLU at chance),
not corrupted checkpoints.

## The design this enables

HellaSwag reaches 90% of its total gain by `ckpt_114`, 32% of the way
through training, and 95% by `ckpt_212`. So there is a long stretch of
training, roughly two thirds of it, over which measured commonsense
capability is nearly flat.

If the rated literary properties that arrive last in Pythia (binding,
subject stability) are still near the floor at `ckpt_114`, then a model
holding 90% of its final capability still cannot sustain a consequence
structure or a stable referent, and "it is just competence" fails as an
explanation, using numbers this project did not compute.

Suggested weight sample if that is run: `ckpt_000, 001, 002, 004, 008,
016, 032, 064, 114, 212, 310, 358`. Twelve checkpoints at roughly 13.5 GB
each, about 160 GB. Dense where capability moves, then the 90% point and
two post-saturation points where any continued rise in literary form is by
construction not a capability gain. Note that the older branches ship
`pytorch_model-*.bin` rather than safetensors.
