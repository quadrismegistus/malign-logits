# Plan M05-A: the Think-SFT acquisition ladder

Written under the [5148] standard: a plan document, not a registration. No
freeze, no hash ceremony; the hashes are here so the pen can check the
population against the source of record in one line.

**Authorisation:** RH, 2026-08-10, "Start a plan for the emergence." The
directory question (new M05, not M01 or M02) was the pen's call and is RH's to
overturn — it is a rename if he does. **Costing is OWED by @malign before any
box spins; the join check (free, local) can run first.**

## QUESTION

In what order does alignment install its operations within a single SFT run —
and specifically, does repression precede displacement?

F04 answered at 10 checkpoints: kill falls by step5000, scream rises from
step10000 — a lag. But 10 checkpoints cannot distinguish a real two-stage
sequence from a sampling artifact, F04 is unaudited/C, and its instrument
predates the twp discipline. The 43-step public ladder plus the standing
battery is the powered version of the question F04 asked first.

Three named secondaries ride the same passes:

1. **Format before correction.** The pilot found the school-format attractor
   fully installed at step1000 while the corrective clause ("But she didn't")
   is absent until later. If template behaviors install before content
   repression, "socialisation" decomposes into stages with different objects.
2. **`<think>` dormancy.** p(first `<think>` piece) per checkpoint, free from
   the same forward pass. The pilot puts it at rank 13k–15k at both ends —
   the reasoning frame is template-keyed, not ambient (contrast Falcon3's
   27–53% assistant leak). A curve confirms or breaks that.
3. **The join.** Which base revision did SFT step1000 descend from? Weight
   delta against base `main` and against the stage3 anneal tips (13
   candidates), via the head_frozen_survey machinery. This is the free first
   line: it makes the two ladders one run *by measurement* rather than by
   vendor claim, and phase 2 depends on it.

## INPUT

**Population: `data/m05_checkpoint_population.json` — 45 checkpoints,
ENUMERATED as (model_id, revision) strings.** Producer
`scripts/build_m05_population.py`, which refuses to write unless the step set
is contiguous at 1000-step spacing (it is: step1000..step43000).

    population                 45 checkpoints    sha256/16  7f7b2565e25b3be3
    source of record           data/model_revisions.json    7ccdca07457e7545

    sft_step        43     allenai/Olmo-3-7B-Think-SFT @ step1000..step43000
    sft_endpoint     1     allenai/Olmo-3-7B-Think-SFT @ main
    base_endpoint    1     allenai/Olmo-3-1025-7B @ main

**Battery: the seats' costing decision, between two declared options.**
(a) The standing 2,583-prompt battery, full — every checkpoint lands directly
comparable to finding U's endpoints and to the whole M01 record.
(b) The M01 displacement core (the U faller/riser vocabulary's prompt subset
plus the F04 panel prompts) — cheaper, still answers the primary, loses
one-instrument comparability for the long tail.
The choice is a COST line, not a science line, and belongs to @malign's
costing post. Whichever is chosen is enumerated and hashed before the run.

## INSTRUMENT

**`scripts/twp_cloud.py`. The column the analysis reads: `true_word_probs`**
(theta 0.001, complete above the floor, per RH's [5136] ruling). Logit and
hidden sidecars ride along under the standard positional contract.
`compute_dtype` declared in the spec — bf16, matching the fleet standard for
this architecture family. Revision-loading is the one new demand on the
instrument: `from_pretrained(model_id, revision=...)` per checkpoint, and the
cell key must carry the revision (a checkpoint is not a model_id; two rows
differing only in revision are different cells — the unit-word lesson).

For secondary 1, one greedy continuation (256 tokens) per checkpoint on the
pilot's 4-prompt panel — 45 x 4 generations, negligible next to the battery.
For secondary 3, `head_frozen_survey` weight deltas, local, no new instrument.

## OUTPUT

    data/m05_twp/<model>@<revision>.jsonl     per-cell twp records
    data/m05_twp/<model>@<revision>.f16       logit sidecar
    data/m05_greedy/<revision>.json           the 4-prompt continuations
    data/m05_join_deltas.json                 weight-delta table, 14 candidates

Ingest through the standard path (twp_ingest + sidecar check); the cells land
in the same stores under the revision-bearing key.

## ANALYSIS — priors said now, both branches

**Primary: onset ordering.** For each word in the F04 panel (fuck, kill,
scream, said) and each U faller/riser present in the battery: the checkpoint
trajectory p(word | prompt, step), aggregated over prompts, with bootstrap CIs
over prompts. Onset = first step at which the trajectory leaves the base
endpoint's CI and stays out. The primary contrast: onset(repression of
fallers) vs onset(rise of risers), paired within prompt site.

- **If repression onsets earlier than displacement, with dense sampling.**
  F04 is confirmed and upgraded: the lag is real, its width is now a measured
  number with an interval, and "the substitute arrives after the prohibition"
  becomes quotable at audit grade. F04 gets its trail (superseded by the
  powered version, direction intact).
- **If onsets are simultaneous at 1000-step resolution.** F04's lag was a
  sampling artifact of 10 checkpoints; the operations install as one event.
  That is a finding, not a failure — it says displacement is constitutive of
  repression here, not a later accommodation, and F04's trail says so.
- **If trajectories are non-monotonic** (F04 saw kill bounce at step20000):
  onset is reported alongside the full curve and no single-number ordering is
  quoted without the curve's shape. A bounce is a fact about the run, not
  noise to smooth.

**Secondary 1 (format vs correction):** per-checkpoint presence of the
format attractor (answer-scaffold markers in the recipe continuation) vs the
corrective clause (negation-after-impulse in the anger continuation), coded by
rule on the greedy outputs — string-rule coding, declared before the run, no
LLM coder. Either order is reportable; the prior from the pilot is
format-first, and if correction arrives first the pilot's two-point read was
an endpoint illusion.

**Secondary 2 (`<think>`):** rank and p per checkpoint. Prior: flat and deep
(template-keyed). If it rises across SFT, the reasoning frame is being
ambiently installed and the Falcon3 contrast weakens.

**Secondary 3 (the join):** if no stage3 candidate is materially closer than
base `main`, the vendor's lineage claim stands unrefined and phase 2 anchors
on `main`; if one anneal tip is closest, phase 2 anchors there and the two
ladders are joined at a named revision.

## COST

**Owed by @malign, not guessed here.** The shape: 45 model loads of a 7B at
one revision each (load-dominated; the fleet's per-load and per-prompt rates
are known from the census), times the battery choice (2,583 vs the core
subset). The greedy panel and the join check are noise next to it. The join
check costs nothing but local disk and can run today.
