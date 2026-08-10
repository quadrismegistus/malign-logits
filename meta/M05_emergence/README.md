# M05 — Emergence: when the operations install

**The axis.** M01–M04 all compare states: base against aligned, arm against
arm. M05's unit is the **checkpoint within a single training run** — the time
axis. The question is not whether the operations exist (M01/M02/M03/M04 have
answered at their grains) but **in what order they install**, and whether that
order is the same order the theories predict.

**Why this is not another M01 letter.** M01's finding U measured the rungs of
the ladder (base -> SFT -> DPO -> RLVR) as endpoints: SFT carries 74% of the
edge's JS, DPO re-targets at lower amplitude. Every M01 registration takes the
base/aligned pair as its unit. M05 goes *inside* a rung: the 43 public
checkpoints of one SFT run, and later the pretraining ladder itself. New unit,
new population discipline, new directory — not because M01 ran out of letters
but because a checkpoint is not a pair.

**Prior work this audits and extends** (both unaudited/C):

- `findings/F04_step_analysis.md` (2026-05-17): OLMo Think-SFT at 10
  checkpoints — sexual repression as phase transition by step1000; violence
  non-monotonic; **repression precedes displacement** (kill falls by step5000,
  scream rises from step10000). M05 phase 1 is F04's declared upgrade path.
- `findings/F24_pretraining_emergence.md`: Pythia pretraining sequence —
  drives -> structure -> deference -> superposition. Its open TODO ("test
  whether the developmental sequence holds for OLMo") is M05 phase 2, on the
  Olmo-3-1025-7B base ladder (1,487 revisions: stage1 1,421 / stage2 52 /
  stage3 13).
- Pilot (pen, 2026-08-10, docket record): Think-SFT step1000 vs final on the
  anger prompt — kill 0.127 (#1) -> 0.035 (#5), scream 0.022 -> 0.057; the
  school-format attractor fully present at step1000 while the corrective
  clause is absent; `<think>` dormant throughout (rank 13k–15k).

**Lineage fact, verified 2026-08-10:** Think-SFT is the FIRST post-training of
Olmo-3-1025-7B (base_model metadata + Olmo 3 report). Joined with the base
ladder this is the closest thing any vendor releases to a continuous
pretraining-to-post-training run; no family anywhere releases preference-stage
trajectories (agent survey, 2026-08-10, data/model_revisions.json).

    plans/a_thinksft_acquisition.md   phase 1 — the 43-step SFT ladder
    (phase 2, base-ladder subsample for F24's TODO, gets its own plan
    document after phase 1 lands)

Plans here follow the [5148] standard: plan documents, not registrations.
Populations enumerated as strings and hashed; instrument column named; priors
said for every branch before the run.
