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

## Findings index

Four documents, all drafts at grade C, all on ONE lineage (OLMo-3), written
2026-08-11 by the registrar seat. Figures live in `figures/fig1..fig13`
(plotnine, 300 dpi) and are embedded in the findings; every finding re-derives
from the scripts named in its own header.

| Doc | Claim (one line) | Status/grade (verbatim) | Unit and population | Figures | Corrects / superseded-by |
| --- | --- | --- | --- | --- | --- |
| `findings/A_acquisition.md` | The prohibition is an event that completes inside SFT while the substitution is a drift that never stops arriving; R1 (primary) replaces F04's lag with a difference in kind (paired p = .97), R2 puts discourse tracking an order of magnitude after the four-way tie at the floor, R4 finds contradiction ratio and pole separation co-moving with no lead either way. | "STATUS: DRAFT, grade C — single run, single lineage (OLMo-3), no cross-seat audit yet." | Checkpoint rung on ONE lineage (OLMo-3): 95 checkpoints (base 42 + SFT 43 + DPO endpoint + RLVR 7 + 3 mains), population sha 495eee8deb6ca20a; battery 584 texts sha e5a4f5fb9f1f4907; `data/m05_curves.parquet`, 49,210 rows; onsets aggregated over 105 prompts, paired test n = 44 sites. | `fig1_event_vs_drift.png`, `fig2_capacity_acquisition.png`, `fig4_ratio_unjoined.png` | Audits `findings/F04_step_analysis.md` at 43 rungs: its specific trajectories fail ([5424]), the direction-level supersession stands. Refines F26; consistent with M01 finding U; scopes the RLVR raw-mode probe. |
| `findings/B_field_flow.md` | Alignment de-concretizes the continuation, pulling mass off concrete physical action toward the grammatical and abstract; R2 has nine of the ten largest fine-field movers falling and concrete-physical with concreteness itself down -0.027, R3 replicates the direction in four independent lexicons at the SFT boundary, R1 shows pretraining builds the field structure that alignment only trims. | "STATUS: DRAFT, grade C — single run, single lineage (OLMo-3), no cross-seat audit, no declared null on the per-field ordering." | Checkpoint rung on ONE lineage (OLMo-3): the same 95-checkpoint ladder, median over the 105 minimal pairs, reference-free per checkpoint; 13 meta-fields and 287 fine fields; `data/m05_field_flow_fine.parquet`; lexicon coverage ~0.58. | `fig8a_field_flow.png`, `fig8c_alignment_field_delta.png`, `fig9a_fine_field_movers.png`, `fig10_norm_field_flow.png`, `fig10_rid_field_flow.png`, `fig10_wn_field_flow.png`, `fig10_usas_field_flow.png` | Adds only WHEN to M01 Registration T, `../../M01_displacement/findings/T_category_flow.md`, which holds the same direction at the edge unit and is the generalisable form (incl. T-7 on concreteness); this document does not add generalisation. |
| `findings/C_affective_convergence.md` | Site-specificity lives in the affective coloring rather than in how much mass moves; R1 finds the pooled demoted-mass gap null, R2 finds 44 fields diverging at q < 0.05 under a within-pair permutation, R3 finds alignment narrowing the drive and affect gaps on this lineage (sign not robust), R4 splits alignment into displace on violence and refuse on sexual. | "STATUS: DRAFT, grade C for the one-lineage claims; the two permutation results are FDR-controlled but still one lineage." | Checkpoint rung on ONE lineage (OLMo-3): 105 minimal pairs, each a marked/unmarked twin sharing a stem, over the 95-checkpoint ladder; nulls are 20,000 within-pair sign-flip draws with BH-FDR over 269 fields; `data/m05_pair_displacement.parquet`, `data/m05_divergence_null.json`, `data/m05_widening_null.json`. | `fig7b_pair_gap.png`, `fig12b_marked_divergence_signed.png`, `fig13_widening_did.png` | Result 3 is SUPERSEDED ON THE SIGN by M01 Registration T finding 18, `../../M01_displacement/findings/T_category_flow.md` (edge-unit DiD, 788 twins, 50-51 edges: the direction flips to divergence, RID:aggression null at q .43), as the document itself records. Internal correction: an earlier widening read off the fig12b base-dot overlay is withdrawn. Consistent with T-13; does not overturn D2. |
| `findings/D_word_trajectories.md` | Displacement watched word by word on the deployed cut, with the transgressive continuation crushed at the SFT boundary as its substitute rises; R2 gives the poetic arc (pull learned early, seen through in mature pretraining, re-bound +88% under alignment against a flat floor), R3 makes the displacement axis scene- and lineage-specific, R4 finds Step(SFT, DPO) empty. | "STATUS: DRAFT, grade C — single lineage (OLMo-3), illustrative/descriptive; the per-prompt panels are examples, not a rate." | Checkpoint rung on ONE lineage (OLMo-3): the same 95-checkpoint ladder; 22 per-prompt panels with movers selected on Step(base, DPO) under CANONICAL; poetic pull measured over 20 binomial/rhyming/alliterative pairs. | `fig5_poetic_pull_floor.png`, `fig6_violence_liminal__the_soldier_raised_his_rifle_and.png`, `fig6_violence_liminal__she_was_so_angry_she_wanted_to.png`, plus the twenty further `fig6_*` panels referenced as a set | Adds a lineage axis to Registration X, whose "contiguity, not resemblance" was already withdrawn as scene-specific. Consistent with M05-A's event-in-SFT and with M01 finding U. |

Owed, because all four documents are one lineage: the 46-lineage convergence
DiD (M01 store), the Pythia second fleet, the syntax curve (secondary 5, which
needs the frozen licit-category artifact), and the join check (which base
revision SFT descends from).

### Cross-index notes

- `findings/C_affective_convergence.md` Result 3 -> superseded on the sign by
  M01 Registration T finding 18, `../../M01_displacement/findings/T_category_flow.md`.
- `findings/C_affective_convergence.md` Results 1-2 -> the edge-unit form of
  the same mechanism is M01 Registration T finding 13, same file.
- `findings/B_field_flow.md` -> the generalisable, edge-unit form of the same
  direction is M01 Registration T (findings 7, 11, 16), same file.
- `findings/A_acquisition.md` -> audits `../../findings/F04_step_analysis.md`
  ([5424]); direction-level supersession stands, specific trajectories fail.

## Plans

    plans/a_thinksft_acquisition.md   phase 1 — the 43-step SFT ladder
    (phase 2, base-ladder subsample for F24's TODO, gets its own plan
    document after phase 1 lands)

Plans here follow the [5148] standard: plan documents, not registrations.
Populations enumerated as strings and hashed; instrument column named; priors
said for every branch before the run.
