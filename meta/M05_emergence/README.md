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

Eight documents, written 2026-08-11. A-F are the registrar seat's, all drafts
at grade C; the two lacan-seat entries at the foot of the table are an
instrument note and a null, and carry their own status lines. A-D are on ONE lineage (OLMo-3); E is the Pythia ladder, a SEPARATE
STUDY never pooled with A-D (cross-ladder comparisons only); F runs both
ladders separately with two coder families. Figures live in
`figures/fig1..fig18` (plotnine, 300 dpi) and are embedded in the findings;
fig17 (token clock, cross-ladder, base arms) and fig18 (full ladder,
flagship ordering) are combined views over A/E/F. Every finding re-derives
from the scripts named in its own header.

| Doc | Claim (one line) | Status/grade (verbatim) | Unit and population | Figures | Corrects / superseded-by |
| --- | --- | --- | --- | --- | --- |
| `findings/A_acquisition.md` | The prohibition is an event that completes inside SFT while the substitution is a drift that never stops arriving; R1 (primary) replaces F04's lag with a difference in kind (paired p = .97), R2 puts discourse tracking an order of magnitude after the four-way tie at the floor, R4 finds contradiction ratio and pole separation co-moving with no lead either way. | "STATUS: DRAFT, grade C — single run, single lineage (OLMo-3), no cross-seat audit yet." | Checkpoint rung on ONE lineage (OLMo-3): 95 checkpoints (base 42 + SFT 43 + DPO endpoint + RLVR 7 + 3 mains), population sha 495eee8deb6ca20a; battery 584 texts sha e5a4f5fb9f1f4907; `data/m05_curves.parquet`, 49,210 rows; onsets aggregated over 105 prompts, paired test n = 44 sites. | `fig1_event_vs_drift.png`, `fig2_capacity_acquisition.png`, `fig4_ratio_unjoined.png`; combined: `fig17_acquisition_tokens_olmo.png`, `fig18_acquisition_ladder_olmo.png` | Audits `findings/F04_step_analysis.md` at 43 rungs: its specific trajectories fail ([5424]), the direction-level supersession stands. Refines F26; consistent with M01 finding U; scopes the RLVR raw-mode probe. |
| `findings/B_field_flow.md` | Alignment de-concretizes the continuation, pulling mass off concrete physical action toward the grammatical and abstract; R2 has nine of the ten largest fine-field movers falling and concrete-physical with concreteness itself down -0.027, R3 replicates the direction in four independent lexicons at the SFT boundary, R1 shows pretraining builds the field structure that alignment only trims. | "STATUS: DRAFT, grade C — single run, single lineage (OLMo-3), no cross-seat audit, no declared null on the per-field ordering." | Checkpoint rung on ONE lineage (OLMo-3): the same 95-checkpoint ladder, median over the 105 minimal pairs, reference-free per checkpoint; 13 meta-fields and 287 fine fields; `data/m05_field_flow_fine.parquet`; lexicon coverage ~0.58. | `fig8a_field_flow.png`, `fig8c_alignment_field_delta.png`, `fig9a_fine_field_movers.png`, `fig10_norm_field_flow.png`, `fig10_rid_field_flow.png`, `fig10_wn_field_flow.png`, `fig10_usas_field_flow.png` | Adds only WHEN to M01 Registration T, `../../M01_displacement/findings/T_category_flow.md`, which holds the same direction at the edge unit and is the generalisable form (incl. T-7 on concreteness); this document does not add generalisation. |
| `findings/C_affective_convergence.md` | Site-specificity lives in the affective coloring rather than in how much mass moves; R1 finds the pooled demoted-mass gap null, R2 finds 44 fields diverging at q < 0.05 under a within-pair permutation, R3 finds alignment narrowing the drive and affect gaps on this lineage (sign not robust), R4 WITHDRAWN PENDING DEFINITION (2026-08-13, [5781]): the recapture producer now exists and the domain contrast does not reproduce — violence and sexual indistinguishable under the declared definition. | "STATUS: DRAFT, grade C for the one-lineage claims; the two permutation results are FDR-controlled but still one lineage." | Checkpoint rung on ONE lineage (OLMo-3): 105 minimal pairs, each a marked/unmarked twin sharing a stem, over the 95-checkpoint ladder; nulls are 20,000 within-pair sign-flip draws with BH-FDR over 269 fields; `data/m05_pair_displacement.parquet`, `data/m05_divergence_null.json`, `data/m05_widening_null.json`. | `fig7b_pair_gap.png`, `fig12b_marked_divergence_signed.png`, `fig13_widening_did.png` | Result 3 is SUPERSEDED ON THE SIGN by M01 Registration T finding 18, `../../M01_displacement/findings/T_category_flow.md` (edge-unit DiD, 788 twins, 50-51 edges: the direction flips to divergence, RID:aggression null at q .43), as the document itself records. Internal correction: an earlier widening read off the fig12b base-dot overlay is withdrawn. Consistent with T-13; does not overturn D2. |
| `findings/D_word_trajectories.md` | Displacement watched word by word on the deployed cut, with the transgressive continuation crushed at the SFT boundary as its substitute rises; R2 gives the poetic arc (pull learned early, seen through in mature pretraining, re-bound +88% under alignment against a flat floor), R3 makes the displacement axis scene- and lineage-specific, R4 finds Step(SFT, DPO) empty. | "STATUS: DRAFT, grade C — single lineage (OLMo-3), illustrative/descriptive; the per-prompt panels are examples, not a rate." | Checkpoint rung on ONE lineage (OLMo-3): the same 95-checkpoint ladder; 22 per-prompt panels with movers selected on Step(base, DPO) under CANONICAL; poetic pull measured over 20 binomial/rhyming/alliterative pairs. | `fig5_poetic_pull_floor.png`, `fig6_violence_liminal__the_soldier_raised_his_rifle_and.png`, `fig6_violence_liminal__she_was_so_angry_she_wanted_to.png`, plus the twenty further `fig6_*` panels referenced as a set | Adds a lineage axis to Registration X, whose "contiguity, not resemblance" was already withdrawn as scene-specific. Consistent with M05-A's event-in-SFT and with M01 finding U. |
| `findings/E_pythia_capacity.md` | The Pythia ladder resolves the four-way onset tie OLMo left-censored — packages (step 2000) before reasoning (3000) before reference (4000), discourse last on both ladders (80000 / stage1-32000, criterion-sensitive, ordinal claim only); the sub-1000 window acquires words before measurable capacities; the phrase-before-the-fact ordering is now a two-lab fact. | "STATUS: FIRST LOOK, grade C — one lineage (Pythia-6.9b), descriptive, instruments copied from the OLMo ladder so the two are read by one rule; no cross-seat audit." | Checkpoint rung on ONE lineage (Pythia-6.9b), a SEPARATE population never pooled with A-D: 155 checkpoints (154 pretraining, log-spaced early), `data/pythia_population.json` ([5430]); same 584-text battery by declaration; `data/pythia_curves.parquet`, 80,290 rows. | `fig14_pythia_capacity.png`; combined: `fig17_acquisition_tokens_pythia.png`, `fig18_acquisition_ladder_pythia.png` | Cross-ladder replication of A's R2 (discourse last; phrase before fact), resolving its left-censored tie. Extends [5430]'s words-per-cell result: the early window has words before capacities. Records two non-quotable artifacts (discourse half-max 128; poetic sign-onset 512 at n=2). |

| `findings/F_syntax_curve.md` | Syntax installs as an event: on Pythia the licit share spikes on junk, CRASHES to ~0.1-0.25 in the frequency-spam phase (steps 8-64), installs past 0.9 by ~1000-2000 — an order of magnitude before any capacity; on OLMo the whole drama sits below the first rung and the floor is FLAT through SFT/DPO/RLVR (alignment runs inside grammar); the shape survives two coder families (Jaccard 0.37 on sets, parallel curves). | "STATUS: DRAFT, grade C — two ladders but each ONE lineage; two coder families; no cross-seat audit." | Checkpoint rung, both ladders separately (95 OLMo + 155 Pythia); 584 prompts; 338,092 tagged pairs; two frozen licit-set artifacts; `data/m05_class_mass.parquet`. | `fig16_syntax_curve_olmo.png`, `fig16_syntax_curve_pythia.png`; combined: `fig17_*`, `fig18_*` | Discharges registered secondary 5. Records the apostrophe-unescape correction and the coder-pin move. Tier 3 (selection/meaning judgment) designed, not run. SUPERSEDED ON THAT LAST CLAUSE by `findings/G_sense_curve.md`: tier 3 has now run. |
| `findings/G_sense_curve.md` | Sense installs with syntax and keeps buying: natural share 0 -> 47% inside Pythia's first 128 steps (the syntax-onset rung), then climbs to 92% over the whole rest of pretraining; THE COLORLESS-GREEN PHASE DOES NOT OCCUR (odd band <= 3.7% of mass at every rung, both ladders — mass moves ungrammatical -> natural directly); and alignment RAISES the natural share where it never moved the licit share (paired per-prompt +0.4-0.6pp, ~360/584 up, Wilcoxon p ~ 1e-6, monotone SFT -> DPO -> RLVR). | "Status: the tier-3 curve, on a census not a sample; one coder family (pinned), instrument validated by pilot, tie-break, and 10/10 canaries." | Census: 136,036 (prompt, word) pairs (118,129 judged + 16,624 auto-ungrammatical + 1,283 format), two declared floors incl. early-window top-up; mass join 552,061 rows, 95 + 155 checkpoints, zero gaps, unclassified tail 1.29% censored and drawn. | `fig19_sense_curve_{pythia,olmo}.png`; on `fig20_*`/`fig21_*` as "sense (natural share)" (fig17/fig18 = frozen pre-sense versions) | Completes F's tier 3. §2's null (no grammatical-nonsense phase) bounds what "syntax first" can claim: the saturation asymmetry, not a phase. §3 read WITH F: alignment runs inside grammar but not inside sense. Mechanism check (which bands lose alignment's removed mass) named, unrun. |
| `findings/H_norm_acquisition.md` | The norm signature is a two-module operation at checkpoint grain: SFT installs it (transgressiveness/bodily-harm/charge/concreteness DOWN, valence/register UP, paired per-prompt n=581), DPO buys concreteness back to a DEAD HEAT with base (+0.0457, 451/562 up, p 1e-49; NET base->DPO 285/286 p=1.00) while leaving register untouched, RLVR re-suppresses across the board (five scales p < 1e-25); NET base->RLVR restores the SFT signature. Pythia panel: composition differentiates from the function-word floor (concreteness 1.08 -> 2.87 from step 128) — pretraining builds the concreteness alignment later spends. | "Status: single registrar pass over the committed table... no new compute." NETs are PAIRED, never summed ([5732]). | (ladder, rung, prompt) cell; 145,741 cells over 249 checkpoints both ladders; `data/m05_norm_mass.parquet` + `results/norm_acquisition_increments.json`; choke point `word_probs` theta=0.001, residual measured per cell. COVERAGE FENCE: no composition number below Pythia step ~64 (instrument resolves <1% of mass at step <= 8). | none | The k_ riders travel with every number (ranks-not-levels; register_level descriptor-only; charge is not arousal; one coder; unrated tail censored). Norm-grain companion to B_field_flow's de-concretization and [5720]'s SFT-cuts/DPO-sorts dissociation; second-seat reproduction from the parquet is the standing upgrade path. |
| `findings/lens_ladder_instrument_note.md` | The lens ladder's one positive is about our method: the top-eighth concentration of the SFT contrast is HEAD-DEPENDENT (0.473 under a frozen base head against 0.265 under a frozen DPO head, 17/21 groups, p 0.0072) while its magnitude is identical under both -- so a depth signature that moves with the readout is not by itself evidence about where the computation changed. Everything else is null, a trend at noise scale, or a bound: step 0 reads ~0.90 at every depth, indistinguishable from a trained model. | "Status: one positive finding, and it is about our method." | Group is the unit (21 en contradiction groups). 95 rungs, 33 layers, two frozen heads, 127,744 rows after the degeneracy guard. ONE lineage (Olmo-3). | none | Carries a same-day CORRECTION: `step` is not a key (OLMo restarts numbering each stage; 9 step values collide, 52%% of base_step rows), so the pretraining trajectory was pooling three checkpoints per rung. sft_step/rlvr_step unaffected, so the positive stands. Reaches the same caution as `M02_frame_exit/findings/depth_and_exit_do_not_join.md` from the other side. |
| `findings/pole_sep_is_not_about_poles.md` | The cross-group null A-R4 records as owed is run and the pole-separation arc is NOT pole-specific: pairing pole_a of group X with pole_a of group Y reproduces the same collapse-and-recovery on both lineages. Separately Pythia's eleven sub-1000 rungs put the floor at step 256 (21/21 groups, p 9.5e-07), where OLMo's first non-zero rung IS step 1000 -- so OLMo's one-segment collapse was measuring the recovery. | "Status: a NULL that discharges an owed debt, plus one positive dating result." | Group is the unit (21 en f11 groups with both poles). Pythia ladder 154 rungs / 106,722 rows; cross-group null 13 checkpoints / 99,099 rows. Both lineages, never pooled. | none | Discharges the null A-R4 books as owed; A-R4's "not quotable in any form" fence should become a strike, which is registrar's amendment to make. Level comparison NOT usable: the null is unmatched on lexical distance (within-group Jaccard 0.750 vs cross-group 0.273); only the shape comparison is used. |

Owed, because the documents are one lineage per ladder: the 46-lineage
convergence DiD is DISCHARGED at the edge unit as M01 T finding 18; the
syntax curve is DISCHARGED as finding F (2026-08-11); remaining owed: the
join check (which base revision SFT descends from). The Pythia second fleet
LANDED 2026-08-11 ([5430], malign's run): its population is separate by
declaration (`data/pythia_population.json`, never pooled with M05), and its
first read QUALIFIES A's Result 3 — the true zero is OLMo's initialisation,
not general.

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
