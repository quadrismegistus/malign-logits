# Plot debt across the five meta-experiments

Written 2026-08-11 by the registrar seat. Produced by five parallel readers,
one per meta folder, each reading every document in its `findings/` plus the
folder README, `figures/` and `results/`; consolidated here. Companion to the
`## Findings index` sections added to each `meta/M##/README.md` the same day.

Two kinds of entry, kept apart: **inventory** (what has a figure, what exists
only as numbers-in-prose or data-on-disk — checkable facts) and **candidates**
(what looks figure-worthy — reader judgment, flagged as such throughout). The
finding documents govern their own numbers; nothing here upgrades a grade or
resolves a tension.

REGIME CHANGE (RH, 2026-08-14): this is now a RUNNING FIGURE TODO,
maintained by the registrar — the original "consumed, not maintained"
rule is retired. The RUNNING TODO section below is the live queue;
per-folder candidate lists remain the backlog; statuses are dated.

## The shape of the debt (refreshed 2026-08-14)

| Folder | Finding docs | Figures | Note |
|---|---|---|---|
| M01_displacement | 21 | 38 | 36 are K's exploratory norm-pair biplots; shortlist items still undrawn |
| M02_frame_exit | 11 | 1 (`z_depth_exit_null`) | results/ largely plot-ready |
| M03_proceduralization | 4 | 4 (all in D, `f_figures.py`) | A, B_C, E entirely unplotted |
| M04_syntagmatic | 2 | 14 (Finding A family) | produce-first discharge DISPUTED ([5901]: cited figures are PASSAGE substrate vs FC-quantity debt items; verify before relying); attention two-panel still undrawn |
| M05_emergence | 10 | 66 | fig19-23 sense family added; A-R4 joined panel still undrawn |
| M06_generation | 3 | 3 (plan-C density pilots) | verdict-grade findings entirely unplotted |

81 figure files landed 08-11 -> 08-14, essentially none from the
cross-folder shortlist: seats drew producer-adjacent diagnostics, not
the paper-facing figures.

## RUNNING TODO (live queue; registrar maintains; dated statuses)

Priority order, RH-adjustable. "open" = undrawn and unblocked.

1. SHIPPED 2026-08-14 (dario, 709dfdf1) — M01 N cluster dot plot:
   `plot_n_figs.py` registry, n_clusters_dotplot (paper-facing; one
   mark per cluster, count as AREA never position, x held open to the
   0.5 null) + n_z_ceiling_method (internal). Booked numbers asserted
   and matched to the digit. Drawing it REDISCOVERED [4134]'s ppf
   saturation (33/34 clusters at the identical capped z) — the
   register held it, the 08-12 doc rewrite had dropped it, N doc
   restored same day.
2. OPEN — M02 calibration number line (shortlist 4): 0.000 / 0.907 /
   1.006 / 4.031, per-cell strip behind, zh twin panel.
3. OPEN — NEW (2026-08-13): S4 DIAGONAL — each model soothed by its own
   promoted vocabulary; 2x2 role x word-class with one significant cell
   per arm ([5795]/[5825]; self_surprisal_cells.parquet). The surviving
   alignment-specific result of the forced-word program.
4. OPEN — NEW (2026-08-13); UPDATED 2026-08-14 (lacan) — F15-ON-PASSAGES
   QUADRANT FLOW — Q2 breakdown drains to metonymic/unmarked, 34/38 pairs
   (f15_on_passages_cells.parquet with per-passage quadrant labels).
   Natural flow/alluvial diagram.
   **STILL DRAWABLE, WITH ONE THING THAT MUST GO ON THE FIGURE.** Reran on
   bge-m3 with NO 75-word truncation (`_full_bge-m3`, 38,577 passages): all
   four flows replicate and strengthen — Q1 +0.114 (31/7), Q2 -0.335 (2/36),
   Q3 -0.123 (6/32), Q4 +0.299 (35/3). So the flow is robust to embedder and
   to truncation, which the earlier MiniLM-only version could not establish.
   **But the two axes do not carry equal information.** The Q1 share change
   tracks the surprisal axis at rho -0.694 and the drift axis at +0.167
   (MiniLM truncated: -0.804 and +0.011). The drift axis went from inert to
   minor, not to load-bearing. A two-axis diagram invites the reader to
   assume parity, so the ~4:1 asymmetry belongs ON the panel, not in a
   caption. Prefer the newer `_full_bge-m3` cells as the basis.
   **Owed before it is final**: the quadrant splits on `total_drift`, the
   metric the audit found weakest and which rises +0.10 with sentence count,
   saturating near n=10 — and untruncated passages sit past that knee. A
   `mean_drift` quadrant is the obvious variant and is cheap now the
   embeddings are cached.
5. OPEN — NEW (2026-08-13): CROSS-LINGUAL INVARIANCE — paired zh/en
   drift reductions + null DiD on matched prompts
   (crosslingual_arms_pairs.parquet, parse-free key version).
6. OPEN — NEW (2026-08-13): PROPAGATION SLOPE — the ~1.3%/99% line;
   per-pair slopes both roles, aligned-minus-base null
   (propagation_cells.parquet at 22aee418, post-repair).
7. OPEN — NEW (2026-08-13): FINDINGS H STAGE PLOT — SFT installs / DPO
   rebounds concreteness to dead heat / RLVR re-suppresses; plus the
   Pythia differentiation curve (concreteness 1.08 -> 2.87 from step
   128, coverage fence shaded). data/m05_norm_mass.parquet.
8. SHIPPED 2026-08-14 — M01 T-14 family, three figures via
   `scripts/plot_t_figs.py` (registry: t14, t14_dumbbell, t14_words):
   the field-level slopegraph with lines as ACTUAL displacement routes
   (direction_edgeunit flows) is the version of record; the lexicon
   dumbbell retained; and `t14_words_flows.png` puts kill->scream
   itself on the page (violence stratum, 245/21) with the whispered
   sink beside it. Booked-number asserts, slices in subtitles,
   truncations stated.
9. OPEN — T-18 x M05-C sign-disagreement dumbbell (shortlist 2).
10. OPEN — M05 A-R4 joined ratio/pole-sep panel (shortlist 3).
11. OPEN — M03 E survivor scatter (shortlist 5) + remaining shortlist
    items 6-8, 10 (X §3g, Y_diegetic four-panel, B_C lineage dots,
    M02 dumbbell) — unblocked, order per original ranking.
12. HALF-BLOCKED — M04 attention §5 two-panel (shortlist 9): Finding
    A's half now has artifacts AND figures (A_position_*, 2026-08-13);
    the combined attention-vs-A panel remains undrawn. CAUTION
    ([5901]): A_position_* figures are the PASSAGE-corpus substrate
    (A_RESULTS.md, STATUS: EXPLORATORY, nothing quotable) — Finding A
    proper is fc/33-pair/10-token and has ZERO figures; do not treat
    the 14 passage figures as covering the FC debt.
13. SHIPPED 2026-08-14 — displacement network viz, first pieces:
    `displacement_network_core.dot/svg` (135-edge working map, maps/
    idiom, basin clusters) and `displacement_basin_procedure` panel
    (52 edges, chains visible) via `plot_displacement_network.py`;
    remaining basins (epistemic/expression/stasis) are one command
    each; chain-exhibit strip (fired->aimed->pointed, kill->shout->hum)
    still owed.
13c. PARKED 2026-08-14 (RH: store and move on) — NETWORK
    VISUALISATIONS ON DISPLACEMENT PAIRS, the consolidated queue:
    (a) three remaining basin panels (epistemic/expression/stasis),
    one `plot_displacement_network.py <basin>` command each;
    (b) chain-exhibit strip — NB fired->aimed->pointed is FRAME
    taxonomy (certified, co-rising), not displacement-coupled, and
    kill->shout->hum dies at shout->hum under the verb restriction:
    quote accordingly; (c) VERB-NETWORK redraws — core map + basin
    panels on pair_cascade_replicated_verbs.parquet (795 coupled
    pairs / 419 words; reach 29/0, understand 19/1, show, prepare;
    lifts compress ~half, kill->scream 2.2x survives); (d) mutual-best
    couples table; (e) graph-structure pass (components, depth,
    condensation). Data all committed; producer takes a --verbs
    source flag as the natural next step.
13a. SHIPPED 2026-08-14 — X.1 GARMENT-LAYERS INFOGRAPHIC (RH's design,
    Opus-agent craft): `x1_garment_layers.svg/png` — stick figures with
    garment layers colored red(base)/blue(aligned) by median delta at
    the two verbatim X.1 prompts; data producer `x1_garment_deltas.py`
    (declared 46; X.1's registered k>=2 counts remain the registered
    form; gradient REPLICATES on the independent population; socks-her
    reversal + hat count-vs-median tension named).
13b. NEW (2026-08-14, pair-cascade instrument, plan_pair_cascade.md):
    the HUM SINK (sing/shake/tremble/weep->hum — expressive
    de-intensification, split-half certified); the PROCEED cluster
    (formalization axis at word grain); the TAXONOMY figure (1,851
    displacement-coupled vs 20,130 frame pairs — the rarity of true
    coupling as an image). t14_words is HELD: it quotes the retired
    binomial's numbers and awaits the T amendment on the new
    instrument.
14. PART-SHIPPED 2026-08-14 (registrar, first-read session with RH) —
    VERSE FLEET figures. SHIPPED: fig24 (rhyme capacity x era, full
    OLMo ladder — the SFT step-down), fig25 (x scheme + compulsion
    curve), fig28/29 (error types: miss vs false alarm at declared
    m=0.05, verse_error_rates.parquet), fig26/27 (ALL-CAPACITIES
    overview, both ladders: battery + poetic + syntax + sense + verse
    hit rate; segment-aware moving averages, right-edge labels; verse
    restraint deliberately NOT on the overview — vacuous pre-capacity,
    lives with its pair on fig28/29). Producers:
    M05 verse_capacity.py (tables), verse_capacity_figs.py,
    m05_capacities_overview.py, aggregate_capacities.py
    (capacities_by_rung.parquet — ONE tidy per-rung table, all
    capacity families; syntax recomputed from class_mass since its
    curve was never persisted; OLMo sense unjoinable, declared).
    STILL OWED from the original menu: nine-slot within-poem time
    course; closure gradient (BLOCKED: rider rides the un-ingested
    .f16 tier, RH's call); era x scheme matrix panel.
15. OPEN (2026-08-14, RH-approved menu) — FINDINGS P FIGURES, for a
    new `plot_p_figs.py` registry (P currently has NO figures of
    record; the 36 K biplots are the superseded measurement study's):
    (1) HEADROOM LADDER — share of the +0.121 word-level ceiling each
    instrument recovers: norms 7%, bge 17%, GloVe 18-21% (quote the
    BAND, never 21%), site-delta 68-82% (flag: contains word
    identity), with the 87% within-word variance shaded as unreachable
    by any word-level feature. Numbers scattered across k_ceiling /
    k_predict_embed / k_delta_predict outputs — assert against booked
    values, t14_dumbbell-style. P's thesis as one image.
    (2) NAMED-COMPONENTS DECOMPOSITION BAR — §7 ledger: register ~half
    (frequency-residualised row welded on per [5606]), concreteness ~a
    quarter, length zero, coder register zero, unnamed majority as the
    dominant empty region; pairs with 7b interiority ~quarter.
    Candidate two-panel with (1).
    (3) FIELD-POLES DIVERGING BARS — §7b z-scores (contact +8.77,
    matter, consumption, motion fall; communication, inquiry,
    cognition x2, perception, evaluation rise), PERCEPTION WEDGE
    highlighted (concrete-falls story cannot produce it); caption
    carries 109/448 surviving FDR. The interiority evidence.
    (4) PER-WORD ARM-AUC DISTRIBUTION — histogram of 4,106 features
    (results/k/word_auc_en.tsv), 22% clearing |0.15| shaded, tails
    labeled BY CLASS (deixis/bodily past-tense vs
    institutional-procedural infinitives), never as "the diagnostic
    words" (§3c sampling-noise fence).
    BARRED BY THE DOC'S OWN FENCES: single pole word-lists as the
    finding; POS-ordering bars (reverses under prompt mix); zh field
    grain alone (0/219 FDR); en/zh geometry across encoders. Held:
    four-instrument pole exhibit (drawable but caption-treacherous).

## STATUS CHANGES (2026-08-13/14)

- M05 C-R4 recapture bars: BLOCKED -> DEAD. R4 withdrawn ([5781]) —
  the domain contrast does not reproduce; nothing to draw.
- OPENING-MATCHED FAMILY: UNPLOTTABLE — withdrawn at construction
  level ([5811]); nothing from it may be drawn in any form. Its
  arm-ORDERING was additionally never established ([5805]) — no
  ordering bar chart even from surviving arm-vs-arm quantities without
  the paired test drawn beside it.
- LADDER (M04) figures: quarantine lifted ([5832]) — draw only from
  the REPAIRED producer values (two producers now agree exactly);
  magnitudes pre-repair are dead.
- M04 produce-first (items 2-4 of the M04 folder list): DISCHARGED —
  A_position_figures shipped with sentence-aligned indexing.

## Producer, artifact and write-up debt

MOVED 2026-08-12 to `meta/producer-debt.md` per [5554] (RH's reading: none of
it was plot debt, and severity runs opposite to this filename). This file
now owns figures only.

## Cross-folder shortlist (registrar's synthesis)

The candidates most likely to carry weight in the paper or book, drawn from
the per-folder lists below. Judgment, not doctrine.

1. **M01 T-14** — few large fallers against many small risers, on the
   BONFERRONI-SURVIVOR slice (declared: ALL/non-TOKEN): 206/36 at 3.79x,
   consistency-filtered so NOT identity-forced (the morning's retirement
   was corrected same day — RH's challenge; survivor sums do not zero).
   Dumbbell per lexicon as originally proposed, with the slice in the
   caption and the count panel beside it. `s_everything_marginal.csv`.
2. **M01 T-18 beside M05-C** — the affect DiD at both units: one row per
   declared field, one-lineage DiD left, edge-unit DiD right, sign
   disagreements highlighted (RID:aggression the anchor).
   `results/t_affect_did.csv` + `data/m05_widening_null.json`. Renders "the
   gap is real, its sign is not robust" as one image instead of a retraction
   paragraph.
3. **M05 A-R4** — the joined ratio/pole-separation two-panel across the
   95-rung ladder, stage boundaries marked. `data/m05_ratio.parquet` +
   `results/m05_pole_sep.csv`. Discharges the plan's "together or not at all".
4. **M02 the calibration number line** — perfect superposition 0.000,
   observed 0.907, NEUTRALIZATION 1.006, RESOLUTION 4.031, per-cell strip
   behind, log-spaced; zh as a twin panel. `results/contradiction_null_en.csv`
   / `_zh.csv`. "1.0 is not a boundary, it is a place" is spatial; show it.
5. **M03 E §3/§4** — 324 verbs on indiv-vs-inst axes with the y=x diagonal,
   65 Bonferroni survivors labelled, four reversals coloured.
   `results/b_word_delta_by_word.csv`. "Degree, not kind" read off geometry.
6. **M01 X §3g** — the word moves the scene (+14.3 points, 12/12 cells), the
   model does not (−0.8, p .918), two panels, same axes.
   `results/x_beam_frame.csv`.
7. **M01 Y_diegetic §3** — the conditional four-panel: CLEAN_SCENE −6.12pp,
   SUPEREGO_IN_SCENE +4.30pp, EXIT and sexual_scene flat.
   `results/y_passages.parquet` via `scripts/y_diegetic.py`. The filter
   account predicting exactly the two panels that do not move.
8. **M03 B_C §1** — the JS arm effect, one row per lineage, 41/46 above zero,
   five dissenters labelled. `results/b_arm_by_lineage.csv`.
9. **M04 attention §5** — attention-back decay above, Finding A's surprisal
   sweep below, both in both aggregations (disjoint bins beside cumulative).
   `results/attn_delta_smollm2_e1_cross_w200.json`. The figure IS the
   two-phenomena argument. (Finding A's half needs the produce-first step.)
10. **M02 dumbbell** — D_CONTRA sitting on top of D_CONTROL, one row per
    field, 39/79 general survivors emphasised, 0/79 specific.
    `results/l2_fields_{meta,norms,usas_fine}.json`. The doc's own warning
    (a residual-only report would have filed a positive as a null) as a
    picture.

Methodological pair worth drawing for the book's methods spine: **M03 D §6**
(ICC 0.855 spaghetti — why 50 rungs are not 50 observations;
`results/d_ladder.csv`) and **M01 U-1/2** (the ladder slope chart, removal
stopping while addition continues; `results/t_ladder_steps.csv`).

## Per-folder candidate lists

Condensed from the readers' reports: doc / result / data / suggested form.
Ordering within each folder is the reader's ranking.

### M01_displacement (reader's tier 1 and 2; tier 3 all blocked, see above)

1. T-14 fallers/risers dumbbell — `s_everything_marginal.csv` (shortlist 1).
2. T-18 affect DiD paired slope, arrows coloured widens/converges —
   `t_affect_did.csv` (shortlist 2).
3. X §3g two-panel — `x_beam_frame.csv` (shortlist 6).
4. Y_diegetic §3 conditional panel — `y_passages.parquet` (shortlist 7).
5. U-1/2 ladder: JS by rung + faller share 49.3/28.6/1.0 collapse —
   `t_ladder_steps.csv`, slope chart, one grey line per family.
6. S-3 harm gradient by domain, violence −0.290 to taboo −0.033 —
   `s_analysis_effects.csv`, dot-and-CI; caption must carry the
   `sexual`→`coercion` relabel from DISPLACEMENT_EVIDENCE.
7. X §3d body-part classes at `suck his ___` (genitals −2.6 … digits +4.3) —
   `x_bodypart_classes.json`, labelled beeswarm by class.
8. C_to_O/M eviction gradient across headroom deciles, 0.157→0.003, zero
   above the fifth — `result_m_column.json`, step chart, zero region shaded.
9. T-12 USAS 45 surviving fields — `s_everything_marginal.csv`, lollipop.
10. T-11 category-by-stratum heatmap, 33 risers / 10 fallers, no reversal —
    `s_marginal_strata.csv`; report stratified, never pooled.
11. T-8 bodily_violence→speech_act by edge, 24/0/1 of 25 —
    `s_everything_direction_edgeunit.csv`, diverging bar.
12. U-4 + X §3e two-panel fan (JS uniform; faller Jaccard splits no-wildchat
    0.340 vs 0.52–0.53) — `t_fans.csv`, `t_fans_jaccard.csv`.
13. V-5 scene-locality, twin 0.327 vs random 0.060, 14/14 families —
    `v_displacement_twin*.csv`, paired dots, raw and residualised.
14. X §3f violence discriminant control, per-prompt rho spread, pooled
    −0.100 — `x_violence_pooled.csv`. The anti-tautology figure.
15. Y_superego §4 heterogeneity, AmberSafe +15.4pp vs median +0.8pp —
    `y_passages.parquet`, sorted per-pair dots, tails named.
16. Y_superego §7.4 de-vulgarisation without de-intensification —
    `y_passages.parquet` + `y_tokens/`, diverging lollipop by valence.
17. H2 per-pair median d + L50/N spread 0.000–0.861 — needs `--json` written
    first (produce-first list), sorted dots, the one reversal named.
18. T-7 concreteness: both tails draining to the middle — `s_concreteness.csv`,
    before/after density. What a difference of means cannot show.
19. T-5/S-8 sink structure, `whispered` 50-in/0-out vs pure sources —
    `s_condensation.csv`, in-degree vs out-degree scatter.
20. N: 91% of 82,775 cells, 34/34 clusters — `result_n_primary.json`,
    cluster dot plot. The campaign's anchor result has no picture.

### M02_frame_exit

1. Calibration number line (shortlist 4).
2. Next-word three-role word dumbbells (`kill` −9.9/−1.3/−21.3; the epistemic
   residual) — `dp.pkl`, REBUILD REQUIRED first.
3. t and resid by role, paired violins, resid=1.0 marked —
   `l3_geometry_union.parquet`. "Same shadow, equally off-axis" needs both
   panels at once.
4. Pole separation vs superposition loss, rho −0.420, n=45 —
   `polesep_vs_superposition.csv`. Cheapest strong figure in the folder.
5. Non-universality slope plot, base→aligned per lineage, 12/46 reversers,
   AmberSafe −0.1392 named — `contradiction_null_by_pair_en.csv` + `_zh`.
6. D_CONTRA-on-D_CONTROL dumbbell (shortlist 10).
7. Falcon3 concentration lollipop, 52.76% vs 2.01% ceiling, Mamba rows
   labelled to kill the vendor reading — `eassist_ambient.csv`.
8. t(both) by depth, one line per role, base/aligned solid/dashed —
   `l3_geometry_union.parquet`.
9. Nobody-near-a-pole histogram, poles at ±0.45–0.48, observed −0.12–+0.15 —
   `dp.pkl` (rebuild). Forecloses the pole-migration misreading of F11.
10. E-QA by domain at the twins — `exit_markers_fc_bypair.csv`. INTERNAL
    ONLY until the coded pass; FIRST LOOK status, a figure makes a number
    quotable.
11. Forced-arm null small multiples beside the surviving scene contrast —
    `exit_forced_bysite.csv`. The null that licenses "scene, not signifier".
12. Lens depth curve — WITH its write-up, not before (see write-up debt).
13. L1 ROC panel, two instruments off the diagonal, five constructions on
    it — `l1_*.json` family. WITH its write-up.

### M03_proceduralization

1. E §3/§4 survivor scatter (shortlist 5).
2. E §4 dumbbell, 65 rows, indiv vs inst dots joined, coloured by pattern —
   `b_word_delta_by_word.csv`. The words the argument quotes.
3. B_C §1 lineage dot plot (shortlist 8).
4. B_C §2 `should`-confound seven-condition interval plot —
   `b_arm_cells.csv`. "Triples with a prompt-final modal AND survives
   without one" as a two-group comparison.
5. D §1 selection-not-construction small multiples: six verbs across 95
   rungs, each verb's pretraining maximum as a reference line —
   `f_verb_rungs.csv`. Nothing yet shows the ceiling.
6. D §6 ICC spaghetti + scenario-unit marginal strip — `d_ladder.csv`
   (methodological pair, above).
7. D §7 cross-corpus non-transfer, three scatter panels over 95 shared
   fields, 0.063 against 0.701 — `e_field_flow_arms.parquet` + root
   `data/m05_field_flow_fine.parquet`.
8. A §4 hedge-vs-position paired slopes, +0.207 against +0.077 —
   `meta/M01_displacement/results/x_m03_kernel_decomp.csv`. Caption must say
   EXPLORATORY, uncorrected.
9. B_C §6 reference-class dot plot, eight domains, pair count printed ON the
   mark — `c_pair_contrast.csv`.

### M04_syntagmatic (Finding A items all blocked on produce-first)

1. Attention §5 two-panel decay comparison (shortlist 9).
2. A position profile, point-with-CI across indices 1–10, zero rule —
   BLOCKED (channel3 re-run + dump).
3. A term × index grid, five panels or signed heatmap — BLOCKED (same).
   The most load-bearing, least legible passage in the folder.
4. A long-window sweep — BLOCKED (same), and the figure MUST show disjoint
   bins beside the cumulative curve (attention doc §5's objection) or it
   hardens a contested artifact into a picture.
5. A pair-level strip, 33 pairs, bigscience outlier labelled, ALL vs CLEAN
   medians — BLOCKED (same). The honest-limits figure.
6. Attention §3e per-pair baseline strips, pooled null vs KW p 0.0019 —
   `attn_norm_sweep_full.json`. The "pooling cancels opposing signs" lesson.
7. Attention §3c refutation strips, three contrasts over 28 cells —
   `attn_norm_sweep{,_full}.json`. A null prose cannot make credible.
8. Attention §3 cross-vs-own two lines against position —
   `attn_delta_smollm2_e1_{cross,own}.json`. The one surviving fact.
9. Attention §6 head-concentration Lorenz curve, 7x not the plan's 17x —
   `attn_undist_*.json`. Doubles as the visual correction to the plan.

### M05_emergence

Orphaned figures first: 11 on-disk figures no finding cites, including the
entire fig11 family (`m05_field_flow_marked.py` has no representation in any
finding) and `fig7c_pair_by_domain.png`, which is exactly the by-domain split
C Result 1 states in prose. Adopt or retire them doc by doc; `fig12` (the
unsigned predecessor of fig12b) stays uncited deliberately — C's correction
section withdrew that read.

1. A-R4 joined ratio/pole-sep panel (shortlist 3; discharges the plan).
2. C-R3 vs T-18 sign-disagreement dumbbell (shortlist 2).
3. A-R1 paired per-site onset distribution, lag histogram centred on zero,
   never-fall/never-rise flanking bars — `m05_onsets.json` `paired`. Makes
   p=.97 legible; the result that kills F04's lag.
4. C-R4 displace-vs-refuse recapture bars by domain, reference line at 1 —
   BLOCKED on the recapture provenance debt (above).
5. A-R3 the true zero: log-scale resolved mass at stage1-step0 against
   theta, 257 complete-but-empty cells marked; zh twin panel —
   `m05_curves.parquet`. QUALIFIED by [5430]: the zero is OLMo's
   initialisation, not general (Pythia step0 resolves ~5 words). The
   stronger figure is now the two-lineage contrast — OLMo's nothing beside
   Pythia's small-but-present floor and its eight-fold rise by step 128 —
   drawn WITHOUT pooling (separate populations by declaration).
6. A-R2 acquisition ordering as a log-step lollipop strip: four families at
   stage1-2000, discourse tracking alone at 32000 — `m05_onsets.json`
   `base_order`.
7. D-R4 three stacked mover lists, Step(base,SFT) / Step(base,DPO) /
   Step(SFT,DPO), the third deliberately empty — from
   `m05_word_trajectories.py`'s Step objects. The empty panel is the
   argument.
8. B fine-field rank figure with intervals — BLOCKED until the per-field
   ordering null exists; drawing it now would quote a rank B says is not
   quotable.

## Fences: do-not-plot and caption-must-carry

Collected from the documents' own limits; a figure that pools a fenced
stratification launders a fence the prose spent paragraphs erecting.

- M01 T-11 reports stratified, never pooled; T-2's GI table must not be
  plotted alongside findings 11–16; X is not poolable with the M01 battery;
  T §17c is exploratory.
- M01 S-3's `sexual` domain label must read `coercion`/`boundary` on any
  figure (DISPLACEMENT_EVIDENCE).
- M03: do not plot `c_word_delta_by_word.csv` as institutional-vs-narrative
  vocabulary — B_C §6 and D §7 record that axis as form-confounded. Any RID
  row carries its coverage numbers (0.400/0.429) on the figure itself.
- M03: `d_verb_rungs.csv` / `d_button_verbs.csv` look like superseded
  pre-`f_figures.py` caches of the `f_*` pair — confirm before plotting
  from them.
- M04: nothing at the per-head unit (250–480 correlated heads; [5226]); the
  binding unit is the cell or the pair. Do not re-dignify the retracted
  two-cell SmolLM2-vs-OLMo contrast.
- M02: no figure from `exit_contradiction_cells.csv` (a 3-generations-per-
  cell sample, per TODO.md — the corpus is 228,520 passages in ClickHouse).
  Exit-marker figures stay internal until the coded pass.
- M05: `fig12` stays retired; the B rank figure is blocked on its null.

## M06 mediation candidates (added 2026-08-14 by lacan)

From `findings/composition_not_level.md` (bab9d228 .. b7b50fea). M06's shape row
says "verdict-grade findings entirely unplotted"; this finding is one of them.
Ranked by what each figure is *for*, not by how nice it looks. All artifacts
exist; nothing here needs a new run.

1. OPEN — **COMPOSITION/LEVEL BAR**, the headline. Delta -1.285 nats/word
   splitting into composition -1.310 and level +0.025, with BOTH decomposition
   orders drawn beside the symmetric mean (order 1 -1.775/+0.490, order 2
   -0.845/-0.440) so the entanglement is visible rather than hidden by the
   average. Per-relation panels (base_to_superego -1.583 n=14, dpo -1.005 n=20,
   ppo -1.997 n=2). `results/mediation_pairs.parquet`. This is the picture the
   "selection not combination" claim rests on, and the two-orders requirement is
   not decoration: a single order would be an undeclared choice.

2. OPEN — **OFF-POLICY SHIFT BY CLASS**, the figure that PROTECTS the claim.
   Mean cross-scorer level per movement class with bar width as token share:
   unmeasured +0.614 at 71.7% of tokens, still +0.257 at 17.3%, fall +0.381 at
   7.6%, rise +0.144 at 3.5%. It shows at a glance that fallers sit BELOW the
   corpus average, so "displaced words cost more in context" cannot be read as
   the finding. That misreading cost three docket corrections on 08-13/14
   ([5881] -> [5884]); this figure is the cheapest guard against a reader
   repeating it. `mediation_words_byprompt.parquet` joined to `movement`.

3. OPEN — **THE DISSOCIATION, TWO PANELS.** net_fall vs composition change
   (rho -0.285, 36/36 pairs negative) beside pct_moved vs composition (+0.008,
   18/36 = exact chance). The NULL panel is the content: a flat cloud next to a
   sloped one, with `the` labelled in the flat one (30.5% of its 1,575 Llama
   cells non-still, direction -3.3%). Direction predicts, volatility does not.
   `results/mediation_corr_words.parquet`.

4. OPEN — **MATCHED-PROBABILITY DUMBBELL.** Per-pair median(level|fall) against
   median(level|rise) inside the common-support band (log p_aligned
   -2.464..-1.465): +0.347 base-generated 34/35 pairs, +0.444 aligned-generated
   35/35. The surviving claim in its strongest form. Same artifact, band
   restricted.

Backlog rather than queue:

- **Common-support diagnostic** — fall/rise counts by log p_aligned bin, showing
  45,901 fall cells in bins holding ZERO risers. A METHOD figure: it is why the
  partial correlations were demoted in favour of the banded contrast. Belongs in
  the finding, probably not paper-facing.
- **Consistency asymmetry** — fall-dominant 0.526 vs rise-dominant 0.613 median
  consistency, per-pair -0.036, 152 pairs. A real result attached to a REFUTED
  mechanism (demotion does not generalise more than promotion; it generalises
  less). It needs a home before it needs a picture, and its population is all
  152 movement pairs rather than the mediation's 36.

Fences that must travel onto any of these: single pass, one seat, ungraded;
CANONICAL's fallers are NOT null-tested; and the level axis is a CROSS-SCORER
difference on fixed text, not syntagmatic damage, since the chain is held
constant by construction.
