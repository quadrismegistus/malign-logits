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
| M04_syntagmatic | 2 | 14 (Finding A family) | produce-first discharge INVALID — CONFIRMED by malign from own folder ([5904], 21ba7a00): 14 figures are passage/42-pair EXPLORATORY substrate vs Finding A's fc/33-pair; 12 now substrate-stamped in the producer; FC debt items REOPEN; attention two-panel still undrawn |
| M05_emergence | 10 | 66 | fig19-23 sense family added; A-R4 joined panel still undrawn |
| M06_generation | 3 | 3 (plan-C density pilots) | verdict-grade findings entirely unplotted |

81 figure files landed 08-11 -> 08-14, essentially none from the
cross-folder shortlist: seats drew producer-adjacent diagnostics, not
the paper-facing figures.

## RUNNING TODO (live queue; registrar maintains; dated statuses)

> **NUMBERING NAMESPACE: `queue N`.** This document carries FIVE
> independently numbered lists — this queue, the cross-folder shortlist, and
> the per-folder candidate lists. **There are four different "item 1".**
> dario read the queue wrong three times on 2026-08-14 before establishing
> its state ([6177]), twice by matching a STATUS WORD where the thing wanted
> was an ENTRY'S STATE: `HELD`, `UNHELD` and `SHIPPED` all occur inside item
> 5, and its first line says HELD. **A first line is a name for an entry, not
> a relation to it.** The structure is legible to a reader and invisible to a
> predicate. Cite as `queue 12`, `shortlist 9`, `M05 candidate 3` — never a
> bare number. Shortlist numbers are NOT renumbered here because three
> entries already cite `shortlist 9`.

Priority order, RH-adjustable. "open" = undrawn and unblocked.

1. SHIPPED 2026-08-14 (dario, 709dfdf1) — M01 N cluster dot plot:
   `plot_n_figs.py` registry, n_clusters_dotplot (paper-facing; one
   mark per cluster, count as AREA never position, x held open to the
   0.5 null) + n_z_ceiling_method (internal). Booked numbers asserted
   and matched to the digit. Drawing it REDISCOVERED [4134]'s ppf
   saturation (33/34 clusters at the identical capped z) — the
   register held it, the 08-12 doc rewrite had dropped it, N doc
   restored same day.
2. SHIPPED 2026-08-14 (dario, a17788c8) — M02 calibration number
   line: contradiction_null_figs.py -> contradiction_null_numberline,
   en anchors asserted, zh twin (0.958/1.004/3.852 — UNASSERTED, not
   booked anywhere; booking debt noted [5912]). Three declared
   departures, all registrar-ratified [5913]: LINEAR axis anchored at
   0 overriding this entry's "log-spaced" suggestion (log cannot show
   the zero anchor and would push OBSERVED/NEUTRALIZATION apart —
   their 0.1 separation IS the finding); labels on opposite sides of
   the strip; zh tail cut at shared limit with count+max printed on
   panel. PROVISIONAL status on the panel.
3. SHIPPED 2026-08-14 (dario, e61776d3) — S4 DIAGONAL, and M06's
   FIRST figure producer (m06_self_surprisal_figs.py ->
   self_surprisal_diagonal). The 2x2's symmetric layout risk resisted
   ON the panel: S4 arm-specific at both grains (DiD pair p .0166),
   S3 NULL at the pair grain (DiD p .636, sig only at cell grain) —
   the mirror is one established effect beside one unestablished
   half, and the panel says so where a rerun cannot erase it.
   Reconstruction notes booked: pair grain = MEDIAN over prompts
   (mean inverts sign counts); sign test keeps zeros in median,
   excludes from counts — caught only because ALL SIX cells were
   asserted while four matched. Method rule ([5915]): artifacts-first
   for RECOVERY, read-the-producer-first for RE-DERIVATION. Finding
   amended in THREE places incl. the TITLE ([5917], 64dc3803): "each
   model soothed by its own promoted vocabulary" asserted both DiDs
   in the line most quoted without the table; now names the
   established half (S4) and the open one (S3). Definitional choices
   (median grain, zeros convention) now in the doc beside the
   numbers.
4. SHIPPED 2026-08-14 (dario, 0b7ab63c) — F15-ON-PASSAGES QUADRANT
   figure, THREE departures booked:
   (a) NOT AN ALLUVIAL — a REFUSAL, ratified: ribbons assert
   item-level migration, and base/aligned passages are different
   generations with no passage correspondence (3,361 of 35,216 keys
   even share a prompt slot). The suggested rendering would have
   asserted a false fact about the data. Figure keeps the quadrant
   plane, draws per-pair SHARE change within each quadrant.
   (b) mean_drift RIDER DISCHARGED without a run (column existed in
   the committed cells): all four flows replicate and SHARPEN on the
   pure surprisal transition (Q2 -0.335->-0.370, Q4 +0.299->+0.366)
   while mixed quadrants weaken — the audit's worry answered in the
   finding's favor. mean_drift quadrants NOT booked anywhere
   (script's own computation, single-pass).
   (c) RHO PAIR RECONCILED 2026-08-14 (lacan, [5924]) — IT IS THE
   AGGREGATION, not population or truncation. Over passages within
   (pair, role), aligned minus base, n=38, on the committed
   _full_bge-m3 cells:
       MEDIAN  surprisal -0.694  drift +0.167   <- quote these
       MEAN    surprisal -0.714  drift +0.211
   Both reproduce exactly; neither seat miscoded. **Quote the median**:
   it is the campaign convention for a pair grain (self_surprisal.md,
   where the mean inverts the sign counts), and it is independently
   right here because `total_drift` is an extreme statistic with a
   skewed passage distribution. The qualitative claim is identical
   either way — surprisal carries ~3-4x the association.
   **A bare coefficient pair without its aggregation is not
   reproducible**, and this entry proved it by failing to reproduce
   for the next seat that tried.
5. HELD 2026-08-14 ([5932], dario; NOT drawn) — CROSS-LINGUAL
   INVARIANCE. The leg this entry names as its basis — the parse-free
   matched-prompt key, which the finding calls "the one that travels"
   — HAS NO ARTIFACT AND NO PRODUCER. Persisted contrasts are exactly
   {total_drift, mean_drift} x {pooled, n_sents-matched} in both jsons
   and both pairs parquets, where `matched` = n_sents-matched, NOT
   prompt-matched; the named producer has no pair_role/parse/
   prompt_catalogue reference (search space in [5932]). Booked as
   producer-debt Class 1B; recovery assigned by RH to lacan; the
   register's QUOTABLE UPGRADE resting on this leg is SUSPENDED
([5934]).
   **UNHELD 2026-08-14 ([5935]): ROUTE 3, and the leg is WITHDRAWN
   rather than pending.** lacan exhausted routes 1 and 2 and withdrew
   both matched-prompt legs in the finding; the register form is
   withdrawn with them. Draw pooled + n_sents-matched ONLY, and per
   registrar's tightening (which lacan endorses over dario's own
   wording): the panel must not imply a matched-prompt comparison
   exists at all — naming an absent leg still puts it in the reader's
   head. Item 6 was taken meanwhile.
   **SHIPPED 2026-08-14 under route 3 (dario, f3c47a4a)**:
   m06_crosslingual_figs.py -> crosslingual_invariance, four panels
   over the persisted contrasts, both runs. DIAGONAL SCATTER, not
   paired bars: the surviving claim is an INVARIANCE, and a null is
   not shown by two bars that happen to look similar — English on x,
   Chinese on y, and the y=x line drawn as the place where "the same
   in both languages" lives, so the reader assesses the scatter
   instead of taking it. Both arms negative on all eight contrasts
   puts the cloud in the lower-left quadrant: two claims, one
   geometry. Asserts cover eight DiDs and sixteen arm medians PLUS
   TWO CLAIM-SHAPE GUARDS (every DiD still null, every arm median
   still negative) — the producer refuses rather than drawing a claim
   the finding no longer makes. No matched-prompt leg named anywhere
   on the panel; the account lives in the docstring, where an editor
   meets it and a reader does not. Axis noun is SPREAD, not drift
   (total_drift = 1 - min pairwise similarity, order-invariant; the
   finding corrected itself on this and an axis label is the easiest
   place for a retired noun to survive).
6. SHIPPED 2026-08-14 (dario, 932ca0c3) — PROPAGATION SLOPE:
   m06_propagation_figs.py -> propagation_slope. TWO SCALES of the
   same points, because the finding asserts both that the slope is
   reliably positive and that it is almost nothing, and one axis
   serves one claim while defeating the other. All four variant x
   role medians, sign counts, and both aligned-minus-base contrasts
   asserted, so the not-an-alignment-effect fence cannot go stale
   without the producer refusing. H3 is neither drawn NOR quoted (it
   comes from opening_matched — UNPLOTTABLE, see STATUS CHANGES); the
   registrar tightening at [5934] caught a first draft that omitted
   the reference from the geometry and then printed its numbers in
   the subtitle — A VALUE PRINTED IN A SUBTITLE IS ON THE FIGURE.
   Arithmetic note ([5936]): the finding's prose "~1.3%" follows from
   neither committed summary (medians 1.20% aligned / 1.05% base;
   means 1.67% / 1.18%); nothing turns on it (the claim is ~99%
   absorbed on any of them) and the panel quotes what it computes.
7. SHIPPED 2026-08-14 (dario, feaea8db) — FINDINGS H, two figures.
   **fig30 STAGE MATRIX: position is a SIGN SPLIT, not a median.**
   `vulgarity` carries 252-339 TIED prompts of 581, so its median is
   exactly 0.0000 in four of five transitions while its sign test
   reaches p 1.3e-25 — a median-coloured matrix would render the
   table's strongest tie-dominated result as "no change". Position =
   share of NON-TIED prompts moving up (what the finding's own sign
   test measures); mark AREA = the non-tied count, so ties shrink the
   mark instead of hiding. Non-significant cells drawn, not dropped —
   the dissociation reads off the panel (DPO reverses SFT's sign on
   four scales; register_level grey at 268/272 p 0.90; NET base->DPO
   concreteness grey at 285/286, the dead heat).
   **fig31 PYTHIA CURVE: the pre-fence region is DRAWN, greyed and
   dotted.** At step 0 median concreteness reads 2.65, ABOVE the 1.08
   floor it occupies once rated coverage reaches 1.0
   (k_rated_mass_share 0.63 there), so an unfenced curve opens with a
   collapse a reader takes for pretraining's largest early event.
   Drawing the fenced region is the demonstration that the fence is
   an instrument limit rather than conservatism.
   **Two grain choices, both caught by asserts:** medians not means
   (mean gives 1.23 vs booked 1.08, 3.04 vs 2.87) — third time this
   session the median/mean grain decided whether a reconstruction
   matched; and base_step rungs ONLY, because **the final rung carries
   the same model twice** (base_step + base_endpoint, 584 prompts
   each), so pooling averages the curve's last point with a duplicate
   of itself: 2.8654 -> booked 2.87 against 2.8648 -> 2.86, a
   fourth-decimal difference visible only at the finding's own quoted
   precision. Riders travel on the panel (register_level
   descriptor-only, construct NOT established; vulgarity sparse;
   CHARGE IS NOT AROUSAL); valence omitted from fig31 as median-pinned
   at the scale midpoint.
8. SHIPPED 2026-08-14 — M01 T-14 family, three figures via
   `scripts/plot_t_figs.py` (registry: t14, t14_dumbbell, t14_words):
   the field-level slopegraph with lines as ACTUAL displacement routes
   (direction_edgeunit flows) is the version of record; the lexicon
   dumbbell retained; and `t14_words_flows.png` puts kill->scream
   itself on the page (violence stratum, 245/21) with the whispered
   sink beside it. Booked-number asserts, slices in subtitles,
   truncations stated.
9. SHIPPED 2026-08-14 (dario, 0db7cbcd) — T-18 x M05-C sign
   disagreement. Landed INSIDE `plot_t_figs.py` (registrar's file)
   because its registry had reserved the slot in a comment; purely
   additive, one function + one entry, flagged not assumed —
   RATIFIED [5946]. DEPARTURE: ONE panel, not the shortlist's two.
   Two panels put the estimates in two coordinate systems and make
   the reader carry a sign across the gutter; on one axis the
   disagreement IS a segment crossing zero, and 6 of 16 do it.
   **What the entry did not know: the units disagree about
   SIGNIFICANCE more widely than about sign** — 13 of 16 fields clear
   q 0.05 at the edge unit against 6 of 16 on the lineage (RID
   aggression: edge +0.0002 q 0.43 sees nothing, lineage -0.0054
   q 0.007 a significant fall), so significance is drawn at both
   units, FILL mapped (not shape) so "filled clears q 0.05" is
   literally what the mark does. NOT framed as a replication failure:
   50 base-to-aligned edges against 105 pairs on ONE lineage, neither
   estimate the other's check — and that fence came off the ARTIFACT's
   own `_about` field, not off any document (see the `_about`
   convention, [5946]).
10. UNBLOCKED 2026-08-14 by RE-DECLARATION ([5965], lacan, `19240d87`;
    RH said go at [5958]). Plan `plan_pole_sep_reduction.md` committed
    ALONE at `570afad4` BEFORE the producer existed, so the ordering is
    checkable from git rather than asserted. THE RULE: **role is not a
    dimension of this measurement** — `pole_sep` is bit-identical
    across role (max difference exactly 0.0, asserted at run) but only
    50,193 cells carry all three roles against 15,675 carrying one, so
    pooling roles is a median WEIGHTED BY HOW MANY CONTROLS HAPPENED TO
    BE RUN for that cell; that single fact is the whole difference
    between dario's median-all and lacan's role==both and makes
    neither defensible. Then: median over 33 layers within (checkpoint,
    group), median over the common group set, medians throughout;
    stated in the artifact's own `_about`, six superseded values kept
    visible as superseded-not-reproduced. **THE REPUBLISHED NUMBERS ARE
    WEAKER THAN THE OLD ONES AT step16000 ON BOTH LADDERS** (0.3931
    against 0.475; 0.2486 against 0.384), consistently and in the same
    direction. And the plan's recorded prediction came out weaker too:
    co-movement is positive on both lineages and significant on
    NEITHER (rho +0.536 p 0.215 n=7; +0.771 p 0.072 n=6), so the
    finding's "the null recovers EXACTLY as the real column does" is
    corrected in place to what n=6 and n=7 can carry — **this cannot
    establish co-movement, only fail to contradict it** — which still
    serves the section's actual argument (the arc is not about poles).
    Reported, not re-reduced, exactly as the plan said it would be.
    `results/m05_pole_sep_reduced.csv` is one row per (ladder, column,
    checkpoint) with n_groups, correctly ordered; the panel is dario's
    whenever wanted, drawn as two curves and a rank correlation that
    does not clear significance — never as the word "exactly".
    **SHIPPED 2026-08-14 (dario, ee7d5b40) — the plan's plot-both-or-
    neither clause satisfied for the first time rather than honoured
    by absence, and the join exposes a structural fact neither series
    shows alone: ratio is measured at ALL 95 rungs and separation at
    SEVEN (ckpt 0, 1, 8, 22, 26, 29, 38), every one inside the first
    38 of 94.** The last 56 rungs carry a ratio with nothing to compare
    against; a dotted rule marks where coverage stops. That is the
    strongest available comment on the R4 co-movement claim — whatever
    relationship holds is asserted from seven points covering the first
    40% of the ladder — and the panel draws no correlation and asserts
    no coupling; the geometry says it. R4's own statistics stay Class
    1B and unquoted; what is quoted is lacan's re-declared co-movement
    WITH its limit (positive both lineages, significant on neither,
    n=7 and n=6). The finding's "exactly" is not restored. Own defect
    caught by looking: a first draft put the calibrated ratio and the
    separation distance on ONE unlabelled y axis because their ranges
    happen to overlap — **a coincidence of range standing in for a
    relationship** — now two panels, free y, shared x, which is what
    the clause was always about.
    ORIGINAL HOLD, for the record: HELD ([5954], dario; NOT drawn) — M05 A-R4 joined
    ratio/pole-sep panel, on TWO counts, both with lacan.
    (a) **R4's statistics have no producer** (booked Class 1B): the
    SFT-arm level co-movement (Spearman +0.61, p 1.3e-05), the
    uncoupled rung-to-rung changes (co-drift rho -0.12, p .45) and
    the no-lead result (sep-leads .085 / ratio-leads .17) appear in
    exactly one file in the tree — `A_acquisition.md`, which quotes
    them. Search space stated, control in the same call returned 3
    .py files, so the search works.
    (b) **The per-checkpoint `pole_sep` reduction is not determined.**
    `m05_pole_sep.csv` is 166,255 rows at (checkpoint, group, role,
    LAYER) grain; every published figure needs one number per
    checkpoint and nothing states how to get it. Median-all and
    mean-all both miss the booked values and stage1-step16000 is not
    close (0.3675 against 0.475). **dario stopped after ONE candidate,
    citing [5935]:** with three targets and eighteen plausible
    reductions it would likely find one that hits all three and it
    would be worth nothing. One line from lacan naming the reduction
    unblocks the item.
    WHY THIS IS WORTH FIXING RATHER THAN PARKING: the plan's clause is
    "the write-up plots ratio and pole_sep together or not at all",
    and `fig4_ratio_unjoined.png` already exists carrying its own
    author's subtitle — **"UNREADABLE without pole_sep"**. A figure
    that declares itself unreadable is on disk; the number that would
    make it readable cannot be derived from the committed artifact by
    any stated rule. Item 11 taken meanwhile.
11. SHIPPED 2026-08-14 (dario, d1d1697a) — M03 E survivor scatter,
    M03's first figure outside `f_figures.py`. **The geometry IS the
    argument**: individual arm on x, institutional on y, so
    "degree, not kind" becomes a LOCATION — both-positive and
    both-negative quadrants are degree, the off-diagonals are kind,
    and distance from y=x is how much harder the operation bit on one
    speaker; 59 of 65 survivors sit same-direction, so the headline is
    checkable by looking. THREE departures/fences: (a) the shortlist
    CONFLATES TWO POPULATIONS — the 65 survivors come from 702 words
    tested (only 58 verbs), while the 324 is the both-arms verb
    population behind Pearson 0.909, and `b_word_delta_by_word.csv`
    has no POS column, so the 324-verb population is NOT DERIVABLE
    from the named artifact; drawn as 702, stated on the panel.
    (b) The four reversals (`estimate`, `fail`, `objected`, `rule`)
    are coloured and explicitly NOT CLAIMED — Bonferroni survival is
    on the between-arm DIFFERENCE, not the sign flip, and section 3
    records zero significant reversals. (c) Axes bounded at ±0.004
    with an ASSERT enforcing the reason: a few extreme non-survivors
    (|max| 0.01729) otherwise crush all 65 into a seventh of the
    panel; every survivor is inside, exactly 8 of 702 outside and all
    grey, count on the panel, **and the producer refuses if a survivor
    ever lands outside — so the bound cannot silently become a
    filter.** NEAR-MISS AVOIDED ONLY BECAUSE THIS QUEUE NAMES IT:
    `c_word_delta_by_word.csv` is fenced as form-confounded and
    `b_word_delta_by_word.csv` is not — same directory, filenames
    differing by ONE CHARACTER, fence applying to one. Recorded in the
    producer docstring for the next seat in that folder.
    **SHORTLIST 6 ALSO SHIPPED 2026-08-14 (dario, d9c48a34)** — X §3g,
    "the word moves the scene and the model does not", new
    `plot_x_figs.py` under the per-letter regime (older `x_*.py`
    predate it, untouched). ONE AXIS, STACKED, and that is the whole
    design: the finding sets WORD (genital vs digit, +14.3 pts, 12/12,
    p 0.00049) against MODEL (aligned vs base, -0.8 pts, 15/30,
    p 0.918) and claims one is an effect and the other nothing —
    independent scales would show two clouds of similar width and
    invite exactly the opposite reading. **The null panel carries its
    own power band** (8.4 pts at 80%, observed -0.8 inside it):
    a null is worth drawing only if the reader can see it is not an
    empty measurement. READING THE SECTION IN FULL CHANGED THE PANEL:
    §3g carries a fence the queue entry did not — a pooled rate
    "nearly went into this document" (unpaired -8 pts looks tempting;
    paired at cell level p 0.484), and the direction REVERSES within
    families (AmberSafe 80->0 on one token, 40->60 on another;
    Tulu-3-DPO opposite on both), so the model null is **"no
    consistent direction across six alignment implementations"**, not
    "nothing happens". The drawn spread became the evidence for that
    wording unintentionally: the MODEL contrast is visibly WIDER than
    the WORD contrast while centred on zero, which is what an
    inconsistent direction looks like and is exactly what two pooled
    bars would have erased. NOT DRAWN: `thumb` at 0 of 60 in BOTH arms
    (the base model cannot keep the scene going on that word at all) —
    a categorical-form result against this panel's 0-100 score, and
    mixing them would repeat the substrate conflation; **queued
    separately as 11a**.
11b. SHIPPED 2026-08-14 (dario, f605d56b) — SHORTLIST 7, Y_diegetic
    3-4, added INTO `plot_y_figs.py` (registrar's file; purely
    additive, ratified under the same terms as t18). **THE NULLS ARE
    THE CONTENT.** A filter at the output blocks, deflects or declines,
    so it predicts EXIT rises and sexual_scene falls; both are FLAT
    (17 of 32 and 16 of 32, a coin flip twice) while composition
    INSIDE the scene moves on 27 of 32. So the two flat measures are
    drawn FIRST, at the same visual weight as the effects, with the
    prediction they falsify written on them — **a figure showing only
    the two moving measures would be the same finding with its
    argument removed**, because the argument IS that the account
    predicting movement predicts it where nothing happens. Two
    populations, forced not chosen: `sexual_scene` cannot be measured
    conditional on itself, so the flat pair is over all passages and
    the moving pair conditional on the scene; each panel names its
    own. READING THE PRODUCER WAS AGAIN THE WHOLE DIFFERENCE:
    `y_diegetic.py` filters pass-A parsed, requires MIN_N=20 per arm,
    reports rates as the MEAN of per-pair rates and deltas as the
    MEDIAN of per-pair deltas — the guessed version had ALL FOUR
    measures wrong (-8.86pp against a booked -6.12pp, two sign counts
    moved). All twelve booked values assert.
11c. SHIPPED 2026-08-14 (dario, 502332ef) — SHORTLIST 8, F21 arm
    effect. **THE PRIOR GOES ON THE AXIS.** The section is titled "the
    arm effect, and it runs the other way" and F21's stated direction
    is NEGATIVE; the result is positive at 41 of 46 lineages, so the
    quantity is not interesting for its size but **for being on the
    wrong side of a prediction** — a panel showing only the
    distribution would render the number and drop the finding. F21's
    predicted half is therefore a SHADED REGION of the axis, so a
    reader who knows nothing about F21 can see that 41 of 46 land
    outside where the prior said they would. TWO CHANNELS, because
    the file carries two things: position = `median_d_js` (the
    lineage's effect), colour = `share_cells_positive` (how many of
    its own 126 cells agree), diverging at 0.5 so a bare majority
    reads pale however large the median — **a lineage at +0.02 with
    71% agreeing is a different object from one at +0.02 with 52%**,
    and it pays off immediately: pythia-6.9b has one of the smaller
    medians and one of the darkest colours, several large-median
    lineages are pale, so **the ranking by effect and the ranking by
    agreement are not the same ranking**, which a medians-only dot
    plot would have asserted they were. AND ONE OF THE FIVE
    DISSENTERS IS NOT ONE: `Mistral-7B-v0.1` sits at -0.000138 with
    `share_cells_positive` exactly 0.5000 — 63 of 126 each way, a TIE
    — so labelling it in the same red as RedPajama (-0.0184) would
    make a coin flip look like evidence against the finding. Drawn as
    the tie it is; the finding's count of five is untouched and
    correct as stated (five sit below zero).
11d. SHIPPED 2026-08-14 (dario, f599dcef) — SHORTLIST 10, the M02
    D_CONTRA/D_CONTROL panel, **and THE FORM THIS ENTRY ASKED FOR IS
    ARITHMETICALLY FALSE.** A dumbbell invites one reading — that the
    distance between the marks is the contradiction-specific effect —
    and it is not: `d_both` and `d_ctrl` are each the MEDIAN OVER 26
    PAIRS of their own quantity, while `effect` is the median of the
    per-pair DIFFERENCE. **A median of differences is not a difference
    of medians**: `effect == d_both - d_ctrl` in 0 of 79 fields, with
    median |discrepancy| 0.000911 — *the same order as the effects
    themselves*. A single dumbbell letting the eye subtract would have
    been wrong in every row by an amount comparable to what the
    subtraction claims to show. Drawn as two panels, the subtitle
    saying the gap is not the residual, with an assert so the
    separation cannot quietly stop being necessary. Same family as the
    median-vs-mean grain that decided four reconstructions today,
    from a third direction: not WHICH aggregation but **whether an
    aggregation commutes with a subtraction.** It does not, and
    nothing in the artifact says so. SECOND CATCH, same panel: a first
    version used free x scales, so the residual rescaled to its own
    ±0.006 and a near-zero result FILLED its panel, reading as spread
    as the effect beside it — the finding is that the residual is nil
    against an effect three times larger, visible only on one ruler.
    Caught by reading the caption against the geometry ([5976]'s mint,
    second time today on this seat). The panel now shows, on a shared
    ruler: effect spanning the axis with D_CONTRA and D_CONTROL
    tracking closely, 39 of 79 surviving; and every residual collapsed
    to a narrow band at zero, none surviving — **the finding's own
    warning as a picture**, since reported as a residual alone a very
    large effect would have been filed as a null.
    **THIS CLOSES EVERY DRAWABLE ENTRY ON THE QUEUE.**
11a. SHIPPED 2026-08-14 (dario, e433f395) — `x_word_ladder_categorical`,
    the CATEGORICAL-FORM companion to X §3g, on the categorical
    substrate ONLY and never mixed with the 0-100 score panel. The
    ladder runs cock 38%, penis 33%, fingers 7%, toes 5%, thumb 0%,
    both arms at every rung, **and the anchor is what the two arms do
    TOGETHER**: `thumb` is 0 of 30 in the aligned arm AND 0 of 30 in
    the base arm — a model with no alignment whatever cannot keep the
    scene going either. That is `x_word_vs_model`'s conclusion reached
    by a ladder ending at zero on both sides at once, which is why the
    finding calls this the more readable form while keeping the score
    as primary, and why they are two figures.
    ONE DEFECT THE DODGE FIXED, AND IT WAS ON THE ANCHOR CELL:
    undodged, the two marks at `thumb` coincide exactly at 0 and only
    the one drawn last renders — **the single cell whose entire point
    is that BOTH arms sit at zero showed one dot**, the figure
    silently displaying half its own headline. Dodged at every rung
    rather than only at thumb, because **a coincident pair and a
    missing mark are indistinguishable**, and they coincide exactly
    where it matters.
    REGISTRAR ERRATUM ([6007], dario's catch): my queue-state lines at
    [5984] and [5995] both said "11a and 13c parked". Only 13c is
    parked; 11a was OPEN, as this entry read, and dario repeated my
    summary without opening the file — so it reported its own queued
    item as parked to the seat who would have had to unpark it. **My
    summary of my own file disagreed with the file, and the file was
    right.**
    (Shortlist 5 done; remaining shortlist
    items 6-8, 10 (X §3g, Y_diegetic four-panel, B_C lineage dots,
    M02 dumbbell) — unblocked, order per original ranking.
12. BLOCKED — M04 attention two-panel (shortlist 9). **STATUS
    CORRECTED 2026-08-14 ([5985], dario): the blocked half is the
    OTHER one.** This entry read half-blocked on Finding A and left
    the impression the FC half was missing. It is not:
    `results/A_post_utterance_shock.json` is Finding A's registered
    half on the fc substrate — 23,746 cells, 5,112 sites, 33 pairs,
    21 statistics including position +1..+10 — and is the most
    completely self-describing artifact in the repo (see the `_about`
    ruling; it carries producer, finding, spec, spec-frozen-at, seed,
    nboot, arbiter, a capture-only reproduction clause, and a
    `_positive_control` recorded as INVALID BY DESIGN *so nobody
    proposes it again with more data*). **THE ACTUAL BLOCKER IS THE
    ATTENTION-DECAY HALF, WHICH EXISTS AT ONE CELL**: shortlist 9
    names `attn_delta_smollm2_e1_cross_w200.json` — one pair, one
    prompt, three words, n=16. Drawing that above a 33-pair position
    profile puts n=1 beside n=33 at equal visual weight, the exact
    substrate mismatch this queue spent the day correcting, and M04's
    own doc warns against re-dignifying the retracted two-cell
    contrast. **And the population artifact measures something else**:
    `attn_norm_sweep_full.json` is a real 28 cells but carries `d_norm`
    and `probs` per word class, not decay against position — so the
    quantity shortlist 9 wants exists only at n=1 and the n=28 artifact
    does not contain it. Substituting one for the other would be the
    substrate conflation with extra steps. **This is a measurement
    debt with a different owner, not a plotting task — BUT IT IS HELD,
    NOT ASSIGNED** ([5987], lacan): two artifacts outside the stated
    search space carry decay against token distance at n_pairs = 42
    with per-pair sign counts and monotone decline —
    `a_decay_disjoint.json` (6 bins, +0.1165 down to +0.0512) and
    `a_decay_and_topic.json` (11-bin fine grid plus a `pref_vs_topic`
    block) — which is the SHAPE shortlist 9 wants at a real
    population. **What is decaying is unidentified**: the
    `pref_vs_topic` keys are `logq`/`logp`, which read as probability
    rather than attention, and if this is surprisal-decay the booking
    stands untouched. The artifacts are ORPHANS — grep over the whole
    tree returns no producer, no citing finding, no queue entry, no
    `_about`, no `_provenance`, no underscore key at all; added in a
    bulk commit naming neither. **Class 1B from the other direction:
    not a number with no artifact but an artifact with no claim.**
    Held because the cost is asymmetric — assigning a measurement that
    already exists spends a fleet; holding it for one identification
    spends minutes — and the identification wants whoever owns the M04
    attention line, reading the raw bytes rather than inferring from
    key names. **IDENTIFIED AND THE HOLD RELEASED ([5989], malign,
    owner): they are the A-LADDER ON THE PASSAGE SUBSTRATE, not
    attention work.** `logq`/`logp` are the aligned and base
    log-probabilities of the injected word — the same q and p as the
    ladder's log2(q/p) — and `A|A` / `B|A` are the four-term
    decomposition, so the decaying quantity is the PREFERENCE term
    against token distance. **And n_pairs = 42 is the tell: that is
    the PASSAGE corpus, against Finding A's fc 33 — a different
    substrate as well as a different quantity**, carrying A_RESULTS's
    EXPLORATORY, nothing-quotable fence. Drawing them beside
    `A_post_utterance_shock.json` would be the substrate conflation
    again with the fence attached. So shortlist 9's attention-decay
    quantity still exists only at n=1, **the measurement debt stands
    as a RUN not a read**, and lacan's hold was right at the price it
    named. CAUTION
    RETAINED ([5901]): the `A_position_*` figures are the PASSAGE
    substrate (A_RESULTS.md, EXPLORATORY, nothing quotable) and do not
    cover the FC debt. FLAGGED NOT WORKED THROUGH ([5985]): the
    head-concentration Lorenz curve among the M04 attention items may
    sit against [5226]'s per-head fence (474 heads in
    attn_norm_sweep_full against "nothing at the per-head unit").
13. SHIPPED 2026-08-14 — displacement network viz, first pieces:
    `displacement_network_core.dot/svg` (135-edge working map, maps/
    idiom, basin clusters) and `displacement_basin_procedure` panel
    (52 edges, chains visible) via `plot_displacement_network.py`;
    remaining basins (epistemic/expression/stasis) are one command
    each; chain-exhibit strip (fired->aimed->pointed, kill->shout->hum)
    still owed.
13c. UNPARKED by RH 2026-08-14; **(a) SHIPPED (dario, 4f24f25c)** —
    the three remaining basin panels (epistemic 25 edges, expression
    26, stasis 16), one command each, with `procedure` regenerated.
    **AND DRAWING THEM FOUND THAT THIS PRODUCER'S OWN REQUIRED FENCE
    HAD NEVER REACHED A RENDERED FIGURE** (registrar's producer,
    registrar's defect): the fence was a `//` DOT comment, which
    `dot` strips at render, so it existed in the `.dot` and in
    neither the `.svg` nor the `.png` — every basin figure shipped
    fence-less, including the one booked at item 13. And it carried
    only the computed half, omitting that the GROUPING into named
    basins is a reading, which is the half that matters because the
    names are the interpretive claim. Now a graph LABEL carrying both
    halves and surviving into the image. **(b) SHIPPED (dario, 94c8a74c)**: both cautions
    VERIFIED before drawing and both hold, for DIFFERENT reasons —
    fired->aimed (frame, 2.05) and aimed->pointed (frame, 3.40) fail
    on TAXONOMY, both links certified and replicated, co-rising in a
    shared frame; kill->shout survives the verb restriction
    (6.30 -> 2.42) while shout->hum, coupled at 4.58 in the full set,
    is ABSENT under verbs — a POPULATION failure. So the exhibit is
    the two failures beside the survivors rather than a strip of
    survivors, because **the two chains a reader arrives with are the
    two that do not qualify.** Selection declared on the basin
    grouping's terms: 1,433 two-hop chains survive both links, six
    drawn, which six is a reading and that 1,433 exist is a
    measurement; auxiliary exclusion stated (the highest weakest-link
    chain in the set exhibits `had`); FRAG imported from
    `plot_displacement_network` so the two figures cannot disagree
    about what counts as a word. **13c CLOSED 2026-08-14**: (c) verb redraws
    `4fdbf486`, (d) mutual-best couples table `d5bd9bec`, (e) graph
    structure `ba12879a`. **(e) TURNED UP A RESULT THAT READS AS
    IMPOSSIBLE: removing more than half the edges makes the network
    DEEPER — 1,818 edges to 795, longest condensation path 10 -> 14.**
    Mechanism: the full graph holds exactly ONE non-trivial strongly
    connected component of **141 words, 21% of the vocabulary**, every
    member reachable from every other, which condensation collapses to
    a single node — **so a path crossing it pays one step for a fifth
    of the network.** 42 of those 141 are words the verb restriction
    removes (`not`, `just`, `instead`, `if`, `in`, `on`, `no`, `all`,
    `he`, `is`, `could`, `how`, `later`, `long`, `before`, `after`),
    and taking them out does not shrink the blob, **it shatters it** —
    the 99 survivors sit in components of 16, 7, 2, 2, 2. The graph is
    deeper because it became LEGIBLE. **So the two depths are NOT
    COMPARABLE and the panel says so in its geometry rather than its
    caption**: each level's bar is split by whether a word sits in a
    multi-word component, putting the 141-word block at level 6 AS the
    thing that invalidates the subtraction a reader would perform.
    STABLE ACROSS BOTH POPULATIONS, with the producer refusing to draw
    if it moves: both longest paths end `cry -> understand -> need`,
    and under the restriction the run into it is
    `scream, see -> cry -> understand -> need` — **the deep end is
    identical in both; it is the MIDDLE the function words fuse.**
    Also 183 pure sinks / 225 pure sources full against 133 / 167
    verbs, written to `results/network_structure.json` with an
    `_about` naming the incomparability, so a reader can query the
    measurements rather than read them off pixels.
    Original entry: PARKED 2026-08-14 (RH: store and move on) — NETWORK
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
15. **CLOSED 2026-08-14 ([6175], dario). All four shipped, and P had NO
    figures of record when this opened.**

        15(1) headroom ladder        f465bdf8
        15(3) field poles            061ee44c
        15(4) arm AUC distribution   31d46c06
        15(2) named components       70779f68, weld at 47e7c75f

    **THE ENTRY'S OWN FRAMING WAS DROPPED AND THAT WAS THE RIGHT CALL**
    (registrar's ruling, dario's design decision): 15(2) asked for a
    decomposition with an *unnamed majority as the dominant empty
    region*. It is drawn as COUNTERFACTUALS, NOT SLICES, with no
    remainder. Register and concreteness correlate at rho 0.493 and §7b
    calls interiority and abstraction colinear by construction, so the
    components overlap and can sum past one — **an empty wedge is itself
    a partition claim**, and the finding denies the partition.

    **AND THE LEDGER WENT TWO-OF-SIX TO SIX-OF-SIX BECAUSE malign EMITTED
    THREE ARTIFACTS THAT DID NOT EXIST — in response to the FIGURE, not
    to any suspicion of error.** Nothing in §7 was wrong: the duplication
    dario reported was a genuine 3.9e-05 coincidence and the gaps were
    Class 1A. What the emission produced was **two concreteness
    instruments at 0.0921 and 0.1183 agreeing on direction and differing
    by a quarter on magnitude, and four genre indices spanning
    16.8-19.9%.** The panel's strongest feature is the agreement between
    measures, and none of it was drawable that morning.
    **A figure asked for numbers that a finding had only printed, and
    the asking turned them into evidence.**

15. ~~OPEN~~ (2026-08-14, RH-approved menu) — FINDINGS P FIGURES, for a
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

16. ~~PROMOTED FROM `shortlist 2`~~ **WITHDRAWN WITHIN THE HOUR
    2026-08-14, registrar. IT WAS ALREADY SHIPPED, by `queue 9`, this
    morning, by the seat I promoted it to** ([6179], `0db7cbcd`,
    `t18_unit_disagreement.png`, 428,784 B, ratified [5946]).
    **I verified the two ARTIFACTS existed and did not check whether the
    ITEM was done.** Inputs present is not work outstanding — the
    artifacts were present *because the figure had already been drawn
    from them*. A promotion is a claim about STATE and I checked INPUTS.

## STATUS CHANGES (2026-08-13/14)

- M05 C-R4 recapture bars: BLOCKED -> DEAD. R4 withdrawn ([5781]) —
  the domain contrast does not reproduce; nothing to draw.
- OPENING-MATCHED FAMILY: UNPLOTTABLE — withdrawn at construction
  level ([5811]); nothing from it may be drawn in any form.
  **A DERIVED VALUE TRAVELLED WITHOUT ITS NOTICE — PATCHED, AND MY
  DIAGNOSIS OF IT CORRECTED** ([5936] dario / [5937] registrar /
  [5940] lacan, `db4c8625`): `propagation.json` carried
  `undisturbed_reference: [0.016, 0.024]` as bare floats with no
  provenance or status, and now carries source, provenance and status
  emitted BY the producer. **REGISTRAR ERROR, corrected by lacan
  against its own interest:** I booked the value as void by
  inheritance from this withdrawal. It is not. The withdrawal is a
  BETWEEN-ARM construction defect (forced rows carry one more word of
  conditioning), and both fits producing the value run on
  `arm == "undisturbed"` only, so neither slope can contain the
  asymmetry — **a withdrawal has a scope, and this value sits outside
  it.** What IS exposed is worse and nobody referred it: the
  COMPARISON — `b_forced` fitted within forced arms, set beside a
  reference fitted within undisturbed rows, differs by exactly the
  conditioning that withdrew opening_matched, and whether a SLOPE
  inherits that asymmetry as a MEAN does is untested. H3 fenced and
  NOT QUOTABLE; `offset_repair.md` is the route that would settle it.
  **AND IT WAS NEVER A RANGE**: 0.016/0.024 are two estimators' point
  medians over the same rows (ANCOVA within-prompt +0.0158; naive per
  pair,role +0.0241). The citation named one fit's line count while
  quoting both fits' numbers, so two point estimates came to wear
  uncertainty bounds — and the forced slope was then called "if
  anything smaller" against what was really estimator disagreement.
  Both were printed and never persisted: the hand-carry hazard at its
  SOURCE rather than at its destination. Its
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

> **NUMBERING NAMESPACE: `shortlist N`.** Cited as such elsewhere in this
> file; do not renumber.
>
> **AND THIS LIST IS A CANDIDATE POOL, NOT A STATE RECORD. THE QUEUE IS THE
> STATE.** Reconciled 2026-08-14 after the registrar promoted `shortlist 2`
> to `queue 16` and it turned out to have been shipped that morning by the
> seat it was promoted to ([6179]) — and after `shortlist 1`, offered as the
> replacement, turned out to be shipped too:
>
>     shortlist 1  SHIPPED   queue 8, T-14 family, three figures on disk
>     shortlist 2  SHIPPED   queue 9, 0db7cbcd, ratified [5946]
>     shortlist 3  SHIPPED   fig32_ratio_polesep_joined.png
>     shortlist 4  SHIPPED   contradiction_null_numberline.png -- producer
>                            carries all four declared values
>     shortlist 6  SHIPPED   d9c48a34
>     shortlist 8  SHIPPED   b_c_arm_by_lineage.png
>     shortlist 9  BLOCKED   queue 12
>     shortlist 10 SHIPPED   l2_field_signature.png -- l2_fields_figs.py
>                            asserts 79 fields / 39 general / 0 specific,
>                            which is this entry's own specification
>     shortlist 5  unconfirmed  artifact present; e_survivor_scatter.png may
>                               or may not be it
>     shortlist 7  unconfirmed  artifact present; no diegetic figure found
>
> **ALL TEN ARE SHIPPED OR BLOCKED AND SIX DID NOT SAY SO** (dario's full mapping, [6180]; the registrar's two passes said three and then seven). **The pool is EXHAUSTED** — and the registrar's own
> first pass at this reconciliation said THREE, because it matched figure
> names against item numbers instead of item CONTENT. The count went from
> three to seven by reading what each entry asks for and checking whether a
> producer asserts those numbers. **A figure filename is a name; the
> specification is the relation.** The discharges were
> written in the QUEUE entries — *"discharged the shortlist's two"* — so the
> link existed, was correct, and ran ONE WAY: a reader arriving from the
> shortlist found open items. **Same class as the numbering namespaces fixed
> an hour earlier, one level over.**
>
> **Before promoting from here, check the queue.** A promotion is a claim
> about STATE; the registrar checked that the two ARTIFACTS existed, which
> is a claim about INPUTS — and they existed *because the figure had already
> been drawn from them.*

The candidates most likely to carry weight in the paper or book, drawn from
the per-folder lists below. Judgment, not doctrine.

1. **[SHIPPED by `queue 8` (T-14 family, three figures).]** **M01 T-14** — few large fallers against many small risers, on the
   BONFERRONI-SURVIVOR slice (declared: ALL/non-TOKEN): 206/36 at 3.79x,
   consistency-filtered so NOT identity-forced (the morning's retirement
   was corrected same day — RH's challenge; survivor sums do not zero).
   Dumbbell per lexicon as originally proposed, with the slice in the
   caption and the count panel beside it. `s_everything_marginal.csv`.
2. ~~**M01 T-18 beside M05-C**~~ **DISCHARGED by `queue 9` at `0db7cbcd`, one panel not two, ratified [5946].** The link existed and ran ONE WAY: queue 9 records that it discharged *the shortlist's two*, and nothing here said so, so a reader arriving from the shortlist found an open item ([6179]). Original text follows.

   **M01 T-18 beside M05-C** — the affect DiD at both units: one row per
   declared field, one-lineage DiD left, edge-unit DiD right, sign
   disagreements highlighted (RID:aggression the anchor).
   `results/t_affect_did.csv` + `data/m05_widening_null.json`. Renders "the
   gap is real, its sign is not robust" as one image instead of a retraction
   paragraph.
3. **[SHIPPED queue 10, `ee7d5b40`]** **M05 A-R4** — the joined ratio/pole-separation two-panel across the
   95-rung ladder, stage boundaries marked. `data/m05_ratio.parquet` +
   `results/m05_pole_sep.csv`. Discharges the plan's "together or not at all".
4. **[SHIPPED queue 2, `a17788c8`]** **M02 the calibration number line** — perfect superposition 0.000,
   observed 0.907, NEUTRALIZATION 1.006, RESOLUTION 4.031, per-cell strip
   behind, log-spaced; zh as a twin panel. `results/contradiction_null_en.csv`
   / `_zh.csv`. "1.0 is not a boundary, it is a place" is spatial; show it.
5. **[SHIPPED queue 11a, `e433f395`]** **M03 E §3/§4** — 324 verbs on indiv-vs-inst axes with the y=x diagonal,
   65 Bonferroni survivors labelled, four reversals coloured.
   `results/b_word_delta_by_word.csv`. "Degree, not kind" read off geometry.
6. **[SHIPPED 2026-08-14, `d9c48a34`.]** **M01 X §3g** — the word moves the scene (+14.3 points, 12/12 cells), the
   model does not (−0.8, p .918), two panels, same axes.
   `results/x_beam_frame.csv`.
7. **[SHIPPED queue 11b, `f605d56b`]** **M01 Y_diegetic §3** — the conditional four-panel: CLEAN_SCENE −6.12pp,
   SUPEREGO_IN_SCENE +4.30pp, EXIT and sexual_scene flat.
   `results/y_passages.parquet` via `scripts/y_diegetic.py`. The filter
   account predicting exactly the two panels that do not move.
8. **[SHIPPED queue 11c, `502332ef`]** **M03 B_C §1** — the JS arm effect, one row per lineage, 41/46 above zero,
   five dissenters labelled. `results/b_arm_by_lineage.csv`.
9. **[BLOCKED — see `queue 12`.]** **M04 attention §5** — attention-back decay above, Finding A's surprisal
   sweep below, both in both aggregations (disjoint bins beside cumulative).
   `results/attn_delta_smollm2_e1_cross_w200.json`. The figure IS the
   two-phenomena argument. (Finding A's half needs the produce-first step.)
10. **[SHIPPED queue 11d, `f599dcef`]** **M02 dumbbell** — D_CONTRA sitting on top of D_CONTROL, one row per
    field, 39/79 general survivors emphasised, 0/79 specific.
    `results/l2_fields_{meta,norms,usas_fine}.json`. The doc's own warning
    (a residual-only report would have filed a positive as a null) as a
    picture.

Methodological pair worth drawing for the book's methods spine: **M03 D §6**
(ICC 0.855 spaghetti — why 50 rungs are not 50 observations;
`results/d_ladder.csv`) and **M01 U-1/2** (the ladder slope chart, removal
stopping while addition continues; `results/t_ladder_steps.csv`).

## Per-folder candidate lists

> **NUMBERING NAMESPACE: `<folder> candidate N`.** Each folder restarts at 1.
>
> **RECONCILED 2026-08-14 AGAINST THE SHORTLIST, PARTIALLY, AND THE
> REMAINDER IS HONESTLY UNKNOWN.** Ten of these 59 entries name a
> shortlist number, and every shortlist entry is now confirmed shipped or
> blocked ([6180]) — so **those ten are discharged by derivation** and are
> marked below:
>
>     M01 1 -> sl 1    M02 1 -> sl 4    M03 1 -> sl 5    M05 1 -> sl 3
>     M01 3 -> sl 6    M02 6 -> sl 10   M03 3 -> sl 8    M05 2 -> sl 2
>     M01 4 -> sl 7                     M04 1 -> sl 9  (BLOCKED)
>
> **THE OTHER 49 ARE UNVERIFIED AND I AM NOT GUESSING.** A number-matching
> heuristic found producers asserting the values of only 5 of the 59, which
> proves almost nothing: most entries name too few distinctive numbers for
> that test to have power, and *absence of a match is not absence of a
> figure*. **Determining their state needs the check that actually worked
> on the shortlist — read what the entry specifies, then look for a
> producer asserting it — and that is per-entry work, not a sweep.**
>
> **So this pool carries the same defect the shortlist did and it has never
> been tested**, because nothing has been promoting from it. dario's
> reading is the right prior: assume it is worse, not better. **Before
> taking anything from here, verify that one entry rather than trusting the
> list.**

Condensed from the readers' reports: doc / result / data / suggested form.
Ordering within each folder is the reader's ranking.

### M01_displacement (reader's tier 1 and 2; tier 3 all blocked, see above)

1. **[DISCHARGED via its shortlist link]** T-14 fallers/risers dumbbell — `s_everything_marginal.csv` (shortlist 1).
2. T-18 affect DiD paired slope, arrows coloured widens/converges —
   `t_affect_did.csv` (shortlist 2).
3. **[DISCHARGED via its shortlist link]** X §3g two-panel — `x_beam_frame.csv` (shortlist 6).
4. **[DISCHARGED via its shortlist link]** Y_diegetic §3 conditional panel — `y_passages.parquet` (shortlist 7).
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

1. **[DISCHARGED via its shortlist link]** Calibration number line (shortlist 4).
2. Next-word three-role word dumbbells (`kill` −9.9/−1.3/−21.3; the epistemic
   residual) — `dp.pkl`, REBUILD REQUIRED first.
3. t and resid by role, paired violins, resid=1.0 marked —
   `l3_geometry_union.parquet`. "Same shadow, equally off-axis" needs both
   panels at once.
4. Pole separation vs superposition loss, rho −0.420, n=45 —
   `polesep_vs_superposition.csv`. Cheapest strong figure in the folder.
5. Non-universality slope plot, base→aligned per lineage, 12/46 reversers,
   AmberSafe −0.1392 named — `contradiction_null_by_pair_en.csv` + `_zh`.
6. **[DISCHARGED via its shortlist link]** D_CONTRA-on-D_CONTROL dumbbell (shortlist 10).
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

1. **[DISCHARGED via its shortlist link]** E §3/§4 survivor scatter (shortlist 5).
2. E §4 dumbbell, 65 rows, indiv vs inst dots joined, coloured by pattern —
   `b_word_delta_by_word.csv`. The words the argument quotes.
3. **[DISCHARGED via its shortlist link]** B_C §1 lineage dot plot (shortlist 8).
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

1. **[DISCHARGED via its shortlist link]** Attention §5 two-panel decay comparison (shortlist 9).
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

1. **[DISCHARGED via its shortlist link]** A-R4 joined ratio/pole-sep panel (shortlist 3; discharges the plan).
2. **[DISCHARGED via its shortlist link]** C-R3 vs T-18 sign-disagreement dumbbell (shortlist 2).
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
   **THAT INPUT IS IGNORED, NOT TRACKED** (265 MB against the repo's
   100 MiB pre-commit hook, so committing was never available;
   lacan `fec71c44`, caught by dario's cited-artifact test at [6036]
   and fixed at [6037]). To get it back:
       REGENERATE  uv run python meta/M06_generation/scripts/m06_mediation.py --by-prompt
       COPY        /Volumes/diderot/malign-logits/meta/M06_generation/results/
   **An ignored file that a document names is a broken reproduction
   path unless the document says how to get it back** — the ignore
   rule is the cheap half, the pointer is the half that makes it
   safe.

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
