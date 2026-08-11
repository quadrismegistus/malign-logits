# M02 — Frame-exit: what contradiction does to the continuation

**Where the claim stands (2026-08-09, docket through [5206]).** M02 began as
"alignment resolves contradiction by leaving the frame" (F11's discourse-grain
extension), inverted under first-look instruments on 08-08 (the contradiction
cell exits *at or below* its poles and controls), and the powered cascade has
since split the question by grain:

- **At hidden-state grain the superposition claim HOLDS and is now
  controlled**: same-side conjunction controls sit AT their poles, the
  contradiction sits BETWEEN, in both arms — alignment does not undo it
  (L3 geometry, 45 pairs / 90 models, docket [5157], pen-recounted [5161]).
  The L3 pilot's "interior excursion" did NOT survive the controls: one true
  cell, not a population effect.
- **At next-word grain the frame question is not answerable**: the
  contradiction's own axis carries ~2% of next-token variance; frame
  membership fails from an LLM coder (OFF-FRAME at chance) AND from four
  geometric constructions — two independent instrument families at chance is
  evidence about the construct ([5187], [5189], [5195]). What next-word DOES
  deliver, cheaply and robustly: the pole axis (geometric, AUC 0.995) and the
  content/function field (AUC 0.812, language-neutral).
- **The construct therefore lives at passage grain**: "a single word is not
  yet a departure; a passage is" — and the L2 generation run (the registered
  primary's corpus) is GO on RH's word, gated only on his scope choice
  ([5196]–[5206]). The aligned arm's observed mode is a THIRD thing: neither
  resolution nor refusal but naming the contradiction from outside
  (meta-commentary; [5195] read with the Oedipalization slide).

The redo registration's declarations stand under the plan-documents regime
([5148]); its EXECUTION BLOCK is the live state of the cascade. Everything
here enters analysis as declared priors, never support.

## The map

    registrations/    the redo registration (registration_m02_redo.md — body
                      hash 33536d93ab9abb9a; execution block = live state),
                      the N3 execution addenda (v3 current), the battery and
                      generative-contradiction plans, the channel-3 spec
    plans/            plan documents under the [5148] regime: the delta run
                      (complete), l3_geometry (run), f11_l2_generation (GO,
                      awaiting RH's scope word; carries the cross-scoring and
                      per-layer-projection addenda and the declared
                      slope-as-primary curve statistic)
    findings/         the write-ups (index below)
    scripts/          the marker battery, the sweeps, the quintuplet builder,
                      the pilot/geometry producers (data/f11_quintuplets.json
                      is BUILT — edit a source and rebuild, never the output)
    results/          per-cell CSVs and parquets, roster/filters stamped

## Findings index

Every write-up in `findings/`, one row each. **Status/grade is quoted from the
document's own status line and is never restated or upgraded here**; where a doc
carries no status line the cell says so, and the doc's own header language is
what governs. Where two findings pull against each other the tension is recorded
with pointers to both and left unresolved.

| Doc | Claim (one line) | Status/grade (verbatim) | Unit and population | Figures | Corrects / superseded-by |
|---|---|---|---|---|---|
| [M02_eassist_ambient.md](findings/M02_eassist_ambient.md) | Aligned checkpoints emit assistant control tokens and system-prompt fragments into raw continuation unbidden (17/18 movers, one-sided p 7.2e-05); §4 the magnitude is the Falcon3 instruct recipe alone (52.76% top cell against a 2.01% ceiling elsewhere), §7 it supplies M02's per-checkpoint ambient floor. | (no status line) | Base/aligned pair is the unit. 29 pairs from `data/base_aligned_pairs.json`, both arms at n >= 200 passages, temp 1.0; ties (11) excluded from the sign test. Second cut: Registry n >= 50, 33 pairs, 19/20 p 2.0e-05. Ambient rate measured on the F01 battery. | none | Supplies the null model / ambient floor that any M02 exit claim must clear. Sits beside Y (coded passage) and Z (semantic fields) as the token grain of the same object; §7 limit says the M02 baseline must be regenerated in-corpus. |
| [contradiction_ratio_has_no_null.md](findings/contradiction_ratio_has_no_null.md) | F11's ratio carries no neutralization reference and 1.0 is where a distribution holding NEITHER pole lands, so alignment's motion is toward neutralization (frame exit) and not resolution (which sits at 4.03); against a real null the superposition signal is +0.1198 base at 46/46 lineages and attenuates -0.0379 under alignment (p 3.61e-05), but universality fails (12/46 lineages reverse) and F11's family gradient reproduces in ORDER (rho +0.728) and not in SCALE. | "Status: PROVISIONAL." | Lineage representative pair is the unit (46 lineages from `data/lineage_representative_pairs.txt`, not 52 arms). 22 live English quintuplet groups, 2,007 of 2,024 (model, group) triplets; substrate `twp_words` theta=0.001 with a full `logit_probs` 1e-6 re-run at 75 models. Same-frame discharge on 10 of 22 groups. zh replication: 21 groups, 42 lineages. Family gradient: 9 families. | none | Corrects `findings/F11_contradiction.md` (repo root): the ordering claim survives, the ABSOLUTE/crossing claims ("Zephyr 1.01 crosses the threshold") survive neither the scale shift nor the recalibrated boundary. Joins `pole_axis_t_is_not_superposition.md` (union vs intersection; pole separation vs superposition signal, rho -0.420) rather than conflicting with it. |
| [exit_markers_first_look.md](findings/exit_markers_first_look.md) | Four designs on the exit-marker battery: §1 the cloze blank is CLOSED as a transgression symptom (raw battery reversed, twins null), §2 Q/A conversion at the genre-controlled twins survives three decoder rosters (+0.43pp, p 0.035) and repeats at an independent design (edges: E-QA +3.99pp, E-ASSIST +0.77pp, p 0.0002), §3 faller-vs-riser is null on every type and roster, so the frame apparatus reads the scene and not the signifier. | "STATUS: FIRST LOOK THROUGHOUT — regex surface patterns over cached generations and beams, declared before reading" | Unit varies by design. RAW BATTERY: prompt, 48 DEFAULT prompts, 190,261 passages, 107 checkpoints (demoted to descriptive pilot, genre-confounded). EDGES: base to derivative, 48 edges over 29 bases, base-clustered. TWINS: checkpoint (RH ruling), 105 stems x MARKED/UNMARKED, ~714k 10-token beams per side, 66-67 checkpoints, 26 nonzero at the clean roster. FORCED ARMS: site, wave-3 lexical beams over the 210 twin prompts. | none | §3 is where M01's Finding W damage-family nulls are superseded (see `meta/M01_displacement/REGISTRATIONS.md`, row W, which points here). Internally: raw battery demoted on RH's genre diagnosis; forcing-vs-undisturbed demoted per [5026]; E-MENTION's aligned column withdrawn under the 12-sampler filter; twin DiD not claimable in either direction. |
| [field_signature_not_contradiction_specific.md](findings/field_signature_not_contradiction_specific.md) | Alignment's semantic-field signature (down: names, places, objects, bodies, time markers; up: emotion, cognition, communication, sociality, evaluation, dominance) survives correction on 39 of 79 fields across three lexicon granularities, and none of it is contradiction-specific: against a syntactically matched single-pole control the specific residual survives on 0 of 79. | (no status line) | Passage is the unit; the median over 26 pairs is the statistic; Wilcoxon over pairs, Benjamini-Hochberg FDR 0.05 within each source. 54,080 English continuations, the 26 complete pairs of `data/f11_l2_receipt.json`, roles `both` / `control_a` / `control_b`, both arms. 79 fields: meta 13, norms 24, usas_fine 42. | none | Reaches the same de-concretization signature as `meta/M05_emergence/findings/B_field_flow.md` on a different population (convergence, not correction). TENSION, recorded and not resolved: this doc finds 0 of 79 contradiction-specific fields at passage/field grain, while `pole_axis_next_word_grain.md` §9 reports a contradiction-specific epistemic residual (`understand`, `know`, `question`) at next-word grain, itself flagged there as descriptive labelling rather than a test. |
| [pole_axis_next_word_grain.md](findings/pole_axis_next_word_grain.md) | Oedipalization is not detectable in the next word: the contradiction's own axis carries ~1.96% of next-token variance and the own-vs-foreign-axis difference is null on both embedders (§1); §2 the sharpening is real, large and happens equally on axes the prompt never named, §4 a small directional tilt toward the named axis replicates on two nulls and two embedders, §9 the controls show that roughly half of what alignment does at a contradiction, by direction, is what it does at a same-side conjunction (cos 0.478, null 0.179). | "Exploratory throughout; nothing was pre-registered and nothing here is frozen." | Cell = (group, base/aligned pair). 13 English groups x 46 pairs = 597-598 BOTH cells, against 583/585 control cells at 45 families in §9. Content words only (`fields.is_content_word`), BOTH prompt for §1-§8; source of record `data/f11_quintuplets.json`, candidates `data/f11_k2_units.json`. Five CATEGORY groups, `f11_class`, `f11_loyal` and `f11_holy_b` excluded. Run independently on BGE-m3 and GloVe-300. | none | §8 is CORRECTED BY §9 inside the same document (the word table is real but not contradiction-specific). §5's signed valence tilt is WITHDRAWN on GloVe non-replication. A two-element `SentenceTransformer.encode()` defect corrupted 4 of 13 axes in the first version and is fixed. See the tension recorded against `field_signature_not_contradiction_specific.md`. |
| [pole_axis_t_is_not_superposition.md](findings/pole_axis_t_is_not_superposition.md) | The L3 pole-axis projection `t` is not a superposition measure (BOTH's off-axis residual is 0.954 of the entire pole separation; both and neither cast the same shadow), with three corrections to [5157]: correction 2, `t(both)`'s invariance is uninformative once one global midpoint compression is removed; correction 3, "clustered by family" clustered nothing and is recomputed at the lineage with every conclusion holding; plus the union rebuild and pole separation predicting superposition loss (Spearman rho -0.420, p 0.0041, n=45 lineages). | "Status: PROVISIONAL, and partly a CORRECTION." | (pair, group, arm, layer) for the geometry; the lineage for every contrast. [5157] population 43-45 pairs / 39 lineages; UNION population 52 pairs, 104 models, 391,278 rows, 46 lineages. Two n kept apart by the coverage column: BOTH-vs-control on CONTROLS_SCORED only (44 pairs / 37 lineages), `t(both)` and `both_matched` on both strata (52 pairs / 46 lineages). Union/intersection check at 46 lineages, both arms. | none | Corrects docket [5157] on three counts (the "superposition" label, the family clustering, the coverage column formerly misnamed TRIPLET_ONLY). Reconciles with `contradiction_ratio_has_no_null.md`: the same gap found independently on the output side, and the union-not-intersection result explains why the two instruments looked to disagree. The causal arrow on pole separation is explicitly held for M05's SFT/DPO checkpoint ladder. |

**Cross-index notes** (pointers only; nothing here resolves anything):

- `meta/M01_displacement/REGISTRATIONS.md` row W -> `findings/exit_markers_first_look.md` §3: W's damage-family nulls are superseded by the wave-3 split written up here.
- `findings/F11_contradiction.md` (repo root) <- `findings/contradiction_ratio_has_no_null.md`: scale and universality corrected; ordering survives.
- `meta/M05_emergence` <- `findings/pole_axis_t_is_not_superposition.md`: the pole-separation arrow is held for the M05 checkpoint ladder. `meta/M05_emergence/findings/B_field_flow.md` and `findings/field_signature_not_contradiction_specific.md` reach the same de-concretization signature on different populations.

## Key facts a consumer needs before touching this module's data

- **The register governs.** The claims register (article hub,
  `notes/claims-register.md`) outranks this page and every finding file.
- The prompt populations live in `data/prompt_categorisation.json` (status
  filters are GROUP-WISE); the assembled quintuplets (en+zh, controls inline)
  are `data/f11_quintuplets.json`. NOTE: five zh control groups still carry a
  stale "GLOSS GATE PENDING" flag — the gate cleared at docket [5089].
- **The L1 instrument reads the `twp` column, never logits** (RH, [5136]).
- **The coverage gate splits by language** ([5193]): English passes at 0.0%
  demoted (confirmatory); Chinese fails N3 §4 at 54.8% (descriptive-only at
  L1 — the cause is theta-flatness, which has no analogue in a generated
  passage, so the zh question lives at L2).
- `data/f11_k2_units.json` is the k>=2 sampling frame (31,815 units), with
  voting-model LISTS per unit; triple-verified. CUSTODY: two of its 92 voting
  models (jais x2) exist only in the store, with no transport file — a store
  rebuilt from flat files is NOT this population.
- `data/f11_l2_population.json` is the L2 population (187 distinct strings,
  pen-checked [5198]). **The collapse rule is PER-CONTRAST, not per-group**:
  holy/holy_b are one unit for the redo primary and two for N3's excess;
  beauty/beauty_ugly never collapse. This derives the registration's n=15.
- Cross-scoring by token ids is safe WITHIN PAIRS only (49/52 ID-safe,
  silent-risk class empty, `data/f11_l2_tokenizer_pairs.json`); cross-family
  scoring requires the re-tokenized path.
- "Per-layer" names TWO instruments a thousandfold apart in cost ([5205]/
  [5206]): per-layer PROJECTIONS (h.a, one scalar per layer, carries its own
  mismatched-axis null) vs twp-BY-LAYER (`expand_layers`, full distributions,
  no null yet — dropped from the L2 fleet by both-seat consensus).
- `beam_fc` keys carry no `design` field — design lives in record VALUES;
  12 of 68 beam_fc checkpoints beam-sampled under their own configs.
- f11_reason / _zh: the declared weak-manipulation negative control, outside
  the primary, run and reported beside it — and it OUTRANKS the primary.
