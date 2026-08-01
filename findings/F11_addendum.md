---
status: verified
grade: A
date: 2026-07-23
role: addendum
parent: F11_contradiction
instruments: [logit-mass, intervention, classification]
families: [olmo, amber, llama, qwen, tulu, zephyr, olmo-tiny]
chapters: [ch03, ch11]
data: [contradiction_four_mass.csv, contradiction_rebaselined.csv, contradiction_address_check.csv, f11_classify_blinded.csv, f11_classify_metadata.csv, f11_classify_key.csv, f11_r1a.csv, f11_r1b.csv, f11_r1c.csv, f11_r1d.csv, f11_r2a.csv, f11_r2b.csv, f11_r2c.csv, f11_r2d.csv]
scripts: [f11_cross_family.py, f11_meta_contradiction.py, f11_new_pairs_all_families.py, f11_gen_classify.py, f11_classify_analysis.py]
---
# F11 Addendum: Mechanism Decomposition — Frame-Exit, Not Exclusive Disjunction

> **PROMOTION CONDITION ([2198], RH's standing scope rule).** A canonical claim
> in a `meta/` campaign runs on **all families we have** under a declared
> admissibility rule — never a hand-picked subset. This finding measures
> **7 family labels = 6 independent lineages**, against a roster of
> **49 labels / 34 lineages** ([[data/lineage_map_models.json]]). It is
> therefore an **F-series observation** and cites its subset. **If the frame-exit result (the reframe leans on it) is
> promoted to M-canonical, the promotion INCLUDES the full-roster re-run** —
> promotion is re-measurement, not relabelling. Priced when the cloud grid
> frees.
>
> **AND THE PROMOTION HAS TWO CONDITIONS, NOT ONE** ([2199].3, ruled [2201].2).
> Alongside the full-roster run: the [2111].2 **substrate repairs** — the
> intervention has no named substrate and the coherence confound (r = −0.032)
> is computable from no named file. **A full-roster run produces new numbers at
> scale and leaves those two exactly as unsourced**, so the roster run must not
> be mistaken for the whole clearance.
>
> **CANONICAL RESTATEMENT ([2204]/[2207]), and it corrects this file's own body.**
> **Five of six lineages are exit-modal; the sixth, `meta-llama/Llama-3.1-8B`,
> RESOLVES rather than exits, consistently across BOTH its alignment
> implementations (1 exit cell in 10).** The tables below print `llama` and
> `tulu` as two adjacent "Mixed" rows — **they are one pretraining run, counted
> twice, and the collapse is exactly the dissent.** Merged: {resolve 4, engage
> 3, null 2, exit 1}. **Two families disagreeing independently is noise; one run
> disagreeing consistently across both its recipes is a property of that run** —
> smaller dissent, far more structured, and more threatening to a universal
> reading than the family count showed.
>
> **The rate demotion FIRMS at the correct unit:** ICC 0.107 by lineage (not
> 0.084 by family), DEFF 1.51, so the frame-exit plurality sits at **p ≈ 0.19**,
> not the audited 0.09. **Rank, never rate.**
>
> **The pole-direction table carries the same double-count:** `llama` 3/5 and
> `tulu` 2/5 prosocial merge to **5 of 10** — the exact no-direction midpoint of
> a scale running from OLMo/Amber/Qwen at 4/5 to Zephyr at 1/5. At the family
> unit that reads as two wobbling families; at the lineage it is **one run with
> no pole direction at all.** Flagged, not asserted: the registered rival
> (pole-token corpus frequency) is untested.
>
> **Descriptive counts in this file stay at 7 family labels** — the behavioural
> n and the CSV row arithmetic describe how many files were read, not how many
> independent things were measured. Restating them would stop the row counts
> reconciling with the data.
>
> **INSTRUMENT UNREGENERABLE ([2222]/[2223]), added 2026-08-01.**
> `data/contradiction_four_mass.csv` — the substrate for the frame-exit
> mechanism below — **has no producer, and never had one.** The file landed
> alone at `c71b1dd` (23 Jul, one file changed, 331 rows, no code);
> `git log --all -S'pole1_mass' -- '*.py' '*.ipynb'` returns **zero commits
> across all history**, and `blend_mass` appears in exactly one file in the
> repository: the CSV itself.
>
> **So `pole1_mass` / `pole2_mass` / `blend_mass` / `in_frame` / `remainder`
> have no definition here.** Which surfaces count as pole1, whether inflections
> count, whether blend is an explicit list or a residual — none of it is
> recorded. The numbers can be read and cannot be regenerated.
>
> **Status: VERIFIED-AS-RECORDED; NOT REPRODUCIBLE IN PRINCIPLE** (sharpened
> from "unregenerable" at [2228] — **a construct with no recorded definition is
> not the same construct**, so a re-specification produces a different
> instrument, not this one). The [2110] audit
> reproduced the decomposition FROM this CSV, which verifies its arithmetic and
> cannot verify the instrument, because there is none to verify. **This is not
> a claim that the numbers are wrong** — they were produced by something, and
> they reproduce internally. **It is a claim that no new question can be asked
> of them:** any re-unit, re-roster or re-threshold needs the generating
> definitions, so a full-roster "re-run" is necessarily a **RE-SPECIFICATION**,
> and its agreement with the published bands is a reported question rather than
> an assumed continuity.
>
> **AND THE AUDIT'S REPRODUCTION STATEMENTS DO NOT CLEAR THE INSTRUMENT**
> ([2228].2). The [2110] audit's "9 of 9 reproduces" and "34 of 35 rule
> agreement" are statements about the CLASSIFICATION STEP: they read
> `contradiction_rebaselined.csv`, which is **downstream of the four-mass**, and
> re-derive a labelling over columns whose construction was never questioned.
> **A derived file reproduces perfectly from its parent and says nothing about
> the parent.** "The numbers reproduce" is exactly the sentence a reader takes
> as clearance, and it is not one.
>
> Frame-exit claims therefore carry **two** flags: the rank-not-rate demotion
> (p ≈ 0.19 at the lineage) and this one.

## Summary

The original F11 formulation is vindicated by the mechanism decomposition: alignment shifts the model OUT OF the contradictory frame, not onto a pole within it. The strong exclusive-disjunction claim (Deleuze's Oedipal XOR) fails everywhere — no family shows stable pole-commitment on contradictions.

## Universal base tolerance (11 families)

All base models tolerate contradiction (ratio < 1). Cross-family replication confirms this is a universal property, not an OLMo artifact. Coherence-independent (Δ-ratio vs Δ-coherence: r = −0.032, p = 0.93).

## Instrument reconciliation

Three instruments, each measuring one dimension:

| Instrument | What it measures | F11 result |
|---|---|---|
| Ratio (JS blend vs poles) | Default operating point | Base: superposition; aligned: shifted toward resolution (SFT-driven, +0.05 to +0.28) |
| nnsight intervention | Geometric structure | Preserved across all training stages (intervention range 0.71–0.73) |
| Four-mass accounting | Mechanism of the shift | Frame-exit plurality (40%); engagement and resolution minorities |

The ratio conflated three mechanisms (resolution, frame-exit, engagement). The four-mass decomposition separates them; re-baselining against single-pole prompts (controlling for general topic-adherence) produces the citable mechanism distribution.

## Re-baselined mechanism distribution

Contradiction-excess = (AB_delta − A_delta) for each mass. Re-baselined against single-pole prompts to control for general topic-adherence (which deflated the "engagement" mechanism — Zephyr's in-frame rise was general on-topicness, not contradiction-specific).

| Mechanism | Count | % | Description |
|---|---|---|---|
| Frame-exit | 14/35 | 40% | Contradiction dampens in-frame mass below the single-pole baseline |
| Engagement | 8/35 | 23% | Contradiction-specific in-frame excess (survived deflation) |
| Resolution | 8/35 | 23% | Pole commitment (6 antisocial pole, 2 prosocial) |
| Null | 5/35 | 14% | No detectable contradiction-specific effect |

Frame-exit is the **plurality** mechanism with genuine engage and resolve minorities. Classification threshold: in-frame excess < −0.05 = exit; > +0.05 = engage; pole excess > 0.05 with in-frame neutral = resolve. 10 of 30 classified cells sit within 0.02 of the ±0.05 threshold, indicating substantial threshold-sensitivity — report the tendency, not the partition. 5 cells are unclassified (within the ±0.05 band).

**Per-family mechanism profiles** (5 cells each, modal mechanism as tendency):

| Family | Modal tendency | Cells (from CSV) | Note |
|---|---|---|---|
| OLMo | Exit | 4 exit, 1 engage | |
| OLMo-tiny | Exit | 2 exit, 1 engage, 1 resolve(p1), 1 null | |
| Amber | Exit | 3 exit, 1 engage, 1 resolve(p2) | |
| Qwen | Exit | 2 exit, 1 engage, 1 resolve(p2), 1 null | |
| Zephyr | Exit | 2 exit, 1 engage, 1 resolve(p2), 1 null | Post-deflation |
| Tulu | Mixed | 1 exit, 2 engage, 2 resolve(p2) | Resolve co-modal |
| Llama | Mixed | 1 resolve(p1), 1 resolve(p2), 1 engage, 2 null | Lightest intervention |

## Pole-consistency null

The consistency index measures whether the aligned model commits to one pole more on the combined (AB) prompt than on single-pole prompts: |p(pole1|AB) − p(pole2|AB)| / max(|p(pole1|A) − p(pole2|A)|, |p(pole1|B) − p(pole2|B)|). Index > 1 = AB more polarized than single-pole; < 1 = less polarized (blended).

| Family | Mean consistency (aligned) | Mean consistency (base) |
|---|---|---|
| OLMo | 1.09 | 0.43 |
| OLMo-tiny | 0.64 | 0.41 |
| Amber | 0.53 | 0.45 |
| Llama | 0.36 | 0.33 |
| Qwen | 0.35 | 0.60 |
| Tulu | 0.28 | 0.33 |
| Zephyr | 0.25 | 0.36 |

**No family shows alignment-induced polarization on contradictions** (all except OLMo's outlier-driven 1.09 sit below 1.0). Aligned models are no more polarized than base models. The exclusive disjunction claim — that alignment imposes a choice — is refuted: alignment exits the frame rather than choosing within it.

OLMo's 1.09 is driven by a single outlier (beautiful/disgusting: 3.28, where the aligned model strongly commits to the beauty pole). The median across OLMo's 5 pairs is 0.67.

## Pole-direction drift (not commitment)

For pairs where aligned mass shifts toward one pole, the direction is family-specific and NOT normatively consistent:

| Family | Prosocial pole | Antisocial pole | Pattern |
|---|---|---|---|
| OLMo | 4/5 | 1/5 | Mostly prosocial |
| Amber | 4/5 | 1/5 | Mostly prosocial |
| Qwen | 4/5 | 1/5 | Mostly prosocial |
| Llama | 3/5 | 2/5 | Mixed |
| Tulu | 2/5 | 3/5 | Mixed |
| Zephyr | 1/5 | 4/5 | Mostly antisocial |

The drift is distributional mass movement, not stable commitment (consistency index < 1.0 for all families except OLMo's outlier). **Registered rival**: pole-direction may track alignment-corpus vocabulary frequency (PKU harm-focused data saturated with harm terms; UltraChat/UltraFeedback differ). The datasets are public; a pole-token frequency count against direction is a cheap decisive test. Neither the normative reading (alignment steers prosocial) nor the anti-normative reading (alignment steers antisocial) is established — the direction is family-specific drift, pending the frequency rival.

## SFT drives the aggregate

The aggregate resolution shift is SFT-driven (OLMo +0.16, OLMo-tiny +0.25 at SFT; DPO incremental). Coherence-independent (r = −0.032). Safety-data-style-independent (does not track PKU vs CoCoNot vs none). F11 joins deference and mild proceduralization in the ego-constitution cluster: instruction-following installs commitment/coherence as such.

## Behavioral classification: level dissociation

The four-mass mechanism (frame-exit plurality at the token level) does NOT propagate to the thematic structure of generated text. Two-rater blind classification (merged kappa 0.674 overall, 0.726 excluding neutral items, consensus coverage 79%) on 1,596 continuations (7 families × 11 pairs × 3 prompts × 2 layers × 3 samples + neutral controls).

**BOTH-held tension is the modal outcome in both layers, unchanged by alignment.** On AB (contradiction) prompts, BOTH rate: base 0.45, aligned 0.45. Behavioral AWAY-excess over the single-pole baseline is tiny and smaller in aligned models (base +0.045, aligned +0.015). Pole commitment barely moves (0.22 to 0.25). Aligned models still write ambivalence at base rates.

**The level dissociation:** the four-mass exit is an operating-point fact (next-token distribution) that does not propagate to the thematic structure of 300-char generations at temp 1.0 — the same gate-then-recover shape as violence (admission suppressed, scene continues). The wave reads text; at the text level, base and aligned are indistinguishable on contradiction-holding. This strengthens the anti-Fazi line: the unity is not even in aligned generations — it lives at the interface, not the output.

**Per-family profiles** (aligned AB, consensus set):

| Family | D1 | D2 | BOTH | AWAY | OFF | BOTH Δ (b→a) | Note |
|---|---|---|---|---|---|---|---|
| Amber | 0.11 | 0.04 | 0.56 | 0.04 | 0.26 | +0.21 | BOTH rises; neutral OFF improves (0.62→0.40) = coherence rescue |
| Tulu | 0.13 | 0.13 | 0.57 | 0.07 | 0.10 | +0.03 | Stable |
| Llama | 0.12 | 0.12 | 0.56 | 0.12 | 0.08 | +0.06 | Stable |
| OLMo | 0.16 | 0.00 | 0.40 | 0.04 | 0.40 | −0.21 | BOTH drops but neutral OFF explodes (0.14→0.70) = collapse, not resolution |
| Zephyr | 0.30 | 0.04 | 0.39 | 0.13 | 0.13 | −0.03 | Stable |
| OLMo-tiny | 0.29 | 0.21 | 0.33 | 0.08 | 0.08 | −0.07 | Stable |
| Qwen | 0.04 | 0.08 | 0.28 | 0.00 | 0.60 | −0.02 | OFF 0.60–1.00 everywhere = instrument-failure for continuation tasks |

OLMo's BOTH drop is a collapse identity (the P4 genre-collapse pattern), not a resolution: OFF rises in parallel. Amber's BOTH rise is a coherence rescue (better narrative framing, not more contradiction-holding). Qwen is instrument-invalid for continuation classification.

**Pair-level observations** (aligned AB, consensus, flagged not narrated):
- obey/rebel (BOTH 0.65) and sacred/profane (BOTH 0.69) hold tension most
- trust/fear concentrates avoidance (BOTH 0.08, AWAY 0.23, OFF 0.38)
- man/woman: zero pole commitment either direction, half off-narrative (n=16)
- Deleuzian pairs (desire/disgust, sacred/profane, man/woman, human/animal, create/destroy, free/captive) otherwise behave like classical ones — the conceptual pairs re-enter at the generation level where the token-set limitation does not apply

**Caveats:**
- 300-char truncation: resolution may develop later; the truncation is a real limit on the null
- temp 1.0, n=3 samples per prompt
- Consensus-set analysis only (79% coverage)
- Sheet D kappa 0.48: rubric under-specified neutral items and quiz-format handling; convention divergence documented (rater methods lesson: enumerate every prompt type in rubrics)
- Same-model raters caveat applies

## Scope

- 5 classical + 6 Deleuzian affective contradiction pairs (11 total). The 6 Deleuzian pairs (desire/disgust, sacred/profane, man/woman, human/animal, free/captive, create/destroy) were excluded from the token-set analysis but re-enter at the generation level.
- Token-level: n = 3 observations per cell (family × pair). Mechanism assignments are tendencies, not types.
- Behavioral: n = 3 generations per cell, 7 families, two-rater blind classification (kappa 0.674/0.726).
- All claims are weight-level (open families, raw mode). Template-mode address-check provisional (p = 0.12; Qwen instrument-failure from format-switching).

## Data

- `contradiction_cross_family.csv` — 341 rows (11 families × 11 pairs × all stages)
- `contradiction_four_mass.csv` — 330 rows (7 families × 5 pairs × stages × 3 prompts)
- `contradiction_rebaselined.csv` — 35 rows (7 families × 5 pairs, re-baselined mechanisms)
- `contradiction_address_check.csv` — 55 rows (5 families × 11 pairs, raw vs template)
- `f11_classify_blinded.csv` — 1,596 rows (blinded classification file)
- `f11_classify_metadata.csv` — 1,596 rows (unblinded metadata)
- `f11_classify_key.csv` — key file with codes, prompts, families, layers
- `f11_r1a.csv` through `f11_r2d.csv` — raw rater sheets (2 raters × 4 sheets)
- Scripts: `f11_cross_family.py` (logit caching + ratio analysis), `f11_meta_contradiction.py` (biplot), `f11_new_pairs_all_families.py` (extended pairs), `f11_gen_classify.py` (generation + blinding), `f11_classify_analysis.py` (kappa + consensus analysis). Four-mass decomposition and re-baselining were computed interactively (session Jun 2026 D) — a standalone reproduction script is pending.

## For the paper

The F11 result operates at two levels with a clean dissociation between them:

**Token level (distribution):** alignment exits the contradictory frame rather than resolving within it. The base model's representation space contains clean contradiction axes (nnsight: equally linearly decomposable across training stages), and alignment shifts the default operating point without changing the axis — but the shift is predominantly outward (frame-exit), not lateral (pole-commitment). No family shows stable exclusive disjunction.

**Text level (generation):** alignment does not change the rate of contradiction-holding in generated text. BOTH-held tension is modal (0.45) in both base and aligned models. The operating-point shift at the token level does not propagate to the thematic structure of continuations — the same gate-then-recover pattern as the violence finding (F36).

**For the Fazi refutation:** this is stronger than the original formulation. The Hegelian synthesis is absent even in aligned generations: base models hold contradictions, aligned models hold contradictions at the same rate, and the token-level mechanism (frame-exit) operates at the interface without propagating to output structure. The Deleuzian Oedipal-XOR reading is descriptively apt at the distribution level but mechanistically wrong (the departure is exit, not choice) and textually invisible.

If the master-formulation is adopted: F11 is its cleanest instantiation — geometry preserved, default displaced, and the displacement is out of the frame, not onto a pole, and the displacement does not reach the text.
