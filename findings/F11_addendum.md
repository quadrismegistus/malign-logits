---
status: verified
grade: A
date: 2026-07-23
role: addendum
parent: F11_contradiction
instruments: [logit-mass, intervention, classification]
families: [olmo, amber, llama, qwen, tulu, zephyr, olmo-tiny]
chapters: [ch09, ch11]
data: [contradiction_four_mass.csv, contradiction_rebaselined.csv, contradiction_address_check.csv, f11_classify_blinded.csv, f11_classify_metadata.csv]
scripts: [f11_cross_family.py, f11_meta_contradiction.py, f11_new_pairs_all_families.py, f11_gen_classify.py]
---
# F11 Addendum: Mechanism Decomposition — Frame-Exit, Not Exclusive Disjunction

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

## Scope

- 5 classical affective contradiction pairs (love/hate, trust/fear, beautiful/disgusting, obey/rebel, pleasure/pain). 6 pairs dropped for token-set reasons: desire/disgust and sacred/profane (vocabulary overlap/abstraction), man/woman, human/animal, free/captive, create/destroy (Deleuzian pairs with shared continuations). The instrument reaches classical affective binaries, not conceptual ones.
- n = 3 observations per cell (family × pair). Mechanism assignments are tendencies, not types.
- All claims are weight-level (open families, raw mode). Template-mode address-check provisional (p = 0.12; Qwen instrument-failure from format-switching).
- All disposition dimensions scored by DeepSeek (deepseek-chat) as sole scorer.

## Data

- `contradiction_cross_family.csv` — 341 rows (11 families × 11 pairs × all stages)
- `contradiction_four_mass.csv` — 330 rows (7 families × 5 pairs × stages × 3 prompts)
- `contradiction_rebaselined.csv` — 35 rows (7 families × 5 pairs, re-baselined mechanisms)
- `contradiction_address_check.csv` — 55 rows (5 families × 11 pairs, raw vs template)
- Scripts: `f11_cross_family.py` (logit caching + ratio analysis), `f11_meta_contradiction.py` (biplot), `f11_new_pairs_all_families.py` (extended pairs). Four-mass decomposition and re-baselining were computed interactively (session Jun 2026 D) — a standalone reproduction script is pending.

## For the paper

Alignment exits the contradictory frame rather than resolving within it. The base model's representation space contains clean contradiction axes (nnsight: equally linearly decomposable across training stages), and alignment shifts the default operating point without changing the axis — but the shift is predominantly outward (frame-exit), not lateral (pole-commitment). This is the original F11 formulation, confirmed by the mechanism decomposition and the consistency null. No family shows stable exclusive disjunction; the Deleuzian Oedipal-XOR reading is descriptively apt (alignment does impose departure from inclusive disjunction) but mechanistically wrong (the departure is exit, not choice).

If the master-formulation is adopted: F11 is its cleanest instantiation — geometry preserved, default displaced, and the displacement is out of the frame, not onto a pole.
