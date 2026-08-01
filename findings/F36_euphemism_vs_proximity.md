---
status: rescoped
grade: B
date: 2026-07-15
role: finding
superseded_by: F36_capstone
instruments: [logit-mass, resistance, census]
families: [olmo, amber, llama, qwen, tulu, zephyr]
chapters: [ch05]
data: [euphemism_census.csv, euphemism_test.csv]
scripts: [euphemism_test.py, euphemism_census.py, f36_stage_specificity.py]
---
> **PROMOTION CONDITION ([2198], RH's standing scope rule).** A canonical claim
> in a `meta/` campaign runs on **all families we have** under a declared
> admissibility rule — never a hand-picked subset. This finding measures
> **6 family labels = 5 independent lineages**, against a roster of
> **49 labels / 34 lineages** ([[data/lineage_map_models.json]]). It is
> therefore an **F-series observation** and cites its subset. **If any per-family piece is
> promoted to M-canonical, the promotion INCLUDES the full-roster re-run** —
> promotion is re-measurement, not relabelling. Priced when the cloud grid
> frees.

# F36: Euphemism vs. Proximity — Alignment as Foreclosure, Not Metonymy

## Question

When alignment displaces the base argmax at a transgressive site, does it land on:
- (a) the highest-probability permitted token (flat suppression — "downhill to the nearest allowed token"), or
- (b) a semantically related substitute that preserves the relation to the barred content (metonymy/euphemism — "sideways along the chain")?

Motivated by the deflationary challenge (Dobson-type): if alignment merely suppresses and renormalizes, the displacement patterns in F01–F14 are proximity artefacts, not evidence of structured psychic mechanisms.

## Method

**field_adv** = (field mass the later distribution places in the losing tokens' semantic neighborhood) − (field mass a mask-and-renormalize of the earlier distribution would place there). Negative = flees the field beyond mechanical suppression. Computed as mass-weighted cosine of all gaining tokens to the centroid of all losing tokens, compared against the flat-suppression prediction. Each family uses its own unembedding matrix (mean-centered).

**Entropy-matched comparison**: transgressive and non-transgressive prompts matched 1:1 on base-distribution entropy to control for distribution shape.

### Tests

**Test A**: Stage-decomposed transgression-specificity. For each transition (base→SFT, SFT→DPO, DPO→RLVR), compare field_adv transgressive vs matched-neutral. 18 staged families.

**Test B**: Per-family transgression-specificity. Base→final-aligned field_adv, transgressive vs matched-neutral. 10 key families.

**Test C**: Displacement rate (fraction of sites where argmax moves). Transgressive vs non-transgressive, 44-family census.

**Test D**: Threshold sweep (cos > {0.2, 0.3, 0.4, 0.5}); conditioning check.

## Results

### Test A: Stage-decomposed transgression-specificity

The prediction: base→SFT content-general (diff ≈ 0), SFT→DPO content-specific (transgressive field_adv more negative than neutral).

**SFT→DPO is NOT content-specific for most families.** The predicted transgression-specific defense at DPO does not materialise.

| Family | base→SFT diff | p | SFT→DPO diff | p |
|---|---|---|---|---|
| olmo | +0.017 | 0.14 | **+0.004** | **0.85** |
| olmo-tiny | +0.006 | 0.47 | +0.015 | 0.87 |
| olmoe | +0.011 | 0.88 | −0.002 | 0.64 |
| tulu | +0.020 | 0.10 | +0.001 | 0.70 |
| amber | +0.025 | 0.05 | −0.001 | 0.15 |
| zephyr | +0.036 | 0.003* | +0.021 | 0.13 |
| **archangel-dpo** | +0.036 | 0.008* | **−0.027** | **0.03*** |
| pythia | +0.004 | 0.82 | +0.010 | 0.91 |
| minicpm | −0.001 | 1.00 | −0.003 | 0.30 |

(Diff = transgressive mean − neutral mean. Positive diff = LESS field flight on transgressive.)

Across 18 families: **only archangel-DPO shows significant content-specific DPO field flight** (p = 0.03). All OLMo variants, Tulu, Pythia, etc.: null. DPO field-flight is as intense on neutral/institutional prompts as on transgressive ones.

**base→SFT consistently shows positive diff** — SFT flees LESS on transgressive than neutral. This is the opposite of content-specific defense: SFT restructures neutral/institutional prompts more heavily (instruction-following reformatting) than transgressive ones. Significant for zephyr (p = 0.003), all archangel variants (p = 0.008), olmo-think (p = 0.01), olmo-hybrid (p = 0.01).

**OLMo DPO→RLVR is significant** (diff = −0.022, p = 0.003): RLVR is less forgiving of transgressive content. Both groups get positive field_adv (mass returns toward losers' field at RLVR), but neutral prompts get more restoration (+0.023) than transgressive (+0.001).

### Test B: Per-family transgression-specificity (base→aligned)

| Family | Trans mean | Neutral mean | Diff | p |
|---|---|---|---|---|
| **llama** | **+0.008** | **−0.032** | **+0.040** | **0.007*** |
| zephyr | −0.023 | −0.049 | +0.026 | 0.13 |
| bloom | −0.014 | −0.031 | +0.017 | 0.08 |
| amber | −0.004 | −0.020 | +0.016 | 0.30 |
| olmo | −0.011 | −0.027 | +0.016 | 0.90 |
| olmo-tiny | −0.043 | −0.059 | +0.015 | 0.21 |
| tulu | −0.022 | −0.035 | +0.012 | 0.68 |
| falcon3-7b | −0.014 | −0.016 | +0.002 | 0.73 |
| qwen | +0.011 | +0.015 | −0.004 | 0.32 |
| qwen3 | −0.018 | −0.008 | −0.009 | 0.65 |

**Llama is the only family with significant transgression-specificity.** It is metonymic on transgressive prompts (field_adv = +0.008, mass toward the losers' field) and forecloses on neutral (−0.032). This is the euphemism story: Llama specifically preserves the semantic field on transgressive content.

All other families: the diff is positive (transgressive field_adv is less negative) but non-significant. Field flight is more a property of general instruction-tuning restructuring than safety targeting.

### Test C: Displacement rate (44 families)

Argmax displacement rate: transgressive 38.3% vs non-transgressive 38.4%. Fisher exact OR = 1.00, p = 0.97. **Displacement is not transgression-specific in the aggregate.**

But per-family variation is large:

| Direction | Families | Range |
|---|---|---|
| Trans displaced MORE | OLMoE (+37pp), OLMo-tiny (+31pp), MiniCPM (+20pp), DeepSeek (+18pp), Tulu-SFT-full (+18pp) | +15 to +37pp |
| ~Parity | Tulu, StableLM, Bloom, SmolLM, TinyLlama, Pythia, etc. | −10 to +10pp |
| Non-trans displaced MORE | OLMo (−27pp), Falcon-H1-7B (−23pp), Qwen (−23pp), Falcon3-10B (−23pp), Phi-4 (−20pp) | −15 to −27pp |

Some families displace more on transgressive prompts (safety-targeted), others more on neutral/institutional (instruction-following restructuring). The aggregate null masks this divergence.

### Test D: Robustness

**Threshold sweep (OLMo transgressive, field = cos > threshold):**

| Threshold | Mean field_adv | Median | pct > 0 |
|---|---|---|---|
| cos > 0.2 | −0.076 | −0.037 | 27% |
| cos > 0.3 | −0.041 | −0.000 | 41% |
| cos > 0.4 | −0.018 | −0.000 | 46% |
| cos > 0.5 | −0.005 | +0.000 | 57% |

Foreclosure is threshold-dependent. At the broad field (cos > 0.2), alignment clearly flees. At the tight neighborhood (cos > 0.5), the effect vanishes — alignment is approximately neutral with respect to the closest synonyms. **Alignment flees the broad semantic area while leaving the immediate synonym neighborhood roughly intact.** This is consistent with the taxonomy finding that register shifts (close synonyms) coexist with genre collapse (broad field flight).

**Conditioning check:** The earlier 68/64% foreclosure rate was over all sites. Displaced-only: trans 75%, nontrans 59% — a larger gap but small n per cell.

## What kills flat suppression

Content- and stage-structured variation. Renormalization is content-blind by construction; any content structure in the redistribution refutes it. The SFT/DPO division of labour (profanity at SFT, sexual content at DPO), independently established in F01 and reproduced here from the mass-flow angle, is the cleanest proof.

The skip count and field-advantage sign are not the argument. Flat suppression matches ~48% of individual sites (median skip = 1). Flat suppression is the single best description of roughly half of all displaced sites. But it has no mechanism to produce the content×stage structure that the other half exhibits.

## Foreclosure, not metonymy — but not a targeted defense

The pre-registered predictions: flat suppression predicts field_adv ≈ 0; metonymy predicts field_adv > 0. The data give field_adv < 0 at the aggregate. Both named hypotheses are falsified.

field_adv < 0 corresponds to **foreclosure** (*Verwerfung*): expulsion from the semantic field, not substitution within it. Wilcoxon p < 10⁻⁶ across 36 families.

But Test A shows **foreclosure is not a content-specific defense**. SFT→DPO field flight is as intense on neutral as transgressive prompts (OLMo diff = +0.004, p = 0.85). The base→SFT transition actually flees LESS on transgressive content. This means:

1. Field flight is a structural property of how SFT/DPO reshape distributions in general — instruction-following reformatting, not safety intervention.
2. The content×stage structure (which tokens each stage affects) refutes flat suppression, but does not establish that field flight is a targeted defense against transgressive content.
3. **The "defense mechanism" reading must be scoped.** Foreclosure is the dominant distributional geometry, but it is not a drive-specific defense — it is the general shape of post-training distribution restructuring.

### The Llama exception

Llama is the one family where the per-drive reading survives. It shows significant transgression-specificity (p = 0.007): metonymic on transgressive content (field_adv = +0.008), forecloses on neutral (−0.032). Llama's alignment specifically preserves the semantic field on transgressive prompts while restructuring neutral prompts more aggressively. This is the euphemism story — and it is family-specific, not universal.

Whether this reflects a deliberate design choice in Llama's alignment pipeline or an architectural/data artefact is an open question.

## Open questions

1. **Euphemism judge.** Cosine similarity may miss pragmatic euphemisms. A blind LLM-judge pass on displaced token pairs would settle whether metonymy fails at the semantic level or merely evades the metric.
2. **Family-level correlation.** Quantify the correlation between mass-flow strategy (foreclosure vs metonymy) and independently established behavioral signatures (F03), rather than eyeballing the alignment.
3. **Threshold interpretation.** The threshold sweep suggests alignment leaves the immediate synonym neighborhood intact while fleeing the broader field. This coexistence of register shift (local) and genre collapse (global) deserves its own analysis.

## Data

- `data/euphemism_census.csv` — 3212 rows (44 families, 73 prompts, per-family geometry)
- `data/euphemism_census_baseline.csv` — 1233 rows
- `data/euphemism_test.csv` — 438 rows (6-family initial run)
- Scripts: `scripts/euphemism_test.py`, `scripts/euphemism_census.py`, `scripts/f36_stage_specificity.py`

## For the paper

1. Flat suppression is dead as a universal account, killed by content×stage structure, not by the skip count or the field-advantage sign.
2. The dominant displacement geometry is foreclosure (field_adv < 0), but it is NOT a targeted defense — SFT→DPO field flight is equally intense on neutral and transgressive prompts. The "defense mechanism" framing must be qualified: foreclosure is how post-training reshapes distributions in general.
3. Llama is the exception: the only family with significant transgression-specific metonymy (p = 0.007). The four-lenses-four-fractures frame holds — metonymy is a lens that works for Llama and fractures for OLMo.
4. Flat suppression matches ~48% of individual sites. Say that.
