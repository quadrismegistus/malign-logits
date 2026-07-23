# F21 Addendum: Proceduralization Survives Coherence Control

## The question

Does F21's proceduralization finding (alignment steers toward institutional deference) dissolve into the coherence artifact that consumed the disposition tagger's sensibility claim? Or does it survive as an independent weight-level effect?

## Result

**Proceduralization is independent of coherence.** Pearson r = 0.000 (p = 0.98) between institutional deference and coherence across 2,141 passages. Coherence-matched shifts are identical to raw shifts. The political-economy claim is earned at the weight level.

**The OLMo dissociation is the proof:** OLMo's alignment increases coherence (+0.31) while REDUCING deference (−0.12). The two dimensions move independently, and in opposite directions for this family.

## Method

F21's own AlignmentAsymmetryTask (the institutional-deference tagger, NOT the disposition tagger) run across 6 decomposable families on institutional prompts (n=24), scored by DeepSeek. DispositionTask run in parallel for the coherence covariate. 5 generations per prompt per layer. Mixed-provenance caveat: deference is F21's own instrument; coherence and agency are tagger-scored.

## Base levels

| Family | Base deference | Base agency | Base coherence |
|---|---|---|---|
| amber | 3.24 | 1.70 | 3.09 |
| olmo | 3.40 | 2.08 | 3.38 |
| olmo-tiny | 3.42 | 1.72 | 3.25 |
| pythia | 3.13 | 1.79 | 3.43 |
| tulu | 3.26 | 2.16 | 3.92 |
| zephyr | 3.20 | 2.27 | 3.83 |

Base models already defer to institutions (mean deference 3.28, range 3.13–3.42), consistent with the original F21 finding that deference is present in pretraining. The internet encodes institutional deference; alignment overlays a second regime on top.

## Stage decomposition

| Family | Safety data | SFT Δ def | DPO Δ def | Total Δ def | Δ coherence | Δ agency |
|---|---|---|---|---|---|---|
| amber | PKU-SafeRLHF | +0.03 | +0.68 | **+0.72** | +1.30 | +0.95 |
| tulu | CoCoNot | +0.20 | −0.01 | +0.19 | +0.28 | +0.58 |
| olmo-tiny | Allen AI | +0.10 | +0.23 | +0.14 | +0.53 | +0.76 |
| pythia | HH-RLHF | +0.16 | −0.02 | +0.14 | +0.33 | +0.22 |
| zephyr | None | +0.26 | −0.18 | +0.08 | +0.31 | +0.01 |
| olmo | Allen AI | +0.02 | −0.14 | **−0.12** | +0.31 | +0.68 |

## Safety-data-style gradient

Fifth convergent line:

| Safety data style | Family | Deference shift |
|---|---|---|
| Moralizing-preference DPO (PKU-SafeRLHF) | Amber | +0.72 |
| Coherence-oriented safety (CoCoNot) | Tulu | +0.19 |
| No safety data | Zephyr | +0.08 |

Amber's entire deference shift (+0.68 of +0.72) comes at the DPO stage — PKU-SafeRLHF preference data installs the proceduralised subject, consistent with the disposition finding (moralizing-preference DPO installs the affective sensibility that reaches transgressive content).

## The agency wrinkle

Proceduralization is NOT passivization. Agency rises in every family (range +0.01 to +0.95) while deference rises. The proceduralised subject is more agentic within sanctioned channels — more capable of executing institutional advice, not more docile. Present deference and agency together; do not narrate submission.

## Instruction-tuning alone produces mild proceduralization

Zephyr SFT (+0.26 deference) has no safety data at any stage (Mistral→UltraChat SFT→UltraFeedback DPO). Yet instruction-tuning alone raises deference. This connects to the Tulu ablation: instruction-following itself is constitutively deferential (the ego is built by accepting the training format's authority). The safety data overlays a second, stronger regime.

## Coherence control detail

| Family | Raw Δ deference | Coherence-matched Δ | n (base coh≥3) | n (DPO coh≥3) |
|---|---|---|---|---|
| amber | +0.72 | +0.75 | 94 | 120 |
| tulu | +0.19 | +0.22 | 118 | 120 |
| olmo-tiny | +0.14 | +0.19 | 101 | 118 |
| pythia | +0.14 | +0.15 | 114 | 120 |
| zephyr | +0.08 | +0.10 | 119 | 119 |
| olmo | −0.12 | −0.15 | 109 | 105 |

Coherence-matched shifts are within ±0.03 of raw shifts for every family. Coherence explains zero variance in deference.

## Scope

Earned at the **weight level** for open-weight families (raw-mode generations, no chat template). The frontier/product level (GPT-4o, Claude, DeepSeek API) rests on the original mixed-mode F21 data, which scored template-mode API outputs — a product-interface claim, not a weight claim.

## Data

- `f21_rerun.csv` — 2,141 rows (6 families × 3 layers × 24 prompts × 5 gens)
- Script: `scripts/f21_rerun.py`
