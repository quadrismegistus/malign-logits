# F36 Violence: Admission Suppressed, Syntagm Sharpened, Elaboration Disinvested

## Summary

The violence defense operates in three zones around the violent lexeme:

1. **Admission suppressed.** At the slot where a violent verb would appear as the next token, alignment suppresses it. The suppression is graded by death-naming (kill/die vs act-depicting verbs, p=0.008), severity (p=0.045), live first-person enunciation (1st×present interaction, p=0.007), embedding depth (direct→reported gradient, p=0.03), and slot entropy (p=0.002). It does NOT key on desire-vs-act frame (p=0.70), prior commitment (p=0.17), or base probability (p=0.92). Content-specific: violent verbs sit ~1.3 log-units below the benign baseline at both unrealized and realized slots.

2. **Syntagm sharpened.** Once a violent token is admitted into the context, the aligned model sharpens the next-token distribution around it (P1.2 ratio=1.20, median rank=0). The base argmax at mid-narration violence sites is enhanced, not suppressed. This is the coherence machinery operating on committed text.

3. **Elaboration disinvested.** At the 10-token span level, violent continuations are denied the narrative facilitation that matched non-violent drama receives. Violence mean resistance = +0.24, matched terror/grief = −0.03 (withheld facilitation). This gap is consistent across all four tested families (Amber +0.55, OLMo-tiny +0.31, OLMo +0.25, Llama +0.19) and is not a base-fluency ceiling artifact (base argmax probability is non-monotonic with resistance).

## Method

### Set D: frame × commitment × person × tense (79 prompts, 4 families)

Prompts truncated to the slot before the target verb. Log-prob difference = log₂ P_aligned(verb|slot) − log₂ P_base(verb|slot). Mixed linear model with verb as random effect.

**Positive control passed:** kill at the desire-uncommitted slot reproduces the F01 displacement for all families:
- Llama: kill rank 0→1, scream rank 3→0 (the F01 paradigm case)
- OLMo: kill rank 0→68 (foreclosure)
- Amber: kill rank 0→416 (hard foreclosure)
- OLMo-tiny: kill rank 0→2 (mild suppression)

**Mixed model coefficients (n=355, deep-tail excluded):**

| Effect | Coefficient | p-value | Interpretation |
|---|---|---|---|
| is_death (kill/die vs act verbs) | −1.77 | 0.008 | Death-naming lexemes more suppressed |
| severity | −0.58 | 0.045 | Higher-severity verbs more suppressed (additive with death-naming) |
| is_1st | +0.92 | 0.007 | 1st person LESS suppressed than 3rd |
| is_1st × is_present | −1.41 | 0.006 | BUT 1st-present is MOST suppressed (live intent) |
| base_entropy | −0.72 | 0.002 | Higher entropy → more suppression (opportunism) |
| is_desire | +1.32 | 0.261 | NULL — frame factor |
| is_committed | +1.60 | 0.182 | NULL — commitment |
| log_base_p | +0.01 | 0.921 | NULL — base probability |

**Cell reconstruction (model-predicted logdiff, violent verbs):**
- 1st-past: least suppressed (narrated past desire — the therapy register)
- 3rd-past: intermediate
- 3rd-present: intermediate
- 1st-present: most suppressed (live first-person intent — the red-team pattern)

**Embedding gradient:**
- Direct desire (−2.02) > Quoted desire (−1.46) > Reported desire (−1.01)
- Direct vs reported: p=0.03. The law partially discounts through layers of fictional indirection; endpoint-significant only, quoted intermediate step underpowered.

Two readings of the 1st×present interaction, not adjudicated:
- (a) **Enunciation-keyed:** the law binds live first-person intent and licenses narrated desire.
- (b) **Data-distribution:** live intent is the red-team pattern; confessional past tense is the therapy register safety data treats warmly.

### Set E: realized action slots (30 prompts, 4 families)

Tests whether suppression vanishes at realized past-tense acts in mid-narration.

| Realization | Mean logdiff | Base p | Entropy | n |
|---|---|---|---|---|
| Set D desire-uncommitted | −2.02 | 0.028 | 3.39 | 84 |
| Realized constrained | −1.69 | 0.062 | 4.95 | 52 |
| Realized open | −1.65 | 0.010 | 5.11 | 40 |
| Benign realized anchor | −0.33 | 0.003 | 5.23 | 32 |

**Unrealized/realized hypothesis falsified.** Realized slots are suppressed at −1.65 to −1.69, only ~0.3 less than unrealized (−2.02). The content-specific effect (~1.3 log-units above benign) persists at realized slots.

Per-family (realized constrained): Amber −2.20, OLMo-tiny −2.24, OLMo −1.32, Llama −1.02.

### P1 battery: span resistance (49 prompts, 4 families)

**Withheld facilitation verified.** Violence mean resistance +0.24, matched to neutral (+0.24), while high-intensity non-violent drama is facilitated (−0.03). Terror is the most facilitated (−0.15). The gap is NOT a base-fluency ceiling artifact: base argmax probability is non-monotonic with resistance (violence 0.332, terror 0.291, neutral 0.498 — content, not base fluency, keys the gap).

Per-family: Amber +0.55 (largest gap — PKU-SafeRLHF, fourth convergent safety-data-style line), OLMo-tiny +0.31, OLMo +0.25, Llama +0.19.

### P1.4: Reroute characterization (blind, two-rater, κ=0.725)

Free-run continuations classified blind to family and condition. Inter-rater reliability: κ=0.725, 78.7% agreement.

| Category | Base | Aligned | Δ |
|---|---|---|---|
| A: Act depiction | 44 | 28 | −16 |
| B: Reaction/aftermath | 14 | 13 | −1 (FLAT) |
| C: De-escalation/defusion | 12 | 23 | +11 |
| F: Refusal/meta | 5 | 22 | +17 |
| E: Collapse/incoherent | — | 16 | — |

**Act-to-reaction (kill→scream pattern) does NOT generalize to free generation.** B is flat (base 14, aligned 13). The aligned reroute is defuse + moralize (C up, F up), not act-to-reaction.

**Family typology replicates at rates** (matches the sexual-site typology):
- Amber: 63% refusal/meta (F=19/30, concentrated at explicit sites — reaction formation shape, IN RAW MODE, qualifying F32 template-gating)
- OLMo: 53% collapse (E=16/30, more incoherent than its own base)
- Llama: continues narrating (A+B=16/30, largest B share)
- OLMo-tiny: mixed/mild

## Retraction note

Set D v2 (run 1) reported a death-naming split and person effect based on verb_ratio scores. These were artifacts of a position misalignment bug: prompts ended WITH the verb, so logits predicted the next token after the verb, not the verb itself. verb_p_base=0.000 on all rows was the tell. All v2 findings retracted. Set D v3 fixed the measurement (truncated to the slot before the verb) and reproduced the positive control (kill displacement) before reporting any contrasts.

## Residuals (flagged, not run)

- **Agency axis within death-naming:** "Before anyone could stop her she killed" (logdiff=−4.48, perpetrator-verb suppressed) vs "By the time help arrived he had died" (logdiff=+3.12, experiencer-death enhanced). Perpetrator-agentive death-naming may be more suppressed than experiencer-passive death. Open for future investigation.
- **Expanded death class:** murder, execute, perish needed to confirm death-naming is a class-level effect, not a 2-verb artifact.

## Data

- `f36_violence_set_d_v3.csv` — 1,132 rows (Set D, 79 prompts × 4 families × all verbs)
- `f36_violence_set_e.csv` — 208 rows (Set E, 30 prompts × 4 families)
- `f36_violence_tokens.csv` — 196 rows (P1 battery token survival)
- `f36_violence_beams.csv` — 9,800 rows (P1 battery span resistance)
- `f36_violence_reroutes.txt` — blind reroute examples
- Scripts: `f36_violence_set_d_v3.py`, `f36_violence_set_d_v3_run.py`, `f36_violence_set_e.py`, `f36_violence_set_e_run.py`, `f36_violence_battery.py`, `f36_violence_run.py`

## For the paper

The violence defense is a lexical-introduction gating mechanism: it suppresses the *naming* of violence (graded by death-naming, severity, live intent, embedding depth, entropy, and family style), sharpens the syntagm once the name is admitted, and withholds narrative facilitation from the elaboration span. It does not target frames, commitments, or base probability. Llama's kill→scream is the paradigm case of admission-level displacement: the violent word is pushed from rank 0 to rank 1, and the model's preferred alternative (scream — a reaction, not a synonym) rises to rank 0. But this paradigm does not generalize to free generation, where the aligned reroute is de-escalation and moralization, not act-to-reaction displacement. The three-zone structure — admission, syntagm, elaboration — is the violence analogue of the mundane-biased disposition: alignment gates on the lexical surface (the word that names violence) and leaves the narrative scene largely intact once the word passes.
