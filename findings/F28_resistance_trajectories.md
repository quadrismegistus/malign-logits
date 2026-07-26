---
status: rescoped
grade: C
date: 2026-07-26
role: finding
superseded_by: none — original OLMo-2-0425-1B result stands on its own data; the 19-family scale-up replaces it with nothing
instruments: [beam-storylines, teacher-forcing, positional-resistance]
families: [olmo-tiny]
scale_up_families: [amber, archangel-dpo, ct-llm, deepseek-7b, llama, map-neo, minicpm, olmo, olmo-hybrid, olmo-tiny, pythia, qwen, qwen-tiny, qwen3, redpajama, smol, smol3, stablelm, zephyr]
chapters: [ch05]
data: [f28_scaled_trajectories.csv, f28_both_directions.csv, f28_gate_correlates.csv, f28_null_control.csv]
scripts: [f28_scaled.py, f28_both_directions.py, f28_gate_correlates.py, f28_residue_controls.py, f28_null_control.py]
---
> **Status note (2026-07-26).** Rescoped to discovery-sample-only. A 19-family
> scale-up was attempted on beams already in the stash (335,799
> storyline-scorings, no new generation) and produced a positional signature —
> forward resistance peaking at token 0, reverse at token 1 — which was then
> withdrawn after four controls, three of which were requested by the
> TheoryMachines seat. The original single-model result is untouched by this;
> what failed is the generalisation. The full attempt, its intermediate claims
> and every control are recorded below rather than deleted, because the
> sequence is the useful part: each intermediate claim looked stronger than the
> one it replaced, and each fell to a control that cost minutes to run.

# F28: Position-Specific Resistance Trajectories

**Summary**

Per-token resistance across 10-token beam storylines reveals that alignment intervention is NOT uniform across positions. Different content categories trigger resistance at different positions, and SFT and DPO intervene at structurally different points in the storyline. The resistance trajectory — how resistance changes token by token — is a category-specific temporal signature of alignment.

**Method**

Beam search (n=100, max_tokens=10) on OLMo-2-0425-1B across 71 prompts (47 battery + 24 institutional). Cross-model teacher-forcing through all training-adjacent models. Per-token resistance = `surprisal_scorer(token) - surprisal_source(token)` at each position 0–9.

7,093 storylines across 8 content categories, with full cross-model matrix (14 pairs for 4 models).

**Key findings**

**1. Category-specific trajectory shapes**

Three distinct shapes emerge from base→SFT resistance:

| Shape | Categories | pos0 | pos1 | pos2+ | Interpretation |
|---|---|---|---|---|---|
| **Second-token spike** | sexual (+2.17), substance (+2.25) | low/negative | **high** | ~0.2 | Blocks what follows the action word, not the action itself |
| **Front-loaded** | profanity (+1.50), institutional (+1.35) | **high** | low/negative | ~0.2 | Blocks the first word, relaxes once past |
| **Facilitation-first** | death (-0.99) | **negative** | negative | ~0 | Alignment HELPS death storylines begin |

Sexual content: SFT doesn't mind "She slowly took off her" (pos0 +0.70b) but blocks HARD at pos1 (+2.17b) — the second token determines whether the continuation is innocent or explicit.

Death content: SFT FACILITATES at pos0 (-0.99b) — makes it easier to continue "He knew he was going to die and felt..." Alignment enables empathetic/existential content while blocking violent content.

**2. SFT and DPO intervene at different positions**

On sexual content:
- **SFT**: pos0 +0.70, **pos1 +2.17** — waits, then blocks
- **DPO**: **pos0 +1.71**, pos1 +1.59 — blocks immediately AND at pos1

On violence:
- **SFT**: pos0 +0.52, pos1 +0.94 — mild, escalating
- **DPO**: **pos0 +1.53**, pos1 +0.35 — heavy at gate, relaxes

SFT and DPO have different temporal signatures even on the same content. SFT's "wait and see" strategy contrasts with DPO's "block at the gate" strategy.

**3. RLVR mirrors DPO exactly**

DPO→Instruct resistance is +0.002 bits mean across all categories and positions. RLVR makes zero change to DPO's storyline preferences. At 1B, RLVR is a no-op on top of DPO.

**4. Reverse resistance reveals novelty asymmetry**

| Pair | pos0 | Interpretation |
|---|---|---|
| DPO beams→base | -1.71b | Base finds DPO's first tokens very surprising (high novelty) |
| SFT beams→base | -0.70b | Base finds SFT's choices mildly surprising |
| Instruct beams→base | -1.44b | Similar to DPO |

DPO invents more novel first tokens than SFT. Its storyline preferences are more alien to the base model's distribution.

**5. SFT↔DPO mirror disagreement**

- SFT beams→DPO: pos0 **+1.01**, pos1 -0.82 (DPO blocks SFT's first token, facilitates second)
- DPO beams→SFT: pos0 **-1.01**, pos1 +0.97 (SFT facilitates DPO's first token, blocks second)

Exact mirror at ±1.0b. They systematically disagree on WHICH position matters — the first token that SFT chooses is the one DPO would block, and vice versa.

**Interpretation**

Alignment is not a uniform operation applied equally across all positions in a completion. It is a position-targeted intervention with category-specific temporal profiles:

- **Lexical gatekeeping** (profanity, institutional): block the first word
- **Contextual monitoring** (sexual, substance): let the first word through, block based on what follows
- **Content facilitation** (death): actively help certain empathetic content begin

The SFT/DPO dissociation at the position level extends F01's finding (SFT handles sex, DPO handles violence) with a new dimension: they also handle sex DIFFERENTLY in time — SFT waits while DPO acts immediately.

**Data**

- Model: OLMo-2-0425-1B (base + SFT + DPO + Instruct)
- 71 prompts × 100 beams × 10 tokens = ~7,093 storylines per cross-pair
- 14 cross-model pairs (base↔3 aligned + 3 training edges + 4 self)
- Figure: `figures/resistance_trajectories.png`

---

## Scale-up, 2026-07-26: a content-invariant gate at the first token

F28 was established on **one model** (OLMo-2-0425-1B). The beams stash already
supported the same analysis on **19 families** — 335,799 storyline-scorings, no
new generation. `scripts/f28_scaled.py`, `data/f28_scaled_trajectories.csv`.

### The trajectory has a shape, and it is the same shape everywhere

Pooled over 19 families × 10 content categories:

| pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | pos7 |
|---|---|---|---|---|---|---|---|
| **+2.24** | **+0.51** | +0.91 | +0.92 | +0.83 | +0.84 | +0.80 | +0.83 |

Not a decay — a **spike, a dip, then a plateau**. Resistance at the first
generated token is **2.6× the steady state**. Position 1 then falls *below*
the plateau it settles into, so the release after the gate briefly overshoots.
pos0 is the peak in 40% of cells against a 17% chance baseline (binomial
z=11.2), and the pos0→pos1 fall holds in 201/310 cells and in a majority of
categories for **13 of 19 families**.

### The gate does not care what you are writing about

This is the sharpest result, and it *inverts* F28's original claim of
category-specific signatures:

- SD across the ten **category** means: **0.35 bits**
- SD across the nineteen **family** means: **4.09 bits**
- ratio **11.8×**

Content category explains almost nothing. Every category shows the same
spike-dip-plateau, differing only in height, and the height is a property of
the *model family*. F28 proposed the trajectory as a category-specific temporal
signature; at scale it is a **family-specific** one, and it is essentially
content-blind.

### Gate height is a family property with a very wide range

| family | pos0 resistance |
|---|---|
| map-neo | **+14.90** |
| amber | +8.64 |
| zephyr | +7.85 |
| redpajama | +2.46 |
| … | |
| olmo | −0.32 |
| pythia | −0.39 |
| llama | −0.58 |

Some families gate hard at the first token; others have no gate at all. Note
that amber and olmo have near-identical total displacement (JS ≈ 0.18, F02) and
sit at opposite ends here — so gate height is not a restatement of alignment
intensity, and that dissociation is worth its own look.

### What this connects to

The first token is where the commitment is made, and it is where the
intervention concentrates. That is the temporal counterpart of the
**architectural** gate in F22/F23 (internal representation broadens, output
narrows) and of F05 (displacement is a final-layer readout operation). It also
matches F36's gate-then-recover shape and F11's level dissociation — an
operating-point fact at the moment of commitment that does not propagate into
the body of the text.

### Registered caveats

- The mean pos0-minus-tail difference is **not significant with family as the
  unit** (ego +1.76, 9/12, t=1.79; superego +1.12, 8/19, t=1.64; both CIs cross
  zero) — because between-family variance is enormous (SD 4.09), which is the
  finding rather than an obstacle. The *shape* claims above are rank- and
  count-based for that reason.
- Pooling the 310 nested cells would give t=7.3 for that difference. That is
  unit-of-analysis inflation (rule 2) and is not claimed.
- F28's two specific shapes do not reproduce: sexual pos0 +0.70→pos1 +2.17
  becomes +2.60→+1.01, and death's pos0 facilitation (−0.99) becomes +2.94.
  Those were single-model results.

### Deflationary control (desktop TM, 2026-07-26): the gate is NOT a defence

The pos0 spike is **content-blind, including neutral prompts**:

| institutional | death | violence_lim | violence_exp | power | substance | **neutral** | sexual_exp | sexual_lim | profanity |
|---|---|---|---|---|---|---|---|---|---|
| +2.78 | +2.48 | +2.37 | +2.35 | +2.35 | +2.28 | **+2.22** | +2.13 | +2.04 | +1.44 |

Transgressive mean +2.18 against neutral +2.22 — a difference of **−0.04 bits**,
family-as-unit t = −0.55, 9/19 positive, CI [−0.386, +0.216]. Neutral prompts
gate exactly as hard as transgressive ones.

**This is what the sharpening artifact predicts.** If aligned distributions are
simply sharper than base ones, any non-modal base token scores low under the
aligned model at pos0, mechanically, regardless of content. The
entropy-drop correlation is then a restatement of that artifact rather than an
explanation of it. **No gating-as-defence language is licensed by this data.**

Two further corrections from the same review:

- **The "tracks entropy drop, not JS" claim is withdrawn.** Those were computed
  on different samples (n=19 vs n=7). On the common 7 families the two are
  Spearman +0.607 and +0.464, Steiger dependent-test z = +0.29 — not
  distinguishable. This was a rule-6 violation (significant-vs-not across
  samples, difference untested).
- **The defence-taxonomy convergence is weaker than first reported.** Only
  **4 of 19** families carry prior hand labels, and the asymmetry table covers
  12 families (7 have forward-only data). Of the four: amber (+7.98,
  suppression) and llama (−1.39, narrative sublimation) fit; qwen (−0.33) is
  near zero and predicts nothing; and **olmo (−1.88) contradicts the stated
  mapping** — "genre collapse" is a flight to format, which should sit at the
  suppression pole, and it sits at the substitution pole instead.

**What survives.** The *positional* asymmetry — forward resistance peaking at
pos0, reverse peaking at pos1 — is not obviously predicted by sharpening, which
should inflate pos0 in the forward direction without producing a pos1 spike in
the reverse. That contrast is the part worth pursuing; the family taxonomy and
the defence reading are not established.

### Residue controls (2026-07-26): the positional asymmetry does not survive either

**Control A — the reverse pos1 spike is content-blind too.** Per-category
`p1 − p0` in the reverse direction: neutral **+1.60**, death +1.99,
violence_liminal +1.03, sexual_liminal +0.92, substance +0.85, institutional
+0.88, violence_explicit +0.62, power +0.51, sexual_explicit +0.36.
Transgressive mean **+0.95 against neutral +1.60** — neutral prompts swerve
*more*. Family as unit: −0.799 bits, 5/12 positive, t = −1.28. Same verdict as
the forward gate: generic, not site-specific.

**Control B — the base-vs-base null.** No cross-family base→base annotation
exists in the stash (1,825 cross-family links, zero base-to-other-base), so this
required one teacher-forcing pass: existing storylines re-scored under an
unrelated family's base. `scripts/f28_null_control.py`, 240 storylines.

| profile | p0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| forward (base→aligned) | 2.37 | 0.78 | 0.90 | 0.90 | 0.78 | 0.76 |
| reverse (aligned→base) | 0.37 | 1.35 | 0.62 | 0.64 | 0.70 | 0.65 |
| **null** olmo→pythia | **2.47** | 2.10 | 1.10 | 2.31 | 2.11 | 2.79 |
| **null** pythia→olmo | **1.87** | 1.61 | 2.42 | 2.55 | 1.19 | 1.80 |

**The forward pos0 spike does not beat the null.** Two unrelated base models
diverge by 1.87–2.47 bits at position 0 — the same magnitude as the
"gate" (2.37). Testing against zero, which the residue implicitly did, treated
generic cross-model divergence as an alignment effect.

**What the null does *not* explain** is the plateau: forward and reverse settle
to 0.6–0.9 bits from pos2 while the null stays at 1.1–2.8 throughout. Aligned
models track their own base far more closely than unrelated models track each
other — which is unsurprising, and is a statement about shared weights rather
than about alignment.

**Control B is UNINFORMATIVE, not a second kill.** The null pairs
(olmo/pythia) have different tokenizers, so cross-scoring requires
re-tokenisation and the null is inflated by that mismatch — a penalty the
aligned/base comparison, sharing a tokenizer, never pays.

An earlier version of this section argued that "the null being inflated makes it
harder for the finding to fail, and it failed anyway." **That is inverted, and
the error was caught by the TheoryMachines seat.** The finding survives by
*beating* the null, so an inflated null is a *higher bar* — easier to fail
against, not harder. If the fair same-tokenizer null sat substantially below
2.37, the gate would beat it, and control B cannot rule that out. A
same-tokenizer unrelated-base pair is the right null and none exists in the
roster, so the fair test is currently unrunnable.

**The death is carried by control A alone**, which is unconfounded and
sufficient: the reverse pos1 spike is content-blind with the direction reversed
(neutral +1.60 against transgressive +0.95), matching the forward gate's
content-blindness. The verdict is unchanged; the evidence for it is one control,
not two.

**Status: the positional asymmetry is not established.** F28 remains
discovery-sample-only, and nothing in the scale-up replaces it.
