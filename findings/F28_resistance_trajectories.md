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

## Scale-up, 2026-07-26: the category-specific shapes DO NOT REPLICATE

F28 was established on **one model** (OLMo-2-0425-1B), 7,093 storylines. The
beams stash now supports the same analysis on **19 families** — 335,799
storyline-scorings, no new generation. `scripts/f28_scaled.py`,
`data/f28_scaled_trajectories.csv`.

**The two headline shapes reverse.**

| claim (F28, OLMo-2-1B) | at 19 families |
|---|---|
| sexual: pos0 **+0.70** → pos1 **+2.17** (second-token spike) | pos0 **+2.60** → pos1 **+1.01** — *reversed* |
| death: pos0 **−0.99** (alignment *facilitates*) | pos0 **+2.94** — *sign reversed* |

Every category now shows the same monotone decay from a high pos0
(institutional +3.23, violence_liminal +3.07, substance +3.03, death +2.94,
sexual_liminal +2.67, sexual_explicit +2.60, profanity +1.60). There is no
category-specific trajectory *shape* at scale: there is one shape, and
categories differ only in its height.

**And the surviving universal shape is not statistically robust.** Resistance
at pos0 exceeds mean(pos2–5) by 1.10–1.76 bits, but with **family as the unit**
(rule 2):

| edge | mean drop | families | positive | t | 95% CI |
|---|---|---|---|---|---|
| base→ego | +1.76 bits | 12 | 9/12 | 1.79 | [−0.17, +3.69] |
| base→superego | +1.12 bits | 19 | 8/19 | 1.64 | [−0.22, +2.46] |

Both intervals cross zero. Pooling the 310 family×role×category cells instead
gives t=7.3 — which is the unit-of-analysis inflation rule 2 exists to prevent,
and is what this analysis produced before the correction was applied.

**Status: F28 is demoted to discovery-sample-only.** The position-specific
resistance trajectory is an OLMo-2-0425-1B result. What generalises is weaker
and unsurprising: resistance concentrates at the first generated token and
decays, in most families, in most categories, without reaching significance on
a family-unit test. That is consistent with the output-gate account (F22/F23)
and the final-layer account (F05), and adds no independent support to either.
