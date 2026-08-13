---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [installation, displacement, axis, alignment-stack]
description: "Plan: WHERE IN THE ALIGNMENT STACK the P axis installs, Tier 0 — the four OLMo rungs already in twp_words, no new compute. The declared question is not when displacement arrives (Finding U: SFT does the cutting) but whether the axis DIRECTION arrives with it, and whether the NAMED quarter (concreteness/register/frequency) and the UNNAMED residual install at the same rung. Directions declared before any number exists."
---
# Plan: where the axis installs — P run down the alignment stack

Drafted 2026-08-13 by the lacan seat on RH's word ("run P in M05-style,
across training checkpoints, to learn when the axis is installed... yes
inside M05"). Under the plan regime of [5687]: the population exists and
does not change, so the dated declaration below does the work a freeze was
a proxy for. No hashes, no content pins; tests and directions first.

## The question, and why it has more structure than "when"

P (M01, `P_unnamed_axis.md`) established a word-level direction alignment
sorts on, with a decomposition: a NAMED component worth roughly a quarter
(concreteness / register / frequency — the interiority face) and an UNNAMED
residual that outpredicts every name. Finding U already answers the
magnitude question: SFT does the cutting; displacement does not come from
safety data. What U cannot answer is whether SFT installs the DIRECTION or
only the movement — and the sharper question this plan declares:

**Do the named component and the unnamed residual install at the same
rung?** If they arrive at different rungs, they are different mechanisms,
which no decomposition of final checkpoints can show.

## Population — exists, verified, does not change

The four OLMo-3 rungs in `twp_words`, verified by malign at [5698]:

    allenai/Olmo-3-1025-7B            base       366,419 rows
    allenai/Olmo-3-7B-Instruct-SFT    SFT        270,701
    allenai/Olmo-3-7B-Instruct-DPO    DPO        277,326
    allenai/Olmo-3-7B-Instruct        Instruct   238,487

Prompts: ACTIVE English catalogue, intersected across all four rungs (rung
coverage differs; the intersection and its size are reported before any
measure). Cells: (prompt, word) present in BASE with p_base >= 0.003 — the
canonical movement rule's own floor, so displacement is measured where the
base actually held mass. Reads are FINAL plus GROUP BY (model, prompt,
word) with avg(p), per the engine-state clause.

The displacement of rung r at cell (prompt, word):

    d_r(prompt, word) = log10( p_r / p_base )    with p_r floored at 1e-6
                        (the floor is reported; cells where the word is
                        absent at rung r are floored, not dropped —
                        dropping selects on the outcome)

Word grain: mean of d_r over the word's cells (n_cells reported; words
with < 5 cells excluded, threshold declared here).

## Measures, all declared before the run

  M1  AXIS PROJECTION PER RUNG. Spearman of word-mean d_r with (a) the
      GloVe movement-axis position, (b) the per-word arm AUC (fall/base
      oriented), (c) the delta word score. Three instruments, one curve
      each, rungs SFT -> DPO -> Instruct.

  M2  NAMED vs RESIDUAL PER RUNG. The axis position is regressed (OLS,
      ranks) on coder concreteness + coder register_level + log COCA-fic
      frequency over covered words; NAMED = fitted component, RESIDUAL =
      what remains. Per rung: Spearman(d_r, NAMED) and
      Spearman(d_r, RESIDUAL). The declared comparison is the RATIO of
      these two correlations across rungs — not their absolute sizes,
      which inherit coverage.

  M3  RUNG INCREMENTS. d_DPO - d_SFT and d_Instruct - d_DPO, same
      correlations: does any LATER rung's increment carry axis direction,
      or is everything after SFT magnitude and noise?

## Directions, declared now

  P1 (from U, directional): the axis direction is PRESENT AT SFT —
     rho(d_SFT, axis) is at least half of rho(d_Instruct, axis), same sign.
  Q1 (open, no direction): whether DPO's increment carries more RESIDUAL
     than NAMED share, or the reverse, or neither. Either answer is a
     finding; the null (increments carry no direction at all) is the
     largest claim and is written here as such.
  Q2 (open): monotonicity of M1 across rungs.

## Statistics and the unit

Words are the unit; CIs by word bootstrap (1,000 draws). This is ONE
LINEAGE — every sentence below the fence is a fact about OLMo-3's stack,
not about alignment. The bloomz/SmolLM2 clause applies verbatim: a property
measured on one member of a class is not a fact about the class. No verdict
language; the deliverable is a described curve with its declared
comparisons, quotable per the register's usual discipline.

## What would make it uninterpretable, stated now

If the prompt intersection across rungs falls below half the battery, the
word-means are over different site populations per rung and the curve is
not the one described here — coverage is reported first and the run stops
there if the floor fails. If the 1e-6 flooring drives any correlation
(checked by rerunning M1 with floored cells dropped), the floored and
dropped variants are both reported and neither is quoted alone.

Producer: `scripts/m05_p_axis_installation.py`. Results:
`results/p_axis_installation.json`. No new compute; everything reads the
store.
