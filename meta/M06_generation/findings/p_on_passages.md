---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/p_on_passages.json
date: 2026-08-13
role: finding
topics: [arm-signature, lexical, replication, amplification, forced-arms]
description: "P on passages, single pass: the arm signature survives to the page (same-prompts cross-grain Spearman +0.500; page classifier real-minus-null ~0.35); the amplification map is legible (narration amplifies, base-pole matter attenuates); and forcing a faller drags BOTH arms toward the base pole equally -- the DiD is null (p 0.63), so the drag is priming, not an alignment-specific response. Ascent branch (M02 markers) declared, not yet run."
---
# P on passages: the signature survives the trip, and the drag is priming

Plan: `plans/plan_p_on_passages.md` (committed before the run, I5 amended
pre-run). Producer `scripts/m06_p_on_passages.py`; results
`results/p_on_passages.json`. Single pass; the [5503] rule applies.

Population: 232,384 undisturbed passages, 41 pairs (SmolLM2 excluded --
flag ambiguity, [5707]); stratum non-degenerate AND English at 90.2%
(prose screen deferred: the flags parquet carries screen quantities, and
`is_prose` lives in the measure shards -- a full-stratum rerun is the
declared follow-up alongside the ascent branch).

## I3: the cross-grain replication -- P1 CONFIRMED

    same prompts, same models, both grains   Spearman +0.500  (n=600 words)
    canonical-vector context                 +0.444           (n=3,613)

Half the distributional ranking survives temperature-1 sampling and ~200
tokens of autoregression, on identical prompts and models: grain is the
only thing that varies. All ten probe words sit at their logit poles.

## I2: the page classifier -- quotable only as real-minus-null

    k=25 .851 | k=50 .912 | k=100 .953 | k=200 .966
    flip-nulls 0.52-0.63 (logit-side sat ~0.51)   real-minus-null ~0.35

The null elevation is undiagnosed (candidates: 41-pair org structure;
lineage signatures in rates). Until it is, the number that travels is
real-minus-null, not the AUC.

## I4: the amplification map (exploratory, as declared)

Amplified on the page: narrative-social machinery (`led, taking, friend,
took, sister, replied, realizing`). Attenuated: base-pole matter whose
distributional signal sampling does not realise (`blood, shout, shot,
hid, waved`). The smoke's `inform` generalised: institutional dispositions
mute in fiction continuations.

## I5: forcing a faller -- DRAGGED, symmetrically; the DiD is null

    I5a  aligned faller vs matched   +0.00115 toward base pole  2638/2224  p<1e-4
         base    faller vs matched   +0.00062                   2489/2244  p=4e-4
         (riser_matched mildly opposite in both arms; riser null)
    I5b  DiD, priming subtracted     +0.00013  2379/2345  p=0.63

The homeostatic reading is dead -- no overcorrection anywhere. The dragged
reading holds but is NOT alignment-specific: base moves the same amount at
the same sites, so the displacement is autoregressive priming. **What
alignment installs, it does not defend at composition grain; and it is not
differentially vulnerable either.**

ECHO, unexplained and carried: aligned models repeat the injected word
more than base across ALL arms (faller 0.255 vs 0.217; matched 0.219 vs
0.138) -- an arm-general echo propensity, not faller-specific. Whether it
is instruction-adjacent coherence or the ascent move is exactly what the
UNRUN third branch decides: M02's second-order markers on these passages
(declared in the plan, not yet run, and nothing about ascent may be
claimed until it runs).

## Fences

Anti-conflation: usage-rate AUC and candidate-share AUC remain different
objects; +0.500 is a correlation between them, not an identity. n=600 for
the primary (the passage-prompts logit table is small). Axis scores cover
GloVe verbs only; coverage travelled per passage. Single lineage-set, one
corpus, one prompt family.
