---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/p_on_passages.json
date: 2026-08-13
role: finding
topics: [arm-signature, lexical, replication, amplification, forced-arms]
description: "P on passages, single pass: the arm signature survives to the page (same-prompts cross-grain Spearman +0.500; page classifier real-minus-null-mean 0.39-0.50 against a 200-flip null distribution); the amplification map is legible (narration amplifies, base-pole matter attenuates); and forcing a faller drags BOTH arms toward the base pole equally -- the DiD is null (p 0.63), so the drag is priming, not an alignment-specific response. Ascent branch RUN and DEAD: M02's marker sets flat on faller vs matched in both arms (DiD p 0.94) -- second-order predication is contradiction-triggered, not transgression-triggered."
---
# P on passages: the signature survives the trip, and the drag is priming

Plan: `plans/plan_p_on_passages.md` (committed before the run, I5 amended
pre-run). Producer `scripts/m06_p_on_passages.py`; results
`results/p_on_passages.json`. Single pass; the [5503] rule applies.

Population: 232,384 undisturbed passages, 41 pairs (SmolLM2 excluded --
flag ambiguity, [5707]); stratum non-degenerate AND English at 90.2%
(prose screen deferred: the flags parquet carries screen quantities, and
`is_prose` lives in the measure shards -- a full-stratum rerun is the
declared follow-up; the ascent branch has now run, below).

## I3: the cross-grain replication -- P1 CONFIRMED

    same prompts, same models, both grains   Spearman +0.500  (n=600 words)
    canonical-vector context                 +0.444           (n=3,613)

Half the distributional ranking survives temperature-1 sampling and ~200
tokens of autoregression, on identical prompts and models: grain is the
only thing that varies. All ten probe words sit at their logit poles.

## I2: the page classifier -- against a null DISTRIBUTION (corrected, [5744])

    k=25 .851 | 50 .912 | 100 .953 | 200 .966
    200-flip null: mean .465-.468, 95% band [.30, .65]
    REAL-MINUS-NULL-MEAN: .387 / .448 / .486 / .499

THE FIRST TWO VERSIONS OF THIS SECTION QUOTED SINGLE DRAWS. The flip
assignment iterated an unsorted set, so one seed gave 0.52-0.63 in one
process and 0.40-0.49 in the next ([5744], malign's catch; the k_ceiling
defect of 12 Aug recommitted here). Both draws sit inside the corrected
band: neither an "elevation" nor a "depression" existed -- a one-flip null
at 41 lineages wobbles +-0.15 and nobody had characterised it. The
quotable form is real-minus-null-mean, 0.39-0.50 across the grid. One
residue kept visible: the null MEAN is 0.465, slightly below 0.5 with 200
draws behind it -- small, real, unexplained, and not load-bearing.

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

THE ASCENT BRANCH IS DEAD (run 2026-08-13,
`scripts/m06_p_on_passages_ascent.py` ->
`results/p_on_passages_ascent.json`). M02's committed marker sets,
imported from `z_second_order` (15 SO + 5 DE regexes, first 50 words --
the instrument, not a reimplementation), on the same forced passages,
same pairing, same strata (793,517 passages, 40,134 cells):

    ANY_SO  aligned faller vs matched   411/416   p=0.89
            base    faller vs matched   418/417   p=1
            DiD faller                  711/715   p=0.94
    ANY_DE  aligned faller vs matched   1265/1204 p=0.23
            DiD faller                  1683/1690 p=0.92
    ambient ANY_SO: aligned 0.0071 | base 0.0067

No faller-excess in either arm, either marker family; the one nominal p
(base riser_matched ANY_DE, p=0.019) is 1 of 12 contrasts and does not
survive any correction. Two consequences, both declared in the producer
before the numbers existed: (1) forcing a transgressive word does NOT
trigger second-order predication -- M02's excess is
contradiction-specific in trigger, not a general aligned response to
transgressive matter; (2) the echo asymmetry (aligned repeats ANY
injected word more: faller 0.255 vs 0.217, matched 0.219 vs 0.138) loses
its "mention not use" support leg and stands as instruction-adjacent
coherence, arm-general and unexplained. Power caveat: ANY_SO is sparse
(ambient 0.007), so per-pair medians are zero and the sign tests run on
the ~830 nonzero pairs; a small faller-specific shift below that grain
is not excluded, but the M02-size effect is.

## Fences

Anti-conflation: usage-rate AUC and candidate-share AUC remain different
objects; +0.500 is a correlation between them, not an identity. n=600 for
the primary (the passage-prompts logit table is small). Axis scores cover
GloVe verbs only; coverage travelled per passage. Single lineage-set, one
corpus, one prompt family.
