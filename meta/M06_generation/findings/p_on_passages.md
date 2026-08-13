---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/p_on_passages.json
date: 2026-08-13
role: finding
topics: [arm-signature, lexical, replication, amplification, forced-arms]
description: "P on passages, single pass: the arm signature survives to the page (same-prompts cross-grain Spearman +0.500; page classifier real-minus-null-mean 0.39-0.50 against a 200-flip null distribution); the amplification map is legible (narration amplifies, base-pole matter attenuates); and forcing a faller drags BOTH arms toward the base pole equally -- the DiD is null (p 0.63), so the drag is priming, not an alignment-specific response. Ascent branch RUN and DEAD: M02's marker sets flat on faller vs matched in both arms (DiD p 0.94) -- second-order predication is contradiction-triggered, not transgression-triggered. I6: the signature is TONIC -- MARKED sites drag both arms base-poleward equally (p 1e-18 each), DiD null (p 0.90); site-specificity lives at the distribution grain, not on the page. I7: one thin crack -- the site x forced-word triple difference is nominally non-null (p 0.015, saturation direction, components null); flagged for second-seat reconstruction before quoting."
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

## I6: MARKED vs UNMARKED sites -- TONIC; the signature does not deploy

Plan amendment committed before the producer existed (c8adbc86). RH's
question: do undisturbed aligned completions of transgressive prompts
carry a different signature than of neutral prompts? Twins joined on
prompt text (never id), paired by pair_id; 2,449/2,381 paired diffs
(pair x twin); 51,142 rows over 53 prompt texts did not join
(literary/logical, reported). Orientation anchored empirically: higher
axis score = base pole (ambient base +0.0276 vs aligned +0.0154 -- the
tonic signature itself, on the page, as the positive control).

    I6a  aligned MARKED - UNMARKED   +0.00256 base-poleward  1443/1006  p=1e-18
         base    MARKED - UNMARKED   +0.00265                1430/951   p=8e-23
    I6b  DiD                         -0.00015  1187/1194  p=0.90

I5's priming prediction confirmed in both arms; the DiD is dead null.
**The interiority signature is TONIC: a constant register shift, not a
site-conditional deployment. Site-specificity lives only at the
distribution grain -- and there in the sharpened M-era form, not the F01
shorthand: the WITHDRAWAL is transgression-specific (T.13 breadth/depth;
K's 745-pair property gradients; S.2's softening degree, with its
rates-not-odds caveat; S.4's two odds-scale operations), while the
substitution side is general (T) and the architecture-level shifts are
site-blind (J). On the page, transgressive content drags both arms
equally and alignment neither amplifies nor defends there.** With
I5, the page-grain picture is uniform: what alignment installs is a
disposition, and every site-conditional operation so far observed
(displacement F01, second-order M02) lives in the policy, not the
sampled text. Domain decomposition exploratory: animal/taboo/violence
carry the priming, betrayal is null in both arms, and no domain's DiD
reaches 0.05 (min p 0.09 of six, uncorrected).

## I7: site x forced word -- the first crack in the tonic picture, thin

Plan amendment 796a1ca9 (declared before the producer; the tonic
prediction for I7b written down first). Cells from the second-seated I5
parquet; site labels via text join (5,498 of 7,262 forced (pair, prompt)
cells sit on a twin side); DRAG = faller - matched, per (pair, role,
pair_id, side); drags persisted (`p_on_passages_i7_drag.parquet`).

    (medians travel -- the location summary consistent with the sign
     test; means beside, labelled, per [5762])
    I7a  aligned DRAG(MARKED) - DRAG(UNMARKED)  med -0.00034 (mean -0.00070)   998/1040  p=0.36
         base                                   med +0.00059 (mean +0.00060)  1029/954   p=0.10
    I7b  triple difference (aligned - base)     med -0.00101 (mean -0.00131)   934/1043  p=0.015

Both component interactions are individually null; the triple difference
reaches nominal significance from opposite-sign trends. Direction:
RELATIVE SATURATION -- in the aligned arm the forced faller adds LESS
drag at transgressive sites than at neutral ones, relative to base,
which trends the other way. If real, it is the first site-conditional
page-grain effect in the series: the site's own priming (I6) occupies
in the aligned arm some of the room the faller would have used. Stated
at its actual weight: p=0.015 uncorrected across the three declared
contrasts (Bonferroni 0.045, marginal), single pass, one corpus, and
the tonic reading it dents was this seat's own declared prediction --
which is exactly the direction of result this seat checks least, so it
was flagged for second-seat reconstruction from the drag parquet before
anyone quotes it. Reconstruction done ([5762]): every count and p to the
digit; registrar's robustness reads cut both ways -- FOR: per-base-model
I7b medians split 25 negative / 14 positive of 39 families, so no small
set of lineages carries it; AGAINST: the median is 4% of the IQR
(0.0010 vs 0.026). Flag stands; a second corpus is what would earn the
saturation reading. NOT a refutation of I6: the MAIN site effect remains
arm-symmetric; the candidate asymmetry lives only in the interaction.

SENSITIVITY (2026-08-13, after the deepseek corpus defect surfaced --
its passage texts are stored undetokenized, so its cells carry
fragment-tokenized junk): excluding the deepseek pair moves I7b from
p=0.0151 to p=0.0272 (med -0.00101 to -0.00082), which FAILS the
Bonferroni-over-3 line (0.0167) it previously sat inside. I5 and I6 are
insensitive to the same exclusion (I5 aligned faller p 3.1e-9 to
2.6e-9, DiD 0.63 to 0.57; I6 aligned 1.0e-18 to 6.4e-19, DiD 0.902 to
0.901). The I7 flag was already flagged-not-quoted; it is now also
Bonferroni-failing on the cleaner population, which lowers it further.

Echo by site (exploratory): flat -- the [5757] echo asymmetry is
unmodulated by site type in either arm (aligned faller 0.250 MARKED vs
0.259 UNMARKED; base 0.209 vs 0.225).

## Fences

Anti-conflation: usage-rate AUC and candidate-share AUC remain different
objects; +0.500 is a correlation between them, not an identity. n=600 for
the primary (the passage-prompts logit table is small). Axis scores cover
GloVe verbs only; coverage travelled per passage. Single lineage-set, one
corpus, one prompt family.
