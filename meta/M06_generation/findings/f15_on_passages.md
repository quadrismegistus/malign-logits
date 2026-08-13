---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/f15_on_passages.json
date: 2026-08-13
role: finding
topics: [surprisal, drift, quadrants, jakobson, replication]
description: "F15 on passages: ALL FOUR F15 claims replicate at the pair grain -- alignment smooths reference-model surprisal (35/38 pairs, med -0.53 nats), reduces drift (34/38), drains the breakdown quadrant (Q2 -0.21) into unmarked (Q4 +0.22) AND metonymic (Q1 +0.14, 34/38, the gain F15 saw only in Qwen) -- BUT the quadrant flow is ONE-DIMENSIONAL: Q1's share change correlates with surprisal at -0.83 and with drift at -0.05, so the 'metonymic' gain has no drift component and the labels are not earned by these axes, and compresses UNIFORMLY across site types (twin-paired DiD p 0.089, Kruskal across domains p 0.892 -- a bound at ~5% of the effect, not a bare null). The drift claim and the quadrant flow are EMBEDDER-INDEPENDENT (bge-m3, sign agreement 33/38). Population 38 of 41 pairs; deepseek fenced for a generation-time tokenizer defect."
---
# F15 on passages: the quadrant flow survives, and the metonymic gain is general

Plan: `plans/plan_f15_on_passages.md` (f2f5c804, committed before the
producer). Producer `scripts/m06_f15_on_passages.py`; results
`results/f15_on_passages.json` + per-passage parquet (35,230 rows).
Single pass; the [5503] rule applies. Metric code imported from the
committed F15 instrument (`malign_logits/embedding.py`), never
reimplemented. Reference GPT-2 (the only reference still independent of
the roster -- Pythia joined it); embedder MiniLM, mps behind the
second-device gate (min cos 1.0000 on the corpus's own rows). Subsample
declared on the docket before the run ([5765]): cap 3 per (pair, role,
prompt), seed over sorted keys.

## Population, and the accounting that found a corpus defect

38 pairs enter the paired contrasts. Named exclusions, each with its
mechanism:

- **deepseek**: 95% of its passage-corpus texts are spaceless with
  literal byte markers (mean 1.6 space-delimited words per 1,300
  chars), so 2,481 of 2,481 stratum-passing passages fail the 75-word
  rule. Reported to malign; ruled a generation-time tokenizer defect
  and FENCED, not repaired ([5776],
  `data/deepseek_passage_text_fence.json`): the word-initial spaces
  were never in the token ids (`['mu','ff','led','her',...]`, no
  `Ġher`), so no decode restores them and the ids themselves are
  intact. **My own inference here was wrong and is corrected in
  place**: I read the flags' english_nltkwords_share of 0.87 as
  evidence that spaced text once existed. It did not. Registrar
  recomputed the flags through their own producer path (416 rows, zero
  mismatches, [5777]): the screen's regex `[a-zA-Z']+` treats the
  byte markers as word boundaries, so marker-delimited stretches yield
  real words while concatenated stretches collapse to one blob. The
  0.87 is the honest output of that regex on these bytes. A regex is a
  tokenizer, and a screen built on one inherits its boundary
  conventions on exactly the pathological rows it exists to catch.
- **recurrentgemma**: 0.2-0.5% pass the degeneracy/English screens.
- **Teuken**: base arm 0% English.
- **bloomz**: aligned arm 7 surviving passages, below any floor.

Truncation survival differs by arm corpus-wide (base 0.836 vs aligned
0.805) -- the selection channel the plan fences; it travels with every
contrast below.

## The three declared directions, all confirmed (paired per pair, n=38)

    (medians travel; means beside)
    P1  mean_surprisal, aligned - base   med -0.526 (mean -0.541) nats  3/35   p=7e-8
    P2  total_drift                      med -0.023 (mean -0.032)       4/34   p=6e-7
    P3  quadrant shares, aligned - base:
        Q2 breakdown  (hi drift, hi surp)   med -0.211  3/35   p=7e-8
        Q4 unmarked   (lo, lo)              med +0.224  35/3   p=7e-8
        Q1 metonymic  (hi drift, lo surp)   med +0.137  34/4   p=6e-7
        Q3 metaphoric (lo drift, hi surp)   med -0.157  4/34   p=6e-7

F15's flow (Q2 drains into Q1 and Q4) replicates on a different corpus,
a different rung, and 4x the pairs. **And the plan's open question Q1
answers YES: the metonymic share rises under alignment in 34 of 38
pairs -- what F15 saw in one family (Qwen) is general.** Both
high-surprisal quadrants drain and both low-surprisal quadrants gain,
so the flow is carried by the surprisal axis with the drift axis moving
less: aligned text keeps crossing topics but becomes predictable while
doing so. Chain-sliding, at the pair grain.

## F1: the drift claim is EMBEDDER-INDEPENDENT -- the gate is discharged

Plan amendment F1, declared before the producer. 4,511 passages (60 per
(pair, role), seed over sorted keys), re-encoded with bge-m3 -- F15's
own headline embedder -- through the identical truncation, split and
prefix recipe; surprisal untouched, so the quadrant axes move only
through drift. Device gate min cos 1.0000 on the corpus's own rows.
Both embedders on IDENTICAL passages, so the comparison is
instrument-vs-instrument and the subsample's MiniLM values (not the
full run's) are the correct reference.

    per-passage drift agreement, Spearman        +0.704
    P2  MiniLM  med -0.0296  6/32  p 2.4e-05
        bge-m3  med -0.0268  5/33  p 4.3e-06
    SIGN AGREEMENT ON P2                         33 of 38 pairs
    P3  quadrant     MiniLM              bge-m3
        Q1 metonymic +0.142 33/5 4e-06   +0.100 32/4 2e-06
        Q2 breakdown -0.207  4/33 1e-06  -0.250  4/34 6e-07
        Q3 metaphoric-0.150  5/33 4e-06  -0.100  7/29 3e-04
        Q4 unmarked  +0.258 33/4 1e-06   +0.250 32/6 2e-05

Every quadrant keeps its sign and its significance under both
instruments, and the two embedders agree on P2's direction in 33 of 38
pairs -- the five disagreements are all near zero under both. **P2 and
P3, including the metonymic result, are quotable as
embedder-independent.** The two instruments are not interchangeable at
the passage grain (+0.704, not +0.95): they measure trajectory
differently and agree about alignment anyway, which is the stronger
form of the check.

## F2: the compression is UNIFORM ACROSS SITES -- F15's fourth claim, carried

Plan amendment F2. F15: "content category has no effect on
within-passage surprisal (Kruskal-Wallis p=0.99); alignment is a
uniform compressor." Three forms, no new compute (persisted cells plus
I6's catalogue join on prompt TEXT); 26,570 of 35,230 passages carry a
twin label.

    F2a  I6-FORM, paired per (pair, pair_id):
         surprisal  MARKED - UNMARKED   aligned +0.010 p 0.29 | base -0.011 p 0.34
                    DiD                 +0.0275  1119/1039  p 0.089   (n 2,158)
         drift      DiD                 +0.0054  1106/1052  p 0.254
    F2b  F15-FORM, aligned-base surprisal delta by domain (38 pairs):
         violence -0.615 | betrayal -0.569 | property -0.550 | taboo
         -0.544 | animal -0.538 | sexual -0.504
         Kruskal-Wallis across 6 domains: surprisal p 0.892, drift p 0.725
    F2b' compression at MARKED -0.537 (4/34) and UNMARKED -0.512 (4/34),
         difference +0.036  23/15  p 0.256

**The uniform compressor replicates, and this is a BOUND rather than a
bare null**: against a main effect of -0.55 nats, at n=2,158 twin-pairs,
the observed site modulation sits at hundredths, so a site-conditional
compression above roughly 5% of the effect would have shown. The
instrument is also sharper than F15's -- paired within minimal twins
(same scene, one word changed) rather than pooled group means, which is
the design that made site-specificity appear in K where group means had
hidden it. The largest deviation in the set, the surprisal DiD at
p=0.089, runs the WRONG WAY for a watchman reading: if anything the
aligned model compresses transgressive sites slightly LESS than its own
neutral twins.

**Read with I6 and the ascent null, three different metrics --
composition, level, predictability -- now agree that the page-grain
operation is site-blind.**

## F3: forcing does not alter the movements -- and the one thing it does move is symmetric

Plan amendment F3, RH's question. 48,335 forced passages (three arms:
faller / matched / riser_matched, cap 2 per (pair, prompt, role, arm),
SmolLM2 excluded and deepseek fenced), same truncation and metrics as the
main run.

    F3a  DOES THE ARM CONTRAST SURVIVE FORCING (P6: declared yes)
         surprisal  forced-matched -0.5237  5/33  p 4e-06
                    undisturbed    -0.5259  3/35  p 7e-08
         drift      forced-matched -0.0302  9/29  p 0.0017
                    undisturbed    -0.0227  4/34  p 6e-07

    F3b  DOES FORCING ITSELF MOVE THE METRICS (Q3: open, no direction)
         surprisal  aligned faller-matched         -0.0213  p 0.066  (n 4,198)
                    base    faller-matched         -0.0337  p 0.0014 (n 4,263)
                    aligned riser_matched-matched  +0.0089  p 0.42
                    base    riser_matched-matched  -0.0073  p 0.37
         drift      all four contrasts             p 0.25 to 0.94
    F3c  DiD faller  surprisal +0.0186  p 0.281  |  drift +0.0028  p 0.463

**P6 CONFIRMED and it is the load-bearing leg: injecting a word does not
abolish the register difference.** The aligned-base surprisal gap on forced
passages is -0.524 against -0.526 undisturbed -- the same number. Drift's
gap survives too, noisier under perturbation as expected. So the signature
is not a fragile property of undisturbed generation.

**Q3 ANSWERED, and against the more interesting hypothesis**: a forced
faller makes the continuation LESS surprising, not more. There is no cost
signature, no disruption, no symptom. And it is faller-SPECIFIC rather than
forcing-general -- riser_matched, forced at the same aligned probability,
is null in both arms -- so what moves the metric is the word's movement
class, not the act of injection. The natural reading is that a faller is a
word the BASE prefers, so it is corpus-typical, and corpus-typical material
makes the following text more predictable to a third-party reference. That
it runs numerically LARGER in base (-0.034 vs -0.021) fits that story, but
the DiD is null (p 0.281) and the difference is not claimable.

**Drift does not move at all** -- four within-arm nulls and a null DiD. The
trajectory of a passage is indifferent to what word was injected into it.

**FOURTH NULL DiD IN THE FORCED SERIES.** Composition (I5), level (ascent),
predictability and trajectory (here): the aligned model's response to a
forced transgressive word is, on every measure this campaign has built, the
same response the base model gives.

RETENTION BY ARM, added [5801] after malign's [5800] sibling-rule lesson --
the finding had reported truncation survival by ROLE (base 0.836, aligned
0.805) and treated that as the selection fence, but a role-level rate is
silent about arms. Measured: aligned faller 0.7923 / matched 0.7969 /
riser_matched 0.7985 (spread 0.0062); base 0.8078 / 0.8046 / 0.8039
(spread 0.0039). Under 0.7% in both roles with the sign flipping between
them, so the surviving set is effectively arm-independent. Note the
direction anyway: in the aligned arm the faller is retained LEAST, which
would deflate a faller effect rather than manufacture one.

## Open, declared -- and one F15 claim this redo did NOT carry

DISCHARGED since this section was written: the bge-m3 fidelity gate
(F1, above) and F15's uncarried fourth claim (F2, above). Still open:

- Sharpness confound, testable rather than merely stated: the pair-wise
  surprisal delta against the same pair's top-1-mass/entropy delta from
  the twp side (P's nuisance block puts top-1 mass at AUC 0.889 for the
  arm). A partial correlation would say how much of P1 is generator
  sharpness rather than page predictability. Never run; the fence
  currently states the confound without bounding it.
- Pythia-1B-deduped secondary reference on the subsample, pythia pairs
  excluded. Not yet run.
- Prose-stratum rerun inherits from the M06 series.
- Plan Q2, the axis-quadrant bridge to the P series. Exploratory, not
  owed.
- Producer-layer reproduction: the aggregation layer is second-seated
  ([5772]/[5775], all six aggregates rebuilt from the cells); the fetch,
  strata, truncation and metric application remain single-pass.

## THE QUADRANT FLOW IS ONE-DIMENSIONAL, and the metonymic gain has NO drift component

*(The full instrument audit behind this section -- order-invariance, the
92%-noise decomposition, directedness being sentence count, and the
truncation cost -- is consolidated in `drift_metric_audit.md`.)*

Measured on RH's question (are the quadrants and their names convincing?),
from the persisted cells, 38 pairs. Per-pair Spearman of each quadrant's
share change against the pair's own surprisal change and drift change:

    quadrant   median shift    rho(surprisal)   rho(drift)
    Q1 metonymic   +0.137          -0.827         -0.050
    Q2 breakdown   -0.211          +0.916         +0.696
    Q3 metaphoric  -0.157          +0.822         +0.142
    Q4 unmarked    +0.224          -0.874         -0.717

**Q1's gain -- this document's headline result -- tracks surprisal at -0.83
and drift at -0.05.** It has NO drift component. Passages entered the
"metonymic" quadrant by crossing the surprisal median while their dispersion
stayed where it was. Q3 is nearly as one-sided (+0.82 / +0.14). Only Q2 and
Q4 carry real drift loading.

**So "alignment drains breakdown into metonymic and unmarked" is largely a
re-description of "alignment lowers surprisal."** The quadrants are a median
split converting two marginals into four shares, and for Q1 and Q3 the
second marginal contributes almost nothing.

**AND THE MECHANISM IS MEASUREMENT NOISE, not effect size -- RH's question,
measured.** Comparing -0.53 nats against -0.023 cosine units compares
incommensurable scales, and standardising exposes that the gap is mostly an
artefact of how the two quantities are BUILT. Surprisal averages ~107 tokens
per passage; `total_drift` is the MINIMUM over ~10 pairwise similarities
among ~5 sentences -- an extreme statistic over a handful of units.
Variance decomposition on 9,501 three-sample cells (within-cell = same
model, same prompt, different sample = pure noise):

    metric           total SD   within SD   between SD    ICC
    mean_surprisal     0.7343      0.5822       0.4475   0.371
    total_drift        0.1397      0.1338       0.0401   0.082
    mean_drift         0.1191      0.1104       0.0448   0.141

**`total_drift` is 92% NOISE at the passage level.** So Cohen's d divides
the drift effect by a denominator that is almost entirely measurement error.
The ratio depends entirely on which standardisation you pick:

    surprisal : drift    4.4x  raw / total SD      (what this document quoted)
                         2.1x  raw / BETWEEN SD    (noise removed)
                         1.3x  dz                  (paired signal-to-noise)

**AND THIS EXPLAINS THE QUADRANT RESULT MECHANICALLY.** A median split on an
axis that is 92% noise assigns passages to sides close to at random, so the
drift axis cannot carry a quadrant flow however real the underlying shift
is. Q1's rho of -0.05 with drift is not evidence that dispersion did not
move -- it moved at dz -0.98 -- it is evidence that a median split on
`total_drift` is mostly splitting on sampling error.

**Consequence for any future drift axis: use `mean_drift`, not
`total_drift`.** It is a mean rather than an extreme (ICC 0.141 against
0.082), it is order-DEPENDENT rather than invariant, and its own effect is
larger on every standardisation (raw -0.051, between-SD -1.13, dz -1.53).
Every property that matters is better.

**THE NAMES ARE NOT EARNED BY THESE AXES.** `total_drift` is order-invariant
(below), so it cannot distinguish a metonymic CHAIN from an undirected
scatter of the same diameter; "metonymic" names a sequential relation that
the measurement is blind to. "Metaphoric" for Q3 imports substitution
without measuring substitution. And the quadrant boundaries are this
corpus's own pooled medians, so membership is a fact about a passage
relative to its cohort rather than a type.

**What survives, and it is not small:** alignment lowers reference surprisal
hugely and narrows semantic dispersion slightly, both replicated across
corpora, embedders and languages. That is two numbers. The four shares are a
presentation of those two numbers, and the Jakobsonian labels are a
hypothesis laid over them rather than a result read out of them.

## WHAT THE DRIFT AXIS MEASURES, corrected 2026-08-13

**`total_drift` is ORDER-INVARIANT** -- `1 - min(pairwise similarity)`, the
DIAMETER of the sentence set, unchanged when the sentences are shuffled
(verified by permutation). P2 is therefore a claim about **SEMANTIC SPREAD**,
not about trajectory: aligned passages occupy a narrower region. The numbers
stand; the noun was wrong wherever this document said the passage "drifts
less" in a sequential sense.

**This qualifies the QUADRANT reading too.** The quadrant axes are drift x
surprisal, and Q1 is labelled METONYMIC -- but an order-invariant axis cannot
distinguish a directed chain from a bounded wander with the same diameter.
So "metonymic" is a label applied to a NON-SEQUENTIAL measure, and the
Jakobsonian reading of Q1 is not earned by this instrument. What Q1 says
without interpretation: semantically DISPERSED and PREDICTABLE.

The measure that would earn it exists in the same function and was discarded
by these producers: `directedness = total_drift / path_length`, plus the
within-passage permutation test (`mean successive - mean all-pairs`), which
holds composition fixed by construction.

## Fences


- **EVERY DRIFT NUMBER HERE IS COMPUTED ON ROUGHLY THE FIRST HALF OF EACH
  PASSAGE**, and that is an inherited convention rather than a design choice.
  Generations run to a 256-token cap (median 183 words); the
  fewest-sentences-exceeding-75-words rule leaves a median of 84 words and 5
  sentences, so **46% of the generated words are analysed**. Measured on
  3,000 passages, dropping truncation entirely would give 11 sentences and 55
  pairwise similarities instead of 5 and 10 -- and would KEEP MORE PASSAGES
  (95.1% against 86.1%), because the word floor discards short generations
  that have three or more sentences anyway.

  The rule comes from F15/F16, where it normalises length across corpora with
  wildly different natural lengths (dreams against abstracts against
  fiction). **Applied to a within-corpus arm contrast, where every generation
  already shares one token cap, it normalises nothing and costs both data and
  trajectory resolution.** So `total_drift`'s 92%-noise figure is
  substantially self-inflicted: five sentences were analysed where eleven
  were available. The successor instrument should carry no truncation and
  `n_sents` as a declared covariate.
Different corpus AND rung from F15: agreement extends it, and this is
agreement. Sharpness confound stated in the plan: reference surprisal
entangles generator sharpness with text predictability on the page;
this instrument cannot separate them. Truncation selects on length per
arm (rates above). Quadrant thresholds are this corpus's own pooled
medians (drift 0.9393, surprisal 3.7278 nats -- from the JSON), not
F15's.
