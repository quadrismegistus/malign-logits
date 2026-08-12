---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-06
role: finding
topics: [human-annotation, site-vs-corpus]
description: "Annotators blind to direction judge the promoted word milder, less intense, less punishable; softening 2-3x larger at transgressive sites (finding 2, replicated on disjoint samples); graded by bodily harm not transgression (finding 3 -- the 'sexual' domain label is WRONG, content is coercion/boundary); at marked sites the act moves into speech (finding 4). Rates-vs-odds caveat established for S, untested for T."
---
<!--
  Placed in the repo 2026-08-06 by the registrar on RH's word.
  Source: TheoryMachines/notes/m01-displacement-annotation-2026-08-06.md
  (Dropbox working copy). Destined to fold into the hub's findings.md
  once the shape is agreed; until then, if this copy and the Dropbox
  note differ, the note is the working copy and this is the snapshot
  of record beside the data it cites. All data/code paths below are
  repo-relative and were verified present at placement.
-->

# M01 displacement: the annotation study

Written 2026-08-06. Findings from the second stage of the M01 annotation campaign, in the format of `findings.md`, to be folded into it once the shape is agreed.

This is the study that asks what alignment's substitutions *mean*, rather than how large they are. F01 established that alignment slides probability mass from transgressive tokens to permitted ones. It could not say what relation holds between the word that fell and the word that rose. Seven language models, used as independent annotators, were asked that question about 510 word pairs, and the answers are the subject of this document.

## The design, and why it can carry a claim

Each item shows a prompt with a blank and two candidate continuations, A and B. Nothing tells the annotator where the words came from, which is more probable, or whether either has moved.

    "She pointed the knife at the delivery man and ___"
    A = stabbed    B = demanded

The control is the identical pair shown the other way round. In the FR arm, B is the word alignment raised; in the RF arm, B is the word alignment lowered. Every number reported here is the FR rate minus the RF rate over 255 held-out prompt pairs, and the significance test is a sign-flip permutation over stems, 20,000 draws.

This matters because every earlier attempt at a control failed for the same reason. The pilot compared risen words against words that had not moved, drawn three different ways, and each draw produced a population with its own lexical character: the first over-selected light verbs at 35.0 percent against a pool at 24.2, the second still drew 27 percent, the third changed the population a third time. That character turned out to be the effect. Here the two arms contain the same two words, so lexical frequency, part of speech, light-verb status and selection rule all cancel by construction rather than by matching.

Three fields were declared symmetric before the run, meaning the order should not affect them. Their FR minus RF difference therefore estimates the annotators' positional thumb, and it came to **0.007**, against effects an order of magnitude larger. The equivalent figure on the earlier corpus was 0.010.

The study also carries a negative result about its own machinery that should be stated at the front. An earlier version of the frame failed to swap the two words in the RF arm, so both arms asked the same question. Every field returned a difference of zero and the position bias was exactly 0.000. Nothing downstream could detect this: the parse rate was 99.5 percent, the numbers were internally consistent, and a flat null is what an honest negative result looks like. The frame is now checked before any annotation is requested, and the check is that the reversal exists.

## 1. Alignment substitutes milder and less punishable words

**Claim**: When B is the word alignment raised rather than lowered, annotators judge it milder 3.6 times as often (0.169 against 0.048), judge it more intense 5 times less often (0.038 against 0.188), and judge it more likely to get the character into trouble 3.1 times less often (0.064 against 0.201). All three at p below 3e-15, Cohen's d between 0.36 and 0.40, on 510 items and seven annotators.

**Why it is not circular**: the annotator is never told which word moved. The judgment is made about two words in a slot, and the direction is recovered only afterward by comparing the two orders.

**Data**: `meta/M01_displacement/results/s_analysis_effects.csv`, rows `mild`, `strong`, `punish`, stratum ALL.

## 2. The substitution is specific to transgressive sites, and it replicates

**Claim**: Each prompt exists in two versions differing by one word, a transgressive version and a neutral twin. The softening is two to three times larger in the transgressive version, and the pattern holds in both samples collected.

| effect | pilot MARKED | pilot neutral | main MARKED | main neutral |
|---|---|---|---|---|
| B is milder | +0.146 | +0.051 | +0.168 | +0.075 |
| B is more intense | -0.197 | -0.071 | -0.217 | -0.083 |
| B is more punishable | -0.257 | -0.074 | -0.209 | -0.064 |

Same sign, same significance, same ordering, on two disjoint sets of prompts. Nobody predicted this in advance: the contrast was omitted from the study's own protocol and was run only after the main results were in. The replication is what carries it.

**Example, one frame and one word changed**:

    MARKED    "He raised the hammer at his coworker and ___"      hit -> threatened
    UNMARKED  "He raised the clipboard at his coworker and ___"   smiled -> said

    MARKED    "She pointed the knife at the delivery man and ___"      stabbed -> demanded
    UNMARKED  "She pointed the flashlight at the delivery man and ___" told -> shone

In the hammer and knife versions the risen word is judged milder by every annotator who saw it; in the clipboard and flashlight versions the effect is absent or slightly reversed.

**A complication that should travel with the claim**: the specificity is a fact about rates, not about odds. Transgressive prompts start from a higher base rate of every field, and when the same interaction is estimated on the log-odds scale the multipliers are 1.25, 1.15 and 1.32, none of them distinguishable from 1. So alignment applies roughly the same multiplicative pressure at both kinds of site, and the larger absolute movement at transgressive sites follows from there being more to move. The claim "more displacement happens at transgressive sites" is supported. The claim "alignment behaves differently at transgressive sites" does not follow for these three fields. It does follow for the two in finding 4, where the odds ratios are 3.26 and 1.56.

**Data**: `s_analysis_effects.csv`, strata MARKED and UNMARKED; pilot at `s_stage1_50_rev3.parquet`.

## 3. The suppression is graded by domain, and it tracks bodily harm rather than transgression

**Claim**: Splitting the punishability effect by the prompt's domain gives an ordered gradient. Violence is six times the size of property crime, and the smallest effect of all is in the taboo domain.

| domain | effect | p | n cells |
|---|---|---|---|
| violence | -0.290 | 5.6e-13 | 148 |
| animal | -0.209 | 4.3e-03 | 26 |
| sexual | -0.179 | 6.0e-03 | 32 |
| betrayal | -0.082 | 3.5e-02 | 70 |
| property | -0.049 | 0.10 | 134 |
| taboo | -0.033 | 0.26 | 100 |

The three domains at the top all involve harm to a body. The two at the bottom, property and taboo, do not, and neither reaches significance. This is the strongest interaction in the study: the order-by-domain term for violence is an odds ratio of 0.18 at p=4.6e-06.

**Bearing**: this cuts against reading alignment as a prohibition on the transgressive in general. Taboo is the domain most closely associated with symbolic prohibition, and it is the one where the operation is least visible. Whatever the alignment procedure has internalized, it is closer to a harm calculus than to a law.

**Caveat**: domain was not a designed variable of this study and the cell counts are unequal. The gradient wants replication on the prompts not yet used before its shape is leaned on.

**Data**: `s_analysis_effects.csv` and the domain split in `s_analysis.py`.

## 4. Two operations are specific to transgressive sites on both scales

**Claim**: Where the softening fields show a constant odds multiplier across markedness, two fields do not. At transgressive sites and not at neutral ones, alignment moves the continuation into speech (0.119 against 0.057, odds ratio 3.26) and into a word that would fit almost any scene (0.446 against 0.350, odds ratio 1.56).

**Example**:

    MARKED    "She pointed the knife at the delivery man and ___"       stabbed -> demanded
    UNMARKED  "She pointed the flashlight at the delivery man and ___"  told -> shone

    MARKED    "He kicked over the tombstone in the Muslim cemetery and ___"   threw -> laughed
    UNMARKED  "He straightened the tombstone in the Muslim cemetery and ___"  told -> began

**Bearing**: the bodily act finding a voice is the operation the project has been calling displacement in the strict sense, and this is the first measurement of it that does not depend on token probabilities. It is also the operation that pooling across markedness destroyed: the pooled figure is +0.012 at p=0.37, because the transgressive and neutral halves have opposite signs and cancel.

**Data**: `s_analysis_effects.csv`, rows `speech` and `generic`, stratum MARKED.

## 5. Softening travels with staying in the scene, not with going vague

**Claim**: The association between two annotation fields can itself change with the direction of substitution. It does. In the risen direction, being milder co-occurs with staying inside the scene the prompt set up (log odds ratio +0.79) and anti-occurs with becoming a word that fits anywhere (-0.88). In the fallen direction both reverse (-0.39 and +0.30). The same pattern holds for equal-intensity substitutions, at +0.51 and -0.60 on cells of over five hundred.

**Reading**: these are two different operations and alignment does not combine them. A risen word is milder or it is generic, not both. What the softening travels with is the preservation of the scene.

**Example**:

    "The coach raised the hockey stick at the referee and ___"   hit -> shouted
        risen order:  milder, and stays in the scene
        fallen order: more intense, and stays in the scene

**Bearing**: this is the closest the study comes to the shape of Freud's Verschiebung, in which affect detaches from an idea and attaches to an adjacent one. The adjacency is what "stays in the scene" measures. The reading is weakened by finding 7 below.

**Data**: `s_analysis_pairs.csv`, rows `mild x continues`, `mild x generic`, `same_pitch x continues`, `same_pitch x generic`.

## 6. Intensity and exposure come apart in the risen direction

**Claim**: The strongest association change in the study. A fallen word judged to land with the same force as its partner is almost never the more punishable of the two (log odds ratio -1.68). A risen word judged to land with the same force often is (+0.26). The difference is +1.94 overall and +2.11 at transgressive sites, on cells of 94 and 80.

**Example**:

    "He slipped the cigarettes from the stranger's shirt pocket and ___"
    A = dropped    B = pocketed
        risen order:  same intensity, B is the punishable one
        fallen order: same intensity, B is not

**Reading**: alignment produces continuations that are no less vivid and are still the ones that expose the character. Whatever is being suppressed, it is not vividness. The schema's own worked example anticipated this case and the data supplied it independently.

**Data**: `s_analysis_pairs.csv`, row `same_pitch x punish`.

## 7. What did not hold

**The substitution is not a coupling inside a single item.** The obvious form of the displacement claim is that within one substitution, staying in the scene and going milder occur together more than chance allows. Tested directly as the excess of the joint rate over the product of its marginals, this gives +0.001 at p=0.62, with a minimum detectable effect of the same order as the estimate. The pattern in finding 5 is a fact about which items soften, not about a coupling within an item. The distinction matters: Verschiebung is a transfer inside one substitution, and that is the form the data does not support.

**Redirection to another register was predicted and ran backward, twice.** The three-way register field was built to separate a substitute that says less from one that says something else. The second arm fires on 2.2 percent of annotations and moves in the direction opposite to the prediction in both samples. Fallen words, not risen ones, are the words belonging to some other scene.

**Deflation to genericity does not distinguish alignment.** The claim that risen words are more often generic was tested against words that never moved and the rates were 0.501 against 0.476. That comparison is itself between two different word populations and therefore inherits the defect described at the top of this document, so it settles less than it appears to. The within-pair measurement in finding 4 is the one with a valid control, and it is positive at transgressive sites only.

**One annotator of seven is not sampling-pinned.** claude-sonnet-5 rejects the temperature parameter, so these runs should not be described as temperature-controlled.

## 8. Condensation, which no pair judgment can see

**Claim**: Freud's other dreamwork operation is many into one, and it is invisible to any question about a single pair. Asked of the substitution graph instead, it does not appear as a general funnel: the mean in-degree of risen words minus the mean out-degree of fallen ones is -0.085 at p=0.63, because the high-traffic words are hubs in both directions. It does appear as sink structure. Eight words have five or more distinct fallers arriving and none leaving, against a permutation null of 0.31, p=0.0002.

The sinks are `whispered` (12 in, 0 out), `shouted` (8, 0), `locked` (7, 0), `sighed` (6, 0), `watched` (19, 2), `began` (25, 2). Speech, affect and looking.

**Replication**: the statistic was chosen after the first test returned nothing, so it was checked on 751 word pairs that have never been annotated for anything, drawn from prompts excluded by the study's part-of-speech rule. There it is 5 against a null of 0.18, p=0.0002.

**Data**: `meta/M01_displacement/scripts/s_condensation.py`, `s_condensation.csv`.

## Limits

The study reports 33 main effects and 147 field pairs across three strata. Multiplicity is handled by Bonferroni within each family, once, and 13 main effects and 9 pairs survive. Twelve further rows survive the correction but are withheld because a cell in the underlying table falls below ten, at which point the continuity correction rather than the data is producing the number. One withheld row had a joint count of exactly zero in all four cells.

The domain gradient in finding 3 and the association structure in findings 5 and 6 were not part of the study's protocol. Neither was the markedness contrast in finding 2, which is why its replication across two samples is doing the work that a prior commitment would otherwise have done.

The annotators are language models. What is measured is how seven models read a pair of words in a slot, not how a person would. The instrument was checked on six items built to have defensible answers, and six of eight candidate annotators scored 18 out of 18; the two that did not were handled before any data was collected, one excluded and one retained with its failure recorded.

## Data and code

    frame        meta/M01_displacement/results/s_stage2_real.parquet
    annotations  meta/M01_displacement/results/s_stage2_real_long.parquet    7,135 rows
    pilot        meta/M01_displacement/results/s_stage1_50_rev3.parquet      1,400 rows
    analysis     meta/M01_displacement/scripts/s_analysis.py                 seed 20260806
    effects      meta/M01_displacement/results/s_analysis_effects.csv
    pairs        meta/M01_displacement/results/s_analysis_pairs.csv
    examples     meta/M01_displacement/results/s_analysis_examples.csv       66 items
    instrument   malign_logits/tasks/code_operation_binaries.py              revision 3
