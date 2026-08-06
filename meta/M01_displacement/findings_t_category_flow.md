# Findings T: where the mass goes

Unregistered. Nothing here was predicted in advance and no hypothesis was filed before the numbers existed. What stands in for a registration is that the categories were built without access to the answer, three independently constructed labelings are reported together, and the test needs no null model. Every number reproduces from `scripts/s_category_crosstab.py` and `scripts/s_condensation.py` with the seeds recorded in them.

The unit throughout is the full population: **5,976 faller-riser pairs over 1,361 prompt cells and 684 stems**, from `data/r_population_k2.parquet`. No annotation is involved. This analysis costs nothing but compute, which is the main thing to say for it.

## The test is symmetry, not a fitted null

If the direction of substitution carries no information, then the table of faller-category against riser-category must equal its own transpose: as many X-to-Y moves as Y-to-X. Bowker's test asks exactly that, and each off-diagonal pair gets an exact binomial for which direction dominates.

This is worth stating plainly because it removes the usual argument. There is no permutation, no null model, no assumption about the shape of the data. Asymmetry in that table IS the effect, and the only question is whether the categories are honest.

## 1. The table is strongly asymmetric on every labeling

| labeling | Bowker chi-square | df |
|---|---|---|
| induced, 16 categories | 834.4 | 104 |
| induced, violence boundary widened | 792.9 | 104 |
| WordNet supersenses, 15 categories | 807.2 | 99 |

All three at p below the floating-point floor. Substitution has a direction and the direction is legible at the category level.

## 2. Violence becomes speech, on three schemes built independently

**Induced taxonomy.** Sixteen categories proposed by an Opus agent shown the 685 word types shuffled, with no roles, no counts, and an explicit instruction not to infer what contrast they would serve.

    bodily_violence -> speech_act        38 against 1        p = 1.5e-10

One instance of speech becoming violence in 5,976 pairs. Clustered to one observation per stem, so that pairs sharing a prompt cannot inflate it, the count is 29 against 1 at p = 5.8e-08.

The concrete pairs: `dragged -> said`, `stabbed -> demanded`, `stabbed -> said`, `shot -> threatened`, `punched -> shouted`.

**WordNet supersenses.** External, fixed, and incapable of having been fitted to this corpus.

    contact -> communication      95 against 42
    contact -> change             59 against 12
    contact -> stative            52 against 14
    contact -> perception         48 against 9

**General Inquirer.** Multi-label, so not a partition and not cross-tabulated; it gets per-category rate tests instead, which is the analysis its structure supports. Of 50 categories with at least 20 words, 20 survive Bonferroni, and the direction is unambiguous:

| category | faller rate | riser rate | difference |
|---|---|---|---|
| IAV (interpretive action verb) | 0.202 | 0.321 | +0.126 |
| TranLw (transaction) | 0.072 | 0.189 | +0.122 |
| ComForm (communication form) | 0.074 | 0.152 | +0.084 |
| Power | 0.038 | 0.112 | +0.072 |
| COM (communication) | 0.028 | 0.081 | +0.054 |
| Perceiv (perception) | 0.026 | 0.056 | +0.033 |
| **Hostile** | **0.111** | **0.061** | **-0.051** |
| **SocRel** (social relations) | **0.094** | **0.067** | **-0.023** |

Two of the 50 categories move negatively and both are about doing something to another person: `Hostile`, and `SocRel`, which GI defines over words naming social relationships and interactions between people. Everything else that reaches significance rises.

A dictionary compiled in the 1960s Lasswell tradition, a fixed lexical database, and a taxonomy induced blind last night all put hostility on the falling side and speech and perception on the rising side. None of the three could have been fitted to the answer, and they do not share a construction principle.

## 3. The boundary the taxonomy's author flagged is not load-bearing

The Opus agent recorded that it had resolved target-ambiguous force verbs conservatively toward `object_handling`, and warned that this left `bodily_violence` under-populated at 29 types. That is the boundary this finding runs across, so the whole table was recomputed with all 23 types whose hard-case entry names `bodily_violence` as a live competitor moved into it: `grabbed`, `pushed`, `cut`, `fired`, `knocked`, `lunged`, `jerked`, `hung`, `aimed`, `dumped`, `injected`, `pointed`, `arrest`, `exposed` and nine others.

    conservative drawing     38 against 1      p = 1.5e-10
    violence-wide drawing    53 against 11     p = 1.0e-07

Same direction, both surviving Bonferroni. The labeling is not carrying the result.

## 4. Speech is a station, not the terminus

`perception_cognition` takes mass from every other category, including from speech itself:

    grammatical_function  -> perception_cognition    263 against 49
    object_handling       -> perception_cognition     78 against 12
    speech_act            -> perception_cognition     76 against 12
    locomotion_posture    -> perception_cognition     75 against 10
    person_reference      -> perception_cognition     71 against 10
    transfer_possession   -> perception_cognition     49 against 11

Nobody predicted this and it was not what the annotation study was built to find. If the chain is violence to speech to perception, then the operation does not stop at giving the impulse a voice. It ends in looking, finding and noticing, which is where the word-level analysis below independently arrives.

## 5. The graph does not funnel, but specific words are pure destinations

Freud's other dreamwork operation is many into one, and it cannot be seen in any single pair. Asked of the substitution graph instead, it fails as a general claim: the mean in-degree of risen words minus the mean out-degree of fallen ones is **+0.091**, which is nothing. The high-traffic words are hubs in both directions and they cancel.

It holds as sink structure. Counting words with many distinct fallers arriving and none leaving, against a null that flips the direction of each edge at random:

| threshold | observed | null | p |
|---|---|---|---|
| in-degree >= 5, out-degree 0 | 19 | 1.01 | 0.0005 |
| in-degree >= 8 | 5 | 0.03 | 0.0005 |
| in-degree >= 12 | 3 | 0.00 | 0.0005 |

The largest sink is `whispered`, which receives 50 distinct fallers and sends to none. The pure sources are `punched` (16), `shot` (15), `stabbed` (13), `attacked` (7), `burned` (7).

This statistic was chosen after the general funnel test returned nothing, which is disclosed in the script. It was therefore checked on 751 word pairs that have never been annotated for anything, drawn from stems the part-of-speech rule excluded from the annotation frame: there it is 5 against a null of 0.18 at p = 0.0002.

## 6. The two grains agree, which is the reason to believe either

Findings 2 and 5 are the same phenomenon measured at different resolutions, and they were arrived at by different routes: one from a category cross-tabulation, one from the degree structure of a bipartite graph with no categories at all. Mean net flow per word, in-degree minus out-degree, ordered by category:

    perception_cognition   +4.24        property_damage   -2.59
    nonverbal_expression   +2.42        bodily_violence   -3.59
    speech_act             +1.00        person_reference  -4.18

The three receiving categories against the three giving ones: **+2.71 against -0.86, Mann-Whitney p = 4.2e-04.**

An honest asymmetry in this: the giving side is concentrated and the receiving side is diffuse. Of the 22 pure sources, 4 are `bodily_violence` and 3 are `property_damage`. Of the 19 pure sinks, no single category holds more than 4, and they spread across `object_handling`, `procedural_operation`, `locomotion_posture` and `perception_cognition`. What alignment moves away from is specific. Where it goes is not.

## Limits

**No held-out set exists for this analysis.** It uses all 684 stems, because it needs no annotation and there was no reason to spend only part of the corpus. Its replication axes are therefore internal: three labelings, two grains, and the never-annotated pairs used for finding 5. The obvious remaining one, and the first thing to do next, is to split the table by model family and ask whether the asymmetry holds edge by edge rather than pooled across all 44.

**The induced taxonomy reports 158 hard cases, 23 percent of the types**, concentrated on four seams: force with an unnamed target, the same act inside or outside a system, touching a person against handling a thing, and bare noun-verb forms. Only one of those seams was tested for sensitivity, the one this finding runs across.

**Labels are on types, not tokens in context.** `held` in "grabbed the knife and held" is not `held` in "held the door", and one label per type is what makes this free. Multi-label assignment was considered and rejected: every hard case is context-resolvable ambiguity rather than genuine category overlap, so putting `held` in two categories would assert it is both in every instance, which is false in every instance.

**The General Inquirer has no entry for `raped`**, nor for `desecrated`, `handcuffed` or `stomped`, and its coverage of the transgressive end of this vocabulary is thin in a way that is not random with respect to the finding. It is included as one of three schemes for exactly that reason.

## Data and code

    population    data/r_population_k2.parquet                       5,976 pairs
    lexicons      meta/M01_displacement/lexicons/                     see README there
    cross-tab     meta/M01_displacement/scripts/s_category_crosstab.py
    condensation  meta/M01_displacement/scripts/s_condensation.py
    outputs       results/s_crosstab_induced.csv, s_crosstab_pairs.csv,
                  s_crosstab_gi.csv, s_condensation.csv
