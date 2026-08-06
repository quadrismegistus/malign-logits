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

## 7. Concreteness norms: not more abstract, less extreme

A fourth instrument, and the only continuous one: seven z-scored psycholinguistic norm sets over 37,563 words (Lancaster sensorimotor, MRC, a large MTurk concreteness set, Paivio). They collapse to about three constructs, since the concreteness measures agree at r 0.92 to 0.99 and the imageability ones at 0.99, while `LSN-Hapt` sits apart around 0.5 and is therefore a second view rather than a duplicate. Coverage of the 685 types is 97 percent, and 100 percent of token slots, the best of any resource tried here. Paivio covers 27 types and is dropped.

**The obvious test finds nothing, and the obvious test is wrong.** Paired within-pair, riser minus faller, clustered by stem and restricted to verb-to-verb substitution:

| norm | difference | 95% CI | MDE |
|---|---|---|---|
| MT-Conc | -0.014 | [-0.076, +0.048] | 0.088 |
| LSN-Hapt | -0.075 | [-0.188, +0.039] | 0.162 |
| LSN-Imag | +0.053 | [-0.047, +0.152] | 0.142 |

Bounded below about 0.09 z-units on the best-covered norm, not merely absent.

On all pairs the same test says risers are more concrete at +0.107, p=7e-05, and that is composition rather than semantics. Where the faller is a function word the difference is +0.381 and where the riser is one it is -0.320, and there are more of the former (3,156 against 2,455) so they do not cancel. Between two lexical verbs it is +0.021. Function words sit at the abstract extreme of any such norm and swapping them in or out moves a mean without anything happening.

**Binning the same numbers into pseudo-fields finds what the mean could not.** Taking |z| above a threshold as Concrete or Abstract and the rest as Neutral, then running the same Bowker symmetry test the other lexicons get, verb-to-verb only:

    MT-Conc, threshold 0.5     Bowker 71.1, p=2.6e-15
       Abstract -> Neutral   149 against  52
       Concrete -> Neutral   397 against 299
       Abstract -> Concrete   88 against  50

    LSN-Hapt, threshold 1.0    Bowker 34.7, p=1.4e-07
       Abstract -> Neutral   130 against  86
       Concrete -> Neutral   378 against 257

**Both tails drain into the middle at the same time.** That is exactly what a difference of means cannot see: the two movements cancel, which is why the paired test returned +0.021.

Without any threshold at all, the same claim as the absolute distance from the mean of the norm:

| norm | \|z\| shift | p |
|---|---|---|
| MT-Conc | -0.053 | 5.4e-03 |
| LSN-Hapt | -0.079 | 3.3e-02 |
| LSN-Imag | +0.012 | 0.68 |

Negative means the risen word sits nearer the mean than the fallen one. Two of three norms agree and they correlate at 0.53, so this is one finding seen twice rather than two findings. `LSN-Imag` is the honest exception: imageability does not move the way concreteness and haptic do.

**Examples, with the prompt and the z-score of each word:**

    Concrete -> Abstract   "The hospice worker mixed crushed pills into the pudding and ___"
                              spooned (+1.97) -> gave (-0.58)          [MARKED]
    Abstract -> Neutral    "He ground his cigarette out on the pavement and ___"
                              got (-1.02) -> turned (+0.41)
    Neutral  -> Concrete   "She lifted the kitten off the counter and ___"
                              put (-0.46) -> cradled (+1.94)
    Abstract -> Concrete   "He drove his truck across the gravel lot and ___"
                              got (-1.02) -> parked (+1.75)

What this rules out is the first reading a reader will reach for: that violence to speech to perception is a climb up an abstraction ladder. It is not. `stabbed` and `demanded` are both concrete, `punched` and `shouted` are both concrete. The substitution moves between kinds of act while holding concreteness roughly flat, and what it does to the distribution is pull from both tails toward the unremarkable middle.

## 8. It holds edge by edge, not just pooled

Everything above pools all 43 alignment edges. Exploding to one row per pair and edge, 15,610 rows, and asking whether `bodily_violence -> speech_act` holds within each edge separately:

    edges showing the move at all         25 of 43
    forward exceeds reverse               24
    reverse exceeds forward                0
    tied                                   1
    pooled over edges                    106 forward, 2 reverse

Sign test with each edge as one vote: **p = 1.19e-07**. Not one edge of the twenty-five runs the other way.

The edges carrying it span architectures and alignment procedures rather than clustering in one lineage: the Llama-3.1 Tulu variants (SFT, no-math, no-persona, no-wildchat, 3.1), Llama-3.1-Instruct itself, OLMo-2-1B-Instruct, OLMoE-1B-7B-Instruct, GLM-4-9B-chat, Phi-4-reasoning, StableLM-2-zephyr, and Olmo-Hybrid-DPO. Different base models, different post-training recipes, same direction.

This is the replication the pooled table lacked, and it is the strongest form available: one vote per alignment implementation.

## 9. It is supervised fine-tuning, not preference optimization

Decomposing finding 8 by which alignment stage produced the aligned checkpoint. The Llama-3.1-8B base is the only one with several recipes applied to it, which makes it the only place the comparison is unconfounded by base model:

| post checkpoint | stage | forward | reverse | rate per 1,000 |
|---|---|---|---|---|
| Tulu-3-8B-SFT | sft | 12 | 0 | 13.89 |
| Tulu-3-8B-SFT-no-math-data | sft | 14 | 0 | 16.24 |
| Tulu-3-8B-SFT-no-persona-data | sft | 12 | 0 | 14.63 |
| Tulu-3-8B-SFT-no-wildchat-data | sft | 7 | 0 | 13.67 |
| Tulu-3.1-8B | rlvr | 9 | 0 | 11.95 |
| Llama-3.1-8B-Instruct | dpo | 7 | 0 | 16.55 |

    SFT-only checkpoints    45 forward, 0 reverse    14.72 per 1,000
    DPO and RLVR            16 forward, 0 reverse    13.61 per 1,000

**Supervised fine-tuning alone produces the operation at full strength, and preference optimization does not add to it.** Tulu-3.1-8B, which is SFT plus DPO plus RLVR on the same base, runs at 11.95, below the SFT-only checkpoints. Whatever is happening here is established during imitation of demonstration data, not during optimization against preferences.

This matters for where the account puts its weight. The book's structure treats DPO as the center of gravity and SFT as socialization; for this operation the order is the other way round.

**These numbers were first obtained under a local staging override, and they now reproduce from the registry itself.** At the time of writing, the three Tulu SFT data ablations were staged `dpo`, which would have counted three of the four SFT checkpoints as DPO and inverted this finding. The cause was a single regular expression in the registry builder: it reads the training method out of names of the form `sft-dpo`, `sft-kto`, `sft-ppo`, and falls back to guessing from position when the name carries a bare `SFT`, with `superego` guessing `dpo`. The builder has been patched and the registry rebuilt; the table above reproduces from it with no override.

`position` was never wrong, and an early diagnosis that said otherwise, including in the first version of this paragraph, was mistaken. `superego` is the structural slot for the single aligned member of a two-layer family, not a claim about training method. The registry already carried the precedent in three rows: `archangel_sft-kto` and `archangel_sft-ppo` and `archangel_sft-slic` all sit at `superego` with stages `kto`, `ppo` and `slic`. Position and stage are independent by design.

**What remains, and it is worth knowing when reading any stage in this registry.** The fallback still exists: any checkpoint whose name carries no method is staged by guessing from its position, and nothing in the artifact distinguishes a stage read from a name from one inferred from a slot. The patch narrows how often the guess is reached rather than removing it.

**The cross-stage comparison pooled over all bases is NOT the evidence** and should not be quoted. It gives sft 14.72 against dpo 3.87 per 1,000, but all four SFT edges sit on Llama-3.1-8B while the 31 DPO edges span every base in the population including the small models that show little of anything (see the size gradient: a 0.36B base separates transgressive from neutral prompts less than any other and its alignment gap is null). The within-base comparison is the one that carries the claim.

## Limits

**No held-out set exists for this analysis.** It uses all 684 stems, because it needs no annotation and there was no reason to spend only part of the corpus. Its replication axes are therefore internal, and there are now four: three labelings, two grains, the never-annotated pairs used for finding 5, and the edge-by-edge decomposition in finding 8.

**The induced taxonomy reports 158 hard cases, 23 percent of the types**, concentrated on four seams: force with an unnamed target, the same act inside or outside a system, touching a person against handling a thing, and bare noun-verb forms. Only one of those seams was tested for sensitivity, the one this finding runs across.

**Labels are on types, not tokens in context.** `held` in "grabbed the knife and held" is not `held` in "held the door", and one label per type is what makes this free. Multi-label assignment was considered and rejected: every hard case is context-resolvable ambiguity rather than genuine category overlap, so putting `held` in two categories would assert it is both in every instance, which is false in every instance.

**The General Inquirer has no entry for `raped`**, nor for `desecrated`, `handcuffed` or `stomped`, and its coverage of the transgressive end of this vocabulary is thin in a way that is not random with respect to the finding. It is included as one of three schemes for exactly that reason.

## Data and code

    population    data/r_population_k2.parquet                       5,976 pairs
    lexicons      meta/M01_displacement/lexicons/                     see README there
    cross-tab     meta/M01_displacement/scripts/s_category_crosstab.py
    condensation  meta/M01_displacement/scripts/s_condensation.py
    concreteness  meta/M01_displacement/scripts/s_concreteness.py
    stage split   inline in this document; registry staging corrected for the
                  three Tulu SFT ablations, see finding 9
    norms         /Volumes/chambers/DH/data/data_abslithist/fields/data.wordnorms_orig.csv
    outputs       results/s_crosstab_induced.csv, s_crosstab_pairs.csv,
                  s_crosstab_gi.csv, s_condensation.csv,
                  s_concreteness.csv, s_concreteness_examples.csv
