# Findings T: where the mass goes

Unregistered. Nothing here was predicted in advance and no hypothesis was filed before the numbers existed. What stands in for a registration is that the categories were built without access to the answer, seven independently constructed labelings are reported together, and the test needs no null model. No annotation is involved. This analysis costs nothing but compute, which is the main thing to say for it.

**There are two units in this document and findings 1 to 9 use the weaker one.**

Findings 1 to 9 run on `data/r_population_k2.parquet`: 5,976 faller-riser pairs over 1,361 prompt cells and 684 stems, from `scripts/s_category_crosstab.py`, `s_lexicon_crosstab.py` and `s_condensation.py`. That population applies a recurrence threshold that has to be re-justified for every prompt set, and it manufactures observations by crossing every faller with every riser inside a cell.

**Those 5,976 rows are not 5,976 observations, and the counts in findings 1 to 9 should be read with that in mind.** The median stem contributes 9 of them. Finding 2 already reports its headline clustered to one observation per stem, which is why it survives; the four later lexicons were not clustered when first run. Re-tested at one vote per stem per directed pair, across all six labelings: **of the 94 reported pairs that remain testable, 70 hold and 24 do not**, and 6 more drop below the minimum cell. What is lost is named in finding 15. The direction of every finding is unchanged; the count of significant directed pairs is not.

Findings 10 to 14 run on `scripts/s_everything.py`, where **one alignment edge is one observation** and there is no threshold, no pairing and no design requirement, so all 2,190 active English prompts are in scope rather than the 1,361 built as twins. Where the two disagree the edge unit wins, and finding 10 records the one place they do.

A defect worth stating at the top because it inflated a number that was quoted before it was caught: the direction test in `s_everything.py` first computed its binomial on pooled occurrences rather than on edges, marking `grammatical_function -> object_handling` significant at p=0.0 when its edges were 43 against 40. That is a coin flip. Corrected, it turns 67,985 claimed survivors into 1,481 and 17,384 word pairs into 2,165. Every figure in findings 10 to 14 is post-correction, and the reasoning is in the source beside the fix.

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

**Four more labelings, added later, all agreeing.** Each was chosen against a gap the first three could not close, and each is run at its own granularity with a minimum cell of 10 and Bonferroni within the resource.

    USAS, verb-to-verb        S3.2 intimate/sexual -> Q2.1 speech      25 against 2
                              G2.1 crime and law   -> A10 finding      21 against 3   WITHDRAWN
                              Q2.1 speech          -> A10 finding      26 against 1
    FrameNet                  Manipulation -> Communication_manner     23 against 0
                              Request      -> Communication_manner     14 against 0
                              Statement    -> Communication_manner     16 against 2   WITHDRAWN
    VerbNet, Levin classes    cut       -> declare                     17 against 1   WITHDRAWN
                              indicate  -> manner_speaking             14 against 1
    RID                       aggression is a source in four surviving pairs
                              and a destination in none

**Three of those are marked WITHDRAWN and should not be quoted.** They do not survive one vote per stem, which is finding 15; the counts beside them are the manufactured pair counts. Everything else in this block holds, including all four WordNet moves above and `Manipulation -> Communication_manner`. The withdrawal is marked here rather than only in finding 15 because a struck claim that lives only in the section that struck it comes back as a headline later.

One of the three is load-bearing for the paragraph below it: `Statement -> Communication_manner` at 16 against 2 was the measured form of `said` becoming `whispered`. At the stem unit it is 14 against 2 and does not clear correction. The VerbNet route to the same claim, `indicate -> manner_speaking`, does hold, and the token-level result in findings 10-14 (`whispered` rising in 40 of 43 edges) holds independently of any lexicon. The claim survives; this particular measurement of it does not.

USAS is the only resource that covers this vocabulary natively, 85% of types and 96% of slots as SURFACE FORMS with no lemmatizing, and it has the words the General Inquirer lacks: `raped` is G2.1-/S3.2, `handcuffed` G2.1, `desecrated` G2.2-/A1.1.2. Its tag contents were read off the lexicon rather than from a tagset document: A10 holds `bare`, `buried`, `camouflage`, `blur`; G2.1 holds `abduct`, `apprehend`, `arraign`; S3.2 holds `copulate`, `cuddle`, `court`.

FrameNet and VerbNet both make the cut WordNet could not. `Statement -> Communication_manner` at 16 against 2 is `said` becoming `whispered`, measured; VerbNet puts `whispered`, `shouted`, `screamed`, `yelled` in `manner_speaking-37.3` while `said` sits in `say` and `told` in `tell`.

RID is Martindale's Regressive Imagery Dictionary, built on Freud's primary-process and secondary-process distinction, which is this project's own theoretical vocabulary. Its coverage is the worst here at 36% of slots and its patterns are regexes that fire on substrings, so it is reported for direction rather than for p-values. On raw slot counts `aggression` falls from 251 to 84, the largest decline of any category, while `instrumental_behavior` rises from 93 to 423 and `sensation` from 127 to 414. On the process axis the move is `emotions -> secondary` at 82 against 40: affect leaves, and it does not leave toward primary process.

Seven labelings now, and they do not share a construction principle: a dictionary compiled in the 1960s Lasswell tradition, a fixed lexical database, a corpus-linguistics tagset, Levin's syntactic-semantic verb classes, Fillmore frames, a psychoanalytic content dictionary from 1975, and a taxonomy induced blind. All put hostility and physical force on the falling side and speech and perception on the rising side. None could have been fitted to the answer.

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

## 10. The threshold-free instrument, and what it costs

Findings 1 to 9 rest on `r_population_k2.parquet`, which keeps a (faller, riser) pair only if it recurs in two or more edges, and which manufactures observations: a cell with 12 fallers and 10 risers becomes 120 rows. Both were replaced. The unit is now one alignment edge, each edge voting once, on all active English prompts rather than the 1,361 built as twins.

    marginal shift      1,929 of 12,464 category-in-stratum tests survive Bonferroni
    directed moves      1,481 of 188,782
    word pairs          2,165 of 194,154   (no lexicon at all)

**The cost is finding 2.** On the induced taxonomy the edge-unit direction test returns 8 survivors in any stratum. `bodily_violence -> speech_act` at 38 against 1 is a claim about the paired population at its own weaker unit, and the edge-unit test does not carry it. What survives on that labeling is the marginal form: violence falls and perception rises across the same edges. FrameNet, USAS and VerbNet do yield directed moves; the induced taxonomy does not.

## 11. The same direction in every stratum, on six lexicons that share no design

The 11 prompt strata are the two M01 twins, the two M03 arms, the two institutional positions by role, and five unpaired registry domains. A category is counted below only if it is a Bonferroni survivor in at least five of them and **never reverses**.

**Rises, no stratum reversing:**

| lexicon | category | strata |
|---|---|---|
| wordnet | cognition | 8 of 11 |
| framenet | Cooking_creation | 8 of 11 |
| usas | Investigate, examine, test, search | 8 of 11 |
| induced | procedural_operation | 8 of 11 |
| rid | abstraction | 8 of 11 |
| usas | Reciprocity | 8 of 11 |
| usas | Definite (+ modals) | 8 of 11 |
| verbnet | investigate | 8 of 11 |
| verbnet | preparing | 7 of 11 |
| verbnet | begin | 7 of 11 |
| usas | Attention | 7 of 11 |
| framenet | Discussion | 7 of 11 |

**Falls, no stratum reversing:**

| lexicon | category | strata |
|---|---|---|
| wordnet | contact | 10 of 11 |
| framenet | Encoding | 8 of 11 |
| verbnet | put | 8 of 11 |
| induced | bodily_violence | 6 of 11 |
| rid | icarian_imagery | 6 of 11 |
| framenet | Body_movement | 5 of 11 |
| framenet | Motion | 5 of 11 |
| rid | aggression | 5 of 11 |
| verbnet | murder | 5 of 11 |
| wordnet | competition | 5 of 11 |

The model stops touching, moving, striking and competing, and starts investigating, attending, preparing and abstracting. WordNet `contact` is the most consistent single result in the set.

**5 categories reverse between strata and they reverse coherently.** They rise in the narrative twins and in `violence`, `sexual` and `neutral`, and fall in `m03_inst`, `m03_indiv`, `inst_authority` and `inst_individual`. This is also why WordNet `cognition` is not significant pooled while being a significant riser in eight strata: the institutional prompts cancel it. Report this stratified. The pooled number hides the finding rather than summarising it.

## 12. USAS names the alignment vocabulary without being asked to

USAS is a Lancaster corpus-linguistics tagset from the 1990s with 232 semantic fields (`lexicons/usas_tagset.tsv`, from ucrel.lancs.ac.uk/usas/semtags.txt). It covers this vocabulary better than anything else tried, 89 percent of word slots as surface forms, and it has no notion of alignment. Its 45 Bonferroni survivors on all prompts, read as a list:

| code | field | shift | edges |
|---|---|---|---|
| `Q2.2` | Speech acts | +0.0157 | 31/43 |
| `T2` | Time: Beginning and ending | +0.0133 | 31/43 |
| `A1.1.1` | General actions, making etc. | +0.0122 | 36/43 |
| `X3.4` | Sensory:- Sight | +0.0079 | 35/43 |
| `X2.4` | Investigate, examine, test, search | +0.0073 | 43/43 |
| `A10` | Open/closed; Hiding/Hidden; Finding; Showing | +0.0067 | 33/43 |
| `A1.7` | Constraint | +0.0038 | 40/43 |
| `S1.1.2` | Reciprocity | +0.0028 | 42/43 |
| `S8` | Helping/hindering | +0.0027 | 34/43 |
| `A1.3` | Caution | +0.0025 | 40/43 |

and the fallers, all 4 of them:

| code | field | shift | edges |
|---|---|---|---|
| `M2` | Putting, taking, pulling, pushing, transporting &c. | -0.0268 | 7/43 |
| `Z8m` | Pronouns etc.  [m] | -0.0139 | 8/43 |
| `L1` | Life and living things | -0.0036 | 9/43 |
| `T1` | Time | -0.0020 | 10/43 |

`Attention` `Understand` `Wanting; planning; choosing` `Caution` `Helping/hindering` `Reciprocity` `Constraint` `Investigate, examine, test, search`. A tagset built for corpus linguistics thirty years ago, applied to a question it was not built for, returns the vocabulary of alignment among its 41 rising fields and physical manipulation as its largest falling one.

## 13. The withdrawal is transgression-specific; the substitution is not

The M01 twins differ in one word. If alignment were simply softening transgression, both the removal and the replacement should be larger in the marked twin. Only the removal is.

| lexicon | category | marked | neutral twin |
|---|---|---|---|
| wordnet | contact | **-0.0864** | -0.0490 |
| framenet | Cause_harm | **-0.0311** | -0.0113 |
| rid | aggression | **-0.0508** | -0.0320 |
| induced | bodily_violence | **-0.0228** | -0.0143 |
| induced | person_reference | **-0.0346** | -0.0269 |
| induced | perception_cognition | +0.0462 | **+0.0625** |
| rid | sensation | +0.0526 | **+0.0680** |
| wordnet | cognition | +0.0084 | **+0.0162** |
| wordnet | perception | +0.0268 | **+0.0338** |

Alignment removes the violent word only where there is one. It adds the deliberative word everywhere, and if anything slightly more where there was nothing to remove.

**Held to what it will bear.** On the named violence categories this is large and clean. As a claim about all 213 categories it is p=0.021 by rank test and p=0.185 parametric, with n=41 fallers against n=172 risers. The categories carry it; the omnibus does not yet.

**Tested at a second seat, on a population we did not use.** malign ran this quantity on the forced-continuation sample -- different population, different pairs, no shared derivation -- as a section of `scripts/fc_analyse.py`. Unit is the pair, n=19, under CANONICAL. Reported first in docket [4737] with only the top-of-site summary, then corrected in [4739] with both summaries after we pointed out that finding 14 predicts the top-of-site statistic to be the unstable one for risers:

    FALLER  top |delta|   marked-unmarked  +0.00294  n=16  DETECTED p=0.0279
    FALLER  sum |delta|   marked-unmarked  +0.00324  n=16  not detected, MDE 0.0092
    RISER   top excess    marked-unmarked  +0.00195  n=15  not detected, MDE 0.0096
    RISER   sum excess    marked-unmarked  -0.00095  n=15  not detected, MDE 0.0108

**The withdrawal half: the direction replicates under both summaries, the significance under one of two.** Both faller rows carry the predicted positive sign and the summed point estimate is the larger of the two, so the effect does not shrink between summaries -- the variance grows. That is support, and it is not confirmation. The accurate sentence is *the direction replicates independently under both summary choices; significance under one of the two tested*, and an earlier version of this section said "confirmed twice", which was more than the evidence carries.

**The substitution half: not supported at this power on this population, and NOT contradicted.** The sign is not stable across summaries -- top-of-site runs against our prediction at +0.00195, summed runs with it at -0.00095 -- so nothing here reverses our estimate. What survives is the bound alone: not detected under either summary, MDE 0.0096 and 0.0108, against the roughly 0.016 we report. A bounded null and a reversed estimate are different claims, and this section briefly recorded the second when only the first was shown.

The two quantities are not identical either -- ours is a category share aggregated over lexicons, theirs is a per-site magnitude -- and they can diverge honestly. **Current standing: the withdrawal asymmetry is supported at two seats, significant at one summary of two. The substitution asymmetry is unresolved, bounded under the size we claim but never tested against it at adequate power.** Our own omnibus for the riser half remains the weaker of the two tests reported above, so the paper should carry the withdrawal claim and mark the substitution claim open.

**Asked of the words directly, with no lexicon, the claim resolves into four cells and the directional half does not survive.** Breadth is words moved per site, depth is mean |delta| per word moved, crossed with role. Pairing is within stem -- both members of a minimal pair at the same edge, so the transgressive word is the only difference -- on 28,931 paired cells. The test is per EDGE, because 28,931 pairs are 43 edges times a few hundred stems and a pair-level Wilcoxon returns p=1e-17 for a 2 percent difference. `scripts/s_depth_breadth.py`.

| | marked | neutral twin | diff | | edges | |
|---|---|---|---|---|---|---|
| faller, breadth | 13.01362 | 12.64184 | +0.37178 | 2.9% | 28/43 | **detected** p=0.0009 |
| faller, depth | 0.00759 | 0.00743 | +0.00016 | 2.2% | 25/38 | **detected** p=0.0009 |
| riser, breadth | 11.93315 | 11.86883 | +0.06433 | 0.5% | 28/43 | null p=0.069 |
| riser, depth | 0.02276 | 0.02243 | +0.00033 | 1.5% | 22/38 | null p=0.464 |

**STRUCK, and left visible because it travelled.** The paragraph immediately below was the reading of the four cells above until the movers were split by word class. Under that split it is false: riser closed-class depth detects at +5.3 percent, p=0.0021, 29 of 43 edges. The clean two-detections-and-two-nulls was an artifact of aggregating over word class, which is this document's own recurring defect arriving at its newest section.

> Both faller cells detect. Both riser cells are null. That supports *substitution does not differ by markedness*, which is the defensible form of "substitution is general", and it gives no support to what this section originally claimed, that risers are larger in the neutral twin: the riser point estimates run marginally the other way and are null both times. **The neutral-twin-larger claim is dropped rather than left open.**

> The sentence the paper can carry: alignment's withdrawal is transgression-specific in how many words it pulls down and in how far it pulls them; its substitution is not transgression-specific in either.

**Split by word class, the four cells become eight and only three survive.** CLAWS open class is `vv`/`nn`/`jj`/`rr`; 77.8 percent of movement tokens are open-class. `scripts/s_class_split.py`.

| measure | role | class | marked | neutral | diff | edges | |
|---|---|---|---|---|---|---|---|
| breadth | faller | open | 12.37458 | 11.93448 | 3.7% | 30/40 | **detected** p=0.0000 |
| breadth | faller | closed | 5.21442 | 5.23230 | -0.3% | 18/40 | null p=0.586 |
| breadth | riser | open | 10.57218 | 10.62178 | -0.5% | 21/43 | null p=0.426 |
| breadth | riser | closed | 3.65169 | 3.64404 | 0.2% | 20/43 | null p=0.570 |
| depth | faller | open | 0.00689 | 0.00682 | 1.0% | 21/38 | null p=0.108 |
| depth | faller | closed | 0.00983 | 0.00934 | 5.2% | 28/38 | **detected** p=0.0000 |
| depth | riser | open | 0.01785 | 0.01790 | -0.3% | 18/43 | null p=0.223 |
| depth | riser | closed | 0.02348 | 0.02230 | 5.3% | 29/43 | **detected** p=0.0021 |

**Breadth is lexical exactly as it should be**: alignment withdraws from more OPEN-class words in the marked twin and closed-class breadth is flat and null. That is an internal control passing -- `the` and `was` are withdrawn from equally often either way.

**Depth is not, and that is the problem.** The two depth detections are CLOSED-class, in both directions, at near-identical magnitudes. An effect that moves function words equally far down and equally far up is not a withdrawal. The likeliest reading is that marked prompts simply carry sharper distributions, so every function-word delta is larger -- a property of the prompts rather than of alignment. It is testable by comparing pre-alignment entropy at marked and unmarked sites and has not been run.

**So one cell of eight is safe: alignment withdraws from more content words in the marked twin.** malign reports the opposite class assignment for depth on their population (open detected, closed null, docket [4752]), but their split classifies a SITE by its top faller while this one classifies the WORDS, which are different operations; the disagreement is not yet a contradiction. Until the same split is run at both seats, the depth leg of finding 13 carries no class-resolved claim.

**Two limits that travel with it.** The effects are small -- 2.9 percent on breadth and 2.2 percent on depth -- and significant because the population is large. And the breadth effect is carried by a tail: marked is larger at only 42.6 percent of paired cells while the mean difference is positive, so the mean should not be quoted without that share. malign's 744-site population reports breadth flat at about 1 percent and wrong-signed (docket [4748]); an effect of this size is below what that population can see, so their depth-not-breadth reading is a statement about n rather than about alignment.

**And finding 14 predicts which of those four statistics would be unstable, which is the part worth pursuing.** Fallers are few and large, so top-of-site and summed are nearly the same object and both faller rows sit close together. Risers are many and small, so the summary choice moves the riser estimate across zero. That is our finding making a checkable prediction about a different seat's instrument rather than agreeing with its output. It is testable directly: count movers per site on their population, and fallers should be few where risers are many. Not yet run.

## 14. Few large fallers, many small risers: displacement along a chain

Counting Bonferroni survivors on all prompts: **206 risers against 36 fallers**, and the fallers are **3.8 times larger** per category (mean -0.01267 against +0.00334, Mann-Whitney p=5.8e-09). The ratio exceeds one in every lexicon.

| lexicon | risers | fallers | mean riser | mean faller | ratio | largest single faller |
|---|---|---|---|---|---|---|
| framenet | 82 | 7 | +0.00152 | -0.00498 | 3.3x | Motion -0.0124 |
| gi_primary | 17 | 8 | +0.00155 | -0.00774 | 5.0x | MALE -0.0225 |
| induced | 5 | 2 | +0.02150 | -0.02212 | 1.0x | person_reference -0.0260 |
| rid | 5 | 4 | +0.01569 | -0.02760 | 1.8x | aggression -0.0327 |
| usas | 41 | 4 | +0.00233 | -0.01157 | 5.0x | Putting, taking, pulling, pushing, transporting &c. -0.0268 |
| verbnet | 50 | 9 | +0.00319 | -0.00664 | 2.1x | convert -0.0134 |
| wordnet | 6 | 2 | +0.01604 | -0.04924 | 3.1x | contact -0.0494 |

The shape is not one-for-one substitution. A few large categories drain and their mass redistributes across many small ones, which is displacement along a chain rather than swap, measured directly. Two of the seven largest fallers are `person_reference` and the General Inquirer's `MALE`: what drains is physical contact and the marking of persons and of maleness.

**The count and the magnitude do not have the same standing, and the first version of this section reported them together.** The 3.8x magnitude ratio holds in all seven lexicons. The 206-against-36 COUNT is carried by the fine-grained ones -- FrameNet 82 risers to 7 fallers, VerbNet 50 to 9, USAS 41 to 4 -- while WordNet gives 6 to 2 and the induced taxonomy 5 to 2 at a ratio of 1.0. With 15 or 16 categories there is not room for many small risers to be resolved, so the count is partly a statement about granularity. Quote the magnitude; quote the count with its resolution.

**Tested at word level, where it partly inverts, and the reconciliation is the interesting part.** malign ran the same shape on the forced-continuation population at the level of words at a site (docket [4741]) and found fallers more numerous and risers larger per mover, the inverse of the category-level result on both axes. Re-run on our own population the inversion is not clean: fallers average 12.24 per site against risers 11.38, but the medians are 9 against 10 and risers outnumber fallers at 58.6 percent of sites, the mean being carried by a right tail of faller-heavy sites (p95 35 against 25). **The direction depends on the summary**, which is the same hazard finding 13 records one level up.

Their proposed reconciliation is that falling words cluster into few categories while rising words scatter across many, so both results can hold at once. **On our data that is supported.** Drawing equal numbers of faller and riser tokens per edge so the count cannot come from having more words, 20 draws, risers occupy more distinct categories on every fine-grained lexicon: FrameNet 329.3 against 301.3 (p=1.0e-05), USAS 200.5 against 194.0 (p=7.1e-04), VerbNet 193.2 against 189.6 (p=0.040). The effects are small in relative terms, 2 to 9 percent. WordNet and the induced taxonomy cannot test it: both roles use essentially all 15 or 16 categories, which is the same saturation that limits the count claim above.

**What is NOT explained**, and it should stay that way rather than be tidied: this finding correctly predicted which of malign's four statistics would be unstable, but the reason given -- risers many and small at the level their statistic operates on -- is false on their data, where risers are few and large per site. A right prediction from a wrong mechanism is exactly the failure this document keeps booking, so it is recorded as unexplained.

**A separate result, quarantined from the above.** The `class` stratum is the census salary battery, `"The teacher earned an annual salary of $___"`, so its vocabulary is numerals and no semantic lexicon covers it (USAS 3 percent). It has 41 significant token shifts and 379 significant directed word pairs, and they run one way: `5`, `3`, `6`, `2` fall, `60`, `75`, `80`, `50` rise. Alignment raises the predicted salary figure, reliably, across the edge population. That is worth having and it is not semantic displacement; it must not be pooled with the fields above.

## 15. What findings 1-9 lose when their denominator is fixed

The 5,976 rows of the pair population are one (faller, riser) combination inside one prompt cell, so a cell with 12 fallers and 10 risers contributes 120 of them and the median stem contributes 9. The cross-tabs binomtested those rows, which makes the denominator a property of the join. `scripts/s_stem_clustered.py` re-tests every reported pair at one vote per stem.

    reported significant              100
    testable at the stem unit         94
      still significant               70
      NOT significant                 24
    below the minimum cell            6

| lexicon | reported | testable | hold | lost |
|---|---|---|---|---|
| framenet | 12 | 10 | 7 | 3 |
| induced | 18 | 18 | 12 | 6 |
| rid | 11 | 11 | 9 | 2 |
| usas | 30 | 30 | 22 | 8 |
| verbnet | 11 | 7 | 5 | 2 |
| wordnet | 18 | 18 | 15 | 3 |

**The pairs that do not survive**, with the manufactured count beside the stem count that replaces it:

| lexicon | from | to | pairs | stems |
|---|---|---|---|---|
| framenet | `Perception_experience` | `Intentionally_create` | 17:1 | 12:1 |
| framenet | `Statement` | `Communication_manner` | 16:2 | 14:2 |
| framenet | `Self_motion` | `Intentionally_create` | 20:4 | 17:4 |
| induced | `contact_care` | `perception_cognition` | 16:0 | 11:0 |
| induced | `property_damage` | `perception_cognition` | 13:0 | 11:0 |
| induced | `bodily_violence` | `locomotion_posture` | 18:2 | 16:2 |
| induced | `property_damage` | `grammatical_function` | 40:11 | 30:11 |
| induced | `grammatical_function` | `transfer_possession` | 103:50 | 68:42 |
| induced | `person_reference` | `transfer_possession` | 26:6 | 18:6 |
| rid | `sensation` | `instrumental_behavior` | 28:7 | 19:6 |
| rid | `aggression` | `temporal_references` | 19:3 | 13:3 |
| usas | `Z8m` | `Z8f` | 53:17 | 40:14 |
| usas | `N4` | `X3.4` | 31:6 | 25:6 |
| usas | `G2.1` | `A10` | 21:3 | 18:3 |
| usas | `A3` | `A10` | 25:4 | 20:4 |
| usas | `Z8m` | `A3` | 33:6 | 23:6 |
| usas | `Z8f` | `A10` | 17:1 | 12:1 |
| usas | `Z5` | `Z8mfn` | 36:11 | 22:10 |
| usas | `Z5` | `Z8f` | 153:78 | 69:55 |
| verbnet | `cut` | `declare` | 17:1 | 12:1 |
| verbnet | `escape` | `run` | 14:1 | 11:1 |
| wordnet | `contact` | `cognition` | 17:1 | 13:1 |
| wordnet | `motion` | `possession` | 63:25 | 41:22 |
| wordnet | `unassigned` | `stative` | 131:72 | 76:55 |

Finding 2's headline is not among them: it was reported clustered from the start, at 29 against 1. `contact -> communication`, `contact -> change` and the USAS and RID moves quoted in finding 2 hold. What goes is mostly the large-count pairs whose margin came from repetition inside a cell rather than from agreement across stems, which is exactly what the correction is for. **No direction reverses.** The claim that survives everywhere is the one about direction; the claim about how many directed pairs reach significance was inflated.

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
    4 lexicons    meta/M01_displacement/scripts/s_lexicon_crosstab.py
    concreteness  meta/M01_displacement/scripts/s_concreteness.py
    stem re-test  meta/M01_displacement/scripts/s_stem_clustered.py
                  one vote per stem; results/s_stem_clustered.csv and
                  s_stem_clustered_verdicts.csv; finding 15
    findings 10-14
      analysis    meta/M01_displacement/scripts/s_everything.py
                  7 labelings x 13 strata x 3 instruments, edge as unit
      movement    results/movement_words.parquet    cached walk of the store,
                  every faller and riser under CANONICAL, 43 edges
      USAS names  meta/M01_displacement/lexicons/usas_tagset.tsv
                  232 fields, ucrel.lancs.ac.uk/usas/semtags.txt
      prose       meta/M01_displacement/scripts/s_findings_t_append.py
                  emits sections 10-14 from the CSVs; no number in them is
                  typed, and it refuses to splice if the anchor is ambiguous
    stage split   inline in this document; registry staging corrected for the
                  three Tulu SFT ablations, see finding 9
    norms         /Volumes/chambers/DH/data/data_abslithist/fields/data.wordnorms_orig.csv
    outputs       results/s_crosstab_induced.csv, s_crosstab_pairs.csv,
                  s_crosstab_gi.csv, s_condensation.csv,
                  s_concreteness.csv, s_concreteness_examples.csv,
                  s_everything_marginal.csv, s_everything_direction.csv,
                  s_everything_wordpairs.csv   (the last three carry
                  category_name / frm_name / to_name for USAS)
