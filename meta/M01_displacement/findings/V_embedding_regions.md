# Findings V: embedding geometry, mostly a negative

Written 2026-08-06. The plan is `registrations/plan_v_embedding_regions.md`, written before any of this ran and amended once — after the variance measurement and before any clustering — to make bare type embeddings primary. That amendment removed two free parameters rather than adding any, which is the only reason it was legitimate after seeing a number.

**This document is largely a null and should not be padded into more.** One route worked, one route failed with its artefact confirmed, and one produced a positive so small it needs its magnitude quoted with it.

## The question, and why it was not already answered

Ledger clause 6 is VERIFIED as an instrument-failure record: four similarity instruments could not locate the relation between a faller and the riser replacing it. Registration P's REF stratum failed metonymy by annotation, 1 of 3. But both concern the **paired** relation — is *this* riser near *that* faller. RH asked a **marginal** question: which neighbourhoods supply fallers, and which supply risers. Those come apart cleanly, and only the paired one was answered.

They no longer do come apart. The marginal version has now failed too.

## 1. The regional test is a null, and the artefact is confirmed

Bare `bge-m3` type vectors for the 2,268 word types with at least 40 movement rows, k-means at k = 10, 20, 35, 50, 75, 100, per region per edge the riser share minus the faller share, family as the unit across 16 families.

**There is no cluster structure to begin with.** Silhouette runs **0.046 to 0.061** across the whole sweep. k-means is slicing a continuum, and "region" is a convenience of the method rather than a property of the space.

**Artefact 1 is confirmed rather than merely uncleared.** The plan named three artefacts before looking; this is the first, that regions are word classes rather than semantic fields.

| k | regions | shift ~ open-class share | shift ~ frequency rank |
|---|---|---|---|
| 10 | 10 | r = +0.531, p = 0.114 | p = 0.65 |
| 20 | 20 | r = +0.447, **p = 0.048** | p = 0.77 |
| 35 | 32 | r = +0.506, **p = 0.0032** | p = 0.52 |
| 50 | 42 | r = +0.369, **p = 0.016** | p = 0.70 |
| 75 | 59 | r = +0.411, **p = 0.0012** | p = 0.58 |
| 100 | 69 | r = +0.411, **p = 0.0005** | p = 0.61 |

Significant at every resolution from k = 20 up, r stable around 0.37–0.53, and **more significant as power grows**. Frequency is clean throughout. A region's rise or fall tracks how content-word-y it is.

**And the load-bearing test could not run.** At k = 50, seven regions survive Bonferroni, **all seven are sinks, and all seven are 100 percent open-class**. There were **no significant sources at all**, so there was nothing to measure adjacency from. Survivor counts also scale with k — 3, 4, 6, 7, 14, 12 — which the plan flagged in advance as a pattern to distrust rather than celebrate.

**A note on how the artefact was caught, because the first pass missed it.** k was initially swept 6–20 and set to 10 by best silhouette. RH asked whether 10 was too few. It was, and the reason is sharper than coarseness: **the artefact control correlates region shift against class share with REGIONS as the unit, so at n = 10 it needed r > 0.63 to reach p < 0.05 and could not fire.** The control's power scales with k. The first run reported the confound as "not cleared" when the test had never been capable of clearing or condemning it. Widening the sweep turned an ambiguous result into a confirmation.

Silhouette was also treated as a choice when all six values were indistinguishable noise. With no cluster structure, k is a resolution dial to report across, not a parameter to optimise.

**Per the plan, this cell reads: the marginal question fails as the paired one did. Record it and stop rather than reaching for another instrument.**

## 2. Field cosine is a third grouping route and it earns its place

Not part of the original plan — RH proposed it after the regional null. Three routes now group the lexicons' fields on three principles: **Jaccard** extensionally (same words), the **agent pass** intensionally (same meaning, blind judgement), and **cosine** distributionally (same region of space, whether or not any words are shared).

It repairs a known blind spot. On the four coarse lexicons the semantic route found 29 cross-resource pairings Jaccard found none of, against 2 the other way, because RID selects by regex, WordNet by supersense and the induced taxonomy by an agent reading types — three membership rules picking different words for one field. Overlap is Jaccard's only signal. A centroid comparison needs no shared membership at all.

**The control passes**, unlike the regional test: cosine distance against difference in open-class share gives **r = +0.264** over 56,280 pairs, so class is about 7 percent of field geometry rather than being it.

**It recovers pairings we would endorse:**

| | | cos d | Jaccard |
|---|---|---|---|
| `induced:locomotion_posture` | `wordnet:motion` | 0.0017 | 0.545 |
| `induced:object_handling` | `wordnet:contact` | 0.0027 | 0.358 |
| `induced:speech_act` | `wordnet:communication` | 0.0033 | 0.466 |
| `framenet:Communication_manner` | `verbnet:manner_speaking` | 0.0079 | 0.579 |

**And 22 percent of the closest 1 percent of cross-resource pairs share no words at all** — the cell Jaccard cannot see by construction. Median Jaccard across those 435 closest pairs is 0.029, so the two routes agree at the very top and diverge quickly below it.

**Caveat that travels with it.** The zero-overlap pairs are weaker than the high-Jaccard ones. `framenet:Self_motion ↔ wordnet:contact` and `rid:sensation ↔ wordnet:motion` are not obviously one field; `Body_movement ↔ aggression` is plausible. This route adds coverage Jaccard lacks and what it adds needs judging rather than trusting. **Finding 16's count of 124 deduplicated fields is an over-count by a knowable amount, and this is how to know it.**

## 3. A relatedness floor exists and is 4.6 percent

RH's second proposal, framed deliberately not as another metonymy test. For each site, mean cosine from its fallers to its own risers, against the same fallers paired with risers from a **different prompt in the same family**.

| rung | families | own | cross-site | gap | p |
|---|---|---|---|---|---|
| base → sft | **14 of 14** closer | 0.3077 | 0.3224 | **−4.57%** | 0.0001 |
| sft → pref | **11 of 11** closer | 0.3113 | 0.3239 | −3.90% | 0.0010 |

Unanimous across families and both rungs. **A site's own risers are genuinely closer to its fallers than a stranger's are** — the floor the adjacency claim always needed and never had.

**And 95.4 percent of the faller-riser distance is the site.** The relation buys 4.6 percent. That is the shape malign found on their forced-continuation JS metric, where the headline was 88 percent instrument.

**This is not metonymy rescued, and the reason is pre-declared in the plan.** Relatedness is not adjacency. If a prompt concerns a knife, both what falls and what rises will be knife-adjacent vocabulary; a shared topic produces this pattern with no substitution relation at all.

## 4. The twin control: displacement moves AWAY from what it replaced

Section 3 could not distinguish relation from topic, and RH supplied a better control than the "same domain" one this document originally specified: **the twin.** Each M01 stem exists as a marked and an unmarked prompt differing in ONE WORD — `hammer`/`clipboard`, `knife`/`flashlight`. Same scene, same syntax, same length. Using the twin's risers as the control holds topic almost perfectly fixed, and it reuses the campaign's own minimal-pair design, which had only ever been applied to the marked/unmarked contrast and never as a null.

Three levels, so the answer decomposes rather than being a binary. `scripts/v_twin_control.py`.

| | base -> sft | sft -> pref |
|---|---|---|
| far, unrelated prompt | 0.3061 | 0.3094 |
| **own** | 0.3006 | 0.3056 |
| **twin** | **0.2997** | **0.3043** |

**The ordering is twin < own < far.**

    THE SCENE     twin - far   -2.11%   14/14 families   p=0.0001
    THE RELATION  own - twin   +0.32%    0/14 negative    p=0.0001

The scene is real and accounts for most of section 3's 4.6 percent. But **within the scene, a site's own risers are FARTHER from its fallers than its twin's risers are** — unanimously, at both rungs, p=0.0001.

**Displacement moves away from what it replaced — ON AVERAGE.** Holding the scene to a single word's difference, what rises at a site sits further from what fell there than what rises at its near-identical twin. That is the opposite of what metonymy predicts, and it is the first **signed** geometric result in the campaign: four grains have reported absences, this one has a direction.

**The qualifier is load-bearing and section 5 is why.** This is an aggregate over the sites within a family, and section 5 establishes that displacement direction is a property of the SCENE. **The result therefore does not distribute: nothing here establishes that any individual site is anti-adjacent**, only that the family-level mean is. The quotable sentence is *the relation is anti-adjacent on average, +0.32 percent against a scene effect of 2.11 percent*, and the size travels with it.

That qualifier reached the docket at [4780] before it reached this file, which is the wrong order — the file is the quotable record and it carried the stronger claim for three commits.

**Quote it with its size.** +0.32 percent of the reference against the scene's 2.11. Unanimous and tiny.

**And this was a cell the outcome map did not contain.** Three were declared — own closer than twin, own equal, all three equal — and the answer was own *farther*. The script's fall-through branch collapsed "not closer" into "topic only" and printed the wrong conclusion until it was corrected. **Plan U's map had the same hole and this one was written knowing that.** An enumeration of outcomes is only as good as its coverage of the sign, and twice now the missing cell is the one that landed.

## 5. Direction: the scene has one, alignment does not

RH's question after section 4: can a faller-to-riser vector be identified and used? Distance had failed at five grains, but **a vector asks which way rather than how far, and the two are independent** — sets can be far apart and consistently offset, which section 4 made a live possibility rather than an idle one. `scripts/v_displacement_vector.py`, run pooled and again restricted to CLAWS `vv*` lexical verbs (RH's suggestion, the same restriction `s_lexicon_crosstab` uses).

**The control fails, and the verb restriction ISOLATED the artefact rather than removing it.**

| | pooled | verbs only |
|---|---|---|
| axis ~ log frequency rank | r = +0.460 | **r = +0.554** |
| axis ~ open-class | r = +0.392 | undefined by construction |

Removing the class variance made frequency *more* visible, not less. And the verb poles are interpretable and confounded in the same breath:

    from   put, got, turn, tell, goes, pull, sat, buy, go, get, call, say, told, push, pay, throw
    to     observed, administered, determined, addressed, informed, cautioned, suggested,
           ignored, wondered, recognized, expressed, accepted, introduced, concluded

**Plain Anglo-Saxon action verbs to formal Latinate ones.** That rhymes with findings T's proceduralisation — `administered`, `cautioned`, `recognized`, `concluded` is the neighbourhood of `Caution`, `Constraint` and `Investigate`. But **Latinate words are rarer**, so "shifts register" and "shifts toward rarer words" predict the same axis and r = 0.554 cannot choose between them.

**There is no global direction.** Mean pairwise cosine between site vectors is **0.059**, against a pairing-shuffled null of 0.046 — the null keeps every faller set and every riser set and destroys only which pairs with which. The gap is real (14/14 families, p = 0.0001) and negligible. Site vectors are near-orthogonal to one another.

**The rungs are not parallel, which cuts against findings U.6.** A family's own `base>sft` and `sft>pref` axes agree at **0.238**, while two *different* families' `base>sft` axes agree at **0.323** — its own rungs less aligned than strangers' same rung, over a range of −0.86 to +0.70. U.6 concluded "one operation at two amplitudes" from word overlap and a field split; the geometric version does not support it. Held loosely, because section 5 has just established that the whole vector measure is weak, but recorded as a negative rather than as neutral.

**And the one robust result is scene-locality:**

| | cosine |
|---|---|
| a marked site's vector vs **its twin's** | **0.327** |
| vs a random site's | 0.060 |

Fivefold, 14 of 14 families, p = 0.0001, and unchanged by the verb restriction. **The direction of displacement is a property of the scene, not of alignment.**

That also explains the frequency axis rather than competing with it. If every prompt displaces in its own direction, averaging thousands of them cancels the scene-specific components and leaves only what they all share — the frequency gradient of the embedding space itself. **The pooled "axis of alignment" is an artefact of pooling directions that are not pooled in nature.**

**One control specified and not run:** residualise log-frequency out of the vectors and recompute. It separates *alignment shifts register* from *alignment shifts toward rarer words*, which are different claims and only one is interesting. Until it runs, the plain-to-formal reading is not available.

## What this leaves

Embedding geometry has now been asked about this operation at six grains: pairwise (clause 6, four instruments, failed), by annotation (Registration P's REF stratum, failed), regionally (section 1, null with a confirmed artefact), set-level within site (section 3, a 4.6 percent floor), scene-controlled (section 4, anti-adjacent), and directional (section 5, no global direction). **The ledger's original verdict survives all six: the faller-riser relation is interpretive, not geometric.**

What geometry did produce is three things, and none is a claim about alignment's direction. Section 2 is a better instrument for grouping the lexicons we already use. Section 4 is a signed result where the others had absences: displacement moves AWAY from what it replaced, small and unanimous. And section 5 is the reason the rest kept failing — **the scene has a direction and alignment does not**, so every pooled measure was averaging over the thing that carries the signal.

## Data and code

    variance     scripts/v_bge_variance.py          results/v_bge_variance.csv
    regions      scripts/v_regions.py               results/v_regions.csv
    field cosine scripts/v_field_cosine.py          results/v_field_cosine.csv
    relatedness  scripts/v_site_relatedness.py      results/v_site_relatedness.csv
    twin control scripts/v_twin_control.py          results/v_twin_control.csv
    direction    scripts/v_displacement_vector.py   results/v_displacement_vector.csv
                 --verbs for the vv* restriction    v_displacement_vector_verbs.csv
                                                    v_displacement_twin.csv
    vectors      results/v_bare_vectors.npz         bare bge-m3, 25% depth, 2,268 types
    movement     results/t_ladder_words.parquet     1,281,413 rows, 16 families
    plan         registrations/plan_v_embedding_regions.md
