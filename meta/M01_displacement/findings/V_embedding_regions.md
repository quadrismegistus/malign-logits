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

**THE DISCRIMINATOR IS SPECIFIED AND NOT RUN.** The null above draws risers from any other prompt in the family, spanning violence, sexual, institutional and neutral. A tighter null draws them from a different prompt **in the same domain**. If the 4.6 percent survives that, it is relatedness beyond topic and metonymy has its first positive evidence at any grain. If it collapses, the 4.6 percent was topic. **Until it runs, section 3 supports "the sets are related" and nothing stronger.**

## What this leaves

Embedding geometry has now been asked about this operation at four grains: pairwise (clause 6, four instruments, failed), by annotation (Registration P's REF stratum, failed), regionally (section 1, null with a confirmed artefact), and set-level within site (section 3, a 4.6 percent floor). **The ledger's original verdict survives all four: the faller-riser relation is interpretive, not geometric.**

What geometry did produce is section 2 — not a claim about alignment at all, but a better instrument for grouping the lexicons we already use.

## Data and code

    variance     scripts/v_bge_variance.py          results/v_bge_variance.csv
    regions      scripts/v_regions.py               results/v_regions.csv
    field cosine scripts/v_field_cosine.py          results/v_field_cosine.csv
    relatedness  scripts/v_site_relatedness.py      results/v_site_relatedness.csv
    vectors      results/v_bare_vectors.npz         bare bge-m3, 25% depth, 2,268 types
    movement     results/t_ladder_words.parquet     1,281,413 rows, 16 families
    plan         registrations/plan_v_embedding_regions.md
