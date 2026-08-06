# Plan V — do certain regions of embedding space rise and fall? A PLAN, NOT A REGISTRATION.

Written 2026-08-06, **before the measurement**. Filed here because this directory holds plans. **Not a formal registration and not to be cited as one** — nothing frozen, nothing sealed, no escrowed producer, no hash, no gate. See `registration_b_provenance.md` in this directory for what a real one looks like. The findings write-up, if there is one, becomes `findings/V_embedding_regions.md`; until then nothing here is a result.

What is kept from the registration apparatus is the one cheap part, and plan U showed exactly why: **say what each outcome means before running, and include the outcomes where the instrument fools you.** U's outcome map had four cells and the answer landed in a fifth — a mechanical artefact of the movement floor — which is the one thing it should have anticipated. This plan enumerates the artefacts first.

## The question, and why clause 6 does not already answer it

RH's: *do certain regions of the embedding space rise or fall?*

**This is not the question ledger clause 6 settled.** That clause is VERIFIED as an instrument-failure record: four similarity instruments — WordNet, contextual cosine, inverted syntagmatic, embedding percentile — all fail to locate the relation between a faller and the riser that replaces it, and blind judgment reads it instantly. Metonymy-as-adjacency also died in Registration P's REF stratum, 1 of 3.

Clause 6 is about the **paired** relation: *is this riser near that faller?* This plan asks a **marginal** question: *which neighbourhoods supply fallers, and which supply risers?* Those come apart cleanly — every faller can be drawn from one region and every riser from another while which specific riser replaces which specific faller stays arbitrary. That is not a hypothetical: findings T describes exactly that shape, one tight falling field against a diffuse rising one, and findings U.6 adds that the rungs are one operation at two amplitudes.

**Do not re-run the pairwise test. It is answered.**

## Two routes, and the second is RH's

**Route A — the model's own space.** `data/raw/cache/preop_embeddings`: 36 GB, 79,397 records, 14 pre-operation checkpoints, 590 prompts, keyed `{model, prompt, tok}`, valued `role` (faller/riser), `word`, `n_tok`, and a hidden-state array over every layer. Produced 2026-07-29 by `f13_base_embeddings.py` as the fix for docket [442]'s aligned-embedding defect, and **never analysed by anything** — `git log --all -S` returns only the producer, a backfill, a repair, a census and a preservation commit.

Its awkwardness is fatal to a simple design: 14 checkpoints live in 14 incompatible spaces, different dimensionalities and arbitrary rotations, so clustering must happen per model and combine by vote, and **no region has a stable identity across models.**

**Route B — one external encoder, all sites in one space.** Embed `prompt + word` with a single pretrained encoder and take the hidden state at the target word's position. Every site in the study becomes comparable, regions become nameable, and the test is a single clustering rather than fourteen. What is given up is that the geometry is English's rather than the model-under-study's — which mattered for the paired question and matters much less here, since neighbourhoods are the object.

**Route B is primary. Route A is the check**, because agreement between a shared external space and the models' own spaces is worth more than either alone.

**AMENDED 2026-08-06, after the variance measurement and before any clustering.** Route B's step 1 ran (`scripts/v_bge_variance.py`) and changed which arm should be primary, so the change is recorded here with its reason rather than made silently.

    within-prompt share of variance, BAAI/bge-m3, contextual
        layer  6 (25%)   79.9%
        layer 12 (50%)   75.3%
        layer 18 (75%)   73.1%
    agreement between contextual centroids and BARE type vectors, Spearman
        layer  6   +0.866      layer 12   +0.803      layer 18   +0.638
        over 450 word types

Three quarters of the variance is the word rather than the prompt, so the contextual arm is clusterable and the confound this plan worried about is small in practice. **But the bare arm agrees with it at rho 0.87**, which means context refines the arrangement rather than creating it -- and the bare object is strictly better to cluster: one vector per TYPE instead of one per site, **no prompt confound at all rather than a small one**, no centring decision to defend, no layer choice to defend (there is no context for depth to accumulate), and a few thousand vectors instead of tens of thousands.

**So: BARE TYPE EMBEDDINGS ARE PRIMARY. The contextual arm becomes the robustness check, and route A the second check.** This removes two declared free parameters rather than adding any, which is the only reason it is a legitimate change to make after seeing a number.

## What is already established, so it is not re-litigated

**The prompt-dominance confound does not sink this.** The obvious objection is that a contextual embedding is mostly its prompt, so clustering finds prompts and every cluster comes out 50/50 faller/riser by construction. Measured on route A: `scripts/t_preop_variance.py`, 13 checkpoints above a 200-site floor, single-token records, layer at 50% depth. **Median within-prompt share of variance 61.3%, range 37.7–72.1.** The word contributes the majority. Centring within prompt is therefore optional rather than forced, and both raw and centred are defensible.

**The same check must be re-run for route B before clustering anything.** A different encoder can have a different balance and the number above does not transfer.

## Declared before looking

- **Encoder (route B):** one model, fixed in advance. `BAAI/bge-m3` unless the variance check below fails on it, in which case the fallback is `Alibaba-NLP/gte-multilingual-base`, and the reason for switching is recorded. The repo's existing `DEFAULT_EMBEDDER` (`paraphrase-multilingual-MiniLM-L12-v2`) is **not** used: it was chosen for embedding generated passages in the drift work, which is a different job.
- **Position:** the hidden state at the final token of the target word, not a pooled sentence vector. Sentence poolers are trained on the pooled representation; token states are what this needs.
- **Layer (route A):** 50% of depth as primary, 25% and 75% as sensitivity. That is the producer's own registered read and it is fixed now, because choosing a layer after seeing results is the free parameter that would sink this.
- **k:** chosen by silhouette over a declared range, reported with its sensitivity. Not tuned to make a region significant.
- **Statistic:** per region per edge, share of riser sites minus share of faller sites — the same marginal statistic as findings 11–16, so the numbers are comparable to work already done.
- **Unit of test:** the EDGE, one vote each, as everywhere in T and U. Not sites, which would let one verbose edge carry it.
- **Adjacency, if there are sources and sinks:** distance between source centroids and sink centroids against a null that permutes region labels.

## What each outcome means, including the ones where the instrument is fooling us

**ARTEFACT 1 — the regions are word classes.** We already know fallers and risers differ by open/closed class: findings T's class split found content-word breadth detecting at +3.7% while function-word breadth is flat, and the ladder's fallers are heavy on `a, the, he, i, put`. If k-means separates open from closed class, then "regions rise and fall" is the class effect wearing a new hat. **Control: print the open/closed composition of every region before reading any rise/fall number.** If the significant regions are class-pure, the result is not about semantics.

**ARTEFACT 2 — the regions are frequency bands.** High-frequency words move more and also cluster together in embedding space. **Control: report median corpus frequency per region**, and check whether the source/sink split tracks it.

**ARTEFACT 3 — the regions are prompts.** Handled above for route A at 61.3%; must be re-measured for route B and reported, not assumed.

**RESULT 1 — some regions are net sources and others net sinks, and they are not class- or frequency-pure.** This gives **lexicon-free fields**: semantic neighbourhoods discovered from geometry, with no taxonomy, no coverage gap, no misleading category names, and none of finding 16's deduplication problem. It would let the central claim of findings T be restated without depending on any lexicon.

**RESULT 2, the only load-bearing one — the draining and filling regions are ADJACENT.** That is metonymy at the regional grain: the chain operating between neighbourhoods rather than between words. Metonymy is currently twice-failed, geometrically at clause 6 and by annotation in P's REF stratum. Reviving it at a grain where it has not been tested would be a genuine theoretical result and the only outcome here that changes the argument rather than decorating it.

**RESULT 3 — regions are sources and sinks but scattered, no adjacency.** Confirms the field story lexicon-free and leaves metonymy where it is. Worth having, not worth much compute.

**NULL — no region is a reliable source or sink.** Then the marginal question fails as the paired one did, and the honest conclusion is the ledger's: this relation is interpretive, not geometric, and further embedding work is not indicated. **Record it and stop** rather than trying a fifth instrument.

## Scope and what it will not support

Route A is **590 prompts and 14 checkpoints**, against findings T's 2,190 and 43. Route B can cover the full movement vocabulary but only by embedding words in prompts we choose, so its coverage is a design decision rather than a given. Either way this is a claim about the population measured and not about the roster.

And the unit problem from findings U travels: AI2 is 6 of 16 families, the registry has no lineage field, and the campaign's own ruling is that a floor counts independent sources rather than pairs. Any per-edge test here inherits it.

## Results

**Run 2026-08-06. They live in `findings/V_embedding_regions.md`; this file stays as written.**

The declared **NULL** cell is the one that landed: no region is a reliable source, the class artefact this plan named first was CONFIRMED (r +0.37 to +0.53, p to 0.0005, at every k from 20 up), and the space has no cluster structure to cluster (silhouette 0.05). The adjacency test — the only load-bearing outcome — could not run at all, because there were zero significant sources to measure from.

Two things this plan did not anticipate. **The artefact control was itself underpowered at the k it was first run at**: correlating region shift against class share uses REGIONS as the unit, so at k=10 it needed r>0.63 and could not fire. RH caught it. The control's power scales with k and the plan should have said so. And **a third grouping route appeared** — field-centroid cosine — which is not about alignment at all but repairs a known blind spot in finding 16's deduplication.

Section 3's relatedness floor carries a specified, unrun discriminator. Until it runs it supports "the sets are related" and nothing stronger.
