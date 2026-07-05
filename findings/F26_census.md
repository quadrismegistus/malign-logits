# F26: The Token-Tree Census — Variance Decomposition and the Deleter/Redirector Typology (53 models, 5 prompts)

**Summary**

Probing every registered checkpoint through the same token trees turns the collection of families into a census: alignment's structure varies far more by *training relation* (base→SFT vs base→DPO vs base→RLVR) than by organization, country, or prompt. SFT does twice DPO's distributional work, RLVR adds almost nothing, and every family resolves into one of two clinical types — **deleters** (block the base continuation, fall back to the natural substitute: repression) and **redirectors** (block and substitute something foreign: foreclosure).

**Method: token trees**

`Probe(model_id).explore_tree(prompt)` runs the *base* model and builds a branching tree of its high-probability continuations: from the prompt, branch on the tokens covering ~50% of probability mass for the first 2 depths, then follow argmax to depth 10 — ~50 readable storylines with per-fork probabilities. `batch_annotate` then teacher-forces **every other checkpoint in the family** through the same tree, recording per-node entropy, probability-of-base-token, JS divergence, and resistance (bits withheld from the base model's path). The tree is the base model's desire-structure; the annotations measure how hard each training stage pushes back at every fork. 53 models across 17 base families were annotated on 5 probe prompts (anger, sexual, labor, contradiction, neutral); trees live in the `trees` stash (`malign probe batch/status/ingest/census`), graph queries via ArangoDB (`graphdb.py`).

**Variance decomposition: relation dwarfs geography**

Across 150 pairwise comparisons (14 families × 5 prompts), the share of JS-divergence variance explained:

| Factor | Variance explained |
|---|---|
| **relation_type** (sft_of / dpo_of / rlvr_of / …) | **48.9%** |
| org (Meta, Allen AI, Alibaba, …) | 14.0% |
| country | 5.1% |
| prompt | 1.7% |
| org_type (industrial / community / academic) | 1.2% |

What *stage of socialisation* a checkpoint represents explains half the variance in how it diverges from its base; who trained it, and where, explains a fifth. The psychic architecture of alignment is more universal than its cultural provenance.

**SFT dominates DPO 2:1; RLVR adds nothing**

Measured incrementally along each family's pipeline, SFT accounts for roughly twice the distributional displacement of DPO; RLVR's increment is ~0 (consistent with F01's coalignment finding). Think-SFT is ~3× standard SFT, but Think-DPO again adds ~0 — the reasoning variants concentrate even more of the work in the supervised stage. (F33 confirms the 2:1 ratio persists at 32B.)

**Mode dwarfs alignment**

The turn-structure component (raw → chat template on the *same weights*) is JS = 0.685, nearly constant across families — 5–54× larger than the alignment component itself. The costume is bigger than the character: most of what a chatbot's distribution owes to "being a chatbot" is the template, not the weights (developed further in F32).

**The deleter/redirector typology (bidirectional census, 14 families)**

Per-token resistance is spiky, concentrated at action-verb onsets (10+ bits), not spread over semantic content. At blocked forks, two clinical responses:

- **Block + default** — suppress the base token, substitute the *natural* next candidate. Deletion; the paradigmatic neighbourhood survives. Repression.
- **Block + redirect** — suppress and substitute something *foreign* to the base distribution. Active displacement. Foreclosure.

Block+default dominates every family, but the ratio types them: **10/14 families are deleters** (resistance > foreignness), **4/14 are redirectors**. OLMo 7B is the purest redirector; Falcon the purest deleter (zero redirects). Five structural cases cover all annotated forks: repress, constitute, redundant, restore, formatting-only.

**Depth profiles and path dependency**

Hidden-state depth profiles fall into four architectures: total rewrite (OLMo 1B), distributed (Llama/Dolphin), mid-network (Falcon), superficial (SmolLM2) — the internal geography behind F05's logit-lens styles. Teacher-forcing shows 88–96% of free-generation hidden distance is path dependency; position-0 metrics are the clean measurements.

**Null results**

Embedding travel does not track alignment intensity (r = 0.07). Safety data volume does not predict alignment depth: Dolphin (community, no safety corpus) sits *deeper* than Llama-Instruct.

**Data**: `data/tree_census.csv` (53 models × 5 prompts), `data/circuit_census_grid_final.csv`, `data/profiles/`. **Figures**: `figures/F26_three_distributions.png`, `figures/F26_two_timescales.png`, `figures/F26_jakobson_space.png`, `figures/F26_tree_entropy_*.png`, `figures/clinical_signature_census.png`. **CLI**: `malign probe batch/census`.
