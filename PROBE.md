# Probe: Data Collection and Metrics Reference

## What Probe Stores

Per model × prompt × generation × position:

| Stash | Key | Shape | What |
|---|---|---|---|
| `probe_logits` | `{model, prompt, gen, pos, T}` | `(vocab_size,)` float32 | Full logit vector before softmax |
| `probe_hidden` | `{model, prompt, gen, pos, T}` | `(n_layers, hidden_dim)` float32 | Hidden states at all transformer layers |
| `probe_meta` | `{model, prompt, gen, T}` | list of dicts (one per position) | Entropy, chosen_token, argmax, top-5, etc. |
| `probe_embeddings` | .npy file per model | `(vocab_size, hidden_dim)` float32 | Input embedding matrix (lazy, on demand) |

`T` = max_tokens in the cache key. Different-length runs coexist.

Positions 0 through T-1 are stored. Position 0 is computed from the prompt alone (pre-sampling) — clean for cross-model comparison. Positions 1+ process model-specific generated tokens.

---

## Collection

```python
from malign_logits.probe import Probe

# Single model
p = Probe("allenai/Olmo-3-1025-7B")
p.collect(n=2, max_tokens=100)     # 2 gens × 5 prompts × 100 tokens

# All variants of a base
Probe.collect_tree("olmo3-7b", n=1, max_tokens=100)

# Teacher-forced (feeds model A's tokens through model B)
aligned = Probe("allenai/Olmo-3-7B-Instruct-DPO")
aligned.teacher_force("allenai/Olmo-3-1025-7B", "anger", max_tokens=100)
```

Always stores logits + hidden states + meta at every position. Embeddings stored lazily on first `embedding_matrix()` call.

---

## Metrics Catalogue

### Tier 1: Logits only (position 0)

**Cross-model, clean (same input at position 0):**

| Metric | Function | Inputs | Returns | Interpretation |
|---|---|---|---|---|
| JS divergence | `js_divergence(a, b)` | two logit vectors | float (nats) | Total distributional change |
| Bits of resistance | `base_token_surprisal(base, aligned)` | two logit vectors | float (bits) | How much aligned suppresses base's argmax |
| KL divergence | `kl_divergence(p, q)` | two logit vectors | float (nats) | Asymmetric information loss |
| Top-k overlap | `top_k_overlap(a, b, k=50)` | two logit vectors | float [0,1] | Token set preservation |
| Rank correlation | `rank_correlation(a, b)` | two logit vectors | float [-1,1] | Ordering preservation |
| Effective vocab | `effective_vocab(logits)` | one logit vector | int | Tokens with p > 0.001 |
| Tail redistribution | `tail_redistribution(base, aligned, k=100)` | two logit vectors | dict | Head→tail mass movement, thin vs spread |
| Top movers | `top_movers(a, b, k=20)` | two logit vectors | dict | Repressed + amplified token IDs |
| Base top-k mass | `base_top_k_mass(base, aligned, k=10)` | two logit vectors | float [0,1] | How much aligned preserves base's top-k |
| Compare (all T1) | `compare(base, aligned)` | two logit vectors | dict (15 metrics) | All of the above in one call |

**Batch:**

| Metric | Function | Returns |
|---|---|---|
| Circuit summary | `circuit_summary(dive, prompt)` | dict: stage shares, argmax tracking |
| Circuit profile | `circuit_profile(dive, prompt)` | DataFrame: edges + nodes |
| Compare tree | `Probe.compare_tree("olmo3-7b", "anger")` | DataFrame: all variants vs base |
| Mode decomposition | `mode_decomposition(dive, prompt)` | DataFrame: alignment vs template vs tokens |

### Tier 2: Logits + Embedding matrix

| Metric | Function | Inputs | Returns | Interpretation |
|---|---|---|---|---|
| Violence/procedural axes | `violence_procedural_axes(embed, tok)` | embedding matrix + tokenizer | two unit vectors | Semantic axes |
| Axis loading | `axis_loading(logits, embed, axis)` | logits + embed + axis | float | E_p[embed · axis] |
| Word probabilities | `word_probabilities(logits, tok, words)` | logits + tokenizer + word list | dict | P(word) under distribution |
| Formation | `formation_trajectory(probe, ...)` | probe + checkpoints | DataFrame | Token probs across base→SFT→DPO |

### Tier 3: Hidden states (position 0 — clean cross-model)

| Metric | Function | Inputs | Returns | Interpretation |
|---|---|---|---|---|
| Hidden distance | `hidden_distance(h_base, h_aligned)` | two (n_layers, dim) | float [0,1] | Mean cosine distance across layers |
| Linear CKA | `linear_cka(h_a, h_b)` | two (n_examples, dim) | float [0,1] | Representational similarity |
| Layer profile | `p.layer_profile("anger")` | probe with hidden states | DataFrame | Per-layer cosine similarity + logit lens entropy |

### Tier 3b: Hidden states (all positions — within-model, no confound)

| Metric | Function | Inputs | Returns | Interpretation |
|---|---|---|---|---|
| Axis trajectory | `p.trajectory("anger", "violence")` | probe | DataFrame | Violence/procedural loading at each position |
| Centroid distance | `centroid_distance(p_base, p_aligned, "anger")` | two probes | dict | Generation cloud separation in hidden space |
| Internal drift | `internal_drift(dive, cp, prompt)` | probe | dict | Cosine drift between consecutive hidden states |

### Tier 3c: Hidden states (cross-model temporal — CONFOUNDED)

| Metric | Function | Issue | Use instead |
|---|---|---|---|
| Hidden divergence trajectory | `hidden_divergence_trajectory(...)` | 88-96% path dependency | Position 0 only, or teacher-forcing |

### Tier 4: Meta (within-model temporal — clean)

| Metric | Function | Inputs | Returns | Interpretation |
|---|---|---|---|---|
| Narrowing ratio | `narrowing_ratio(entropies)` | entropy array | float | H(last)/H(first): <1 = narrows |
| Narrowing rate | `narrowing_rate(entropies)` | entropy array | float | Linear slope of entropy |
| Logit drift | `logit_drift(dive, cp, prompt)` | probe | dict | JS between consecutive positions |

### Teacher-forced metrics (clean temporal cross-model)

| Metric | Source | What it shows |
|---|---|---|
| Per-token resistance | `aligned.meta("anger::tf_Base", gen=0)` → chosen_prob | How much aligned resists each base token |
| Per-token foreignness | `base.meta("anger::tf_Aligned", gen=0)` → chosen_prob | How foreign aligned's tokens are to base |
| Block+default rate | High fwd surprisal + low rev surprisal | Deletion with natural substitute |
| Block+redirect rate | High fwd surprisal + high rev surprisal | Active conceptual displacement |

---

## Robustness Guide

| Comparison | Position 0 | Positions 1+ | Teacher-forced |
|---|---|---|---|
| **Cross-model logits** | ✅ Clean | ⚠️ Path-dependent | ✅ Clean |
| **Cross-model hidden** | ✅ Clean | ❌ 88-96% artifact | ✅ Clean (but flat, = pos 0) |
| **Within-model temporal** | n/a | ✅ Clean | n/a |
| **Centroid distance** | n/a | ⚠️ Mixed (cloud summary) | n/a |
| **Axis trajectory** | ✅ Clean | ✅ Within-model | n/a |

---

## Tree Metrics (n=100 generations, T=10)

The generation tree: position 0 is deterministic, branching starts at the first sampled token. n=100 samples the tree.

| Metric | Function | Returns | Interpretation |
|---|---|---|---|
| **Tree metrics** | `p.tree("anger")` | dict: branches, entropy, distribution | Tree structure for one model |
| **Tree compare** | `p.tree_compare(other, "anger")` | dict: tree JS, survival, novel/pruned | How alignment reshapes the tree |
| **Branch trajectory** | `branch_trajectory(p, "anger", "kill")` | entropy mean±std per position | How a specific branch evolves |
| **Cross-model branch** | `cross_model_branch(base, aligned, "anger", "kill")` | JS per position on matched branch | Clean temporal comparison |
| **Convergence depth** | `convergence_depth(p, "anger")` | depth where branches become indistinguishable | How deep the tree matters |
| **Branches** | `p.branches("anger")` | dict: token → gen indices | Group gens by first token |

Three alignment archetypes visible in tree structure:
- **Broad (repression)**: many branches, low concentration — energy scattered across substitutes
- **Concentrated (foreclosure)**: few branches, high concentration — tree collapses to one path
- **Kill-dominant**: base models or light alignment — original drive preserved

Universal branches: "hit" in 17/17 models, "kill" 16/17 — the internet's consensus anger.

---

## Data Inventory

```python
Probe.inventory()           # all models with data
Probe.families()            # 17 families with metadata
Probe.compare_tree("olmo3-7b", "anger")  # cross-variant table
p.tree("anger")             # tree structure
p.tree_compare(other, "anger")  # tree comparison
```

Current: collecting n=100 × T=10 × temp=1.0 for all 59 registered models.
