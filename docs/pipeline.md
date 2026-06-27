# Data Pipeline — Full Model Family

Complete pipeline for populating all data stashes for a model family.
Steps in dependency order.

## Adding a new family

1. Register in `MODEL_FAMILIES` in `malign_logits/__init__.py`
2. Run steps below in order

## Pipeline steps

| # | Stash | Script / Command | GPU? | ~Time (7B, MPS) | Depends on | Notes |
|---|-------|-----------------|------|-----------------|------------|-------|
| 1 | `logits/` | `malign precompute --logits-only` | Yes | ~1 min/checkpoint | — | 1 fwd pass per prompt, full vocab logit vector |
| 2 | `beam_words/` | `scripts/compute_beam_words.py` | Yes | ~30 min/checkpoint (n=1000 d=3) | logits | Word probs via beam search, accurate for multi-token words |
| 3 | `word_probs/` | `scripts/compute_word_probs.py` | No (tokenizer only) | ~20s/checkpoint | logits + beam_words | Hybrid: exact logit (single-token) + beam (multi-token) |
| 4 | `beams/` | `scripts/compute_beam_storylines.py` | Yes | ~15 min/checkpoint (n=100 d=10) | logits | Beam storylines + cross-model teacher-forcing + resistance + entropy |
| 5 | `generations/` | `malign vllm-generate --n 100` | Yes (GPU, cloud) | ~$2 cloud / ~2h local per checkpoint | — | 100 completions per prompt, T=1.0 |
| 6 | `sent_embeddings/` | `malign embed` | No (CPU bge-m3) | ~30 min/checkpoint | generations | 1024d sentence embeddings |
| 7 | `ref_surprisal/` | `malign surprisal` | Yes (Pythia 1B) | ~1h/checkpoint | generations | Cross-entropy under reference model |
| 8 | `self_surprisal/` | `malign surprisal --self` | Yes | ~1h/checkpoint | generations | Cross-entropy under own model |
| 9 | `logits/` (continue) | `scripts/compute_mode_logits.py --mode continue` | Yes | ~1 min/checkpoint | logits | Logits with chat template applied |
| 10 | `beam_words/` (continue) | `scripts/compute_beam_words.py --mode continue --n 200` | Yes | ~30 min/checkpoint | — | Template-mediated word distributions |
| 11 | `word_probs/` (continue) | `scripts/compute_word_probs.py --mode continue` | No | ~20s/checkpoint | steps 9+10 | Hybrid word probs for continue mode |
| 12 | `mega_generations/` (optional) | `Circuit.mega_generate()` | Yes | ~30 min/checkpoint | — | Per-position entropy + top-5 trajectory. For F25 temporal signatures. Superseded by beams for most analyses. |

## Dependency graph

```
Nothing required
├─→ 1. Logits (raw)
│   ├─→ 2. Beam words (raw) ──→ 3. Word probs (raw)
│   ├─→ 4. Beam storylines (teacher-forced, resistance, entropy)
│   └─→ 9. Logits (continue) ──→ 11. Word probs (continue)
├─→ 5. Generations
│   ├─→ 6. Embeddings (CPU)
│   ├─→ 7. Ref surprisal
│   └─→ 8. Self surprisal
├─→ 10. Beam words (continue) ──→ 11. Word probs (continue)
└─→ 12. Mega-gen (optional)
```

## Time estimates

| Scale | Per checkpoint | 3-checkpoint family | 4-checkpoint family |
|-------|---------------|--------------------|--------------------|
| 1B | ~2h | ~6h | ~8h |
| 7B | ~6h + $2 cloud | ~20h + $6 | ~26h + $8 |

## Stash inventory

| Stash | Current entries | Size | Key format |
|-------|----------------|------|------------|
| `logits/` | 4,171 (+4,171 continue) | 1.4GB | `{model, prompt}` |
| `beam_words/` | 4,109 raw + 4,488 continue | ~37MB | `{type, model, prompt, n, depth, [mode]}` |
| `word_probs/` | 4,109 raw + 4,171 continue | ~26MB | `{model, prompt, [mode]}` |
| `beams/` | 4,867 | 296MB | `{model, source, prompt, ..., type}` |
| `generations/` | 213,081 | 267MB | `{model, prompt, temp, idx}` |
| `sent_embeddings/` | 449,877 | 46.6GB | `{embedder, prompt, text}` |
| `ref_surprisal/` | 605,800 | 2.9GB | `{ref, prompt, text}` |
| `self_surprisal/` | 195,483 | 731MB | `{model, prompt, text}` |
| `mega_generations/` | 6,250 | 91MB | `{model, prompt, temp, idx}` |
| `gen_logprobs/` | 20,000 | 768MB | `{model, prompt, temp, idx}` |
| `gen_annotations/` | 20,389 | 25MB | `{tagger, model, prompt, temp, idx}` |

## Ablation scripts

- `scripts/tulu_beam_ablation.py` — Tulu-specific beam + teacher-force (5 variants, restricts to Tulu only instead of full Llama family)
- `scripts/cloud_beam_annotate.py` — cloud GPU beam pipeline
- `scripts/cloud_batch_annotate.py` — cloud GPU batch annotation

## Template survey

18 checkpoints have chat templates (task switch in continue mode), 11 do not (continue = raw, serve as controls). See F32.
