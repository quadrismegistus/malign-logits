# `data/` — provenance map

Where each number in the paper comes from: analysis area → producing command → key output files → finding. This is a guide, not an exhaustive listing (261 tracked files); when in doubt the finding file in `findings/` names its own data, and the CLI/script is the source of truth.

**Two layers of data:**
- **`data/raw/cache/`** — lmdb stashes (logits, generations, embeddings, surprisal, beams, trees). **Gitignored** (75 GB). Rebuilt from models via the CLI; every expensive computation is cached here so reruns are instant. Access only through `malign_logits.cache.CacheManager` / `open_stash`.
- **`data/*.csv|*.parquet|*.jsonl`** — committed analysis outputs derived from the cache. The tables below.

---

## Logit-level battery

| Finding | Command | Key outputs |
|---|---|---|
| F01 logit analysis | `malign precompute --logits-only`; `notebooks/F01_logit_analysis.ipynb` | (reads `logits` stash) |
| F02 cross-family logits | `malign battery` | `battery_results.csv`, `battery_<family>.csv` |
| F06 baseline validation | `malign battery` | `transgressive_mass.csv`, `battery_results.csv` (perplexity/rank-corr cols) |
| F09 Tulu vs Llama | `malign battery --family tulu` | `battery_tulu.csv` |
| F10 SFT ablation | `malign ablation`; `scripts/sft_ab_experiment.py` | `ablation_results.csv`, `logits_sft_*/` |

## Displacement taxonomy & Jakobsonian axes

| Finding | Command | Key outputs |
|---|---|---|
| F08 taxonomy | `malign taxonomy [--all-prompts]` | `taxonomy_olmo.csv`, `taxonomy_<family>.csv` (was `displacement_taxonomy.csv`, renamed at 39a3886) |
| F13 paradigmatic/syntagmatic | `malign taxonomy --analyze` | `taxonomy_<family>.csv`, `taxonomy_summary.csv` |
| F14 syntagmatic baseline | `malign taxonomy --baseline --family olmo` | `taxonomy_olmo.csv` (`syntagmatic_js_aligned`) |
| displacement method | `scripts/compute_bits_resistance.py` | `bits_resistance.csv`, `bits_resistance_datadriven.csv`, `displacement_agreement.csv` |

## Generation & passage metrics

| Finding | Command | Key outputs |
|---|---|---|
| F03 cross-family generation | `malign generate-battery` | `gen_battery_metrics.csv`, `gen_battery_raw.parquet` |
| F15 passage metrics | `malign topic-drift`; `scripts/corpus_metrics.py` | `corpus_metrics.parquet`, `passage_metrics.csv` |
| F16 corpus comparison | `scripts/corpus_metrics.py` | `corpus_metrics.parquet`, `corpus_metrics.md`, `dreams.csv`, `c20_fiction_metrics.csv` |
| F17 cross-generation MMD | `scripts/cross_generation_mmd.py` | `mmd_cross_generation.csv` |

## Network-internal analysis

| Finding | Command | Key outputs |
|---|---|---|
| F05 logit lens | `malign logit-lens` | `logit_lens_datadriven.csv` |
| F04 step-level | `malign step-analysis` | `step_analysis_*.csv` |
| F12 fold / trajectory | `malign trajectory` | `intervention_<family>.csv`, `trajectory_geometry_<family>.csv`, `fold_rank_summary.csv` (see `scripts/rederive_f12_heldout.py` for the corrected held-out closure) |
| F22 circuit decomposition | `scripts/decompose_circuit.py` | `circuit_decomposition*.csv`, `attention_*.csv` |
| F35 architecture independence | `scripts/decompose_circuit.py` | `architecture_independence.csv`, `value_vector*.csv` |

## Information-theoretic

| Finding | Command | Key outputs |
|---|---|---|
| F18 Shannon / self-surprisal | `malign surprisal --self` | `shannon_entropy.csv`, `self_surprisal.csv`, `asymmetric_entropy.csv` |
| F19 BOS entropy / BLT | `malign bos-generate`; `scripts/blt_combined.py` | `jakobson.parquet`, `bos_surprisal.csv`, `blt_combined.csv` |
| F24 pretraining emergence | `scripts/pythia_battery_emergence.py` | `pythia1b_battery_emergence.csv` |

## Census, trees & variance decomposition

| Finding | Command | Key outputs |
|---|---|---|
| F26 token-tree census | `malign probe batch` / `census` | `tree_census.csv`, `circuit_census_grid_final.csv`, `profiles/` (magnitude decomposition; see note in F26 vs F31) |
| F31 PERMANOVA | (reads `word_probs` stash; `scripts/compute_score_vocab.py`) | word-prob vectors → PERMANOVA in notebook (position + delta-direction decomposition) |
| F33 scale effects | `scripts/scale_test_full.py` | `logits_32b/`, `logits_70b/` |

## Reasoning, template, institutional, contradiction

| Finding | Command | Key outputs |
|---|---|---|
| F21 institutional alignment | `malign api-generate`; `scripts/score_institutional.py` | `f21_institutional_generations.csv`, `institutional_generations.jsonl` |
| F23 reasoning distillation | `scripts/r1_full_battery.py` | `circuit_complete_*.csv`, `circuit_megagen_*.csv` |
| F25 temporal signature | `Circuit.classify_mega_gen` | `f25_signature_summary.csv`, `mega_gen_*.csv` |
| F32 template-mediated | `scripts/template_comparison.py` | `continue_mode_comparison.csv` |
| F11 contradiction | `scripts/contradiction_test.py` | `contradiction_detail.csv`, `contradiction_lovehate_tagged.csv` |
| F27/F28 beams & resistance | `malign probe batch` (beams stash) | `bidirectional_resistance.csv`, `bits_resistance.csv` |

## Human-corpus reference data (inputs, not outputs)

`dreams.csv`, `arxiv_abstracts_500.csv`, `c20_fiction_metrics.csv`, `lltk_narration_3k.jsonl` — external corpora ingested via `malign ingest` for the corpus-comparison baselines (F16).

---

*Regenerate any group from scratch by deleting its outputs and rerunning the command; the `data/raw/cache/` stashes make reruns cheap. The README/findings are rebuilt with `scripts/build_readme.py build`.*
