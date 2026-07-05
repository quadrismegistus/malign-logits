# TODO — Malign Logits

## Completed model families (2026-06-27 → 2026-06-30)
- [x] **Archangel** — DONE. 4 methods on Pythia 2.8B. Method interchangeable.
- [x] **OLMo Hybrid 7B** — DONE. SSM-Transformer hybrid, displacement works.
- [x] **Falcon-Mamba 7B** — DONE. Pure SSM, displacement works.
- [x] **45 families total** with word_probs, 40 with logit lens, 57 aligned models with reverse beams.

## Deferred (not needed for CI article or book draft)
- DeepSeek V3.1 671B: needs vLLM, abandoned at HF transformers level
- Think mode for word_probs: book-phase experiment
- InternLM2: custom code incompatible, low priority (CT-LLM covers Chinese)

Work items from session 2026-06-20/21. Pick up when idle.

## Code fixes
- [x] OLMo worker "unclassified" — DONE. Base top1 was NaN (newline). Added BLANK_SENTINEL handling. Now correctly classified as de_foreclosure (100%)
- [x] Circuit `Mode.CHAT`/`Mode.THINK` test — DONE. Wiring verified on SmolLM3. CHAT=THINK for SmolLM3 (both → `<think>` at p=1.0). JS(RAW,CHAT)=0.693 (max). Fixed import bug (get_base_logits was in models.py not core.py)
- [x] `malign circuit` CLI command — DONE. `malign circuit [--family X] [--save]` classifies F25 signatures from mega-gen CSVs
- [x] Merge salary CSVs — DONE. `salary_all.csv` already has all 5 families (6000 rows)

## Experiments to run
- [x] **Pythia 6.9B BLT scoring** — DONE. Confirms 1B finding: text plateaus at step 5000 at both scales (1B: 1.60 bpc, 6.9B: 1.50 bpc)
- [x] **Qwen3-8B native thinking** — DONE. Battery + mode comparison + 1500-row mega-gen. BASE forecloses anger (top1=______), aligned RESTORES kill (de-foreclosure inverted). CHAT=THINK (both → <think> p=1.0)
- [x] **olmo-think battery** — DONE. Think-SFT produces LESS displacement on 7/10 prompts. Anger: JS 0.13 vs 0.41 (Think preserves "break", standard forecloses to blanks). Reasoning training produces lighter alignment footprint on same base

## Post-training dataset topic analysis
- [x] **Keyword-based topic proportions** — DONE. Pile 4.42%, Tulu 1.79%, UltraFeedback 2.13%, HH-RLHF 1.09%, OpenAssistant 0.53%. Capital as form not content
- [x] **Pile subset audit** — DONE. FreeLaw 14.99%, Wikipedia 8.48%, Pile-CC 7.00%. Full causal chain quantified

## Full circuit mega-gen campaign
- [x] **Small families** (smol, qwen-tiny, olmo-tiny) — DONE. 10k+15k+20k rows
- [x] **Medium families** (smol3, qwen3) — DONE. SmolLM3 raw cached, qwen3 RAW+CHAT+THINK complete (17.5k rows)
- [x] **7B families** — DONE. 7 cloud families: olmo(35k), olmo-think(25k), llama(15k), amber(15k), qwen(15k), zephyr(25k), deepseek-7b(15k). 146 prompts, ~$2
- [x] **Census grid** — DONE. 11 families × 5 prompts. 6 distinct worker mechanisms. Committed 7fc4ba6
- [x] **COMPLETE mode** — DONE. 6 cloud families + qwen-tiny local. Three-way decomposition: OLMo non-monotonic (It→reopen), OLMo-think compliance (Okay), Qwen tightens, Llama irrelevant, Zephyr transparent on worker
- [x] **Reasoning variants** — DONE (from earlier session). R1-Distill-Llama + R1-Distill-Qwen, 125 gens each, phase-tagged (think/response). Data in cache + CSVs
- [x] **Tulu family** — DONE. Same Llama base, quit→sue vs Llama's quit→file. Different procedural register. 2/5 transparent (no-safety-data SFT)
- [x] **Pythia family** — DONE. 4/5 transparent (lightest alignment of any 7B). Only anger repressed (kill→scream). Worker stays "take". HH-RLHF = minimal displacement

## CircuitProfile pipeline
- [x] **Populate profiles** — DONE. 14 families, 43 files, ~90KB. Per-family axes from own base model embeddings

## Temporal chain analysis
- [x] **Embedding trajectories** — DONE. 8 families, 2880 windows, violence + procedural axes. Repression=steady drift, reaction formation=collapse, transparency=flat, de-foreclosure=increasing trajectory
- [x] **Paired mega-gen** — DONE as teacher-forced projections. Displacement concentrated at position 0. Models converge by step 5 on identical tokens. Alignment changes prediction not interpretation

## Tests
- [x] **Circuit integration test (smol)** — DONE. 10 tests on SmolLM2-360M: from_family, compare, self-JS=0, formation, Mode.RAW/CHAT, entropy, top_tokens, classify. Full suite 64/64
- [x] **Circuit unit tests** — DONE. 21 tests: classify_trajectory (13 cases incl BLANK_SENTINEL, NaN, de_foreclosure), signature_summary (2), Mode (2), tokens (2), sentinel (2). Also fixed "nan" string blank detection
- [x] **CacheManager mega_generations roundtrip** — DONE. 4 tests: roundtrip, miss, count (binary search), separate prompts. Full suite 54/54
- [x] **Salary probe SmolLM3-3B** — DONE. Already uses correct SmolLM3-3B (not 360M). In salary_all.csv
- [x] **SmolLM3 mega-gen at 100 tokens** — DONE. 49,560 rows, all 5 prompts × 50 gens × base+aligned
- [x] **F25 classifier on reasoning mega-gen** — DONE but classifier rules don't apply (step 0 is `<think>` token, not content). Reasoning needs phase-boundary analysis (think H vs response H) instead of step-0 argmax rules. Two complementary approaches.

## Analysis / figures
- [x] **F25 cross-family figure** — DONE. 5×5 pie-chart grid. OLMo=foreclosure(anger), Llama=pure repression, Qwen=most diverse, SmolLM3=transparent on anger+love
- [x] **DPO paradox figure** — DONE. RLVR increases reaction formation (violence +11pp) and return of repressed (anger +3pp). Ego-ideal destabilizes
- [x] **Reasoning phase boundary figure** — DONE. R1-Llama content-blind, R1-Qwen content-sensitive, SmolLM3 inverts (broadens at </think>)
- [x] **Salary cross-family figure** — DONE. Gender gap heatmap + profession bars. Alignment compresses base chaos but introduces new gaps (OLMo teacher +40%)
- [x] **Chinese displacement figure** — DONE. EN triggers exam blanks, ZH preserves semantics (離開, 集體). Sexual ZH entropy RISES after alignment (+0.24)

## Memory / documentation
- [x] Update session memory with reasoning battery results (R1-Qwen, SmolLM3-think)
- [x] Update F25 finding with classifier results (prompt-specific signatures)
- [x] Update CLAUDE.md with Circuit class documentation

---

## Svelte UI improvements (2026-06-24)

Playwright installed for headless iteration. Goal: make the UI useful as a book companion / data explorer, not just a live-analysis tool.

### Data explorer tabs (no model server needed)
- [x] **Beam Explorer** — browse beam storylines from cached data. Top-100 beams per prompt, per-token resistance coloring (red=blocked, green=facilitated), annotator selector (SFT/DPO/RLVR), sort by rank/resistance/probability. Component: `BeamExplorer.svelte`.
- [x] **Survival Decay** — SVG line chart of base top-10 survival in aligned top-100 at prefix lengths 1-10. Family and category selectors. Computed from beam cache (17K rows, `data/survival_decay.csv`). Component: `SurvivalDecay.svelte`.
- [x] **Cross-Family Heatmap** — 10×10 Spearman correlation of 10-token survival rates across 71 prompts. Color-coded cells, significance asterisks. Mean |r|=0.126. Data: `data/cross_family_beam_correlation.csv`. Component: `CrossFamilyHeatmap.svelte`.
- [x] **Census Grid** — 13 families × 5 prompts heatmap from `circuit_census_grid_final.csv`. Color by mechanism type or entropy change. Single-prompt list view with token shifts. Legend with 6 mechanism types. Component: `CensusGrid.svelte`.
- [x] **Resistance Trajectories** — per-position resistance (bits) across 10 token positions. All-families view (grouped by family) or single-family view (grouped by category). Data: `data/resistance_trajectories.csv` (17K rows). Component: `ResistanceTrajectories.svelte`.

### Existing tab improvements
- [ ] **Passages tab: add BLT bits/char** — `jakobson.parquet` has 143K passages with BLT scores. Show distribution, human vs AI comparison, quadrant plot.
- [ ] **Passages tab: family filter** — currently loads all passages. Add family dropdown + category filter.
- [x] **Token Shifts tab** — multi-family institutional token displacement from `f21_token_shifts_multi.csv`. Grid view (6 families × 17 tokens, paired bars) and single-family detail view with base/aligned probabilities. Component: `TokenShifts.svelte`.
- [ ] **Formation tab: multi-family** — load `f01_meta_formation.csv` to compare formation across families without server.

### Infrastructure
- [x] **Static data mode** — `malign serve --data-only` starts server without loading models. UI shows "data only" badge, only data tabs (Beams, Passages) visible. API endpoints `/api/beam/*` and `/api/data/csv` serve cached data. Server: `serve(data_only=True)`.
- [x] **Sankey component** — multi-stage displacement flow (base→SFT→DPO→Instruct). Model/prompt/depth selectors. Computed from beam cache via `/api/beam/sankey`. Component: `DisplacementSankey.svelte`.

### Pending beam data
- [x] **Llama integration** — DONE. 30h local run, 497 entries, 7 variants (R1-Distill, Dolphin, Hermes, Tulu). 11 families total in beam cache.

---

## Scale remaining analyses to full census (2026-07-01)

### Cheap (from existing cached data, no model loads)
- [x] **Taxonomy at 43 families** — DONE (2026-07-01). 504 pairs. 64% synonym, 32% register shift, 4% category shift. `data/taxonomy_full_census.csv`.
- [x] **Resistance trajectories at 41 families** — DONE (2026-07-01). 5,542 rows, both directions. `data/resistance_trajectories.csv`.
- [x] **Cross-family resistance at 41 families** — DONE (2026-07-01). 1,670 rows, up to 41 families per pair. `data/cross_family_resistance.csv`.

### Moderate (new forward passes, local MPS)
- [ ] **System prompt effect at 45 families** (currently 3) — 4 conditions (raw/chat-default/chat-safety/chat-permissive) × 5 prompts per model. ~5 hours on MPS.
- [ ] **SFT↔DPO cross-resistance** — beam search on DPO, teacher-force through SFT and vice versa. For 3+ layer families (~20 families). ~8 hours on MPS.
- [x] **Teacher-forced prompt extension** (done in fee76ad) — append target word (e.g. "kill") to prompt, compare next-token distributions between base and aligned. Geography-independent displacement measure. ~10 min per word across all families.

### Data / findings cleanup
- [x] **Mark stale TODOs** — DONE (2026-07-01). Cleaned up top-of-file items.
- [x] **F05 revision** — DONE (2026-07-01). Updated with 40-family contradiction + SFT/DPO depth gradient.
- [ ] **Bidirectional resistance CSV** — save the 28-pair forward/reverse table as a browsable CSV for Data Explorer.
- [ ] **Publication figures** — new findings (bidirectional, category-level, logit lens depth, system prompt) need figures for CI article / book.
- [ ] **Summary table CSV** — one row per family with headline numbers (kill delta, reverse/forward bits, displacement style, architecture, alignment intensity). Master reference for drafting.

### Expensive (need cloud GPU / generation cache)
- [ ] **Surprisal / Shannon entropy at 45 families** (currently ~10) — needs n=100 generations per model per prompt. Cloud vLLM.
- [ ] **BLT byte-level scoring at 45 families** (currently ~10) — needs generations + BLT 1B scoring. Cloud.
- [ ] **Generation-level analysis / MMD at 45 families** (currently ~13) — needs generation cache. Cloud.
- [ ] **Fold dimensionality at 45 families** (currently ~10) — needs generation cache. Cloud.
- [ ] **Cross-generation MMD at 45 families** (currently ~10) — needs generation cache. Cloud.
- [ ] **Step-level checkpoints** (currently 1 family) — OLMo Think-SFT only. Allen AI specific.
