# TODO — Malign Logits

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
- [ ] **Tulu family** — needs gated Llama base
- [x] **Pythia family** — DONE. 4/5 transparent (lightest alignment of any 7B). Only anger repressed (kill→scream). Worker stays "take". HH-RLHF = minimal displacement

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
