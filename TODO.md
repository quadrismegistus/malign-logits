# TODO — Malign Logits

Work items from session 2026-06-20/21. Pick up when idle.

## Code fixes
- [x] OLMo worker "unclassified" — DONE. Base top1 was NaN (newline). Added BLANK_SENTINEL handling. Now correctly classified as de_foreclosure (100%)
- [ ] Circuit `Mode.CHAT`/`Mode.THINK` test with live models (wired but untested)
- [x] `malign circuit` CLI command — DONE. `malign circuit [--family X] [--save]` classifies F25 signatures from mega-gen CSVs
- [x] Merge salary CSVs — DONE. `salary_all.csv` already has all 5 families (6000 rows)

## Experiments to run
- [x] **Pythia 6.9B BLT scoring** — DONE. Confirms 1B finding: text plateaus at step 5000 at both scales (1B: 1.60 bpc, 6.9B: 1.50 bpc)
- [ ] **Qwen3-8B native thinking** — HF pipeline (already in LM Studio as GGUF). Mega-gen + formation
- [x] **Salary probe SmolLM3-3B** — DONE. Already uses correct SmolLM3-3B (not 360M). In salary_all.csv
- [ ] **SmolLM3 mega-gen at 100 tokens** — current data is 50 tokens, longer traces for transparent signature
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
