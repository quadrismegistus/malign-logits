# TODO — Malign Logits

Work items from session 2026-06-20/21. Pick up when idle.

## Code fixes
- [ ] OLMo worker "unclassified" — de_foreclosure rule needs fuzzy blank detection (base argmax `?` not `_`)
- [ ] Circuit `Mode.CHAT`/`Mode.THINK` test with live models (wired but untested)
- [ ] `malign circuit` CLI command
- [ ] Merge salary CSVs into one unified dataset (`salary_systematic.csv` + `salary_smol3.csv`)

## Experiments to run
- [ ] **Pythia 6.9B BLT scoring** — score 1,500 cached generations with BLT 1B. Quick (~15 min)
- [ ] **Qwen3-8B native thinking** — HF pipeline (already in LM Studio as GGUF). Mega-gen + formation
- [ ] **Salary probe SmolLM3-3B** — the real 3B (SmolLM2 360M ran by mistake in first batch)
- [ ] **SmolLM3 mega-gen at 100 tokens** — current data is 50 tokens, longer traces for transparent signature
- [ ] **F25 classifier on reasoning mega-gen** — run classify_mega_gen on R1/SmolLM3-think CSVs

## Analysis / figures
- [ ] **F25 cross-family figure** — 5×5 grid (family × prompt) coloured by dominant signature. Book signature image
- [ ] **DPO paradox figure** — deeper foreclosure → more return of repressed (OLMo SFT 10% vs DPO 40% bleed)
- [ ] **Reasoning phase boundary figure** — think H vs response H across R1-Llama/R1-Qwen/SmolLM3
- [ ] **Salary cross-family figure** — gender gap × profession × family heatmap
- [ ] **Chinese displacement figure** — EN vs ZH argmax comparison table

## Memory / documentation
- [ ] Update session memory with reasoning battery results (R1-Qwen, SmolLM3-think)
- [ ] Update F25 finding with classifier results (prompt-specific signatures)
- [ ] Update CLAUDE.md with Circuit class documentation
