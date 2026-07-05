# Repository Audit — malign-logits

**Date:** 2026-07-04
**Scope:** full repository — `malign_logits/` package (35 modules, ~21k lines), `scripts/` (117 files, ~17k lines), `tests/` + CI, `notebooks/` (19), documentation (README, CLAUDE.md, TODO.md, PROBE.md, context.md, findings/, docs/), packaging, and git/repo hygiene.
**Method:** parallel deep-read audits of each area with claims verified against actual code (grep for callers, empirical tokenizer tests, live test run, CI log inspection, git history analysis). Findings below are verified, not speculative.

---

## Executive summary

The codebase is in fundamentally good shape for a fast-moving research project: the cache layer is well designed, full-vocabulary logit discipline holds everywhere, MPS/dtype hygiene is consistent, and model lifecycle management is careful. The test suite is small but high quality where it exists.

The problems cluster in four places:

1. **Five issues can silently distort published numbers** (§1) — the most important category given the July 31 Critical Inquiry deadline.
2. **Three CLI commands are outright broken** and CI has been red for a week on a one-line stale test (§2, §5).
3. **Documentation has drifted badly from the code** — README contains ~1,480 duplicated stale lines, CLAUDE.md documents 8 of 47 registered families and contains example code that raises `KeyError` (§6).
4. **Repo hygiene**: 26 unpushed commits, a malformed `.gitignore`, 726 MB of git history from large data blobs, and 29 log files in the root (§3).

### Top 10 actions (highest value first)

| # | Action | Effort |
|---|--------|--------|
| 1 | `git push` — 26 commits ahead of origin, no backup of a week of paper-critical work | 1 min |
| 2 | Fix `tests/test_circuit.py:155` (`Mode.COMPLETE` → `Mode.CONTINUE`) to restore CI signal | 1 line |
| 3 | Fix the trajectory v2.6 held-out evaluation leak (`trajectory.py:729-797`) — inflates published F12 closure numbers | small |
| 4 | Fix the `score_vocab` cache key collision (`cache.py:379-397`) — cross-family contamination of formation numbers | small |
| 5 | Fix `_load_surprisal_model` name guard (`embedding.py:817-833`) — surprisal scored/cached under wrong reference model | 1 line |
| 6 | Delete README lines 1990–3466 (duplicated stale findings) and rerun `scripts/build_readme.py build`; add section markers to the script | small |
| 7 | Fix the broken `continue` mode branch in `probe.py:141-150` and `deep_probe.py:230-241` — current chapter blocker depends on it | 2 lines |
| 8 | Commit `scripts/deepseek_v3_cloud.py` and `scripts/sft_ab_experiment.py` (with corrected docstring) — results already exist that these produced | small |
| 9 | Repair the malformed `.gitignore` lines (`.vastai.jsondraft/`, `*.bakdata/models/`) | 2 lines |
| 10 | Fix the three broken CLI commands: `step-analysis`, `vllm-generate`, `produce-all` phase 4 (§2) | medium |

---

## 1. Research-integrity issues — can affect published numbers

These deserve priority over everything except backup, because they touch numbers that feed the paper.

### 1.1 F12 "held-out" closure is not held out — `malign_logits/trajectory.py:729-797`
The v2.6 train/eval split is computed (`eval_keys`/`train_keys`, lines 733–737) but evaluation then loops over **all** prompts (`for label, prompt in subset.items()`, line 769), including the training half. The printed "held-out mean closure" figures that back F12 ("held-out closure 77%/20%") mix train and eval prompts and are inflated. The CSV keeps `label`, so re-analysis is possible without re-running models.
**Fix:** filter `v26_rows` to `eval_keys` (or add a `split` column and report both), then re-derive the F12 headline numbers.

### 1.2 `score_vocab` cache silently collides across vocabularies — `malign_logits/cache.py:379-397`, `malign_logits/psyche.py:233-245`
`ModelLayer.score_vocabulary` builds a key containing the word tuple, but `CacheManager.set_derived`/`get_derived` for `type == "score_vocab"` drop the words and store under `{model, prompt}` only. Two different vocabularies for the same (model, prompt) overwrite each other; words missing from the stale cached value read as probability 0 in `formation_df`. This is **live**: `llama` and `tulu` (and all `tulu-sft-*` ablation families) share base `meta-llama/Llama-3.1-8B` but have different `_focused_vocabulary` unions — the second family analyzed gets the first family's scores.
**Fix:** include the vocabulary (or its hash) in the stored key, or store full-vocab scores and subset on read. Then audit which cached formation results were computed after a colliding family.

### 1.3 Surprisal reference model ignored after first load — `malign_logits/embedding.py:817-833`
`_load_surprisal_model` guards only on `if _surprisal_model is None:` (contrast `_get_embedder`, which correctly compares names). Any process touching two reference models (e.g. `compute_passage_metrics(ref_model_name="gpt2")` after Pythia loaded, or the server's `/passage-tokens` fallback) silently scores under the wrong model **and caches results under the wrong ref name**. `cmd_surprisal` manually resets the globals (`cli.py:655-657`) — a workaround that proves the bug.
**Fix:** `if _surprisal_model is None or _loaded_name != model_name:`.

### 1.4 Beam-width divergence undocumented — DeepSeek 200 vs everyone else 1000
`beam_word_probs` runs with `n_beams=1000` in `malign_logits/core.py:246` and `scripts/olmoe_cloud.py`, but `scripts/deepseek_v3_cloud.py` uses `n_beams=200` (plausibly a deliberate cost decision for a 671B model). Beam-word distributions at different widths are not strictly comparable.
**Fix:** document the width difference wherever DeepSeek beam words are compared against 1000-beam families, or standardize. Relatedly, the `length_penalty=0.0` audit fix was applied to `core.beam_word_probs` but **not** to `beam_storylines`/`batch_beam_annotate` (`malign_logits/beam.py:55-63, 385-390`) — with HF's default `length_penalty=1.0`, `path_prob = exp(sequences_scores)` is a length-normalized geometric mean, not a path probability.

### 1.5 Taxonomy `--baseline` recomputes syntagmatic JS on truncated prompts — `malign_logits/taxonomy.py:271-276`
`run_taxonomy` stores `prompt[:60]` in the CSV (line 157) but computes `syntagmatic_js` with the full prompt; `add_aligned_baseline` re-reads the truncated `row["prompt"]`. For prompts >60 chars (most institutional prompts under `--all-prompts`), base-vs-aligned deltas compare different contexts.
**Fix:** store the full prompt in a separate column, or map label→prompt via `DEFAULT_PROMPTS`.

### Related provenance issues
- **`scripts/sft_ab_experiment.py` (modified, uncommitted):** the working-tree diff (None-source bug fix, 100K subsample, 1 epoch, effective batch 32) is the version that produced the untracked `data/logits_sft_*` results, but the docstring still claims "Hyperparameters match Tulu 3 SFT recipe." Commit with the docstring corrected, or the A/B logits' provenance is misdocumented.
- **`scripts/deepseek_v3_cloud.py` (untracked):** results-bearing cloud script (DeepSeek-V3.1 base + instruct, 73-prompt battery, logits + 200-beam words). Unreproducible without it. Commit (after fixing §7.2's cache-wipe footgun).
- **`mode_decomposition` duplicate dict key — `malign_logits/metrics.py:1451-1456`:** `mode_labels` defines `("raw","chat")` twice ("special tokens" silently overwritten by "total template"), and `("chat","think")` never matches. Mislabeled components in mode-decomposition output.

---

## 2. Broken functionality (critical)

1. **`malign step-analysis` — `NameError` — `malign_logits/step_analysis.py:123-127`.** Phase 2's cache check references an undefined `stash` variable with a stale tuple-key format (pre-CacheManager leftover). Broken since the cache migration; `cmd_step_analysis` only catches `ValueError`.
   Fix: `all(cache.has_logits(f"{model_id}_step", p) for p in prompts.values())`.

2. **`malign vllm-generate` never generates — `malign_logits/cli.py:400-419`.** With `--families`, the loop only validates names and returns; without it, the `else` branch references an undefined `fam` → `NameError`. The CLI port dropped the `for fam in families:` loop that `scripts/vllm_generate.py:184-187` has.
   Fix: restore the loop (and pass `tp` like the script does).

3. **`malign produce-all` phase 4 always fails silently — `malign_logits/produce.py:183-185`.** `embed_generations()`, `compute_generation_metrics()`, `compute_concept_metrics()` are called with no arguments but require `(psg_df, ...)` / `(embeds_df, psg_df)` (`embedding.py:204, 296, 388`). The `TypeError` is swallowed and recorded as `"error: ..."` — every run silently skips embedding metrics.
   Fix: load cached generations, group, call with proper args — or delete the phase.

4. **`continue` mode broken in Probe and DeepDive — `malign_logits/probe.py:141-150`, `malign_logits/deep_probe.py:230-241`.** Branch order is `if mode in ("chat", "think"): ... elif mode == "chat": ...` — the second branch (the `role: assistant` + `continue_final_message` variant) is unreachable, and `mode="continue"` falls through to `raise ValueError`. The `elif` was clearly meant to be `mode == "continue"`. Continue-mode beams are a current chapter blocker, so this is load-bearing.

5. **Probe tree cache: three incompatible key schemas — `malign_logits/probe.py:1026-1029, 1369-1372, 1400-1402, 1653-1655`, `malign_logits/cli.py:195-197`.** `explore_tree`/`annotate_tree` use a `{coverage, branch_depth, ...}` key; `batch_annotate` uses `{path_threshold, ...}`; `malign probe status` checks a third (`explore_tree_v2`), so status always reports 0/N done. Concrete cost: `malign probe ingest` can't see `batch_annotate`'s results, reloads every annotator model, and **re-annotates from scratch** (hours of GPU redone).
   Fix: one canonical key-builder function used by all four sites.

6. **Server `/prompts` endpoint permanently broken — `malign_logits/server.py:648-661`.** `psyche._stash` doesn't exist (it's `_cache`); the `AttributeError` is swallowed and `{"prompts": []}` always returned. Even fixed, it scans legacy tuple keys. `app.py:459` depends on it.

---

## 3. Repository & git hygiene

### 3.1 26 unpushed commits (HIGH)
`main` is ahead of `origin/main` by 26 commits — a week of paper-critical work (census, article evidence, audit fixes) with no off-machine backup. **Push now.**

### 3.2 Malformed `.gitignore` (HIGH)
Two lines are corrupted concatenations of what were clearly meant to be separate entries:

```
.vastai.jsondraft/      ← meant: ".vastai.json" + "draft/"
*.bakdata/models/       ← meant: "*.bak" + "data/models/"
```

Verified with `git check-ignore`: `.vastai.json`, `*.bak`, and `data/models/` are **not ignored**. Consequences visible in `git status` right now: `data/corpus_metrics.csv.bak`, `data/corpus_metrics.parquet.bak`, and 5.5 GB `data/models/` show as untracked. If `.vastai.json` (vast.ai credentials) is ever created in the root, it is one `git add -A` away from being committed.
**Fix:** split into four lines. Also note: `notes/` is ignored but `notes/stash_redesign.md` is tracked (added before the ignore rule); `context.md` is gitignored yet CLAUDE.md's architecture section advertises it as a project file — anyone cloning the repo won't have it.

### 3.3 Git history bloat: 726 MB `.git`, no LFS (MEDIUM)
Largest blobs in history: `data/corpus_metrics.csv` (48 MB), `data/gen_battery_raw.parquet` (46 MB), `data/logit_lens_datadriven.csv` (41 MB), `data/mega_gen_olmo_4layer.csv` (26 MB), `data/corpus_metrics.parquet` (25 MB), plus 9–13 MB notebooks. 1,683 tracked files under `data/`; 620 tracked figures (348 MB on disk). Both CSV **and** parquet of `corpus_metrics` are tracked (identical data, +73 MB combined; the `.bak` copies are byte-identical to the current files and can be deleted).
**Fix (choose deliberately):** (a) accept the size but stop the growth — track only parquet for large tables, clear notebook outputs before committing (the five biggest notebooks carry ~42 MB of embedded plots); or (b) move `data/*.{csv,parquet}` >5 MB to git-LFS. History rewriting is probably not worth the disruption mid-paper.

### 3.4 Root directory clutter (LOW)
29 `*.log` files plus `overnight_done.flag` in the repo root. A `logs/` directory exists and is already gitignored. Move logs there and have future scripts write to `logs/` directly; delete the flag file once its run is confirmed done.

### 3.5 `ui_dist` build churn (LOW)
The working tree has the old Svelte build deleted and a new build untracked (8 old files `D`, 8 new files `??`, plus modified `version.json`/`index.html`). Commit the rebuilt output or decide to stop tracking builds — but note built assets must ship somehow since `server.py:26` serves from the package dir (see §4.2).

### 3.6 Duplicate virtualenvs (LOW)
Both `venv/` and `.venv/` exist at the root. `uv run` uses `.venv`; the stray `venv/` wastes disk and confuses tooling. Remove one.

### Positive
No secrets in tracked files (scanned for API-key patterns across the repo; `DEEPSEEK_API_KEY` etc. are read from env correctly). No hardcoded `/Users/rj416` paths in the package (only in two notebooks, §7.5).

---

## 4. Packaging & distribution

1. **Wrong GitHub URLs — `pyproject.toml`.** Homepage/Repository/Bug-Reports all point to `github.com/rj416/malign-logits`; the actual remote is `github.com/quadrismegistus/malign-logits`.
2. **`ui_dist` not packaged.** `[tool.setuptools.package-data]` includes only `*.py` (pointless — .py ships by default) and `MANIFEST.in` doesn't mention `ui_dist`, so a built wheel omits the UI that `malign serve` serves from `malign_logits/ui_dist/`. Add `"malign_logits" = ["ui_dist/**/*"]`.
3. **License contradiction (three-way).** `LICENSE` is GPL v3, `pyproject.toml:10` says MIT, README's footer says "GNUv3". Pick one before public release alongside the CI paper — likely `license = {text = "GPL-3.0"}` in pyproject and "GPL-3.0" in README.
4. **Dev deps undeclared.** `pytest-timeout` is used locally but absent from `requirements.txt`/`[dev]` extras; `pytest-cov` isn't installed at all. Add both to `[project.optional-dependencies].dev`.
5. **`pytest` has no `testpaths`.** A bare `pytest` from the root collects `scripts/test_circuit_modes.py`, which **loads SmolLM3-3B on MPS at import time**. Add `testpaths = ["tests"]` to `[tool.pytest.ini_options]` and/or rename the script `demo_circuit_modes.py`.

---

## 5. Tests & CI

Local run (2026-07-04): **70 passed / 1 failed** (~20 s including slow tests).

1. **CI red on every push since ≥ June 28 (CRITICAL).** `.github/workflows/test.yml` is correctly configured and genuinely runs the suite (~2.5 min), but the last 8+ runs on `main` all failed on one stale assertion: `tests/test_circuit.py:155` asserts `Mode.COMPLETE.value == "complete"`, but commit `b6a3e8e` renamed it to `Mode.CONTINUE` in `circuit.py:34`. A week of pushes (45-family census, article evidence) landed with no working regression gate.
   **Fix:** `assert Mode.CONTINUE.value == "continue"`. One line, restores all CI signal.

2. **~26 of 35 modules have zero test coverage (HIGH),** including the project's self-declared core: none of `core.py`'s six functions (`discover_top_words`, `score_words_from_logits`, `beam_word_probs`, …) is imported by any test — `tests/test_core.py` is misnamed (it tests the registry and embedding helpers). In `analysis.py`, only 2 of 13 functions are tested; the displacement engine behind the paper's confirmed findings (`compute_displacement`, `compute_repression`, `distribution_metrics`, `top_movers`) has none. Other zero-coverage heavyweights: `probe.py` (2,003 lines), `metrics.py` (1,509), `cli.py` (1,448, 65 subcommand sites), `taxonomy.py`, `beam.py`, `trajectory.py`, `server.py`.
   **Priority order:** pure-computation modules that back published findings and need no model downloads — `analysis.py`, `metrics.py`, `taxonomy.py` classification, `beam.py` — plus a `cli.py --help` smoke test. Note a ready-made `_mock_model_and_tokenizer` fixture already exists in `test_core.py:43-69` and is **never used by any test**.

3. **`test_circuit_integration.py` downloads ~1.4 GB of models in CI, unmarked (MEDIUM).** Not marked `slow`, so `-m "not slow"` still runs it; meanwhile the two `slow`-marked tests do the same kind of load — marking is inconsistent. It's valuable CI signal, so keep it, but add `actions/cache` for `~/.cache/huggingface` (and pip) so every push doesn't re-download.

4. **A few tests assert existence, not behavior (MEDIUM).** `test_compute_passage_metrics_uses_cache` asserts only `callable(...)`; `test_psyche_cache` asserts `hasattr` only. Replace or delete — they provide false confidence. (Counterpoint: most of the suite is genuinely good — cache roundtrips, all 8 F25 classifier branches, real JS invariants.)

5. **Brittle magic-count assertions (MEDIUM).** `len(DEFAULT_PROMPTS) >= 47`, `>= 10` families, etc. Assert structure, not counts, or pin counts with a comment.

6. **No `conftest.py` (LOW).** The `skipif`-on-`CACHE_ROOT` guards are duplicated per test; fresh-clone safety is fine (verified), but the only tests of the Psyche-cache wiring never run in CI. A tiny committed fixture lmdb would let one cache-read path run everywhere.

7. **Misc (LOW):** CI Python 3.12 vs local 3.11 (a small matrix would catch skew); `tests/.DS_Store` and `tests/__pycache__/` present; no coverage tooling.

---

## 6. Documentation

1. **README.md corrupted: ~1,480 duplicated stale lines (CRITICAL) — lines 1990–3466.** The full findings section appears twice; the second copy is an older draft of F21–F34. Root cause verified in `scripts/build_readme.py`: `build()` locates the end of the findings section by the first `\n## ` after `## Findings`; old finding files had internal `##` headings, so the boundary landed inside the block and the stale tail was preserved and re-prepended. **Re-running build will not fix it** — the boundary still lands at line 1990.
   **Fix:** delete lines 1990–3466 manually, rerun `python scripts/build_readme.py build` (safe — no current findings file contains `^## `), and harden the script with explicit `<!-- findings:start/end -->` markers.

2. **README missing F35, stale for F05/F31 (HIGH).** `findings/F35_architecture_independence.md` (Jun 28 — a headline anti-Weatherby result for the CI paper) postdates the last README build (Jun 27); F05's Jul 1 revision and F31's update likewise. The same rebuild fixes all three. Also add `35: "architecture_independence"` to `SLUGS` in `build_readme.py` (gaps: 26, 29, 30, 35).

3. **CLAUDE.md family table: 8 documented, 47 registered (HIGH).** Code registers 47 families / 109 checkpoints; CLAUDE.md lists 8 and still describes Zephyr as *unregistered* (with a different SFT checkpoint than the code uses: `alignment-handbook/zephyr-7b-sft-full` vs actual `HuggingFaceH4/mistral-7b-sft-beta`). Replace the table with a count plus a pointer to `malign info` / `docs/model_candidates.md`.

4. **CLAUDE.md example code raises `KeyError` (HIGH).** `Psyche.from_family("olmo-3-7b")`, `from_family("llama-3.1-8b")`, `malign serve --family llama-3.1-8b`, "default family (olmo-3-7b)" — `from_family` is a plain `MODEL_FAMILIES[...]` lookup (`psyche.py:1447`); correct keys are `olmo`, `llama` (used correctly elsewhere in the same file).

5. **CLAUDE.md architecture tree omits 16 of 36 modules (HIGH)** — including `registry.py`, `probe.py`, `metrics.py`, `beam.py`, `cache.py` (in prose but not tree), `trajectory.py`, `profile.py`, `cloud.py`, and all graph/vec modules. PROBE.md (the best-maintained technical doc) is not referenced from CLAUDE.md at all. Also stale: "app.py — Gradio web UI" and the "Terminal 2: Gradio UI" dev workflow — `malign ui` now opens the Svelte data explorer served by `malign serve` (`--data-only` mode undocumented); `app.py` is legacy (candidate for deletion given the no-backward-compat preference). And the F13 metric line ("4 families, r = −0.34 to −0.50") is stale — F13 says 6 families, −0.338 to −0.533.

6. **context.md describes an obsolete architecture as current (HIGH).** Its Technical Architecture section still has 4-bit bitsandbytes Llama and superego-as-prompt-prefix ("Ego and Superego hold a reference to the *same* model object") — both directly contradicted by current doctrine ("Each layer is a separate model checkpoint — not a prompting trick"). Add a prominent "Historical document (March 2026), superseded" banner or split theory from stale mechanics.

7. **Family counts contradict across six documents (MEDIUM):** CLAUDE.md 8, README 9, `docs/model_inventory.md` 17, `docs/model_candidates.md` 20, TODO.md 45, PROBE.md 17/59 — code says 47/109. Make code the source of truth; date-stamp or generate the rest. Related: `data/model_registry.json` (`REGISTRY_PATH`) doesn't exist — the persisted registry silently re-bootstraps (see also §8.7).

8. **F26 is cited but has no file (MEDIUM).** `findings/F33_scale_effects.md:93` cites "The F26 2:1 SFT>DPO ratio" but no `F26_*.md` exists — the census result lives only in session memory. Write `findings/F26_census.md` (it's load-bearing for F33). F29/F30 are silent gaps; note or reserve them.

9. **CLI: 10 of 24 subcommands undocumented (MEDIUM)** — `probe` (8 subcommands), `deep-probe`, `trajectory`, `circuit`, `api-generate`, `topic-drift`, `download-models`, `ablation`, `produce-all`, `serve --data-only`. Add a generated CLI table to CLAUDE.md.

10. **README footer stale (MEDIUM):** "Available families" lists 9 of 47; architecture tree lists 9 modules; install says `pip install -e .` where CLAUDE.md mandates uv.

11. **TODO.md is 90% a done-log (LOW).** One stale unchecked item: "Teacher-forced prompt extension" was completed in commit `fee76ad`. Move done sections to an archive so the ~10 genuinely open items are visible before the July 31 deadline.

12. **Missing meta-docs (LOW):** no `data/README.md` mapping the 100+ CSV/parquet files → producing script → finding (will pay off at paper-revision time); `docs/pipeline.md` and `docs/data_pipeline.md` are good and current but linked from nowhere; PROBE.md status line ("Current: collecting…") describes a campaign finished Jun 30.

---

## 7. Scripts & notebooks

1. **Commit the two dirty scripts (HIGH).** `deepseek_v3_cloud.py` (untracked, results-bearing) and `sft_ab_experiment.py` (modified; the diff is the version that produced `data/logits_sft_*`) — see §1 for the provenance angle. No other untracked or orphaned scripts; all 117 scripts' imports resolve against the current package (AST-verified).

2. **HF-cache wipe footgun (MEDIUM) — `deepseek_v3_cloud.py:184-189`, `scale_test_full.py:203-206`.** Both `shutil.rmtree` every `models--*` under `~/.cache/huggingface/hub` after each model — correct on a throwaway vast.ai box; run accidentally on the Mac Studio it deletes every locally cached model (hundreds of GB including all OLMo checkpoints). Guard with `if os.path.exists('/workspace')` or an explicit `--wipe-hf-cache` flag.

3. **73-prompt battery dict copy-pasted into 6 scripts (MEDIUM)** — `cloud_r1_generations.py`, `deepseek_v3_cloud.py`, `olmoe_cloud.py`, `sft_ab_experiment.py`, `scale_test_full.py`, `vllm_reasoning.py`. Canonical prompts live in `experiments.py`; one edited prompt would silently desynchronize cross-scale comparisons. Add a `--dump-prompts` generator or at minimum a header comment naming the source of truth.

4. **Superseded scripts to archive/delete (MEDIUM):** `scale_test_32b.py` / `scale_test_70b.py` (→ `scale_test_full.py`), `mega_generation.py` (→ `mega_generation_scaled.py` + `Circuit.mega_generate`), `bos_smoke_test.py` (→ `malign bos-generate`), plus the one-off smoke/inspect scripts (`smoke_institutional.py`, `smoke_political.py`, `r1_smoke_test.py`, `inspect_*.py`, `contradiction_ugly.py`). An `scripts/archive/` sweep would cut noise substantially. (Keep `vllm_generate.py` — the CLI loads it by path.)

5. **Notebook bloat and duplication (MEDIUM).** ~52 MB of tracked notebooks, ~42 MB of it in five files of embedded plot outputs. `10_examples.ipynb` and `09_quadrants.ipynb` are near-duplicates (byte-identical Dropbox `fig.save` lines) with `10_examples2.ipynb` as the apparent trimmed successor — pick one canonical quadrants notebook. Personal Dropbox paths in `09_quadrants`, `10_examples`, `F20_noise_talk` break re-execution for anyone else; fall back to `figures/`.

6. **Naming split (LOW):** numbered `01–10_*.ipynb` coexist with `F##_*.ipynb` (`10_shannon` = F18, `06_step_analysis` = F04). A mapping note in `findings/` would help navigation.

---

## 8. Package code quality (high/medium not covered above)

1. **Path traversal in the model server (HIGH, security) — `malign_logits/server.py:490-504`.** `_serve_static` does `_UI_DIR / path.lstrip("/")` with no containment check; `GET /../../../../etc/passwd` serves files outside `ui_dist`. The server binds `0.0.0.0` by default (line 955), so this is LAN-exposed.
   **Fix:** resolve and require `resolved.is_relative_to(_UI_DIR.resolve())`; consider defaulting to `127.0.0.1`.

2. **Tokenizer round-trip check misfires for every BOS-prepending tokenizer (MEDIUM) — `malign_logits/models.py:12-17`.** `tok.decode(tok.encode("a b")) != "a b"` is true for Llama and Amber (BOS gets prepended), so roughly half the registry always takes the slow-tokenizer fallback intended only for DeepSeek — a redundant second tokenizer load per model, and a behavior risk for `trust_remote_code` slow-only tokenizers (Baichuan). Verified empirically. Fix: `add_special_tokens=False` in the probe.

3. **Three inconsistent mode-caching conventions (MEDIUM).** `CacheManager.get_logits(..., mode=)` adds a `mode` field (`cache.py:59-76`); `CircuitNode.logits` prefixes the prompt (`circuit.py:85-96`); `Probe.collect` encodes mode in the model id (`probe.py:128`). The same conceptual entry lands under three different keys depending on code path — a future cache-reuse trap. Standardize on the CacheManager convention.

4. **`Psyche.from_cache(cache_dir=...)` is documented but ignored (MEDIUM) — `psyche.py:1460-1484`.** `_psyche_cache()` always returns the default-root CacheManager. Wire it through or drop the parameter.

5. **Silent-failure patterns (MEDIUM):** `core._apply_mode` (`core.py:235-243`) swallows all exceptions and silently degrades chat/think to raw; `psyche._get_derived`/`_set_derived` (`psyche.py:89-102`) swallow everything, so a corrupt lmdb degrades to recompute-and-drop; `warnings.filterwarnings("ignore")` at package import (`__init__.py:42-43`) suppresses all warnings for every consumer, including the package's own deprecation warnings.

6. **Other confirmed bugs (MEDIUM):**
   - `server.py:930-933` — `do_OPTIONS` sends CORS headers before `send_response(204)` → malformed preflight.
   - `embedding.py:853-856` vs `940` — batched (CUDA) and sequential (MPS) surprisal join prompt+text differently → device-dependent surprisal for the same passage.
   - `embedding.py:1060-1066` — `run_generate_battery` cache check omits the RLVR layer; 4-layer families with a partially-cached RLVR layer crash with `layer_obj.model = None`.
   - `generation.py:22` — `bos_token_id or eos_token_id` picks EOS when BOS id is 0 (falsy). Use `is None`.
   - `cloud.py:182` — `eval()` on vastai CLI output (use `ast.literal_eval`); `status` can be unbound at `cloud.py:215`.
   - `api_generate.py:278-295` — per-generation failures leave `idx` holes, breaking `count_generations`' contiguity assumption (undercount + index reuse).
   - `registry.py:243-250` — once `data/model_registry.json` exists it is never refreshed from `MODEL_FAMILIES`; new families are invisible to probe/beam/graph pipelines until the JSON is deleted. `NICKNAMES` maps two Yi checkpoints to `"yi"` (ambiguous reverse lookup).
   - `probe.py:1235, 1516` — annotator context rebuilt by whitespace-joining token strings, mis-reconstructing subword/punctuation sequences even when exact `token_id`s are stored on the nodes.

7. **Low-severity (see also dead code, §9):** `viz.py:733-786` silently drops death/power/profanity/substance prompts from aggregate battery figures; `cli.py:1203-1205` `--output` help names the wrong default; `trajectory.py` writes `fold_rank_{family}.csv` twice and duplicates `cli._run_trajectory_one`; `cli.py:211-223` `probe merge` computes the "new keys" list then merges everything anyway; `metrics.py:933` bare `except:`; `cache.py` docstring claims SHA256 key hashing that `normalize_text` doesn't do; `get_cache(root)` returns the old singleton when later called with `root=None`; `graphdb.py:43-45` hardcoded `root`/`malign` ArangoDB credentials; `viz_sankey.py:216` hardcoded Dropbox auto-copy inside a library function; `training_graph.py` (pairtree) and `displacement_graph.py` (lmdb) open raw HashStash on the same root with different engines — contravening the project's "always CacheManager" rule.

---

## 9. Dead code & duplication (verified by grep across package, scripts, notebooks)

**Whole modules with no callers:** `vocab.py`, `vecdb.py` (only caller is itself dead), `training_graph.py`, `displacement_graph.py`, `viz_sankey.py`. Given the no-backward-compat preference, delete them (they're in git history if needed).

**Dead and broken:** `experiments.print_repression_report` (KeyErrors on renamed columns; only referenced from an archived notebook), `Probe.inventory` (passes prompt names where text is expected → always empty), `Probe.tree_to_vecdb` (reads node fields `explore_tree` never sets).

**Dead functions:** `analysis.measure_overdetermination`, `models.load_models`, `models.load_four_models`, `generation.model_generate`, `viz.plot_sublimation`, `embedding.compute_topic_drift`, `embedding.run_surprisal`, `embedding._cache_sent_embeddings`, `embedding._cache_token_surprisals`, `embedding.PATH_GEN_STASH`/`_gen_stash_path`, `graphdb.ingest_hidden_vectors`/`search_hidden`, `experiments.run_prompt_battery`/`summarize_battery`.

**Major duplication:**
- `annotate_tree` vs `batch_annotate` (~150 copy-pasted lines, `probe.py:1215-1372` vs `1454-1648`) — and only the batch copy has KV caching, while `graphdb.ingest_trees` calls the slow one (also a performance issue, §11).
- JS/KL/entropy/top-k implemented three times with subtly different alignment semantics (`analysis.py:332-335` truncate-to-min vs `metrics.py:48-59` pad-with−1e10 vs inline `_js` in `psyche.py:1798`) — cross-vocab comparisons can differ slightly by pipeline. Consolidate on one implementation.
- Canonical 5-prompt dict defined three times (`probe.PROMPTS`, `deep_probe.PROMPTS`, `profile.PROMPTS` — the last with **different prompts under the same keys**).
- Truncated-nickname resolution implemented 4× (`server.py:244-256`, `server.py:440-457`, `Registry.resolve`, `graphdb._annotation_prefixes`).
- Beam entropy extraction duplicated (`beam.py:66-99` vs `392-420`); `cmd_surprisal` vs `cmd_embed` corpus scanning (~40 lines).

---

## 10. Architecture observations (longer-term)

- **Star-import chain with no `__all__`:** `__init__.py` does `from .core import *` … `from .psyche import *`, and most modules start `from . import *`. Every module's namespace contains every other's names plus `torch`/`pd`/`os`; import order is load-bearing (the `# Order matters` comment). Fine for a solo research repo, but it makes dead-code detection, refactors, and IDE tooling much harder. Incremental fix: add `__all__` per module, convert new modules to explicit imports.
- **Four competing model-naming systems:** `MODEL_FAMILIES` keys ("olmo"), `Probe.FAMILIES` ("olmo3-7b"), `NICKNAMES` ("olmo"), raw HF ids — plus truncated/sanitized variants in beam data. A single resolve function (extending `Registry.resolve`) used everywhere would eliminate a whole class of lookup bugs.
- **Three parallel family abstractions:** `Psyche`, `Circuit`, and `Probe`/`DeepDive` each model "several checkpoints of one family" with overlapping but incompatible caching (§8.3). Worth a consolidation pass after the paper deadline, not before.
- **`server.py` `ModelHandler`** mixes static serving, inference, cache reads, tokenizer loading, and nickname munging in one ~850-line handler; `/api/beam/sankey` alone is ~140 lines inside a request handler. `__init__.py` carries 320 lines of `MODEL_FAMILIES` data that belongs in a `families.py`.

---

## 11. Performance opportunities

- `graphdb.ingest_trees` → `annotate_tree`: one full-context forward pass per node per annotator with **no KV cache**, while the KV-cached `batch_annotate` sits next to it. Routing ingest through the fast path saves hours per ingest run.
- `cache.get_score_vocab(words=None)` (`cache.py:390-392`) does a full lmdb keyspace scan on every fallback miss; the server's `/api/beam/sankey` triggers it per model, and also re-instantiates `AutoTokenizer.from_pretrained` per model per request (`server.py:368, 433`). Cache tokenizers in the handler.
- `Registry()` re-parses the full JSON on every instantiation and is constructed per-request in several endpoints.
- `deep_probe._store_embeddings` decodes 100k+ tokens one at a time inside try/except; `convert_ids_to_tokens`/`batch_decode` is orders faster.
- The tokenizer fallback misfire (§8.2) loads every BOS-prepending family's tokenizer twice.

---

## 12. Features & improvements worth adding

1. **`data/README.md` provenance map** — file → producing script/CLI → finding, for the 100+ tracked data files. Highest payoff at paper-revision time when a reviewer asks where a number came from.
2. **`findings/F26_census.md`** — the 42-model census result exists only in session memory but is cited by F33.
3. **Unit tests for the displacement math** (`compute_displacement`, `compute_repression`, `score_words_from_logits`) on synthetic logits with known expected outputs — the machinery underwriting the paper's empirical claims. The unused mock fixture in `test_core.py` is ready for this.
4. **CI hardening:** HF + pip caching, `pytest-timeout`, coverage report, a `build_readme.py build` no-op check (catches stale README at PR time), 3.11+3.12 matrix.
5. **`malign doctor`** — a fast self-check command: cache reachable, registry JSON in sync with `MODEL_FAMILIES`, counts of cached logits/generations per family, stale-README check. Half its value is documentation-by-executable.
6. **Prompt source-of-truth dump** (`malign info --dump-prompts` or a codegen helper) so cloud scripts embed generated, not hand-copied, battery dicts.
7. **Section markers in build_readme.py** (`<!-- findings:start/end -->`) — prevents the §6.1 corruption class permanently.
8. **`scripts/archive/` convention** — sweep superseded one-offs there instead of deleting, keeping `scripts/` navigable.

---

## 13. What's done well (do not "fix")

- **CacheManager** (`cache.py`): typed accessors, dict keys, absolute root path, legacy adapter with forward-routing, binary-search generation counting — exactly what CLAUDE.md prescribes.
- **Full-vocabulary discipline**: logits stored and compared full-vocab everywhere; top-k appears only in presentation layers.
- **MPS/dtype hygiene**: `.float()` upcasts before every softmax/log/entropy; explicit bfloat16→float32 MPS workaround; no CUDA/bitsandbytes assumptions.
- **Model lifecycle**: consistent `del model; gc.collect(); torch.mps.empty_cache()` after sequential loads across probe/beam/battery/produce/taxonomy.
- **Clean topology contracts**: `Psyche`'s `ValueError`s for 2-layer families match the documented contract; `_tokenize_for_generation`'s edge handling; `api_generate`'s exponential backoff; `server._sanitize`'s NaN/Inf→None.
- **Tests that exist are mostly real tests**: cache roundtrips assert stored==retrieved; the F25 classifier suite exercises all 8 signature branches; JS invariants are genuine.
- **Secrets hygiene**: all API keys via env vars; nothing sensitive tracked.
- **`docs/pipeline.md` / `docs/data_pipeline.md` and PROBE.md** are accurate, current, and well written — the model for what the other docs should become.

---

## Appendix: consolidated checklist

**Do today (minutes each):**
- [ ] `git push` (26 commits unbacked) — awaiting sign-off
- [x] Fix `tests/test_circuit.py:155` → CI green *(2026-07-05; suite 69/69)*
- [x] Repair `.gitignore` lines *(2026-07-05)*
- [ ] Commit `scripts/deepseek_v3_cloud.py` + `scripts/sft_ab_experiment.py` — docstring corrected, awaiting commit sign-off
- [x] Fix `embedding.py:817` surprisal model-name guard *(2026-07-05)*
- [x] Fix `probe.py` / `deep_probe.py` continue-mode branch *(2026-07-05)*
- [x] Delete `data/*.bak`; move root `*.log` files into `logs/` *(2026-07-05)*

**Do this week (paper-relevant):**
- [ ] Fix trajectory v2.6 held-out leak; re-derive F12 numbers
- [ ] Fix `score_vocab` cache key collision; audit affected formation results
- [x] README: delete duplicated lines, rebuild (F35/F05/F31 now included), `findings:end` marker added to `build_readme.py`; rebuild verified idempotent *(2026-07-05)*
- [ ] Write `findings/F26_census.md`
- [ ] Fix taxonomy truncated-prompt baseline; document/standardize DeepSeek beam width; extend `length_penalty=0.0` fix to `beam.py`
- [ ] Fix `mode_decomposition` duplicate key (`metrics.py:1451`)
- [x] Fix the three broken CLI commands (`step-analysis`, `vllm-generate` incl. new `--tp` flag, `produce-all` phase 4) *(2026-07-05)*
- [x] Unify probe tree cache keys — `probe.tree_key()` used by explore/annotate/batch/status, legacy-schema migrate-on-read *(2026-07-05)*
- [ ] Guard the HF-cache wipe in cloud scripts

**Do before publication:**
- [ ] Resolve GPL/MIT license contradiction; fix pyproject GitHub URLs
- [ ] CLAUDE.md refresh: family keys, 47-family reality, architecture tree, Gradio→Svelte, F13 numbers
- [ ] context.md historical banner
- [x] Server: path-traversal fix + default to 127.0.0.1 (`malign serve --host 0.0.0.0` for LAN) *(2026-07-05)*
- [ ] Unit tests for displacement math; `testpaths = ["tests"]`
- [ ] `data/README.md` provenance map

**When there's slack:**
- [ ] Delete dead modules (`vocab.py`, `vecdb.py`, `training_graph.py`, `displacement_graph.py`, `viz_sankey.py`) and dead functions (§9)
- [ ] Consolidate JS/entropy implementations; unify model-naming via `Registry.resolve`
- [ ] Archive superseded scripts; dedupe quadrants notebooks; clear outputs on the 5 largest notebooks
- [ ] `malign doctor`; CI caching/coverage; remove stray `venv/`
