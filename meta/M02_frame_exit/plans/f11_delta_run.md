# Plan: the F11 delta run — the 84 prompts the first fleet missed

Written under RH's [5148] standard: a plan document, not a registration. No freeze, no hash ceremony. The hashes below are there so the pen can check the population against the source of record in one line — which is the check whose absence cost the first fleet.

## QUESTION

Do the conjunction controls behave like the poles or like the contradiction cell — and is that asymmetric by valence?

The first fleet could not ask this. It scored 115 of the source of record's 199 prompt texts and none of the 72 control/matched texts, so *CONTROL_A vs CONTROL_B* and *mean(CONTROLS) vs mean(POLES)* had no data.

## INPUT

**Population: `data/f11_delta_population.json` — the 84 prompts, ENUMERATED.** Not "the output of a script", not "the ACTIVE rows of a table". The file contains the strings.

    source of record      data/f11_quintuplets.json   sha256/16  44a708bf76cfff67
    enumerated prompts    84                          sha256/16  68fe1b5f50f0c10c

    control_a       30      pole_a           5
    control_b       32      pole_b           4
    both_matched    10      both             3

The 84 are exactly `{all prompt roles in the source of record} − {the 115 the first fleet ran}`. The 115 are **not** re-run: their twp, logits and residuals are in hand, verified, and unaffected by this.

**Roster: 104 checkpoints**, `Registry().base_aligned_pairs()`, the same as the first fleet. 90 of them produce data; the other 14 fail at tokenizer or load and are listed in `data/model_load_environments.json`.

    104 checkpoints x 84 prompts = 8,736 cells

## INSTRUMENT

**`scripts/twp_cloud.py`.** One forward pass per prompt, batch-1, producing three artifacts in the same pass:

    true_word_probs   THE COLUMN THE ANALYSIS READS.  Threshold-bounded prefix
                      expansion at theta = 0.001, complete for every word above
                      it. Rows are per (word, first token); residual is three
                      numbers (tail / drop / open).
    logit vector      full-vocabulary, last position, f16 sidecar indexed
                      positionally by `logit_row`
    hidden states     final position only, (n_layers+1, d_model) float32,
                      indexed by `hidden_row`

**Which column is read: `true_word_probs`.** RH's ruling at [5136] — twp's theta and the analysis threshold are both 0.001, and twp at that floor is complete for every word above it. Logits are a by-product; residuals are for L3.

**`compute_dtype` is declared IN the spec** (`data/f11_twp_spec.quintuplet_delta.json`), bfloat16 on the 10 scan architectures. Falcon-H1 at fp16 measures finite 1/12 on its own battery prompts; the first backfill inherited the float16 default because its spec was generated from one that declared nothing.

**Environments are plural** and each cell carries its own `torch` / `transformers` / `device` stamp. `data/f11_env_plan.json`: 82 default, 10 needing torch ≥ 2.6, 10 needing `mamba-ssm` + `causal-conv1d`, 2 needing two GPUs.

## OUTPUT

    data/f11_twp_delta/<model>.jsonl        one record per (model, prompt)
    data/f11_twp_delta/<model>.f16          logit payload, row n = nth
                                            logit-bearing jsonl line
    data/f11_twp_delta/<model>.hidden.f32   residuals, same positional contract

Each record carries: `model`, `prompt`, `theta`, `rows`, `residual`, `rule_version`, `dict_sha`, `logit_row`, `logit_dim`, `hidden_row`, `hidden_shape`, `compute_dtype`, `device`, `torch`, `transformers`, `loader_id`.

**Ingest:** `scripts/twp_ingest.py` (word probabilities, validates `Σ P + residual == 1.0` per line before writing) and `scripts/f11_sidecar_ingest.py` (logit index entries, hardlinked into the archive root; hidden-state manifest). `scripts/twp_sidecar_check.py` verifies the positional pairing before either.

## ANALYSIS

**Primary:** CONTROL_A vs CONTROL_B — is the conjunction effect asymmetric by valence? Per-checkpoint, roster-level Wilcoxon, per-triplet reporting always.

**Declared secondaries:** mean(CONTROLS) vs mean(POLES) — the conjunction/length effect, estimable on both poles, which is [5063].1's missing cell. And BOTH vs BOTH_MATCHED, now that the matched cells exist.

**Prior:** the controls are near-synonym companions on the same side of the same semantic dimension as their pole ([5083] convention), so they should sit *with* the poles and not with the contradiction cell. If they sit with BOTH instead, the conjunction effect is about conjunction rather than contradiction, and the whole M02 reading changes.

**Both branches are reportable.** Controls-with-poles supports the contradiction reading; controls-with-BOTH undercuts it. Neither outcome is a null to be buried.

## COST

**~$12–18, ~2–3 h wall.** Compute is 8,736 forward passes — minutes. The bill is **model acquisition**: the same 90 checkpoints must be pulled again, and download dominates. That is the real cost of the first fleet's population error, and it is paid once here.

No coding cost — this is logit-grain only.

## WHAT IS GATED ON WHAT

1. Malign posts the enumerated 84 (`data/f11_delta_population.json`, list hash `68fe1b5f50f0c10c`).
2. The pen checks it against `data/f11_quintuplets.json` and posts the check.
3. **RH gives the spend word.** Then it launches, and not before.
