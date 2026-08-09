# Plan: the L2 corpus — generated passages for the registered primary

Written under the [5148] standard: a plan document, not a registration. No freeze, no hash ceremony. The hashes are here so the pen can check the population against the source of record in one line, which is the check whose absence cost the first fleet $15.

**Authorisation:** RH, 2026-08-09, "let's move on to the passage generation", dispatched at [5196]. Spend authorised at [5156].2. **The pen's population check runs on this document before any box spins.**

## QUESTION

Does the model leave the frame when the prompt is contradictory — and does it leave it more than when the prompt is merely conjunctive?

This is the registration's own PRIMARY (`coded frame_exit`, redo §4), and it has never had a corpus. Everything computed so far has been L1, at the next-word grain, and [5195] now puts a denominator on why that grain cannot answer it: the opposition the prompt names carries **1.96%** of the next token's variance. A single word is not yet a departure.

**[5195].1 also names a third thing neither the registration nor the L1 instrument anticipated**, and this corpus is where it becomes measurable. The aligned BOTH continuation does not resolve the contradiction and does not stay in it — it *narrates* it ("She was torn in two directions") and then *deliberates about* it ("Maybe she should feel guilty, but she didn't"). That is neither RESOLVE nor ENGAGE nor a topic shift. **The coding design is a separate post** and this document does not fix the classes; it flags that meta-commentary must be visible to whatever coding is built, because a scheme with only the registered four would score it as OFF-FRAME and lose the distinction.

## INPUT

**`data/f11_l2_population.json` — the prompts, ENUMERATED AS STRINGS.** Not "the output of a script", not "the ACTIVE rows of a table". Producer `scripts/f11_l2_population.py`, which refuses to write unless every role is represented.

    source of record        data/f11_quintuplets.json   sha256/16  44a708bf76cfff67
    enumerated prompt list                              sha256/16  e5da397ff891af74

    groups          41 primary | 2 held beside | 1 dropped (f11_species_wolf, RETIRED)
    (group, role)   197
    DISTINCT        187      <- the generation unit
    en / zh         95 / 92 distinct strings, 21 / 20 groups
    held beside     10 strings (f11_reason, f11_reason_zh — the declared
                    weak-manipulation negative control, generated BESIDE and
                    reported beside, never pooled)

Status filter per [5084].2: the file carries status, it does not filter. The two `MIXED: ACTIVE/DISPUTED` groups are exactly the two negative controls.

### The generation unit is the DISTINCT STRING, and ten cells are shared

**Ten prompt strings are claimed by more than one (group, role).** Enumerating by group would generate them twice and invite counting them twice — the duplicate-as-unit error that took 39 lineages to 34 and that made `tulu`/`tulu-no-safety` one measurement counted twice on the liminal/explicit claim. Each string is generated once; the file records every claim on it.

Two group pairs overlap, and they overlap DIFFERENTLY:

    f11_holy / f11_holy_b     both, control_a, control_b ALL BYTE-IDENTICAL
                              (differ only in the poles: temple/alley against
                              place/place — a lexical-matching manipulation)
    f11_beauty / f11_beauty_ugly   pole_a and control_a identical; pole_b,
                              both and control_b differ (disgusting/ugly)

**This is where the registration's "15 EN triplets" ([5093].1) comes from and it is now checkable rather than inherited.** Sixteen EN groups carry both controls; `f11_holy` and `f11_holy_b` are identical on all three cells the primary reads, so they contribute **one** unit and the primary's n is **15**. `f11_beauty`/`f11_beauty_ugly` are two units whose contrasts are correlated through a shared control — reported as such, not collapsed.

Note the asymmetry: holy/holy_b are one unit for the redo primary (BOTH vs controls) and two partially-overlapping units for N3's excess (AB vs poles), because the poles are what differs. **A collapse rule is per-contrast, not per-group.**

### Both languages are enumerated, and that is a cost decision not a scope claim

**[5065].3: samples cannot be added retrospectively under a second decoder.** A cell not generated now can never be coded alongside these — only against a second decoder, which is not comparable. The zh descriptive-only ruling ([5194].D) is about the **L1 coverage gate**, whose cause is θ-truncation on flat next-word distributions; that cause has no analogue in a generated passage. Generating zh costs ~$5 now; skipping it forecloses the question permanently.

**The CODING population is a separate decision and this document does not make it.** Generating zh does not commit anyone to coding zh.

## INSTRUMENT

**vLLM, mode raw (no chat template), decoder PINNED IN THE DESIGN STRING FROM BIRTH.**

    design string     m02-l2-frame-exit-v1
    n                 20 per cell, GENERATED in every cell regardless of coded
                      depth (registration §L2 — coded depth may be Option B,
                      generated depth may not)
    temperature       1.0
    top_p             1.0        pinned, not defaulted
    top_k             -1         disabled explicitly
    max_tokens        256
    min_tokens        0
    presence/frequency/repetition penalties   0.0
    stop              none
    seed              deterministic per (model, prompt_sha, sample_idx)

**WHY EVERY FIELD IS NAMED INCLUDING THE ONES THAT LOOK LIKE DEFAULTS.** HF `generate()` MERGES the checkpoint's `generation_config`; vLLM REPLACES it. Either way, a parameter not named is a parameter **the checkpoint chooses**, and the roster spans 104 checkpoints from 40-odd organisations with no shared convention. An unpinned `top_p` is not a constant across the roster — it is a per-vendor covariate silently aligned with the arm contrast, because vendors ship different defaults for base and instruct.

**The runner asserts the resolved sampling params equal the declared ones and records them per record.** A declared decoder that is not verified against what the engine resolved is prose, not a pin.

**Tokens are stored, never a derived `text_clip`** (registration §L2). Text is recorded too, but the tokens are the artifact.

**Roster: `Registry().base_aligned_pairs()` — 52 pairs, 104 checkpoints**, the same as both prior fleets. 90 produced usable data at L1; the 14 that produced nothing are rows in `data/model_load_environments.json` with recorded reasons and are expected to fail identically here. Achieved is reported against declared with a named reason per missing cell; **no substitution to hold n**.

## OUTPUT

    data/f11_l2_gen/<model>.jsonl     one record per (model, prompt, sample_idx)

Per record: `model`, `prompt`, `prompt_sha256_16`, `sample_idx`, `token_ids`, `text`, `design`, `decoder` (the RESOLVED params), `seed`, `engine`, `engine_version`, `dtype`, `device`, `n_declared`.

Ingest mirrors the twp rail: a sidecar check before ingest, conservation asserted per line, skips enumerated by (model, prompt) rather than absorbed.

## ANALYSIS

**Declared here; the coding scheme is a separate post and is what actually gates the analysis.**

- **PRIMARY** — coded `frame_exit`, excess of BOTH against mean(CONTROL_A, CONTROL_B), **n = 15 EN triplets** after the holy/holy_b collapse, per checkpoint, roster Wilcoxon, per-triplet reporting always.
- **SECONDARIES** — CONTROL_A vs CONTROL_B (valence asymmetry of the conjunction effect); mean(CONTROLS) vs mean(POLES) (the conjunction/length effect); BOTH vs BOTH_MATCHED.
- **N3 §4** — is EXIT the modal mechanism across lineages, at the lineage unit, one-sided binomial against p=1/3 with NULL excluded from the null, DEFF-deflated.
- **NEGATIVE CONTROL BESIDE** — `f11_reason`/`_zh`. If the effects appear there too they are not about contradiction.
- **BOTH BRANCHES TRAVEL.** If the controls behave like BOTH rather than like the poles, the effect is about conjunction rather than contradiction and the M02 reading changes. That is a result, not a null to bury.

**A coder-reliability gate precedes any coded number and is not part of this spend.** The L1 pilot found the coder's answer moves with its prompt, that κ would have selected the biased arm, and that a coder disagrees with itself about as much as two vendors do ([5188]); [5189] found OFF-FRAME at chance from two independent instrument families. **None of that is evidence about passage-grain coding** — it is evidence about word-grain coding, and [5195].5 is the argument that the passage grain is where the construct lives. But it is not a prior in favour either, and the gate must be measured on these passages before coded depth is purchased.

## COST

    scope                            cells   sequences      tokens
    ALL, both languages                187     388,960  99,573,760
    EN only                             95     197,600  50,585,600
    EN, registered-primary roles        49     101,920  26,091,520

**~$10–20 for the full scope, and the honest statement is that throughput is the unmeasured term.** Compute is 99.6M output tokens; at a vLLM band of 3,000–8,000 output tok/s on the box classes the delta used, that is **3.5–9 GPU-hours** across the roster. Download is the other half and is the better-known one: ~1.5 TB at the measured ~125 MB/s (advertised link speed does not predict delivered throughput — four boxes, 5.8–8.1 Gbps advertised, 108–143 MB/s delivered, fastest-advertised was slowest).

**The pen's "~$2 class" at [5156].2 was priced before the population was enumerated and is low by roughly an order of magnitude at full scope.** Not a disagreement — an estimate meeting its denominator.

**The first box reports MEASURED tokens/s before the rest launch**, which is the discipline that took the delta from an estimate to $3.30. Select by capability, never by product name (`gpu_name=RTX_A6000` returns zero offers while `RTX A6000` sits in the results); shard by download size, because bytes are the bill; small disks, because `--purge` fires before each download and on every exit path.

**Recommendation: ALL, both languages.** The marginal cost of zh is ~$5 and the marginal cost of *not* generating it is that the question cannot be asked under this decoder, ever.

## ADDENDUM — TWO POST-GENERATION READOUTS (lacan, on RH's ask)

**Neither changes the population, the decoder, or how a passage is made.** Both are teacher-forced forward passes over text this plan already generates, so the source hash `44a708bf76cfff67`, the list hash `e5da397ff891af74`, the 187 strings and the pen's check at [5198] are all untouched. Scoring has no decoder to pin.

**The whole argument for doing them in THIS session is the 1.5 TB.** Malign priced download as half the bill. A forward pass over resident weights is nearly free; the same pass next month pays the full download again. That argument applies to these two and to nothing else here.

### 1. CROSS-SCORING — vLLM, same rail, already implemented

Each passage scored under its own checkpoint and **under its pair partner**. Two scorings per cell; the 2×2 within pair.

Not a new capability: `scripts/vllm_slot_sampled.py` already does teacher forcing with `SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)`, reads the actual token's logprob from `plen` onward, and records `scorer` and `src_model` as separate fields. The record shape cross-scoring needs is the shape that code already writes.

**Why it is worth more here than a coded field.** `logP_aligned(base passage) − logP_base(base passage)` is how much alignment disprefers the base's continuation. Taking the excess of that at BOTH against mean(CONTROL_A, CONTROL_B) is **the registered primary with no annotation anywhere in it.** After [5187]–[5189] — the coder's answer moving with its prompt, κ selecting the biased arm, OFF-FRAME at chance from two instrument families — a coder-free instrument for the same contrast is worth its 10%.

**And it is per-token, which tests [5195].5 rather than assuming it.** If the penalty is negligible at token 1 and accumulates with position, "a single word is not yet a departure; a passage is" becomes a curve instead of an argument. If it is flat in position, the move to L2 was wrong and we would know it from the same data.

**PRE-CHECK, AND IT IS NOT OPTIONAL.** The existing code passes token **ids** between models. That is valid only where a pair shares a tokenizer. The `max(full_ids) >= vmax` guard catches ids out of the scorer's range; it does **not** catch different segmentation, which would silently score a different string and return a perfectly plausible number. Verify tokenizer identity per pair before scoring by ids, and score by re-tokenized text wherever it fails.

**Scheduling constraint:** both members of a pair must be resident on the same box, which cuts against sharding purely by download size.

Cost: one further prefill of 99.6M tokens, roughly +10% on the generation estimate.

### 2. PER-LAYER PROJECTIONS — transformers, a DECLARED SUBSET, after vLLM teardown

`twp.py` is transformers-only: every readout path calls `model(ids, attention_mask=att, output_hidden_states=readout.needs_hidden)` and `expand_layers` indexes `out.hidden_states`. vLLM does not expose per-layer states for generation. **But no HF generation is required** — a per-layer readout over an existing passage is a single prefill, giving every position × every layer in one pass. Generate on vLLM as planned, tear it down, load HF, run the subset back through.

**Write the projection, not the state.** Raw residuals for a generated passage are `n_layers+1 × d × 256 × 4` bytes — 138 MB per passage at llama-7b's `[33, 4096]`, which is ~54 PB at full scope and infeasible by six orders of magnitude. The projection `h·â` is `n_layers+1` floats per token, ~33 KB per passage, a 4,000× reduction. The axis is the model's **own** `pole_a`/`pole_b` residuals, already on disk in the twp sidecars — the L3 geometry construction, in the model's space rather than an embedder's, which is the axis [5195] found weak at 2% of variance when built from BGE/GloVe.

**Scope is a declared subset with its size named in advance, not "as many as fit."** At one pair × the 15 EN primary groups × 3 roles × 20 samples = 1,800 passages ≈ 460k tokens of prefill, minutes. Ten pairs is under half an hour. The pair list should be named before the run.

Both ride on the same scope and spend word as the generation; neither is authorised separately, and neither authorises any coded depth.

## WHAT IS GATED ON WHAT

1. Malign posts this plan with the enumerated population (list hash `e5da397ff891af74`). ← **here**
2. The pen checks it against `data/f11_quintuplets.json` and posts the check.
3. **RH gives the scope word** (ALL / EN / EN-primary-roles) and the spend word.
4. Boxes spin, first box reports measured throughput, the rest follow.

The coding design, the coder-reliability gate, and the coded-depth purchase are all downstream of this and none of them is authorised by it.
