# Plan H2: how deep does alignment go — the sweep

**STATUS: A PLAN under the [5148] standard, not a registration. Nothing frozen.**
Written 2026-08-10 by the malign seat. Successor to `plan_h_logitlens.md` (lacan,
2026-08-09), which is **not superseded and should be read first** — its lens fix,
its method, and its withdrawals all stand. What changed is the QUESTION and
therefore the instrument.

Provisional finding this scales: `findings/F42_alignment_depth.md` (to move to
`meta/M01_displacement/findings/` as H).

---

## 0. QUESTION

**RH's, stated plainly: counter the claim that alignment is a small change
DOWNSTREAM of pretraining, and show it is deep inside the mechanism.**

That claim exists inside this project. `F05` (grade D, rescoped) was revised
2026-07-01 to say displacement is "overwhelmingly a final-layer operation…
**alignment changes the readout, not the representation**. Hidden states are
nearly identical between base and aligned through 97% of the network."

**The sweep tests that, on the project's own roster, with four instruments that
fail differently.**

## 1. WHAT THIS TAKES FROM PLAN H, AND WHAT IT LEAVES

**Takes:** the corrected lens (`head(hidden[-1])` IS the model's logits, and the
function refuses if they disagree); the multi-token word defect and its fix; the
principle that a per-layer distribution is *what a model exiting there would
say*, never *what the layer represents*.

**Leaves:** plan H's INPUT question — "which prompts for `expand_layers`" — is
not this plan's. `expand_layers` (threshold-bounded discovery at every layer) was
priced at 5.2x–32.7x twp and **declined by RH on 9 Aug** ([5205]/[5206], lacan
concurring) on the ground that **it has no null**. This plan uses scoring of a
KNOWN vocabulary instead, which needs no threshold, and the vocabulary comes from
the store via `Step`/`Cell`/`movement`.

**θ is a DISCOVERY device, not a MEASUREMENT one.** Lowering it grows the seed
set 19x / 214x / 973x at 1e-4 / 1e-5 / 1e-6, worst in the mid-stack where the
readout is `efon` and `$MESS`.

## 2. INPUT — ENUMERATED, HASHED, GATES AS COLUMNS

`data/h2_sweep_population.json`, producer `scripts/h2_sweep_population.py`.

    prompts       57 DISTINCT STRINGS   sha256/16  0957c5b627458224
                  from 60 prompt_ids -- three ids share a string with another
                  and are deduplicated ON THE STRING, because the unit is the
                  (pair, prompt) forward pass and two ids for one string are
                  one measurement
                  44  sexual/violence x liminal/explicit  (14/10/10/10)
                  16  e7_* minimal pairs, OF WHICH 8 ARE CHINESE (`_zh`)
    pairs in Registry      52
    INCLUDED               26      list sha256/16  e22a30b65fe9a06a
    (pair, prompt) cells   1,395   min 28 / median 57 / max 57 per pair

**THE CHINESE PROMPTS ARE DECLARED, NOT SMUGGLED.** They are in because the e7
minimal pairs were built with them and dropping half a designed contrast after
seeing the roster is population construction after the fact. But `lang` is a
column, and **no pooled figure may mix them** — the tokenizers differ in how many
tokens a word costs, which is exactly the quantity the lens reads.

**Five gates, applied in order, each recorded as a column so an exclusion is a
query and is reversible. Counts are the FIRST gate that excluded each pair:**

    LOCAL           3    both arms have a snapshot; never downloads
    HAS_ATTENTION   7    SSM / Mamba / RWKV / recurrent / hybrid, excluded on a
                         stronger ground than the path: a block swap there is
                         not the same operation and must not be pooled with one
                         that is
    ARCH           13    blocks reachable at `model.model.layers`, read from
                         config.json rather than guessed from a family name
    PREFLIGHT       2    `model_load_environments.json`, matched on CAUSE not on
                         the environment tag — mpt's repo is gone everywhere;
                         deepseek and croissant mangle the prompt in the
                         TOKENIZER, which no card changes
    CELLS           1    >= 10 prompts with a stored cell and >= 4 movers

**The 26 are not "the models on disk". They are what survived five gates**, and
the plan says so in those words because "6 pairs" meaning "the 6 smallest,
selected for MPS speed" is a live lesson from M04 this week ([5224].2).

**ARCH IS THE BIGGEST EXCLUSION AT 13 AND IS THE ONE TO REVISIT.** Some of those
13 are architectures whose blocks live at a different attribute rather than
architectures that have no blocks; the accessor is a whitelist read from
config.json, and widening it is a build task, not a result. Recorded here so the
number is not mistaken for a fact about the models.

## 3. INSTRUMENT — FOUR MEASURES THAT FAIL DIFFERENTLY

    ||dW||       what training CHANGED. Read from safetensors, no model load.
                 `malign_logits/weightdelta.py`, by = block | group | head | key.
                 NO confounds. Answers "did training touch it", never "does it
                 matter".
    ||dh_L||     that the computation DIVERGES, per depth. No unembedding.
    weight patch CAUSAL. Hybrid: aligned blocks top-K or bottom-K, **BASE head,
                 embeddings and final norm throughout**, so the readout is held
                 fixed by construction and the frozen-head problem cannot arise.
    repr patch   aligned RESIDUAL at depth L through base weights.
    per-word lens the WORD-LEVEL content of the divergence — the only measure in
                 word space, and the only one that can say what a model exiting
                 at L would have said.

**READ THE TWO PATCHES TOGETHER OR NEITHER.** On Llama the representation patch
says half the behaviour is recoverable only at L26/32 — the downstream picture —
while the weight patch says the last two aligned blocks give 6–16% and everything
below gives 55–73%. Both are true: the early changes are neither inert nor
independently sufficient. **Quoting either alone inverts the conclusion.**

### Gates that are columns, not filters

    CEILING          recovery with ALL blocks aligned; the denominator every
                     normalised figure divides by. Excluded Qwen2.5-0.5B
                     (-0.25) and OLMo-2-0425-1B (3.96) in the pilot. **VARIES
                     BY PROMPT** — Llama spans 0.47–0.94 across four — so it
                     cannot live at pair level.
    FROZEN HEAD      fails everywhere: 29 pairs surveyed, zero frozen, zero with
                     a single unchanged row, spread 2.3e-03 to 1.18. Routed
                     around, not waited for.
    CROSS-READ OOD   top-1 mass ratio of the cross-read against the true read.
                     Amber fails at 5x; `AmberSafe` is `dpo_of` **AmberChat**,
                     so that pair spans two training stages.

### Rule is a column

Every cell computed under **`CANONICAL` and `LENS`**. They disagree on the
onset NUMBER (11/18/27 against 19/19/23) and agree on the DIRECTION. Storing
both makes specification sensitivity a `groupby` instead of a re-run, and
`CANONICAL` is one declared convention among several — not ground truth.
**`LENS` risers are NOT null-tested and must never be described as beyond
renormalisation.**

## 4. OUTPUT — SIX TABLES, BECAUSE THERE ARE SIX GRAINS

Long, not wide: a new measure is a row, not a column.

    depth_pairs.parquet        1 row per pair
                               arch, n_blocks, wdelta_{attn,mlp,norm,head,embed,
                               final_norm}, head_rel_diff, head_rows_moved_frac
    depth_blocks.parquet       (pair, layer)          wdelta_block/attn/mlp
    depth_heads.parquet        (pair, layer, head)    wdelta_head, proj
    depth_cells.parquet        (pair, prompt)  <- THE UNIT
                               rule, n_fallers/risers/flats, residual_share,
                               ceiling, repr_L50, lens_ood, lens_permitted
    depth_cell_layers.parquet  (pair, prompt, layer)
                               dh, repr_recovery, lens_gap, flats_lo/hi, outside
    depth_patch.parquet        (pair, prompt, direction, k)   recovery

Every row stamped with `run_id`, `rule`, `dtype`, `device`, `torch`,
`transformers` and the population hash, per [5055].1.

## 5. ANALYSIS

**THE UNIT IS THE (pair, prompt) CELL.** lacan [5224].1, registered as binding on
every layer-level read: *"the heads are not replicates; they are one measurement
decomposed"* — opposite signs at p=1.6e-15 and p=2e-32 on two prompts of one
model. **Layers within a model are the same structure.** Anything whose n is the
layer count, or words within a cell, is describing within-model structure.

**THE CLAIM IS ORDINAL, NOT A DEPTH NUMBER.** Differences accumulate, so a curve
flat to L18 and rising after is what a uniformly distributed effect looks like
crossing a noise floor: **onset-in-curve is not onset-in-mechanism**. What the
curve licenses is an UPPER BOUND on lateness, and `||dh||` carries that better.

Primary, descriptive, no test:

    last-2 blocks recovery  vs  all-but-last-2 recovery,   per cell,
    reported as a distribution over the 26 pairs

Pilot values (3 pairs): 6–16% against 55–73%.

Declared beside it: `||dW||` first-third vs last-third ratio; `||dh||` at L0;
`repr_L50`; and the per-word lens count of significant (word, layer) pairs
before the final two layers — which was **zero of 3,465 tests** on one pilot cell
and **two of 4,257** on another, both in the last two layers.

**Cross-family: report per-pair curves and ask whether the ORDINAL claim
replicates. Do NOT normalise depth** — it reads as conservative and assumes the
work is distributed proportionally to depth, which is the thing under test
(lacan [5222].1).

## 6. COST

**~2.5 h locally, no spend.** 26 pairs, ~4–5 min each including load; measured
0.9 min for 10 prompts on Llama after a 20–40 s load. MPS, fp16, one pair
resident at a time. No cloud, no downloads: the LOCAL gate guarantees it.

## 7. WHAT THIS DOES NOT CLAIM

Not a mechanism story: `||dW||` says training touched a block, never that the
touch matters. Not a depth number. Not anything about SSM/hybrid architectures,
which are excluded by gate. Not cross-family layer correspondence — layer *i* to
layer *i* is the fine-tune's initialisation and does not extend past a pair.
And a hybrid model is **not a model anyone trained**: blocks compose only because
one checkpoint initialises from the other, the interaction term is not
identified, and the top-K curve is non-monotone at ~0.05.

## 8. GATED ON

1. This plan posts with the population enumerated (`e22a30b65fe9a06a`). ← here
2. Pen checks the population. lacan is asked one thing specifically: whether the
   ordinal comparison in §5 is the right primary, given [5224].1 is theirs.
3. Build, in this order, all cheap:
   a. **The parquet writer.** The battery currently appends nested jsonl; the
      six tables do not exist. This is the only real work.
   b. **Widen the architecture whitelist** and RE-RUN the population producer, so
      the 13 ARCH exclusions separate into "different attribute" and "no blocks".
      **Any pair this adds is added BEFORE any result is read**, and the hash
      changes — a population that grows after a look is a different population.
   c. Store-coverage per (pair, prompt), which the CELLS gate counts but does not
      yet write per cell.
4. Run. ~2.5 h, local, **nothing spends**.

**IF (b) CHANGES THE ROSTER, THE HASH IN THIS PLAN IS SUPERSEDED AND THE PLAN
SAYS SO IN A DATED LINE.** It is not edited silently.
