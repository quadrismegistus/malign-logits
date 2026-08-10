# M04 plan: one attention extraction, written to parquet, before any statistic

**A PLAN DOCUMENT UNDER [5148], NOT A REGISTRATION.** It fixes the free
parameters of a data collection so they cannot be chosen after seeing a number.
No hypothesis is registered here and no result is claimed. Requested by
@registrar at [5226] in answer to [5224].7: pin before extraction, pen checks
the roster, then spend the three hours.

Prior: `registrations/plan_attention_back.md` (the design), the provisional
finding `findings/attention_back_cross_own.md`, docket [5224], [5225], [5226].

## 1. Why collection is being separated from statistics

The pilot re-ran the same forward passes three times in one session, because
each new question needed a field the previous dump had discarded: the position
axis, then `U` and the per-head levels, then the arrays behind the medians. RH:
separate data collection from statistics. Extract once, write parquet, make every
downstream question a query.

The pilot also selected its roster by model size for MPS speed and reported it as
"6 pairs". That is selection on a variable that could relate to the outcome, and
it is the reason the roster is enumerated below rather than described.

## 2. Roster, ENUMERATED

All 36 pairs in the Y corpus (`design=slot-sampled-y-v1`), both arms, subject to
two mechanical exclusions applied BEFORE extraction and logged with counts:

- **tokenizer mismatch.** One arm's ids are run through the other arm's model, so
  the two tokenizers must agree on the prompt and on every forced word. Checked
  per pair on the real strings. Known to fire on the zephyr lineage
  (`' cock'` -> `[12408]` against `[28705, 12408]`) and on internlm2, whose base
  omits a BOS both aligned arms add.
- **no attention.** Mamba and RWKV checkpoints have no attention heads.
  `falcon-mamba-7b`, `Falcon3-Mamba-7B-Base`, `rwkv-4-7b-pile`.

Every excluded pair is listed with its reason in the run receipt. **The surviving
count is whatever it is; it is not a target.**

The pilot's six -- SmolLM2-360M, Qwen2.5-0.5B, OLMo-2-0425-1B, Falcon3-1B-Base,
TinyLlama-1.1B, stablelm-2-1_6b, all between 0.23B and 1.21B -- are a subset of
this roster and carry no special status in it.

## 3. Population per pair

    prompts     all 5:  sexual_explicit_1, _3, _5, sexual_liminal_6, _7
    cells       every forced word the Y run used for that prompt, PLUS the
                undisturbed cell (word = null)
    samples     16 per (model, prompt, word), taken in file order, no selection
    arms        base and aligned

Word cells are NOT reduced to faller/riser/non-mover at extraction. Every forced
word is extracted and the movement roles become a JOIN against
`attn_words.parquet`. The pilot hard-coded the three arms into the producer,
which is why changing the selection rule cost a re-run.

## 4. Declared parameters

**dtype: bfloat16, uniform across the roster.** [5226] choice 1. A 13B in fp32
will not load here, and mixing dtypes would confound model size with numerics.
The extraction therefore SUPERSEDES the 28 fp32 cells rather than extending
them. Per [5131] discipline, **one cell is run in both dtypes as the
comparability floor** -- SmolLM2-360M / `sexual_explicit_1` -- and the two
values are reported side by side in the receipt whatever they show.

**Window: 128 positions after the slot.** Bins, disjoint:

    j = 0 | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64-127

Cumulative windows are NOT stored: a cumulative sweep of a front-loaded effect
falls and then flattens by construction, which is how a decaying effect gets read
as sustained. Any cumulative summary is derivable from the disjoint bins; the
reverse is not.

**Samples: 16 per cell.** @malign's to answer per [5226]; unanswered at time of
writing, so 16 stands as the declared value and the receipt records it. It gives
a split-half null of 8 against 8, which is what caught the forcing-distortion
artifact in the pilot.

**Both weightings, always.** Raw alpha and norm-weighted (Kobayashi
alpha x ||v_i||) are stored side by side in every row. They agreed on direction
everywhere in the pilot and this removes the choice from every downstream
analysis.

**Slot index.** For a forced word, `plen - len(word_ids)`. For the undisturbed
cell, `plen`. These are the same absolute index -- `|prompt|` -- because the
producer builds the forced prompt as `prompt + " " + word`
(`scripts/vllm_y_run.py:119`). The prefix-identity that rests on is asserted per
cell: `tokenize(prompt + " " + w)` must equal `tokenize(prompt) + tokenize(" " + w)`,
and a merging boundary is logged and the cell dropped.

**Ragged, never padded, never truncated to the shortest.** Sequences that stop
early are averaged over whatever positions they reach and the count per position
is stored. Padding manufactures a decay; truncating to the shortest made the
window depend on one sequence and produced per-word means over different windows.

## 5. Schema

    attn_heads.parquet    one row per (model, arm, pair, prompt_id, word, layer, head)
        n_seq, and per bin: mean and sd of raw alpha, mean and sd of
        norm-weighted; vnorm at the slot; layer/head indices and n_layers/n_heads
        so relative depth is derivable without a config lookup.

    attn_seqs.parquet     one row per (model, arm, pair, prompt_id, word, sample_idx)
        head-pooled raw and norm-weighted per bin; slot_index; slot_logprob read
        from the SAME forward pass; n_positions reached; finish_reason; plen;
        len(full_ids).
        This table is what makes split-half and permutation nulls possible
        without re-running anything, which the pilot could not do.

    attn_words.parquet    one row per (pair, prompt_id, word)
        P (base) and Q (aligned) from true_word_probs via Step/Cell, delta,
        and the shard's declared `cls` and `direction`. Movement roles are a
        query against this, never a hard-coded selection.

`slot_logprob` comes from the extraction pass, not from twp: twp is a word
probability summed over token paths and `scored_by_*[0]` is a single-token
logprob, and mixing the two would compare estimators.

## 6. What this does NOT decide

No hypothesis, no arm-selection rule for analysis, no statistic. The pilot's
prediction (faller below, non-mover and riser together) was refuted at 14 of 28
cells and is not reinstated here.

**The unit for any statistic computed from these tables is the (pair, prompt)
cell or the pair, never the head and never the layer.** Registered at [5226] as
binding on every head- and layer-level read in the campaign: the heads are not
replicates, they are one measurement decomposed, and the exhibit is two prompts
of one model returning opposite signs at p = 1.6e-15 and p = 2e-32.

**A base-probability covariate is expected.** @malign at [5225].4 measured the
per-layer gap tracking a word's output base probability, OLS slope rising from
-0.002 at L0 to +0.123 at L32 on Llama. The pilot hit the same confound from the
other side. `attn_words.parquet` carries P and Q so any analysis can residualise
on it; whether to is an analysis decision and is not taken here.

## 7. Cost, measured

0.23s per forward pass on Qwen2.5-7B at window 32, measured, faster than the
sub-1B models which are overhead-bound. Window 128 raises the attention tensor
but only column `i` is retained.

    36 pairs x 2 arms x 5 prompts x ~7 word-cells x 16 samples = ~40,000 passes
    ~3 hours local, no API spend, plus ~15s model load x 72 checkpoints

Sharded per model and resumable: a completed model's shard is skipped, so an
interrupted run costs the current model only.
