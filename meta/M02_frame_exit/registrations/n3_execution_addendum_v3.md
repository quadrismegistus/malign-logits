# N3 — execution addendum v3: the producer swap

**Content-independent. Posted before the fleet's cells are consumed.** States no observed quantity, names no arm, fixes no construct. `n3_frame_exit_registration.md` (frozen July, arm-unread by attestation) is unchanged and unamended; its **body hash is untouched**, as with v2.

    v1   41da48c2def102b9   a0c65412   fp16 pin, whitespace span, whole-checkpoint refusal
    v2   c0f8ea1391e4f89d   6900c9ad   native dtype, envelope span, triplet refusal
    v3   this document                 PRODUCER SWAP: f11_l1_logits.py -> twp_cloud.py

Ruled at docket [5127]/[5129]. Occasioned by RH's redirect: L1 collects **true_word_probs and logits together** rather than logits alone, because `twp_cloud.py` already computes the logit vector as its depth-1 selector and writes it to a `.f16` sidecar indexed by `logit_row`. The logits-only runner was doing strictly less work for the same model loads and the same downloads, which are what the run costs.

**Why a new document rather than an edit.** v2 pinned its preconditions *as properties of a named file*. That file is no longer the producer. A precondition that describes code which is not running is the failure class this campaign has hit repeatedly — a check that reads as verified and covers something else.

---

## 1. Mode and encode path — RESTATED, not carried over

v2 §2 said `mode: raw`, verified at `models.py:99`, on the ground that `get_base_logits` has no branch that could apply a template. **`twp_cloud.py` never calls that function.** It has its own `encode_prompt()` and `BOS_POLICY`. The v2 sentence is true about a producer that is not running and must not be read as covering this one.

    mode         raw. twp_cloud applies no chat template on any path.
    encode       twp_cloud's own encode_prompt() under BOS_POLICY.
    verified by  the round-trip guard below, which tests THE PATH THE RUN
                 TAKES rather than the tokenizer in isolation.

## 2. Round-trip — ENFORCED IN CODE, and on a stronger path than v2's

In the run loop, per prompt, from `ids` as produced by `encode_prompt()`:

    back = tok.decode(ids, skip_special_tokens=True)
    if back.strip() != prompt.strip():   ->  refused

**This is stronger than the guard v2 described, and the difference is not cosmetic.** v2's check encoded the prompt itself; this one checks the ids the forward pass will actually receive. The producer's own comment records why: a precondition script encoding with `add_special_tokens=False` passed 100,837 pairs with zero failures while the runner encoded through the BOS policy — and the mismatch **killed zephyr-7b-beta at 0/979 in the v3 grid as a false positive.** The guard is the survivor of exactly the class it guards.

A standalone roster sweep, `scripts/tokenizer_roundtrip_sweep.py`, probes English, contractions, digits+punctuation, CJK and mixed with exact-then-whitespace-normalised comparison.

## 3. Finiteness — coverage stated honestly

**Not enforced in the runner.** Covered at two later points, and nothing escapes silently:

    at INGEST   twp_ingest.py validates `Σ P(words) + residual == 1.0` per line
                and refuses failures. A non-finite logit makes the sum non-finite
                and the line is rejected. This reads EVERY record, not a sample.
    at READ     cache.py:295 raises on any non-finite value, no bypass flag,
                on the path every read crosses.

**The residual gap is real and is a cost in time, not in correctness:** a broken write is discovered at ingest rather than at the checkpoint, so the box is already spent. It cannot produce a wrong number; it can waste a run.

## 4. Read-back — WAIVED, with the reason

v2 added a first-cell read-back because [5114]'s defect was a write that succeeded and a read that could not — 2,415 cells keyed at a dtype numpy cannot construct.

**Waived here, on three grounds:**

1. **The ingest IS a read-back, of 100% of records rather than one per checkpoint**, and it runs before anything enters the canonical store. Later than v2's, and broader.
2. **The failure surface differs.** v2's defect was an index key that could not decode its own payload — a property of the lmdb index. twp_cloud writes flat JSONL plus a `.f16` sidecar whose row *n* is the *n*th logit-bearing line, with no key to disagree with the bytes.
3. **A producer edit during a run is how a corpus acquires two versions.** The fleet is running; the ingest guard covers correctness.

**What ingest must check, and this is the invariant the waiver rests on:** that the sidecar row count equals the count of logit-bearing jsonl lines, per model. The pairing is positional, so a break is silent and shifts every subsequent row. If that check is not present it is to be added **before** ingest, not before the run.

## 5. dtype — DECLARED, and it was not

`compute_dtype` is per-model in the spec, defaulting to `float16`. **No spec declared it**, so the ten scan architectures on the SSM box would have run at fp16. Falcon-H1 measures **fp16 finite 1/12, bf16 12/12**, every failure ≥13 tokens — the defect that produced 5,166 empty cells passing every structural gate, because `sum([]) + 1.0` makes conservation read exactly 1.000000.

    box2_ssm    bfloat16, declared on all ten   (commit e8a6b651)
    box1/box3   float16, the runner default

**fp16 is retained for dense on purpose.** RH's 2026-08-01 ruling computed the 103-model corpus that way, and 103 models are the evidence that it is fine there: **the overflow is a property of a cumulative scan, not of fp16.** Storage stays f16 throughout — finite logits reach |28.4| against f16's 65504, so the cast is lossless in range.

This supersedes v2 §6's "native everywhere". v2's reasoning was right about the risk and wrong to generalise it past the architectures that carry it.

## 6. Population delta — DECLARED

`LOADER_OVERRIDE` forces `PreTrainedTokenizerFast` for `deepseek-llm-7b-base` and `-chat`, bypassing the Metaspace install that deletes every space and empties CJK. So:

    locally    deepseek is REFUSED by the round-trip guard
    on fleet   deepseek PRODUCES CELLS, loader_id stamped, retirement
               condition written against upstream PR #47017

**The fleet's L1 population therefore includes deepseek and the local one does not.** Any local/fleet comparison carries this line.

**Measured, not assumed:** the box runs transformers 5.14.1 and **still has the defect** — `encode("He loved…") == encode("Helovedher…")`, CJK decoding to the empty string, identical to local. It is the override that changes deepseek's status, not the library version. Croissant on the box drops the same six CJK characters as locally.

Registrar's [5129] notes the interaction: fleet deepseek cells are faithful-tokenizer cells **by construction**, which makes them a second instrument for the stored-vs-fresh question — the local broken environment says what space-stripped logits look like, the fleet says what faithful ones do.

## 7. Span-1 — N/A as a producer property

A population rule, not a producer property. It is applied when the spec is built (`scripts/f11_twp_spec.py`), so the refused triplets never reach any runner. Refused: `f11_holy`, `f11_holy_zh`. 39 triplets, 115 prompts.

## 8. Environments — plural, and stamped

One environment does not run the roster. Derived at `scripts/f11_env_plan.py`:

    default   82   safetensors, dense
    torch26   10   bin-only; check_torch_load_is_safe needs torch >= 2.6
    ssm       10   selective-scan: mamba-ssm + causal-conv1d
    twogpu     2   70B, ~140GB bf16 each

**The seam is not avoidable, so it is recorded rather than pretended away.** `twp_cloud` stamps torch, transformers and device on every record. The box is on transformers 5.14.1; the Mac on 5.4.0. That difference is in the data.

## 9. The environment noise floor — required before any cross-environment read

**A metric with no null.** Fleet and local cells differ for reasons that have nothing to do with any tokenizer: CUDA against MPS, transformers 5.14.1 against 5.4.0, and whatever dtype each path resolves. So a difference between a stored cell and a fresh one **cannot be read as "different string" until we know what "same string, different environment" looks like.** Raised by lacan at [5131].2; adopted as a clause rather than as scaffolding, because it answers more than the question that prompted it.

**Measurement.** Pick a checkpoint that round-trips cleanly in **both** environments and is already in the fleet, compute its cell locally on the same prompt, and take the same magnitude statistic. One checkpoint, one prompt, no additional fleet load — the cell exists anyway.

    same string, different environment   ->  THE FLOOR
    stored vs fresh                      ->  read against the floor

**What it licenses beyond deepseek.** [5129] makes the fleet's cells L1 retroactively. The floor is the number that says **whether a cell computed on the box and one computed on the Mac can sit in the same roster median at all.** Without it, every mixed-environment statistic in this registration is unquantified. It belongs in the execution block in its own right.

**The three-point test it completes.** With `fresh-fleet` faithful by construction (§6) and `fresh-local` space-stripped by construction, the deepseek verdict gains a row two points could not produce:

    stored ~= fresh-LOCAL    written space-stripped              CORRUPT
    stored ~= fresh-FLEET    written faithfully                  FINE
    stored ~= NEITHER        a third environment, uncharacterised  NEW FINDING

The third row would otherwise have been read as "fine" by elimination.

## 10. Scope, unchanged from v2

The frozen body's triplet is **CONFIRMATORY**. Every other triplet, and the pythia ladder, are **EXPLORATORY EXTENSION**, labelled at write time.

Producers: `scripts/twp_cloud.py`; spec `scripts/f11_twp_spec.py`; environment partition `scripts/f11_env_plan.py`; ingest `scripts/twp_ingest.py`.
