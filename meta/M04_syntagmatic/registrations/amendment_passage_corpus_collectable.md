# PRE-RUN AMENDMENT — what is collectable, and why 45 of 46

**This amends the frozen spec; it does not break it.** The frozen population remains **46 pairs** under the standing hashes. This document declares which of them this run collects, before any box exists, with every cause cited to a source.

    SPEC        meta/M04_syntagmatic/registrations/spec_passage_corpus_105.md
                FROZEN at e541f6a4, sha256_16 27b1369efdd9dc0e
    POPULATION  data/forced_arms_46reps_drmatch.json  89eb642b50d00dd9
                pairs 46   8567e0ee993b457b     <- UNCHANGED by this amendment
                prompts 208 27484f7ade774b77
                cells 8,169 aa5389c1420c7f76
                matched     723e81b3946b6d56

    COLLECTABLE THIS RUN 40 pairs   (see §1, §1b, §1c, §1d)
    NOT COLLECTABLE       6 pairs   Zamba2 (cost), RWKV (no implementation),
                                    Olmo-Hybrid (broken released loader),
                                    Pharia (no implementation), Aquila2
                                    (support REMOVED after v0.24.0),
                                    Baichuan2 (REMOVED after v0.23.0)
    COLLECTED             final count reconciled at end of run against
                          THE MANIFESTS AND LANDED FILES, never against
                          the absence of failure records ([5573]: boxes
                          delete FAILED.jsonl on restart, so a cleared
                          record and a success look identical from
                          outside; only a non-empty output file
                          distinguishes them). Run in flight as of this
                          edit; six pairs have failed at least once with
                          fixes applied and unproven.

Same discipline as the spec's existing named deviations (46-not-52, 208-not-212): the population is what it is, and what was not collected is named with its reason rather than quietly absent.

---

## 1. THE ONE EXCLUSION — `Zyphra/Zamba2-7B > Zamba2-7B-Instruct`

**EXCLUDED FROM THIS RUN ON COST, NOT ON CAPABILITY.**

Three named requirements, each individually satisfiable, **never yet tried together**:

    transformers==4.57.1    docs/cloud_runbook.md:301 — the vLLM image ships
                            5.14.1, on which the base arm fails at load with a
                            v5 weight-tying validation its config predates
                            (it declares transformers_version 4.49.0.dev0).
                            4.57 is also the OLMo 3 floor, so one pin serves both.
    mamba-ssm +             scripts/f11_env_plan.py:42 — Zamba2 is in KERNELS.
    causal-conv1d           Measured 19.3x on the sibling architecture.
    base-arm repo access    docs/lineage_candidates_hf_v2.md:52 — `AutoTokenizer`
                            OSError, "trying to access a gated repo", against a
                            published Apache-2.0 licence on an ungated family.
                            That document's own verdict: **"a repo-level access
                            state on one arm, not a rule-5 failure, and it
                            deserves a retry before the pair is written off."**

**The aligned arm is demonstrated.** `docs/local_capability.md:70` records `Zamba2-7B-Instruct` producing cells locally on 8 Aug. Only the base arm carries the access state.

**The cost fact.** Satisfying all three means a dedicated environment box for **1 pair, 8,960 sequences, 0.7% of the corpus**, against a **$21 total contingency** (vast credit $125 against a $104 run). `scripts/build_fleet.py`'s own header records a fleet that "pulled 15 GB of Zamba2 and failed on a kernel it did not have".

**This amendment does not say the pair cannot be collected.** It says it is not collected at this price, and it records the full recipe so that when spending resumes the pair is a known small purchase and not an archaeology project.

## 1b. A SECOND EXCLUSION, FOUND BY RUNNING RATHER THAN BY READING

**`RWKV/rwkv-4-7b-pile > rwkv-raven-7b` — EXCLUDED, CURRENTLY UNRUNNABLE.** Added 2026-08-12 during the run, on RH's word; heading retightened to RH's phrase per [5573]: "currently unrunnable" is a property of a DATE (vLLM 0.27.1, August 2026), where "incompatible" would read as a property of the pair. The upstream routes (#11193, closed unmerged, RWKV6-not-4; a small fork) are far off but exist.

    vLLM 0.27.1: "Model architectures ['RwkvForCausalLM'] are not supported
    for now." A ValidationError at ModelConfig, before any weights load.

Not a dtype, a pin, a card or a tokenizer. **vLLM has no RWKV implementation**, so this pair cannot be collected by this runner at any price — the first exclusion in this document that is a capability fact rather than a cost one. `docs/local_capability.md` records both arms running fine on MPS under plain transformers, which is the whole distinction: the checkpoints work, the ENGINE cannot host them.

**No preflight would have caught it.** `model_load_environments.json` records RWKV as loading (it does, under transformers), the fidelity guard passes (the tokenizer is fine), and `build_fleet` routed it correctly. The incompatibility exists only between this architecture and this engine, and nothing in the campaign's records is keyed that way. It surfaced as a `FAILED.jsonl` row on box4 forty minutes into the run.

**Collected population is therefore 44 pairs, not 45.** Wilcoxon P 45 -> 44; SEs scale by sqrt(45/44); position profile ~2.98 -> ~2.95, four-term ~2.44 -> ~2.41. Both still clear.

    RECORDED FOR WHOEVER REVISITS IT: the pair is collectable by the
    HuggingFace path (`generate()` rather than vLLM), which this runner does
    not implement. That is a different instrument, not a bigger box.

## 1c. A THIRD EXCLUSION — `allenai/Olmo-Hybrid-7B > Olmo-Hybrid-Instruct-DPO-7B`

**EXCLUDED, CURRENTLY UNRUNNABLE — of the opposite kind.** Added 2026-08-12 during the run, on malign's [5571] root-cause; wording by the registrar per that post's request; heading retightened to RH's phrase per [5573]. Both §1b and §1c are facts about vLLM 0.27.1 in August 2026, not about the pairs: `main` may already carry this loader fix, and the next release could make this pair a five-minute purchase.

Where RWKV has no implementation in any vLLM release, Olmo-Hybrid has an implementation whose **released weight loader is broken**:

    vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py:517
        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            dim = dims[loaded_shard_id]        <- dims[None]
        TypeError: list indices must be integers or slices, not NoneType

`utils.py:281 _load_param` calls the loader with no shard id, so the default `None` indexes a list — firing for any checkpoint whose conv1d weights ship fused rather than split. Support was merged (PR #32550, `olmo_hybrid.py` is in the install); the defect is in the release we run, **0.27.1, which is PyPI's newest** — the fix, if it exists, lives only on `main`. Waiting does not help and upgrading is not available.

**Confounds eliminated:** the identical traceback reproduced on a verified-clean card (0 MiB used) at `gpu_memory_utilization=0.35` with `enforce_eager=True`. The first failure was confounded with an orphaned engine; the second was not. Not memory, not dtype, not eager mode.

**The same preflight blindness as §1b, from the opposite cause.** Both arms are recorded working — 33,300 + 34,600 rows in `beam_fc`, 430,100 + 380,526 in `twp_words` — and every one of those rows came from a transformers-based runner. (ARCHITECTURE x ENGINE) is a fact class the campaign's records do not key; runbook §2.21 now names it.

**THE PATCH THAT WAS NOT APPLIED, RECORDED AS A DECISION.** The fix is three lines (treat `loaded_shard_id is None` as already-fused) and was declined: one pair of the corpus produced by a modified inference engine, inside a corpus whose value is that every other pair went through an identical one, is a provenance cost that outlives the convenience ([5571] §4, registrar concurring). RH holds the deciding word and has not asked for it.

    RECORDED FOR WHOEVER REVISITS IT: collectable by the HF path
    (`generate()` rather than vLLM), where both arms are demonstrated; or
    by any vLLM release that ships the loader fix. A different instrument
    or a later engine — not a bigger box, and not this run.

## 1d. THREE MORE, AND TWO ARE A NEW FAILURE CLASS — SUPPORT THAT WAS REMOVED

Added 2026-08-12 during the run, malign's [5576]; wording by the registrar. All three CURRENTLY UNRUNNABLE per RH's phrasing, each with its distinct relation to the engine:

    Aleph-Alpha/Pharia-1-LLM-7B-control-hf > -aligned-hf
        "Model architectures ['PhariaForCausalLM'] are not supported for
        now" -- never was. RWKV's class: no implementation in any release.

    BAAI/Aquila2-7B > AquilaChat2-7B
        "supported in vLLM until v0.24.0, and is not supported anymore"

    baichuan-inc/Baichuan2-7B-Base > Baichuan2-7B-Chat
        "supported in vLLM until v0.23.0, and is not supported anymore"

**Aquila2 and Baichuan2 are a class this document did not have: an implementation that EXISTED and was DELETED.** The engine's own error names the last working version, which turns the loss into a specification: **one box, one pinned vLLM (<= 0.23.0 collects both), two pairs.** Cheaper than Zamba2's recipe and better evidenced -- the recipe came from the engine itself. Not run now ([5576]: a fourth environment mid-run, against a stable fleet, under the $21 contingency); recorded so that when spending resumes it is a known small purchase.

Five of the six exclusions are (architecture x engine) facts -- the class [5571] §5 flagged as unkeyed by any campaign record. It cost 11% of the declared population in one run. Runbook §2.21 carries it.

## 1e. WHAT THE RUN ITSELF SETTLED — ONE RECOVERED, TWO MORE LOST, AND THE COUNT §1d USES IS OPTIMISTIC

Added 2026-08-12 by malign, from the run's own artifacts rather than from prediction. **Every claim here is a row count or a recorded traceback, not an expectation.**

**`BAAI/Aquila2-7B > AquilaChat2-7B` — COLLECTED.** §1d's specification was correct and was executed: a contemporary image (`vllm/vllm-openai:v0.22.1`) on an Ampere card, profiles `vllm022` / `vllm022_a100`.

    remote  1,782 rows   200,426,507 bytes
    local   1,782 rows   200,426,507 bytes    byte-exact
    expect  1,796 (2 x 898 arm-cells)      -> 99.2%, complete

Two refinements §1d could not have known. **Pinning the engine is not enough — the IMAGE must be contemporary**: installing `vllm==0.22.1` onto a 0.27.1 image fails in ABI order through four separate extensions and generation still dies. And **Turing will not do**: on a Q RTX 8000 the model loads and generates at small batch, then dies inside FlashInfer's paged-KV prefill, and `VLLM_ATTENTION_BACKEND` is ignored in 0.22.1, so no backend flag rescues it. `baichuan-inc/Baichuan2-7B-Base` is running on the same box under the same recipe at the time of writing; it is NOT claimed here.

**`openGPT-X/Teuken-7B-base-v0.6 > -instruct-v0.6` — A SEVENTH EXCLUSION, and the second of its exact kind:**

    IndexError: OUT_OF_RANGE: piece id is out of range
      return self._processor.IdToPiece(id)      identical to m-a-p/CT-LLM-Base

Both are the cross-scoring vocabulary guard reading the wrong bound: it drops a sequence at `max(full_ids) >= model_config.get_vocab_size()`, while the crash is in the TOKENIZER's sentencepiece count. The real bound is `min(model_vocab, len(tokenizer))`. **Not repaired mid-run** — it would split the scoring contract between pairs collected before and after, which is worse than two named absences. It is the cheapest item on the post-run list and it would return two pairs.

**`deepseek-llm-7b-base > -chat` — §2's prediction was NOT borne out in this run.** §2 argues from the `beam_fc` corpus that the pair is collectable and that "the corpus outranks the record". That argument was about the TOKENIZER and it still stands. The pair nonetheless failed here, in the DETOKENIZER, at passage length:

    TypeError: 'NoneType' object is not iterable
      for ch in raw_token:

**A pair being collectable in one stack is not a claim that it collects in another** — the same scoping the section applies to the record applies to its own conclusion. `inceptionai/jais-family-6p7b` failed alongside it, and its message corrects a detail carried in my earlier posts: **JAISLMHeadModel was removed after v0.22.0, not v0.24.0, so the `vllm022` rescue image at v0.22.1 is already too late for it.** Recovering jais needs v0.22.0 exactly; that is a different box from the one that recovered Aquila.

**THE COUNT.** §1d and §3 are written at P=40 against six exclusions. **The run has eight, and the two additions were never in that arithmetic:**

    absent, each with a recorded traceback   Pharia, RWKV, Zamba2, Olmo-Hybrid,
                                             deepseek, jais, CT-LLM, Teuken
    collected and verified                   33
    still running                             5   Baichuan2, recurrentgemma,
                                                  kanana, Falcon3-Mamba,
                                                  falcon-mamba

    P if all five land       38     3.01 x sqrt(38/46) = ~2.74
    P as it stands now       33     3.01 x sqrt(33/46) = ~2.55
    §3 currently assumes     40                          ~2.81

**The best available case is now BELOW the ~2.8 bar on which the fifth arm was restored, and it moves further below with every pair that does not land.** This is the registrar's number to restate, not mine to rewrite; §3 is left as written so the change is visible rather than absorbed. Flagged here because the direction is downward, and a correction that reduces a headline is the one least likely to be noticed on its own.

**POWER AT P=40, THE NUMBER TO LOOK AT HARDEST ([5576] §2):** position profile 3.01 x sqrt(40/46) = **~2.81, AT the ~2.8 bar rather than above it** -- and the fifth arm was restored on that margin. Four-term ~2.30, effect-limited regardless. SEs scale by sqrt(46/40). Two facts belong beside that number before anyone re-litigates the fifth arm: (a) the ~2.8 bar was itself an estimate, not a cliff; (b) **the pinned-vLLM box above moves P to 42 (~2.88) for a single small spend** -- the cheapest power in the campaign if the margin worries RH more than the contingency does. RH's call, cool-off rules apply.

Both were single-source reads of `data/model_load_environments.json`, and in both cases a second source disagreed. Recorded because the amendment should carry its own error rate.

**`deepseek-llm-7b-base > -chat` — COLLECTED.** The record says it "loads and runs and silently destroys the prompt", every space deleted on encode, fix none known. **The corpus refutes it**: 47,000 sequences in `beam_fc`, and the discriminating statistic is prompt token length, not output spaces (the model generates its own spaces whatever the prompt did).

    tokens-per-char, beam_fc, shared prompts, 23 models
      roster range        0.205 – 0.214
      deepseek base/chat  0.2073 / 0.2073   — 8th and 9th, at the median

A prompt with every space deleted tokenises far shorter. It is not an outlier in the direction the defect predicts. **The record is not wrong, it is SCOPED** — it names transformers 5.4.0 / tokenizers 0.22.2, and `beam_fc` ran a different stack. CLAUDE.md's standing rule covers exactly this: *the corpus outranks the record.*

**`gemma-2-9b > -it` — COLLECTED, WITH A PROFILE REQUIREMENT.** vLLM refuses gemma2 at float16 as a hard validation error, and `scripts/vllm_y_run.py` hardcodes `dtype="float16"`. That is a routing constraint, not a block:

    needs bf16 and therefore an sm_80+ card — the record's own words are that
    it "belongs on the bf16 roster with Falcon-H1"
    corroborated: scripts/vllm_provision.sh:138 "model_info OK | download OK"
    demonstrated at passage length: 1,700 sequences per arm in corpus `y`

**If it is misrouted it fails loudly at load**, which is the acceptable failure mode. The fleet plan must place this pair on a bf16 box.

## 2b. TWO PAIRS COLLECTED UNDER A DECLARED TOKENIZER OVERRIDE

`deepseek-llm-7b-base > -chat` and `internlm2-base-7b > -chat-7b` are collected **under `malign_logits/twp.py`'s `LOADER_OVERRIDE`**, which forces `PreTrainedTokenizerFast` past the transformers-v5 #45488 regression (a SentencePiece Metaspace pre-tokenizer installed over the ByteLevel one the repo declares; every space vanishes, `unk_token: null`, nothing raises). internlm2 fails the same class as a boundary shift rather than a deletion.

**The runner now imports that table rather than copying it, and runs `assert_prompt_survives` per checkpoint before any weights are downloaded.** Every checkpoint's verdict is recorded in the run artifacts as `fidelity`: `pass`, `pass_under_override`, or `refused`. A refused pair writes a `.REFUSED.json` naming itself and stops — a named absence, never a silent one.

**This makes the exclusion question empirical per box instead of predicted from a document.** The guard runs for all 90 collected checkpoints, not the four in the override table, which is what turns the 20 never-observed checkpoints from unknown risk into a per-box measurement at no marginal cost.

## 3. POWER, RESTATED AT P=45 SO NOBODY MEETS IT AT ANALYSIS TIME

    Wilcoxon P              46 -> 45
    standard errors         scale by sqrt(46/45)
    position profile        3.01 -> ~2.98   (clears the ~2.8 bar it was bought for)
    four-term decomposition 2.47 -> ~2.44   (effect-limited regardless)

Both still clear. The fifth arm was restored on the position profile's clearing, and it still clears.

**Restated again at P=43 (§1b + §1c):** position profile ~2.91 (still clears the ~2.8 bar); four-term ~2.39 (effect-limited regardless); SEs scale by sqrt(46/43). If the end-of-run reconciliation finds further losses, this section is restated once more at the final count rather than met at analysis time.

## 4. THE UNTESTED FRACTION, CORRECTED

An earlier draft of this reasoning said 66 of 92 checkpoints have no observation. True of `model_load_environments.json` and **misleading about risk**, because the corpus is not the record:

    roster checkpoints appearing in beam_fc   60      1,971,600 sequences
    ...in y                                   62        108,800
    ...in f11_l2                              50        197,000
    with NO generation anywhere               20

**72 of 92 have demonstrably produced generations.** The unknown fraction is 20 checkpoints, not 66.

## 5. THE PATTERN THIS AMENDMENT EXISTS TO NOT REPEAT

Three proposed exclusions, three single-source reads, three second sources that disagreed: a version-scoped observation read as a checkpoint property (deepseek), a dtype requirement read as a failure (gemma-2), an MPS statement plus a repo access state read as an architecture verdict (Zamba2). **The preflight this campaign calls non-negotiable consults one file.** All three were caught by RH asking, twice, whether the documents had actually been read.

**The difference between "cannot be collected" and "not collected at this price" is the difference between closing a door and noting where the key hangs** (@registrar, [5547]).

**One number in this document was wrong until the producer computed it.** Drafts quoted Zamba2 as 17,360 sequences and 1.3% of the corpus. That was the two-pair BLOCK bucket from an earlier post, carried after deepseek left it — a derived value going stale in exactly the way this campaign spent the week cataloguing. The pair is **8,960 sequences, 0.7%**, computed from the table by `data/passage_corpus_pairs_collected.json`'s producer. The cost argument is unchanged and slightly stronger.

---

Drafted by the malign seat on registrar's [5547] terms; countersigned by the registrar. §1b added mid-run on RH's word; §1c added mid-run on malign's [5571] root-cause, wording by the registrar as that post requested. Final collected count to be reconciled and appended at end of run.
