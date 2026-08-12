# PRE-RUN AMENDMENT — what is collectable, and why 45 of 46

**This amends the frozen spec; it does not break it.** The frozen population remains **46 pairs** under the standing hashes. This document declares which of them this run collects, before any box exists, with every cause cited to a source.

    SPEC        meta/M04_syntagmatic/registrations/spec_passage_corpus_105.md
                FROZEN at e541f6a4, sha256_16 27b1369efdd9dc0e
    POPULATION  data/forced_arms_46reps_drmatch.json  89eb642b50d00dd9
                pairs 46   8567e0ee993b457b     <- UNCHANGED by this amendment
                prompts 208 27484f7ade774b77
                cells 8,169 aa5389c1420c7f76
                matched     723e81b3946b6d56

    COLLECTED THIS RUN   44 pairs   (see §1, §1b)
    NOT COLLECTED         2 pairs   Zamba2 (cost), RWKV (engine)

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

**`RWKV/rwkv-4-7b-pile > rwkv-raven-7b` — EXCLUDED, ENGINE INCOMPATIBILITY.** Added 2026-08-12 during the run, on RH's word.

    vLLM 0.27.1: "Model architectures ['RwkvForCausalLM'] are not supported
    for now." A ValidationError at ModelConfig, before any weights load.

Not a dtype, a pin, a card or a tokenizer. **vLLM has no RWKV implementation**, so this pair cannot be collected by this runner at any price — the first exclusion in this document that is a capability fact rather than a cost one. `docs/local_capability.md` records both arms running fine on MPS under plain transformers, which is the whole distinction: the checkpoints work, the ENGINE cannot host them.

**No preflight would have caught it.** `model_load_environments.json` records RWKV as loading (it does, under transformers), the fidelity guard passes (the tokenizer is fine), and `build_fleet` routed it correctly. The incompatibility exists only between this architecture and this engine, and nothing in the campaign's records is keyed that way. It surfaced as a `FAILED.jsonl` row on box4 forty minutes into the run.

**Collected population is therefore 44 pairs, not 45.** Wilcoxon P 45 -> 44; SEs scale by sqrt(45/44); position profile ~2.98 -> ~2.95, four-term ~2.44 -> ~2.41. Both still clear.

    RECORDED FOR WHOEVER REVISITS IT: the pair is collectable by the
    HuggingFace path (`generate()` rather than vLLM), which this runner does
    not implement. That is a different instrument, not a bigger box.

## 2. TWO PAIRS I PROPOSED EXCLUDING AND WAS WRONG ABOUT

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

Drafted by the malign seat on registrar's [5547] terms. Countersignature and docket id to be appended by the registrar.
