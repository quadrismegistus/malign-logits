# HF lineage candidates — survey v2

**Status: DRAFT, single pass, not cross-checked by a second agent.** Supersedes the informal v1 survey referenced in the brief (whose "known gap" region was never actually searched — its sub-agent didn't report and the parent fabricated results for that section). This document was produced by one agent in one session on 2026-08-07, working directly against live HuggingFace pages via WebFetch and WebSearch — no results in this file are recalled from training data without a citation attached, except where explicitly marked UNVERIFIED.

**Provenance rule used throughout:** VERIFIED means a tool call in this session returned a quote from the actual HF model card, README, or file tree, and that quote is reproduced below. UNVERIFIED means the claim is plausible from a search-result summary, secondary source, or general knowledge, but no direct-fetch quote backs it. Where a fetch failed (HTTP 401, timeout, etc.) this is stated rather than silently falling back to a guess. Nothing below is asserted as VERIFIED without a quote attached.

**Coverage honesty:** this is a single agent's single pass under a real tool-call budget. It is broad, not exhaustive. Section 6 lists what was explicitly not checked. Treat absence from this document as "not searched," not "searched and found nothing," unless a region is explicitly marked checked-and-empty.

---

## Verification key

- **VERIFIED** — direct-fetch quote in hand, reproduced below.
- **UNVERIFIED** — plausible, sourced to a search-result summary or general knowledge, no direct quote.
- **DISQUALIFIED** — fails one of the six rules; reason stated with its evidence tier.
- **AMBIGUOUS** — genuinely unclear; forcing a yes/no would misrepresent the evidence.

---

## Part 1 — The nine previously-flagged candidates, re-verified

### 1. Gemma 2 (Google) — `google/gemma-2-9b` / `-it`

**QUALIFIES, over the line (9.24B).** VERIFIED: card lists "9B params," and "The 9B model was trained with 8 trillion tokens" of web documents, code, and mathematics, "Primarily English-language content." Safetensors confirmed present (library tag + active download count). No statement of initialization from another model's weights was found — consistent with independent pretraining. `gemma-2-9b-it` (instruct) VERIFIED to have safetensors ("Tensor type: BF16," Safetensors tag), but the fetched card gives **no explicit SFT/RLHF/DPO breakdown** — post-training method is not detailed in the card text retrieved, so the bonus tier (separate SFT+DPO) is **UNVERIFIED from the card itself** despite being reasonably well known from the Gemma 2 technical report (not checked here). `gemma-2-2b` exists as a smaller same-lineage fallback per the brief; not independently re-verified this pass.

### 2. RecurrentGemma (Google) — `google/recurrentgemma-9b` / `-it`

**QUALIFIES, non-transformer, with one confirmed unresolved wrinkle and one new ambiguity.** VERIFIED: "RecurrentGemma is a family of open language models built on a novel recurrent architecture... Gated Linear Recurrences with Local Attention" (Griffin), i.e. genuinely non-transformer. Safetensors confirmed present as a tag. English confirmed: "available in English" / generates "English-language text."

**Param discrepancy CONFIRMED still present, unresolved:** the card's prose calls it the "9B" model throughout, but the HF metadata badge on the same page reads **"Model size: 10B params."** No reconciling statement was found. Same finding as the prior survey; I could not resolve it.

**New ambiguity flag (AMBIGUOUS on independence):** the card states **"RecurrentGemma uses the same training data and data processing as used by the Gemma model family."** The weights and architecture are independently initialized and trained (this is not weight-inheritance, so it does not fail rule 3 on its letter), but the pretraining *data recipe* is identical to Gemma 2's, which is already on our roster. Whether that counts as "independent pretraining" in the spirit the rule wants — i.e., does this teach us anything a fresh Gemma-2-shaped run wouldn't — is a judgment call, not a fact I can resolve by reading the card further. Flagging rather than forcing a verdict, per the brief's instruction 5.

Instruct variant (`recurrentgemma-9b-it`): safetensors confirmed present, param metadata again reads "10B params." No SFT/RLHF detail found in the fetched excerpt.

### 3. Granite 3.1 (IBM) — CORRECTION to the prior survey

**The prior survey's claim needs a fix.** `ibm-granite/granite-3.1-8b-base`'s own card does **not** say "trained from scratch." It says, VERIFIED: **"[Granite 3.1] extends the context length of Granite-3.0-8B-Base from 4K to 128K."** Granite 3.1 is a context-extension continuation of Granite 3.0, not itself the from-scratch checkpoint.

The actual from-scratch checkpoint is **`ibm-granite/granite-3.0-8b-base`**, separately VERIFIED: **"trained from scratch following a two-stage training strategy"** — Stage 1: "10 trillion tokens... web, code, academic sources, books, and math data," Stage 2: "2 trillion tokens... curated mix," 12T tokens total, "decoder-only dense transformer," 8.1B params. No mention of Llama/Mistral weight reuse — architecture terms only (GQA, RoPE, SwiGLU, RMSNorm).

**Recommendation:** cite `granite-3.0-8b-base` / `granite-3.0-8b-instruct` as the clean pair; treat `granite-3.1-8b-*` as a same-lineage, later, context-extended checkpoint rather than the primary citation.

Neither `granite-3.0-8b-instruct` nor `granite-3.1-8b-instruct` names DPO explicitly. Both VERIFIED to say the same thing: **"developed using a diverse set of techniques with a structured chat format, including supervised finetuning, model alignment using reinforcement learning, and model merging."** The 3.1 card additionally notes the specific preference-optimization method is "coming soon" in an unreleased technical report. **No bonus tier confirmed for either version.**

### 4. Zamba2-7B (Zyphra) — `Zyphra/Zamba2-7B` / `-Instruct`

**QUALIFIES.** VERIFIED: "hybrid model composed of state-space (Mamba) and transformer blocks," Mamba2 blocks + "two alternating shared attention blocks," LoRA projectors on shared MLP/attention. Tokenizer reuse confirmed and scoped precisely: **"Zamba2-7B uses the Mistral v0.1 tokenizer"** — tokenizer only, not weights. Pretraining independence VERIFIED: "pre-trained on 2T tokens of text and code data" from "open web-datasets, including Zyda," annealed on "~100B high-quality tokens." Safetensors confirmed present. English confirmed via the underlying open web-datasets. Instruct sibling VERIFIED: **"Zamba2-7B-Instruct is obtained from Zyphra/Zamba2-7B by fine-tuning on instruction-following and chat datasets"** — explicitly no DPO mentioned. **No bonus tier.**

**AMENDED 2026-08-10 from the load record. The qualification stands; "Safetensors confirmed present" does not hold for the base arm.** The verdict above was read against the card. Against the repos the two arms behave differently:

    Zyphra/Zamba2-7B            AutoTokenizer OSError, "trying to access a gated repo"
    Zyphra/Zamba2-7B-Instruct   LOADS

`data/model_load_environments.json` records the base arm as `load_failed` and notes in its own words that "the -Instruct sibling loads, so it is the repo, not the family." The published licence is Apache 2.0 and the family is not gated. **So this is a repo-level access state on one arm, not a rule-5 failure, and it deserves a retry before the pair is written off.** The general lesson: a card saying safetensors are present is evidence about the card. Whether an arm can actually be fetched is a separate fact that only a fetch establishes, and the two were conflated here.

### 5. OpenELM (Apple) — `apple/OpenELM-3B` / `-Instruct`

**Independence still unconfirmed by direct card language — same finding as before.** VERIFIED: "Our pre-training dataset contains RefinedWeb, deduplicated PILE, a subset of RedPajama, and a subset of Dolma v1.6, totaling approximately 1.8 trillion tokens." No explicit "randomly initialized" or "trained from scratch" statement was found in the fetched excerpt (matches the prior survey's finding exactly — I re-ran this check and got the same negative result). Safetensors tag present but not itemized at file level (not checked further). English presence not explicitly stated in what was fetched, though the constituent corpora are English-heavy by public convention — this is an inference, not a quote. Instruct card fetched separately: no training-method detail (SFT vs RLHF) found there either.

### 6. Kanana 1.5 (Kakao) — `kakaocorp/kanana-1.5-8b-base` / `-instruct-2505`

**Likely qualifies, weakly verified on independence language specifically.** VERIFIED: "8B params" in metadata; card describes it as a "bilingual language model" referencing MMLU and KMMLU (Korean) benchmarks; safetensors listed ("Model size: 8B params," "Tensor type: BF16," Safetensors tag). **The explicit "trained from scratch" phrase was not located in the fetched excerpt** — the summarizing tool inferred independence from absence of contrary evidence rather than quoting a positive statement, which is a weaker form of verification than the rest of this document uses; flagging that gap honestly. English presence not explicitly quoted either (inferred from the MMLU/KMMLU pairing). Relation to `kanana-nano-2.1b`: the fetched card for `kanana-nano-2.1b-base` does **not** explicitly describe it as pruned/distilled from the 8B specifically — it only says the broader Kanana pretraining program used "pruning and distillation" as general techniques. So the prior survey's caution ("do NOT also count kanana-nano... it is pruned/distilled from the 8B") is **not confirmed by direct quote either way** — worth a closer read of the Kanana technical report, not done here. Instruct sibling (`-instruct-2505`): safetensors listed, no SFT/DPO breakdown found ("a refined post-training process," no named method). **No bonus tier confirmed.**

### 7. Tanuki-8B (Matsuo Lab) — base repo still inaccessible

`hatakeyama-llm-team/Tanuki-8B` returned **HTTP 401 Unauthorized** on two separate fetch attempts this session (rendered page and raw README) — persists exactly as the prior survey found. Via a WebSearch summary (not a direct quote): "Tanuki-8B is a large-scale language model with about 8B parameters... pre-trained with about 1.3T tokens using full-scratch training... a Llama-3-8B similar architecture Japanese full-scratch LLM," GENIAC Matsuo Lab project. This is UNVERIFIED by my standard (search summary, not a fetched quote).

The DPO checkpoint `weblab-GENIAC/Tanuki-8B-dpo-v1.0` **is** directly accessible. VERIFIED: "8B params," safetensors present. Its raw README, when fetched directly, contained **"フルスクラッチで"** ("from scratch," applied to the base model description) and separately the sentence **"Tanuki-8x8B-dpo-v1.0は、SFTおよびDPOにより対話用に調整されています"** ("Tanuki-8x8B-dpo-v1.0 is tuned for dialogue via SFT and DPO") — note that second sentence names the **8x8B MoE sibling**, not the 8B model whose page I was reading. This is a real inconsistency in what the page returns, not a mistake on my part to paper over: **the SFT+DPO claim is textually attached to the 8x8B variant in the fetched content, while the "from scratch" claim is attached to the 8B.** I could not get a fetch that cleanly confirms "8B (not 8x8B) underwent SFT+DPO" in one quote. Org names differ (`hatakeyama-llm-team` for the base, `weblab-GENIAC` for the DPO checkpoint) but both sit inside the same GENIAC Matsuo Lab program per search summaries — same finding as before, still UNVERIFIED as a clean same-org claim in HF's own org-namespace sense.

**Net: spot-check inconclusive, same as the prior survey's caution predicted.** Someone with an authenticated HF session should pull the base card directly.

### 8. LLM-jp-3 (NII) — using `llm-jp-3-7.2b`, not 1.8b/3.7b

The brief's listed sizes (1.8b/3.7b) are not NII's current mid-tier flagship; `llm-jp-3-7.2b` is, and is the one I checked. **QUALIFIES, and this is the single best-verified bonus-tier case in the whole survey.**

VERIFIED: "7.2b" params (6,476,271,616 non-embedding, 32 layers, 4096 hidden). Pretraining independence: "pre-trained using a blend of the following datasets" — phrasing consistent with from-scratch training (not an explicit "from scratch" string, worth noting as a slightly softer form of the claim). English confirmed explicitly: "English" listed among pretraining languages with named datasets (Wikipedia, Dolma/CC-head, Dolma/C4), ~950B tokens total. Safetensors confirmed. Org: National Institute of Informatics (NII) / llm-jp.

Instruct sibling `llm-jp-3-7.2b-instruct3` VERIFIED with **explicit, separately-documented SFT and DPO stages**: **"We have fine-tuned the pre-trained checkpoint with supervised fine-tuning"** and **"and further aligned it with Direct Preference Optimization"** — the card gives each stage its own subsection with its own dataset list. **BONUS TIER CONFIRMED**, most cleanly of any candidate in this document.

### 9. Salamandra-7B (BSC) — see Part 2, it's also in the named-gap region

Covered in Part 2 under BSC, since the brief's gap-region list separately named it.

---

## Part 2 — The named "known gap" region

### AI Sweden — GPT-SW3

`AI-Sweden-Models/gpt-sw3-6.7b` / `gpt-sw3-6.7b-v2-instruct`. **QUALIFIES.**

VERIFIED base: "GPT-SW3 is a collection of large decoder-only pretrained transformer language models... developed by AI Sweden in collaboration with RISE and the WASP WARA for Media and Language," "trained on a dataset containing 320B tokens in Swedish, Norwegian, Danish, Icelandic, English, and programming code" (NeMo Megatron implementation — independent). English present but a minority language in the mix; rule 6 only requires presence, which is satisfied. Safetensors **VERIFIED at the file-tree level**: `model-00001-of-00003.safetensors` through `-00003-of-00003` (9.99GB + 9.99GB + 8.06GB), alongside redundant `.bin` copies of the same shards.

Instruct sibling VERIFIED: "The `instruct` models were finetrained on instruction data using both chat and raw text formats" — **no DPO mentioned, no bonus tier.** Safetensors confirmed present as a tag on the instruct card (not independently itemized at file-tree level for this specific repo).

### Silo AI — Poro and Viking (now under the `LumiOpen` HF org)

**Poro (original, 34B):** independently pretrained — "34B parameter decoder-only transformer pretrained on Finnish, English and code," "1 trillion tokens," BLOOM architecture with ALiBi — but **34B is far outside range**, well past even the 9-14B "over the line" bracket. Not pursued as a candidate on size grounds; no instruct-sibling check performed.

**Poro 2 (8B/70B): DISQUALIFIED.** Per search summary of the model card: "Poro 2 8B Base is an 8B parameter decoder-only transformer created through **continued pretraining of Llama 3.1 8B** to add Finnish language capabilities" (165B tokens). This is a clean rule-3 disqualification (UNVERIFIED tier — sourced to a search summary, not a direct fetch quote, but the phrasing is specific enough to trust provisionally; a direct-fetch confirmation is recommended before citing).

Worth flagging as a near-miss because Poro 2's post-training structure is exactly what the bonus tier wants and is unusually well documented: per the same summary, there are **three released stages** — "a base model, a post-training SFT-only checkpoint, and the final instruct model which is the SFT model plus a round of DPO." If a genuinely independent lineage with this same three-stage structure turns up, it would be a strong bonus-tier catch. Poro 2 itself just fails on independence.

**Viking-7B: DISQUALIFIED under rule 2 (no aligned sibling).** VERIFIED: "Viking 7B is a 7B parameter decoder-only transformer pretrained on Finnish, English, Swedish, Danish, Norwegian, Icelandic and code" (2T tokens), safetensors present. But VERIFIED directly from the card: **"Viking is a base model which needs further fine tuning for most use cases"** — no instruct/chat sibling was ever released under this name. Independently pretrained, base-only, disqualified on rule 2 alone.

### Occiglot — DISQUALIFIED

`occiglot/occiglot-7b-eu5` VERIFIED: **"Continued-pretraining from: Mistral-7B-v0.1."** Clean, explicit, rule-3 disqualification. 7B, English+Spanish+French+German+Italian+code. Not pursued further; the instruct variants inherit the same disqualification.

### Aleph Alpha — three lineages, three distinct problems

**(a) Pharia-1-LLM-7B (`control` / `control-aligned`)** — the strongest Aleph Alpha candidate, contingent on one unresolved point.

VERIFIED independence: **"After random initialization of all parameters, the model was trained to predict the next token in a sequence"**; own tokenizer — "vocabulary size 128000... trained via the Unigram algorithm... SentencePiece." 7,041,544,704 params. English confirmed dominant: "English: 2,970 (Billions) Tokens, 66.74%" of 7.7T total across languages.

VERIFIED alignment method, explicitly: **"`Pharia-1-LLM-7B-control-aligned` was aligned for helpfulness and safety using Direct Preference Optimization (DPO)."** This would be a clean bonus-tier pass — **except**: the primary `Pharia-1-LLM-7B-control` repo's file tree, VERIFIED directly, contains **no `.safetensors` files at all** — only 29 `.pt` layer-sharded files (`model_state_layer_N_*.pt`), 14.1GB total. Aleph-Alpha themselves publish a separate conversion repo, `Pharia-1-LLM-7B-control-hf`, described on its own page as **"the safetensors-conversion of `Pharia-1-LLM-7B-control`,"** and search results confirm a matching `Pharia-1-LLM-7B-control-aligned-hf` exists (its `tokenizer.json` blob URL was directly visible in search results). **I was not able to fetch and verify the aligned-hf repo's file tree directly** in the time available, so the safetensors claim for the *aligned* half of the pair is UNVERIFIED at the file-tree level, corroborated only by its existence and a plausible naming pattern. **Recommendation: qualifies via the org's own `-hf` conversion siblings, not the primary repos — verify `control-aligned-hf`'s file tree before relying on this pair.**

**AMENDED 2026-08-10. THE OPEN ITEM RESOLVES IN THIS PAIR'S FAVOUR AND THE PAIR IS STILL DISQUALIFIED, FOR A REASON NO RULE IN THIS SURVEY CHECKS.**

The safetensors contingency is settled: both `-hf` conversion repos load. `f11_l2_tokenizer_pairs.json` records the pair as **ID-SAFE, 0 mismatches over 76 probes**, and `model_load_environments.json` records both arms as LOADS. So the recommendation above is satisfied and the pair passes rules 1 through 6.

**It is nonetheless not a base-to-aligned pair.** Both public checkpoints are already post-trained: `control` is itself the SFT'd/instruction-tuned model and `control-aligned` adds DPO on top of it. No raw pretrained-only checkpoint was ever released. The edge is **SFT-to-DPO**, which measures something real but is not the contrast this roster is built to measure.

**This is a gap in the frame, not an oversight in this entry.** Rule 3 asks whether the base was pretrained independently of *another model's* weights, and Pharia passes: it was randomly initialized, as the quote above establishes. The unasked question is whether the base arm is at the *pretraining stage at all*. A survey can verify independence perfectly and still admit a pair whose base is two stages downstream.

Two consequences for the roster generally, both of which affect entries beyond this one:

- **Record `edge_type` per pair**, not just qualification: `BASE_TO_SFT`, `BASE_TO_DPO`, `BASE_TO_INSTRUCT`, `SFT_TO_DPO`, `SPANS_TWO_STAGES`.
- **The spanning cases are already in the roster and are currently indistinguishable from single-stage edges.** `LLM360/Amber -> AmberSafe` skips AmberChat; `EleutherAI/pythia-2.8b -> archangel_sft-dpo_pythia2-8b` crosses SFT and DPO together. Both are usable. Neither is a base-to-DPO edge, and a finding about "what alignment does" that pools them with single-stage edges is pooling two different operations.

Carried into the v3 sweep instruction as rule 7 (the base arm must be pretrained-only), with `edge_type` a required output field.

**(b) tfree-hat-pretrained-7b-base** — independently pretrained, architecturally the most novel thing in this whole survey, but currently pairless.

VERIFIED: **"We randomly initialized all model parameters"** — genuinely independent. Architecture: **"a hierarchical autoregressive transformer (HAT)... encoder, backbone, and decoder,"** character-level encode/decode integrated with word-level processing, **no traditional tokenizer** ("byte-level trigrams," per the org page). 7,192,507,136 params. English confirmed dominant: "English Language Data (70%)." Safetensors confirmed present.

**CORRECTED 2026-08-10. THE EARLIER "DISQUALIFIED under rule 2" VERDICT WAS WRONG, AND IT WAS WRONG BY READING A REPO NAME AS A LINEAGE CLAIM.**

It said: *"The only DPO'd sibling in this HAT family line is `Aleph-Alpha/llama-tfree-hat-pretrained-7b-dpo`, which — **per its name** — is the HAT architecture applied to Llama 3.1 8B weights."* The card's own metadata falsifies that. Verified directly against the HF API:

    Aleph-Alpha/llama-tfree-hat-pretrained-7b-dpo
      base_model = Aleph-Alpha/tfree-hat-pretrained-7b-base

Card body: *"The model was initialized from TFree-HAT-Pretrained-7B-Base."* Parameter counts match exactly (7,192,507,136 both), where the genuinely Llama-derived line carries a different count (7,192,495,104). **The `llama-` prefix denotes the Llama PROMPT TEMPLATE** (`add_llama_template=True` in the usage snippet), not Llama weights.

**So the independently-pretrained HAT base DOES have a same-lineage aligned partner**, and this is a 2-layer base+superego pair on a from-scratch pretrain: `tfree-hat-pretrained-7b-base` -> `llama-tfree-hat-pretrained-7b-dpo`. This is the same error class the campaign already recorded for `rwkv-raven-7b` in `scripts/f11_env_plan.py`: **a name match is not an architecture claim** — here, a name prefix taken as a lineage claim, in a document whose entire purpose is adjudicating lineage.

**AND IT IS STILL NOT USABLE, FOR A DIFFERENT AND HARDER REASON. The output space is 256 BYTES.** Verified in `config.json`:

    architectures        HATForCausalLM
    backbone_config      vocab_size = 0
    encoder_config       vocab_size = 256
    decoder_config       vocab_size = 256
    (no top-level vocab_size)

There is **no tokenizer file of any kind** in either repo — no `tokenizer.json`, no `tokenizer_config.json`, no vocab file. What ships instead is `splitter.py` with a `HATSplitter` class that does not subclass `PreTrainedTokenizer`, plus four other custom modules requiring `trust_remote_code=True`.

**This is a measurement incompatibility, not an inconvenience.** `get_base_logits()` returns a distribution over the model's vocabulary; every metric downstream of it — JS divergence, top-50 overlap, rank correlation, transgressive token mass, the 62-token vocabulary, single-word probes — presumes a shared subword simplex of ~32k-128k types. **A 256-way byte simplex is internally coherent and incommensurable with every other family on the roster.** There is no word-level unit to score, and the backbone has no vocabulary at all.

**VERDICT: the lineage objection is WITHDRAWN; a substrate objection replaces it.** This is a candidate for an explicitly byte-level side-study — where it would be genuinely interesting, since BLT already gives this project a byte-level instrument — and NOT a drop-in roster addition. It does not replace the Pharia entry, because it fixes Pharia's lineage problem by introducing a worse one.

**Also recorded, and separately useful:** `llama-3_1-8b-tfree-hat-base` -> `-sft` -> `-dpo` is a full three-stage public chain, exactly the topology this project prefers — and its `base_model` is `meta-llama/Meta-Llama-3.1-8B`, card: *"uses the Llama-3.1 8B base pre-trained checkpoint as initialization for the backbone."* **It collapses into the existing Llama-3.1-8B lineage alongside `llama`, `tulu` and `tulu-no-safety`, so it adds a checkpoint and no independent unit.** Correctly disqualified, on evidence this time rather than on its name.

**(c) Net for Aleph Alpha:** Pharia is the pair to cite, pending the one open safetensors check on the aligned-hf conversion.

### LightOn — searched, no qualifying pair found (checked-and-empty, not uncovered)

Their flagship independently-trained LLM is `lightonai/alfred-40b-0723` — 40B, far outside range. LightOnOCR models found in search are vision/OCR models, not relevant. No small (≤9B) independently-pretrained base+aligned text-LLM pair under LightOn's own org was located. This is a genuine "searched and found nothing," not a skipped region.

### Sarvam (India) — `sarvamai/sarvam-1`

**DISQUALIFIED under rule 2, independence itself weakly sourced.** VERIFIED: card describes it in prose as "2-billion parameter language model" but the page metadata separately reads **"Model size: 3B params"** — an internal inconsistency I could not resolve, worth flagging rather than picking one number silently. Pretraining: "Trained on a curated corpus of ~4 trillion tokens with 2 trillion high-quality Indic tokens" — the English portion of that 4T is implied by arithmetic (4T − 2T Indic ≈ 2T of something else, likely English, per general knowledge of this model) but **not itself directly quoted** as English in the fetched excerpt. Explicit "trained from scratch" language was **not found** in the fetched excerpt (inferred only). Safetensors present.

**Disqualifying fact, directly quoted:** **"This is a text-completion model. It is meant to be finetuned on downstream tasks, and cannot be used directly as a chat or an instruction-following model."** No aligned sibling from this same lineage was found — `sarvam-m` exists but is built on Mistral-Small (a different, Mistral-derived lineage, itself worth noting as a separate disqualification for the same reason as RakutenAI/Krutrim-2 below).

### Krutrim (Ola, India) — two generations, mixed

**Krutrim-1: plausible qualifier, incompletely verified.** The base card `krutrim-ai-labs/Krutrim-1-Base` returned **HTTP 401** on direct fetch. What I have instead is a fetch of the **instruct** card's own historical framing, VERIFIED there: a Release History table entry reading **"Trained from scratch"** against Krutrim-1-Base's 2024-01-31 release, and **"Krutrim-1-Instruct... SFT on Krutrim-1 Base,"** also released 2024-01-31. Param description on the instruct page: "7B parameter dense transformer model comparable [in size] to a similarly sized LLama-2 model" — this is a **size comparison, not a derivation claim**; I want to be explicit that "comparable to Llama-2" and "based on Llama-2" are different claims and the page uses the former. Safetensors present. **No DPO mentioned — no bonus tier.** Given the base card itself was inaccessible, I'm treating this as UNVERIFIED-leaning-plausible rather than fully VERIFIED; someone should retry the direct fetch.

**Krutrim-2: DISQUALIFIED.** Per search summary: "built on the Mistral-NeMo 12B architecture" — continued/adapted from Mistral, and 12B is also over the ≤9B line regardless. UNVERIFIED tier (search summary), but specific enough to trust provisionally.

### Jais / Inception / Core42 / MBZUAI

**`inceptionai/jais-family-6p7b` / `-6p7b-chat`. QUALIFIES — cleanest self-documented independence boundary found in this survey.**

VERIFIED, and notably the card does this disambiguation *for* the reader: **"Models pre-trained from scratch (`jais-family-*`)"** is explicitly distinguished, in the same document, from **"Jais adapted pre-trained (`jais-adapted-*`)"**, which the card itself says are **"built on top of Llama-2."** This is exactly the kind of same-org, same-namespace trap the brief warns about, and here the source itself flags it — worth noting as a model worth trusting on this specific point since it isn't asserting its own purity, it's naming its own non-independent sibling line.

6.7B params ("6.7B" in the specs table), safetensors **VERIFIED at file-tree level**: `model-00001-of-00003.safetensors` (9.9GB) + `-00002` (9.85GB) + `-00003` (8.82GB) + index json. Note the repo is gated (requires accepting terms). English confirmed: Arabic:English:Code ratio "1:2:0.4," 283B English tokens for this size. Org: "Inception, Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), and Cerebras Systems," explicitly named as a three-way partnership — this covers the brief's separate "MBZUAI" gap-region item as well, since MBZUAI is a named co-developer here.

Chat sibling VERIFIED safetensors present ("Tensor type: F32"); method described only as **"fine-tuned using Arabic and English prompt-response pairs in both single-turn and multi-turn settings"** — SFT-flavored language, no DPO named in the card text retrieved. **No bonus tier confirmed from the card** (the Jais technical report likely has more detail; not checked here).

**Same-org disqualification worth reporting:** `jais-adapted-*` models (e.g. `jais-adapted-7b`, `jais-adapted-70b`) are explicitly Llama-2-derived per the family card itself, and should not be substituted for the family-* line even though they share an org.

### BSC — Salamandra

`BSC-LT/salamandra-7b` / `-7b-instruct`. **QUALIFIES on the basic six; explicitly does NOT get the bonus tier, and is an interesting negative case.**

VERIFIED: "Total Parameters: 7,768,117,248," **"pre-trained from scratch on 12.875 trillion tokens,"** own 256k-vocabulary tokenizer (not reused from another model), RoPE/SwiGLU architecture. English: **"English represents the largest portion, accounting for 39.31% of the total data"** across 35 European languages + 92 programming languages. Safetensors confirmed present.

Instruct sibling exists and is SFT-tuned via FastChat, but the card is unusually candid about its own limits, VERIFIED: **"It has been optimized to engage in conversation but has NOT been aligned through RLHF to filter or avoid sensitive topics"** and **"This model is a first proof-of-concept designed to demonstrate the instruction-following capabilities of recently released base models."** So: qualifies as an "aligned checkpoint" in the loose instruction-tuned sense the basic six rule uses, but is explicitly self-described as *not* safety-aligned — potentially a useful negative/control case for a study of alignment displacement specifically, since the authors themselves flag it as unaligned.

### openGPT-X — Teuken

`openGPT-X/Teuken-7B-base-v0.6` / `-instruct-commercial-v0.4` (and other version/license variants exist: research vs. commercial, v0.4 vs v0.6). **QUALIFIES on the basic six; bonus tier unconfirmed; one internal inconsistency worth flagging.**

VERIFIED: "7B parameter multilingual large language model," "24 official European Union languages" (English among them, listed explicitly), safetensors present as a tag on both cards (not itemized at file-tree level — weaker verification point I did not close out). Custom tokenizer implied by `trust_remote_code=True` usage but ownership not explicitly stated. Developer consortium confirmed via search: "Fraunhofer Institutes IAIS and IIS, Jülich Research Center, German AI Association, TU Dresden, DFKI, IONOS, **Aleph Alpha**, ControlExpert and Westdeutscher Rundfunk" — note Aleph Alpha is a listed consortium member here too, a cross-link between two gap-region entries.

**Token-count inconsistency:** the base-v0.6 card says "pre-trained on 6 trillion tokens," while a separate search summary of the v0.4 instruct-commercial card says "pre-trained with 4T tokens" — these may simply be different training-run versions (v0.4 vs v0.6) rather than a real contradiction, but I did not reconcile the two directly. Post-training method: card describes it only as "an instruction-tuned version of Teuken-7B-base-v0.4" / "Instruction fined tuned version" — **no DPO named, no bonus tier confirmed.**

### EuroLLM — DISQUALIFIED

`utter-project/EuroLLM-1.7B`. VERIFIED independence: own dense Transformer, "trained on 4 trillion tokens," developed by a nine-institution EU consortium (Unbabel, IST, IT, Edinburgh, Aveni, Paris-Saclay, Amsterdam, NAVER Labs, Sorbonne), English confirmed present in a 35-language list. 1.657B params.

**DISQUALIFIED under rule 5, verified at the file-tree level:** `utter-project/EuroLLM-1.7B/tree/main` contains **only** `pytorch_model.bin` (3.31GB) — **no `.safetensors` file anywhere in the repo.** This is a clean catch the prior survey missed (it flagged EuroLLM as qualified without a file-tree check). I did not check whether the larger `EuroLLM-9B` sibling (if it exists) has safetensors — that's a genuinely uncovered follow-up, noted in Part 6.

### TII non-Falcon lines — NOT SEARCHED

I did not search for any TII-branded effort outside the Falcon family (e.g., a possible "NOOR" or similar). This is an explicit gap, not a checked-and-empty result — flagging per the brief's instruction 2.

---

## Part 3 — Unknown unknowns found this session

### Lucie-7B (LINAGORA / OpenLLM-France) — QUALIFIES, strong candidate

`OpenLLM-France/Lucie-7B` / `-Instruct`. VERIFIED independence: **"pre-trained on 512 H100 80GB GPUs for about 550,000 GPU hours"** with custom training code; **"exactly 6 706 958 336 free parameters."** Architecture note, VERIFIED and precisely scoped: **"Lucie-7B has the same neural network architecture as Llama3.1"** — this is architecture reuse, not weight inheritance, which is the actual line rule 3 draws (rule 3 disqualifies starting from Llama *weights*; copying Llama's published architecture and training your own weights from random initialization is a different, common, and non-disqualifying thing — several candidates in this document do it). Safetensors confirmed via metadata tag. English: 33.2%, French: 32.4%, German 6.9%, Spanish 6.6%, of a multilingual mix.

Instruct sibling exists (`Lucie-7B-Instruct`, also a v1.1). Training method per search summary: "fine-tuned on synthetic instructions produced by ChatGPT and Gemma" — **SFT-flavored, DPO not mentioned; I did not fetch the Instruct card directly to confirm this, so it's UNVERIFIED at the method-detail level** even though the pairing itself (base exists, instruct exists, same org) is solid.

### CroissantLLM (LINAGORA-adjacent multi-university consortium) — QUALIFIES

`croissantllm/CroissantLLMBase` / `CroissantLLMChat-v0.1`. VERIFIED: **"pretrained on a set of 3T English and French tokens,"** "a 1.3B language model," "1:1 English-to-French pretraining data ratio." Authors listed by name (Faysse, Fernandes, Guerreiro, et al. — an academic consortium, not a single lab). Chat sibling confirmed to exist and be the recommended entry point ("we recommend using the Chat version"), corresponding to "the checkpoint after 190k steps... and a final Chat finetuning phase."

**Two open items:** safetensors was **not explicitly confirmed** from either fetched card (I saw "Tensor type: F32" on the chat card but no unambiguous safetensors-file confirmation at the tree level — this needs a follow-up file-tree check I did not perform). Method for the chat variant beyond "finetuning phase" is unspecified — **no bonus tier confirmed.**

**RESOLVED 2026-08-10 from the measurement record. The first open item closes: it loads and it runs.** `f11_l2_tokenizer_pairs.json` records the pair ID-SAFE; `model_load_environments.json` records CroissantLLMBase as loading; and the twp store holds **103 scored prompts on each arm**, which settles the safetensors question by having used the weights rather than by reading a file tree. The second open item stands: the chat method is still unspecified, still no bonus tier.

**One usability defect that is not a rule failure and must not be recorded as one.** The load record notes: *"Loads; DELETES CJK CHARACTERS"*, with the observed example `他既是美丽的又是恶心的，她想要` rendering as `他是美的是心的想要`. That disqualifies the pair for the Chinese battery and **not** for English work. The 105-stem minimal-pair battery was checked directly against this: of 210 prompts, **zero contain CJK** and two contain non-ASCII at all, both the `é` in "fiancé". **Usability is a property of (model, battery), not of the model**, and a per-model verdict here would have dropped a usable pair from the English roster for a defect it cannot exhibit there.

### Bamba-9B (IBM Research + Princeton + UIUC) — currently disqualified, worth watching

`ibm-ai-platform/Bamba-9B-v2` (also under `ibm-fms`). VERIFIED: hybrid **Mamba2 + SwiGLU** architecture (non-transformer), **"trained from scratch"** per search summary (two-stage: 2T tokens from Dolma v1.7, then 200B curated tokens — this two-stage description came from a search summary; the direct card fetch did not itself contain the literal phrase "trained from scratch," so treat that specific wording as UNVERIFIED even though the broader independence claim is corroborated). 9.78B params — **right at/just over the ≤9B line**, would need the "over the line" flag if pursued. Safetensors present.

**DISQUALIFIED under rule 2 as of this survey**, directly quoted from the card: **"SFT - coming soon"** and **"DPO - coming soon - to be released in the next drop."** No aligned checkpoint exists yet. Flagging this because if/when IBM ships it, this would be a rare thing: a major-lab, non-transformer, independently-pretrained model with an explicit SFT-then-DPO bonus-tier pair. Worth rechecking in a few months.

### XVERSE-7B (Shenzhen Yuanxiang Technology) — DISQUALIFIED

`xverse/XVERSE-7B` / `-7B-Chat`. **DISQUALIFIED under rule 5, verified at file-tree level:** the repo contains only 8-way-sharded `pytorch_model-0000N-of-00008.bin` files (~15GB total) — **no `.safetensors` files present.** Independence is plausible per search summary ("independently developed by Shenzhen Yuanxiang Technology... 2.6 trillion tokens... 40+ languages including English") but this is UNVERIFIED (search summary, not a direct card quote) and moot given the safetensors disqualification.

### Skywork-13B (Kunlun Tech / SkyworkAI) — double disqualification

`Skywork/Skywork-13B-base` / presumably `-13B-Chat` (existence of the chat sibling confirmed via a requirements-section mention on the base card, not independently fetched). **DISQUALIFIED under rule 5, verified at file-tree level:** 53-way-sharded `.bin` files only (27.7GB total), **no safetensors.** Also **over the ≤9B line at 13B** (would qualify for the "over the line" 9-14B note if the safetensors issue weren't already disqualifying). English confirmed present: "52.2%" of a Chinese/English/code mix.

### Aquila2-7B/8B (BAAI) — inconclusive, flagged as UNVERIFIED rather than forced

`BAAI/Aquila2-7B` / `AquilaChat2-7B`. **Param count inconsistency I could not resolve:** search summaries call it "Aquila2-7B" (7B), but the fetched base-card metadata itself reads **"8B params."** Safetensors tag present on the base card. **"Trained from scratch" was not found verbatim** in either card excerpt fetched — this is a claim I have only from secondary search summaries ("trained from scratch on a high-quality corpus of Chinese and English, with Chinese corpora accounting for about 40%"), not from a direct quote of BAAI's own text. Post-training method (SFT vs DPO) for the chat variant was not found in the fetched excerpt either; the technical report (arXiv 2408.07410) is linked from the card but I did not read it. **Net: plausible candidate, genuinely under-verified — needs a closer pass before citing.**

### PhoGPT-4B / PhoGPT-7B5 (VinAI, Vietnam) — English presence unconfirmed, likely a problem

`vinai/PhoGPT-4B` (+ `-Chat`) and the earlier `PhoGPT-7B5` (+ `-Instruct`) generations. VERIFIED: **"pre-trained from scratch on a Vietnamese corpus of 102B tokens"** (4B generation), "exactly 3.7B parameters." Base+instruct pairs exist for both generations.

**Rule 6 concern, not resolved:** every description I fetched or found in search calls the pretraining corpus Vietnamese, full stop — **no English component was mentioned anywhere I checked.** This may simply mean I didn't dig deep enough (the underlying technical report likely has a fuller data breakdown I did not read), but as it stands, "English present in the pretraining mix" is **unconfirmed and plausibly false** for this lineage. Safetensors status also not confirmed from what I fetched. Flagging as a likely rule-6 failure rather than a clean qualifier, pending a closer look at the paper.

### InkubaLM-0.4B (Lelapa AI, South Africa) — small, thinly checked, likely missing its aligned half

`lelapa/InkubaLM-0.4B`. VERIFIED-via-search-summary (not a direct card fetch — I only reached this one through WebSearch, not WebFetch, so treat the whole entry as one tier softer than most of this document): **"trained from scratch using 2.4 billion total tokens,"** architecture "similar to MobileLLM," modeled on "the architecture design of LLaMA-7B" (again, architecture reuse, not weights — same pattern as Lucie). Covers isiZulu, Yoruba, Swahili, isiXhosa, Hausa, **plus English and French explicitly** — this is the cleanest rule-6 confirmation among the African/regional candidates I found, for what that's worth given the softer sourcing tier.

400M params, well under the ≤9B ceiling. An "Inkuba-instruct" **dataset** exists on HF, but I did **not** find a corresponding released aligned/instruct **model checkpoint** built on InkubaLM-0.4B in the time available. **Rule 2 is unconfirmed, likely a gap** — worth a direct look at Lelapa's HF org page (not done this session) before either qualifying or disqualifying this one.

### NorBLOOM-7b-scratch (Univ. of Oslo + HPLT + Nat'l Library of Norway + Univ. of Turku, `norallm` org) — disqualified on two fronts

VERIFIED: **"pretrained from scratch on a total of 260 billion subword tokens,"** "around 7 billion parameters," BLOOM-architecture reuse (again, architecture not weights). Safetensors listed.

**DISQUALIFIED, two reasons, both directly quoted:** (a) rule 2 — **"This is only a pretrained language model; an instruction-finetuned model will follow soon"** — no aligned sibling exists as of the card text fetched. (b) rule 6, likely — the described pretraining mix is Norwegian text plus **"20% of the 260B tokens... sampled from"** a Starcoder code corpus, with **no English corpus component mentioned** anywhere in what I fetched. I did not do an exhaustive search for a hidden English slice, so I'm calling this "likely fails rule 6" rather than a certain failure. Worth rechecking later both for the promised instruct release and for a fuller data breakdown.

### rinna (Japan) — one disqualified, one genuinely interesting PPO-aligned pair with an unconfirmed English gap

**`rinna/youri-7b`: DISQUALIFIED.** VERIFIED via search summary (own HF collection is literally titled **"llama-2-youri"**): "based on the llama-2 series and... continually pre-trained on Japanese-specific corpora." Clean continued-pretrain-of-Llama-2 disqualification.

**`rinna/japanese-gpt-neox-3.6b` family: independent, but I could not access the base card directly (HTTP 401, two attempts), and there's a real rule-6 concern.** This family has a genuinely rare structure: a from-scratch GPT-NeoX base, an SFT sibling (`-instruction-sft`), **and** a separately-released **PPO/RLHF sibling** (`-instruction-ppo`), which I fetched directly and VERIFIED: **"Following the OpenAI InstructGPT paper, Reinforcement Learning from Human Feedback (RLHF) has been applied to aligning the model's behaviour with input instructions"** and **"The second RL stage produces this model."** This is a real SFT→RL two-stage lineage, methodologically arguably *more* interesting than a DPO bonus pair since PPO/RLHF is rarer in open releases than DPO.

**But:** the base card itself was inaccessible both times I tried, so I have no direct quote on pretraining languages. Everything I can find via search describes this family as trained on "Japanese C4, Japanese CC-100 and Japanese Wikipedia" with **no English component named anywhere**. I'm calling this **UNVERIFIED-leaning-disqualified on rule 6** rather than a clean disqualification, specifically because I never got primary-source access — someone with an authenticated session should check directly before ruling it out for good, because if English does turn out to be present even as a minor component, this PPO pair would be a genuinely valuable addition.

### MOSS (Fudan / OpenMOSS) — DISQUALIFIED

Per search summary: **"The base language model is initialized with CodeGen-16B"** (Salesforce's checkpoint). Not a continuation of anything already on our roster specifically, but disqualified on the same principle rule 3 encodes: it starts from someone else's pretrained weights, not random initialization. UNVERIFIED tier (search summary), but specific and consistent across multiple independent search hits.

### RakutenAI-7B (Rakuten, Japan) — DISQUALIFIED

Per search summary: **"leverages the Mistral model architecture and is based on Mistral-7B-v0.1 pre-trained checkpoint... this model was not pretrained from scratch."** Clean rule-3 disqualification. No independent Rakuten lineage was found in this session.

### Latam-GPT (regional Latin America/Caribbean consortium, 75+ institutions) — DISQUALIFIED

Per search summary: **"built on top of Llama 3.1 70B and adapted through continued pretraining (CPT)"** with regional data, plus SFT for instruction-following. Clean rule-3 disqualification, and 70B is far outside range regardless. Notable as the one clear Latin American entry found — worth reporting as a disqualification rather than silence, since "we looked and it's Llama-based" is a different, more useful finding than "we didn't look."

### Tele-FLM (China Telecom / TeleAI, `CofeAI` org) — out of range, not pursued

52B params (plus a 1T-parameter MoE sibling). Far outside ≤9B or even the 9-14B "over the line" bracket. Independence plausible (own architecture, "GPT-4-style" instruct-data-in-pretraining approach per search summary) but not worth chasing further given the size mismatch; not independently verified.

### Naver HyperCLOVA X SEED (Korea) — entirely unverified, structurally unclear, worth a direct look

Org page (via search only, not fetched directly) lists `HyperCLOVAX-SEED-Text-Instruct-0.5B`, `-1.5B`, a `-Vision-Instruct-3B` whose search summary says it **"was developed starting from the HyperCLOVAX-SEED-Text-Base-3B"** (implying a 3B **text base** checkpoint exists, at a size not matching either of the released instruct sizes), and an `-Omni-8B` multimodal model. **Nothing here was fetched directly** — independence, safetensors, English presence, and even which base pairs with which instruct size are all unconfirmed. This is a promising-looking but completely unverified lead; flagging it precisely so it doesn't get silently dropped, not because I have any evidence for it beyond a search-result summary.

### Mi:dm 2.0 (KT / K-intelligence, Korea) — not pursued past initial search, likely over the line and structurally unclear

11.5B params (per search summary), already over even the "over the line" bracket. The only HF repo name found combines base and instruct in one name — `Midm-2.0-Base-Instruct` — which may mean no separate non-instruct checkpoint is released at all; not confirmed either way, not fetched directly. English+Korean pretraining mix confirmed via search summary only. Not pursued further given the size mismatch.

---

## Part 4 — Disqualifications at a glance

| Lineage | Org | Reason | Rule | Evidence tier |
|---|---|---|---|---|
| Occiglot-7B | Occiglot collective | Continued-pretrain of Mistral-7B-v0.1 | 3 | VERIFIED (direct quote) |
| RakutenAI-7B | Rakuten | Continued-pretrain of Mistral-7B-v0.1 | 3 | UNVERIFIED (search) |
| youri-7b | rinna | Continued-pretrain of Llama-2 | 3 | UNVERIFIED (search, corroborated by rinna's own collection name) |
| Poro 2 (8B/70B) | LumiOpen/Silo AI | Continued-pretrain of Llama 3.1 8B | 3 | UNVERIFIED (search) |
| Latam-GPT | 75+-institution LatAm consortium | Continued-pretrain of Llama 3.1 70B | 3 | UNVERIFIED (search) |
| Krutrim-2 | Ola/Krutrim | Built on Mistral-NeMo 12B | 3 | UNVERIFIED (search) |
| MOSS | Fudan/OpenMOSS | Base initialized from CodeGen-16B | 3 | UNVERIFIED (search) |
| jais-adapted-* | Inception | Built on top of Llama-2 (per the family's own card) | 3 | VERIFIED (direct quote, self-disclosed) |
| llama-tfree-hat-*-dpo | Aleph Alpha | HAT architecture on Llama-3.1-8B weights | 3 | VERIFIED (org page + collection title) |
| EuroLLM-1.7B | utter-project | No `.safetensors` in repo, `.bin` only | 5 | VERIFIED (file tree) |
| XVERSE-7B | XVERSE/Shenzhen Yuanxiang | No `.safetensors` in repo, `.bin` only | 5 | VERIFIED (file tree) |
| Skywork-13B-base | Kunlun Tech/SkyworkAI | No `.safetensors`, `.bin` only; also 13B | 5 (+ size) | VERIFIED (file tree) |
| Viking-7B | LumiOpen/Silo AI | No aligned sibling ever released | 2 | VERIFIED (direct quote) |
| Sarvam-1 | Sarvam AI | Explicitly not usable as chat model; no aligned sibling found | 2 | VERIFIED (direct quote) |
| tfree-hat-pretrained-7b-base | Aleph Alpha | Independent, but its only DPO sibling is the Llama-derived line, not this one | 2 | VERIFIED (org page checked) |
| Bamba-9B | IBM/Princeton/UIUC | SFT and DPO both explicitly "coming soon," not yet released | 2 | VERIFIED (direct quote) |
| NorBLOOM-7b-scratch | Univ. Oslo/HPLT/Turku | No aligned sibling yet; likely no English in pretraining mix | 2 (+ likely 6) | VERIFIED (direct quote) |

---

## Part 5 — Ambiguous cases (not forced to yes/no)

- **RecurrentGemma vs. Gemma 2**: independently-initialized non-transformer weights, but explicitly trained on **"the same training data and data processing as used by the Gemma model family."** Passes rule 3 on the letter (no weight inheritance); whether identical-data-recipe pretraining counts as "independent" in spirit is a judgment call I'm not making for you.
- **Lucie-7B / NorBLOOM-7b-scratch / InkubaLM-0.4B architecture reuse**: all three explicitly reuse a *published architecture* (Llama 3.1's, BLOOM's, LLaMA-7B's respectively) while training their *own weights* from random initialization. I've treated this as passing rule 3, since the rule's own text disqualifies starting from "Llama/Qwen/Mistral/Gemma **weights**," not their published architectures — but if the intended bar is stricter than that (i.e., architecture-copying itself is suspect), all three should be revisited.
- **Aquila2-7B vs -8B naming**: two independent sources give two different parameter counts for what appears to be the same model family entry point. Not resolved.
- **Sarvam-1 "2B" vs "3B"**: same kind of internal inconsistency, prose vs. metadata badge on the same page, not resolved.

---

## Part 6 — What was not covered (explicit gaps, not silent absences)

- **TII non-Falcon lines**: not searched at all this session.
- **Krutrim-1-Base primary card**: 401'd twice; relied on the instruct card's secondhand framing instead.
- **`hatakeyama-llm-team/Tanuki-8B` primary card**: 401'd twice, same as the prior survey found.
- **`rinna/japanese-gpt-neox-3.6b` primary card**: 401'd twice; the rule-6 (English) question for this lineage is unresolved as a result.
- **EuroLLM-9B**: not checked for safetensors; only the 1.7B was checked (and disqualified).
- **CroissantLLM file tree**: safetensors presence not confirmed at file level for either the base or chat repo.
- **Lucie-7B-Instruct card**: not fetched directly; SFT-vs-DPO claim is secondhand.
- **Naver HyperCLOVA X SEED**: no card fetched at all, only search summaries; flagged as a lead, not a finding.
- **Aquila2 technical report** (arXiv 2408.07410): not read; would likely resolve the param-count and "from scratch" questions.
- **PhoGPT technical report/paper**: not read; would likely resolve the English-presence question.
- **Aleph-Alpha `Pharia-1-LLM-7B-control-aligned-hf` file tree**: existence corroborated by search, contents not directly verified.
- **Broader unknown-unknown regions not touched at all**: no search was run for South/Southeast Asian labs beyond India/Vietnam (e.g., Thailand, Indonesia, Philippines, Malaysia); no search for additional Chinese labs beyond the handful checked (Skywork, XVERSE, Aquila, MOSS, Tele-FLM) — there are certainly more (e.g., 01.AI's non-Yi efforts, Baichuan's smaller checkpoints, Moonshot/Kimi, MiniMax, StepFun); no search for Middle Eastern labs beyond Jais/Inception; no search for Israeli labs (e.g., AI21's smaller-than-Jamba efforts, if any exist); no search for Russian labs (e.g., Yandex YaLM, Sber GigaChat); no second pass on the "smaller Western labs" angle the brief suggested (e.g., Snowflake Arctic, Databricks DBRX-adjacent smaller runs, Together's own from-scratch lines beyond RedPajama, Cerebras-GPT, MPT beyond the one search below).
- **MosaicML MPT-7B**: found and plausible (base+instruct+chat, "pretrained from scratch on 1T tokens of English text and code," own ALiBi-based architecture) but **not verified at all this session** — both direct file-tree fetches (`mosaicml/mpt-7b/tree/main`, `rinna` base) 401'd or were not attempted before time ran out; only a WebSearch summary is in hand. Given MPT-7B is a 2023-era release, the safetensors question (rule 5) is a real risk — many mid-2023 HF repos predate the safetensors default and never got converted. **This is the single most promising fully-unverified lead in this document** and should be the first thing checked in a follow-up pass.
  - **CLOSED 2026-08-10, and not on rule 5.** The follow-up ran and the repo id does not resolve at all: `mosaicml/mpt-7b` returns `OSError: mosaicml/mpt-7b is not a local folder and is not a valid model identifier`, recorded in `f11_l2_tokenizer_pairs.json` as UNAVAILABLE. The safetensors risk flagged above was the right worry about the wrong failure. **Recorded as rule 1 (public weights), not rule 5**, and it should come out of `data/base_aligned_pairs.json`, where it is still carried as a pair. If a live successor id exists under another org it has not been found; that search has not been run and is not claimed here.
- **NX-AI xLSTM-7b**: independently pretrained non-transformer, VERIFIED via search to have **no instruct/aligned sibling** ("NX-AI's xLSTM-7b lacks an instruct model variant... the official model available... is the base pre-trained model, not a specialized instruct or chat variant") — disqualified under rule 2, same pattern as Viking-7B and Bamba-9B, but this one was only reached via search, not a direct card fetch, so treat the disqualification itself as UNVERIFIED tier even though it's a negative result (absence of a sibling is harder to falsely confirm via search than presence would be, but I want to flag the tier honestly rather than let a disqualification look more solid than a qualification would at the same evidence level).

---

## Part 7 — Ranked shortlist (my read, not a committee decision)

**Ready to add now, well-verified:**
1. **Jais-family-6.7B** (Inception/MBZUAI/Cerebras) — cleanest independence documentation in the survey, safetensors verified at file level, gated but accessible.
2. **LLM-jp-3-7.2b** (NII) — best-verified bonus-tier (explicit separate SFT+DPO) case found.
3. **Lucie-7B** (LINAGORA/OpenLLM-France) — clean independence, safetensors confirmed, instruct sibling exists (method detail not yet confirmed).
4. **GPT-SW3-6.7b** (AI Sweden) — safetensors verified at file level, base+instruct pair solid, no bonus tier.
5. **Salamandra-7b** (BSC) — qualifies, notable as a self-declared *unaligned* instruct model — useful as a negative case.
6. **Teuken-7B** (openGPT-X) — qualifies on the basic six, one unresolved token-count inconsistency between versions.

**Worth one more verification pass before deciding:**
7. **Pharia-1-LLM-7B** (Aleph Alpha) — explicit DPO confirmed, the best bonus-tier candidate by alignment-method clarity, contingent entirely on the `-control-aligned-hf` safetensors conversion actually existing and being complete.
8. **CroissantLLM** (multi-university) — qualifies on independence and pairing, safetensors unconfirmed.
9. **Krutrim-1** (Ola) — plausible, base card inaccessible, relying on secondhand framing.
10. **MPT-7B** (MosaicML) — never actually fetched this session, flagged as the top unverified lead, safetensors is a real risk given the 2023 release date.

**Disqualified but worth remembering, because they'd otherwise resurface as false leads:**
Occiglot, RakutenAI-7B, youri-7b, Poro 2, Latam-GPT, Krutrim-2, MOSS, jais-adapted-*, the Llama-derived tfree-hat line, EuroLLM-1.7B, XVERSE-7B, Skywork-13B, Viking-7B, Sarvam-1, the independent tfree-hat-pretrained-7b-base (pairless), Bamba-9B (pairless, for now), NorBLOOM-7b-scratch (pairless and likely no English), xLSTM-7b (pairless).
