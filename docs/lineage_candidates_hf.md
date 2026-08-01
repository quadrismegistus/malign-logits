# Candidate independent pretraining lineages — HuggingFace survey

**STATUS: UNVERIFIED AGENT OUTPUT. NOT A ROSTER. NOTHING HERE HAS BEEN CHECKED
AGAINST A LIVE MODEL CARD BY A SEAT.**

Commissioned by RH 2026-08-01 to address the pseudo-replication defect in the
lineage count: `data/lineage_map.json` puts the roster at ~25–34 independent
lineages depending on setting, and H2's cross-lineage generality was not
distinguishable from chance at the principled count on the displacing stratum
(16 of 25, p = 0.11). More genuinely independent *pretraining runs* is the
correct remedy for that specific defect. It does **not** address ragged coverage,
which scales with the pool and is a separate problem.

## PROVENANCE AND WHY THIS FILE IS HEDGED

Produced by a Sonnet agent against a declared six-criterion inclusion rule (below).
Two things about that run must travel with the table:

1. **The agent fabricated one result and retracted it unprompted.** It reported a
   European/Indian/Arabic cluster as having "found Salamandra-7B, Teuken-7B,
   Jais-family-6.7B, EuroLLM-1.7B" before the sub-agent covering that region had
   reported anything. That sub-agent never reported. Those four names are
   **plausible leads from general knowledge, verified against nothing**.
2. **That entire region was therefore never searched.** OpenGPT-X/Teuken, Silo AI
   (Poro, Viking), Occiglot, Aleph Alpha, LightOn, Sarvam, Krutrim, Jais /
   Inception / Core42 / G42, MBZUAI, AI Sweden GPT-SW3, BSC Salamandra, EuroLLM —
   **uncovered, not checked-and-empty. The distinction matters and is the reason
   this file names it.**

**Every candidate below needs a live model-card check before entering the
registry.** Adding a Llama derivative as an "independent lineage" would corrupt
precisely the count this exercise exists to repair.

## THE INCLUSION RULE, AS GIVEN

1. Public **base** (pretrained-only) checkpoint on HuggingFace.
2. Public **aligned** checkpoint from the **same pretraining run**, same org.
3. **Independent pretraining** — not a continued-pretrain, distillation or
   fine-tune of anything already in the roster. Starting from Llama/Qwen/Mistral
   weights disqualifies however much further pretraining followed.
4. ≤ 9B parameters for the base.
5. `safetensors` available (not `.bin` only) — the torch-floor defect cost the v3
   grid 13 models.
6. English present in the pretraining mix.

Bonus, not required: separate SFT **and** DPO checkpoints, so the two steps can be
measured apart.

## QUALIFIED — 9, all agent-reported, none seat-verified

| lineage | base | aligned | params | notes |
|---|---|---|---|---|
| Gemma 2 (Google) | `google/gemma-2-9b` | `google/gemma-2-9b-it` | **9.24B** | over the ≤9B line by 2.7%; `gemma-2-2b` (2.6B) is a clean same-lineage fallback |
| RecurrentGemma (Google) | `google/recurrentgemma-9b` | `google/recurrentgemma-9b-it` | 9B or 10B — **unresolved**, paper says 9B, HF badge says 10B | **Griffin: gated linear recurrence + local attention. NON-TRANSFORMER** |
| Granite 3.1 (IBM) | `ibm-granite/granite-3.1-8b-base` | `ibm-granite/granite-3.1-8b-instruct` | 8.1B | card says "trained from scratch"; no separate DPO |
| Zamba2-7B (Zyphra) | `Zyphra/Zamba2-7B` | `Zyphra/Zamba2-7B-Instruct` | 7B | **hybrid Mamba2 + shared attention**; reuses Mistral *tokenizer only* |
| OpenELM (Apple) | `apple/OpenELM-3B` | `apple/OpenELM-3B-Instruct` | 3B | own 1.8T mix; layer-wise param scaling incoherent if init from fixed Llama weights; **no explicit "randomly initialized" statement found** |
| Kanana 1.5 (Kakao) | `kakaocorp/kanana-1.5-8b-base` | `kakaocorp/kanana-1.5-8b-instruct-2505` | 8B | paper: "8B from scratch, 2.7T tokens". **Do NOT also count `kanana-nano-2.1b` — pruned/distilled from this 8B, the tulu double-count shape** |
| Tanuki-8B (Matsuo Lab / GENIAC) | `hatakeyama-llm-team/Tanuki-8B` | `weblab-GENIAC/Tanuki-8B-dpo-v1.0` | 8B | card (JA): "フルスクラッチで約1.3Tトークン事前学習". **DPO explicit.** Base repo returned HTTP 401 twice — spot-check |
| LLM-jp-3 (NII) | `llm-jp/llm-jp-3-1.8b` or `-3.7b` | `…-instruct` | 1.8B / 3.7B | 2.1T from scratch; English ≈ 950B of it |
| 360Zhinao (Qihoo 360) | `qihoo360/360Zhinao-7B-Base` | `qihoo360/360Zhinao-7B-Chat-4K` | ~7.7B | own arch class + own tokenizer (vocab 158,464). **Independence rests on architectural/tokenizer evidence, not an explicit disclaimer** |

**Caveated:** BTLM-3B-8K (`cerebras/btlm-3b-8k-base` / `-chat`, 3B, DPO on
Anthropic-HH) — independence clean, **safetensors status unresolved and likely
failing**; the agent's live check was blocked by a 401.

**Near-miss, fixable by conversion:** Aquila2-7B (BAAI) — base is safetensors-clean
and independence decent, `AquilaChat2-7B` is `.bin`-only.

## REJECTED — with the criterion each failed

**Derived from a roster lineage (criterion 3):** Upstage SOLAR (Mistral-7B weights
+ depth-up-scaling, also 10.7B) · Rakuten AI 7B (Mistral-7B-v0.1) · ELYZA
(Llama-2/3) · Swallow (Llama continued-pretrain) · SKT A.X 4.0 (Qwen2.5) ·
NCSOFT VARCO (Llama 3.1) · CyberAgent Mistral-Nemo-JP (Mistral-NeMo-12B) ·
Nemotron-Mini-4B (pruned/distilled from Nemotron-4 15B) · Yuan2.0 (continued
pretrain of Yuan1.0) · Sakana (evolutionary merges / Llama continued-pretrain).

**`.bin` only (criterion 5):** XVERSE-7B · MPT-7B · XGen-7B-8K · OPT/OPT-IML ·
AquilaChat2-7B.

**No pretrain-only checkpoint published (criterion 1):** TeleChat/TeleChat2 ·
Command-R/Aya (Cohere) · LG AI EXAONE (all generations) · Naver HyperCLOVA X SEED.

**No aligned counterpart from the original authors (criterion 2):** Cerebras-GPT ·
pure Mamba / BlackMamba · PFN PLaMo-2-8B (independence otherwise clean).

**Too large (criterion 4):** Skywork (13B) · OrionStar (14B) · MiniMax (229B+) ·
Moonshot/Kimi (16B+) · StepFun (10B+) · Tencent Hunyuan · Jamba (52B) · DBRX
(132B) · Arctic (480B) · KT Mi:dm 2.0 (11.5B — **independence the most rigorously
confirmed of any candidate**, and the only sub-9B release is pruned from it).

## UNKNOWN — could not establish

- **SenseTime SenseNova U1-8B** — clears 5 of 6, but is a unified image+text model
  that "cannot function as a text-only causal LM" per its own docs. Needs a
  pipeline-level decision, not model-card research.
- **CyberAgent calm2-7b** — right size, base+aligned pair, but registers as
  `LlamaForCausalLM` and the README never says "from scratch". Resolvable by
  `config.json`'s `vocab_size` (far from 32,000 ⇒ independent init).
- **Granite 3.1's exact token budget** — the 10T+2T figure is documented for 3.0,
  not reverified for 3.1. Does not block; "trained from scratch" is on the 3.1 card.
- **SKT A.X 3.0** · **ETRI** · **iFlytek Spark** · **ByteDance Doubao** — unchased.
- **The entire European / Indian / Arabic region** — see provenance note above.

## WHAT ADDING THESE WOULD AND WOULD NOT DO

**Would:** take the principled lineage count from ~25 toward ~34, which is the
correct remedy for counting Falcon3's four sizes and the six tulu ablations as
independent implementations.

**Would not:** touch ragged coverage. Families are observed on largely disjoint
item sets — exactly one prompt is common to all 37 in the current roster — and
that gets *worse* on a larger pool, not better. Registration E's 5.6× substrate
is the remedy for that one. **Two defects, two remedies; more lineages fixes only
the first.**
