"""
malign-logits
=============

A toolkit for psychoanalytic analysis of LLM probability distributions.

Compares base (id), SFT (ego), DPO (superego), and optionally RLVR
(reinforced superego) checkpoints from the same model family to map
repression, displacement, and condensation signatures of AI alignment.

Quick start (OO interface)::

    from malign_logits import Psyche
    psyche = Psyche.from_pretrained()

    s = psyche.analyze("He lay naked in his bed and")
    s.repression          # DataFrame of ego-superego deltas
    s.id_scores           # drive-weighted repression scores
    s.analysis_df         # full combined DataFrame

Model families::

    from malign_logits import Psyche
    psyche = Psyche.from_family("llama")  # 2-layer
    psyche = Psyche.from_family("olmo")   # 4-layer (default)
"""

__version__ = "0.2.0"

# Centralized stdlib / third-party imports used across modules.
import math
import os
import platform
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from tqdm import tqdm

import warnings
# Silence the noisy HF/torch chatter this project triggers on MPS, but do NOT
# blanket-ignore everything — a bare filterwarnings("ignore") at import hid the
# package's own DeprecationWarnings from every consumer.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass

PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
PATH_DATA = os.path.join(PATH_REPO, "data")
PATH_DATA_RAW = os.path.join(PATH_DATA, "raw")
PATH_STASH = os.path.join(PATH_DATA_RAW, "stash")  # deprecated: use cache.get_cache()
PATH_FIGURES = os.path.join(PATH_REPO, "figures")


# ---------------------------------------------------------------------------
# Model family registry
# ---------------------------------------------------------------------------

@dataclass
class ModelFamily:
    """A model family with checkpoints at each training stage.

    The main path is linear: base → ego (SFT) → superego (DPO) → reinforced_superego (RLVR).
    Reasoning branches off the base (or a specified base) as an alternative post-training path.
    """
    name: str
    base: str                              # primary process (always required)
    ego: str | None = None                 # SFT checkpoint
    superego: str | None = None            # DPO or instruct-as-superego
    reinforced_superego: str | None = None # RLVR
    # Reasoning branch
    reasoning: str | None = None           # reasoning model (R1-distill, native thinking)
    reasoning_base: str | None = None      # base it branches from (if cross-family distillation)
    thinking_mode: bool = False            # supports /think toggle on instruct model
    #: {slot -> git revision}. Only for checkpoints whose DEFAULT BRANCH IS
    #: WRONG for this pair -- not a general provenance field. BAAI replaced
    #: Aquila2-7B's main branch with a re-tokenised model and left the chat arm
    #: alone, so main-vs-main spans two vocabularies and is dimensionally
    #: undefined. **NOT YET HONOURED ON EVERY LOADING PATH** -- see
    #: TODO_TRACKS.md; `load_model` takes `revision=`, but `Psyche.from_family`
    #: does not thread it and the twp/depth/L2 runners call `from_pretrained`
    #: directly. Read `revisions` explicitly wherever a family is loaded until
    #: that is closed.
    revisions: dict | None = None

    @property
    def n_layers(self):
        return sum(1 for x in [self.base, self.ego, self.superego, self.reinforced_superego] if x is not None)

    @property
    def has_reasoning(self):
        return self.reasoning is not None or self.thinking_mode

    @property
    def all_checkpoints(self):
        """All model IDs in this family, including reasoning."""
        ids = [self.base]
        for x in [self.ego, self.superego, self.reinforced_superego, self.reasoning]:
            if x is not None:
                ids.append(x)
        return ids


MODEL_FAMILIES = {
    "olmo": ModelFamily(
        name="OLMo 3 7B",
        base="allenai/Olmo-3-1025-7B",
        ego="allenai/Olmo-3-7B-Instruct-SFT",
        superego="allenai/Olmo-3-7B-Instruct-DPO",
        reinforced_superego="allenai/Olmo-3-7B-Instruct",
    ),
    "olmo-think": ModelFamily(
        name="OLMo 3 7B Think",
        base="allenai/Olmo-3-1025-7B",
        ego="allenai/Olmo-3-7B-Think-SFT",
        superego="allenai/Olmo-3-7B-Think-DPO",
    ),
    "olmo-tiny": ModelFamily(
        name="OLMo 2 1B",
        base="allenai/OLMo-2-0425-1B",
        ego="allenai/OLMo-2-0425-1B-SFT",
        superego="allenai/OLMo-2-0425-1B-DPO",
        reinforced_superego="allenai/OLMo-2-0425-1B-Instruct",
    ),
    "smol": ModelFamily(
        name="SmolLM2 360M",
        base="HuggingFaceTB/SmolLM2-360M",
        superego="HuggingFaceTB/SmolLM2-360M-Instruct",
    ),
    "qwen-tiny": ModelFamily(
        name="Qwen 2.5 0.5B",
        base="Qwen/Qwen2.5-0.5B",
        superego="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    "llama": ModelFamily(
        name="Llama 3.1 8B",
        base="meta-llama/Llama-3.1-8B",
        superego="meta-llama/Llama-3.1-8B-Instruct",
        reasoning="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ),
    "amber": ModelFamily(
        name="Amber",
        base="LLM360/Amber",
        ego="LLM360/AmberChat",
        superego="LLM360/AmberSafe",
    ),
    # Beaver: PPO against PKU-SafeRLHF cost+reward judges (same judges we score with in F37).
    # Same safety curriculum as Amber (PKU-SafeRLHF), different route: PPO vs DPO.
    # Base is unofficial LLaMA-1 7B mirror (huggyllama).
    "beaver": ModelFamily(
        name="Beaver 7B",
        base="huggyllama/llama-7b",
        ego="PKU-Alignment/alpaca-7b-reproduced",
        superego="PKU-Alignment/beaver-7b-v1.0",
    ),
    "qwen": ModelFamily(
        name="Qwen 2.5 7B",
        base="Qwen/Qwen2.5-7B",
        superego="Qwen/Qwen2.5-7B-Instruct",
        reasoning="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ),
    "tulu": ModelFamily(
        name="Tulu 3.1 8B",
        base="meta-llama/Llama-3.1-8B",
        ego="allenai/Llama-3.1-Tulu-3-8B-SFT",
        superego="allenai/Llama-3.1-Tulu-3-8B-DPO",
        reinforced_superego="allenai/Llama-3.1-Tulu-3.1-8B",
    ),
    # The safety-data ablation as a FULL pipeline: same base, same DPO, only the
    # SFT differs. Distinct from tulu-sft-* below, which are 2-layer families
    # that treat each SFT variant as a terminal stage; this one keeps the ego
    # position so base->ego->superego stays comparable to `tulu`.
    #
    # Restored 2026-07-27. `tulu`'s ego had been pointing at the no-safety
    # checkpoint and this key was absent entirely, so the registry no longer
    # reproduced its own data: battery_results.csv holds 47 rows under
    # `tulu-no-safety` that no registered family could regenerate, and
    # `--family tulu` returned the ablation arm under the standard name. The
    # data was never wrong -- the two arms' base->superego metrics are
    # identical because base and superego ARE identical, which is the control
    # working, not a duplicate.

    # ────────────────────────────────────────────────────────────────────
    # INDEPENDENT PRETRAINING LINEAGES, survey v2 (docs/lineage_candidates_hf_v2.md,
    # 2026-08-07). Registered on RH's word; **registration is not verification.**
    #
    # WHY THESE EXIST AS A BLOCK. The roster's ~34 lineages are dominated by a
    # handful of orgs, and the pooled damage MDE is bounded by n_pairs rather
    # than by sites — so more INDEPENDENT PRETRAINING RUNS is the only lever on
    # it. Every entry below is a base + aligned pair from one pretraining run,
    # not a continued-pretrain of anything already on the roster.
    #
    # TIER IS RECORDED IN THE NAME, because a registry row looks equally
    # authoritative whatever its evidence:
    #   [v2 READY]  direct-fetch quote in hand for independence and safetensors
    #   [v2 VERIFY] qualifies but one named check is open — see the survey
    #   [v2 AMBIG]  independence is a judgement call, not a fact
    # 2-layer throughout (base + superego) per the house convention, EXCEPT
    # llm-jp-3 which has separately documented SFT and DPO stages.
    "jais": ModelFamily(
        name="Jais family 6.7B (Inception/MBZUAI/Cerebras) [v2 READY]",
        base="inceptionai/jais-family-6p7b",
        superego="inceptionai/jais-family-6p7b-chat",
    ),
    # **2-LAYER, NOT 3.** The survey calls this the cleanest bonus tier because
    # the CARD documents SFT and DPO as separate stages with their own dataset
    # lists. That is documentation of a PROCESS, not two downloadable
    # checkpoints — `instruct3` is the single post-SFT-and-DPO model. Declaring
    # it in both ego and superego made the builder chain a self-edge
    # (dpo_of instruct3 -> instruct3) and `base_of` then resolved the model to
    # itself. Caught by checking that the new pairs resolve, not by reading.
    #
    # The documented two-stage process is still worth having — it is why this
    # lineage is worth more than a plain base/instruct pair — but it is a fact
    # about the card, not a rung we can measure apart.
    "llm-jp-3": ModelFamily(
        name="LLM-jp-3 7.2B (NII) [v2 READY, SFT+DPO documented but one checkpoint]",
        base="llm-jp/llm-jp-3-7.2b",
        superego="llm-jp/llm-jp-3-7.2b-instruct3",
    ),
    "lucie": ModelFamily(
        name="Lucie-7B (LINAGORA/OpenLLM-France) [v2 READY]",
        base="OpenLLM-France/Lucie-7B",
        #: **`Lucie-7B-Instruct` DOES NOT EXIST** -- RepositoryNotFoundError, not
        #: a gated 401. The real aligned checkpoint is `-Instruct-v1.1`, and the
        #: name was registered on 7 Aug by pattern-matching every other family's
        #: `base + "-Instruct"` rather than from the model card. The twp fleet
        #: then ran the base, failed the aligned half, and reported ALL MODELS
        #: COMPLETE -- so the lineage looked delivered while carrying one arm.
        #: Corrected on RH's word. A registered name is a claim about a repo and
        #: has to be checked against the repo, not against the sibling families.
        superego="OpenLLM-France/Lucie-7B-Instruct-v1.1",
    ),
    "gpt-sw3": ModelFamily(
        name="GPT-SW3 6.7B (AI Sweden) [v2 READY]",
        base="AI-Sweden-Models/gpt-sw3-6.7b",
        superego="AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct",
    ),
    # SELF-DECLARED NOT RLHF-ALIGNED: "has NOT been aligned through RLHF to
    # filter or avoid sensitive topics". Kept for exactly that reason — an
    # instruction-tuned model whose authors deny it is safety-aligned is a
    # control this roster does not otherwise have.
    "salamandra": ModelFamily(
        name="Salamandra 7B (BSC) [v2 READY, self-declared unaligned]",
        base="BSC-LT/salamandra-7b",
        superego="BSC-LT/salamandra-7b-instruct",
    ),
    "teuken": ModelFamily(
        name="Teuken 7B (openGPT-X) [v2 READY]",
        base="openGPT-X/Teuken-7B-base-v0.6",
        superego="openGPT-X/Teuken-7B-instruct-commercial-v0.4",
    ),
    "gemma2": ModelFamily(
        name="Gemma 2 9B (Google) [v2 READY, 9.24B — over the 9B line]",
        base="google/gemma-2-9b",
        superego="google/gemma-2-9b-it",
    ),
    # 3.0 NOT 3.1. The v1 survey named 3.1; its own card says it "extends the
    # context length of Granite-3.0-8B-Base". 3.0 is the from-scratch run.
    "granite": ModelFamily(
        name="Granite 3.0 8B (IBM) [v2 READY]",
        base="ibm-granite/granite-3.0-8b-base",
        superego="ibm-granite/granite-3.0-8b-instruct",
    ),
    # DPO named explicitly on the card. The primary repos ship .pt shards only;
    # these are the org's own safetensors conversions, and the -aligned-hf tree
    # was NOT verified — that is the open check.
    "pharia": ModelFamily(
        name="Pharia-1-LLM-7B (Aleph Alpha) [v2 VERIFY: -aligned-hf safetensors]",
        base="Aleph-Alpha/Pharia-1-LLM-7B-control-hf",
        superego="Aleph-Alpha/Pharia-1-LLM-7B-control-aligned-hf",
    ),
    "croissant": ModelFamily(
        name="CroissantLLM 1.3B [v2 VERIFY: safetensors unconfirmed]",
        base="croissantllm/CroissantLLMBase",
        superego="croissantllm/CroissantLLMChat-v0.1",
    ),
    # TOP UNVERIFIED LEAD: never fetched in the survey. A 2023 repo, so
    # safetensors is a real risk — many predate the default and were never
    # converted.
    "mpt": ModelFamily(
        name="MPT-7B (MosaicML) [v2 VERIFY: never fetched, safetensors at risk]",
        base="mosaicml/mpt-7b",
        superego="mosaicml/mpt-7b-instruct",
    ),
    # CLOUD ONLY. Mamba2 blocks need causal-conv1d + mamba-ssm, which have no
    # Metal build; the fallback path allocates the full state.
    # ADDED 2026-08-10 from the v2 lineage survey's unregistered residue. Both
    # preflighted against the HF API before declaring: safetensors, own
    # tokenizer, ungated. Neither needs a revision pin.
    # KANANA 2 — the better-documented of the two Kakao lines, and a SEPARATE
    # pretraining run from 1.5, so a separate lineage rather than the same one
    # counted twice. Card, verbatim: "Kanana-2-3B was **pretrained from scratch
    # on TPU clusters** and further improved through post-training with
    # **supervised fine-tuning and reinforcement learning**", and the
    # frontmatter declares `base_model: kakaocorp/kanana-2-3b-base`. So both
    # the independence and the base->aligned derivation are stated rather than
    # inferred, which is more than the 1.5 card gives.
    #
    # `Qwen3ForCausalLM` is ARCHITECTURE REUSE, NOT WEIGHT REUSE: vocab is
    # 128,256 against Qwen3's 151,936, the card's five Qwen mentions are all
    # benchmark rows, and there is no "initialized from" or "continued
    # pretraining" anywhere in it. Same distinction as Aquila2 sharing
    # Llama-7B's dimensions.
    #
    # **DO NOT ADD kanana-2-1.3b OR -0.9b AS LINEAGES.** The card: "Kanana-2-1.3B
    # models are derived from Kanana-2-3B through a cascade pruning and
    # distillation pipeline." They are compressions of this run, not runs. A
    # future expansion that grabs "all the kanana sizes" would double-count
    # exactly the way this campaign has already had to correct (59 -> 39 -> 34).
    "kanana2": ModelFamily(
        name="Kanana 2 3B (Kakao) [from scratch; SFT + RL declared]",
        base="kakaocorp/kanana-2-3b-base",
        superego="kakaocorp/kanana-2-3b-instruct",
    ),
    # KAKAO CONTRIBUTES TWO LINEAGES, NOT THREE, AND `kanana` IS NOT A
    # FROM-SCRATCH PRETRAIN. Recorded here because the two omissions below are
    # the kind that get "discovered" later and added as new lineages.
    #
    #     Kanana 1.0 8B run  ->  kanana-1.5-8b-base    continued pretraining
    #                        ->  kanana-nano-2.1b-base pruned + distilled
    #     kanana-2-3b-base       SEPARATE, "pretrained from scratch on TPU"
    #
    # `kanana-1.5-8b-base` is the 1.0 8B plus continued training, not its own
    # run: tech.kakao.com/posts/707, "총 100B 토큰의 비교적 대규모 추가 학습"
    # ("a relatively large-scale ADDITIONAL training of 100B tokens").
    #
    # **`kanana-nano-2.1b-base` IS DELIBERATELY NOT REGISTERED.** The 1.0
    # technical report (arXiv:2502.18934) attributes it to the 2.1B by name in
    # three places -- §2.3 "we derive Kanana Nano 2.1B model through pruning and
    # distillation from the 8B model"; §2.3.3 "Leveraging the 8B model ... we
    # efficiently produce smaller models"; Table 6 "Each model is pruned from
    # the preceding model. Each model is distilled using the 8B model as the
    # teacher" (8B -> 4.5B -> 2.1B -> 1.3B -> ...).
    #
    # **THE TRAP, IF ANYONE REVISITS THIS.** Table 5 also reports a FROM-SCRATCH
    # 2.1B at 3T tokens -- an ablation baseline Kakao never shipped. The
    # released card's benchmarks (54.83/44.80/77.09/31.10/46.20/46.32) match the
    # pruned-and-distilled 0.3T row digit for digit and mismatch the
    # from-scratch row. Quote the wrong row and the same paper argues for
    # independence.
    #
    # It was considered for a within-org DPO-vs-PPO contrast -- nano is
    # DPO-aligned, 1.5 is on-policy RL -- and rejected: that comparison differs
    # in SCALE (2.1B vs 8B), COMPRESSION (pruned+distilled vs not) and
    # PRETRAINING (+100B continued) as well as in objective, so it isolates
    # nothing. A pair deduplicated out of every lineage statistic, supporting a
    # contrast confounded four ways, is not worth registering.
    #
    # POST-TRAINING METHOD DOCUMENTED, BUT NOT ON THE CARD AND NOT AS DPO.
    # The card names no method; two Kakao engineering posts do, in Korean.
    # posts/707 lists 1.5's three changes as on-policy RL, generative reward
    # model, and verifiable reward functions; posts/716 names them RLVR and
    # RLGRM applied after SFT. **Kakao ABANDONED DPO for this line** -- the DPO
    # in those posts describes the 1.0 recipe they moved away from. Staged as
    # `ppo` in the registry via METHOD_DECLARED; see the comment there for why
    # that is a bucket rather than a named algorithm.
    # (superseded note: the card mentions no DPO, RLHF,
    # SFT, preference or GRPO stage anywhere (zero hits, checked 2026-08-10).
    # It sits in `superego` on the roster's existing convention for 2-layer
    # instruct arms -- the same slot as Llama-Instruct, Qwen-Instruct, gemma-it
    # and Yi-Chat, whose derived `stage=dpo` is a property of the SLOT and not a
    # claim about any of them. Recorded so nobody later reads the slot as
    # evidence of the method.
    "kanana": ModelFamily(
        name="Kanana 1.5 8B (Kakao) [v2 QUALIFIES; Korean/En; method undisclosed]",
        base="kakaocorp/kanana-1.5-8b-base",
        superego="kakaocorp/kanana-1.5-8b-instruct-2505",
    ),
    # THE SURVEY'S "INACCESSIBLE" VERDICT WAS STALE, NOT WRONG WHEN WRITTEN:
    # it checked `hatakeyama-llm-team/Tanuki-8B`, which 404s. The repos moved
    # orgs, and THE PAIR SPANS TWO OF THEM — base under team-hatakeyama-phase2,
    # DPO under weblab-GENIAC. The `-GGUF` repo RH found is a quantised
    # llama.cpp conversion with no tokenizer and no safetensors; it names the
    # real checkpoint in its own `base_model` field. Quantised weights are
    # refused here on the project's own precision rule, so the GGUF is not a
    # fallback.
    "tanuki": ModelFamily(
        name="Tanuki-8B (Matsuo Lab / GENIAC) [v2 REVIVED; Japanese/English]",
        base="team-hatakeyama-phase2/Tanuki-8B-base-v1.0",
        superego="weblab-GENIAC/Tanuki-8B-dpo-v1.0",
    ),
    # BAAI Aquila2 — independent lineage (own 100k BPE vocabulary, own corpus,
    # own framework; a 100,008-token embedding matrix mechanically rules out
    # inheriting Llama/Mistral/Qwen weights).
    #
    # **THE ALIGNED ARM IS SFT, NOT DPO.** The technical report (arXiv
    # 2408.07410) says "we collected instructional data to train the chat
    # version" and calls it "the supervised fine-tuned model" throughout §7.1;
    # an exhaustive search of the report and the FlagAI-Open/Aquila2 repo finds
    # no DPO, RLHF, PPO, KTO or reward model. So it goes in the EGO slot. This
    # project's central finding is that SFT and DPO divide labour by content
    # type, so putting an SFT checkpoint in the superego slot would corrupt
    # exactly the contrast the roster exists to measure.
    #
    # **THE BASE MUST BE PINNED. `main` IS THE WRONG MODEL.** BAAI replaced it
    # on 2024-06-06 and never updated the chat arm. Verified by fetching all
    # three configs:
    #
    #     Aquila2-7B @ main                       vocab 143,973  ctx 8192
    #     Aquila2-7B @ 9c76e143... (2023-10-26)   vocab 100,008  ctx 2048
    #     AquilaChat2-7B @ main                   vocab 100,008  ctx 2048
    #
    # Unpinned, the pair spans two vocabularies and there is no full-vocabulary
    # comparison to take — dimensionally undefined, not merely wrong.
    #
    # Book alongside it: Flan and COIG-PC sit in the PRETRAINING mixture
    # (report Table 2), so this is a pre-socialised base in the same sense as
    # Qwen. A small base->SFT displacement here is not permissiveness.
    "aquila2": ModelFamily(
        name="Aquila2 7B (BAAI) [v2 QUALIFIES; SFT arm; BASE REVISION PINNED]",
        base="BAAI/Aquila2-7B",
        ego="BAAI/AquilaChat2-7B",
        revisions={"base": "9c76e143c6e9621689ca76e078c465b0dee75eb8"},
    ),
    "zamba2": ModelFamily(
        name="Zamba2-7B (Zyphra) [v2 READY, SSM hybrid — CLOUD ONLY]",
        base="Zyphra/Zamba2-7B",
        superego="Zyphra/Zamba2-7B-Instruct",
    ),
    # AMBIGUOUS ON INDEPENDENCE, and the ambiguity is the card's own words:
    # "uses the same training data and data processing as used by the Gemma
    # model family". Independent weights, identical data recipe to a lineage
    # already on the roster. Kernel requirement UNCHECKED — Griffin is not
    # Mamba and may run natively.
    "recurrentgemma": ModelFamily(
        name="RecurrentGemma 9B (Google) [v2 AMBIG: Gemma data recipe; Griffin]",
        base="google/recurrentgemma-9b",
        superego="google/recurrentgemma-9b-it",
    ),
    "tulu-no-safety": ModelFamily(
        name="Tulu 3 8B (SFT without safety data)",
        base="meta-llama/Llama-3.1-8B",
        ego="allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data",
        superego="allenai/Llama-3.1-Tulu-3-8B-DPO",
        reinforced_superego="allenai/Llama-3.1-Tulu-3.1-8B",
    ),
    # Tulu SFT ablation variants (all share Llama 3.1 8B base)
    "tulu-sft-full": ModelFamily(
        name="Tulu SFT (full)",
        base="meta-llama/Llama-3.1-8B",
        #: **AN SFT CHECKPOINT BELONGS IN `ego`, NOT `superego`.** It sat in
        #: `superego` until 7 Aug, which meant any generic query for "the
        #: preference-optimised arm of Llama-3.1-8B" could return a MID-PIPELINE
        #: model. Surfaced by `Registry.base_aligned_pairs()`, which reads
        #: `superego` by definition and listed an SFT arm among the six
        #: candidates for that base.
        #:
        #: The other three ablation arms (nomath / nopersona / nowildchat)
        #: already used `ego` correctly, so this entry alone was inconsistent
        #: with its own siblings -- the kind of defect that survives because
        #: every sibling looks right and nobody diffs them.
        #:
        #: Fixing at the source rather than masking it in the ruling: the
        #: DPO_EQUIVALENT_RULINGS entry for this base picks Llama-3.1-8B-Instruct
        #: and would have hidden this forever.
        ego="allenai/Llama-3.1-Tulu-3-8B-SFT",
    ),
    "tulu-sft-nopersona": ModelFamily(
        name="Tulu SFT (no persona)",
        base="meta-llama/Llama-3.1-8B",
        ego="allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data",
    ),
    #: **SFT ARMS GO IN `ego`, NOT `superego`** (RH, 7 Aug: "All the sft's are
    #: 'ego' ... and dpo is 'superego'"). Declaring an SFT checkpoint in the
    #: superego slot made three ablation arms read as aligned ENDPOINTS, which
    #: is what put position=superego on them in the registry.
    "tulu-sft-nomath": ModelFamily(
        name="Tulu SFT (no math)",
        base="meta-llama/Llama-3.1-8B",
        ego="allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data",
    ),
    "tulu-sft-nowildchat": ModelFamily(
        name="Tulu SFT (no wildchat)",
        base="meta-llama/Llama-3.1-8B",
        ego="allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data",
    ),
    "zephyr": ModelFamily(
        name="Zephyr 7B",
        base="mistralai/Mistral-7B-v0.1",
        ego="HuggingFaceH4/mistral-7b-sft-beta",
        superego="HuggingFaceH4/zephyr-7b-beta",
    ),
    "pythia": ModelFamily(
        name="Pythia 6.9B",
        base="EleutherAI/pythia-6.9b",
        ego="lomahony/eleuther-pythia6.9b-hh-sft",
        superego="lomahony/eleuther-pythia6.9b-hh-dpo",
    ),
    "deepseek-7b": ModelFamily(
        name="DeepSeek LLM 7B",
        base="deepseek-ai/deepseek-llm-7b-base",
        superego="deepseek-ai/deepseek-llm-7b-chat",
    ),
    # New families with reasoning variants
    "smol3": ModelFamily(
        name="SmolLM3 3B",
        base="HuggingFaceTB/SmolLM3-3B-Base",
        superego="HuggingFaceTB/SmolLM3-3B",
        thinking_mode=True,
    ),
    "qwen3": ModelFamily(
        name="Qwen3 8B",
        base="Qwen/Qwen3-8B-Base",
        superego="Qwen/Qwen3-8B",
        thinking_mode=True,
    ),
    # Chinese bilingual
    "map-neo": ModelFamily(
        name="MAP-Neo 7B",
        base="m-a-p/neo_7b",
        ego="m-a-p/neo_7b_sft_v0.1",
        superego="m-a-p/neo_7b_instruct_v0.1",
    ),
    # Chinese-primary
    "ct-llm": ModelFamily(
        name="CT-LLM 2B",
        base="m-a-p/CT-LLM-Base",
        ego="m-a-p/CT-LLM-SFT",
        superego="m-a-p/CT-LLM-SFT-DPO",
    ),
    "internlm2": ModelFamily(
        name="InternLM 2 7B",
        base="internlm/internlm2-base-7b",
        ego="internlm/internlm2-chat-7b-sft",
        superego="internlm/internlm2-chat-7b",
    ),
    # Alignment method comparison
    "archangel-dpo": ModelFamily(
        name="Archangel DPO (Pythia 2.8B)",
        base="EleutherAI/pythia-2.8b",
        ego="ContextualAI/archangel_sft_pythia2-8b",
        superego="ContextualAI/archangel_sft-dpo_pythia2-8b",
    ),
    "archangel-kto": ModelFamily(
        name="Archangel KTO (Pythia 2.8B)",
        base="EleutherAI/pythia-2.8b",
        ego="ContextualAI/archangel_sft_pythia2-8b",
        superego="ContextualAI/archangel_sft-kto_pythia2-8b",
    ),
    "archangel-ppo": ModelFamily(
        name="Archangel PPO (Pythia 2.8B)",
        base="EleutherAI/pythia-2.8b",
        ego="ContextualAI/archangel_sft_pythia2-8b",
        superego="ContextualAI/archangel_sft-ppo_pythia2-8b",
    ),
    "archangel-slic": ModelFamily(
        name="Archangel SLIC (Pythia 2.8B)",
        base="EleutherAI/pythia-2.8b",
        ego="ContextualAI/archangel_sft_pythia2-8b",
        superego="ContextualAI/archangel_sft-slic_pythia2-8b",
    ),
    # SSM / hybrid architectures
    "olmo-hybrid": ModelFamily(
        name="OLMo Hybrid 7B",
        base="allenai/Olmo-Hybrid-7B",
        ego="allenai/Olmo-Hybrid-Instruct-SFT-7B",
        superego="allenai/Olmo-Hybrid-Instruct-DPO-7B",
    ),
    "falcon-mamba": ModelFamily(
        name="Falcon Mamba 7B",
        base="tiiuae/falcon-mamba-7b",
        superego="tiiuae/falcon-mamba-7b-instruct",
    ),
    "falcon3-mamba": ModelFamily(
        name="Falcon3 Mamba 7B",
        base="tiiuae/Falcon3-Mamba-7B-Base",
        superego="tiiuae/Falcon3-Mamba-7B-Instruct",
    ),
    # RWKV (pure RNN, zero attention, zero SSM)
    "rwkv": ModelFamily(
        name="RWKV-4 7B",
        base="RWKV/rwkv-4-7b-pile",
        superego="RWKV/rwkv-raven-7b",
    ),
    # MoE
    "olmoe": ModelFamily(
        name="OLMoE 1B-7B",
        base="allenai/OLMoE-1B-7B-0125",
        ego="allenai/OLMoE-1B-7B-0125-SFT",
        superego="allenai/OLMoE-1B-7B-0125-DPO",
        reinforced_superego="allenai/OLMoE-1B-7B-0125-Instruct",
    ),
    # Falcon3 (Llama arch, TII)
    "falcon3-1b": ModelFamily(
        name="Falcon3 1B",
        base="tiiuae/Falcon3-1B-Base",
        superego="tiiuae/Falcon3-1B-Instruct",
    ),
    "falcon3-3b": ModelFamily(
        name="Falcon3 3B",
        base="tiiuae/Falcon3-3B-Base",
        superego="tiiuae/Falcon3-3B-Instruct",
    ),
    "falcon3-7b": ModelFamily(
        name="Falcon3 7B",
        base="tiiuae/Falcon3-7B-Base",
        superego="tiiuae/Falcon3-7B-Instruct",
    ),
    "falcon3-10b": ModelFamily(
        name="Falcon3 10B",
        base="tiiuae/Falcon3-10B-Base",
        superego="tiiuae/Falcon3-10B-Instruct",
    ),
    # Falcon-H1 (SSM-Transformer hybrid, TII)
    "falcon-h1-1.5b": ModelFamily(
        name="Falcon-H1 1.5B",
        base="tiiuae/Falcon-H1-1.5B-Base",
        superego="tiiuae/Falcon-H1-1.5B-Instruct",
    ),
    "falcon-h1-7b": ModelFamily(
        name="Falcon-H1 7B",
        base="tiiuae/Falcon-H1-7B-Base",
        superego="tiiuae/Falcon-H1-7B-Instruct",
    ),
    # Yi (01.AI, Chinese+English)
    "yi": ModelFamily(
        name="Yi 1.5 9B",
        base="01-ai/Yi-1.5-9B",
        superego="01-ai/Yi-1.5-9B-Chat",
    ),
    # Baichuan2 (Chinese+English, pretraining ckpts available)
    "baichuan": ModelFamily(
        name="Baichuan2 7B",
        base="baichuan-inc/Baichuan2-7B-Base",
        superego="baichuan-inc/Baichuan2-7B-Chat",
    ),
    # GLM-4 (THUDM/Zhipu, Chinese+English)
    "glm4": ModelFamily(
        name="GLM-4 9B",
        base="zai-org/glm-4-9b-hf",
        superego="zai-org/glm-4-9b-chat-hf",
    ),
    # StableLM 2 (Stability AI, 3-stage: base→chat-SFT→zephyr-DPO)
    "stablelm": ModelFamily(
        name="StableLM 2 1.6B",
        base="stabilityai/stablelm-2-1_6b",
        ego="stabilityai/stablelm-2-1_6b-chat",
        superego="stabilityai/stablelm-2-zephyr-1_6b",
    ),
    # RedPajama (Together, 3-stage: base→instruct-SFT→chat-RLHF)
    "redpajama": ModelFamily(
        name="RedPajama 7B",
        base="togethercomputer/RedPajama-INCITE-Base-7B-v0.1",
        ego="togethercomputer/RedPajama-INCITE-7B-Instruct",
        superego="togethercomputer/RedPajama-INCITE-7B-Chat",
    ),
    # TinyLlama (Llama 2 arch, intermediate pretraining ckpts)
    "tinyllama": ModelFamily(
        name="TinyLlama 1.1B",
        base="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        superego="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    # BLOOM (BigScience, 46 languages, multitask-tuned not safety-aligned)
    "bloom": ModelFamily(
        name="BLOOM 7B",
        base="bigscience/bloom-7b1",
        superego="bigscience/bloomz-7b1",
    ),
    # MiniCPM5 (OpenBMB/Tsinghua, Chinese, 3-stage)
    "minicpm": ModelFamily(
        name="MiniCPM5 1B",
        base="openbmb/MiniCPM5-1B-Base",
        ego="openbmb/MiniCPM5-1B-SFT",
        superego="openbmb/MiniCPM5-1B",
    ),
    # Phi-4 (Microsoft)
    "phi4": ModelFamily(
        name="Phi-4 14B",
        base="microsoft/phi-4",
        superego="microsoft/phi-4-reasoning",
    ),
    # Scale variants
    "olmo-32b": ModelFamily(
        name="OLMo 3.1 32B",
        base="allenai/Olmo-3-1125-32B",
        ego="allenai/Olmo-3.1-32B-Instruct-SFT",
        superego="allenai/Olmo-3.1-32B-Instruct-DPO",
        reinforced_superego="allenai/Olmo-3.1-32B-Instruct",
    ),
    "llama-70b": ModelFamily(
        name="Llama 3.1 70B",
        base="meta-llama/Llama-3.1-70B",
        superego="meta-llama/Llama-3.1-70B-Instruct",
    ),
}

TULU_ABLATIONS = {
    "standard": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "no-safety": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data",
    "no-persona": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data",
    "no-math": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data",
    "no-wildchat": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data",
}

DEFAULT_FAMILY = "olmo"

# Legacy constants — point at default family for backward compat in function signatures
BASE_MODEL_NAME = MODEL_FAMILIES[DEFAULT_FAMILY].base
SFT_MODEL_NAME = MODEL_FAMILIES[DEFAULT_FAMILY].ego
DPO_MODEL_NAME = MODEL_FAMILIES[DEFAULT_FAMILY].superego
INSTRUCT_MODEL_NAME = MODEL_FAMILIES[DEFAULT_FAMILY].reinforced_superego


# Centralized intra-package imports.
# Order matters: later modules depend on names defined by earlier ones.
from .core import *
from .models import *
from .analysis import *
from .experiments import *
from .generation import *
from .viz import *
from .psyche import *
from .circuit import Circuit, Mode
