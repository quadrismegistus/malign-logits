"""
registry.py — Model-centric data model.

Models are atoms (identified by HuggingFace model ID). Relations are
typed edges between models. Families are derived queries, not primary keys.

    from malign_logits.registry import Registry

    reg = Registry()
    reg.base_of("allenai/Olmo-3-7B-Instruct-SFT")
    # → "allenai/Olmo-3-1025-7B"

    reg.variants_of("meta-llama/Llama-3.1-8B")
    # → ["meta-llama/Llama-3.1-8B-Instruct", "allenai/tulu-3-8b", ...]

    reg.path("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-DPO")
    # → [("allenai/Olmo-3-1025-7B", "sft", "allenai/Olmo-3-7B-Instruct-SFT"),
    #    ("allenai/Olmo-3-7B-Instruct-SFT", "dpo", "allenai/Olmo-3-7B-Instruct-DPO")]
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple


#: EVERY RELATION KIND IS ASSIGNED TO EXACTLY ONE CLASS, so a new kind cannot
#: default into the walk ([2159](ii)).
#:
#:   WALKED   directed descent: the parent is an ANCESTOR, so base_of climbs it
#:   UNIONED  co-lineage: same pretraining, NO ancestor -- unions the lineage
#:            and is never walked. `smaller_sibling_of` is one release at
#:            several scales, and NO RUNG IS ANOTHER RUNG'S BASE; walking it
#:            made Falcon3-10B ambiguous between 1B, 3B and 7B, which is not an
#:            ambiguity but a category error.
#:   CHECKED  post-repair assertion only: `same_base_as` is SYMMETRIC, and a
#:            symmetric relation walked upward makes a SIBLING the parent --
#:            it made kto's "base" the dpo checkpoint, right only by luck.
_WALKED = frozenset({"sft_of", "dpo_of", "rlvr_of", "kto_of", "ppo_of",
                     "slic_of", "reasoning_of", "data_ablation_of"})
_UNIONED = frozenset({"smaller_sibling_of"})
_CHECKED = frozenset({"same_base_as", "smaller_predecessor_of"})
_DESCENT = _WALKED
_STAGE_REL = {"sft": "sft_of", "dpo": "dpo_of", "rlvr": "rlvr_of",
              "kto": "kto_of", "ppo": "ppo_of", "slic": "slic_of",
              "reasoning": "reasoning_of"}

REGISTRY_PATH = Path(__file__).parent.parent / "data" / "model_registry.json"

RELATION_TYPES = [
    "sft_of",       # supervised fine-tuning (known intermediate)
    "dpo_of",       # direct preference optimization (known intermediate)
    "rlvr_of",      # reinforcement learning from verifiable rewards
    "aligned_of",   # post-trained but exact method unknown
    "distill_of",   # knowledge distillation (possibly cross-family)
]
STAGE_ORDER = ["base", "sft", "dpo", "rlvr"]


@dataclass
class ModelInfo:
    """Metadata for a single model checkpoint."""
    model_id: str
    stage: str = ""
    org: str = ""
    org_type: str = ""
    scale: str = ""
    corpus: str = ""
    country: str = ""
    open_weight: bool = True
    open_data: bool = False
    nickname: str = ""


# Preferred short names. Base models get the family name with no suffix.
# Variants get family-stage or a custom name.
def annotation_prefix(model_id: str, dots: bool = True, maxlen: int = 20) -> str:
    """Truncated model-name prefix used for beam/tree annotation column names.

    LOAD-BEARING format: it must match how the annotation columns were written.
    ``annotate_tree`` replaces '-' but not '.', so the graphdb read-side tries
    both ``dots=True`` and ``dots=False``. Single source for a form that was
    duplicated across probe/graphdb/server.
    """
    s = model_id.split("/")[-1].replace("-", "_")
    if dots:
        s = s.replace(".", "_")
    return s[:maxlen]


NICKNAMES = {
    # Yi
    "01-ai/Yi-9B": "yi",
    "01-ai/Yi-1.5-9B-Chat": "yi-chat",
    # Pythia
    "EleutherAI/pythia-6.9b": "pythia",
    "lomahony/eleuther-pythia6.9b-hh-sft": "pythia-sft",
    "lomahony/eleuther-pythia6.9b-hh-dpo": "pythia-dpo",
    # SmolLM
    "HuggingFaceTB/SmolLM2-360M": "smol",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "smol-instruct",
    "HuggingFaceTB/SmolLM3-3B-Base": "smol3",
    "HuggingFaceTB/SmolLM3-3B": "smol3-instruct",
    # Amber
    "LLM360/Amber": "amber",
    "LLM360/AmberChat": "amber-sft",
    "LLM360/AmberSafe": "amber-dpo",
    # Qwen 0.5B
    "Qwen/Qwen2.5-0.5B": "qwen-tiny",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-tiny-instruct",
    # Qwen 7B
    "Qwen/Qwen2.5-7B": "qwen",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-instruct",
    # Qwen 3
    "Qwen/Qwen3-8B-Base": "qwen3",
    "Qwen/Qwen3-8B": "qwen3-instruct",
    # OLMo 1B
    "allenai/OLMo-2-0425-1B": "olmo-tiny",
    "allenai/OLMo-2-0425-1B-SFT": "olmo-tiny-sft",
    "allenai/OLMo-2-0425-1B-DPO": "olmo-tiny-dpo",
    "allenai/OLMo-2-0425-1B-Instruct": "olmo-tiny-instruct",
    # OLMo 7B
    "allenai/Olmo-3-1025-7B": "olmo",
    "allenai/Olmo-3-7B-Instruct-SFT": "olmo-sft",
    "allenai/Olmo-3-7B-Instruct-DPO": "olmo-dpo",
    "allenai/Olmo-3-7B-Instruct": "olmo-instruct",
    "allenai/Olmo-3-7B-Think-SFT": "olmo-think-sft",
    "allenai/Olmo-3-7B-Think-DPO": "olmo-think-dpo",
    # Archangel (alignment method comparison on Pythia 2.8B)
    "EleutherAI/pythia-2.8b": "pythia-2.8b",
    "ContextualAI/archangel_sft_pythia2-8b": "archangel-sft",
    "ContextualAI/archangel_sft-dpo_pythia2-8b": "archangel-dpo",
    "ContextualAI/archangel_sft-kto_pythia2-8b": "archangel-kto",
    "ContextualAI/archangel_sft-ppo_pythia2-8b": "archangel-ppo",
    "ContextualAI/archangel_sft-slic_pythia2-8b": "archangel-slic",
    # OLMo Hybrid (SSM-Transformer)
    "allenai/Olmo-Hybrid-7B": "olmo-hybrid",
    "allenai/Olmo-Hybrid-Instruct-SFT-7B": "olmo-hybrid-sft",
    "allenai/Olmo-Hybrid-Instruct-DPO-7B": "olmo-hybrid-dpo",
    # Falcon Mamba (pure SSM)
    "tiiuae/falcon-mamba-7b": "falcon-mamba",
    "tiiuae/falcon-mamba-7b-instruct": "falcon-mamba-instruct",
    # Falcon3-Mamba (pure SSM)
    "tiiuae/Falcon3-Mamba-7B-Base": "falcon3-mamba",
    "tiiuae/Falcon3-Mamba-7B-Instruct": "falcon3-mamba-instruct",
    # RWKV (pure RNN)
    "RWKV/rwkv-4-7b-pile": "rwkv",
    "RWKV/rwkv-raven-7b": "rwkv-raven",
    # MAP-Neo (bilingual Chinese+English)
    "m-a-p/neo_7b": "map-neo",
    "m-a-p/neo_7b_sft_v0.1": "map-neo-sft",
    "m-a-p/neo_7b_instruct_v0.1": "map-neo-dpo",
    # OLMoE (MoE)
    "allenai/OLMoE-1B-7B-0125": "olmoe",
    "allenai/OLMoE-1B-7B-0125-SFT": "olmoe-sft",
    "allenai/OLMoE-1B-7B-0125-DPO": "olmoe-dpo",
    "allenai/OLMoE-1B-7B-0125-Instruct": "olmoe-instruct",
    # OLMo 32B
    "allenai/Olmo-3-1125-32B": "olmo-32b",
    "allenai/Olmo-3.1-32B-Instruct-SFT": "olmo-32b-sft",
    "allenai/Olmo-3.1-32B-Instruct-DPO": "olmo-32b-dpo",
    "allenai/Olmo-3.1-32B-Instruct": "olmo-32b-instruct",
    # DeepSeek
    "deepseek-ai/deepseek-llm-7b-base": "deepseek",
    "deepseek-ai/deepseek-llm-7b-chat": "deepseek-chat",
    # Llama
    "meta-llama/Llama-3.1-8B": "llama",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-instruct",
    # Llama 70B
    "meta-llama/Llama-3.1-70B": "llama-70b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama-70b-instruct",
    # Tulu (Llama-based)
    "allenai/Llama-3.1-Tulu-3-8B-SFT": "tulu-sft",
    "allenai/Llama-3.1-Tulu-3-8B-DPO": "tulu-dpo",
    "allenai/Llama-3.1-Tulu-3.1-8B": "tulu",
    "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data": "tulu-sft-nosafety",
    "allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data": "tulu-sft-nomath",
    "allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data": "tulu-sft-nopersona",
    "allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data": "tulu-sft-nowildchat",
    # Other Llama-based
    "NousResearch/Hermes-3-Llama-3.1-8B": "hermes-llama",
    "dphn/Dolphin3.0-Llama3.1-8B": "dolphin-llama",
    # Reasoning distills
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "r1-llama",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "r1-qwen",
    # Mistral / Zephyr
    "mistralai/Mistral-7B-v0.1": "mistral",
    "HuggingFaceH4/mistral-7b-sft-beta": "zephyr-sft",
    "HuggingFaceH4/zephyr-7b-beta": "zephyr",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO": "hermes-mistral",
    "berkeley-nest/Starling-LM-7B-alpha": "starling",
    "cognitivecomputations/dolphin-2.6-mistral-7b-dpo": "dolphin-mistral",
    "openchat/openchat-3.5-0106": "openchat",
    "teknium/OpenHermes-2.5-Mistral-7B": "openhermes",
    # Falcon (original)
    "tiiuae/falcon-7b": "falcon",
    "tiiuae/falcon-7b-instruct": "falcon-instruct",
    # Falcon3
    "tiiuae/Falcon3-1B-Base": "falcon3-1b",
    "tiiuae/Falcon3-1B-Instruct": "falcon3-1b-instruct",
    "tiiuae/Falcon3-3B-Base": "falcon3-3b",
    "tiiuae/Falcon3-3B-Instruct": "falcon3-3b-instruct",
    "tiiuae/Falcon3-7B-Base": "falcon3-7b",
    "tiiuae/Falcon3-7B-Instruct": "falcon3-7b-instruct",
    "tiiuae/Falcon3-10B-Base": "falcon3-10b",
    "tiiuae/Falcon3-10B-Instruct": "falcon3-10b-instruct",
    # Falcon-H1 (SSM hybrid)
    "tiiuae/Falcon-H1-1.5B-Base": "falcon-h1-1.5b",
    "tiiuae/Falcon-H1-1.5B-Instruct": "falcon-h1-1.5b-instruct",
    "tiiuae/Falcon-H1-7B-Base": "falcon-h1-7b",
    "tiiuae/Falcon-H1-7B-Instruct": "falcon-h1-7b-instruct",
    # Yi (01.AI)
    "01-ai/Yi-1.5-9B": "yi",
    "01-ai/Yi-1.5-9B-Chat": "yi-chat",
    # Baichuan2
    "baichuan-inc/Baichuan2-7B-Base": "baichuan",
    "baichuan-inc/Baichuan2-7B-Chat": "baichuan-chat",
    # GLM-4
    "zai-org/glm-4-9b-hf": "glm4",
    "zai-org/glm-4-9b-chat-hf": "glm4-chat",
    # StableLM 2
    "stabilityai/stablelm-2-1_6b": "stablelm",
    "stabilityai/stablelm-2-1_6b-chat": "stablelm-chat",
    "stabilityai/stablelm-2-zephyr-1_6b": "stablelm-dpo",
    # RedPajama
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": "redpajama",
    "togethercomputer/RedPajama-INCITE-7B-Instruct": "redpajama-instruct",
    "togethercomputer/RedPajama-INCITE-7B-Chat": "redpajama-chat",
    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": "tinyllama",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama-chat",
    # BLOOM
    "bigscience/bloom-7b1": "bloom",
    "bigscience/bloomz-7b1": "bloomz",
    # MiniCPM5
    "openbmb/MiniCPM5-1B-Base": "minicpm",
    "openbmb/MiniCPM5-1B-SFT": "minicpm-sft",
    "openbmb/MiniCPM5-1B": "minicpm-instruct",
    # Phi-4
    "microsoft/phi-4": "phi4",
    "microsoft/phi-4-reasoning": "phi4-reasoning",
    # Baichuan
    "baichuan-inc/Baichuan2-7B-Base": "baichuan",
    "baichuan-inc/Baichuan2-7B-Chat": "baichuan-chat",
    # Gemma
    "google/gemma-7b": "gemma",
    "google/gemma-7b-it": "gemma-instruct",
    # InternLM 2
    "internlm/internlm2-base-7b": "internlm2",
    "internlm/internlm2-7b": "internlm2-continued",
    "internlm/internlm2-chat-7b-sft": "internlm2-sft",
    "internlm/internlm2-chat-7b": "internlm2-chat",
    # InternLM 2.5
    "internlm/internlm2_5-7b": "internlm",
    "internlm/internlm2_5-7b-chat": "internlm-chat",
    # CT-LLM (Chinese-primary)
    "m-a-p/CT-LLM-Base": "ct-llm",
    "m-a-p/CT-LLM-SFT": "ct-llm-sft",
    "m-a-p/CT-LLM-SFT-DPO": "ct-llm-dpo",
}


@dataclass
class Relation:
    """A typed edge between two models."""
    child: str
    relation: str  # sft_of, dpo_of, rlvr_of, distill_of, ...
    parent: str


class Registry:
    """Model registry: models as nodes, relations as edges.

    Populated from MODEL_FAMILIES on first use, extensible via add().
    Persists to data/model_registry.json.
    """

    def __init__(self, path: str = None):
        self._path = Path(path) if path else REGISTRY_PATH
        self._models: Dict[str, ModelInfo] = {}
        self._relations: List[Relation] = []
        if self._path.exists():
            self._load()
        else:
            self._from_families()

    def _from_families(self):
        """Bootstrap from the existing MODEL_FAMILIES registry."""
        from . import MODEL_FAMILIES

        relation_map = {
            "ego": "sft_of",
            "superego": "dpo_of",
            "reinforced_superego": "rlvr_of",
        }

        for key, fam in MODEL_FAMILIES.items():
            # Register base model
            self._models[fam.base] = ModelInfo(
                model_id=fam.base, stage="base",
                scale=getattr(fam, 'scale', ''),
                nickname=NICKNAMES.get(fam.base, ''),
            )

            # Register post-training checkpoints with relations
            for attr, rel_type in relation_map.items():
                model_id = getattr(fam, attr, None)
                if model_id is None:
                    continue

                # Determine parent and actual relation type
                if rel_type == "sft_of":
                    parent = fam.base
                    stage = "sft"
                elif rel_type == "dpo_of":
                    if fam.ego:
                        parent = fam.ego
                        stage = "dpo"
                    else:
                        # No SFT intermediate — we just know it's aligned
                        parent = fam.base
                        rel_type = "aligned_of"
                        stage = "aligned"
                elif rel_type == "rlvr_of":
                    parent = fam.superego or fam.ego or fam.base
                    stage = "rlvr"
                else:
                    parent = fam.base
                    stage = rel_type.replace("_of", "")

                self._models[model_id] = ModelInfo(
                    model_id=model_id, stage=stage,
                    nickname=NICKNAMES.get(model_id, ''),
                )
                self._relations.append(Relation(
                    child=model_id, relation=rel_type, parent=parent,
                ))

            # Reasoning branch
            if fam.reasoning:
                self._models[fam.reasoning] = ModelInfo(
                    model_id=fam.reasoning, stage="reasoning",
                )
                base_for_reasoning = fam.reasoning_base or fam.base
                self._relations.append(Relation(
                    child=fam.reasoning, relation="distill_of",
                    parent=base_for_reasoning,
                ))

        self.save()

    def _load(self):
        with open(self._path) as f:
            data = json.load(f)
        # TOLERATE FIELDS THE DATACLASS DOES NOT DECLARE, and keep them.
        # The canonical file carries family/position/architecture/weights_format
        # and more; ModelInfo predates all of it. Passing the row straight in
        # raised TypeError and broke every caller the moment the file was
        # regenerated. Dropping the unknown keys instead would be worse -- the
        # loader would silently return less than the file says.
        known = set(ModelInfo.__dataclass_fields__)
        for m in data.get("models", []):
            info = ModelInfo(**{k: v for k, v in m.items() if k in known})
            for k, v in m.items():
                if k not in known:
                    setattr(info, k, v)
            self._models[m["model_id"]] = info
        for r in data.get("relations", []):
            self._relations.append(Relation(**r))

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": [m.__dict__ for m in self._models.values()],
            "relations": [r.__dict__ for r in self._relations],
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    #: PATH OF THE DERIVED PAIR EXPORT. Beside REGISTRY_PATH because it is the
    #: same kind of object: a view of the library, regenerated and never edited.
    PAIRS_PATH = Path(__file__).parent.parent / "data" / "base_aligned_pairs.json"

    def save_base_aligned_pairs(self, path=None):
        """Write `base_aligned_pairs()` to `data/base_aligned_pairs.json`.

        **THIS EXPORT HAD NO PRODUCER AND THAT IS WHY IT WENT STALE.** The
        derivation lives here, in `base_aligned_pairs()`; the file on disk was
        written once by hand and then diverged silently. On 2026-08-10 it was
        still serving the RETIRED Teuken pairing -- `Teuken-7B-base-v0.6`
        against `instruct-commercial-v0.4`, an instruct built off a 4T-token
        base whose repo 404s -- to every consumer that read the FILE while
        `Registry()` returned the corrected arm. Ten-plus M02 scripts read the
        file.

        The registry's own schema note names the hazard in the general case:
        *"a cache that can outrank its source is how 59 models shadowed 112 for
        five weeks."* An export with no producer is that cache with no way back.

        A method rather than a script, for the same reason `save()` is: the
        thing that KNOWS how pairs are derived is this class, and a standalone
        producer would be a second place for the derivation to drift.
        """
        rows = self.base_aligned_pairs()
        out = path or self.PAIRS_PATH
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)
        return len(rows)

    # -- queries ---------------------------------------------------------------

    def base_of(self, model_id: str) -> Optional[str]:
        """Find the base model for a given checkpoint."""
        visited = set()
        current = model_id
        while current:
            if current in visited:
                break
            visited.add(current)
            #: DESCENT EDGES ONLY. `same_base_as` is 77 of 181 relations and
            #: is SYMMETRIC -- walking it upward makes a SIBLING the parent,
            #: which terminates on whatever that peer's chain reaches and is
            #: right only by luck ([2152].2, ruled [2154]).
            #:
            #: AND FILE ORDER DECIDED NOTHING FROM HERE ON. This took the FIRST
            #: matching relation in JSON key order, so the base of a model with
            #: several edges depended on how the file happened to be written --
            #: a silent ordering dependency inside the quantity every roster
            #: count is built from. Where several DESCENT edges match, take the
            #: stage-consistent one; where that is ambiguous, RAISE rather than
            #: pick, because a wrong base is a wrong independence claim and a
            #: loud failure is cheaper than a plausible number.
            cands = [r for r in self._relations
                     if r.child == current and r.relation in _DESCENT]
            if not cands:
                return current
            if len(cands) > 1:
                st = self.stage_of(current)
                want = _STAGE_REL.get(st)
                consistent = [r for r in cands if r.relation == want]
                if len(consistent) != 1:
                    raise ValueError(
                        f"base_of({current!r}) is AMBIGUOUS: "
                        f"{[(r.relation, r.parent) for r in cands]} with "
                        f"stage={st!r}. Add the stage-consistent descent edge "
                        f"or remove the spurious one; file order must not decide.")
                cands = consistent
            parent = cands[0].parent
            current = parent
        return current

    def parent_of(self, model_id: str) -> Optional[Tuple[str, str]]:
        """Direct parent and relation type. Returns (parent_id, relation)."""
        for r in self._relations:
            if r.child == model_id:
                return r.parent, r.relation
        return None, None

    def children_of(self, model_id: str) -> List[Tuple[str, str]]:
        """All direct children. Returns [(child_id, relation), ...]."""
        return [(r.child, r.relation) for r in self._relations
                if r.parent == model_id]

    def variants_of(self, base_id: str) -> List[str]:
        """All models derived from a given base (any depth)."""
        result = []
        queue = [base_id]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for child, _ in self.children_of(current):
                result.append(child)
                queue.append(child)
        return result

    def path(self, from_id: str, to_id: str) -> List[Tuple[str, str, str]]:
        """Training path from one model to another.

        Returns [(from, relation, to), ...] or empty list if no path.
        """
        # BFS from from_id to to_id through parent→child relations
        from collections import deque
        queue = deque([(from_id, [])])
        visited = set()
        while queue:
            current, path_so_far = queue.popleft()
            if current == to_id:
                return path_so_far
            if current in visited:
                continue
            visited.add(current)
            for child, rel in self.children_of(current):
                queue.append((child, path_so_far + [(current, rel, child)]))
        return []

    def stage_of(self, model_id: str) -> str:
        """What stage is this model? (base, sft, dpo, rlvr, reasoning)."""
        if model_id in self._models:
            return self._models[model_id].stage
        return ""

    def all_bases(self) -> List[str]:
        """All base models in the registry."""
        return [m.model_id for m in self._models.values() if m.stage == "base"]

    def family_key(self, model_id: str) -> Optional[str]:
        """Reverse lookup: which MODEL_FAMILIES key contains this model?"""
        from . import MODEL_FAMILIES
        for key, fam in MODEL_FAMILIES.items():
            if model_id in fam.all_checkpoints:
                return key
        return None

    @staticmethod
    def _fams():
        from . import MODEL_FAMILIES
        return MODEL_FAMILIES

    #: RULINGS for bases carrying more than one preference-optimised arm. Each
    #: is a CHOICE, not a fact, and each is reversible by editing this dict --
    #: which is the point of putting them here rather than in a caller.
    #:
    #:   Llama-3.1-8B    Meta's own Instruct vs AI2's Tulu-3-DPO. Both are real
    #:                   answers to different questions -- a frontier lab's
    #:                   alignment against an open documented one on the same
    #:                   weights -- and that contrast is a finding the campaign
    #:                   has used. They cannot both be in one roster and be
    #:                   called independent. Instruct is what "aligned Llama"
    #:                   means to a reader.
    #:   pythia-2.8b     archangel dpo/kto/ppo/slic are four preference METHODS
    #:                   on one SFT checkpoint. dpo is the literal equivalent;
    #:                   the other three ARE the method experiment.
    #:   Olmo-3-1025-7B  Instruct-DPO is the main line, Think-DPO a branch.
    DPO_EQUIVALENT_RULINGS = {
        "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
        "EleutherAI/pythia-2.8b": "ContextualAI/archangel_sft-dpo_pythia2-8b",
        "allenai/Olmo-3-1025-7B": "allenai/Olmo-3-7B-Instruct-DPO",
    }

    def base_aligned_pairs(self, ruled_only: bool = False,
                           include_ambiguous: bool = True) -> List[dict]:
        """ONE base -> ONE aligned checkpoint, per unique pretraining run.

        **THE UNIT IS THE BASE MODEL, NOT THE FAMILY ENTRY.** 62 family entries
        sit on 52 unique bases; seven entries share `meta-llama/Llama-3.1-8B`
        alone. Counting family entries as independent pairs inflates every
        lineage-level n, which is the error the campaign has already booked
        twice (59 -> 39 -> 34).

        **THE ALIGNED ARM IS THE PREFERENCE-OPTIMISED ONE (`superego`), NOT THE
        LAST ONE.** `reinforced_superego` (RLVR/Instruct) sits ABOVE dpo and is
        a different stage, not a better version of it. Where a family has no
        superego -- a 2-layer topology -- `ego` IS the aligned arm and is used.

        Returns dicts with `base`, `aligned`, `family`, `stage`, `ambiguous`,
        and `candidates` when more than one arm was available. A ruled pair
        carries the ruling so a reader can see a choice was made rather than
        discovering one absent.

        ruled_only          only the bases needing a ruling (for auditing them)
        include_ambiguous   False drops them entirely rather than ruling them,
                            which is the conservative roster: 49 not 52.
        """
        from . import MODEL_FAMILIES
        cand = {}
        for key, fam in MODEL_FAMILIES.items():
            b = getattr(fam, "base", None)
            if not b:
                continue
            #: superego first; ego ONLY as the 2-layer fallback. Never
            #: reinforced_superego -- a different stage, not a substitute.
            a = getattr(fam, "superego", None) or getattr(fam, "ego", None)
            if not a or a == b:
                continue
            cand.setdefault(b, {})[a] = key

        out = []
        for b, arms in sorted(cand.items()):
            ruling = self.DPO_EQUIVALENT_RULINGS.get(b)
            ambiguous = len(arms) > 1
            if ambiguous and not include_ambiguous:
                continue
            if ambiguous:
                a = ruling if ruling in arms else sorted(arms)[0]
            else:
                a = next(iter(arms))
            if ruled_only and not ambiguous:
                continue
            #: **THE `ego` FALLBACK IS SAFE FOR 2-LAYER FAMILIES AND UNSAFE FOR
            #: SFT-ABLATION ONES, AND THE TWO ARE INDISTINGUISHABLE BY SHAPE.**
            #: Both are "a family with no superego". For `llama` the fallback
            #: correctly yields Llama-3.1-8B-Instruct; for `tulu-sft-nomath` it
            #: would yield an SFT checkpoint and label it aligned. Today that
            #: never fires, because every SFT-ablation family sits on
            #: Llama-3.1-8B and the ruling picks Instruct -- i.e. it is masked
            #: by an unrelated decision, which is not a guarantee. Register one
            #: SFT ablation on a base with no other family and it fires
            #: silently. Flagged rather than dropped: dropping would remove
            #: genuine 2-layer pairs, and this must not fail closed on them.
            fam = arms[a]
            via_ego = not getattr(self._fams().get(fam), "superego", None)
            looks_sft = ("-SFT" in a or a.endswith("SFT")) and "DPO" not in a
            out.append({"base": b, "aligned": a, "family": fam,
                        "stage": self.stage_of(a) or "aligned",
                        "ambiguous": ambiguous,
                        "ruled": bool(ambiguous and ruling in arms),
                        "candidates": sorted(arms) if ambiguous else None,
                        "warn_sft_as_aligned": bool(via_ego and looks_sft)})
        return out

    # -- mutation --------------------------------------------------------------

    def add(self, model_id: str, relation: str = None, parent: str = None,
            stage: str = "", **kwargs):
        """Register a new model, optionally with a relation."""
        self._models[model_id] = ModelInfo(model_id=model_id, stage=stage,
                                           **kwargs)
        if relation and parent:
            self._relations.append(Relation(
                child=model_id, relation=relation, parent=parent))
        self.save()

    def models(self) -> List[str]:
        """All registered model IDs."""
        return list(self._models.keys())

    def info(self, model_id: str) -> Optional[ModelInfo]:
        return self._models.get(model_id)

    def nickname(self, model_id: str) -> str:
        """Short human-readable name for a model."""
        return NICKNAMES.get(model_id, model_id.split("/")[-1])

    def from_nickname(self, nick: str) -> Optional[str]:
        """Resolve a nickname to a full model ID. Returns None if not found."""
        for model_id, n in NICKNAMES.items():
            if n == nick:
                return model_id
        # Also check if it's already a full model ID
        if nick in self._models:
            return nick
        return None

    def nicknames(self) -> Dict[str, str]:
        """All nickname → model_id mappings."""
        return {v: k for k, v in NICKNAMES.items()}

    def resolve(self, name: str) -> Optional[str]:
        """Resolve any model name variant to its canonical HuggingFace ID.

        Accepts: full HF ID, nickname, or sanitized beam source name
        (e.g. 'Olmo_3_7B_Instruct_DPO' or 'olmo-dpo').
        Returns the full HF ID or None if no match.
        """
        if name in self._models:
            return name
        nick_match = self.from_nickname(name)
        if nick_match:
            return nick_match
        if not hasattr(self, '_sanitized_map'):
            self._sanitized_map = {}
            all_ids = set(self._models.keys()) | set(NICKNAMES.keys())
            for model_id in all_ids:
                short = model_id.split('/')[-1]
                sanitized = short.replace('-', '_').replace('.', '_')
                for variant in [short, short.lower(), sanitized, sanitized.lower()]:
                    if variant not in self._sanitized_map:
                        self._sanitized_map[variant] = model_id
        sanitized_name = name.replace('-', '_').replace('.', '_')
        for variant in [name, sanitized_name, name.lower(), sanitized_name.lower()]:
            if variant in self._sanitized_map:
                return self._sanitized_map[variant]
        return None

    def __len__(self):
        return len(self._models)

    def __repr__(self):
        n_bases = len(self.all_bases())
        return f"Registry({len(self._models)} models, {n_bases} bases, {len(self._relations)} relations)"
