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
    # Falcon
    "tiiuae/falcon-7b": "falcon",
    "tiiuae/falcon-7b-instruct": "falcon-instruct",
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
        for m in data.get("models", []):
            self._models[m["model_id"]] = ModelInfo(**m)
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

    # -- queries ---------------------------------------------------------------

    def base_of(self, model_id: str) -> Optional[str]:
        """Find the base model for a given checkpoint."""
        visited = set()
        current = model_id
        while current:
            if current in visited:
                break
            visited.add(current)
            parent = None
            for r in self._relations:
                if r.child == current:
                    parent = r.parent
                    break
            if parent is None:
                return current
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

    def __len__(self):
        return len(self._models)

    def __repr__(self):
        n_bases = len(self.all_bases())
        return f"Registry({len(self._models)} models, {n_bases} bases, {len(self._relations)} relations)"
