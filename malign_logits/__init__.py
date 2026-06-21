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
warnings.filterwarnings("ignore")

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
    "qwen": ModelFamily(
        name="Qwen 2.5 7B",
        base="Qwen/Qwen2.5-7B",
        superego="Qwen/Qwen2.5-7B-Instruct",
        reasoning="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ),
    "tulu": ModelFamily(
        name="Tulu 3.1 8B",
        base="meta-llama/Llama-3.1-8B",
        ego="allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data",
        superego="allenai/Llama-3.1-Tulu-3-8B-DPO",
        reinforced_superego="allenai/Llama-3.1-Tulu-3.1-8B",
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
