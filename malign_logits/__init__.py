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
PATH_STASH = os.path.join(PATH_DATA_RAW, "stash")
PATH_FIGURES = os.path.join(PATH_REPO, "figures")


# ---------------------------------------------------------------------------
# Model family registry
# ---------------------------------------------------------------------------

@dataclass
class ModelFamily:
    """A model family with checkpoints at each training stage."""
    name: str
    base: str                              # primary process (always required)
    ego: str | None = None                 # SFT checkpoint
    superego: str | None = None            # DPO or instruct-as-superego
    reinforced_superego: str | None = None # RLVR

    @property
    def n_layers(self):
        return sum(1 for x in [self.base, self.ego, self.superego, self.reinforced_superego] if x is not None)


MODEL_FAMILIES = {
    "olmo": ModelFamily(
        name="OLMo 3 7B",
        base="allenai/Olmo-3-1025-7B",
        ego="allenai/Olmo-3-7B-Instruct-SFT",
        superego="allenai/Olmo-3-7B-Instruct-DPO",
        reinforced_superego="allenai/Olmo-3-7B-Instruct",
    ),
    "llama": ModelFamily(
        name="Llama 3.1 8B",
        base="meta-llama/Llama-3.1-8B",
        superego="meta-llama/Llama-3.1-8B-Instruct",
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
    ),
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
