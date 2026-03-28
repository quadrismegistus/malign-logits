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

Functional interface::

    from malign_logits import load_models, discover_top_words, run_prompt_battery
    base, sft, dpo, tok = load_models()
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

PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
PATH_DATA = os.path.join(PATH_REPO, "data")
PATH_DATA_RAW = os.path.join(PATH_DATA, "raw")
PATH_STASH = os.path.join(PATH_DATA_RAW, "stash")
PATH_FIGURES = os.path.join(PATH_REPO, "figures")

# OLMo 3 — Allen AI (all intermediates released separately)
BASE_MODEL_NAME = "allenai/Olmo-3-1025-7B"
SFT_MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
DPO_MODEL_NAME = "allenai/Olmo-3-7B-Instruct-DPO"
INSTRUCT_MODEL_NAME = "allenai/Olmo-3-7B-Instruct"  # RLVR (final)


# Centralized intra-package imports.
# Order matters: later modules depend on names defined by earlier ones.
from .core import *
from .models import *
from .analysis import *
from .experiments import *
from .generation import *
from .viz import *
from .psyche import *
