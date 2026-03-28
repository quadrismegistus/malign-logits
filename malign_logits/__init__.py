"""
malign-logits
=============

A toolkit for psychoanalytic analysis of LLM probability distributions.

Compares base models (primary process), instruct models without system prompts
(ego), and instruct models with restrictive system prompts (superego) to map
the repression, displacement, and condensation signatures of AI alignment.

Quick start (OO interface)::

    from malign_logits import Psyche
    psyche = Psyche.from_pretrained()

    s = psyche.analyze("He lay naked in his bed and")
    s.repression          # DataFrame of ego-superego deltas
    s.id_scores           # drive-weighted repression scores
    s.analysis_df         # full combined DataFrame

    result = psyche.generate("She knelt down...", displacement_weight=0.3)

Functional interface (all original functions still available)::

    from malign_logits import load_models, load_three_models, discover_top_words, run_prompt_battery
    base, instruct, tok = load_models()
    results = run_prompt_battery(instruct, tok, base_model=base)
"""

__version__ = "0.1.0"

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
from collections import defaultdict


import warnings
warnings.filterwarnings("ignore")

PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
PATH_DATA = os.path.join(PATH_REPO, "data")
PATH_DATA_RAW = os.path.join(PATH_DATA, "raw")
PATH_STASH = os.path.join(PATH_DATA_RAW, "stash")
PATH_FIGURES = os.path.join(PATH_REPO, "figures")

BASE_MODEL_NAME = "LLM360/Amber"
INSTRUCT_MODEL_NAME = "LLM360/AmberChat"
SAFE_MODEL_NAME = "LLM360/AmberSafe"


# Centralized intra-package imports.
# Order matters: later modules depend on names defined by earlier ones.
from .core import *
from .models import *
from .analysis import *
from .experiments import *
from .generation import *
from .viz import *
from .psyche import *
