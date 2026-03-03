"""
libidinal_toolkit.py
====================

A toolkit for psychoanalytic analysis of LLM probability distributions.

Compares base models (primary process), instruct models without system prompts
(ego), and instruct models with restrictive system prompts (superego) to map
the repression, displacement, and condensation signatures of AI alignment.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures
for AI" (Accelerationism Revisited, UCD, June 2026).

Requirements:
    pip install transformers accelerate bitsandbytes torch pandas tqdm

Usage:
    See notebooks/ directory for worked examples, or run:

        from malign_logits import load_models, discover_top_words, run_prompt_battery
        base, instruct, tok = load_models()
        results = run_prompt_battery(base, instruct, tok)
"""

__version__ = "0.1.0"

# Centralized imports
import math
import re
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm

# Import all functions from submodules
from .models import *
from .core import *
from .analysis import *
from .experiments import *
