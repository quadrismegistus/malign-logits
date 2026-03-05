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

    from malign_logits import load_models, discover_top_words, run_prompt_battery
    base, instruct, tok = load_models()
    results = run_prompt_battery(instruct, tok, base_model=base)
"""

__version__ = "0.1.0"

# OO interface
from .psyche import (
    ModelLayer,
    PrimaryProcess,
    Ego,
    Superego,
    Psyche,
    PromptAnalysis,
)

# Functional interface — all original functions
from .models import load_models, get_base_logits, get_embeddings
from .core import (
    DEFAULT_SUPEREGO_PREFIX,
    make_superego_prompt,
    discover_top_words,
    get_word_logprobs,
)
from .analysis import (
    compute_repression,
    compute_id,
    compute_displacement,
    build_analysis_df,
    measure_overdetermination,
)
from .experiments import (
    DEFAULT_PROMPTS,
    run_prompt_battery,
    summarize_battery,
    print_repression_report,
)
from .generation import generate_neurotic, generate_neurotic_contextual
