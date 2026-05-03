"""Cross-family prompt battery driver.

Loads each model family in turn, calls ``Psyche.battery_metrics()``, and
concatenates the per-family DataFrames into a single CSV. The per-family
metric computation lives on ``Psyche`` and ``experiments.run_prompt_battery``;
this module is the multi-family driver that handles model lifecycle.
"""
import gc

import pandas as pd
import torch

from . import MODEL_FAMILIES
from .psyche import Psyche


def run_battery(families=None, output_path=None):
    """Run the prompt battery across one or all model families.

    Args:
        families: List of family keys, or None for all registered families.
        output_path: CSV destination (default ``data/battery_results.csv``).

    Returns:
        DataFrame with columns ``[family, label, prompt, ...metrics]``.
    """
    keys = families if families else list(MODEL_FAMILIES.keys())
    output_path = output_path or "data/battery_results.csv"

    all_metrics = []
    for key in keys:
        fam = MODEL_FAMILIES[key]
        print(f"\n{'=' * 60}\n  {key}: {fam.name} ({fam.n_layers} layers)\n{'=' * 60}")
        psyche = Psyche.from_family(key, load=True)
        metrics = psyche.battery_metrics()
        metrics["family"] = key
        all_metrics.append(metrics)

        del psyche
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    combined = pd.concat(all_metrics, ignore_index=True)
    id_cols = ["family", "label", "prompt"]
    cols = id_cols + [c for c in combined.columns if c not in id_cols]
    combined = combined[cols]

    combined.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"\n{combined.to_string()}")
    return combined
