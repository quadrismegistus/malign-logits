"""SFT data-mixture ablation.

Compares base -> SFT shifts across Tulu 3 SFT checkpoints trained without
specific data subsets (no-safety, no-persona, no-math, etc.). Same base
model, same architecture, different SFT data mixtures. Isolates the
contribution of each data component to ego-stage displacement.
"""
import gc

import pandas as pd
import torch

from . import MODEL_FAMILIES, TULU_ABLATIONS
from .analysis import (
    _align_logits, distribution_entropy, js_divergence, rank_correlation,
    top_k_overlap,
)
from .experiments import DEFAULT_PROMPTS
from .models import load_model


def run_ablation(ablation_keys=None, output_path=None):
    """Run SFT-ablation comparison and write per-(ablation, prompt) metrics.

    Args:
        ablation_keys: Subset of ``TULU_ABLATIONS.keys()``. Default: all.
        output_path: CSV destination (default ``data/ablation_results.csv``).

    Returns:
        DataFrame with one row per (ablation, prompt) pair and columns:
        ``js_base_ego``, ``entropy_base``, ``entropy_ego``, ``entropy_drop``,
        ``top50_overlap``, ``rank_corr``.
    """
    base_id = MODEL_FAMILIES["tulu"].base
    keys = list(ablation_keys) if ablation_keys else list(TULU_ABLATIONS.keys())
    output_path = output_path or "data/ablation_results.csv"

    print(f"Loading base: {base_id}")
    base_model, base_tok = load_model(base_id)

    all_rows = []
    for abl_key in keys:
        if abl_key not in TULU_ABLATIONS:
            print(f"  Unknown ablation: {abl_key}, skipping")
            continue
        sft_id = TULU_ABLATIONS[abl_key]
        print(f"\n{'=' * 60}\n  {abl_key}: {sft_id}\n{'=' * 60}")

        sft_model, _ = load_model(sft_id)
        n_prompts = len(DEFAULT_PROMPTS)

        for j, (label, prompt) in enumerate(DEFAULT_PROMPTS.items()):
            print(f"    [{j+1}/{n_prompts}] {label}", flush=True)
            try:
                inputs = base_tok(prompt, return_tensors="pt").to(base_model.device)
                with torch.no_grad():
                    base_logits = base_model(**inputs).logits[0, -1, :]
                    sft_logits = sft_model(**inputs.to(sft_model.device)).logits[0, -1, :]

                base_l, sft_l = _align_logits(base_logits, sft_logits)
                ent_base = distribution_entropy(base_l)
                ent_sft = distribution_entropy(sft_l)
                all_rows.append({
                    "ablation": abl_key,
                    "label": label,
                    "prompt": prompt[:60],
                    "js_base_ego": js_divergence(base_l, sft_l),
                    "entropy_base": ent_base,
                    "entropy_ego": ent_sft,
                    "entropy_drop": ent_base - ent_sft,
                    "top50_overlap": top_k_overlap(base_l, sft_l, k=50),
                    "rank_corr": rank_correlation(base_l, sft_l),
                })
            except Exception as e:
                print(f"  Skipping {label}: {e}")

        del sft_model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    del base_model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    if not df.empty:
        summary = df.groupby("ablation")[["js_base_ego", "entropy_drop", "top50_overlap"]].mean()
        print(f"\n{summary.to_string(float_format='{:.4f}'.format)}")

    return df
