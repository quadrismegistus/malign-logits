"""Drive all data-production tasks for the paper.

Loads each family once, runs battery / generation / taxonomy under it,
then runs cross-family analyses (logit lens, ablation, embedding metrics)
that do their own model lifecycle. Each phase delegates to the
corresponding analysis module so cli.py stays thin.
"""
import gc
import time

import pandas as pd
import torch

from . import MODEL_FAMILIES
from .experiments import DEFAULT_PROMPTS, TIER1_PROMPTS
from .psyche import Psyche


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def produce_all(families=None, skip=None, gen_n=30):
    """Run all data-production tasks across families.

    Args:
        families: List of family keys, or None for all registered.
        skip: Set of phase names to skip
            (``battery``, ``generate``, ``taxonomy``, ``logit-lens``,
            ``ablation``).
        gen_n: Generations per prompt during the generation phase.

    Returns:
        Dict mapping per-phase keys to ``"done"`` or an error string.
    """
    keys = families if families else list(MODEL_FAMILIES.keys())
    skip = set(skip) if skip else set()
    results = {}
    t0 = time.time()

    # ── Phase 1: per-family tasks (models loaded once per family) ──
    all_battery = []
    for key in keys:
        fam = MODEL_FAMILIES[key]
        print(f"\n{'=' * 60}\n  {key}: {fam.name} ({fam.n_layers} layers)\n{'=' * 60}")
        psyche = Psyche.from_family(key, load=True)

        if "battery" not in skip:
            print(f"\n  ── Battery ({key}) ──")
            try:
                metrics = psyche.battery_metrics()
                metrics["family"] = key
                all_battery.append(metrics)
                metrics.to_csv(f"data/battery_{key}.csv", index=False)
                print(f"  Saved data/battery_{key}.csv ({len(metrics)} prompts)")
                results[f"battery-{key}"] = "done"
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"battery-{key}"] = f"error: {e}"

        if "generate" not in skip:
            print(f"\n  ── Generation ({key}) ──")
            try:
                from .embedding import generate_many
                for label, prompt in TIER1_PROMPTS.items():
                    print(f"    {label}: {prompt[:40]}...")
                    generate_many(psyche, prompt, n=gen_n, max_new_tokens=100)
                results[f"generate-{key}"] = "done"
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"generate-{key}"] = f"error: {e}"

        if "taxonomy" not in skip:
            print(f"\n  ── Taxonomy ({key}) ──")
            try:
                from .taxonomy import run_taxonomy
                run_taxonomy(
                    family_key=key, all_prompts=True,
                    output_path=f"data/taxonomy_{key}.csv",
                    psyche=psyche,
                )
                results[f"taxonomy-{key}"] = "done"
            except ImportError:
                print(f"  Skipping taxonomy (spacy/wordfreq not installed)")
                results[f"taxonomy-{key}"] = "skipped (missing deps)"
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"taxonomy-{key}"] = f"error: {e}"

        del psyche
        _free()

    if all_battery and "battery" not in skip:
        combined = pd.concat(all_battery, ignore_index=True)
        id_cols = ["family", "label", "prompt"]
        cols = id_cols + [c for c in combined.columns if c not in id_cols]
        combined = combined[cols]
        combined.to_csv("data/battery_results.csv", index=False)
        print(f"\nCombined battery: data/battery_results.csv ({len(combined)} rows)")

    # ── Phase 2: logit lens (caches to stash; one psyche per family) ──
    if "logit-lens" not in skip:
        lens_prompts = list(TIER1_PROMPTS.items())[:6]
        for key in keys:
            psyche = Psyche.from_family(key, load=True)
            for label, prompt in lens_prompts:
                print(f"\n  ── Logit lens: {key} / {label} ──")
                try:
                    analysis = psyche.analyze(prompt)
                    data = analysis.logit_lens_df
                    print(f"  {len(data['rows'])} data points, {len(data['word_sources'])} tracked")
                    results[f"logit-lens-{key}-{label}"] = "done"
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[f"logit-lens-{key}-{label}"] = f"error: {e}"
            del psyche
            _free()

    # ── Phase 3: ablation (loads base once, swaps SFT variants) ──
    if "ablation" not in skip and "tulu" in keys:
        print(f"\n{'=' * 60}\n  SFT Ablation comparison\n{'=' * 60}")
        try:
            from .ablation import run_ablation
            run_ablation()
            results["ablation"] = "done"
        except Exception as e:
            print(f"  ERROR: {e}")
            results["ablation"] = f"error: {e}"

    # ── Phase 4: embed + compute generation metrics on cached generations ──
    if "generate" not in skip:
        print(f"\n{'=' * 60}\n  Embedding + generation metrics\n{'=' * 60}")
        try:
            from .embedding import (
                compute_concept_metrics, compute_generation_metrics, embed_generations,
            )
            embed_generations()
            gen_metrics = compute_generation_metrics()
            compute_concept_metrics()
            if gen_metrics is not None:
                gen_metrics.to_csv("data/gen_battery_metrics.csv", index=False)
                print(f"  Saved data/gen_battery_metrics.csv")
            results["embed-metrics"] = "done"
        except Exception as e:
            print(f"  ERROR: {e}")
            results["embed-metrics"] = f"error: {e}"

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}\n  ALL TASKS COMPLETE ({elapsed / 3600:.1f}h)\n{'=' * 60}")
    for k, v in sorted(results.items()):
        status = "✓" if v == "done" else "✗"
        print(f"  {status} {k:30s} {v}")

    return results
