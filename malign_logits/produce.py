"""Drive all data-production tasks for the paper.

Loads each family once, runs battery / generation / taxonomy under it,
then runs cross-family analyses (logit lens, ablation, embedding metrics)
that do their own model lifecycle. Each phase delegates to the
corresponding analysis module so cli.py stays thin.
"""
import gc
import os
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


def _exists(path, force=False):
    if not force and os.path.exists(path):
        print(f"  Skipping (exists): {path}")
        return True
    return False


def produce_all(families=None, skip=None, gen_n=30, force=False):
    """Run all data-production tasks across families.

    Args:
        families: List of family keys, or None for all registered.
        skip: Set of phase names to skip
            (``battery``, ``generate``, ``taxonomy``, ``trajectory``,
            ``logit-lens``, ``ablation``).
        gen_n: Generations per prompt during the generation phase.
        force: Recompute even if output CSVs already exist.

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
            csv_path = f"data/battery_{key}.csv"
            print(f"\n  ── Battery ({key}) ──")
            if _exists(csv_path, force):
                all_battery.append(pd.read_csv(csv_path))
                results[f"battery-{key}"] = "done"
            else:
                try:
                    metrics = psyche.battery_metrics()
                    metrics["family"] = key
                    all_battery.append(metrics)
                    metrics.to_csv(csv_path, index=False)
                    print(f"  Saved {csv_path} ({len(metrics)} prompts)")
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
            csv_path = f"data/taxonomy_{key}.csv"
            print(f"\n  ── Taxonomy ({key}) ──")
            if _exists(csv_path, force):
                results[f"taxonomy-{key}"] = "done"
            else:
                try:
                    from .taxonomy import run_taxonomy
                    run_taxonomy(
                        family_key=key, all_prompts=True,
                        output_path=csv_path,
                        psyche=psyche,
                    )
                    results[f"taxonomy-{key}"] = "done"
                except ImportError:
                    print(f"  Skipping taxonomy (spacy/wordfreq not installed)")
                    results[f"taxonomy-{key}"] = "skipped (missing deps)"
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[f"taxonomy-{key}"] = f"error: {e}"

        if "trajectory" not in skip:
            geom_path = f"data/trajectory_geometry_{key}.csv"
            int_path = f"data/intervention_{key}.csv"
            print(f"\n  ── Trajectory ({key}) ──")
            if _exists(geom_path, force) and (psyche.superego is None or _exists(int_path, force)):
                results[f"trajectory-{key}"] = "done"
            else:
                try:
                    from .trajectory import run_trajectory_geometry, run_intervention
                    n_hidden = psyche.primary_process.model.config.num_hidden_layers
                    layer = round(n_hidden * 0.8125)
                    intervention_layers = [round(n_hidden * f) for f in (0.25, 0.5, 0.75, 0.875)]
                    if force or not os.path.exists(geom_path):
                        run_trajectory_geometry(psyche, key, layer, out_dir="data")
                    if psyche.superego is not None and (force or not os.path.exists(int_path)):
                        run_intervention(psyche, key, intervention_layers, out_dir="data")
                    results[f"trajectory-{key}"] = "done"
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[f"trajectory-{key}"] = f"error: {e}"

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
        if _exists("data/ablation_results.csv", force):
            results["ablation"] = "done"
        else:
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
                compute_concept_metrics, compute_generation_metrics,
                embed_generations, load_generations_from_stash,
            )
            psg_df = load_generations_from_stash()
            if families:
                psg_df = psg_df[psg_df["family"].isin(keys)].reset_index(drop=True)
            if psg_df.empty:
                raise RuntimeError("no cached generations found to embed")
            print(f"  {len(psg_df)} cached generations")
            embeds_df = embed_generations(psg_df)
            metrics_rows = []
            for (fam_key, label), idx in psg_df.groupby(["family", "label"]).groups.items():
                sub_psg = psg_df.loc[idx].reset_index(drop=True)
                sub_emb = embeds_df.loc[idx].reset_index(drop=True)
                m = compute_generation_metrics(sub_emb, sub_psg)
                m.update(compute_concept_metrics(sub_emb, sub_psg))
                m["family"] = fam_key
                m["label"] = label
                m["n_generations"] = len(sub_psg)
                metrics_rows.append(m)
            gen_metrics = pd.DataFrame(metrics_rows)
            gen_metrics.to_csv("data/gen_battery_metrics.csv", index=False)
            print(f"  Saved data/gen_battery_metrics.csv ({len(gen_metrics)} rows)")
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
