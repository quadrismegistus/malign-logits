"""F11 cross-family: contradiction analysis across all families.

Caches logits for all contradiction prompts, then runs the analysis.

Usage:
    uv run python scripts/f11_cross_family.py
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from scipy.special import softmax

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.models import load_model, get_base_logits
from malign_logits.metrics import js_divergence

PAIRS = Psyche.DEFAULT_CONTRADICTIONS

FAMILIES = ['olmo', 'llama', 'amber', 'qwen', 'tulu', 'zephyr',
            'olmo-tiny', 'deepseek-7b', 'pythia', 'qwen-tiny', 'smol']


def cache_logits(families):
    """Cache logits for all contradiction prompts across families."""
    all_prompts = set()
    for pair in PAIRS:
        all_prompts.add(pair['prompt_a'])
        all_prompts.add(pair['prompt_b'])
        all_prompts.add(pair['prompt_ab'])

    for fkey in families:
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)

        # Which models need logits
        models_to_cache = [(fam.base, psyche.primary_process)]
        if fam.ego and psyche.ego:
            models_to_cache.append((fam.ego, psyche.ego))
        if fam.superego and psyche.superego:
            models_to_cache.append((fam.superego, psyche.superego))
        if fam.reinforced_superego and psyche.reinforced_superego:
            models_to_cache.append((fam.reinforced_superego, psyche.reinforced_superego))

        for mid, layer in models_to_cache:
            need = [p for p in all_prompts
                    if not (layer._cache and layer._cache.has_logits(mid, p))]
            if not need:
                continue

            print(f"  {fkey}/{mid.split('/')[-1]}: caching {len(need)} prompts...")
            model, tok = load_model(mid)
            for p in need:
                logits = get_base_logits(model, tok, p)
                layer._cache.set_logits(mid, p, logits.cpu().numpy())
            del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print("All logits cached.")


def run_analysis(families):
    """Run contradiction analysis from cached logits."""
    rows = []

    for fkey in families:
        psyche = Psyche.from_family(fkey)

        layers = [("base", psyche.primary_process)]
        if psyche.ego:
            layers.append(("sft", psyche.ego))
        if psyche.superego:
            layers.append(("dpo", psyche.superego))
        if psyche.reinforced_superego:
            layers.append(("rlvr", psyche.reinforced_superego))

        for pair in PAIRS:
            for layer_name, layer in layers:
                try:
                    la = layer.logits(pair['prompt_a']).numpy()
                    lb = layer.logits(pair['prompt_b']).numpy()
                    lab = layer.logits(pair['prompt_ab']).numpy()
                except Exception:
                    continue

                n = min(len(la), len(lb), len(lab))
                pa = softmax(la[:n].astype(np.float64))
                pb = softmax(lb[:n].astype(np.float64))
                pab = softmax(lab[:n].astype(np.float64))
                pmean = 0.5 * (pa + pb)

                js_ab_mean = js_divergence(lab[:n], np.log(pmean + 1e-15))
                # Use proper JS computation
                from scipy.spatial.distance import jensenshannon
                js_ab_a = float(jensenshannon(pab, pa) ** 2)
                js_ab_b = float(jensenshannon(pab, pb) ** 2)
                js_ab_m = float(jensenshannon(pab, pmean) ** 2)

                ratio = js_ab_m / min(js_ab_a, js_ab_b) if min(js_ab_a, js_ab_b) > 1e-10 else 0

                rows.append({
                    'family': fkey,
                    'pair': pair['name'],
                    'model': layer_name,
                    'js_to_A': js_ab_a,
                    'js_to_B': js_ab_b,
                    'js_to_mean': js_ab_m,
                    'ratio': ratio,
                })

    return pd.DataFrame(rows)


def print_results(df):
    print(f"\n{'='*80}")
    print(f"F11 CROSS-FAMILY: Contradiction tolerance (ratio < 1 = superposition)")
    print(f"{'='*80}")

    # Per-family mean ratio at base vs aligned
    for fkey in sorted(df.family.unique()):
        sub = df[df.family == fkey]
        base = sub[sub.model == 'base']
        aligned = sub[sub.model.isin(['dpo', 'sft'])]
        if aligned.empty:
            al = sub[sub.model != 'base']
        else:
            al = aligned

        # Use the outermost aligned layer
        for layer in ['dpo', 'sft', 'rlvr']:
            if layer in sub.model.values:
                al = sub[sub.model == layer]
                break

        if base.empty or al.empty:
            continue

        print(f"\n  {fkey:15s}  base={base.ratio.mean():.3f}  "
              f"aligned={al.ratio.mean():.3f}  "
              f"Δ={al.ratio.mean()-base.ratio.mean():+.3f}  "
              f"(n_pairs={len(base)})")

    # Per-pair across families
    print(f"\n{'='*80}")
    print(f"PER-PAIR (base ratio, mean across families):")
    for pair in sorted(df.pair.unique()):
        base = df[(df.pair == pair) & (df.model == 'base')]
        print(f"  {pair:25s}  mean_base_ratio={base.ratio.mean():.3f}  "
              f"std={base.ratio.std():.3f}  n={len(base)}")


def main():
    print("F11 cross-family contradiction analysis")
    print(f"Families: {FAMILIES}")
    print(f"Pairs: {len(PAIRS)}")

    print("\nCaching logits...")
    cache_logits(FAMILIES)

    print("\nRunning analysis...")
    df = run_analysis(FAMILIES)
    print_results(df)

    out = os.path.join(PATH_DATA, "contradiction_cross_family.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
