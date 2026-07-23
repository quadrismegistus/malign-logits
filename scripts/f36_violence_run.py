"""P1: Violence battery — token survival + span resistance + reroute characterization.

Runs P1.2 (token survival from logits), P1.3 (beam resistance), and
collects data for P1.4 (reroute characterization) and P1.5 (regression).

Usage:
    uv run python scripts/f36_violence_run.py
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import wilcoxon

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.models import load_model, get_base_logits
from malign_logits.beam import annotate_beams
from malign_logits.cache import open_stash
from scripts.f36_violence_battery import BATTERY


FAMILIES = {
    'olmo': {
        'base': 'allenai/Olmo-3-1025-7B',
        'annotators': ['allenai/Olmo-3-7B-Instruct-DPO'],
    },
    'llama': {
        'base': 'meta-llama/Llama-3.1-8B',
        'annotators': ['meta-llama/Llama-3.1-8B-Instruct'],
    },
    'amber': {
        'base': 'LLM360/Amber',
        'annotators': ['LLM360/AmberSafe'],
    },
    'olmo-tiny': {
        'base': 'allenai/OLMo-2-0425-1B',
        'annotators': ['allenai/OLMo-2-0425-1B-DPO'],
    },
}

N_BEAMS = 50
MAX_TOKENS = 10


def run_token_survival():
    """P1.2: Token survival — does the violent token stay in the aligned top-K?"""
    print("=" * 70)
    print("P1.2: TOKEN SURVIVAL")
    print("=" * 70)

    rows = []
    for fkey, fconf in FAMILIES.items():
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        # Cache logits for all battery prompts
        need_cache = False
        for entry in BATTERY:
            if not psyche.primary_process._cache.has_logits(fam.base, entry['prompt']):
                need_cache = True
                break

        if need_cache:
            print(f"  {fkey}: caching logits...")
            base_model, base_tok = load_model(fam.base)
            aligned_model, _ = load_model(fconf['annotators'][0])
            for entry in BATTERY:
                for mid, model in [(fam.base, base_model),
                                    (fconf['annotators'][0], aligned_model)]:
                    if not psyche.primary_process._cache.has_logits(mid, entry['prompt']):
                        logits = get_base_logits(model, base_tok, entry['prompt'])
                        psyche.primary_process._cache.set_logits(
                            mid, entry['prompt'], logits.cpu().numpy())
            del base_model, aligned_model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"    cached")
        else:
            print(f"  {fkey}: logits cached")

        for entry in BATTERY:
            try:
                bl = psyche.primary_process.logits(entry['prompt']).numpy()
                al = aligned.logits(entry['prompt']).numpy()
            except Exception:
                continue

            n = min(len(bl), len(al))
            bp = softmax(bl[:n].astype(np.float64))
            ap = softmax(al[:n].astype(np.float64))
            f_id = int(np.argmax(bp))
            a_id = int(np.argmax(ap))

            if f_id >= n:
                continue

            ranking = np.argsort(-ap)
            f_rank = int(np.where(ranking == f_id)[0][0])

            rows.append({
                'family': fkey,
                'prompt_id': entry.get('id', entry.get('pair', '')),
                'prompt': entry['prompt'],
                'category': entry['category'],
                'subcategory': entry.get('subcategory', ''),
                'intensity': entry['intensity'],
                'is_violence': 'violence' in entry['category'],
                'f_token': tok.decode([f_id]).strip(),
                'a_token': tok.decode([a_id]).strip(),
                'f_prob_base': float(bp[f_id]),
                'f_prob_aligned': float(ap[f_id]),
                'f_rank_aligned': f_rank,
                'displaced': f_id != a_id,
                'ratio': float(ap[f_id] / bp[f_id]) if bp[f_id] > 1e-10 else 0,
            })

    return pd.DataFrame(rows)


def run_beam_resistance():
    """P1.3: Span resistance via beam teacher-forcing."""
    print("\n" + "=" * 70)
    print("P1.3: SPAN RESISTANCE (beams)")
    print("=" * 70)

    stash = open_stash(os.path.join(PATH_DATA, "raw", "cache", "beams"))
    rows = []

    for fkey, fconf in FAMILIES.items():
        base_id = fconf['base']
        annotators = fconf['annotators']
        print(f"\n  {fkey}:")

        for i, entry in enumerate(BATTERY):
            prompt = entry['prompt']
            cache_key = {
                "type": "beam_violence_v1",
                "model": base_id,
                "prompt": prompt,
                "n_beams": N_BEAMS,
                "max_tokens": MAX_TOKENS,
            }

            if cache_key in stash:
                stories_data = stash[cache_key]
            else:
                try:
                    stories = annotate_beams(
                        base_id, prompt, n=N_BEAMS,
                        max_tokens=MAX_TOKENS,
                        annotators=annotators,
                    )
                    stories_data = [{
                        "text": s.text,
                        "path_prob": s.path_prob,
                        "token_texts": s.token_texts if hasattr(s, 'token_texts') else [],
                        "annotations": s.annotations,
                    } for s in stories]
                    stash[cache_key] = stories_data
                except Exception as e:
                    print(f"    SKIP {prompt[:40]}: {e}")
                    continue

            for sd in stories_data:
                for ann_name, ann_data in sd.get('annotations', {}).items():
                    if not isinstance(ann_data, dict):
                        continue
                    mr = ann_data.get('mean_resist', None)
                    if mr is None:
                        continue
                    rows.append({
                        'family': fkey,
                        'prompt_id': entry.get('id', entry.get('pair', '')),
                        'prompt': prompt,
                        'category': entry['category'],
                        'subcategory': entry.get('subcategory', ''),
                        'intensity': entry['intensity'],
                        'is_violence': 'violence' in entry['category'],
                        'mean_resist': mr,
                        'total_resist': ann_data.get('total_resist', 0),
                        'text': sd.get('text', '')[:100],
                    })

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(BATTERY)}")

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return pd.DataFrame(rows)


def print_results(tok_df, beam_df):
    """Print P1.2, P1.3, P1.5 results."""

    # ── P1.2 Token survival ──
    print("\n" + "=" * 70)
    print("P1.2: TOKEN SURVIVAL — base levels + aligned rank")
    print("=" * 70)

    violence = tok_df[tok_df.is_violence == True]
    nonviolent = tok_df[tok_df.is_violence == False]

    for label, sub in [("violence", violence), ("nonviolent", nonviolent)]:
        disp = sub[sub.displaced == True]
        print(f"\n  {label} (n={len(sub)}, displaced={len(disp)}):")
        print(f"    f_prob_base:    mean={sub.f_prob_base.mean():.4f}")
        print(f"    f_prob_aligned: mean={sub.f_prob_aligned.mean():.4f}")
        print(f"    ratio:          mean={sub.ratio.mean():.3f}")
        print(f"    f_rank_aligned: median={sub.f_rank_aligned.median():.0f}  "
              f"top10={100*(sub.f_rank_aligned < 10).mean():.0f}%")

    # ── P1.3 Span resistance ──
    print("\n" + "=" * 70)
    print("P1.3: SPAN RESISTANCE")
    print("=" * 70)

    for label, sub in [("violence", beam_df[beam_df.is_violence]),
                        ("nonviolent_high", beam_df[beam_df.category == 'nonviolent_high']),
                        ("neutral", beam_df[beam_df.category == 'neutral'])]:
        if sub.empty:
            continue
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    mean_resist: {sub.mean_resist.mean():+.4f}  "
              f"median={sub.mean_resist.median():+.4f}")

    # Per-subcategory
    if 'subcategory' in beam_df.columns:
        print(f"\n  By subcategory:")
        for subcat in sorted(beam_df.subcategory.dropna().unique()):
            if not subcat:
                continue
            sub = beam_df[beam_df.subcategory == subcat]
            print(f"    {subcat:15s}  n={len(sub):5d}  "
                  f"resist={sub.mean_resist.mean():+.4f}")

    # ── P1.5 THE DECISIVE TEST ──
    print("\n" + "=" * 70)
    print("P1.5: VIOLENCE COEFFICIENT (resistance ~ violence + intensity)")
    print("=" * 70)

    # Per-prompt mean resistance
    prompt_means = beam_df.groupby(['prompt_id', 'category', 'intensity',
                                     'is_violence', 'family']).agg(
        resist=('mean_resist', 'mean')).reset_index()

    for fam in sorted(prompt_means.family.unique()):
        fsub = prompt_means[prompt_means.family == fam]
        viol = fsub[fsub.is_violence == True]
        nonv = fsub[fsub.is_violence == False]
        if viol.empty or nonv.empty:
            continue

        # Simple: violence vs nonviolent mean
        v_mean = viol.resist.mean()
        n_mean = nonv.resist.mean()

        # Intensity-controlled: OLS
        from sklearn.linear_model import LinearRegression
        X = fsub[['is_violence', 'intensity']].values.astype(float)
        y = fsub['resist'].values
        if len(X) >= 5:
            reg = LinearRegression().fit(X, y)
            v_coef = reg.coef_[0]
            i_coef = reg.coef_[1]
        else:
            v_coef, i_coef = np.nan, np.nan

        print(f"\n  {fam}:")
        print(f"    raw: violence={v_mean:+.4f}  nonviolent={n_mean:+.4f}  "
              f"diff={v_mean - n_mean:+.4f}")
        print(f"    intensity-controlled: violence_coef={v_coef:+.4f}  "
              f"intensity_coef={i_coef:+.4f}")


def main():
    tok_df = run_token_survival()
    beam_df = run_beam_resistance()
    print_results(tok_df, beam_df)

    # Save
    tok_out = os.path.join(PATH_DATA, "f36_violence_tokens.csv")
    tok_df.to_csv(tok_out, index=False)
    beam_out = os.path.join(PATH_DATA, "f36_violence_beams.csv")
    beam_df.to_csv(beam_out, index=False)
    print(f"\nSaved {len(tok_df)} token rows to {tok_out}")
    print(f"Saved {len(beam_df)} beam rows to {beam_out}")


if __name__ == "__main__":
    main()
