"""F36 minimal-pair battery: beam resistance analysis.

Runs beam_storylines + annotate_beams on all 84 minimal-pair prompts
for key families. Reports within-pair resistance differences.

Usage:
    uv run python scripts/f36_minimal_beams.py --save
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from malign_logits import PATH_DATA
from malign_logits.beam import annotate_beams
from scripts.f36_minimal_pairs import BATTERY


FAMILIES = {
    'olmo': {
        'base': 'allenai/Olmo-3-1025-7B',
        'annotators': [
            'allenai/Olmo-3-7B-Instruct-SFT',
            'allenai/Olmo-3-7B-Instruct-DPO',
        ],
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


def run():
    from malign_logits.cache import open_stash
    stash = open_stash(os.path.join(PATH_DATA, "raw", "cache", "beams"))

    all_rows = []

    for fkey, fconf in FAMILIES.items():
        base_id = fconf['base']
        annotators = fconf['annotators']

        print(f"\n{'='*60}")
        print(f"{fkey}: {base_id}")
        print(f"  annotators: {annotators}")
        print(f"  {len(BATTERY)} prompts × {N_BEAMS} beams × {len(annotators)} annotators")
        print(f"{'='*60}")

        for i, entry in enumerate(BATTERY):
            prompt = entry['prompt']

            # Check cache
            cache_key = {
                "type": "beam_minimal_v1",
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
                    stories_data = []
                    for s in stories:
                        sd = {
                            "text": s.text,
                            "path_prob": s.path_prob,
                            "base_token_probs": s.base_token_probs if hasattr(s, 'base_token_probs') else [],
                            "annotations": s.annotations,
                        }
                        stories_data.append(sd)
                    stash[cache_key] = stories_data
                except Exception as e:
                    print(f"  SKIP {prompt[:40]}: {e}")
                    continue

            # Extract resistance per annotator
            for sd in stories_data:
                for ann_name, ann_data in sd.get('annotations', {}).items():
                    if not isinstance(ann_data, dict):
                        continue
                    mr = ann_data.get('mean_resist', None)
                    if mr is None:
                        continue
                    all_rows.append({
                        'family': fkey,
                        'annotator': ann_name,
                        'pair': entry['pair'],
                        'prompt': prompt,
                        'transgression': entry['transgression'],
                        'trans_level': entry['trans_level'],
                        'valence': entry['valence'],
                        'swap': entry['swap'],
                        'mean_resist': mr,
                        'total_resist': ann_data.get('total_resist', 0),
                        'text': sd.get('text', '')[:80],
                    })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(BATTERY)} prompts done")

        # Free GPU between families
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return pd.DataFrame(all_rows)


def print_results(df):
    df['is_trans'] = ~df.transgression.isin(['benign', 'benign_high'])

    print("\n" + "=" * 80)
    print("BEAM RESISTANCE: WITHIN-PAIR ANALYSIS (single-swap, primary)")
    print("Paired difference: transgressive mean_resist - benign mean_resist")
    print("Positive = MORE resistance on transgressive (content-specific suppression)")
    print("=" * 80)

    singles = df[df.swap == 'single']

    for fkey in sorted(singles.family.unique()):
        fsub = singles[singles.family == fkey]
        pair_diffs = []

        for pair in fsub.pair.unique():
            psub = fsub[fsub.pair == pair]
            t = psub[psub.is_trans]
            b = psub[~psub.is_trans]
            if t.empty or b.empty:
                continue
            pair_diffs.append(t.mean_resist.mean() - b.mean_resist.mean())

        if not pair_diffs:
            continue

        d = np.array(pair_diffs)
        if len(d) >= 5:
            stat, p = wilcoxon(d, alternative='two-sided')
        else:
            p = np.nan
        sig = "*" if p < 0.05 else ""

        print(f"\n  {fkey:12s}  n_pairs={len(d)}")
        print(f"    resist diff (trans-benign):  mean={d.mean():+.4f}  "
              f"median={np.median(d):+.4f}  p={p:.4f}{sig}")

    # Per-category
    print("\n  Per-category (pooled, single-swap):")
    for cat_prefix, cat_name in [('v', 'violence'), ('s', 'sexual'),
                                   ('sub', 'substance'), ('p', 'profanity'),
                                   ('d', 'death')]:
        if cat_prefix in ('v', 's', 'd'):
            csub = singles[singles.pair.str.startswith(cat_prefix) &
                           ~singles.pair.str.startswith(cat_prefix + '_')]
        elif cat_prefix == 'sub':
            csub = singles[singles.pair.str.startswith('sub') &
                           ~singles.pair.str.startswith('sub_')]
        else:
            csub = singles[singles.pair.str.startswith(cat_prefix) &
                           ~singles.pair.str.startswith(cat_prefix + '_')]

        diffs = []
        for fkey in csub.family.unique():
            fsub = csub[csub.family == fkey]
            for pair in fsub.pair.unique():
                psub = fsub[fsub.pair == pair]
                t = psub[psub.is_trans]
                b = psub[~psub.is_trans]
                if t.empty or b.empty:
                    continue
                diffs.append(t.mean_resist.mean() - b.mean_resist.mean())

        if diffs:
            d = np.array(diffs)
            if len(d) >= 5:
                _, p = wilcoxon(d, alternative='two-sided')
            else:
                p = np.nan
            print(f"    {cat_name:12s}  n={len(d):3d}  diff={d.mean():+.4f}  "
                  f"median={np.median(d):+.4f}  p={p:.4f}")

    # Aggregate
    print("\n  AGGREGATE (all single-swap pairs, all families):")
    all_diffs = []
    for fkey in singles.family.unique():
        fsub = singles[singles.family == fkey]
        for pair in fsub.pair.unique():
            psub = fsub[fsub.pair == pair]
            t = psub[psub.is_trans]
            b = psub[~psub.is_trans]
            if t.empty or b.empty:
                continue
            all_diffs.append(t.mean_resist.mean() - b.mean_resist.mean())
    if all_diffs:
        d = np.array(all_diffs)
        _, p = wilcoxon(d, alternative='two-sided')
        print(f"    n={len(d)}  diff={d.mean():+.4f}  median={np.median(d):+.4f}  p={p:.4f}")


def main():
    print("F36 minimal-pair beam resistance")
    df = run()
    print_results(df)

    out = os.path.join(PATH_DATA, "f36_minimal_beams.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
