"""Tree analysis pipeline.

Run: python scripts/tree_analysis.py [--prompt anger] [--family olmo2-1b]

Exports:
    data/tree_stats.csv          — per model × prompt tree metrics
    data/tree_comparisons.csv    — base→aligned tree comparisons
    figures/F26_tree_*.png       — tree visualizations
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from malign_logits.probe import Probe, PROMPTS
from malign_logits.metrics import tree_metrics, tree_compare
from malign_logits.registry import Registry
from malign_logits.cache import get_cache


def get_models_with_data(prompt_text, max_tokens=10):
    cache = get_cache()
    s = cache._stash('probe_meta')
    models = set()
    for key in s:
        if (isinstance(key, dict) and key.get('T') == max_tokens
            and key.get('prompt') == prompt_text and key.get('gen') == 0):
            models.add(key['model'])
    return models


def export_tree_stats(prompts=None):
    reg = Registry()
    prompts = prompts or PROMPTS
    rows = []
    for pname, ptext in prompts.items():
        models = get_models_with_data(ptext)
        for model_id in sorted(models):
            t = tree_metrics(Probe(model_id), ptext, n_gens=100, max_tokens=10)
            if not t or t['n_gens'] < 10:
                continue
            info = reg.info(model_id)
            _, rel = reg.parent_of(model_id)
            base_id = reg.base_of(model_id)
            rows.append({
                'model': model_id, 'model_short': model_id.split('/')[-1],
                'family': base_id.split('/')[-1] if base_id else '',
                'relation': rel or 'base',
                'org': info.org if info else '',
                'prompt': pname,
                'n_gens': t['n_gens'], 'n_branches': t['n_branches'],
                'branch_entropy': round(t['branch_entropy'], 3),
                'top_branch': t['top_branch'],
                'top_branch_pct': round(t['top_branch_pct'], 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv('data/tree_stats.csv', index=False)
    print(f"Exported data/tree_stats.csv: {len(df)} rows")
    return df


def export_tree_comparisons(prompts=None):
    reg = Registry()
    prompts = prompts or PROMPTS
    rows = []
    for pname, ptext in prompts.items():
        models = get_models_with_data(ptext)
        for base_id in reg.all_bases():
            if base_id not in models:
                continue
            for v in reg.variants_of(base_id):
                if v not in models:
                    continue
                tc = tree_compare(Probe(base_id), Probe(v), ptext,
                                  n_gens=100, max_tokens=10)
                if not tc:
                    continue
                _, rel = reg.parent_of(v)
                rows.append({
                    'base': base_id, 'variant': v,
                    'relation': rel, 'prompt': pname,
                    'tree_js': round(tc['tree_js'], 4),
                    'branches_base': tc['n_branches_a'],
                    'branches_aligned': tc['n_branches_b'],
                    'H_base': round(tc['branch_entropy_a'], 3),
                    'H_aligned': round(tc['branch_entropy_b'], 3),
                    'n_novel': tc['n_novel'], 'n_pruned': tc['n_pruned'],
                    'n_repressed': len(tc['repressed']),
                    'n_amplified': len(tc['amplified']),
                })
    df = pd.DataFrame(rows)
    df.to_csv('data/tree_comparisons.csv', index=False)
    print(f"Exported data/tree_comparisons.csv: {len(df)} rows")
    return df


def plot_tree_entropy(stats_df, prompt='anger'):
    sub = stats_df[stats_df['prompt'] == prompt].sort_values('branch_entropy')
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'base': '#4e79a7', 'sft_of': '#f28e2b', 'dpo_of': '#e15759',
              'rlvr_of': '#59a14f', 'aligned_of': '#76b7b2'}
    for _, row in sub.iterrows():
        c = colors.get(row['relation'], '#999')
        ax.barh(row['model_short'], row['branch_entropy'], color=c, alpha=0.8)
    ax.set_xlabel('Branch entropy (bits)')
    ax.set_title(f'Tree breadth by model ({prompt})')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'figures/F26_tree_entropy_{prompt}.png', dpi=150)
    print(f"Saved figures/F26_tree_entropy_{prompt}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default=None)
    args = parser.parse_args()

    prompts = {args.prompt: PROMPTS[args.prompt]} if args.prompt else PROMPTS

    stats = export_tree_stats(prompts)
    comps = export_tree_comparisons(prompts)

    for pname in prompts:
        if pname in stats['prompt'].values:
            plot_tree_entropy(stats, pname)
