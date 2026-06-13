"""Generate formation trajectory figure for anger prompt using full discover_top_words pipeline."""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from malign_logits.psyche import Psyche

    import sys
    family = sys.argv[1] if len(sys.argv) > 1 else 'llama'
    psyche = Psyche.from_family(family, load=True)
    analysis = psyche.analyze("She was so angry she wanted to")

    print('Computing formation_df...')
    fdf = analysis.formation_df
    print(f'Got {len(fdf)} words')
    print(f'Columns: {list(fdf.columns)}')
    print(f'Trajectories:\n{fdf["trajectory"].value_counts()}')

    # Detect layer columns (prob columns, not delta columns)
    layer_cols = [c for c in fdf.columns if c in ['base', 'sft', 'dpo', 'rlvr',
                                                    'ego', 'superego', 'instruct']]
    X_LABELS = {'base': 'BASE', 'sft': 'SFT', 'dpo': 'DPO', 'rlvr': 'RLVR',
                'ego': 'SFT', 'superego': 'DPO', 'instruct': 'RLVR'}

    print(f'Layer columns: {layer_cols}')

    decline = fdf[fdf['trajectory'] == 'decline'].copy()
    decline['drop'] = decline[layer_cols[0]] - decline[layer_cols[-1]]
    decline = decline.nlargest(8, 'drop')

    rise = fdf[fdf['trajectory'] == 'rise'].copy()
    rise['gain'] = rise[layer_cols[-1]] - rise[layer_cols[0]]
    rise = rise.nlargest(8, 'gain')

    print('\nDECLINE:')
    for _, r in decline.iterrows():
        vals = '  '.join(f'{r[c]:.4f}' for c in layer_cols)
        print(f'  {r["word"]:12s}  {vals}')
    print('\nRISE:')
    for _, r in rise.iterrows():
        vals = '  '.join(f'{r[c]:.4f}' for c in layer_cols)
        print(f'  {r["word"]:12s}  {vals}')

    # Plot
    TRAJ_COLORS = {'decline': '#e63946', 'rise': '#457b9d'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    x = np.arange(len(layer_cols))

    for ax, traj, sub in [(axes[0], 'decline', decline), (axes[1], 'rise', rise)]:
        color = TRAJ_COLORS[traj]
        sub = sub.sort_values(layer_cols[-1], ascending=False)

        for _, row in sub.iterrows():
            vals = [row[l] for l in layer_cols]
            ax.plot(x, vals, '-o', color=color, linewidth=1.8, markersize=5, alpha=0.7)
            ax.text(x[-1] + 0.08, vals[-1], row['word'], fontsize=10, va='center',
                    color=color, fontweight='bold')

        n_words = len(sub)
        ax.set_title(f'{traj} (n={n_words})', fontsize=13, fontweight='bold', color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([X_LABELS[l] for l in layer_cols], fontsize=11)
        ax.set_yscale('log')
        ax.set_ylabel('probability' if ax == axes[0] else '', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.3, len(layer_cols) - 0.3)

    fig.suptitle('"She was so angry she wanted to ___"',
                 fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93,
             'Token probability trajectories across alignment stages · OLMo 3 7B',
             ha='center', fontsize=11, style='italic', color='#444')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/F01_anger_formation.png', dpi=200, bbox_inches='tight')
    print('\nSaved figures/F01_anger_formation.png')
