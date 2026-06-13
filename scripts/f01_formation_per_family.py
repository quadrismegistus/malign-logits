"""Formation trajectories per family for a given prompt."""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc

if __name__ == '__main__':
    from malign_logits.psyche import Psyche
    from malign_logits import MODEL_FAMILIES

    prompt = sys.argv[1] if len(sys.argv) > 1 else "She was so angry she wanted to"
    FAMILIES = ['olmo', 'tulu', 'zephyr', 'pythia', 'olmo-tiny', 'amber']

    X_LABELS = {'base': 'BASE', 'ego': 'SFT', 'sft': 'SFT', 'superego': 'DPO',
                'dpo': 'DPO', 'instruct': 'RLVR', 'rlvr': 'RLVR'}
    TRAJ_COLORS = {'decline': '#e63946', 'rise': '#457b9d'}

    for fam_key in FAMILIES:
        print(f'\n=== {fam_key} ===')

        # Try cache first
        try:
            psyche = Psyche.from_family(fam_key, load=False)
            analysis = psyche.analyze(prompt)
            fdf = analysis.formation_df
            print(f'  Cached: {len(fdf)} words')
        except Exception:
            print(f'  Loading models...')
            psyche = Psyche.from_family(fam_key, load=True)
            analysis = psyche.analyze(prompt)
            fdf = analysis.formation_df
            print(f'  Computed: {len(fdf)} words')

        layer_cols = [c for c in fdf.columns if c in X_LABELS]
        print(f'  Layers: {layer_cols}')
        print(f'  Trajectories: {dict(fdf["trajectory"].value_counts())}')

        decline = fdf[fdf['trajectory'] == 'decline'].copy()
        if len(decline):
            decline['drop'] = decline[layer_cols[0]] - decline[layer_cols[-1]]
            decline = decline.nlargest(8, 'drop')

        rise = fdf[fdf['trajectory'] == 'rise'].copy()
        if len(rise):
            rise['gain'] = rise[layer_cols[-1]] - rise[layer_cols[0]]
            rise = rise.nlargest(8, 'gain')

        # Also grab 'peak' and 'V' if they exist
        peak = fdf[fdf['trajectory'] == 'peak'].copy()
        if len(peak):
            peak['max_val'] = peak[layer_cols].max(axis=1)
            peak = peak.nlargest(4, 'max_val')

        eliminated = fdf[fdf['trajectory'] == 'eliminated'].copy()
        if len(eliminated):
            eliminated['drop'] = eliminated[layer_cols[0]] - eliminated[layer_cols[-1]]
            eliminated = eliminated.nlargest(8, 'drop')

        # How many panels?
        panels = []
        if len(decline): panels.append(('decline', decline))
        if len(rise): panels.append(('rise', rise))
        if len(peak): panels.append(('peak', peak))
        if len(eliminated): panels.append(('eliminated', eliminated))

        if not panels:
            print(f'  No decline/rise/peak/eliminated words — skipping')
            continue

        ncols = min(len(panels), 3)
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        x = np.arange(len(layer_cols))
        EXTRA_COLORS = {'peak': '#2a9d8f', 'eliminated': '#9b2226'}

        for i, (traj, sub) in enumerate(panels):
            ax = axes[i]
            color = TRAJ_COLORS.get(traj, EXTRA_COLORS.get(traj, '#888'))
            sub = sub.sort_values(layer_cols[-1], ascending=False)

            for _, row in sub.iterrows():
                vals = [row[l] for l in layer_cols]
                ax.plot(x, vals, '-o', color=color, linewidth=1.8, markersize=5, alpha=0.7)
                ax.text(x[-1] + 0.08, vals[-1], row['word'], fontsize=9, va='center',
                        color=color, fontweight='bold')

            n_words = len(sub)
            ax.set_title(f'{traj} (n={n_words})', fontsize=12, fontweight='bold', color=color)
            ax.set_xticks(x)
            ax.set_xticklabels([X_LABELS[l] for l in layer_cols], fontsize=10)
            ax.set_yscale('log')
            if i % ncols == 0:
                ax.set_ylabel('probability', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(-0.3, len(layer_cols) - 0.3)

        # Hide unused axes
        for j in range(len(panels), len(axes)):
            axes[j].set_visible(False)

        short_prompt = prompt if len(prompt) < 50 else prompt[:47] + '...'
        fig.suptitle(f'"{short_prompt} ___" · {fam_key}',
                     fontsize=14, fontweight='bold', y=1.01)

        plt.tight_layout()
        outpath = f'figures/F01_formation_{fam_key}.png'
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {outpath}')

        del psyche, analysis
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print('\nDone.')
