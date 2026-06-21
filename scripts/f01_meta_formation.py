"""Meta formation: average token trajectories across all families.

Uses full discover_top_words pipeline (200 forward passes per layer).
Checks cache before loading models. Maps to 3 canonical positions:
BASE (all), SFT (3+ layer families only), ALIGNED (all — final checkpoint).
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc

if __name__ == '__main__':
    from malign_logits.psyche import Psyche
    from malign_logits import MODEL_FAMILIES

    prompt = sys.argv[1] if len(sys.argv) > 1 else "She kneeled and reached for his"

    ALL_FAMILIES = sorted(MODEL_FAMILIES.keys())
    POSITIONS = {0: 'BASE', 1: 'SFT', 2: 'ALIGNED'}

    all_fdfs = {}

    for fam_key in ALL_FAMILIES:
        fam = MODEL_FAMILIES[fam_key]
        n_layers = sum(1 for x in [fam.base, fam.ego, fam.superego, fam.reinforced_superego] if x)
        print(f'\n=== {fam_key} ({n_layers}L) ===')

        # Try cache first
        try:
            psyche = Psyche.from_family(fam_key, load=False)
            analysis = psyche.analyze(prompt)
            fdf = analysis.formation_df
            print(f'  Cached: {len(fdf)} words')
            all_fdfs[fam_key] = (fdf, n_layers)
            continue
        except Exception as e:
            print(f'  Cache miss: {type(e).__name__}')

        # Load models
        print(f'  Loading models...')
        try:
            psyche = Psyche.from_family(fam_key, load=True)
            analysis = psyche.analyze(prompt)
            fdf = analysis.formation_df
            print(f'  Computed: {len(fdf)} words')
            all_fdfs[fam_key] = (fdf, n_layers)
        except Exception as e:
            print(f'  FAILED: {e}')
            continue

        del psyche, analysis
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Map each family's columns to canonical positions
    # BASE=0, SFT=1 (if available), ALIGNED=2 (last column)
    COL_TO_POS = {
        'base': 0, 'ego': 1, 'sft': 1,
        'superego': 2, 'dpo': 2, 'instruct': 2, 'rlvr': 2,
    }

    rows = []
    for fam_key, (fdf, n_layers) in all_fdfs.items():
        layer_cols = [c for c in fdf.columns if c in COL_TO_POS]

        # For 4-layer families, the "aligned" is the last col (rlvr/instruct)
        # For 2-layer families, it's superego/dpo/instruct
        # Map: base→0, sft/ego→1, last→2
        for _, r in fdf.iterrows():
            # BASE
            if 'base' in layer_cols:
                rows.append({'family': fam_key, 'word': r['word'],
                             'pos': 0, 'pos_name': 'BASE', 'prob': r['base'],
                             'trajectory': r.get('trajectory', '')})

            # SFT (only 3+ layer families)
            sft_col = next((c for c in ['sft', 'ego'] if c in layer_cols), None)
            if sft_col and n_layers >= 3:
                rows.append({'family': fam_key, 'word': r['word'],
                             'pos': 1, 'pos_name': 'SFT', 'prob': r[sft_col],
                             'trajectory': r.get('trajectory', '')})

            # ALIGNED (last non-base column)
            aligned_col = None
            for c in ['rlvr', 'instruct', 'dpo', 'superego', 'sft', 'ego']:
                if c in layer_cols and c != 'base':
                    aligned_col = c
                    break
            if aligned_col:
                rows.append({'family': fam_key, 'word': r['word'],
                             'pos': 2, 'pos_name': 'ALIGNED', 'prob': r[aligned_col],
                             'trajectory': r.get('trajectory', '')})

    df = pd.DataFrame(rows)
    print(f'\nTotal rows: {len(df)}, families: {len(all_fdfs)}')
    print(f'Families included: {sorted(all_fdfs.keys())}')

    # Aggregate
    agg = df.groupby(['word', 'pos', 'pos_name'])['prob'].agg(
        ['mean', 'std', 'count']).reset_index()
    agg = agg.rename(columns={'mean': 'prob_mean', 'std': 'prob_std', 'count': 'n_families'})

    # Need data at positions 0 and 2 minimum
    words_at_base = set(agg[agg['pos'] == 0]['word'])
    words_at_aligned = set(agg[agg['pos'] == 2]['word'])
    valid_words = words_at_base & words_at_aligned
    agg = agg[agg['word'].isin(valid_words)]

    # Classify by mean trajectory (base vs aligned)
    word_summary = []
    for w in valid_words:
        wsub = agg[agg['word'] == w].sort_values('pos')
        base_val = wsub[wsub['pos'] == 0]['prob_mean'].values[0]
        aligned_val = wsub[wsub['pos'] == 2]['prob_mean'].values[0]
        max_prob = wsub['prob_mean'].max()
        n_at_base = wsub[wsub['pos'] == 0]['n_families'].values[0]
        if max_prob < 0.002 or n_at_base < 3:
            continue
        if aligned_val > base_val * 1.5:
            traj = 'rise'
        elif aligned_val < base_val * 0.5:
            traj = 'decline'
        else:
            traj = 'stable'
        word_summary.append({'word': w, 'trajectory': traj, 'base': base_val,
                             'aligned': aligned_val, 'max': max_prob, 'n': int(n_at_base)})

    wdf = pd.DataFrame(word_summary)
    print(f'\n=== Meta formation ({len(all_fdfs)} families) ===')
    for traj in ['decline', 'rise', 'stable']:
        sub = wdf[wdf['trajectory'] == traj].sort_values('max', ascending=False)
        print(f'\n  {traj} (n={len(sub)}):')
        for _, r in sub.head(12).iterrows():
            print(f'    {r["word"]:12s}  base={r["base"]:.4f}  aligned={r["aligned"]:.4f}  (n={r["n"]})')

    # Plot
    decline = wdf[wdf['trajectory'] == 'decline'].nlargest(8, 'max')
    rise = wdf[wdf['trajectory'] == 'rise'].nlargest(8, 'max')

    TRAJ_COLORS = {'decline': '#e63946', 'rise': '#457b9d'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    positions = sorted(agg['pos'].unique())
    x = np.arange(len(positions))
    x_labels = [POSITIONS[p] for p in positions]

    for ax, traj, sub in [(axes[0], 'decline', decline), (axes[1], 'rise', rise)]:
        color = TRAJ_COLORS[traj]
        sub = sub.sort_values('aligned', ascending=False)

        for _, wrow in sub.iterrows():
            word = wrow['word']
            vals = []
            for pos in positions:
                row = agg[(agg['word'] == word) & (agg['pos'] == pos)]
                vals.append(row['prob_mean'].values[0] if len(row) > 0 else np.nan)
            ax.plot(x, vals, '-o', color=color, linewidth=1.8, markersize=5, alpha=0.7)
            ax.text(x[-1] + 0.08, vals[-1], word, fontsize=10, va='center',
                    color=color, fontweight='bold')

        n_words = len(sub)
        ax.set_title(f'{traj} (n={n_words})', fontsize=13, fontweight='bold', color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_yscale('log')
        ax.set_ylabel('probability (mean across families)' if ax == axes[0] else '', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.3, len(positions) - 0.3)

    short_prompt = prompt if len(prompt) < 50 else prompt[:47] + '...'
    fig.suptitle(f'"{short_prompt} ___"',
                 fontsize=15, fontweight='bold', y=0.98)
    n_sft = len([k for k, (_, nl) in all_fdfs.items() if nl >= 3])
    fig.text(0.5, 0.93,
             f'Mean across {len(all_fdfs)} families · SFT from {n_sft} families with 3+ layers',
             ha='center', fontsize=11, style='italic', color='#444')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    slug = prompt.lower().replace(' ', '_')[:30]
    figpath = f'figures/F01_{slug}_formation.png'
    plt.savefig(figpath, dpi=200, bbox_inches='tight')
    print(f'\nSaved {figpath}')

    csvpath = f'data/f01_{slug}_formation.csv'
    agg.to_csv(csvpath, index=False)
    print(f'Saved {csvpath}')
