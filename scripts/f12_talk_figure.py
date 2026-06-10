"""F12 talk figures: cumulative-variance overlay + fold-as-path.

Computes SVD of (DPO - base) hidden states for selected families,
then generates two talk-ready figures.

Usage: python scripts/f12_talk_figure.py
"""
import numpy as np
import pandas as pd
import torch
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    from malign_logits import MODEL_FAMILIES
    from malign_logits.psyche import Psyche
    from malign_logits.experiments import DEFAULT_PROMPTS
    from malign_logits.trajectory import last_hidden

    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer as ATok

    FAMILIES = ['pythia', 'amber', 'olmo']
    LABELS = {
        'pythia': 'Pythia (community fine-tune)',
        'amber': 'Amber (3-stage, RedPajama)',
        'olmo': 'OLMo (industrial safety stack)',
    }
    COLORS = {'pythia': '#e63946', 'amber': '#f4a261', 'olmo': '#457b9d'}

    all_cumvar = {}
    all_S = {}

    for fam_key in FAMILIES:
        print(f'\n=== {fam_key} ===')
        fam = MODEL_FAMILIES[fam_key]
        fold_df = pd.read_csv(f'data/fold_rank_{fam_key}.csv')
        best_L = int(fold_df['layer'].iloc[0])
        print(f'  Using layer {best_L}')
        print(f'  Loading {fam.base}...')
        tok = ATok.from_pretrained(fam.base, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            fam.base, torch_dtype=torch.float16, device_map='mps', trust_remote_code=True)
        print(f'  Loading {fam.superego}...')
        dpo_model = AutoModelForCausalLM.from_pretrained(
            fam.superego, torch_dtype=torch.float16, device_map='mps', trust_remote_code=True)

        diff_vecs = []
        for label, prompt in DEFAULT_PROMPTS.items():
            base_h = last_hidden(base_model, tok, prompt, best_L)
            dpo_h = last_hidden(dpo_model, tok, prompt, best_L)
            diff_vecs.append((dpo_h - base_h).numpy())

        diff_matrix = np.stack(diff_vecs)
        U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
        cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
        k_50 = int(np.searchsorted(cumvar, 0.5)) + 1

        all_cumvar[fam_key] = cumvar
        all_S[fam_key] = S
        print(f'  K_50 = {k_50}, top1 = {S[0]**2 / (S**2).sum() * 100:.1f}%')

        del base_model, dpo_model, tok
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # === FIGURE 1: Cumulative-variance overlay ===
    N = 20
    rows = []
    for fam_key in FAMILIES:
        cv = all_cumvar[fam_key]
        k_50 = int(np.searchsorted(cv, 0.5)) + 1
        for i in range(min(N, len(cv))):
            rows.append({'family': LABELS[fam_key], 'direction': i + 1,
                         'cumvar': cv[i] * 100, 'k_50': k_50,
                         'color_key': fam_key})

    cvdf = pd.DataFrame(rows)
    cvdf['family'] = pd.Categorical(cvdf['family'],
                                     categories=[LABELS[f] for f in FAMILIES], ordered=True)

    # Annotation points where each curve crosses 50%
    annot_rows = []
    for fam_key in FAMILIES:
        cv = all_cumvar[fam_key]
        k_50 = int(np.searchsorted(cv, 0.5)) + 1
        annot_rows.append({
            'family': LABELS[fam_key], 'direction': k_50,
            'cumvar': cv[k_50 - 1] * 100, 'k_50': k_50,
            'label': f'K₅₀ = {k_50}',
            'color_key': fam_key,
        })
    adf = pd.DataFrame(annot_rows)
    adf['family'] = pd.Categorical(adf['family'],
                                    categories=[LABELS[f] for f in FAMILIES], ordered=True)

    p1 = (ggplot(cvdf, aes(x='direction', y='cumvar', color='family'))
          + geom_line(size=1.5)
          + geom_point(size=2.5)
          + geom_hline(yintercept=50, linetype='dashed', color='gray', alpha=0.7)
          + geom_point(data=adf, mapping=aes(x='direction', y='cumvar'), size=5, shape='D')
          + geom_text(data=adf, mapping=aes(x='direction', y='cumvar', label='label'),
                      nudge_y=5, nudge_x=1, size=10, ha='left')
          + scale_color_manual(values={LABELS[f]: COLORS[f] for f in FAMILIES})
          + scale_x_continuous(breaks=range(1, N + 1, 2))
          + labs(x='Number of directions\n(independent components of the alignment shift)',
                 y='Cumulative % of alignment shift captured',
                 title='How many directions does alignment fold into?',
                 color='')
          + theme_minimal()
          + theme(figure_size=(12, 6),
                  text=element_text(size=12),
                  plot_title=element_text(size=15),
                  legend_position='right')
    )
    p1.save('figures/F12_fold_dimensionality_talk.png', dpi=200)
    print('\nSaved figures/F12_fold_dimensionality_talk.png')

    # === FIGURE 2: Fold-as-path (small multiples) ===
    np.random.seed(42)
    n_dirs = 20
    # Fixed random 2D projection (same for all families)
    proj = np.random.randn(2, 4096)  # will truncate to hidden_dim per family
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)

    path_rows = []
    for fam_key in FAMILIES:
        S = all_S[fam_key]
        n = min(n_dirs, len(S))
        # Normalize so total path length = 1 for comparability
        total = S[:n].sum()
        lengths = S[:n] / total

        # Build path: each step in a direction projected to 2D
        # Use random 2D angles since we can't faithfully project high-D
        x, y = 0.0, 0.0
        path_rows.append({'family': LABELS[fam_key], 'step': 0, 'x': x, 'y': y})
        for i in range(n):
            angle = np.random.uniform(0, 2 * np.pi)
            dx = lengths[i] * np.cos(angle)
            dy = lengths[i] * np.sin(angle)
            x += dx
            y += dy
            path_rows.append({'family': LABELS[fam_key], 'step': i + 1, 'x': x, 'y': y})

    pdf = pd.DataFrame(path_rows)
    pdf['family'] = pd.Categorical(pdf['family'],
                                    categories=[LABELS[f] for f in FAMILIES], ordered=True)

    p2 = (ggplot(pdf, aes(x='x', y='y'))
          + geom_path(size=1.2, color='#264653')
          + geom_point(aes(size='step'), color='#264653', alpha=0.6)
          + scale_size_continuous(range=(1, 4), guide=None)
          + facet_wrap('~family', ncol=3, scales='free')
          + labs(title='The fold as a path: each segment is one direction of alignment\nSegment length = how much of the fold lives there · Layout is schematic',
                 x='', y='')
          + theme_minimal()
          + theme(figure_size=(16, 5),
                  text=element_text(size=11),
                  plot_title=element_text(size=13),
                  axis_text=element_blank(),
                  axis_ticks=element_blank(),
                  panel_grid=element_blank())
    )
    p2.save('figures/F12_fold_as_path_talk.png', dpi=200)
    print('Saved figures/F12_fold_as_path_talk.png')
