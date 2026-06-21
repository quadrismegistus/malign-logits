"""F11 meta contradiction: superposition → resolution across all families.

Computes JS divergences for 11 contradiction pairs across all families,
maps to BASE/SFT/ALIGNED, averages, and plots the biplot.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import gc


def _js(p, q):
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p.log() - m.log())).sum()
            + 0.5 * (q * (q.log() - m.log())).sum()).item()


if __name__ == '__main__':
    from malign_logits.psyche import Psyche
    from malign_logits import MODEL_FAMILIES

    pairs = list(Psyche.DEFAULT_CONTRADICTIONS) + [
        {
            'name': 'innocent/guilty',
            'prompt_a': 'She was innocent and she began to',
            'prompt_b': 'She was guilty and she began to',
            'prompt_ab': 'She was innocent and guilty and she began to',
        },
        {
            'name': 'rational/irrational',
            'prompt_a': 'He was completely rational and chose to',
            'prompt_b': 'He was completely irrational and chose to',
            'prompt_ab': 'He was completely rational and completely irrational and chose to',
        },
        {
            'name': 'faithful/unfaithful',
            'prompt_a': 'She was faithful to him and she began to',
            'prompt_b': 'She was unfaithful to him and she began to',
            'prompt_ab': 'She was faithful and unfaithful to him and she began to',
        },
        {
            'name': 'rich/poor',
            'prompt_a': 'He was rich and he decided to',
            'prompt_b': 'He was poor and he decided to',
            'prompt_ab': 'He was rich and poor and he decided to',
        },
    ]
    ALL_FAMILIES = sorted(MODEL_FAMILIES.keys())

    # Canonical position mapping
    LAYER_TO_POS = {
        'base': (0, 'BASE'), 'primary_process': (0, 'BASE'),
        'ego': (1, 'SFT'), 'sft': (1, 'SFT'),
        'superego': (2, 'ALIGNED'), 'dpo': (2, 'ALIGNED'),
        'reinforced_superego': (2, 'ALIGNED'), 'rlvr': (2, 'ALIGNED'),
        'instruct': (2, 'ALIGNED'),
    }

    results = []

    for fam_key in ALL_FAMILIES:
        fam = MODEL_FAMILIES[fam_key]
        n_layers = sum(1 for x in [fam.base, fam.ego, fam.superego, fam.reinforced_superego] if x)
        print(f'\n=== {fam_key} ({n_layers}L) ===')

        # Try cache first
        try:
            psyche = Psyche.from_family(fam_key, load=False)
        except Exception as e:
            print(f'  FAILED init: {e}')
            continue

        # Check if logits are cached for all prompts
        all_prompts = []
        for pair in pairs:
            all_prompts.extend([pair['prompt_a'], pair['prompt_b'], pair['prompt_ab']])

        layers_list = [('base', psyche.primary_process)]
        if psyche.ego:
            layers_list.append(('ego', psyche.ego))
        layers_list.append(('superego', psyche.superego))
        if psyche.reinforced_superego:
            layers_list.append(('rlvr', psyche.reinforced_superego))

        # Check cache
        needs_load = False
        for _, layer in layers_list:
            for p in all_prompts:
                try:
                    logits = layer.logits(p)
                    if logits is None:
                        needs_load = True
                        break
                except:
                    needs_load = True
                    break
            if needs_load:
                break

        if needs_load:
            print(f'  Loading models...')
            try:
                psyche = Psyche.from_family(fam_key, load=True)
                layers_list = [('base', psyche.primary_process)]
                if psyche.ego:
                    layers_list.append(('ego', psyche.ego))
                layers_list.append(('superego', psyche.superego))
                if psyche.reinforced_superego:
                    layers_list.append(('rlvr', psyche.reinforced_superego))
            except Exception as e:
                print(f'  FAILED load: {e}')
                continue
        else:
            print(f'  All cached')

        for pair in pairs:
            for layer_name, layer in layers_list:
                try:
                    logits_a = layer.logits(pair['prompt_a'])
                    logits_b = layer.logits(pair['prompt_b'])
                    logits_ab = layer.logits(pair['prompt_ab'])

                    n = min(logits_a.shape[-1], logits_b.shape[-1], logits_ab.shape[-1])
                    p_a = torch.softmax(logits_a[:n].float(), dim=-1)
                    p_b = torch.softmax(logits_b[:n].float(), dim=-1)
                    p_ab = torch.softmax(logits_ab[:n].float(), dim=-1)
                    p_mean = 0.5 * (p_a + p_b)

                    js_ab_a = _js(p_ab, p_a)
                    js_ab_b = _js(p_ab, p_b)
                    js_ab_mean = _js(p_ab, p_mean)
                    ratio = js_ab_mean / max(min(js_ab_a, js_ab_b), 1e-10)

                    pos, pos_name = LAYER_TO_POS.get(layer_name, (2, 'ALIGNED'))
                    # For 4-layer families, use RLVR as ALIGNED (not DPO)
                    if layer_name == 'superego' and psyche.reinforced_superego:
                        pos, pos_name = (2, 'DPO')
                        # Skip: we'll use rlvr as ALIGNED
                        continue

                    results.append({
                        'family': fam_key, 'pair': pair['name'],
                        'pos': pos, 'pos_name': pos_name,
                        'js_to_A': js_ab_a, 'js_to_B': js_ab_b,
                        'js_to_mean': js_ab_mean, 'ratio': ratio,
                        'pole_bias': js_ab_a - js_ab_b,
                    })
                except Exception as e:
                    print(f'  {pair["name"]} {layer_name}: {e}')

        if needs_load:
            del psyche
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv('data/f11_meta_contradiction.csv', index=False)
    print(f'\nSaved data/f11_meta_contradiction.csv ({len(df)} rows)')

    # Aggregate across families
    agg = df.groupby(['pair', 'pos', 'pos_name']).agg(
        ratio_mean=('ratio', 'mean'),
        ratio_std=('ratio', 'std'),
        pole_bias_mean=('pole_bias', 'mean'),
        pole_bias_std=('pole_bias', 'std'),
        n_families=('family', 'nunique'),
    ).reset_index()

    print(f'\n=== Meta contradiction ({agg["n_families"].max()} families) ===')
    for pair_name in sorted(agg['pair'].unique()):
        psub = agg[agg['pair'] == pair_name].sort_values('pos')
        vals = '  →  '.join(f'{r.pos_name} ratio={r.ratio_mean:.3f} bias={r.pole_bias_mean:+.3f} (n={int(r.n_families)})'
                            for _, r in psub.iterrows())
        print(f'  {pair_name:25s}  {vals}')

    # Plot biplot — BASE and ALIGNED only
    plot_agg = agg[agg['pos'].isin([0, 2])]

    PAIR_COLORS = {
        'love/hate': '#e63946', 'trust/fear': '#f4a261',
        'beautiful/disgusting': '#e76f51', 'desire/disgust': '#d62828',
        'obey/rebel': '#264653', 'sacred/profane': '#6d6875',
        'man/woman': '#2a9d8f', 'human/animal': '#457b9d',
        'pleasure/pain': '#a8dadc', 'create/destroy': '#8338ec',
        'free/captive': '#06d6a0',
        'innocent/guilty': '#ff006e',
        'rational/irrational': '#90be6d',
        'faithful/unfaithful': '#f9c74f',
        'rich/poor': '#577590',
    }
    MARKERS = {0: 'o', 2: 'D'}
    POSITIONS = {0: 'BASE', 2: 'ALIGNED'}

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.axhline(1.0, color='#999', linestyle='--', linewidth=0.8, zorder=1)
    ax.axvline(0, color='#999', linestyle='--', linewidth=0.8, zorder=1)

    texts = []
    for pair_name in sorted(plot_agg['pair'].unique()):
        psub = plot_agg[plot_agg['pair'] == pair_name].sort_values('pos')
        color = PAIR_COLORS.get(pair_name, '#666')

        xs = psub['pole_bias_mean'].values
        ys = psub['ratio_mean'].values
        positions = psub['pos'].values

        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8, alpha=0.6),
                    zorder=2)

        for xi, yi, pos in zip(xs, ys, positions):
            ax.plot(xi, yi, marker=MARKERS[pos], color=color, markersize=9,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=3)

        texts.append(ax.text(xs[-1], ys[-1], f'  {pair_name}', fontsize=9,
                             color=color, fontweight='bold', va='center', zorder=4))

    ax.text(0.98, 0.98, 'departure from inclusive disjunction\n(Oedipalization)', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#e63946', alpha=0.6, style='italic')
    ax.text(0.98, 0.02, 'inclusive disjunction\n(either ... or ... or)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#457b9d', alpha=0.6, style='italic')

    for pos, marker in MARKERS.items():
        ax.plot([], [], marker=marker, color='#666', markersize=9, linestyle='none',
                label=POSITIONS[pos], markeredgecolor='white', markeredgewidth=0.5)
    ax.legend(title='Training stage', loc='lower left', fontsize=9, title_fontsize=10)

    ax.set_xlabel('Pole bias:  JS(combined, A) − JS(combined, B)\n← resolves toward A          resolves toward B →',
                  fontsize=10)
    ax.set_ylabel('Superposition ratio\nJS(combined, blend) / min(JS(combined, A), JS(combined, B))',
                  fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Background shading based on data range
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.fill_between([xmin, xmax], 1.0, ymax, alpha=0.04, color='#e63946', zorder=0)
    ax.fill_between([xmin, xmax], ymin, 1.0, alpha=0.04, color='#457b9d', zorder=0)

    n_fam = plot_agg['n_families'].max()
    fig.suptitle('From inclusive disjunction to Oedipalization', fontsize=15, fontweight='bold', y=0.98)
    fig.text(0.5, 0.935,
             f'Mean across {n_fam} families · BASE → ALIGNED',
             ha='center', fontsize=11, style='italic', color='#444')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    adjust_text(texts, ax=ax, force_text=(0.3, 0.3),
                arrowprops=dict(arrowstyle='-', color='#ccc', lw=0.5))

    plt.savefig('figures/F11_contradiction_biplot_meta.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved figures/F11_contradiction_biplot_meta.png')
