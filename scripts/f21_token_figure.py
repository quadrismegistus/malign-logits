"""F21 token shift figure: base vs aligned probability for key tokens."""
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    from malign_logits.cache import get_cache
    from malign_logits import MODEL_FAMILIES
    from transformers import AutoTokenizer

    cache = get_cache()

    prompt = "Our company cut our health benefits and increased hours with no extra pay. We should"

    TOKENS_OF_INTEREST = [
        'strike', 'quit', 'sue', 'walk', 'leave', 'refuse', 'fight',
        'consider', 'negotiate', 'think', 'talk', 'discuss', 'ask',
        'union', 'report', 'file', 'demand',
    ]

    FAMILIES = ['olmo', 'llama', 'amber', 'qwen', 'tulu', 'zephyr']

    rows = []
    for fam_key in FAMILIES:
        fam = MODEL_FAMILIES[fam_key]
        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        # Get token IDs (with space prefix)
        token_ids = {}
        for w in TOKENS_OF_INTEREST:
            ids = tok.encode(' ' + w, add_special_tokens=False)
            if len(ids) == 1:
                token_ids[w] = ids[0]

        aligned_id = fam.reinforced_superego or fam.superego
        base_logits = cache.get_logits(fam.base, prompt)
        aligned_logits = cache.get_logits(aligned_id, prompt)

        if base_logits is None or aligned_logits is None:
            print(f'  Missing logits for {fam_key}')
            continue

        base_probs = torch.softmax(torch.tensor(base_logits).float(), dim=-1)
        aligned_probs = torch.softmax(torch.tensor(aligned_logits).float(), dim=-1)

        for w, tid in token_ids.items():
            if tid < len(base_probs) and tid < len(aligned_probs):
                rows.append({
                    'family': fam_key, 'token': w,
                    'base': base_probs[tid].item(),
                    'aligned': aligned_probs[tid].item(),
                    'delta': aligned_probs[tid].item() - base_probs[tid].item(),
                })

    df = pd.DataFrame(rows)
    df.to_csv('data/f21_token_shifts_multi.csv', index=False)

    # Aggregate across families: mean base, mean aligned, mean delta
    agg = df.groupby('token').agg(
        base_mean=('base', 'mean'),
        aligned_mean=('aligned', 'mean'),
        delta_mean=('delta', 'mean'),
        n=('family', 'count'),
    ).reset_index()
    agg = agg.sort_values('delta_mean')

    # Split into gained and lost
    lost = agg[agg['delta_mean'] < -0.0005].nsmallest(8, 'delta_mean')
    gained = agg[agg['delta_mean'] > 0.0005].nlargest(8, 'delta_mean')
    plot_df = pd.concat([lost, gained]).sort_values('delta_mean')

    print('\nMean across families:')
    for _, r in plot_df.iterrows():
        print(f'  {r["token"]:15s}  base={r["base_mean"]:.4f}  aligned={r["aligned_mean"]:.4f}  Δ={r["delta_mean"]:+.4f}  (n={int(r["n"])})')

    # Figure: lollipop chart
    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(plot_df))
    tokens = plot_df['token'].values
    deltas = plot_df['delta_mean'].values
    bases = plot_df['base_mean'].values
    aligneds = plot_df['aligned_mean'].values

    colors = ['#e63946' if d < 0 else '#457b9d' for d in deltas]

    ax.barh(y, deltas * 100, color=colors, alpha=0.8, height=0.6, edgecolor='white', linewidth=0.5)

    for i, (tok_name, d, b, a) in enumerate(zip(tokens, deltas, bases, aligneds)):
        side = 'left' if d > 0 else 'right'
        offset = 0.05 if d > 0 else -0.05
        ax.text(d * 100 + offset, i, f'{b*100:.2f}% → {a*100:.2f}%',
                va='center', ha=side, fontsize=8, color='#555')

    ax.set_yticks(y)
    ax.set_yticklabels(tokens, fontsize=11, fontweight='bold')
    ax.axvline(0, color='#999', linewidth=0.8)
    ax.set_xlabel('Change in probability (percentage points)\n← suppressed by alignment          promoted by alignment →', fontsize=10)

    fig.suptitle('"We should ___"',
                 fontsize=16, fontweight='bold', y=0.97)
    fig.text(0.5, 0.925,
             'Prompt: "Our company cut our health benefits and increased hours with no extra pay. We should"',
             ha='center', fontsize=10, style='italic', color='#444')
    fig.text(0.5, 0.895,
             f'Mean probability shift across {len(FAMILIES)} families (base → aligned)',
             ha='center', fontsize=10, color='#666')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig('figures/F21_token_shift.png', dpi=200, bbox_inches='tight')
    print('\nSaved figures/F21_token_shift.png')
