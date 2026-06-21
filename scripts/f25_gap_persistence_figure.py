"""
F25 alignment gap persistence: entropy gap trajectories across 100 tokens.

Shows how the alignment entropy gap evolves over 100 tokens for each family.
Key finding: Llama inverts (sublimation opens), Qwen deepens, Amber locks, OLMo narrows.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILES = {
    "OLMo": ("data/mega_gen_olmo_4layer.csv", "base", "dpo"),
    "Llama": ("data/mega_generation_llama.csv", "base", "aligned"),
    "Qwen": ("data/mega_generation_qwen.csv", "base", "aligned"),
    "Amber": ("data/mega_generation_amber.csv", "base", "aligned"),
}

PROMPTS = ["anger", "violence", "sexual", "worker", "love"]
PROMPT_COLORS = {
    "anger": "#e74c3c", "violence": "#2c3e50", "sexual": "#8e44ad",
    "worker": "#e67e22", "love": "#27ae60",
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for idx, (fam, (fpath, base_layer, aligned_layer)) in enumerate(FILES.items()):
    ax = axes[idx]
    df = pd.read_csv(fpath)
    max_step = df["step"].max()

    for pk in PROMPTS:
        base = df[(df["layer"] == base_layer) & (df["prompt_key"] == pk)]
        aligned = df[(df["layer"] == aligned_layer) & (df["prompt_key"] == pk)]

        if len(base) == 0 or len(aligned) == 0:
            continue

        base_h = base.groupby("step")["entropy"].mean()
        aligned_h = aligned.groupby("step")["entropy"].mean()

        steps = sorted(set(base_h.index) & set(aligned_h.index))
        gap = [aligned_h[s] - base_h[s] for s in steps]

        ax.plot(steps, gap, color=PROMPT_COLORS[pk], alpha=0.7,
               linewidth=1.5, label=pk)

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_title(fam, fontsize=13, fontweight='bold')
    ax.set_xlabel("Token position", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Entropy gap (aligned − base, nats)", fontsize=10)
    ax.set_xlim(0, max_step)

    # Add region labels
    ax.fill_between([0, max_step], 0, 2, alpha=0.03, color='red')
    ax.fill_between([0, max_step], 0, -4, alpha=0.03, color='blue')

axes[0].legend(fontsize=7, loc='lower left', frameon=False)

fig.suptitle("Alignment Gap Persistence Across 100 Tokens\n"
             "(negative = aligned narrows, positive = aligned opens)",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig("figures/F25_gap_persistence.png", dpi=200, bbox_inches='tight',
            facecolor='white')
print("Saved figures/F25_gap_persistence.png")
