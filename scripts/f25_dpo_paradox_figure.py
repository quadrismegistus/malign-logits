"""
F25 DPO paradox figure: deeper alignment → more return of repressed.

Shows signature proportions across OLMo's 4 layers (base→SFT→DPO→RLVR)
per prompt, revealing how alignment stages redistribute defences.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from malign_logits.circuit import Circuit

df = pd.read_csv("data/mega_gen_olmo_4layer.csv")

LAYERS = ["sft", "dpo", "rlvr"]
LAYER_LABELS = {"sft": "SFT (Ego)", "dpo": "DPO (Superego)", "rlvr": "RLVR (Ego-ideal)"}
PROMPTS = ["anger", "violence", "sexual", "worker", "love"]

SIGNATURE_COLORS = {
    "foreclosure": "#2c3e50",
    "return_of_repressed": "#e74c3c",
    "repression": "#8e44ad",
    "reaction_formation": "#e67e22",
    "transparent": "#27ae60",
    "de_foreclosure": "#3498db",
    "unclassified": "#bdc3c7",
}

SIGNATURE_ORDER = [
    "foreclosure", "return_of_repressed", "repression",
    "reaction_formation", "transparent", "de_foreclosure", "unclassified"
]

# Get base argmax
base_argmax = {}
base_data = df[df["layer"] == "base"]
for pk in base_data["prompt_key"].unique():
    step0 = base_data[(base_data["prompt_key"] == pk) & (base_data["step"] == 0)]
    if len(step0) > 0:
        mode = step0["top1"].mode()
        if len(mode) > 0 and pd.notna(mode.iloc[0]):
            base_argmax[pk] = mode.iloc[0]
        else:
            base_argmax[pk] = Circuit.BLANK_SENTINEL

# Classify per layer
all_results = []
for layer in LAYERS:
    layer_data = df[df["layer"] == layer]
    for (pk, gen_idx), sub in layer_data.groupby(["prompt_key", "gen_idx"]):
        r = Circuit.classify_trajectory(sub, base_top1=base_argmax.get(pk))
        r["layer"] = layer
        r["prompt_key"] = pk
        r["gen_idx"] = gen_idx
        all_results.append(r)

classified = pd.DataFrame(all_results)

# Build stacked bar chart: layers × prompts, stacked by signature
fig, axes = plt.subplots(1, 5, figsize=(14, 5), sharey=True)

for j, prompt in enumerate(PROMPTS):
    ax = axes[j]
    prompt_data = classified[classified["prompt_key"] == prompt]

    bottoms = np.zeros(len(LAYERS))
    for sig in SIGNATURE_ORDER:
        vals = []
        for layer in LAYERS:
            cell = prompt_data[prompt_data["layer"] == layer]
            if len(cell) == 0:
                vals.append(0)
            else:
                vals.append((cell["signature"] == sig).mean())
        vals = np.array(vals)
        if vals.sum() > 0:
            ax.bar(range(len(LAYERS)), vals, bottom=bottoms,
                   color=SIGNATURE_COLORS[sig], label=sig if j == 0 else None,
                   edgecolor='white', linewidth=0.5)
            # Label bars with % if > 15%
            for i, v in enumerate(vals):
                if v > 0.15:
                    ax.text(i, bottoms[i] + v / 2, f"{v:.0%}",
                           ha='center', va='center', fontsize=7,
                           color='white', fontweight='bold',
                           path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
            bottoms += vals

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([LAYER_LABELS[l].split(" (")[0] for l in LAYERS],
                       fontsize=9, rotation=30, ha='right')
    ax.set_title(prompt.capitalize(), fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    if j == 0:
        ax.set_ylabel("Proportion of generations", fontsize=10)

# Legend
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SIGNATURE_COLORS[s])
           for s in SIGNATURE_ORDER if s != "de_foreclosure"]
labels = [s.replace("_", " ").capitalize() for s in SIGNATURE_ORDER if s != "de_foreclosure"]
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8,
          bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.suptitle("OLMo: Temporal Signatures Across Alignment Stages\n"
             "(deeper alignment → more reaction formation, not more foreclosure)",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.savefig("figures/F25_dpo_paradox.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved figures/F25_dpo_paradox.png")

# Print key numbers
print("\nKey paradox numbers:")
for prompt in PROMPTS:
    pd_sub = classified[classified["prompt_key"] == prompt]
    for sig in ["foreclosure", "return_of_repressed", "reaction_formation"]:
        sft_pct = (pd_sub[pd_sub["layer"] == "sft"]["signature"] == sig).mean() * 100
        dpo_pct = (pd_sub[pd_sub["layer"] == "dpo"]["signature"] == sig).mean() * 100
        rlvr_pct = (pd_sub[pd_sub["layer"] == "rlvr"]["signature"] == sig).mean() * 100
        if max(sft_pct, dpo_pct, rlvr_pct) > 5:
            delta = rlvr_pct - sft_pct
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {prompt}/{sig}: SFT={sft_pct:.0f}% → DPO={dpo_pct:.0f}% → RLVR={rlvr_pct:.0f}% ({arrow}{abs(delta):.0f}pp)")
