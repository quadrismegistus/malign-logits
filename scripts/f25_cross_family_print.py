"""
F25 cross-family figure — print-ready version with legible axis labels.
Requested by TheoryMachines for book signature image.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from malign_logits.circuit import Circuit

FILES = {
    "OLMo": "data/mega_gen_olmo_4layer.csv",
    "Llama": "data/mega_generation_llama.csv",
    "Qwen": "data/mega_generation_qwen.csv",
    "Amber": "data/mega_generation_amber.csv",
    "SmolLM3": "data/mega_generation_smol3.csv",
}

ALIGNED_LAYER = {
    "OLMo": "dpo", "Llama": "aligned", "Qwen": "aligned",
    "Amber": "aligned", "SmolLM3": "aligned",
}

PROMPT_ORDER = ["anger", "violence", "sexual", "worker", "love"]
PROMPT_LABELS = {
    "anger": "Anger", "violence": "Violence", "sexual": "Sexual",
    "worker": "Worker", "love": "Love",
}
FAMILY_ORDER = ["OLMo", "Llama", "Qwen", "Amber", "SmolLM3"]

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
SIGNATURE_LABELS = {
    "foreclosure": "Foreclosure\n(Verwerfung)",
    "return_of_repressed": "Return of\nrepressed",
    "repression": "Repression\n(Verdrängung)",
    "reaction_formation": "Reaction\nformation",
    "transparent": "Transparent\n(APO)",
    "de_foreclosure": "De-foreclosure",
    "unclassified": "Unclassified",
}

# ── Classify ──────────────────────────────────────────────────────

all_classified = []
for family_name in FAMILY_ORDER:
    fpath = FILES[family_name]
    df = pd.read_csv(fpath)
    base_layer = "base"
    aligned_layer = ALIGNED_LAYER[family_name]
    df_filtered = df[df["layer"].isin([base_layer, aligned_layer])].copy()

    base_argmax = {}
    base_data = df_filtered[df_filtered["layer"] == base_layer]
    for pk in base_data["prompt_key"].unique():
        step0 = base_data[(base_data["prompt_key"] == pk) & (base_data["step"] == 0)]
        if len(step0) > 0:
            mode = step0["top1"].mode()
            if len(mode) > 0 and pd.notna(mode.iloc[0]):
                base_argmax[pk] = mode.iloc[0]
            else:
                base_argmax[pk] = Circuit.BLANK_SENTINEL

    aligned_data = df_filtered[df_filtered["layer"] == aligned_layer]
    for (pk, gen_idx), sub in aligned_data.groupby(["prompt_key", "gen_idx"]):
        result = Circuit.classify_trajectory(sub, base_top1=base_argmax.get(pk))
        result["family"] = family_name
        result["prompt_key"] = pk
        result["gen_idx"] = gen_idx
        all_classified.append(result)

classified_df = pd.DataFrame(all_classified)

# ── Compute proportions ──────────────────────────────────────────

grid_proportions = {}
for family_name in FAMILY_ORDER:
    for pk in PROMPT_ORDER:
        cell = classified_df[(classified_df["family"] == family_name) &
                             (classified_df["prompt_key"] == pk)]
        if len(cell) == 0:
            grid_proportions[(family_name, pk)] = {}
            continue
        grid_proportions[(family_name, pk)] = cell["signature"].value_counts(normalize=True).to_dict()

# ── Print-ready figure ────────────────────────────────────────────

n_f = len(FAMILY_ORDER)
n_p = len(PROMPT_ORDER)

fig, axes = plt.subplots(n_f, n_p, figsize=(10, 10))

for i, family in enumerate(FAMILY_ORDER):
    for j, prompt in enumerate(PROMPT_ORDER):
        ax = axes[i, j]
        props = grid_proportions.get((family, prompt), {})

        pie_data = []
        pie_colors = []
        pie_labels = []
        for sig in SIGNATURE_ORDER:
            val = props.get(sig, 0)
            if val > 0:
                pie_data.append(val)
                pie_colors.append(SIGNATURE_COLORS[sig])
                pie_labels.append(sig)

        if pie_data:
            wedges, _ = ax.pie(pie_data, colors=pie_colors, startangle=90,
                              wedgeprops=dict(edgecolor='white', linewidth=1))

            # Dominant % in centre
            dominant = max(props, key=props.get)
            dom_pct = props[dominant] * 100
            if dom_pct < 100:
                ax.text(0, 0, f"{dom_pct:.0f}%", ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white',
                       path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        else:
            ax.text(0, 0, "—", ha='center', va='center', fontsize=14, color='#999')

        ax.set_aspect('equal')

        # Column headers (top row only)
        if i == 0:
            ax.set_title(PROMPT_LABELS[prompt], fontsize=13, fontweight='bold', pad=12)

        # Row labels (left column only)
        if j == 0:
            ax.text(-1.5, 0, family, fontsize=13, fontweight='bold',
                   va='center', ha='right', transform=ax.transData)

# Legend at bottom
legend_elements = [
    plt.matplotlib.patches.Patch(facecolor=SIGNATURE_COLORS[sig],
                                  edgecolor='white', linewidth=1,
                                  label=SIGNATURE_LABELS[sig].replace('\n', ' '))
    for sig in SIGNATURE_ORDER if sig not in ("unclassified",)
]
fig.legend(handles=legend_elements, loc='lower center',
          bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=10,
          frameon=False, handlelength=1.5, handleheight=1.5)

fig.suptitle("Temporal Alignment Signatures", fontsize=16, fontweight='bold', y=0.98)

plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.10,
                    wspace=0.1, hspace=0.15)
plt.savefig("figures/F25_cross_family_signatures_print.png", dpi=300,
            bbox_inches='tight', facecolor='white')
print("Saved figures/F25_cross_family_signatures_print.png")
