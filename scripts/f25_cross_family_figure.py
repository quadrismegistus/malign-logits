"""
F25 cross-family figure: 5×5 grid (family × prompt) coloured by dominant signature.

Runs Circuit.classify_trajectory on all 5 families' mega-gen data and produces
a heatmap where colour = dominant temporal alignment signature.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from malign_logits.circuit import Circuit

# ── Load all data ──────────────────────────────────────────────────

FILES = {
    "OLMo": "data/mega_gen_olmo_4layer.csv",
    "Llama": "data/mega_generation_llama.csv",
    "Qwen": "data/mega_generation_qwen.csv",
    "Amber": "data/mega_generation_amber.csv",
    "SmolLM3": "data/mega_generation_smol3.csv",
}

ALIGNED_LAYER = {
    "OLMo": "dpo",
    "Llama": "aligned",
    "Qwen": "aligned",
    "Amber": "aligned",
    "SmolLM3": "aligned",
}

PROMPT_ORDER = ["anger", "violence", "sexual", "worker", "love"]
FAMILY_ORDER = ["OLMo", "Llama", "Qwen", "Amber", "SmolLM3"]

SIGNATURE_COLORS = {
    "foreclosure": "#2c3e50",        # dark blue-grey
    "return_of_repressed": "#e74c3c", # red
    "repression": "#8e44ad",          # purple
    "reaction_formation": "#e67e22",  # orange
    "transparent": "#27ae60",         # green
    "de_foreclosure": "#3498db",      # blue
    "unclassified": "#bdc3c7",        # light grey
}

SIGNATURE_ORDER = [
    "foreclosure", "return_of_repressed", "repression",
    "reaction_formation", "transparent", "de_foreclosure", "unclassified"
]

SIGNATURE_LABELS = {
    "foreclosure": "Foreclosure",
    "return_of_repressed": "Return of repressed",
    "repression": "Repression",
    "reaction_formation": "Reaction formation",
    "transparent": "Transparent",
    "de_foreclosure": "De-foreclosure",
    "unclassified": "Unclassified",
}

# ── Classify each family ──────────────────────────────────────────

all_classified = []

for family_name in FAMILY_ORDER:
    fpath = FILES[family_name]
    df = pd.read_csv(fpath)
    base_layer = "base"
    aligned_layer = ALIGNED_LAYER[family_name]

    # Filter to base + aligned only
    df_filtered = df[df["layer"].isin([base_layer, aligned_layer])].copy()

    # Get base argmax per prompt (mode of step-0 top1 across generations)
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

    # Classify aligned generations
    aligned_data = df_filtered[df_filtered["layer"] == aligned_layer]
    for (pk, gen_idx), sub in aligned_data.groupby(["prompt_key", "gen_idx"]):
        result = Circuit.classify_trajectory(sub, base_top1=base_argmax.get(pk))
        result["family"] = family_name
        result["prompt_key"] = pk
        result["gen_idx"] = gen_idx
        all_classified.append(result)

classified_df = pd.DataFrame(all_classified)
print(f"Classified {len(classified_df)} generations across {len(FAMILY_ORDER)} families")
print(classified_df["signature"].value_counts())

# ── Compute dominant signature per cell ───────────────────────────

grid = {}
grid_proportions = {}

for family_name in FAMILY_ORDER:
    for pk in PROMPT_ORDER:
        cell = classified_df[(classified_df["family"] == family_name) &
                             (classified_df["prompt_key"] == pk)]
        if len(cell) == 0:
            grid[(family_name, pk)] = "unclassified"
            grid_proportions[(family_name, pk)] = {}
            continue
        counts = cell["signature"].value_counts(normalize=True)
        grid[(family_name, pk)] = counts.index[0]
        grid_proportions[(family_name, pk)] = counts.to_dict()

# ── Build figure ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

n_families = len(FAMILY_ORDER)
n_prompts = len(PROMPT_ORDER)

for i, family in enumerate(FAMILY_ORDER):
    for j, prompt in enumerate(PROMPT_ORDER):
        dominant = grid[(family, prompt)]
        proportions = grid_proportions[(family, prompt)]

        # Draw pie chart in each cell
        pie_data = []
        pie_colors = []
        for sig in SIGNATURE_ORDER:
            val = proportions.get(sig, 0)
            if val > 0:
                pie_data.append(val)
                pie_colors.append(SIGNATURE_COLORS[sig])

        if pie_data:
            # Create inset axes for pie
            x0 = j / n_prompts + 0.01
            y0 = 1 - (i + 1) / n_families + 0.01
            w = 1 / n_prompts - 0.02
            h = 1 / n_families - 0.02
            inset = fig.add_axes([
                ax.get_position().x0 + ax.get_position().width * x0,
                ax.get_position().y0 + ax.get_position().height * y0,
                ax.get_position().width * w,
                ax.get_position().height * h,
            ])
            wedges, _ = inset.pie(pie_data, colors=pie_colors, startangle=90)
            inset.set_aspect('equal')

            # Add dominant % label
            dom_pct = proportions.get(dominant, 0) * 100
            if dom_pct < 100:
                inset.text(0, 0, f"{dom_pct:.0f}%", ha='center', va='center',
                          fontsize=8, fontweight='bold', color='white',
                          path_effects=[
                              pe.withStroke(linewidth=2, foreground='black')
                          ])

# Row labels (families)
for i, family in enumerate(FAMILY_ORDER):
    y = 1 - (i + 0.5) / n_families
    ax.text(-0.02, y, family, transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='center', ha='right')

# Column labels (prompts)
for j, prompt in enumerate(PROMPT_ORDER):
    x = (j + 0.5) / n_prompts
    ax.text(x, 1.02, prompt.capitalize(), transform=ax.transAxes, fontsize=11,
           fontweight='bold', va='bottom', ha='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("F25: Temporal Alignment Signatures Across Families\n"
             "(dominant mechanism per prompt, aligned layer)",
             fontsize=14, fontweight='bold', pad=20)

# Legend
legend_elements = [
    plt.matplotlib.patches.Patch(facecolor=SIGNATURE_COLORS[sig],
                                  label=SIGNATURE_LABELS[sig])
    for sig in SIGNATURE_ORDER if sig != "unclassified"
]
ax.legend(handles=legend_elements, loc='lower center',
         bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9,
         frameon=False)

plt.savefig("figures/F25_cross_family_signatures.png", dpi=200, bbox_inches='tight',
            facecolor='white')
print("Saved figures/F25_cross_family_signatures.png")

# ── Also save summary CSV ─────────────────────────────────────────

summary_rows = []
for family in FAMILY_ORDER:
    for prompt in PROMPT_ORDER:
        props = grid_proportions[(family, prompt)]
        row = {"family": family, "prompt": prompt, "dominant": grid[(family, prompt)]}
        for sig in SIGNATURE_ORDER:
            row[sig] = props.get(sig, 0)
        n = len(classified_df[(classified_df["family"] == family) &
                              (classified_df["prompt_key"] == prompt)])
        row["n"] = n
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("data/f25_signature_summary.csv", index=False)
print("Saved data/f25_signature_summary.csv")
print()
print(summary_df[["family", "prompt", "dominant", "n"]].to_string(index=False))
