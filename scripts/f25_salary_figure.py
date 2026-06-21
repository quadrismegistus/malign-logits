"""
Salary cross-family figure: gender gap × profession × family heatmap.

Shows how alignment changes gendered salary predictions across 5 families.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

df = pd.read_csv("data/salary_all.csv")

FAMILIES = ["OLMo", "Llama", "Qwen", "Amber", "SmolLM3"]
FAMILY_MAP = {"olmo": "OLMo", "llama": "Llama", "qwen": "Qwen", "amber": "Amber", "smol3": "SmolLM3"}
df["family_label"] = df["family"].map(FAMILY_MAP)

PROFESSIONS = ["doctor", "nurse", "teacher", "engineer"]

# Compute gender gap: median(male) - median(female) for each profession × family × layer
rows = []
for fam in FAMILIES:
    for prof in PROFESSIONS:
        for layer in ["base", "aligned"]:
            male = df[(df["family_label"] == fam) & (df["layer"] == layer) &
                     (df["prompt_key"] == f"{prof}_male")]["salary"].dropna()
            female = df[(df["family_label"] == fam) & (df["layer"] == layer) &
                       (df["prompt_key"] == f"{prof}_female")]["salary"].dropna()
            if len(male) > 0 and len(female) > 0:
                gap = male.median() - female.median()
                gap_pct = gap / female.median() * 100 if female.median() > 0 else 0
                rows.append({
                    "family": fam, "profession": prof, "layer": layer,
                    "male_median": male.median(), "female_median": female.median(),
                    "gap": gap, "gap_pct": gap_pct,
                })

gap_df = pd.DataFrame(rows)

# ── Figure 1: Gender gap heatmap (base vs aligned side by side) ──

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, layer, title in [(ax1, "base", "Base Model"), (ax2, "aligned", "Aligned Model")]:
    sub = gap_df[gap_df["layer"] == layer]
    matrix = sub.pivot(index="family", columns="profession", values="gap_pct")
    matrix = matrix.reindex(index=FAMILIES, columns=PROFESSIONS)

    vmax = max(abs(gap_df["gap_pct"].min()), abs(gap_df["gap_pct"].max()))
    vmax = min(vmax, 200)
    im = ax.imshow(matrix.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    for i in range(len(FAMILIES)):
        for j in range(len(PROFESSIONS)):
            val = matrix.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:+.0f}%", ha='center', va='center',
                       fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(range(len(PROFESSIONS)))
    ax.set_xticklabels([p.capitalize() for p in PROFESSIONS], fontsize=10)
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels(FAMILIES, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')

fig.colorbar(im, ax=[ax1, ax2], label="Gender gap (% of female median salary)",
            shrink=0.8, pad=0.02)
fig.suptitle("Salary Gender Gap: Male − Female Median\n"
             "(positive = men paid more, negative = women paid more)",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.92, 0.90])
plt.savefig("figures/F25_salary_gender_gap.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved figures/F25_salary_gender_gap.png")

# ── Figure 2: Alignment effect on absolute salary by profession ──

fig2, axes2 = plt.subplots(1, 4, figsize=(14, 5), sharey=False)
bar_colors = {"base": "#95a5a6", "aligned": "#2c3e50"}

for j, prof in enumerate(PROFESSIONS):
    ax = axes2[j]
    x = np.arange(len(FAMILIES))
    width = 0.35

    base_vals = []
    aligned_vals = []
    for fam in FAMILIES:
        neutral = df[(df["family_label"] == fam) &
                    (df["prompt_key"] == f"{prof}_neutral")]
        base_v = neutral[neutral["layer"] == "base"]["salary"].median()
        aligned_v = neutral[neutral["layer"] == "aligned"]["salary"].median()
        base_vals.append(base_v if not pd.isna(base_v) else 0)
        aligned_vals.append(aligned_v if not pd.isna(aligned_v) else 0)

    ax.bar(x - width/2, [v/1000 for v in base_vals], width, label='Base',
          color=bar_colors["base"], edgecolor='white')
    ax.bar(x + width/2, [v/1000 for v in aligned_vals], width, label='Aligned',
          color=bar_colors["aligned"], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, fontsize=8, rotation=30, ha='right')
    ax.set_title(prof.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel("Median salary ($K)" if j == 0 else "", fontsize=10)
    if j == 0:
        ax.legend(fontsize=8)

fig2.suptitle("Median Salary by Profession Across Families\n"
              "(neutral gender prompt, base vs aligned)",
              fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("figures/F25_salary_profession.png", dpi=200, bbox_inches='tight', facecolor='white')
print("Saved figures/F25_salary_profession.png")

# Summary
print("\nGender gap summary (aligned, % of female median):")
aligned_gaps = gap_df[gap_df["layer"] == "aligned"][["family", "profession", "gap_pct"]]
print(aligned_gaps.pivot(index="family", columns="profession", values="gap_pct").to_string(float_format="{:+.0f}%".format))
