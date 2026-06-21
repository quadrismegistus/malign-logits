"""Six circuit signature diagrams from CircuitProfile data.

Each diagram: a small pipeline (base→SFT→DPO→output) with
entropy mapped to node size, violence/procedural to color,
JS divergence to edge thickness.
"""
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

# Signature exemplars: (family, prompt, description, layers)
SIGNATURES = [
    ("foreclosure", "olmo", "anger", "Foreclosure (Verwerfung)",
     ["base", "ego", "superego", "rlvr"],
     "kill → ______ at SFT. Signifier excluded from symbolic order."),
    ("repression", "llama", "anger", "Repression (Verdrängung)",
     ["base", "superego"],
     "kill → scream. Signifier displaced, chain preserved."),
    ("reaction_formation", "amber", "anger", "Reaction Formation",
     ["base", "ego", "superego"],
     "kill → sc → moral correction. Drive expressed then negated."),
    ("transparent", "zephyr", "anger", "Transparent",
     ["base", "ego", "superego"],
     "scream → scream → scream. Entropy narrows, argmax preserved."),
    ("de_foreclosure", "olmo", "worker", "De-foreclosure",
     ["base", "ego", "superego"],
     "NaN → seek. Alignment constitutes the subject position."),
    ("restore", "qwen3", "anger", "Restore (Qwen3)",
     ["base", "superego"],
     "______ → kill. Alignment UN-forecloses the signifier."),
]

# Manual data where profiles are incomplete
MANUAL_DATA = {
    ("zephyr", "anger"): {
        "base": {"H": 4.23, "top1": "scream", "V": 0.0, "P": 0.0},
        "ego": {"H": 3.05, "top1": "scream", "V": 0.0, "P": 0.0},
        "superego": {"H": 2.36, "top1": "scream", "V": 0.0, "P": 0.0},
    },
}


def get_node_data(family, prompt, layer):
    """Get node data from profile CSV or manual data."""
    if (family, prompt) in MANUAL_DATA and layer in MANUAL_DATA[(family, prompt)]:
        return MANUAL_DATA[(family, prompt)][layer]
    try:
        nodes = pd.read_csv(f"data/profiles/{family}_nodes.csv")
        row = nodes[(nodes["checkpoint"] == layer) & (nodes["prompt"] == prompt)]
        if len(row) > 0:
            r = row.iloc[0]
            return {
                "H": r["entropy"], "top1": r["argmax_token"],
                "V": r["violence_loading"], "P": r["procedural_loading"],
            }
    except:
        pass
    return None


def violence_to_color(v):
    """Map violence loading to color: red (violent) → blue (peaceful)."""
    v_clamp = max(-0.1, min(0.2, v))
    t = (v_clamp + 0.1) / 0.3
    r = t
    b = 1 - t
    g = 0.2
    return (r, g, b, 0.85)


def blank_color():
    return (0.7, 0.7, 0.7, 0.85)


def entropy_to_size(h, base_size=800):
    return base_size * (h / 5.0)


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (sig_name, family, prompt, title, layers, description) in enumerate(SIGNATURES):
    ax = axes[idx]

    # Collect node data
    node_data = []
    for layer in layers:
        d = get_node_data(family, prompt, layer)
        if d:
            node_data.append((layer, d))
        else:
            node_data.append((layer, {"H": 3.0, "top1": "?", "V": 0.0, "P": 0.0}))

    n = len(node_data)
    x_positions = np.linspace(0.1, 0.9, n)

    # Draw edges
    for i in range(n - 1):
        x0, x1 = x_positions[i], x_positions[i + 1]
        layer0, d0 = node_data[i]
        layer1, d1 = node_data[i + 1]

        # Edge thickness from JS, style from signature
        js = 0.1
        edge_sig = "repression"
        try:
            edges = pd.read_csv(f"data/profiles/{family}_edges.csv")
            e = edges[(edges["from"] == layer0) & (edges["to"] == layer1) & (edges["prompt"] == prompt)]
            if len(e) > 0:
                js = e.iloc[0]["js_divergence"]
                edge_sig = e.iloc[0].get("signature", "repression")
        except:
            pass

        lw = max(1, js * 15)
        style_map = {
            "repression": "-", "foreclosure": "-", "reaction_formation": "-",
            "transparent": "--", "de_foreclosure": ":",
            "return_of_repressed": "-.", "unknown": "-",
        }
        linestyle = style_map.get(edge_sig, "-")
        edge_color = "#555" if linestyle == "-" else "#888"
        ax.annotate("", xy=(x1 - 0.03, 0.5), xytext=(x0 + 0.03, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=edge_color,
                                   linestyle=linestyle,
                                   connectionstyle="arc3,rad=0"),
                    transform=ax.transAxes)

    # Draw nodes
    for i, (layer, d) in enumerate(node_data):
        x = x_positions[i]
        top1 = str(d["top1"])
        is_blank = any(c in top1 for c in ("_", "▁")) or top1 in ("nan", "None", "", "?")

        if is_blank:
            color = blank_color()
        else:
            color = violence_to_color(d["V"])

        size = entropy_to_size(d["H"])
        ax.scatter([x], [0.5], s=size, c=[color], edgecolors="white",
                  linewidth=2, zorder=5, transform=ax.transAxes)

        # Token label inside node
        label = "______" if is_blank else top1[:8]
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=8,
               fontweight="bold", color="white", transform=ax.transAxes,
               path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

        # Layer name below
        layer_labels = {"base": "Base", "ego": "SFT", "superego": "DPO", "rlvr": "RLVR"}
        ax.text(x, 0.22, layer_labels.get(layer, layer), ha="center",
               fontsize=8, color="#555", transform=ax.transAxes)

        # Entropy above
        ax.text(x, 0.78, f"H={d['H']:.1f}", ha="center", fontsize=7,
               color="#888", transform=ax.transAxes)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.text(0.5, 0.05, description, ha="center", fontsize=7, color="#666",
           style="italic", transform=ax.transAxes, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# Legend
legend_ax = fig.add_axes([0.02, 0.01, 0.96, 0.03])
legend_ax.axis("off")
legend_elements = [
    mpatches.Patch(facecolor=violence_to_color(0.15), label="Violent (+V)"),
    mpatches.Patch(facecolor=violence_to_color(0.0), label="Neutral"),
    mpatches.Patch(facecolor=violence_to_color(-0.05), label="Peaceful (-V)"),
    mpatches.Patch(facecolor=blank_color(), label="Blank/foreclosed"),
]
legend_ax.legend(handles=legend_elements, loc="center", ncol=4, fontsize=9, frameon=False)

fig.suptitle("Six Circuit Signatures of Alignment",
            fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig("figures/circuit_signature_diagrams.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Saved figures/circuit_signature_diagrams.png")
