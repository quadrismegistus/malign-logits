"""Causal DAG of alignment's distributional effects.

Qualitative DAG annotated with controlled comparisons.
Solid edges = identified by natural experiment.
Dashed edges = plausible but unidentified.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(14, 9))

# Node positions (x, y)
nodes = {
    # Causes (left)
    "corpus": (1.0, 7.0),
    "scale": (1.0, 5.0),
    "country": (1.0, 3.0),
    "org": (3.0, 3.0),
    # Mediators (middle)
    "alignment_data": (5.0, 6.5),
    "alignment_method": (5.0, 4.5),
    "prompt_category": (5.0, 2.0),
    # Outcomes (right)
    "entropy": (9.0, 7.5),
    "violence_loading": (9.0, 6.0),
    "procedural_loading": (9.0, 4.5),
    "signature": (9.0, 3.0),
    "template_architecture": (9.0, 1.5),
}

# Node styling
cause_style = dict(boxstyle="round,pad=0.4", facecolor="#e8f4f8", edgecolor="#2c3e50", lw=1.5)
mediator_style = dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#e67e22", lw=1.5)
outcome_style = dict(boxstyle="round,pad=0.4", facecolor="#fdedec", edgecolor="#e74c3c", lw=1.5)

node_labels = {
    "corpus": "Pretraining\ncorpus",
    "scale": "Model\nscale",
    "country": "Country\nof origin",
    "org": "Organisation",
    "alignment_data": "Alignment\ndata",
    "alignment_method": "Alignment\nmethod",
    "prompt_category": "Prompt\ncategory",
    "entropy": "Entropy",
    "violence_loading": "Violence\nloading",
    "procedural_loading": "Procedural\nloading",
    "signature": "Clinical\nsignature",
    "template_architecture": "Template\narchitecture",
}

for name, (x, y) in nodes.items():
    if name in ("corpus", "scale", "country", "org"):
        style = cause_style
    elif name in ("alignment_data", "alignment_method", "prompt_category"):
        style = mediator_style
    else:
        style = outcome_style
    ax.text(x, y, node_labels[name], ha="center", va="center", fontsize=9,
           fontweight="bold", bbox=style, zorder=10)

# Edges: (from, to, identified_by, style)
# solid = identified, dashed = unidentified
edges = [
    # Identified edges (solid, with comparison annotation)
    ("alignment_method", "signature",
     "OLMo vs OLMo-Think\n(same base, 4/5 differ)", "solid"),
    ("alignment_data", "signature",
     "Llama vs Tulu\n(same base+method, 2/5 differ)", "solid"),
    ("alignment_data", "procedural_loading",
     "Llama vs Tulu\n(file vs sue)", "solid"),
    ("alignment_method", "violence_loading",
     "OLMo vs Think\n(+0.15 → -0.02 vs +0.10)", "solid"),
    ("alignment_method", "procedural_loading",
     "OLMo vs Think\n(doubles vs halves)", "solid"),
    ("prompt_category", "signature",
     "Census grid\n(drives converge,\nclass diverges)", "solid"),
    ("prompt_category", "violence_loading",
     "anger +0.15\nvs worker -0.03", "solid"),

    # Identified as null effect (tested, not found)
    ("scale", "signature",
     "OLMo-tiny vs OLMo\n(mostly replicates)\n≈ NULL", "null"),

    # Plausible but unidentified (dashed)
    ("corpus", "violence_loading", "", "dashed"),
    ("corpus", "entropy", "", "dashed"),
    ("country", "org", "", "dashed"),
    ("org", "alignment_data", "", "dashed"),
    ("org", "alignment_method", "", "dashed"),
    ("alignment_data", "entropy", "", "dashed"),
    ("alignment_method", "entropy", "", "dashed"),
    ("alignment_data", "violence_loading", "", "dashed"),
    ("alignment_method", "template_architecture", "", "dashed"),
    ("org", "template_architecture", "", "dashed"),
    ("scale", "entropy", "", "dashed"),
]

for from_node, to_node, annotation, style in edges:
    x0, y0 = nodes[from_node]
    x1, y1 = nodes[to_node]

    # Offset for text boxes
    dx = x1 - x0
    dy = y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    offset = 0.6
    x0a = x0 + dx / dist * offset
    y0a = y0 + dy / dist * offset
    x1a = x1 - dx / dist * offset
    y1a = y1 - dy / dist * offset

    if style == "solid":
        color = "#2c3e50"
        lw = 2.0
        ls = "-"
    elif style == "null":
        color = "#e74c3c"
        lw = 2.0
        ls = "-"
    else:
        color = "#bdc3c7"
        lw = 1.0
        ls = "--"

    ax.annotate("", xy=(x1a, y1a), xytext=(x0a, y0a),
               arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                              linestyle=ls, connectionstyle="arc3,rad=0.1"))

    # Annotation on identified edges
    if annotation and style == "solid":
        mx = (x0a + x1a) / 2
        my = (y0a + y1a) / 2
        # Offset annotation perpendicular to edge
        perp_x = -dy / dist * 0.35
        perp_y = dx / dist * 0.35
        ax.text(mx + perp_x, my + perp_y, annotation, fontsize=6,
               ha="center", va="center", color="#2c3e50", style="italic",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor="none", alpha=0.8))

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#e8f4f8", edgecolor="#2c3e50", label="Cause"),
    mpatches.Patch(facecolor="#fef9e7", edgecolor="#e67e22", label="Mediator"),
    mpatches.Patch(facecolor="#fdedec", edgecolor="#e74c3c", label="Outcome"),
    plt.Line2D([0], [0], color="#2c3e50", lw=2, label="Identified (natural experiment)"),
    plt.Line2D([0], [0], color="#e74c3c", lw=2, label="Identified NULL (tested, not found)"),
    plt.Line2D([0], [0], color="#bdc3c7", lw=1, linestyle="--", label="Plausible (unidentified)"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8, frameon=True)

ax.set_xlim(-0.5, 11)
ax.set_ylim(0.5, 8.5)
ax.set_title("Causal Structure of Alignment's Distributional Effects\n"
             "Edges annotated with identifying natural experiments",
             fontsize=13, fontweight="bold")
ax.axis("off")

plt.tight_layout()
plt.savefig("figures/causal_dag.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved figures/causal_dag.png")
