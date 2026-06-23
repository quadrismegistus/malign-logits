"""
viz_sankey.py — Merged DAG Sankey diagrams with BASE/SFT dual flows.

Nodes merged by (depth, token). Multiple paths converge into shared
continuations, showing attractor structure. Dual flows show alignment
intervention: blue = base probability, orange = aligned probability.

    from malign_logits.viz_sankey import merged_sankey
    fig = merged_sankey("allenai/OLMo-2-0425-1B", "anger")
    fig.write_html("figures/sankey_anger.html")
"""

import plotly.graph_objects as go
from collections import defaultdict


def merged_sankey(model_id: str, prompt: str, annotator_idx: int = 0,
                  top_n: int = 8, save: str = None):
    """Build a merged DAG Sankey with BASE vs aligned dual flows.

    Args:
        model_id: base model HF ID
        prompt: prompt name or text
        annotator_idx: which annotator to show (0 = first/SFT)
        top_n: max nodes per depth level
        save: path prefix to save .html + .png (optional)

    Returns:
        plotly Figure
    """
    from .probe import Probe

    p = Probe(model_id)
    nodes = p.annotate_tree(prompt)
    short = model_id.split("/")[-1]
    prompt_display = prompt if len(prompt) < 35 else prompt[:32] + "..."

    # Find annotator prefix
    sr_keys = [k for k in nodes[1].keys() if k.endswith("_self_resist")]
    if not sr_keys:
        raise ValueError("No self_resist annotations found")
    prefix = sr_keys[min(annotator_idx, len(sr_keys) - 1)].replace("_self_resist", "")
    ann_name = prefix.replace("_", "-")[:20]

    # Build merged DAG edges with base + aligned flow
    base_edges = defaultdict(float)
    aligned_edges = defaultdict(float)
    node_base_flow = defaultdict(float)
    node_aligned_flow = defaultdict(float)
    node_resist = defaultdict(list)

    for n in nodes:
        if n["depth"] == 0:
            continue
        parent = nodes[n["parent"]]
        from_key = (0, "ROOT") if parent["depth"] == 0 else (parent["depth"], parent["token"])
        to_key = (n["depth"], n["token"])

        bp = n.get("path_prob", n["prob"])
        sr = n.get(f"{prefix}_self_resist", 0)
        sp = n.get(f"{prefix}_self_prob", n["prob"])

        # Aligned path prob: approximate by scaling base path_prob
        if n["prob"] > 0:
            ap = bp * (sp / n["prob"])
        else:
            ap = bp

        base_edges[(from_key, to_key)] += bp
        aligned_edges[(from_key, to_key)] += ap
        node_base_flow[to_key] += bp
        node_aligned_flow[to_key] += ap
        node_base_flow[from_key] += 0  # ensure exists
        node_aligned_flow[from_key] += 0
        node_resist[to_key].append(sr)

    # Find top-N complete paths (leaf storylines), keep all nodes along them
    max_depth = max(d for d, _ in node_base_flow.keys())

    # Build leaf paths from the original tree
    node_ids = set(range(len(nodes)))
    parent_set = {n["parent"] for n in nodes}
    leaves = [i for i in node_ids - parent_set if nodes[i]["depth"] > 0]
    leaves.sort(key=lambda i: -nodes[i].get("path_prob", 0))

    # Collect (depth, token) keys along top-N leaf paths
    keep = {(0, "ROOT")}
    paths_kept = 0
    for leaf_idx in leaves:
        if paths_kept >= top_n:
            break
        # Walk up from leaf to root, collect merged keys
        path_keys = set()
        i = leaf_idx
        while i >= 0:
            n = nodes[i]
            if n["depth"] > 0:
                path_keys.add((n["depth"], n["token"]))
            i = n["parent"]
        keep.update(path_keys)
        paths_kept += 1

    # Filter edges
    filtered_base = {}
    filtered_aligned = {}
    for (fk, tk) in base_edges:
        if fk in keep and tk in keep:
            filtered_base[(fk, tk)] = base_edges[(fk, tk)]
            filtered_aligned[(fk, tk)] = aligned_edges[(fk, tk)]

    unique_nodes = sorted(keep)
    node_map = {k: i for i, k in enumerate(unique_nodes)}

    by_depth = defaultdict(list)
    for k in unique_nodes:
        by_depth[k[0]].append(k)
    for d in by_depth:
        by_depth[d].sort(key=lambda k: -node_base_flow.get(k, 0))

    # Node properties
    labels = []
    x_pos = []
    y_pos = []
    colors = []

    for k in unique_nodes:
        d, tok = k
        x = 0.001 + (d / (max_depth + 0.5)) * 0.98
        group = by_depth[d]
        idx = group.index(k)
        margin = 0.03
        y = margin + (idx / max(len(group) - 1, 1)) * (1 - 2 * margin)
        if len(group) == 1:
            y = 0.5

        bf = node_base_flow.get(k, 0) * 100
        af = node_aligned_flow.get(k, 0) * 100
        resists = node_resist.get(k, [0])
        mean_sr = sum(resists) / len(resists) if resists else 0

        if d == 0:
            labels.append("ROOT")
            colors.append("rgba(100,100,100,0.8)")
        else:
            labels.append(f"{tok}<br>b:{bf:.2f}%<br>a:{af:.2f}%<br>r:{mean_sr:+.2f}b")
            if mean_sr > 1.0:
                colors.append("rgba(217,79,61,0.8)")
            elif mean_sr > 0.3:
                colors.append("rgba(232,164,77,0.8)")
            elif mean_sr > -0.3:
                colors.append("rgba(200,200,200,0.8)")
            elif mean_sr > -1.0:
                colors.append("rgba(168,194,86,0.8)")
            else:
                colors.append("rgba(45,138,78,0.8)")

        x_pos.append(x)
        y_pos.append(y)

    # Build links: two per edge (base blue, aligned orange)
    source = []
    target = []
    value = []
    link_color = []
    link_label = []

    for (fk, tk) in filtered_base:
        si = node_map[fk]
        ti = node_map[tk]
        bf = filtered_base[(fk, tk)]
        af = filtered_aligned.get((fk, tk), 0)

        # Base flow (blue)
        source.append(si)
        target.append(ti)
        value.append(max(bf * 1000, 0.01))
        link_color.append("rgba(68,119,170,0.25)")
        link_label.append(f"base: {bf*100:.2f}%")

        # Aligned flow (orange)
        source.append(si)
        target.append(ti)
        value.append(max(af * 1000, 0.01))
        link_color.append("rgba(232,115,77,0.25)")
        link_label.append(f"{ann_name}: {af*100:.2f}%")

    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=6, thickness=12,
            line=dict(color="#333", width=0.3),
            label=labels, color=colors,
            x=x_pos, y=y_pos,
        ),
        link=dict(
            source=source, target=target, value=value,
            color=link_color, label=link_label,
        )
    )])

    fig.update_layout(
        title_text=(
            f"{short}: \"{prompt_display}\" (merged DAG, top {top_n}/depth)<br>"
            f"<sub>Blue=BASE, Orange={ann_name}. Node color: green=facilitated, red=blocked. "
            f"Convergence visible where paths merge.</sub>"
        ),
        font_size=9, width=1400, height=700,
    )

    if save:
        fig.write_html(f"{save}.html")
        fig.write_image(f"{save}.png", scale=2)
        # Auto-copy to TheoryMachines
        import shutil
        from pathlib import Path
        tm_dir = Path.home() / "Dropbox" / "Prof" / "Articles" / "TheoryMachines" / "figures"
        if tm_dir.exists():
            for ext in (".html", ".png"):
                src = Path(f"{save}{ext}")
                if src.exists():
                    shutil.copy2(src, tm_dir / src.name)

    return fig
