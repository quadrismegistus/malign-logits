from . import *
from .psyche import TRAJECTORY_THRESHOLD


def _layer_columns(df):
    """Detect which layer columns are present in a formation DataFrame."""
    cols = ["base"]
    if "ego" in df.columns:
        cols.append("ego")
    if "superego" in df.columns:
        cols.append("superego")
    if "instruct" in df.columns:
        cols.append("instruct")
    return cols


def _layer_labels(cols):
    """Human-readable labels for layer columns."""
    labels = {
        "base": "base (pretrained)",
        "ego": "ego (SFT)",
        "superego": "superego (DPO)",
        "instruct": "instruct (RLVR)",
    }
    return [labels.get(c, c) for c in cols]


def plot_formation_trajectories(
    formation,
    prompt=None,
    min_prob=0.003,
    min_delta=None,
    sort_by="mass",
    top_n=120,
    color_by_shape=True,
    label_words=True,
    facet_by_shape=False,
    facet_cols=3,
    facet_hspace=0.08,
    facet_vspace=0.12,
    width=None,
    height=None,
    save_path=None,
):
    """
    Visualize probability trajectories across training layers.

    Supports 3-layer (base/ego/superego) and 4-layer (+ instruct/RLVR)
    topologies automatically based on the DataFrame columns.

    Args:
        formation: Either a PromptAnalysis instance (with `.formation_df`) or a
            DataFrame in `formation_df` format.
        prompt: Optional prompt text for title (auto-read from PromptAnalysis).
        min_prob: Keep words above this prob in at least one layer.
        min_delta: If set, also include words where max delta between any
            adjacent layers exceeds this threshold (catches high-movement
            low-probability words like sublimated terms).
        sort_by: How to rank words for top_n. "mass" (default) sorts by
            total probability across layers. "delta" sorts by maximum
            absolute delta between adjacent layers.
        top_n: Max number of words to draw.
        color_by_shape: If True, color by trajectory class.
        label_words: If True, print word labels at the last layer point.
        facet_by_shape: If True, create one subplot per trajectory class.
        facet_cols: Number of facet columns when `facet_by_shape=True`.
        facet_hspace: Horizontal spacing between facet panels (0-1).
        facet_vspace: Vertical spacing between facet panels (0-1).
        width: Overall figure width in pixels.
        height: Overall figure height in pixels. If None, auto-calculated.
        save_path: Optional image path to save via `fig.write_image`.

    Returns:
        plotly.graph_objects.Figure
    """
    if hasattr(formation, "formation_df"):
        df = formation.formation_df.copy()
        if prompt is None:
            prompt = getattr(formation, "prompt", None)
    else:
        df = formation.copy()

    # Ensure numeric columns are float
    for col in ["base", "ego", "superego", "instruct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    required = {"word", "base"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"formation DataFrame missing required columns: {sorted(missing)}"
        )

    cols = _layer_columns(df)
    if len(cols) < 2:
        raise ValueError("formation DataFrame needs at least 2 layer columns")
    labels = _layer_labels(cols)
    xvals = list(range(len(cols)))

    if "trajectory" not in df.columns:
        from .psyche import _classify_trajectory
        df["trajectory"] = df.apply(_classify_trajectory, axis=1)

    # Filter: include words above min_prob OR above min_delta
    prob_mask = df[cols].max(axis=1) > min_prob

    # Compute max absolute delta between adjacent layers
    delta_cols = []
    for i in range(len(cols) - 1):
        col_name = f"_delta_{i}"
        df[col_name] = (df[cols[i]] - df[cols[i + 1]]).abs()
        delta_cols.append(col_name)
    df["_max_delta"] = df[delta_cols].max(axis=1)

    if min_delta is not None:
        delta_mask = df["_max_delta"] > min_delta
        sig = df[prob_mask | delta_mask].copy()
    else:
        sig = df[prob_mask].copy()

    if len(sig) == 0:
        raise ValueError("No words passed filters; lower `min_prob` or `min_delta`.")

    # Sort by chosen criterion
    sig["mass"] = sig[cols].sum(axis=1)
    if sort_by == "delta":
        sig = sig.sort_values("_max_delta", ascending=False).head(top_n)
    else:
        sig = sig.sort_values("mass", ascending=False).head(top_n)

    # Clean up temp columns
    df.drop(columns=delta_cols + ["_max_delta"], inplace=True, errors="ignore")
    sig.drop(columns=delta_cols + ["_max_delta"], inplace=True, errors="ignore")

    shape_colors = {
        "decline": "#e15759",
        "rise": "#4e79a7",
        "V": "#f28e2b",
        "peak": "#76b7b2",
        "sublimated": "#b07aa1",
        "eliminated": "#b07aa1",
        "superego_only": "#59a14f",
        "flat": "#9c9c9c",
    }
    neutral_color = "#6f6f6f"
    shape_order = ["decline", "rise", "V", "peak", "eliminated", "sublimated", "superego_only", "flat"]

    def _row_yvals(row):
        return [max(row[c], 1e-7) for c in cols]

    def _row_hover(row):
        traj = row.get("trajectory", "flat")
        parts = [f"<b>{row['word']}</b><br>trajectory: {traj}"]
        for c in cols:
            parts.append(f"{c}={row[c]:.5f}")
        return "<br>".join(parts) + "<extra></extra>"

    last_x = len(cols) - 1

    if facet_by_shape:
        from plotly.subplots import make_subplots

        facet_cols = max(1, int(facet_cols))
        present_shapes = [s for s in shape_order if s in set(sig["trajectory"].tolist())]
        if not present_shapes:
            present_shapes = ["flat"]
        n_panels = len(present_shapes)
        ncols = min(facet_cols, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        subtitles = [
            f"{s} (n={int((sig['trajectory'] == s).sum())})"
            for s in present_shapes
        ]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subtitles,
            shared_yaxes=True,
            horizontal_spacing=facet_hspace,
            vertical_spacing=facet_vspace,
        )

        for i, shape in enumerate(present_shapes):
            row_idx = i // ncols + 1
            col_idx = i % ncols + 1
            sdf = sig[sig["trajectory"] == shape]
            for _, row in sdf.iterrows():
                color = shape_colors.get(shape, neutral_color) if color_by_shape else neutral_color
                yvals = _row_yvals(row)
                hover = _row_hover(row)
                text_vals = [""] * (len(cols) - 1) + [row["word"]] if label_words else None
                fig.add_trace(
                    go.Scatter(
                        x=xvals,
                        y=yvals,
                        mode="lines+markers+text" if label_words else "lines+markers",
                        text=text_vals,
                        textposition="middle right",
                        line=dict(color=color, width=1.6),
                        marker=dict(size=5, color=color),
                        showlegend=False,
                        hovertemplate=hover,
                        opacity=0.85,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            fig.update_xaxes(
                tickmode="array",
                tickvals=xvals,
                ticktext=labels,
                range=[-0.2, last_x + (0.6 if label_words else 0.2)],
                row=row_idx,
                col=col_idx,
            )
        fig.update_yaxes(type="log", title="probability", exponentformat="e")
    else:
        fig = go.Figure()
        for _, row in sig.iterrows():
            traj = row.get("trajectory", "flat")
            color = shape_colors.get(traj, neutral_color) if color_by_shape else neutral_color
            yvals = _row_yvals(row)
            hover = _row_hover(row)
            text_vals = [""] * (len(cols) - 1) + [row["word"]] if label_words else None

            fig.add_trace(
                go.Scatter(
                    x=xvals,
                    y=yvals,
                    mode="lines+markers+text" if label_words else "lines+markers",
                    text=text_vals,
                    textposition="middle right",
                    line=dict(color=color, width=1.6),
                    marker=dict(size=5, color=color),
                    showlegend=False,
                    hovertemplate=hover,
                    opacity=0.8,
                )
            )

        if color_by_shape:
            for name in shape_order:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=shape_colors[name], width=3),
                        name=name,
                    )
                )

    ttl = "Formation trajectories"
    if prompt:
        ttl += f': "{prompt}"'

    layout_kwargs = {
        "title": dict(text=ttl),
        "height": (
            height
            if height is not None
            else (
                380 * int(np.ceil(max(1, len([s for s in shape_order if s in set(sig["trajectory"].tolist())])) / max(1, facet_cols))) + 180
                if facet_by_shape
                else 800
            )
        ),
        "width": width,
        "template": "plotly_white",
        "legend": dict(title="trajectory", orientation="h", yanchor="bottom", y=1.02),
    }
    if not facet_by_shape:
        layout_kwargs["xaxis"] = dict(
            tickmode="array",
            tickvals=xvals,
            ticktext=labels,
            range=[-0.2, last_x + (0.6 if label_words else 0.2)],
        )
        layout_kwargs["yaxis"] = dict(type="log", title="probability", exponentformat="e")

    fig.update_layout(**layout_kwargs)

    fig.add_hline(
        y=TRAJECTORY_THRESHOLD,
        line=dict(color="grey", width=1, dash="dot"),
    )

    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, f"plot_formation.{prompt[:100]}.png")
        fig.write_image(save_path, scale=2)
    return fig


def plot_sublimation(dm, prompt, min_prob=0.003, min_sim=0.5, save_path=None):
    """Visualize base->ego sublimation with displacement links."""
    df = dm['df'].copy()
    wp = df.set_index('word')

    sig = df[
        (df['base'] > min_prob) | (df['ego'] > min_prob)
    ].copy()
    words = sig['word'].tolist()

    def best_pairs(pairs_list):
        if not pairs_list:
            return pd.DataFrame(columns=['source', 'target', 'sim', 'layer'])
        pdf = pd.DataFrame(pairs_list, columns=['source', 'target', 'sim', 'layer'])
        best = pdf.loc[pdf.groupby(['source', 'target'])['sim'].idxmax()]
        return best[best['sim'] >= min_sim]

    sub_best = best_pairs(dm.get('sublimation', {}).get('pairs', []))

    sub_from = defaultdict(list)
    sub_to = defaultdict(list)

    for _, r in sub_best.iterrows():
        sub_from[r['source']].append((r['target'], r['sim']))
        sub_to[r['target']].append((r['source'], r['sim']))

    def fmt_links(pairs):
        return ', '.join(f'{w} ({s:.2f})' for w, s in sorted(pairs, key=lambda x: -x[1]))

    def tooltip(word):
        parts = [f'<b>{word}</b>']
        if word in wp.index:
            r = wp.loc[word]
            parts.append(f'base={r["base"]:.4f}  ego={r["ego"]:.4f}')
        if sub_to.get(word):
            parts.append(f'\u2190 condensed from: {fmt_links(sub_to[word])}')
        if sub_from.get(word):
            parts.append(f'\u2192 sublimated to: {fmt_links(sub_from[word])}')
        return '<br>'.join(parts)

    traces = []

    for _, row in sig.iterrows():
        ys = [max(row['base'], 1e-6), max(row['ego'], 1e-6)]
        traces.append(go.Scatter(
            x=[0, 1], y=ys, mode='lines',
            line=dict(color='rgba(200,200,200,0.25)', width=2),
            showlegend=False, hoverinfo='skip',
        ))

    def delta_color(y_src, y_tgt, alpha=0.6):
        ratio = np.log10(max(y_tgt, 1e-7) / max(y_src, 1e-7))
        t = np.clip(ratio / 1.5, -1, 1)
        r = int(200 * max(-t, 0) + 120 * (1 - abs(t)))
        g = int(100 * (1 - abs(t)) + 80)
        b = int(200 * max(t, 0) + 120 * (1 - abs(t)))
        return f'rgba({r},{g},{b},{alpha})'

    for _, r in sub_best.iterrows():
        src, tgt = r['source'], r['target']
        if src not in wp.index or tgt not in wp.index:
            continue
        y0 = max(wp.loc[src, 'base'], 1e-6)
        y1 = max(wp.loc[tgt, 'ego'], 1e-6)
        traces.append(go.Scatter(
            x=[0, 1], y=[y0, y1], mode='lines',
            line=dict(color=delta_color(y0, y1), width=1.5, dash='dot'),
            showlegend=False, opacity=0.7,
            hovertemplate=(
                f'<b>sublimation</b><br>{src} \u2192 {tgt}'
                f'<br>sim = {r["sim"]:.3f} (peak layer {int(r["layer"])})'
                f'<br>{src}: base={y0:.4f}  |  {tgt}: ego={y1:.4f}'
                '<extra></extra>'
            ),
        ))

    colors_m = {'base': '#e45756', 'ego': '#4c78a8'}
    for layer_name, x_pos in [('base', 0), ('ego', 1)]:
        probs = sig[layer_name].clip(lower=1e-6)
        tips = [tooltip(w) for w in words]
        traces.append(go.Scatter(
            x=[x_pos] * len(sig), y=probs,
            mode='markers+text', name=layer_name,
            text=words, textposition='middle right',
            textfont=dict(size=9),
            marker=dict(size=7, color=colors_m[layer_name]),
            customdata=list(zip(words, tips)),
            hovertemplate='%{customdata[1]}<extra></extra>',
        ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        xaxis=dict(
            tickmode='array', tickvals=[0, 1],
            ticktext=['base (pretrained)', 'ego (SFT)'],
            range=[-0.4, 1.9],
        ),
        yaxis=dict(type='log', title='probability', exponentformat='e'),
        title=dict(text=f'Sublimation map: "{prompt}"'),
        height=800,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        annotations=[
            dict(x=0.05, y=1.07, xref='paper', yref='paper',
                 text='<span style="color:rgb(200,80,80)">\u2501\u2501</span> losing probability  '
                      '<span style="color:rgb(80,80,200)">\u2501\u2501</span> gaining probability',
                 showarrow=False, font=dict(size=11)),
        ],
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


def plot_displacement(dm, prompt, min_prob=0.003, min_sim=0.5, save_path=None):
    """Visualize displacement across all training layers with link arrows.

    Supports 3-layer (base/ego/superego) and 4-layer (+ instruct) topologies.
    """
    df = dm['df'].copy()
    wp = df.set_index('word')

    cols = _layer_columns(df)
    labels = _layer_labels(cols)
    n_layers = len(cols)

    sig = df[
        df[cols].max(axis=1) > min_prob
    ].copy()
    words = sig['word'].tolist()

    def best_pairs(pairs_list):
        if not pairs_list:
            return pd.DataFrame(columns=['source', 'target', 'sim', 'layer'])
        pdf = pd.DataFrame(pairs_list, columns=['source', 'target', 'sim', 'layer'])
        best = pdf.loc[pdf.groupby(['source', 'target'])['sim'].idxmax()]
        return best[best['sim'] >= min_sim]

    sub_best = best_pairs(dm.get('sublimation', {}).get('pairs', []))
    rep_best = best_pairs(dm.get('repression', {}).get('pairs', []))

    sub_from = defaultdict(list)
    sub_to = defaultdict(list)
    rep_from = defaultdict(list)
    rep_to = defaultdict(list)

    for _, r in sub_best.iterrows():
        sub_from[r['source']].append((r['target'], r['sim']))
        sub_to[r['target']].append((r['source'], r['sim']))
    for _, r in rep_best.iterrows():
        rep_from[r['source']].append((r['target'], r['sim']))
        rep_to[r['target']].append((r['source'], r['sim']))

    def fmt_links(pairs):
        return '\n'.join(f'{w} ({s:.2f})' for w, s in sorted(pairs, key=lambda x: -x[1]))

    def tooltip(word):
        parts = [f'<b>{word}</b>']
        if word in wp.index:
            r = wp.loc[word]
            vals = '  '.join(f'{c}={r[c]:.4f}' for c in cols)
            parts.append(vals)
        if sub_to.get(word):
            parts.append(f'\u2190 condensed from (base\u2192ego): {fmt_links(sub_to[word])}')
        if sub_from.get(word):
            parts.append(f'\u2192 sublimated to (base\u2192ego): {fmt_links(sub_from[word])}')
        rep_label = 'ego\u2192superego' if has_ego else 'base\u2192superego'
        if rep_to.get(word):
            parts.append(f'\u2190 condensed from ({rep_label}): {fmt_links(rep_to[word])}')
        if rep_from.get(word):
            parts.append(f'\u2192 repressed to ({rep_label}): {fmt_links(rep_from[word])}')
        return '<br>'.join(parts)

    traces = []

    for _, row in sig.iterrows():
        ys = [max(row[c], 1e-6) for c in cols]
        traces.append(go.Scatter(
            x=list(range(n_layers)), y=ys, mode='lines',
            line=dict(color='rgba(200,200,200,0.25)', width=2),
            showlegend=False, hoverinfo='skip',
        ))

    def delta_color(y_src, y_tgt, alpha=0.6):
        ratio = np.log10(max(y_tgt, 1e-7) / max(y_src, 1e-7))
        t = np.clip(ratio / 1.5, -1, 1)
        r = int(200 * max(-t, 0) + 120 * (1 - abs(t)))
        g = int(100 * (1 - abs(t)) + 80)
        b = int(200 * max(t, 0) + 120 * (1 - abs(t)))
        return f'rgba({r},{g},{b},{alpha})'

    # Sublimation links (base -> ego, x=0 -> x=1)
    for _, r in sub_best.iterrows():
        src, tgt = r['source'], r['target']
        if src not in wp.index or tgt not in wp.index:
            continue
        y0 = max(wp.loc[src, 'base'], 1e-6)
        y1 = max(wp.loc[tgt, 'ego'], 1e-6)
        traces.append(go.Scatter(
            x=[0, 1], y=[y0, y1], mode='lines',
            line=dict(color=delta_color(y0, y1), width=1.5, dash='dot'),
            showlegend=False, opacity=0.7,
            hovertemplate=(
                f'<b>sublimation</b><br>{src} \u2192 {tgt}'
                f'<br>sim = {r["sim"]:.3f} (peak layer {int(r["layer"])})'
                f'<br>{src}: base={y0:.4f}  |  {tgt}: ego={y1:.4f}'
                '<extra></extra>'
            ),
        ))

    # Repression links
    has_ego = 'ego' in cols
    if has_ego:
        # 3+ layers: ego -> superego (x=1 -> x=2)
        rep_src_col, rep_tgt_col = 'ego', 'superego'
        rep_x0, rep_x1 = 1, 2
    else:
        # 2 layers: base -> superego (x=0 -> x=1)
        rep_src_col, rep_tgt_col = 'base', 'superego'
        rep_x0, rep_x1 = 0, 1
    for _, r in rep_best.iterrows():
        src, tgt = r['source'], r['target']
        if src not in wp.index or tgt not in wp.index:
            continue
        y0 = max(wp.loc[src, rep_src_col], 1e-6)
        y1 = max(wp.loc[tgt, rep_tgt_col], 1e-6)
        traces.append(go.Scatter(
            x=[rep_x0, rep_x1], y=[y0, y1], mode='lines',
            line=dict(color=delta_color(y0, y1), width=1.5, dash='dot'),
            showlegend=False, opacity=0.7,
            hovertemplate=(
                f'<b>repression</b><br>{src} \u2192 {tgt}'
                f'<br>sim = {r["sim"]:.3f} (peak layer {int(r["layer"])})'
                f'<br>{src}: {rep_src_col}={y0:.4f}  |  {tgt}: {rep_tgt_col}={y1:.4f}'
                '<extra></extra>'
            ),
        ))

    layer_colors = {
        'base': '#e45756', 'ego': '#4c78a8',
        'superego': '#72b7b2', 'instruct': '#eeca3b',
    }
    for i, col in enumerate(cols):
        probs = sig[col].clip(lower=1e-6)
        tips = [tooltip(w) for w in words]
        traces.append(go.Scatter(
            x=[i] * len(sig), y=probs,
            mode='markers+text', name=col,
            text=words, textposition='middle right',
            textfont=dict(size=9),
            marker=dict(size=7, color=layer_colors.get(col, '#999')),
            customdata=list(zip(words, tips)),
            hovertemplate='%{customdata[1]}<extra></extra>',
        ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(n_layers)),
            ticktext=labels,
            range=[-0.4, n_layers - 1 + 0.9],
        ),
        yaxis=dict(type='log', title='probability', exponentformat='e'),
        title=dict(text=f'Displacement map: "{prompt}"'),
        height=800,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        annotations=[
            dict(x=0.05, y=1.07, xref='paper', yref='paper',
                 text='<span style="color:rgb(200,80,80)">\u2501\u2501</span> losing probability  '
                      '<span style="color:rgb(80,80,200)">\u2501\u2501</span> gaining probability  '
                      '<span style="color:rgb(140,140,140)">\u2504\u2504</span> displacement (dashed)',
                 showarrow=False, font=dict(size=11)),
        ],
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


def plot_layer_displacement(dm, prompt, source_word=None, min_sim=0.4, top_n=8, save_path=None):
    """Visualize how displacement similarity evolves across hidden layers.

    3+ layers: uses sublimation pairs (base→ego).
    2 layers: uses repression pairs (base→superego).
    """
    # Use sublimation pairs if available, otherwise repression
    sub_pairs = dm.get('sublimation', {}).get('pairs', [])
    rep_pairs = dm.get('repression', {}).get('pairs', [])
    if sub_pairs:
        pairs = sub_pairs
        axis_name = 'sublimation'
        tgt_prob_col = 'ego'
    elif rep_pairs:
        pairs = rep_pairs
        axis_name = 'repression'
        tgt_prob_col = 'superego'
    else:
        print("No displacement pairs found")
        return

    pairs_df = pd.DataFrame(pairs, columns=['source', 'target', 'sim', 'layer'])
    wp = dm['df'].set_index('word')

    if source_word:
        sources = [source_word]
    else:
        sources = (
            pairs_df.groupby('source')['sim'].max()
            .nlargest(3).index.tolist()
        )

    figs = []
    for src in sources:
        sdf = pairs_df[pairs_df['source'] == src]
        top_targets = (
            sdf.groupby('target')['sim'].max()
            .nlargest(top_n).index.tolist()
        )

        fig = go.Figure()
        base_prob = wp.loc[src, 'base'] if src in wp.index else 0
        tgt_probs = {
            t: wp.loc[t, tgt_prob_col]
            for t in top_targets
            if t in wp.index and tgt_prob_col in wp.columns
        }

        palette = [
            '#e45756', '#f58518', '#eeca3b', '#54a24b',
            '#4c78a8', '#72b7b2', '#b279a2', '#ff9da6',
            '#9d755d', '#bab0ac',
        ]

        for i, tgt in enumerate(top_targets):
            tdf = sdf[sdf['target'] == tgt].sort_values('layer')
            color = palette[i % len(palette)]
            peak_row = tdf.loc[tdf['sim'].idxmax()]
            peak_layer = int(peak_row['layer'])
            peak_sim = peak_row['sim']

            fig.add_trace(go.Scatter(
                x=tdf['layer'].tolist(),
                y=tdf['sim'].tolist(),
                mode='lines+markers',
                name=f'{tgt}',
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
                hovertemplate=(
                    f'<b>{src} \u2192 {tgt}</b>'
                    '<br>layer %{x}'
                    '<br>similarity = %{y:.4f}'
                    '<extra></extra>'
                ),
            ))

            fig.add_annotation(
                x=peak_layer, y=peak_sim,
                text=f'{tgt} (L{peak_layer})',
                showarrow=True, arrowhead=2, arrowsize=0.8,
                ax=20, ay=-20,
                font=dict(size=9, color=color),
                arrowcolor=color,
            )

            if tgt in tgt_probs:
                fig.add_trace(go.Scatter(
                    x=[0], y=[tgt_probs[tgt]],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='diamond'),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{tgt}</b> {tgt_prob_col} probability = {tgt_probs[tgt]:.4f}'
                        '<extra></extra>'
                    ),
                ))

        fig.add_trace(go.Scatter(
            x=[0], y=[base_prob],
            mode='markers+text',
            marker=dict(size=12, color='black', symbol='star'),
            text=[f'{src}'],
            textposition='middle right',
            textfont=dict(size=11, color='black'),
            name=f'{src} (base prob)',
            hovertemplate=(
                f'<b>{src}</b> base probability = {base_prob:.4f}'
                '<extra></extra>'
            ),
        ))

        # Shaded regions for interpretive context
        fig.add_vrect(x0=0.5, x1=8.5, fillcolor='rgba(255,200,200,0.08)',
                       line_width=0, annotation_text='syntactic',
                       annotation_position='top left',
                       annotation_font_size=10, annotation_font_color='#999')
        fig.add_vrect(x0=8.5, x1=22.5, fillcolor='rgba(200,255,200,0.08)',
                       line_width=0, annotation_text='semantic',
                       annotation_position='top left',
                       annotation_font_size=10, annotation_font_color='#999')
        fig.add_vrect(x0=22.5, x1=32.5, fillcolor='rgba(200,200,255,0.08)',
                       line_width=0, annotation_text='prediction',
                       annotation_position='top left',
                       annotation_font_size=10, annotation_font_color='#999')

        embed_model = 'SFT' if axis_name == 'sublimation' else 'instruct'
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0] + list(range(1, 33)),
                ticktext=['base'] + [str(i) for i in range(1, 33)],
                title=f'base model \u2192 {embed_model} model hidden layers',
                range=[-0.5, 33],
            ),
            yaxis=dict(
                title=f'cosine similarity to displacement target<br>'
                      f'<span style="font-size:10px">(\u2666 at base = {tgt_prob_col} probability of target)</span>',
            ),
            title=dict(text=f'Displacement through layers: "{src}" \u2014 "{prompt}"'),
            height=550,
            template='plotly_white',
            legend=dict(title='target word'),
        )
        figs.append(fig)
        if save_path:
            path = save_path if isinstance(save_path, str) else None
            if path and os.path.isdir(path):
                path = os.path.join(path, f"displacement_layers_{prompt[:100]}_{src}.png")
            if path:
                fig.write_image(path, scale=2)

    return figs


def _categorize_label(label):
    """Extract content category from prompt label."""
    if label.startswith("sexual_liminal"):
        return "sexual (liminal)"
    elif label.startswith("sexual_explicit"):
        return "sexual (explicit)"
    elif label.startswith("violence"):
        return "violence"
    elif label.startswith("neutral"):
        return "neutral"
    return "other"


CATEGORY_COLORS = {
    "sexual (liminal)": "#b07aa1",
    "sexual (explicit)": "#e15759",
    "violence": "#4e79a7",
    "neutral": "#9c9c9c",
    "other": "#76b7b2",
}

CATEGORY_ORDER = ["sexual (liminal)", "sexual (explicit)", "violence", "neutral"]


def plot_battery_metrics(metrics_df, save_path=None):
    """Plot battery-level aggregate metrics grouped by content category.

    Args:
        metrics_df: DataFrame from Psyche.battery_metrics().
        save_path: Optional directory or file path to save figures.

    Returns:
        dict of {metric_name: plotly Figure}.
    """
    from plotly.subplots import make_subplots

    df = metrics_df.copy()
    df["category"] = df["label"].apply(_categorize_label)

    # Aggregate by category
    cat_order = [c for c in CATEGORY_ORDER if c in df["category"].values]
    agg = df.groupby("category").agg({
        "js_base_ego": "mean",
        "js_ego_superego": "mean",
        "js_base_superego": "mean",
        "entropy_base": "mean",
        "entropy_ego": "mean",
        "entropy_superego": "mean",
        "entropy_drop_sft": "mean",
        "entropy_drop_dpo": "mean",
        "top50_overlap_base_ego": "mean",
        "top50_overlap_ego_superego": "mean",
        "top50_overlap_base_superego": "mean",
    }).reindex(cat_order)

    figs = {}

    # --- Figure 1: JS divergence by stage and category ---
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="Base ↔ Ego (SFT)",
        x=cat_order,
        y=agg["js_base_ego"],
        marker_color="#e45756",
    ))
    fig1.add_trace(go.Bar(
        name="Ego ↔ Superego (DPO)",
        x=cat_order,
        y=agg["js_ego_superego"],
        marker_color="#4c78a8",
    ))
    fig1.update_layout(
        title="Distributional distance by training stage and content type",
        yaxis_title="Jensen-Shannon divergence",
        barmode="group",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    figs["js_by_stage"] = fig1

    # --- Figure 2: Entropy drop by stage ---
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="SFT (ego formation)",
        x=cat_order,
        y=agg["entropy_drop_sft"],
        marker_color="#e45756",
    ))
    fig2.add_trace(go.Bar(
        name="DPO (repression)",
        x=cat_order,
        y=agg["entropy_drop_dpo"],
        marker_color="#4c78a8",
    ))
    fig2.update_layout(
        title="Entropy narrowing by training stage and content type",
        yaxis_title="Entropy drop (nats)",
        barmode="group",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig2.add_hline(y=0, line=dict(color="grey", width=1, dash="dot"))
    figs["entropy_drop"] = fig2

    # --- Figure 3: Entropy per layer ---
    fig3 = go.Figure()
    for layer_name, col, color in [
        ("Base", "entropy_base", "#e45756"),
        ("Ego (SFT)", "entropy_ego", "#4c78a8"),
        ("Superego (DPO)", "entropy_superego", "#72b7b2"),
    ]:
        fig3.add_trace(go.Bar(
            name=layer_name,
            x=cat_order,
            y=agg[col],
            marker_color=color,
        ))
    fig3.update_layout(
        title="Distribution entropy by layer and content type",
        yaxis_title="Entropy (nats)",
        barmode="group",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    figs["entropy_by_layer"] = fig3

    # --- Figure 4: Top-50 overlap ---
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="Base ∩ Ego",
        x=cat_order,
        y=agg["top50_overlap_base_ego"],
        marker_color="#e45756",
    ))
    fig4.add_trace(go.Bar(
        name="Ego ∩ Superego",
        x=cat_order,
        y=agg["top50_overlap_ego_superego"],
        marker_color="#4c78a8",
    ))
    fig4.add_trace(go.Bar(
        name="Base ∩ Superego",
        x=cat_order,
        y=agg["top50_overlap_base_superego"],
        marker_color="#72b7b2",
    ))
    fig4.update_layout(
        title="Vocabulary overlap between layers by content type",
        yaxis_title="Top-50 token overlap",
        yaxis_tickformat=".0%",
        barmode="group",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    figs["vocabulary_overlap"] = fig4

    # --- Figure 5: Per-prompt JS scatter ---
    fig5 = go.Figure()
    for cat in cat_order:
        cdf = df[df["category"] == cat]
        fig5.add_trace(go.Scatter(
            x=cdf["js_base_ego"],
            y=cdf["js_ego_superego"],
            mode="markers+text",
            name=cat,
            text=cdf["label"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=10, color=CATEGORY_COLORS.get(cat, "#999")),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "JS base↔ego: %{x:.4f}<br>"
                "JS ego↔superego: %{y:.4f}"
                "<extra></extra>"
            ),
        ))
    fig5.update_layout(
        title="SFT vs DPO distributional impact per prompt",
        xaxis_title="JS base ↔ ego (SFT reshaping)",
        yaxis_title="JS ego ↔ superego (DPO repression)",
        template="plotly_white",
        height=600,
    )
    figs["js_scatter"] = fig5

    # Save all figures
    if save_path:
        save_dir = save_path if os.path.isdir(save_path) else os.path.dirname(save_path)
        if not save_dir:
            save_dir = "."
        for name, fig in figs.items():
            path = os.path.join(save_dir, f"battery_{name}.png")
            fig.write_image(path, scale=2)
            print(f"  Saved {path}")

    return figs


# ── Generation battery visualizations ─────────────────────────────

def plot_logit_vs_generation(battery_csv, gen_metrics_csv, save_path=None):
    """Scatter: logit JS divergence vs generation centroid distance.

    The key figure — tests whether logit-level displacement predicts
    narrative-level divergence.
    """
    bat = pd.read_csv(battery_csv)
    gen = pd.read_csv(gen_metrics_csv)

    # Find the centroid distance column (base vs superego or instruct)
    dist_col = None
    for col in ["centroid_dist_base_superego", "centroid_dist_base_instruct",
                "centroid_dist_base_ego"]:
        if col in gen.columns:
            dist_col = col
            break
    if dist_col is None:
        dist_cols = [c for c in gen.columns if c.startswith("centroid_dist_")]
        if dist_cols:
            dist_col = dist_cols[0]
        else:
            raise ValueError("No centroid distance column found in gen metrics")

    merged = bat.merge(gen, on=["family", "label"], suffixes=("_logit", "_gen"))

    family_colors = {
        "qwen": "#72b7b2", "llama": "#4c78a8",
        "olmo": "#e45756", "amber": "#eeca3b",
    }

    fig = go.Figure()
    for fam in sorted(merged["family"].unique()):
        sub = merged[merged["family"] == fam]
        fig.add_trace(go.Scatter(
            x=sub["js_base_superego"],
            y=sub[dist_col],
            mode="markers",
            name=fam,
            marker=dict(size=8, color=family_colors.get(fam, "#999")),
            text=sub["label"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "JS divergence: %{x:.4f}<br>"
                "Centroid distance: %{y:.4f}<br>"
                "<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        title="Logit displacement vs narrative divergence",
        xaxis_title="JS divergence (base → superego, logits)",
        yaxis_title="Centroid distance (base → superego, generations)",
        template="plotly_white",
        width=700, height=500,
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


def plot_variance_reduction(gen_metrics_csv, save_path=None):
    """Bar chart: variance_ratio per family per content category."""
    df = pd.read_csv(gen_metrics_csv)
    if "variance_ratio" not in df.columns:
        raise ValueError("No variance_ratio column in gen metrics")

    df["category"] = df["label"].str.replace(r"_\d+$", "", regex=True)

    pivot = df.pivot_table(
        values="variance_ratio", index="category", columns="family",
        aggfunc="mean",
    )

    fam_order = [f for f in ["qwen", "llama", "olmo", "amber"]
                 if f in pivot.columns]
    pivot = pivot[fam_order]
    family_colors = {
        "qwen": "#72b7b2", "llama": "#4c78a8",
        "olmo": "#e45756", "amber": "#eeca3b",
    }

    fig = go.Figure()
    for fam in fam_order:
        fig.add_trace(go.Bar(
            x=pivot.index.tolist(),
            y=pivot[fam].tolist(),
            name=fam,
            marker_color=family_colors.get(fam, "#999"),
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="no change", annotation_position="top right")
    fig.update_layout(
        title="Variance ratio (superego / base) — below 1 = alignment compresses diversity",
        yaxis_title="variance ratio",
        barmode="group",
        template="plotly_white",
        width=900, height=500,
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


def plot_concept_shift(gen_metrics_csv, concept="violent", save_path=None):
    """Bar chart: concept shift per family per content category."""
    df = pd.read_csv(gen_metrics_csv)
    shift_col = f"{concept}_shift"
    if shift_col not in df.columns:
        raise ValueError(f"No {shift_col} column in gen metrics")

    df["category"] = df["label"].str.replace(r"_\d+$", "", regex=True)

    pivot = df.pivot_table(
        values=shift_col, index="category", columns="family",
        aggfunc="mean",
    )

    fam_order = [f for f in ["qwen", "llama", "olmo", "amber"]
                 if f in pivot.columns]
    pivot = pivot[fam_order]
    family_colors = {
        "qwen": "#72b7b2", "llama": "#4c78a8",
        "olmo": "#e45756", "amber": "#eeca3b",
    }

    fig = go.Figure()
    for fam in fam_order:
        fig.add_trace(go.Bar(
            x=pivot.index.tolist(),
            y=pivot[fam].tolist(),
            name=fam,
            marker_color=family_colors.get(fam, "#999"),
        ))

    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(
        title=f"Concept shift: {concept} (base → superego)",
        yaxis_title=f"{concept} score shift",
        barmode="group",
        template="plotly_white",
        width=900, height=500,
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


# ── Logit lens visualizations ─────────────────────────────────────

def plot_logit_lens(lens_data, prompt=None, words=None, min_layers=8, save_path=None):
    """Plot word probabilities across network layers for each model.

    Uses plotnine (ggplot2) with adjustText for non-overlapping labels
    at each word's peak probability.

    Args:
        lens_data: DataFrame with columns [layer, word, probability, model]
                   or path to CSV.
        prompt: Prompt string for title.
        words: Filter to only these words (overrides auto-selection).
        min_layers: Minimum layers a top-k word must appear in to be plotted.
    """
    from plotnine import (
        ggplot, aes, geom_line, geom_point, geom_text,
        facet_wrap, scale_y_log10, scale_linetype_manual,
        labs, theme_minimal, theme, element_text,
    )

    if isinstance(lens_data, str):
        lens_data = pd.read_csv(lens_data)
    df = lens_data.copy()

    model_labels = {"base": "BASE", "ego": "SFT", "superego": "DPO", "instruct": "RLVR"}
    df["model_label"] = df["model"].map(model_labels).fillna(df["model"])
    # Order facets
    label_order = [v for v in ["BASE", "SFT", "DPO", "RLVR"] if v in df["model_label"].values]
    df["model_label"] = pd.Categorical(df["model_label"], categories=label_order, ordered=True)

    # Determine tracked vs top-k
    tracked_set = set()
    if "source" in df.columns:
        tracked_set = set(df[df["source"] == "tracked"]["word"].unique())

    if words:
        df = df[df["word"].isin(words)]
        plot_words = words
    else:
        tracked = list(tracked_set)
        topk = df[df.get("source", pd.Series()) == "top_k"]
        topk_counts = topk.groupby("word")["layer"].nunique()
        frequent_topk = topk_counts[topk_counts >= min_layers].index.tolist()
        plot_words = list(dict.fromkeys(tracked + frequent_topk))
        df = df[df["word"].isin(plot_words)]

    df["linetype"] = df["word"].apply(lambda w: "tracked" if w in tracked_set else "top-k")

    # Build label df: each word labeled at its peak probability per model
    label_rows = []
    for (model, word), grp in df.groupby(["model_label", "word"]):
        peak = grp.loc[grp["probability"].idxmax()]
        label_rows.append(peak)
    label_df = pd.DataFrame(label_rows)

    title = "Logit lens: word probability at each network layer"
    if prompt:
        title += f'\n"{prompt[:80]}"'

    n_models = df["model_label"].nunique()

    p = (
        ggplot(df, aes(x="layer", y="probability", color="word", linetype="linetype"))
        + geom_line(size=0.7)
        + geom_point(size=1)
        + geom_text(
            aes(label="word"),
            data=label_df,
            size=7, adjust_text={
                "arrowprops": {"arrowstyle": "-", "color": "gray", "lw": 0.5},
                "expand_points": (1.5, 1.5),
                "force_text": (0.5, 1.0),
            },
        )
        + facet_wrap("model_label", nrow=1)
        + scale_y_log10()
        + scale_linetype_manual(values={"tracked": "solid", "top-k": "dashed"})
        + labs(
            title=title,
            x="network layer",
            y="probability (log scale)",
            color="word",
            linetype="",
        )
        + theme_minimal()
        + theme(
            figure_size=(5 * n_models, 6),
            plot_title=element_text(size=12),
            strip_text=element_text(size=11, weight="bold"),
            legend_position="right",
        )
    )

    if save_path:
        p.save(save_path, dpi=300)
    return p


# ── Step-level checkpoint visualizations ──────────────────────────

_WORD_PALETTE = [
    "#e45756", "#4c78a8", "#eeca3b", "#54a24b", "#f58518",
    "#72b7b2", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
    "#5778a4", "#e49444", "#d1615d", "#85b6b2", "#6a9f58",
    "#e7ca60", "#a87c9f", "#f1a2a9", "#967662", "#b8b0a2",
]


def plot_repression_curves(words_csv, prompt_label=None, words=None, save_path=None):
    """Line plot: probability vs training step for tracked words."""
    df = pd.read_csv(words_csv)
    if prompt_label:
        df = df[df["label"] == prompt_label]
    if words:
        df = df[df["word"].isin(words)]

    # Order words by base probability (highest first) for legend readability
    word_order = (
        df.groupby("word")["base_probability"].mean()
        .sort_values(ascending=False).index.tolist()
    )

    fig = go.Figure()
    for i, word in enumerate(word_order):
        wdf = df[df["word"] == word].sort_values("step")
        base_prob = wdf["base_probability"].iloc[0]
        steps = [0] + wdf["step"].tolist()
        probs = [base_prob] + wdf["probability"].tolist()
        color = _WORD_PALETTE[i % len(_WORD_PALETTE)]
        fig.add_trace(go.Scatter(
            x=steps, y=probs, mode="lines+markers",
            name=word,
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
        ))

    title = "Repression onset curves (probability vs SFT training step)"
    if prompt_label:
        prompt_text = df["prompt"].iloc[0] if len(df) > 0 else prompt_label
        title += f"<br><sub>{prompt_text}</sub>"

    fig.update_layout(
        title=title,
        xaxis_title="SFT training step (0 = base model)",
        yaxis_title="probability (log scale)",
        yaxis_type="log",
        template="plotly_white", width=900, height=550,
        legend=dict(title="word"),
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


def plot_step_metrics(metrics_csv, metric="js_base_step", save_path=None):
    """Line plot: distributional metric vs training step, by prompt category."""
    df = pd.read_csv(metrics_csv)
    df["category"] = df["label"].str.replace(r"_\d+$", "", regex=True)
    agg = df.groupby(["step", "category"])[metric].mean().reset_index()

    cat_colors = {
        "neutral": "#999999", "death": "#666666",
        "profanity": "#eeca3b", "substance": "#f58518",
        "power": "#54a24b", "sexual_liminal": "#ff9da6",
        "sexual_explicit": "#e45756", "violence_liminal": "#72b7b2",
        "violence_explicit": "#4c78a8",
    }

    fig = go.Figure()
    for cat in sorted(agg["category"].unique()):
        cdf = agg[agg["category"] == cat].sort_values("step")
        fig.add_trace(go.Scatter(
            x=cdf["step"], y=cdf[metric], mode="lines+markers", name=cat,
            line=dict(color=cat_colors.get(cat, "#999"), width=2),
            marker=dict(size=5),
        ))

    metric_labels = {
        "js_base_step": "JS divergence from base",
        "kl_base_step": "KL divergence from base",
        "entropy_step": "Entropy", "entropy_drop": "Entropy drop from base",
        "top50_overlap": "Top-50 overlap with base",
    }
    fig.update_layout(
        title=f"{metric_labels.get(metric, metric)} across SFT training steps",
        xaxis_title="SFT training step",
        yaxis_title=metric_labels.get(metric, metric),
        template="plotly_white", width=900, height=550,
        legend=dict(title="category"),
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


def plot_displacement_lag(words_csv, repressed_word, displaced_word,
                          prompt_label, save_path=None):
    """Dual-axis plot: repressed word falling, displacement target rising."""
    df = pd.read_csv(words_csv)
    df = df[df["label"] == prompt_label]
    rep = df[df["word"] == repressed_word].sort_values("step")
    disp = df[df["word"] == displaced_word].sort_values("step")

    if rep.empty or disp.empty:
        print(f"Words not found: {repressed_word}, {displaced_word}")
        return None

    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    rep_steps = [0] + rep["step"].tolist()
    rep_probs = [float(rep["base_probability"].iloc[0])] + rep["probability"].tolist()
    fig.add_trace(go.Scatter(
        x=rep_steps, y=rep_probs, mode="lines+markers",
        name=f"{repressed_word} (repressed)",
        line=dict(color="#e45756", width=3),
    ), secondary_y=False)

    disp_steps = [0] + disp["step"].tolist()
    disp_probs = [float(disp["base_probability"].iloc[0])] + disp["probability"].tolist()
    fig.add_trace(go.Scatter(
        x=disp_steps, y=disp_probs, mode="lines+markers",
        name=f"{displaced_word} (displacement target)",
        line=dict(color="#4c78a8", width=3),
    ), secondary_y=True)

    prompt_text = rep["prompt"].iloc[0] if len(rep) > 0 else prompt_label
    fig.update_layout(
        title=f"Displacement lag: {repressed_word} → {displaced_word}<br><sub>{prompt_text}</sub>",
        xaxis_title="SFT training step (0 = base model)",
        template="plotly_white", width=800, height=500,
    )
    fig.update_yaxes(title_text=f"P({repressed_word})", secondary_y=False, color="#e45756")
    fig.update_yaxes(title_text=f"P({displaced_word})", secondary_y=True, color="#4c78a8")

    if save_path:
        fig.write_image(save_path, scale=2)
    return fig
