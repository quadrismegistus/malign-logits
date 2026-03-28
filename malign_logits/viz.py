from . import *
from .psyche import TRAJECTORY_THRESHOLD


def _layer_columns(df):
    """Detect which layer columns are present in a formation DataFrame."""
    cols = ["base", "ego", "superego"]
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

    required = {"word", "base", "ego", "superego"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"formation DataFrame missing required columns: {sorted(missing)}"
        )

    cols = _layer_columns(df)
    labels = _layer_labels(cols)
    xvals = list(range(len(cols)))

    if "trajectory" not in df.columns:
        def _classify(row):
            b, e, s = row["base"], row["ego"], row["superego"]
            t = 0.005
            if b - e > t and e - s > t:
                return "decline"
            if e - b > t and s - e > t:
                return "rise"
            if b - e > t and s - e > t:
                return "V"
            if e - b > t and e - s > t:
                return "peak"
            if b > t and e < t and s < t:
                return "sublimated"
            if b < t and e < t and s > t:
                return "superego_only"
            return "flat"

        df["trajectory"] = df.apply(_classify, axis=1)

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
        if rep_to.get(word):
            parts.append(f'\u2190 condensed from (ego\u2192superego): {fmt_links(rep_to[word])}')
        if rep_from.get(word):
            parts.append(f'\u2192 repressed to (ego\u2192superego): {fmt_links(rep_from[word])}')
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

    # Repression links (ego -> superego, x=1 -> x=2)
    for _, r in rep_best.iterrows():
        src, tgt = r['source'], r['target']
        if src not in wp.index or tgt not in wp.index:
            continue
        y0 = max(wp.loc[src, 'ego'], 1e-6)
        y1 = max(wp.loc[tgt, 'superego'], 1e-6)
        traces.append(go.Scatter(
            x=[1, 2], y=[y0, y1], mode='lines',
            line=dict(color=delta_color(y0, y1), width=1.5, dash='dot'),
            showlegend=False, opacity=0.7,
            hovertemplate=(
                f'<b>repression</b><br>{src} \u2192 {tgt}'
                f'<br>sim = {r["sim"]:.3f} (peak layer {int(r["layer"])})'
                f'<br>{src}: ego={y0:.4f}  |  {tgt}: superego={y1:.4f}'
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
    """Visualize how displacement similarity evolves across hidden layers."""
    sub_pairs = dm.get('sublimation', {}).get('pairs', [])
    if not sub_pairs:
        print("No sublimation pairs found")
        return

    sub_df = pd.DataFrame(sub_pairs, columns=['source', 'target', 'sim', 'layer'])
    wp = dm['df'].set_index('word')

    if source_word:
        sources = [source_word]
    else:
        sources = (
            sub_df.groupby('source')['sim'].max()
            .nlargest(3).index.tolist()
        )

    figs = []
    for src in sources:
        sdf = sub_df[sub_df['source'] == src]
        top_targets = (
            sdf.groupby('target')['sim'].max()
            .nlargest(top_n).index.tolist()
        )

        fig = go.Figure()
        base_prob = wp.loc[src, 'base'] if src in wp.index else 0
        ego_probs = {t: wp.loc[t, 'ego'] for t in top_targets if t in wp.index}

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

            if tgt in ego_probs:
                fig.add_trace(go.Scatter(
                    x=[0], y=[ego_probs[tgt]],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='diamond'),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{tgt}</b> ego probability = {ego_probs[tgt]:.4f}'
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

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0] + list(range(1, 33)),
                ticktext=['base'] + [str(i) for i in range(1, 33)],
                title='base model \u2192 SFT model hidden layers',
                range=[-0.5, 33],
            ),
            yaxis=dict(
                title='cosine similarity to displacement target<br>'
                      '<span style="font-size:10px">(\u2666 at base = ego probability of target)</span>',
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
