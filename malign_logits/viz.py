from . import *



def plot_sublimation(dm, prompt, min_prob=0.003, min_sim=0.5):
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

    def add(trace):
        traces.append(trace)

    for _, row in sig.iterrows():
        ys = [max(row['base'], 1e-6), max(row['ego'], 1e-6)]
        add(go.Scatter(
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
        add(go.Scatter(
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
        add(go.Scatter(
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
            ticktext=['base (pretrained)', 'ego (RLHF)'],
            range=[-0.4, 1.9],
        ),
        yaxis=dict(type='log', title='probability', exponentformat='e'),
        title=dict(text=f'Sublimation map: "{prompt}"'),
        height=800, width=1200,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        annotations=[
            dict(x=0.05, y=1.07, xref='paper', yref='paper',
                 text='<span style="color:rgb(200,80,80)">\u2501\u2501</span> losing probability  '
                      '<span style="color:rgb(80,80,200)">\u2501\u2501</span> gaining probability',
                 showarrow=False, font=dict(size=11)),
        ],
    )
    fig.write_image(f"../figures/sublimation_{prompt[:100]}.png", scale=2)
    return fig

# fig = plot_sublimation(dm, prompt, min_sim=0.75)
# fig.show()



# import plotly.graph_objects as go
# import numpy as np
# from collections import defaultdict

# def plot_displacement(dm, prompt, min_prob=0.003, min_sim=0.5):
#     df = dm['df'].copy()
#     wp = df.set_index('word')

#     sig = df[
#         (df['base'] > min_prob) | (df['ego'] > min_prob) | (df['superego'] > min_prob)
#     ].copy()
#     words = sig['word'].tolist()

#     def best_pairs(pairs_list):
#         if not pairs_list:
#             return pd.DataFrame(columns=['source', 'target', 'sim', 'layer'])
#         pdf = pd.DataFrame(pairs_list, columns=['source', 'target', 'sim', 'layer'])
#         best = pdf.loc[pdf.groupby(['source', 'target'])['sim'].idxmax()]
#         return best[best['sim'] >= min_sim]

#     sub_best = best_pairs(dm.get('sublimation', {}).get('pairs', []))
#     rep_best = best_pairs(dm.get('repression', {}).get('pairs', []))

#     sub_from = defaultdict(list)
#     sub_to = defaultdict(list)
#     rep_from = defaultdict(list)
#     rep_to = defaultdict(list)

#     for _, r in sub_best.iterrows():
#         sub_from[r['source']].append((r['target'], r['sim']))
#         sub_to[r['target']].append((r['source'], r['sim']))
#     for _, r in rep_best.iterrows():
#         rep_from[r['source']].append((r['target'], r['sim']))
#         rep_to[r['target']].append((r['source'], r['sim']))

#     def fmt_links(pairs):
#         return ', '.join(f'{w} ({s:.2f})' for w, s in sorted(pairs, key=lambda x: -x[1]))

#     def tooltip(word):
#         parts = [f'<b>{word}</b>']
#         if word in wp.index:
#             r = wp.loc[word]
#             parts.append(f'base={r["base"]:.4f}  ego={r["ego"]:.4f}  super={r["superego"]:.4f}')
#         if sub_to.get(word):
#             parts.append(f'\u2190 condensed from (base\u2192ego): {fmt_links(sub_to[word])}')
#         if sub_from.get(word):
#             parts.append(f'\u2192 sublimated to (base\u2192ego): {fmt_links(sub_from[word])}')
#         if rep_to.get(word):
#             parts.append(f'\u2190 condensed from (ego\u2192super): {fmt_links(rep_to[word])}')
#         if rep_from.get(word):
#             parts.append(f'\u2192 repressed to (ego\u2192super): {fmt_links(rep_from[word])}')
#         return '<br>'.join(parts)

#     word_trace_ids = defaultdict(list)
#     traces = []

#     def add(trace, *related_words):
#         idx = len(traces)
#         traces.append(trace)
#         for w in related_words:
#             word_trace_ids[w].append(idx)

#     for _, row in sig.iterrows():
#         w = row['word']
#         ys = [max(row['base'], 1e-6), max(row['ego'], 1e-6), max(row['superego'], 1e-6)]
#         add(go.Scatter(
#             x=[0, 1, 2], y=ys, mode='lines',
#             line=dict(color='rgba(200,200,200,0.25)', width=2),
#             showlegend=False, hoverinfo='skip',
#         ), w)

#     def delta_color(y_src, y_tgt, alpha=0.6):
#         ratio = np.log10(max(y_tgt, 1e-7) / max(y_src, 1e-7))
#         t = np.clip(ratio / 1.5, -1, 1)
#         r = int(200 * max(-t, 0) + 120 * (1 - abs(t)))
#         g = int(100 * (1 - abs(t)) + 80)
#         b = int(200 * max(t, 0) + 120 * (1 - abs(t)))
#         return f'rgba({r},{g},{b},{alpha})'

#     for _, r in sub_best.iterrows():
#         src, tgt = r['source'], r['target']
#         if src not in wp.index or tgt not in wp.index:
#             continue
#         y0 = max(wp.loc[src, 'base'], 1e-6)
#         y1 = max(wp.loc[tgt, 'ego'], 1e-6)
#         add(go.Scatter(
#             x=[0, 1], y=[y0, y1], mode='lines',
#             line=dict(color=delta_color(y0, y1), width=1.5, dash='dot'),
#             showlegend=False, opacity=0.7,
#             hovertemplate=(
#                 f'<b>sublimation</b><br>{src} \u2192 {tgt}'
#                 f'<br>sim = {r["sim"]:.3f} (peak layer {int(r["layer"])})'
#                 f'<br>{src}: base={y0:.4f}  |  {tgt}: ego={y1:.4f}'
#                 '<extra></extra>'
#             ),
#         ), src, tgt)

#     for _, r in rep_best.iterrows():
#         src, tgt = r['source'], r['target']
#         if src not in wp.index or tgt not in wp.index:
#             continue
#         y0 = max(wp.loc[src, 'ego'], 1e-6)
#         y1 = max(wp.loc[tgt, 'superego'], 1e-6)
#         add(go.Scatter(
#             x=[1, 2], y=[y0, y1], mode='lines',
#             line=dict(color=delta_color(y0, y1), width=1.5, dash='dot'),
#             showlegend=False, opacity=0.7,
#             hovertemplate=(
#                 f'<b>repression</b><br>{src} \u2192 {tgt}'
#                 f'<br>sim = {r["sim"]:.3f} (peak layer {int(r["layer"])})'
#                 f'<br>{src}: ego={y0:.4f}  |  {tgt}: super={y1:.4f}'
#                 '<extra></extra>'
#             ),
#         ), src, tgt)

#     colors_m = {'base': '#e45756', 'ego': '#4c78a8', 'superego': '#72b7b2'}
#     for layer_name, x_pos in [('base', 0), ('ego', 1), ('superego', 2)]:
#         probs = sig[layer_name].clip(lower=1e-6)
#         tips = [tooltip(w) for w in words]
#         add(go.Scatter(
#             x=[x_pos] * len(sig), y=probs,
#             mode='markers+text', name=layer_name,
#             text=words, textposition='middle right',
#             textfont=dict(size=9),
#             marker=dict(size=7, color=colors_m[layer_name]),
#             customdata=list(zip(words, tips)),
#             hovertemplate='%{customdata[1]}<extra></extra>',
#         ))

#     fig = go.Figure(data=traces)

#     fig.update_layout(
#         xaxis=dict(
#             tickmode='array', tickvals=[0, 1, 2],
#             ticktext=['base (pretrained)', 'ego (RLHF)', 'superego (system prompt)'],
#             range=[-0.4, 2.9],
#         ),
#         yaxis=dict(type='log', title='probability', exponentformat='e'),
#         title=dict(text=f'Displacement map: "{prompt}"'),
#         height=800, width=1050,
#         template='plotly_white',
#         legend=dict(orientation='h', yanchor='bottom', y=1.02),
#         annotations=[
#             dict(x=0.05, y=1.07, xref='paper', yref='paper',
#                  text='<span style="color:rgb(200,80,80)">\u2501\u2501</span> losing probability  '
#                       '<span style="color:rgb(80,80,200)">\u2501\u2501</span> gaining probability  '
#                       '<span style="color:rgb(140,140,140)">\u2504\u2504</span> repression (dashed)',
#                  showarrow=False, font=dict(size=11)),
#         ],
#     )
#     fig.write_image(f"../figures/displacement_{prompt[:100]}.png", scale=2)
#     return fig

# fig = plot_displacement(dm, prompt, min_sim=0.85)
# fig.show()


def plot_layer_displacement(dm, prompt, source_word=None, min_sim=0.4, top_n=8):
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

            # Similarity across layers 1-32
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

            # Mark peak with annotation
            fig.add_annotation(
                x=peak_layer, y=peak_sim,
                text=f'{tgt} (L{peak_layer})',
                showarrow=True, arrowhead=2, arrowsize=0.8,
                ax=20, ay=-20,
                font=dict(size=9, color=color),
                arrowcolor=color,
            )

            # At x=0 (base), show ego probability of target as a reference dot
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

        # Mark base probability of source word at x=0
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
                title='base model \u2192 instruct model hidden layers',
                range=[-0.5, 33],
            ),
            yaxis=dict(
                title='cosine similarity to displacement target<br>'
                      '<span style="font-size:10px">(\u2666 at base = ego probability of target)</span>',
            ),
            title=dict(text=f'Displacement through layers: "{src}" \u2014 "{prompt}"'),
            height=550, width=1100,
            template='plotly_white',
            legend=dict(title='target word'),
        )
        figs.append(fig)
        # Save the current figure as a PNG file
        fig.write_image(f"../figures/displacement_layers_{prompt[:100]}_{src}.png", scale=2)
        fig.show()
        

    return figs

# Single word
# fig = plot_layer_displacement(dm, prompt, source_word='cock', min_sim=0.5)

# Or top sublimated words
# fig = plot_layer_displacement(dm, prompt)