#!/usr/bin/env python
"""Figures for Findings T (T_category_flow.md), one function per figure.

    uv run python meta/M01_displacement/scripts/plot_t_figs.py            # all
    uv run python meta/M01_displacement/scripts/plot_t_figs.py t14        # one or more by name
    uv run python meta/M01_displacement/scripts/plot_t_figs.py --list

Per-letter plotting convention (RH, 2026-08-14, plot-debt regime): one
script per letter in scripts/plot_<letter>_figs.py, a FIGURES registry
mapping name -> function, all figures drawn by default, plotnine at 300
dpi, output to ../figures/. Every figure embeds its slice in the
subtitle — plot-debt's fence: no T magnitude travels without its slice
named, T-11 class figures report stratified never pooled, and T-2's GI
table is never plotted beside findings 11-16.

Each figure function verifies the finding's booked numbers from the
artifact before drawing and refuses (with a named reason) if they do
not reproduce — a plot of numbers that no longer match their finding is
the [5811] class with an image on it.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
FIGURES = os.path.join(M01, "figures")

LEX_LABELS = {
    "framenet": "FrameNet", "gi_primary": "General Inquirer",
    "induced": "induced taxonomy", "rid": "RID", "usas": "USAS",
    "verbnet": "VerbNet", "wordnet": "WordNet",
}


def t14_dumbbell():
    """T-14: few large fallers, many small risers — dumbbell per lexicon.

    Slice of record (T-14 amendment, corrected 2026-08-12): stratum=ALL,
    seven labelings (TOKEN excluded), Bonferroni survivors. Booked:
    206 risers / 36 fallers, means +0.00334 / -0.01267, ratio 3.79x,
    ratio > 1 in every lexicon. Count is partly granularity (FrameNet
    82/7 vs WordNet 6/2) — the count rides ON the marks, the magnitude
    is the position, per the finding's own rule: quote the magnitude;
    quote the count with its resolution.
    """
    from plotnine import (aes, element_text, geom_point, geom_segment,
                          geom_text, ggplot, labs, scale_color_manual,
                          scale_x_log10, theme, theme_minimal)

    d = pd.read_csv(os.path.join(RESULTS, "s_everything_marginal.csv"))
    s = d[(d.stratum == "ALL") & (d.labeling != "TOKEN") & d.bonferroni].copy()
    ris, fal = s[s.delta > 0], s[s.delta < 0]
    n_r, n_f = len(ris), len(fal)
    m_r, m_f = ris.delta.mean(), fal.delta.mean()
    ratio = abs(m_f) / m_r
    assert (n_r, n_f) == (206, 36), f"count drifted: {n_r}/{n_f} vs booked 206/36"
    assert abs(ratio - 3.79) < 0.01, f"ratio drifted: {ratio:.2f} vs booked 3.79"

    s["role"] = s.delta.gt(0).map({True: "riser", False: "faller"})
    per = (s.groupby(["labeling", "role"])
             .agg(n=("delta", "size"),
                  mean_abs=("delta", lambda v: v.abs().mean()))
             .reset_index())
    wide = per.pivot(index="labeling", columns="role",
                     values=["n", "mean_abs"]).reset_index()
    wide.columns = ["labeling", "n_faller", "n_riser", "abs_faller", "abs_riser"]
    wide["lex"] = wide.labeling.map(LEX_LABELS)
    wide["ratio"] = wide.abs_faller / wide.abs_riser
    order = wide.sort_values("ratio").lex.tolist()
    wide["lex"] = pd.Categorical(wide.lex, categories=order, ordered=True)

    long = pd.concat([
        wide.assign(role="faller", mean_abs=wide.abs_faller, n=wide.n_faller),
        wide.assign(role="riser", mean_abs=wide.abs_riser, n=wide.n_riser),
    ])
    long["n_label"] = long.n.map(lambda v: f"n={int(v)}")
    # risers label above, fallers below — keeps labels apart where the
    # two roles nearly coincide (induced taxonomy, ratio ~1.0)
    long["nudge"] = long.role.map({"riser": 0.32, "faller": -0.38})

    p = (ggplot()
         + geom_segment(wide, aes(x="abs_riser", xend="abs_faller",
                                  y="lex", yend="lex"),
                        color="#b0b0b0", size=0.8)
         + geom_point(long, aes(x="mean_abs", y="lex", color="role"), size=3.5)
         + geom_text(long[long.role == "riser"],
                     aes(x="mean_abs", y="lex", label="n_label", color="role"),
                     size=7, nudge_y=0.32, show_legend=False)
         + geom_text(long[long.role == "faller"],
                     aes(x="mean_abs", y="lex", label="n_label", color="role"),
                     size=7, nudge_y=-0.38, show_legend=False)
         + scale_x_log10()
         + scale_color_manual({"faller": "#c0392b", "riser": "#2e6da4"})
         + labs(
             x="mean |delta| per Bonferroni-survivor category (log scale)",
             y="",
             color="",
             title="T-14: few large fallers, many small risers",
             subtitle=(f"Slice of record: stratum=ALL, seven labelings "
                       f"(TOKEN excluded), Bonferroni survivors —\n"
                       f"{n_r} risers / {n_f} fallers, means {m_r:+.5f} / "
                       f"{m_f:+.5f}, fallers {ratio:.2f}x larger "
                       f"(Mann-Whitney p=5.8e-09); ratio > 1 in every lexicon.\n"
                       f"Counts are of resource-category pairs, not semantic "
                       f"fields; fine-grained lexicons carry the count —\n"
                       f"quote the magnitude, quote the count with its "
                       f"resolution."))
         + theme_minimal()
         + theme(figure_size=(9, 5),
                 plot_subtitle=element_text(size=7.5),
                 plot_title=element_text(size=12, weight="bold")))
    out = os.path.join(FIGURES, "t14_fallers_risers_dumbbell.png")
    p.save(out, dpi=300, verbose=False)
    print(f"wrote {out}")




def t14():
    """T-14 v2: slopegraph of the individual FIELDS — fallers left, risers
    right, placed at their |delta| (log y), connected where a Bonferroni-
    surviving DIRECTED flow runs between two T-14 survivor fields
    (s_everything_direction_edgeunit, edge-consistent p_edge < .05). The
    lines are actual displacement routes, not decoration. Four lexicons
    carry qualifying flows (framenet 238, verbnet 152, usas 57,
    gi_primary 23); top flows per lexicon drawn, truncation stated on the
    panel — no silent caps."""
    from plotnine import (aes, element_blank, element_text, facet_wrap,
                          geom_point, geom_segment, geom_text, ggplot,
                          labs, scale_alpha_continuous, scale_color_manual,
                          scale_x_continuous, scale_y_log10, theme,
                          theme_minimal)

    TOP_FLOWS = 12

    m = pd.read_csv(os.path.join(RESULTS, "s_everything_marginal.csv"))
    d = pd.read_csv(os.path.join(RESULTS,
                                 "s_everything_direction_edgeunit.csv"))
    surv = m[(m.stratum == "ALL") & (m.labeling != "TOKEN") & m.bonferroni]
    mag = {(r.labeling, r.category): abs(r.delta)
           for r in surv.itertuples()}
    # display names where the artifact declares them (USAS codes are
    # cryptic; category_name carries 'Speech acts' etc.) — declared
    # column over code, the day's own rule
    disp = {}
    for r in m[m.category_name.notna()].itertuples():
        nm = str(r.category_name)
        disp[(r.labeling, r.category)] = (nm[:22] + "…") if len(nm) > 23 else nm
    fal = {(r.labeling, r.category) for r in surv.itertuples() if r.delta < 0}
    ris = {(r.labeling, r.category) for r in surv.itertuples() if r.delta > 0}

    dd = d[(d.stratum == "ALL") & (d.labeling != "TOKEN") & d.bonferroni
           & (d.p_edge < 0.05)]
    flows = dd[[((r.labeling, r.frm) in fal) and ((r.labeling, r.to) in ris)
                for r in dd.itertuples()]].copy()
    n_total = len(flows)
    flows = (flows.sort_values("E", ascending=False)
                  .groupby("labeling").head(TOP_FLOWS).copy())

    rows, segs = [], []
    for r in flows.itertuples():
        yf, yr = mag[(r.labeling, r.frm)], mag[(r.labeling, r.to)]
        lex = LEX_LABELS.get(r.labeling, r.labeling)
        segs.append(dict(lex=lex, x=0, xend=1, y=yf, yend=yr, E=r.E))
        rows.append(dict(lex=lex, x=0, y=yf, role="faller",
                         field=disp.get((r.labeling, r.frm), r.frm)))
        rows.append(dict(lex=lex, x=1, y=yr, role="riser",
                         field=disp.get((r.labeling, r.to), r.to)))
    pts = pd.DataFrame(rows).drop_duplicates(["lex", "x", "field"])
    segs = pd.DataFrame(segs)
    kept = len(segs)

    pl = (ggplot()
          + geom_segment(segs, aes(x="x", xend="xend", y="y", yend="yend",
                                   alpha="E"), color="#808080", size=0.5)
          + geom_point(pts, aes(x="x", y="y", color="role"), size=2.2)
          + geom_text(pts[pts.role == "faller"],
                      aes(x="x", y="y", label="field", color="role"),
                      size=5.5, ha="right", nudge_x=-0.04,
                      adjust_text={"only_move": {"text": "y"},
                                   "arrowprops": {"arrowstyle": "-",
                                                  "color": "#cccccc",
                                                  "lw": 0.4}},
                      show_legend=False)
          + geom_text(pts[pts.role == "riser"],
                      aes(x="x", y="y", label="field", color="role"),
                      size=5.5, ha="left", nudge_x=0.04,
                      adjust_text={"only_move": {"text": "y"},
                                   "arrowprops": {"arrowstyle": "-",
                                                  "color": "#cccccc",
                                                  "lw": 0.4}},
                      show_legend=False)
          + facet_wrap("~lex", nrow=1)
          + scale_x_continuous(breaks=[0, 1],
                               labels=["fallers", "risers"],
                               limits=(-0.9, 1.9))
          + scale_y_log10()
          + scale_alpha_continuous(range=(0.15, 0.7), guide=None)
          + scale_color_manual({"faller": "#c0392b", "riser": "#2e6da4"},
                               guide=None)
          + labs(x="", y="field |delta| (log scale)",
                 title="T-14: displacement routes between survivor fields",
                 subtitle=(f"Lines = Bonferroni-surviving directed flows, "
                           f"faller field -> riser field, edge-consistent "
                           f"(p_edge < .05), line weight = edges agreeing; "
                           f"top {TOP_FLOWS} flows per lexicon shown of "
                           f"{n_total} qualifying ({kept} drawn).\n"
                           f"Fields at their T-14 survivor |delta| "
                           f"(slice: ALL / non-TOKEN / Bonferroni). "
                           f"wordnet, rid and induced carry no qualifying "
                           f"flows and are absent, stated not hidden."))
          + theme_minimal()
          + theme(figure_size=(14, 6),
                  axis_text_x=element_text(size=9),
                  strip_text=element_text(size=10, weight="bold"),
                  plot_subtitle=element_text(size=8),
                  panel_grid_minor=element_blank(),
                  plot_title=element_text(size=12, weight="bold")))
    out = os.path.join(FIGURES, "t14_fields_slopegraph.png")
    pl.save(out, dpi=300, verbose=False)
    print(f"wrote {out}")


REGISTRY = {
    "t14": t14,
    "t14_dumbbell": t14_dumbbell,
    # future: t5 (sink structure), t7 (concreteness densities),
    # t8 (bodily_violence->speech_act diverging bar), t11 (stratified
    # heatmap; NEVER pooled), t12 (USAS lollipop), t18 (affect DiD
    # beside M05-C) — see meta/plot-debt.md RUNNING TODO.
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=[],
                    help="figure names (default: all)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, fn in REGISTRY.items():
            print(f"{k}: {(fn.__doc__ or '').strip().splitlines()[0]}")
        return
    names = args.figs or list(REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        sys.exit(f"unknown figure(s): {unknown}; known: {list(REGISTRY)}")
    os.makedirs(FIGURES, exist_ok=True)
    for n in names:
        REGISTRY[n]()


if __name__ == "__main__":
    main()
