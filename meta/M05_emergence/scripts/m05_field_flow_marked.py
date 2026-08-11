#!/usr/bin/env python
"""Field flow split by transgressive/neutral: facet by field, free y, two members.

    uv run python meta/M05_emergence/scripts/m05_field_flow_marked.py

RH (2026-08-11): take the fields alignment moves most and split each into its
transgressive (MARKED) and neutral (UNMARKED) trajectory, faceted by field
with each facet on its own y range. Reads data/m05_field_flow_fine.parquet
(now member-carrying). One figure per namespace + one overall top-12.
Window-5 smoothed; median over the 105 pairs within each member.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

from plotnine import (aes, annotate, element_blank, element_line,  # noqa: E402
                      element_rect, element_text, facet_wrap, geom_line,
                      ggplot, labs, scale_color_manual, theme, theme_minimal)

PARQUET = "data/m05_field_flow_fine.parquet"
FIGDIR = "meta/M05_emergence/figures"
ORANGE, BLUE, INK, INK2 = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e"


def figure(df, fields_in_order, out, title):
    d = df[df.field.isin(fields_in_order)].copy()
    med = (d.groupby(["field", "member", "ckpt_idx"]).mass.median()
           .reset_index().sort_values(["field", "member", "ckpt_idx"]))
    med["sm"] = (med.groupby(["field", "member"]).mass
                 .transform(lambda x: x.rolling(5, center=True,
                                                min_periods=1).mean()))
    # short facet label = field without the namespace prefix, ordered by rank
    lab = {f: f.split(": ", 1)[1] if ": " in f else f for f in fields_in_order}
    med["facet"] = pd.Categorical(med.field.map(lab),
                                  categories=[lab[f] for f in fields_in_order],
                                  ordered=True)
    role = df[["ckpt_idx", "role"]].drop_duplicates()
    sft0 = role[role.role == "sft_step"].ckpt_idx.min()
    endx = role.ckpt_idx.max()
    ncol = 3
    nrow = int(np.ceil(len(fields_in_order) / ncol))
    g = (ggplot(med, aes("ckpt_idx", "sm", color="member"))
         + annotate("rect", xmin=sft0 - 0.5, xmax=endx + 0.5, ymin=-np.inf,
                    ymax=np.inf, fill="#efeee9", alpha=0.55)
         + geom_line(size=0.8)
         + facet_wrap("~facet", ncol=ncol, scales="free_y")
         + scale_color_manual({"MARKED": ORANGE, "UNMARKED": BLUE})
         + labs(title=title,
                subtitle="Orange = transgressive (MARKED), blue = neutral (UNMARKED) twin. Median over the pairs, "
                         "window-5, each facet free-scaled. Shaded = post-training.",
                x="training position (base | SFT | DPO | RLVR)",
                y="continuation mass in field")
         + theme_minimal(base_size=10)
         + theme(panel_grid_minor=element_blank(),
                 panel_grid_major=element_line(color="#e8e7e3", size=0.3),
                 text=element_text(color=INK),
                 plot_title=element_text(size=13, weight="bold"),
                 plot_subtitle=element_text(size=8.5, color=INK2),
                 strip_text=element_text(size=8),
                 legend_position="none",
                 plot_background=element_rect(fill="#fcfcfb", color="#fcfcfb"),
                 figure_size=(11, 2.4 * nrow)))
    g.save(out, dpi=300, verbose=False)
    print(f"wrote {out} ({len(fields_in_order)} facets)")


def rank(df, ns=None):
    sub = df if ns is None else df[df.field.str.startswith(ns + ":")]
    med = (sub.groupby(["field", "ckpt_idx"]).mass.median().reset_index()
           .pivot(index="field", columns="ckpt_idx", values="mass").fillna(0))
    base_end = df[df.role == "base_endpoint"].ckpt_idx.iloc[0]
    rlvr = df[df.role == "rlvr_step"].ckpt_idx.max()
    align = (med[rlvr] - med[base_end])
    present = med[base_end] >= 0.003
    a = align[present]
    return list(a.sort_values(ascending=False).head(6).index) + \
        list(a.sort_values().head(6).index)


def main():
    df = pd.read_parquet(PARQUET)
    # overall top-12 (6 risers + 6 fallers), one faceted figure
    figure(df, rank(df),
           f"{FIGDIR}/fig11_marked_vs_neutral_top12.png",
           "Transgressive vs neutral, per field: the 12 alignment movers")
    # and one per namespace
    for ns, nice in [("NORM", "norm bins"), ("RID", "RID categories"),
                     ("WN", "WordNet supersenses"), ("USAS", "USAS categories")]:
        figure(df, rank(df, ns),
               f"{FIGDIR}/fig11_marked_{ns.lower()}.png",
               f"Transgressive vs neutral by field — {nice}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
