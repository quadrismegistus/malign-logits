#!/usr/bin/env python
"""One figure per lexicon namespace: top-5 riser + top-5 faller fields on one axis.

    uv run python meta/M05_emergence/scripts/m05_field_flow_per_namespace.py

RH (2026-08-11): a graph per namespace (USAS, RID, WN, NORM), top-5 fields
alignment RAISES and top-5 it LOWERS, ten trajectories on one shared axis
(not faceted). Reads data/m05_field_flow_fine.parquet (no re-extraction).
Ranked by base-endpoint -> RLVR movement, base-endpoint present >= 0.003.
Colour = direction (riser/faller); identity = the field label at its own
peak. Window-5 rolling mean for legibility (median over 105 pairs first).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

from plotnine import (aes, annotate, element_blank, element_line,  # noqa: E402
                      element_rect, element_text, geom_line, geom_text,
                      ggplot, labs, scale_color_manual, scale_x_continuous,
                      theme, theme_minimal)

PARQUET = "data/m05_field_flow_fine.parquet"
FIGDIR = "meta/M05_emergence/figures"
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
NS = {"USAS": "USAS (145 semantic categories)",
      "RID": "RID (regressive imagery: drives, sensation, cognition)",
      "WN": "WordNet verb supersenses",
      "NORM": "Warriner/Brysbaert norm bins"}


def main():
    df = pd.read_parquet(PARQUET)
    df["ns"] = df.field.str.split(":").str[0]
    med = (df.groupby(["field", "ns", "ckpt_idx"]).mass.median()
           .reset_index())
    role = df[["ckpt_idx", "role"]].drop_duplicates()
    sft0 = role[role.role == "sft_step"].ckpt_idx.min()
    end = role.ckpt_idx.max()
    base_end = role[role.role == "base_endpoint"].ckpt_idx.iloc[0]
    rlvr = df[df.role == "rlvr_step"].ckpt_idx.max()

    for ns, title in NS.items():
        sub = med[med.ns == ns].copy()
        piv = sub.pivot(index="field", columns="ckpt_idx",
                        values="mass").fillna(0)
        align = (piv[rlvr] - piv[base_end])
        present = piv[base_end] >= 0.003
        a = align[present].sort_values()
        fallers = list(a.head(5).index)
        risers = list(a.tail(5).index)[::-1]
        picks = risers + fallers
        colour = {f: BLUE for f in risers}
        colour.update({f: ORANGE for f in fallers})

        d = sub[sub.field.isin(picks)].sort_values(["field", "ckpt_idx"])
        d["sm"] = (d.groupby("field").mass
                   .transform(lambda x: x.rolling(5, center=True,
                                                  min_periods=1).mean()))
        d["dir"] = np.where(d.field.isin(risers), "riser", "faller")
        pmax = d.sm.max()
        labs_rows = []
        for f in picks:
            dd = d[d.field == f]
            pk = dd.loc[dd.sm.idxmax()]
            lx = min(max(float(pk.ckpt_idx), end * 0.05), end * 0.7)
            labs_rows.append(dict(x=lx, y=float(pk.sm) + pmax * 0.02,
                                  field=f, txt=f.split(": ", 1)[1],
                                  dir="riser" if f in risers else "faller"))
        ld = pd.DataFrame(labs_rows)

        g = (ggplot(d, aes("ckpt_idx", "sm", group="field", color="dir"))
             + annotate("rect", xmin=sft0 - 0.5, xmax=end + 0.5, ymin=-np.inf,
                        ymax=np.inf, fill="#efeee9", alpha=0.55)
             + geom_line(size=0.8)
             + geom_text(aes("x", "y", label="txt", color="dir"), data=ld,
                         size=7.5, ha="center", va="bottom",
                         inherit_aes=False)
             + scale_color_manual({"riser": BLUE, "faller": ORANGE})
             + scale_x_continuous(expand=(0.03, 0, 0.03, 0))
             + labs(title=f"{title}: top 5 alignment risers (blue) and fallers (orange)",
                    subtitle="Field-mass across the ladder, median over 105 pairs, window-5 smoothed.\n"
                             "Ranked by base-endpoint -> RLVR movement. Shaded = post-training.\n"
                             "Label at each field’s peak.",
                    x="training position (base | SFT | DPO | RLVR)",
                    y="continuation mass in field")
             + theme_minimal(base_size=11)
             + theme(panel_grid_minor=element_blank(),
                     panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                     text=element_text(color=INK),
                     plot_title=element_text(size=13, weight="bold"),
                     plot_subtitle=element_text(size=8.5, color=INK2),
                     legend_position="none",
                     plot_background=element_rect(fill="#fcfcfb",
                                                  color="#fcfcfb"),
                     figure_size=(10, 5.5)))
        out = f"{FIGDIR}/fig10_{ns.lower()}_field_flow.png"
        g.save(out, dpi=300, verbose=False)
        print(f"wrote {out}")
        print(f"  {ns} risers:", [f.split(': ',1)[1] for f in risers])
        print(f"  {ns} fallers:", [f.split(': ',1)[1] for f in fallers])
    return 0


if __name__ == "__main__":
    sys.exit(main())
