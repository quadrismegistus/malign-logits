#!/usr/bin/env python
"""Findings A Result 4: the contradiction ratio and the separation arc, joined.

    uv run python meta/M05_emergence/scripts/m05_ratio_polesep_figs.py
    uv run python meta/M05_emergence/scripts/m05_ratio_polesep_figs.py joined
    uv run python meta/M05_emergence/scripts/m05_ratio_polesep_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.
Numbered `figNN_` output per this folder's convention, continuing from
fig31.

THIS EXISTS BECAUSE THE PLAN FORBIDS THE UNJOINED VERSION
---------------------------------------------------------
`plans/a_thinksft_acquisition.md`: "The write-up plots ratio and
pole_sep together or not at all." The folder already holds
`fig4_ratio_unjoined.png`, whose own subtitle reads UNREADABLE WITHOUT
POLE_SEP. Until 2026-08-14 the joined version could not be drawn,
because no stated rule turned 166,255 per-layer rows into one number per
checkpoint. lacan re-declared that reduction (plan `570afad4` committed
before the producer, run and republished at `19240d87`), so this is the
first time the plan's clause can be satisfied rather than honoured by
absence.

THE TWO SERIES DO NOT SPAN THE SAME LADDER, AND THE PANEL SAYS SO
-----------------------------------------------------------------
The ratio is measured at all 95 rungs. `pole_sep` is measured at SEVEN,
at ckpt_idx 0, 1, 8, 22, 26, 29 and 38 -- so every separation point
falls in the first 38 rungs of 94, and the last 56 rungs have a ratio
and nothing to compare it to. Drawing two lines of equal visual weight
would imply a co-extensive pair. The separation panel therefore shows
POINTS where it is measured, with a rule marking where its coverage
stops.

THE R4 CO-MOVEMENT STATISTICS ARE NOT ON THIS FIGURE
----------------------------------------------------
`A_acquisition.md` Result 4 reports Spearman +0.61 for levels, co-drift
rho -0.12 for rung-to-rung changes, and lead tests at .085 / .17. **None
of them is produced anywhere**: `co-drift`, `sep-leads` and
`ratio-leads` appear in exactly one file, the finding's own prose, with
zero hits across `*.py`. They are Class 1B and are not quoted here.

What IS quoted is lacan's re-declared and reproducible co-movement of
the two SEPARATION columns, with its own limit stated: positive on both
lineages, significant on neither, n=7 and n=6. The finding's word
"exactly" was corrected in place; this panel does not restore it.

STEP 0 IS NOT A MEASUREMENT, ON BOTH PANELS
-------------------------------------------
At initialisation `pole_sep` is flat across all 32 layers (0.749-0.812,
a 1.1x spread against 18.9x at step 16,000) -- the signature of a random
projection rather than a representation readout. The earlier reading of
step 0 as the largest-separation rung was withdrawn by its author. The
ratio panel has its own reason at the same rung: ckpt_idx 0 carries only
FOUR groups against 21 everywhere else. Both are marked.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
RATIO = os.path.join(ROOT, "data", "m05_ratio.parquet")
SEP = os.path.join(RESULTS, "m05_pole_sep_reduced.csv")

#: OLMo REAL/CROSSGROUP at the seven measured checkpoints, from lacan's
#: re-declaration ([5965]); the superseded six are NOT reproduced here.
BOOKED_SEP = {
    "stage1-step0": (0.8022, 1.3936), "stage1-step1000": (0.2779, 0.6960),
    "stage1-step16000": (0.3931, 1.0458), "stage3-step10000": (0.4689, 1.0078),
}
BOOKED_RATIO_CKPT0 = 1.2675   #: A_acquisition's 1.267, and step 0 is fenced
COMOVE = "olmo Spearman rho +0.536, p 0.215, n=7"


def _load():
    r = pd.read_parquet(RATIO)
    #: READ-SIDE GUARD (registrar, [5969]). Both files in this lineage were
    #: correctly labelled on disk and both had their arm label erased by
    #: pandas' default na_values, which contains the literal strings "null"
    #: and "NULL". The producer-side guard lacan added cannot protect files
    #: that already exist, so the reader asserts too.
    s = pd.read_csv(SEP, keep_default_na=False)
    assert s.column.value_counts().to_dict() == {"REAL": 13, "CROSSGROUP": 13}, \
        f"arm labels unreadable or unbalanced: {s.column.value_counts().to_dict()}"
    return r, s[s.ladder == "olmo"].copy()


def joined():
    """A-R4: the ratio and the separation arc on one ladder.

    Ratio at all 95 rungs above; separation at the seven checkpoints
    where it exists below, both arms, on the same x.
    """
    from plotnine import (aes, element_text, geom_line, geom_point,
                          geom_rect, geom_text, geom_vline, ggplot, labs,
                          facet_grid, scale_color_manual, theme, theme_minimal)

    r, s = _load()

    g = (r.groupby("ckpt_idx")
         .agg(ratio=("ratio", "median"), n=("ratio", "count"),
              stage=("stage", "first"), role=("role", "first"))
         .reset_index())
    assert len(g) == 95, f"ladder drifted: {len(g)} rungs vs 95"
    assert round(float(g.loc[g.ckpt_idx == 0, "ratio"].iloc[0]), 4) == BOOKED_RATIO_CKPT0, \
        "ckpt 0 ratio drifted from the value A_acquisition quotes"
    assert int(g.loc[g.ckpt_idx == 0, "n"].iloc[0]) == 4, \
        "ckpt 0 no longer carries only 4 groups; the fence's reason has changed"

    idx = r[["model", "ckpt_idx"]].drop_duplicates("model")
    s = s.merge(idx, left_on="checkpoint", right_on="model", how="left")
    assert s.ckpt_idx.notna().all(), "a separation checkpoint is not on the ratio ladder"
    s["rev"] = s.checkpoint.str.split("@").str[-1]
    for rev, (real, cross) in BOOKED_SEP.items():
        got = {c: round(float(s[(s.rev == rev) & (s.column == c)].value.iloc[0]), 4)
               for c in ("REAL", "CROSSGROUP")}
        assert got == {"REAL": real, "CROSSGROUP": cross}, \
            f"{rev} drifted: {got} vs booked REAL {real} CROSSGROUP {cross}"

    last_sep = float(s.ckpt_idx.max())
    fence = pd.DataFrame([{"xmin": -1.6, "xmax": 0.6, "ymin": -np.inf,
                           "ymax": np.inf}])

    g["panel"] = "contradiction ratio  (all 95 rungs)"
    s["panel"] = "separation  (the 7 rungs where it is measured)"
    order = [g.panel.iloc[0], s.panel.iloc[0]]
    for f in (g, s):
        f["panel"] = pd.Categorical(f.panel, categories=order, ordered=True)

    #: stage boundaries, from the first rung of each stage
    bnd = (g.dropna(subset=["stage"]).groupby("stage").ckpt_idx.min()
           .reset_index().rename(columns={"ckpt_idx": "x"}))
    bnd = bnd[bnd.x > 0]

    p = (
        ggplot()
        + geom_vline(bnd, aes(xintercept="x"), color="#c9c9c9", size=0.5)
        + geom_text(bnd.assign(panel=pd.Categorical([order[0]] * len(bnd),
                                                    categories=order, ordered=True)),
                    aes("x", 1.24, label="stage"), size=6.2, ha="left",
                    va="top", color="#8a8a8a", nudge_x=0.7)
        + geom_vline(xintercept=last_sep, color="#8a6d1f", linetype="dotted",
                     size=0.6)
        #: ckpt 0, fenced on both panels for different reasons
        + geom_rect(fence, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    fill="#999999", alpha=0.20)
        + geom_line(g, aes("ckpt_idx", "ratio"), color="#1f4e79", size=0.7)
        + geom_point(g, aes("ckpt_idx", "ratio"), color="#1f4e79", size=0.9,
                     alpha=0.7)
        + geom_line(s, aes("ckpt_idx", "value", color="column"), size=0.7,
                    alpha=0.75)
        + geom_point(s, aes("ckpt_idx", "value", color="column"), size=2.6)
        + scale_color_manual(values={"REAL": "#b03030", "CROSSGROUP": "#c98a2b"},
                             name="")
        #: TWO PANELS WITH FREE Y. A single axis was the first draft and it
        #: put a calibrated ratio and a separation distance on one unlabelled
        #: scale purely because their ranges happen to overlap. They are not
        #: the same quantity and a shared y would assert that they are. The
        #: x IS shared, which is the whole point of the plan's clause.
        + facet_grid("panel ~ .", scales="free_y")
        + labs(
            title="Plotted together, as the plan requires: the ratio across the ladder, and the arc that is not about poles",
            subtitle=(
                "OLMo ladder, x is rung index. ABOVE: median calibrated contradiction ratio over 21 "
                "groups, all 95 rungs. BELOW: the separation reduction at the SEVEN checkpoints where "
                "it is measured, both arms.\n"
                "THE TWO SERIES DO NOT SPAN THE SAME LADDER. Every separation point falls within the "
                "first 38 rungs of 94; past the dotted line the ratio has nothing to be compared "
                "against.\n"
                "THE ARC IS NOT ABOUT POLES. CROSSGROUP pairs prompts that are merely different rather "
                "than opposed, and it collapses and recovers alongside REAL, so the arc is what "
                "training does to the distance between any two distinct prompts.\n"
                f"CO-MOVEMENT OF THE TWO ARMS: {COMOVE} -- positive and NOT significant. The finding's "
                "word \"exactly\" was corrected in place and is not restored here; n=7 cannot establish "
                "co-movement, only fail to contradict it.\n"
                "THE R4 RATIO-versus-SEPARATION STATISTICS ARE NOT SHOWN. Spearman +0.61,\n"
                "co-drift -0.12 and the lead tests have no producer anywhere in the repo and are\n"
                "booked Class 1B; this panel plots the two series and asserts nothing about their coupling.\n"
                "GREY BAND: rung 0 is not a measurement. Separation there is flat across all 32 layers "
                "(a random-projection signature) and the ratio there rests on 4 groups against 21 "
                "everywhere else."),
            x="rung index on the OLMo ladder",
            y="",
            caption=("Producer: meta/M05_emergence/scripts/m05_ratio_polesep_figs.py from "
                     "data/m05_ratio.parquet + results/m05_pole_sep_reduced.csv "
                     "(reduction re-declared by lacan, plan 570afad4 before producer, run 19240d87).\n"
                     "The separation file is read with keep_default_na=False and its arm counts "
                     "asserted: both files in this lineage were correctly labelled on disk and had "
                     "their labels erased by pandas' default na_values.\n"
                     "Superseded separation values are NOT reproduced here; the re-declared ones are "
                     "asserted at four checkpoints."),
        )
        + theme_minimal()
        + theme(figure_size=(12.4, 7.0),
                plot_title=element_text(size=12, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                strip_text=element_text(size=8.0, weight="bold"),
                panel_spacing=0.06,
                legend_position="right")
    )
    out = os.path.join(FIGURES, "fig32_ratio_polesep_joined.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    ratio 95 rungs; separation at {int(s.ckpt_idx.nunique())} "
          f"(idx {sorted(set(s.ckpt_idx.astype(int)))}), last at {int(last_sep)} of 94")
    return out


FIGURES_REGISTRY = {"joined": joined}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in FIGURES_REGISTRY.items():
            print(f"  {k:10s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    names = a.names or list(FIGURES_REGISTRY)
    unknown = [n for n in names if n not in FIGURES_REGISTRY]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    os.makedirs(FIGURES, exist_ok=True)
    for n in names:
        print(f"{n}:")
        FIGURES_REGISTRY[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
