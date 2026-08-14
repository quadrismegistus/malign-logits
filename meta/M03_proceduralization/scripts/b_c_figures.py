#!/usr/bin/env python
"""Figures for Findings B_C (B_C_arm_and_reference_class.md).

    uv run python meta/M03_proceduralization/scripts/b_c_figures.py
    uv run python meta/M03_proceduralization/scripts/b_c_figures.py lineages
    uv run python meta/M03_proceduralization/scripts/b_c_figures.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.
Descriptive naming per this folder (`f_figures.py`, `e_figures.py`).

THE SIGN IS THE FINDING, SO THE PRIOR GOES ON THE AXIS
------------------------------------------------------
Section 1 is titled "The arm effect, and it runs the other way", and its
second line is **"Negative is F21's stated direction."** The result is
positive at 41 of 46 lineages. So the quantity here is not interesting
for its size; it is interesting for being on the wrong side of a
prediction, and a panel that shows only the distribution would render
the number and drop the finding.

F21's predicted direction is therefore drawn as a labelled region of the
axis rather than described in a caption. A reader who knows nothing
about F21 can still see that 41 of 46 lineages fall outside the half
where the prior said they would land.

CONSISTENCY AND EFFECT ARE DIFFERENT AXES AND BOTH ARE IN THE FILE
------------------------------------------------------------------
`median_d_js` is the lineage's effect; `share_cells_positive` is how
consistently its 126 cells agree with that effect. A lineage at +0.02
with 71% of cells positive is a different object from one at +0.02 with
52%, and the file carries both. Position is the effect, colour is the
consistency, diverging at 0.5 so a lineage that is barely a majority
reads as pale whatever its median.

ONE OF THE FIVE DISSENTERS IS NOT A DISSENTER
---------------------------------------------
The finding says five lineages sit below zero and the shortlist asks for
them labelled. Four of them are: RedPajama at -0.0184 through
falcon-mamba at -0.0074. **Mistral-7B-v0.1 is at -0.000138 with
`share_cells_positive` of exactly 0.5000** -- 63 of its 126 cells each
way. Labelling it beside RedPajama would present a tie as a
disagreement, so it is labelled as the tie it is. The count of five is
the finding's and is not changed here.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
SRC = os.path.join(RESULTS, "b_arm_by_lineage.csv")

#: Booked in B_C_arm_and_reference_class.md section 1.
BOOKED = {"lineages": 46, "cells": 5796, "per_lineage": 126,
          "median": 0.01187, "above_zero": 41}


def lineages():
    """B_C 1: the arm effect, one row per lineage, and it runs the other way."""
    from plotnine import (aes, element_text, geom_point, geom_rect,
                          geom_text, geom_vline, ggplot, labs,
                          scale_color_gradient2, scale_y_continuous, theme,
                          theme_minimal)

    d = pd.read_csv(SRC)
    assert len(d) == BOOKED["lineages"], \
        f"lineages drifted: {len(d)} vs booked {BOOKED['lineages']}"
    assert set(d.n_cells.unique()) == {BOOKED["per_lineage"]}, \
        f"cells per lineage are no longer uniform: {sorted(d.n_cells.unique())}"
    assert int(d.n_cells.sum()) == BOOKED["cells"], \
        f"total cells drifted: {int(d.n_cells.sum())} vs {BOOKED['cells']}"
    assert round(float(d.median_d_js.median()), 5) == BOOKED["median"], \
        f"median drifted: {round(float(d.median_d_js.median()), 5)} vs {BOOKED['median']}"
    assert int((d.median_d_js > 0).sum()) == BOOKED["above_zero"], \
        f"above-zero count drifted: {int((d.median_d_js > 0).sum())} vs {BOOKED['above_zero']}"

    d = d.sort_values("median_d_js").reset_index(drop=True)
    d["short"] = d.lineage.str.split("/").str[-1]
    #: NUMERIC y with labelled breaks, not a categorical. The shaded prior
    #: region and the axis annotation are rectangles and points in data
    #: space, and mixing those with a discrete scale raises "Discrete value
    #: supplied to continuous scale". Same fix as the propagation panel.
    d["ypos"] = range(len(d))

    below = d[d.median_d_js < 0].copy()
    tie = below[below.median_d_js.abs() < 1e-3]
    real = below[below.median_d_js.abs() >= 1e-3]
    below = below.copy()
    below["lab"] = [
        f"{s}   {v:+.4f}" + ("   a tie, not a dissent: 63/126 each way"
                             if abs(v) < 1e-3 else "")
        for s, v in zip(below.short, below.median_d_js)]

    lo, hi = float(d.median_d_js.min()), float(d.median_d_js.max())
    pad = (hi - lo) * 0.34
    band = pd.DataFrame([{"xmin": lo - pad, "xmax": 0.0,
                          "ymin": -0.8, "ymax": len(d) - 0.2}])

    p = (
        ggplot()
        + geom_rect(band, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    fill="#b03030", alpha=0.07)
        + geom_vline(xintercept=0, color="#333333", size=0.45)
        + geom_point(d, aes("median_d_js", "ypos", color="share_cells_positive"),
                     size=2.8)
        + geom_text(below, aes("median_d_js", "ypos", label="lab"), size=6.0,
                    ha="right", nudge_x=-0.0012, color="#b03030")
        + geom_text(pd.DataFrame([{"x": lo - pad * 0.5, "y": len(d) - 2.0}]),
                    aes("x", "y"), size=6.6, ha="center", color="#b03030",
                    label="F21's STATED DIRECTION")
        + scale_y_continuous(breaks=list(d.ypos), labels=list(d.short),
                             limits=(-0.8, len(d) - 0.2))
        + scale_color_gradient2(low="#b03030", mid="#eeeeee", high="#1f4e79",
                                midpoint=0.5, name="share of the\nlineage's 126\ncells positive")
        + labs(
            title="The arm effect runs the other way: 41 of 46 lineages land outside F21's predicted half",
            subtitle=(
                "d = JS(institutional) - JS(individual), one row per lineage, 126 paired cells each, "
                "5,796 in total. NEGATIVE IS F21's STATED DIRECTION and is shaded.\n"
                "41 of 46 lineages are positive, median +0.01187, p 4.4e-08: the INSTITUTIONAL side "
                "moves further under alignment, which is the opposite of the prior this was built to "
                "replicate.\n"
                "COLOUR IS CONSISTENCY, NOT EFFECT. A lineage's position is its median; its colour is "
                "how many of its own 126 cells agree, diverging at 0.5, so a bare majority reads pale "
                "however large the median.\n"
                "FOUR OF THE FIVE BELOW ZERO ARE DISSENTERS. Mistral-7B-v0.1 is at -0.000138 with "
                "exactly 63 of 126 cells each way: a tie, labelled as one. The count of five is the "
                "finding's and is unchanged.\n"
                "In this contrast the scenario, the person, the modal position and the modal type are "
                "identical across arms; the arm is the manipulation. Single pass, M03's module title "
                "separately CONTESTED."),
            x="d = JS(institutional) - JS(individual), lineage median",
            y="",
            caption=("Producer: meta/M03_proceduralization/scripts/b_c_figures.py from "
                     "results/b_arm_by_lineage.csv.\n"
                     "Asserted before drawing: 46 lineages, 126 cells each and 5,796 in total, median "
                     "+0.01187, and 41 above zero."),
        )
        + theme_minimal()
        + theme(figure_size=(11.0, 8.8),
                plot_title=element_text(size=12, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.1, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_text(size=6.8),
                legend_position="right")
    )
    out = os.path.join(FIGURES, "b_c_arm_by_lineage.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    46 lineages, {int((d.median_d_js > 0).sum())} above zero, "
          f"{len(real)} true dissenters + {len(tie)} tie "
          f"({', '.join(tie.short.astype(str))})")
    return out


FIGURES_REGISTRY = {"lineages": lineages}


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
