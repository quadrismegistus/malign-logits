#!/usr/bin/env python
"""Figures for `field_signature_not_contradiction_specific.md`.

    uv run python meta/M02_frame_exit/scripts/l2_fields_figs.py
    uv run python meta/M02_frame_exit/scripts/l2_fields_figs.py dumbbell
    uv run python meta/M02_frame_exit/scripts/l2_fields_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.
Per-finding naming, as `contradiction_null_figs.py`.

THE FIGURE IS THE DOC'S OWN WARNING
-----------------------------------
The finding is a large, consistent effect that is NOT specific to what
it was built to test. 39 of 79 fields survive correction on the general
effect; 0 of 79 survive on the contradiction-specific residual. Reported
as a residual alone, that reads as "nothing happens" -- and something
very large happens. Two panels, same field order: the effect on the
left, its specificity on the right.

THE GAP IN THE LEFT PANEL IS NOT THE RIGHT PANEL, AND THAT IS MEASURED
----------------------------------------------------------------------
The natural reading of a D_CONTRA-over-D_CONTROL dumbbell is that the
distance between the two marks is the contradiction-specific effect. It
is not. `d_both` and `d_ctrl` are each the MEDIAN OVER 26 PAIRS of their
own quantity, while `effect` is the median over pairs of the per-pair
DIFFERENCE -- and a median of differences is not the difference of
medians.

Measured on these artifacts: `effect` differs from `d_both - d_ctrl` in
**79 of 79 fields**, with a median absolute discrepancy of 0.000911,
which is the same order as the effects themselves. So the gap is wrong
as a residual in every row of the panel by an amount comparable to what
it claims to show.

The panels are therefore separate and the subtitle says the gap is not
the residual. Drawing one dumbbell and letting the eye subtract would
have been arithmetically false 79 times out of 79.
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
SOURCES = ("meta", "norms", "usas_fine")

#: Booked in the finding's result table.
BOOKED = {"fields": 79, "general": 39, "specific": 0,
          "per_source": {"meta": (13, 7), "norms": (24, 11),
                         "usas_fine": (42, 21)}}


def _load():
    rows = []
    for src in SOURCES:
        j = json.load(open(os.path.join(RESULTS, f"l2_fields_{src}.json")))
        for r in j["results"]:
            r["source"] = src
            rows.append(r)
    return pd.DataFrame(rows)


def dumbbell():
    """The signature is large and none of it is contradiction-specific."""
    from plotnine import (aes, element_blank, element_text, facet_grid, geom_point,
                          geom_segment, geom_vline, ggplot, labs,
                          scale_color_manual, scale_y_continuous, theme,
                          theme_minimal)

    d = _load()
    assert len(d) == BOOKED["fields"], \
        f"fields drifted: {len(d)} vs booked {BOOKED['fields']}"
    assert int(d.bh_both.sum()) == BOOKED["general"], \
        f"general survivors drifted: {int(d.bh_both.sum())} vs {BOOKED['general']}"
    assert int(d.bh.sum()) == BOOKED["specific"], \
        f"SPECIFIC survivors are no longer zero: {int(d.bh.sum())}"
    for src, (n, surv) in BOOKED["per_source"].items():
        s = d[d.source == src]
        assert (len(s), int(s.bh_both.sum())) == (n, surv), \
            f"{src} drifted: {(len(s), int(s.bh_both.sum()))} vs {(n, surv)}"
    #: the gap-is-not-the-residual claim, asserted rather than asserted in prose
    gap = (d.d_both - d.d_ctrl)
    agree = int(((gap - d.effect).abs() < 1e-9).sum())
    assert agree == 0, \
        (f"{agree} fields now have effect == d_both - d_ctrl; the panel's "
         "separation of the two is built on their never agreeing")

    d = d.sort_values("d_both").reset_index(drop=True)
    d["ypos"] = range(len(d))
    d["surv"] = d.bh_both.map({True: "survives correction",
                               False: "does not survive"})

    left = d.assign(panel="THE EFFECT\nD_CONTRA and D_CONTROL, median over 26 pairs")
    right = d.assign(panel="ITS SPECIFICITY\nmedian per-pair (D_CONTRA - D_CONTROL)")
    order = [left.panel.iloc[0], right.panel.iloc[0]]
    for f in (left, right):
        f["panel"] = pd.Categorical(f.panel, categories=order, ordered=True)

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.4)
        + geom_segment(left, aes("d_ctrl", "ypos", xend="d_both", yend="ypos",
                                 color="surv"), size=0.6, alpha=0.55)
        + geom_point(left, aes("d_ctrl", "ypos"), color="#c98a2b", size=1.5)
        + geom_point(left, aes("d_both", "ypos", color="surv"), size=1.9)
        + geom_point(right, aes("effect", "ypos", color="surv"), size=1.9)
        + scale_color_manual(values={"survives correction": "#1f4e79",
                                     "does not survive": "#c9c9c9"}, name="")
        + scale_y_continuous(breaks=[], labels=[])
        #: SHARED X, NOT free_x. With a free axis the residual panel rescales
        #: to its own +/-0.006 and a near-zero result FILLS the panel, reading
        #: as spread as the effect beside it. The finding is that the residual
        #: is nil against an effect three times larger, so the two panels have
        #: to be measured against one ruler or the comparison is manufactured.
        #: This is the same rule as the propagation panel and I broke it here
        #: first, then read my own caption against the geometry.
        + facet_grid(". ~ panel")
        + labs(
            title="The field signature is large, consistent, and not specific to contradiction",
            #: WRAPPED FOR THE RENDERER, WHICH CUT THIS SUBTITLE WHEN SHIPPED.
            #: The line beginning "THE GAP ON THE LEFT" ran past the canvas and
            #: lost its tail -- "median absolute discrepancy 0.000911, the same
            #: order as the effects" -- which is the number that makes the claim
            #: quantitative. The figure then asserted that the two disagree
            #: without saying by how much, and read as complete. Found by
            #: meta/figure_text_audit.py and confirmed by looking at the PNG.
            subtitle=(
                "79 fields across three lexicon granularities, 26 model pairs, 54,080 English\n"
                "continuations. One row per field, ordered by the general effect.\n"
                "LEFT: D_CONTRA (blue or grey) against D_CONTROL (orange), each the median over 26\n"
                "pairs. 39 of 79 fields survive Benjamini-Hochberg on the general effect.\n"
                "RIGHT: the contradiction-specific residual. ZERO of 79 survive. Every point is a\n"
                "field whose signature is real on the left and not contradiction-specific on the right.\n"
                "THE GAP ON THE LEFT IS NOT THE PANEL ON THE RIGHT. d_both and d_ctrl are medians of\n"
                "their own quantities; the residual is the median of per-pair DIFFERENCES, and a\n"
                "median of differences is not a difference of medians. MEASURED: they disagree in\n"
                "79 of 79 fields, median absolute discrepancy 0.000911, the same order as the effects.\n"
                "BOTH PANELS SHARE ONE AXIS, so the residual's collapse toward zero is measured\n"
                "against the effect beside it rather than rescaled to fill its own panel.\n"
                "This is the finding's own warning drawn: reported as a residual alone, a very large\n"
                "effect would have been filed as a null.\n"
                "Rates are shares of CLASSIFIED words, so a difference in lexicon coverage between\n"
                "arms cannot masquerade as a field difference."),
            x="difference in field rate (aligned minus base)",
            y="",
            caption=("Producer: meta/M02_frame_exit/scripts/l2_fields_figs.py from "
                     "results/l2_fields_{meta,norms,usas_fine}.json (producer "
                     "l2_semantic_fields.py).\n"
                     "Asserted: 79 fields, 39 general survivors, 0 specific survivors, the per-source "
                     "splits 7/13, 11/24 and 21/42, and that effect never equals d_both - d_ctrl."),
        )
        + theme_minimal()
        + theme(figure_size=(12.0, 7.6),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                strip_text=element_text(size=7.8, weight="bold"),
                legend_title=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                legend_position="right",
                panel_spacing=0.05)
    )
    out = os.path.join(FIGURES, "l2_field_signature.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    79 fields, {int(d.bh_both.sum())} general survivors, "
          f"{int(d.bh.sum())} specific survivors; "
          f"effect == d_both - d_ctrl in {agree} of 79")
    return out


FIGURES_REGISTRY = {"dumbbell": dumbbell}


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
