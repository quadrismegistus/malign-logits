#!/usr/bin/env python
"""Pole separation against superposition loss, across 45 lineages.

    uv run python meta/M02_frame_exit/scripts/poles_superposition_figs.py
    uv run python meta/M02_frame_exit/scripts/poles_superposition_figs.py --list

plot-debt M02 candidate 4. plotnine at 300 dpi, output to ../figures/, booked
numbers asserted before drawing. Case 1 by shape: reads a committed CSV and
writes only pixels.

NO REGRESSION LINE IS DRAWN, AND THAT IS THE WHOLE DESIGN
----------------------------------------------------------
`pole_axis_t_is_not_superposition.md` states the confound in bold: both
quantities may simply track HOW MUCH ALIGNMENT HAPPENED, "a heavily aligned
model would show more pole separation and more superposition loss with neither
causing the other. The correlation is real; the arrow is not established."

**An ordinary least-squares line of y on x asserts precisely the asymmetry the
finding declines.** It minimises error in y given x, which is the graphical form
of "x predicts y", and no caption placed under it survives contact with the
picture. So the summary drawn here is the PRINCIPAL AXIS -- symmetric in the two
variables, the same line whichever is called the predictor -- and it is labelled
as such on the panel.

The section heading in the finding does read "pole separation PREDICTS the loss
of superposition", and the body then withdraws the arrow four paragraphs later.
The panel follows the body.

WHY THE CORRELATION IS WORTH ANYTHING AT ALL
----------------------------------------------
The two axes come from independent substrates: `pole_sep` from the L3 hidden
states, the superposition signal from the twp output ratio. This is agreement
across two instruments rather than one instrument agreeing with itself, which is
the same argument the M01 headroom ladder makes for GloVe against bge.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
SRC = os.path.join(RESULTS, "polesep_vs_superposition.csv")

BOOKED = {"n": 45, "spearman": -0.420, "sp_p": 0.0041,
          "pearson": -0.447, "pe_p": 0.0021}
DOT_C, AXIS_C = "#1f4e79", "#b03030"


def polesep_superposition():
    """M02 candidate 4: two instruments agree, and the arrow is not established."""
    from scipy.stats import pearsonr, spearmanr
    from plotnine import (aes, element_text, geom_hline, geom_point,
                          geom_segment, geom_text, geom_vline, ggplot, labs,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    d = pd.read_csv(SRC)
    assert len(d) == BOOKED["n"], f"{len(d)} lineages, not {BOOKED['n']}"
    assert d.lineage.nunique() == BOOKED["n"], "a lineage appears twice"

    rs, ps = spearmanr(d.d_polesep, d.delta)
    rp, pp = pearsonr(d.d_polesep, d.delta)
    assert abs(rs - BOOKED["spearman"]) < 5e-4, f"Spearman {rs:+.4f}"
    assert abs(ps - BOOKED["sp_p"]) < 5e-5, f"Spearman p {ps:.5f}"
    assert abs(rp - BOOKED["pearson"]) < 5e-4, f"Pearson {rp:+.4f}"
    assert abs(pp - BOOKED["pe_p"]) < 5e-5, f"Pearson p {pp:.5f}"

    #: THE PRINCIPAL AXIS, symmetric in x and y. Standardise, take the leading
    #: eigenvector of the correlation matrix, map back. Computed rather than
    #: fitted, so there is no predictor and no residual direction.
    x, y = d.d_polesep.values, d.delta.values
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    w, v = np.linalg.eigh(np.cov(np.vstack([(x - mx) / sx, (y - my) / sy])))
    ex, ey = v[:, np.argmax(w)]
    #: the symmetric slope in raw units; sign follows the correlation
    slope = (ey / ex) * (sy / sx)
    assert np.sign(slope) == np.sign(rp), \
        "the principal axis and the correlation disagree in sign"

    #: THE LINE MUST BE CLIPPED TO THE PANEL OR IT IS NOT DRAWN AT ALL. Drawn
    #: across the full x span this axis leaves the y limits, and plotnine
    #: silently removes the whole segment -- "geom_segment : Removed 1 rows".
    #: The figure then renders complete, with every number correct and the one
    #: line its argument depends on simply absent. Endpoints are solved against
    #: both limits and the survival of the segment is asserted below.
    span = np.array([x.min(), x.max()])

    d["short"] = [s.split("/")[-1] for s in d.lineage]
    #: NAMED FOR ORIENTATION ONLY, and declared as such: the two largest on each
    #: axis. The finding names no lineage here, so any selection is the panel's.
    named = pd.concat([d.nsmallest(2, "delta"), d.nlargest(2, "d_polesep")])
    named = named.drop_duplicates("lineage").copy()
    named["lx"] = named.d_polesep + 0.004

    xlim = (x.min() - 0.012, x.max() + 0.055)
    ylim = (y.min() - 0.02, y.max() + 0.03)

    def _clip(x0, x1):
        """endpoints of the principal axis, trimmed to the panel."""
        pts = []
        for xx in (x0, x1):
            yy = my + slope * (xx - mx)
            if ylim[0] <= yy <= ylim[1]:
                pts.append((xx, yy))
        for yy in ylim:
            xx = mx + (yy - my) / slope
            if x0 <= xx <= x1 and xlim[0] <= xx <= xlim[1]:
                pts.append((xx, yy))
        pts = sorted(set(pts))
        return pts[0], pts[-1]

    (sx0, sy0), (sx1, sy1) = _clip(span[0], span[1])
    seg = pd.DataFrame([{"x": sx0, "xend": sx1, "y": sy0, "yend": sy1}])
    #: the segment is a LAYER, not data, so nothing else would notice its loss
    assert (xlim[0] <= sx0 <= xlim[1] and xlim[0] <= sx1 <= xlim[1]
            and ylim[0] <= sy0 <= ylim[1] and ylim[0] <= sy1 <= ylim[1]), \
        "the principal axis still leaves the panel and would be dropped"
    assert abs(sx1 - sx0) > 0.3 * (span[1] - span[0]), \
        "the clipped principal axis is too short to read as a summary"
    assert x.min() > xlim[0] and x.max() < xlim[1], "a lineage falls off the x axis"
    assert y.min() > ylim[0] and y.max() < ylim[1], "a lineage falls off the y axis"

    p = (
        ggplot()
        + geom_hline(yintercept=0, color="#999999", size=0.4)
        + geom_vline(xintercept=0, color="#999999", size=0.4)
        + geom_segment(seg, aes("x", "y", xend="xend", yend="yend"),
                       color=AXIS_C, size=0.8, linetype="dashed", alpha=0.8)
        + geom_point(d, aes("d_polesep", "delta"), size=2.8, color=DOT_C,
                     alpha=0.8)
        + geom_text(named, aes("lx", "delta", label="short"), size=6.4,
                    ha="left", color="#444444")
        + scale_x_continuous(limits=xlim)
        + scale_y_continuous(limits=ylim)
        + labs(
            title="Where alignment drives the poles further apart, superposition collapses further -- on two independent instruments",
            subtitle=(
                f"One dot per lineage, {len(d)} lineages. x is the change in pole separation under\n"
                "alignment (aligned minus base); y is the change in the superposition signal. Both are\n"
                "differences, so the origin is 'alignment did nothing to this quantity'.\n"
                f"Spearman rho {rs:+.3f} (p {ps:.4f}), Pearson r {rp:+.3f} (p {pp:.4f}).\n"
                "THE TWO AXES COME FROM INDEPENDENT SUBSTRATES, which is the reason the correlation is\n"
                "worth anything: `pole_sep` is measured on the L3 hidden states, the superposition\n"
                "signal on the twp output ratio. This is two instruments agreeing, not one instrument\n"
                "agreeing with itself.\n"
                "THE DASHED LINE IS A PRINCIPAL AXIS AND NOT A REGRESSION, and the difference is the\n"
                "point. A least-squares line of y on x minimises error in y given x, which is the\n"
                "graphical form of 'x predicts y' -- and the finding's own bolded caveat is that the\n"
                "arrow is NOT established. The principal axis is symmetric: it is the same line\n"
                "whichever variable is called the predictor.\n"
                "IT IS ALSO STEEPER THAN EITHER REGRESSION WOULD BE, and deliberately does not hug the\n"
                "cloud. That is a property of the symmetric solution at a moderate correlation, not a\n"
                "bad fit: the two least-squares lines here would disagree with each other, and this one\n"
                "sits between them without choosing.\n"
                "THE CONFOUND THIS CANNOT RULE OUT. Both quantities may simply track HOW MUCH ALIGNMENT\n"
                "HAPPENED. A heavily aligned model would show more pole separation and more\n"
                "superposition loss with neither causing the other. Settling it needs the SFT/DPO\n"
                "checkpoint ladder, where separation either precedes collapse or does not -- that work\n"
                "is M05 and it is held.\n"
                "FOUR LINEAGES ARE NAMED FOR ORIENTATION and the finding names none, so the selection is\n"
                "this panel's: the two largest superposition losses and the two largest pole\n"
                "separations."),
            x="change in pole separation under alignment  (aligned - base)",
            y="change in superposition signal",
            caption=(
                "Producer: meta/M02_frame_exit/scripts/poles_superposition_figs.py from\n"
                "results/polesep_vs_superposition.csv (producer scripts/poles_and_superposition.py).\n"
                "plot-debt M02 candidate 4.\n"
                "Asserted before drawing: 45 lineages with no duplicate; Spearman rho and Pearson r\n"
                "each within 5e-04 of the finding, and both p values within 5e-05; that the principal\n"
                "axis agrees in sign with the correlation; and that no lineage falls outside either\n"
                "axis limit, since a dot dropped from a 45-point cloud would change the shape a reader\n"
                "reads off it without changing any number in this caption."),
        )
        + theme_minimal()
        + theme(figure_size=(12.2, 8.0),
                plot_title=element_text(size=10.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                plot_caption=element_text(size=6.3, color="#666666", ha="left",
                                          lineheight=1.45))
    )
    out = os.path.join(FIGURES, "polesep_vs_superposition.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    n={len(d)}  Spearman {rs:+.4f} (p {ps:.4f})  "
          f"Pearson {rp:+.4f} (p {pp:.4f})")
    print(f"    principal-axis slope {slope:+.4f} (symmetric; no regression drawn)")
    print(f"    named for orientation: {', '.join(named.short)}")
    return out


REGISTRY = {"polesep_superposition": polesep_superposition}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:24s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    names = a.names or list(REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    os.makedirs(FIGURES, exist_ok=True)
    for n in names:
        print(f"{n}:")
        REGISTRY[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
