#!/usr/bin/env python
"""Figures for `contradiction_ratio_has_no_null.md`, one function per figure.

    uv run python meta/M02_frame_exit/scripts/contradiction_null_figs.py
    uv run python meta/M02_frame_exit/scripts/contradiction_null_figs.py numberline
    uv run python meta/M02_frame_exit/scripts/contradiction_null_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before
drawing. Script naming follows THIS folder rather than M01's per-letter
registry (registrar, [5889]): M02 names its figure scripts for the
finding, as `z_depth_exit_figs.py` does.

WHY A NUMBER LINE, AND WHY IT IS LINEAR AND ANCHORED AT ZERO
------------------------------------------------------------
The finding's claim is spatial and is about a SCALE rather than a
comparison: F11 reads its ratio with "below 1.0 = superposition, above
1.0 = resolution", and the finding's result is that 1.0 is not that
boundary, it is where a distribution holding NEITHER pole lands. So the
figure has to put the four calibration anchors in one space and let the
reader see where the observed value actually sits.

The axis is LINEAR and starts at 0. Both choices are forced by the
argument's first consequence: the interval below 1.0 runs from perfect
blending (0.000, by construction) to neither-pole (1.006), and the
observed 0.907 sits near the NEITHER-POLE end of it. A reader can only
see "near 1.006 and far from 0.000" if 0.000 is on the axis, which
rules out the log spacing the queue entry originally suggested. Log
would also spread 0.907 and 1.006 apart, which is the opposite of the
finding.

The zh panel's tail is the honest difficulty: obs runs to 54.08 against
en's 2.87. The panel is cut at the shared limit and the count and
maximum of what falls outside are printed ON the panel, per plot-debt's
truncations-stated rule. Rescaling to fit the tail would compress the
four anchors into the left eighth of the figure and destroy the only
thing it is for.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")

XMAX = 4.6

#: Booked in contradiction_ratio_has_no_null.md, the calibration block.
#: EN is the block quoted in the finding's headline table; zh is the twin.
BOOKED = {
    "en": {"obs": 0.907, "null": 1.006, "res": 4.031},
}


def _load():
    frames = []
    for lang in ("en", "zh"):
        p = os.path.join(RESULTS, f"contradiction_null_{lang}.csv")
        d = pd.read_csv(p)
        d["lang"] = lang
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def numberline():
    """The calibration number line: 1.0 is a place, not a boundary.

    Four anchors per language over the per-cell distribution of the
    observed ratio. The anchors are medians over cells for the measured
    three; perfect superposition is 0.000 BY CONSTRUCTION and is not a
    measurement, which the panel says rather than implying it is one
    more estimate.
    """
    from plotnine import (aes, element_blank, element_text, facet_wrap,
                          geom_jitter, geom_rect, geom_text, geom_vline, ggplot,
                          labs, scale_color_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)

    d = _load()

    #: OBSERVED and NEUTRALIZATION are ~0.1 apart and that proximity IS the
    #: finding, so they are labelled on OPPOSITE SIDES of the strip rather
    #: than nudged apart. Nudging would buy legibility by denying the result;
    #: opposite sides keep the two lines where they fall and still read.
    anchors = []
    for lang, g in d.groupby("lang"):
        anchors += [
            {"lang": lang, "what": "perfect superposition\n(A+B)/2", "x": 0.0,
             "col": "#8a8a8a", "side": "above"},
            {"lang": lang, "what": "OBSERVED\ncontradiction", "x": g.obs.median(),
             "col": "#b03030", "side": "above"},
            {"lang": lang, "what": "NEUTRALIZATION -- another BOTH", "x": g["null"].median(),
             "col": "#1f4e79", "side": "below"},
            {"lang": lang, "what": "RESOLUTION\n0.9A + 0.1B", "x": g.res.median(),
             "col": "#1f4e79", "side": "above"},
        ]
    a = pd.DataFrame(anchors)
    #: name and value are ONE text block per anchor. Drawn as two layers they
    #: overlapped each other at every anchor, which is the same defect as the
    #: labels colliding: two marks placed independently at the same x.
    a["label"] = [
        (f"{w}\n{x:.3f}" if s == "above" else f"{x:.3f}\n{w}")
        for w, x, s in zip(a.what, a.x, a.side)
    ]
    a["y"] = a.side.map({"above": 1.10, "below": -0.30})
    a["va"] = a.side.map({"above": "bottom", "below": "top"})

    en = a[a.lang == "en"].set_index("what").x
    b = BOOKED["en"]
    got = {"obs": en.filter(like="OBSERVED").iloc[0],
           "null": en.filter(like="NEUTRALIZATION").iloc[0],
           "res": en.filter(like="RESOLUTION").iloc[0]}
    for k, want in b.items():
        assert round(float(got[k]), 3) == want, \
            f"en {k} drifted: {got[k]!r} -> {round(float(got[k]), 3)} vs booked {want}"

    #: what the panel cannot show, counted rather than quietly dropped
    trunc = (d[d.obs > XMAX].groupby("lang")
             .agg(n=("obs", "size"), mx=("obs", "max")).reindex(["en", "zh"])
             .fillna({"n": 0, "mx": float("nan")}))
    lab, band = [], []
    for lang in ("en", "zh"):
        n, mx = int(trunc.loc[lang, "n"]), trunc.loc[lang, "mx"]
        tot = int((d.lang == lang).sum())
        lab.append({
            "lang": lang, "x": XMAX * 0.99, "y": 1.44,
            "t": (f"{tot:,} cells" if not n else
                  f"{tot:,} cells; {n} above {XMAX:g} not shown (max {mx:.2f})"),
        })
        #: one row per facet, or the band renders in the first panel only
        band.append({"lang": lang, "xmin": 0.0, "xmax": 1.0,
                     "ymin": -0.40, "ymax": 1.02})
        band.append({"lang": lang, "x": 0.5, "y": 0.99,
                     "t": 'F11 reads this whole band as "superposition"'})
    labels = pd.DataFrame(lab)
    rect = pd.DataFrame([r for r in band if "xmin" in r])
    bandlab = pd.DataFrame([r for r in band if "t" in r])

    d = d[d.obs <= XMAX].copy()
    for f in (d, a, labels, rect, bandlab):
        f["lang"] = pd.Categorical(f.lang, categories=["en", "zh"], ordered=True)

    p = (
        ggplot()
        #: the band F11's rule calls "superposition". Shaded to be argued
        #: with, not to be believed: its right edge is the whole point.
        + geom_rect(rect, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    fill="#f0b429", alpha=0.11)
        + geom_text(bandlab, aes("x", "y", label="t"), size=6.4, ha="center",
                    va="top", color="#8a6d1f")
        + geom_jitter(d, aes("obs", 0), height=0.30, width=0, size=0.32,
                      alpha=0.11, color="#1f4e79")
        + geom_vline(a, aes(xintercept="x", color="col"), size=0.55)
        + geom_text(a, aes("x", "y", label="label", color="col", va="va"),
                    size=6.5, ha="center", lineheight=1.30)
        + geom_text(labels, aes("x", "y", label="t"), size=5.8, ha="right",
                    va="bottom", color="#666666")
        + scale_color_identity()
        + facet_wrap("lang", ncol=1,
                     labeller=lambda s: {"en": "ENGLISH", "zh": "CHINESE"}[s])
        + scale_x_continuous(limits=(-0.18, XMAX),
                             breaks=[0, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        + scale_y_continuous(limits=(-0.80, 1.62))
        + labs(
            title="1.0 is not the boundary. It is where a distribution holding NEITHER pole lands.",
            subtitle=(
                "F11 scores contradiction as JS(AB, mean(A,B)) / min(JS(AB,A), JS(AB,B)) and reads "
                "below 1.0 as superposition, above 1.0 as resolution.\n"
                "Measured against a real neutralization reference on the same substrate, the band "
                "below 1.0 runs from perfect blending to neither-pole, and the OBSERVED value sits "
                "at the neither-pole end.\n"
                "Anchors are medians over model x group cells; perfect superposition is 0.000 BY "
                "CONSTRUCTION, not a measurement. Points behind are the per-cell observed ratios.\n"
                "STATUS PROVISIONAL: the calibration is solid; effect sizes rest on a null whose "
                "frame-mismatch threat is tested on 10 of 22 groups."),
            x="contradiction ratio",
            y="",
            caption=("Producer: meta/M02_frame_exit/scripts/contradiction_null_figs.py from "
                     "results/contradiction_null_{en,zh}.csv (producer contradiction_null.py).\n"
                     "Axis is linear and anchored at 0 so that the distance from perfect blending "
                     "is legible; log spacing cannot show a 0.000 anchor and would separate the "
                     "observed value from neutralization, which is the opposite of the finding."),
        )
        + theme_minimal()
        + theme(figure_size=(12.0, 7.0),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.2, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                strip_text=element_text(size=8.5, weight="bold"),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                panel_spacing=0.06)
    )
    out = os.path.join(FIGURES, "contradiction_null_numberline.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for lang, g in _load().groupby("lang"):
        print(f"    {lang}: obs {g.obs.median():.4f}  null {g['null'].median():.4f}  "
              f"res {g.res.median():.4f}  n {len(g):,}")
    return out


FIGURES_REGISTRY = {"numberline": numberline}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="figure names; default all")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in FIGURES_REGISTRY.items():
            print(f"  {k:12s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
