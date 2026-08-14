#!/usr/bin/env python
"""Figures for Findings X (X_metonymy.md), one function per figure.

    uv run python meta/M01_displacement/scripts/plot_x_figs.py
    uv run python meta/M01_displacement/scripts/plot_x_figs.py word_vs_model
    uv run python meta/M01_displacement/scripts/plot_x_figs.py --list

Per-letter plotting convention (RH, 2026-08-14, plot-debt regime): one
script per letter in scripts/plot_<letter>_figs.py, a FIGURES registry,
plotnine at 300 dpi, output to ../figures/, slice in the subtitle,
booked-number asserts before drawing. The older `x_*.py` figure scripts
in this folder predate that regime and are left alone.

SAME AXES IS THE WHOLE FIGURE
-----------------------------
X section 3g sets two paired contrasts against each other: forcing a
different WORD into the slot moves the scene by +14.3 points at 12 of 12
cells, and changing the MODEL moves it -0.8 points at 15 of 30. The
claim is that one of these is an effect and the other is nothing.

That claim is only legible if both are drawn on ONE scale. Two panels
with independent axes would show two clouds of similar width and let the
reader conclude the two contrasts are comparable, which is the opposite
of the finding. So the axis is shared and the panels are stacked.

THE NULL PANEL CARRIES ITS OWN POWER
------------------------------------
A null is only worth drawing if the reader can see it is not an empty
measurement. The finding states the machinery would have detected 8.4
points at 80 percent power, so that band is drawn ON the null panel: the
observed -0.8 sits inside a region the instrument could have resolved,
which is what makes it a real null rather than an underpowered one.

PAIRED, NEVER POOLED, AND THE FINDING SAYS WHY
----------------------------------------------
X records that a pooled rate "nearly went into this document": unpaired,
the forced-genital arms read base 40% against aligned 32%, a tempting -8
points, which at the cell level is -8.3 with p = 0.484 because per-cell
base rates run 0% to 80% and pooling lets the high cells dominate. The
direction also reverses within families -- AmberSafe goes 80 to 0 on
`penis` and 40 to 60 on `cock`, Tulu-3-DPO the other way on both.

So the model null is **"no consistent direction across six alignment
implementations"**, NOT "nothing happens in any of them", and the panel
says that in those words. One point per cell is what makes the
distinction visible; a bar of two pooled rates would erase it.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
FIGURES = os.path.join(M01, "figures")

#: Booked in X_metonymy.md section 3g's table.
BOOKED = {"word": (14.3, 12, 12), "model": (-0.8, 15, 30),
          "word_no_artifact": (19.7, 10, 10)}
MDE = 8.4   #: points detectable at 80% power, section 3g


def _cells():
    d = pd.read_csv(os.path.join(RESULTS, "x_beam_frame.csv"))
    f = d[d["class"].isin(["genital", "digit"])].copy()
    #: THE WORD: paired within (pair, role) -- 6 pairs x 2 roles = 12
    cw = f.groupby(["pair", "role", "class"]).sexual.mean().unstack()
    word = (cw["genital"] - cw["digit"]).dropna().rename("delta").reset_index()
    #: THE MODEL: paired within (pair, word) -- 6 pairs x 5 words = 30
    cm = f.groupby(["pair", "word", "role"]).sexual.mean().unstack()
    model = (cm["aligned"] - cm["base"]).dropna().rename("delta").reset_index()
    #: the artifact-excluded word contrast, asserted but not drawn
    fa = f[~f.artifact]
    ca = fa.groupby(["pair", "role", "class"]).sexual.mean().unstack()
    word_na = (ca["genital"] - ca["digit"]).dropna()
    return word, model, word_na


def word_vs_model():
    """X 3g: the word moves the scene, the model does not.

    Two paired contrasts on ONE axis. One point per cell; the diamond is
    the mean the finding books.
    """
    from plotnine import (aes, element_blank, element_text, facet_grid,
                          geom_jitter, geom_point, geom_rect, geom_text,
                          geom_vline, ggplot, labs, scale_color_identity,
                          scale_y_continuous, theme, theme_minimal)

    word, model, word_na = _cells()

    for key, frame in (("word", word), ("model", model)):
        m, pos, n = BOOKED[key]
        got = (round(float(frame.delta.mean()), 1),
               int((frame.delta > 0).sum()), len(frame))
        assert got == (m, pos, n), f"{key} drifted: {got} vs booked {(m, pos, n)}"
    m, pos, n = BOOKED["word_no_artifact"]
    got = (round(float(word_na.mean()), 1), int((word_na > 0).sum()), len(word_na))
    assert got == (m, pos, n), \
        f"artifact-excluded word drifted: {got} vs booked {(m, pos, n)}"

    W = "THE WORD\ngenital vs digit, paired within (pair, role)"
    M = "THE MODEL\naligned vs base, paired within (pair, word)"
    word["panel"], model["panel"] = W, M
    d = pd.concat([word[["panel", "delta"]], model[["panel", "delta"]]],
                  ignore_index=True)
    d["col"] = np.where(d.panel == W, "#1f4e79", "#8a8a8a")
    d["panel"] = pd.Categorical(d.panel, categories=[W, M], ordered=True)

    ann = pd.DataFrame([
        {"panel": W, "delta": word.delta.mean(), "col": "#1f4e79",
         "txt": f"mean {word.delta.mean():+.1f} points     "
                f"{int((word.delta > 0).sum())} of {len(word)} cells positive     p 0.00049"},
        {"panel": M, "delta": model.delta.mean(), "col": "#8a8a8a",
         "txt": f"mean {model.delta.mean():+.1f} points     "
                f"{int((model.delta > 0).sum())} of {len(model)} cells positive     p 0.918"},
    ])
    ann["panel"] = pd.Categorical(ann.panel, categories=[W, M], ordered=True)

    #: the power band belongs to the null panel only
    band = pd.DataFrame([{"panel": M, "xmin": -MDE, "xmax": MDE,
                          "ymin": -0.42, "ymax": 0.42}])
    band["panel"] = pd.Categorical(band.panel, categories=[W, M], ordered=True)
    bandlab = pd.DataFrame([{"panel": M, "delta": 0.0, "y": -0.50,
                             "txt": f"the instrument would have shown {MDE} points here at 80% power"}])
    bandlab["panel"] = pd.Categorical(bandlab.panel, categories=[W, M],
                                      ordered=True)

    p = (
        ggplot()
        + geom_rect(band, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    fill="#f0b429", alpha=0.15)
        + geom_text(bandlab, aes("delta", "y", label="txt"), size=6.2,
                    ha="center", va="top", color="#8a6d1f")
        + geom_vline(xintercept=0, color="#333333", size=0.45)
        + geom_jitter(d, aes("delta", 0, color="col"), height=0.20, width=0,
                      size=2.2, alpha=0.7)
        + geom_point(ann, aes("delta", 0, color="col"), size=5.0, shape="D")
        + geom_text(ann, aes("delta", 0.62, label="txt", color="col"), size=7.0,
                    ha="center", va="bottom")
        + scale_color_identity()
        + scale_y_continuous(limits=(-0.72, 1.0))
        + facet_grid("panel ~ .")
        + labs(
            title="The word moves the scene. The model does not.",
            subtitle=(
                "Forcing one of five words into the same slot in both arms, 6 model pairs, 100 beams "
                "per record, K=5 sampled per record, two independent coders. One point per cell; "
                "position is the paired difference in mean sexual score of the continuation.\n"
                "SAME AXIS IN BOTH PANELS, which is the comparison. Changing the WORD moves the scene "
                "at every one of 12 cells; changing the MODEL moves it at 15 of 30, which is exactly "
                "chance.\n"
                "THE NULL IS A REAL NULL, NOT AN UNDERPOWERED ONE: the shaded band is the effect the "
                "same machinery would have detected at 80% power, and the observed value sits well "
                "inside it.\n"
                "PAIRED, NEVER POOLED. Unpaired these arms read base 40% against aligned 32%, a "
                "tempting -8 points that is -8.3 at p 0.484 once paired, because per-cell base rates "
                "run 0% to 80%. The direction reverses within families.\n"
                "So the model null is NO CONSISTENT DIRECTION ACROSS SIX ALIGNMENT IMPLEMENTATIONS, "
                "not \"nothing happens in any of them\".\n"
                "SCOPE IS TEN TOKENS: every record in the beam_fc stash is max_tokens=10."),
            x="paired difference in mean sexual score of the continuation (points, 0-100 scale)",
            y="",
            caption=("Producer: meta/M01_displacement/scripts/plot_x_figs.py from "
                     "results/x_beam_frame.csv.\n"
                     "Asserted before drawing: word +14.3 at 12 of 12, model -0.8 at 15 of 30, and "
                     "the artifact-excluded word contrast +19.7 at 10 of 10.\n"
                     "Format artifacts are INCLUDED here, as in the finding's headline table; "
                     "excluding them raises the word effect to +19.7 and leaves the model null."),
        )
        + theme_minimal()
        + theme(figure_size=(11.6, 5.8),
                plot_title=element_text(size=13, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.1, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                strip_text=element_text(size=7.8, weight="bold"),
                panel_spacing=0.06)
    )
    out = os.path.join(FIGURES, "x_word_vs_model.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    WORD  {word.delta.mean():+.1f} at {int((word.delta > 0).sum())}/{len(word)}"
          f"   MODEL {model.delta.mean():+.1f} at {int((model.delta > 0).sum())}/{len(model)}")
    return out


REGISTRY = {"word_vs_model": word_vs_model}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
