#!/usr/bin/env python
"""Figures for findings V (scene-locality). Per-letter registry convention.

    uv run python meta/M01_displacement/scripts/plot_v_figs.py
    uv run python meta/M01_displacement/scripts/plot_v_figs.py scene_locality
    uv run python meta/M01_displacement/scripts/plot_v_figs.py --list

plot-debt `queue 18` (M01 candidate 13). plotnine at 300 dpi, output to
../figures/, booked-number asserts before drawing. Case 1 by shape: reads a
committed CSV and writes only pixels.

THE FENCE IS THAT THIS IS NOT A DIRECTION
------------------------------------------
V's own finding is that there is NO global displacement direction: mean
pairwise cosine between site vectors is 0.059. The 14-of-14 result here is
families agreeing on a CONTRAST -- twin sites resemble each other more than
random pairs do -- and not on a shared vector. **The panel must not be
readable as one**, so nothing here is drawn as an arrow, an axis or a
projection; it is two cosines per family and the gap between them.

And the random column is the same number as that null: its mean is 0.060
against V's 0.059 global figure. So the baseline on this panel IS the
no-global-direction result, which is the honest way to show both at once.

THE MEAN AND THE MEDIAN DISAGREE AND THE FINDING USES THE MEAN
----------------------------------------------------------------
    twin    mean 0.3273   median 0.3364
    random  mean 0.0603   median 0.0542

A per-family dot plot reaches for the median by default and the gap is small
enough to read as rounding rather than as a different statistic (registrar,
`queue 18`). Both are drawn, and the one the finding quotes is labelled.

WHICH OF FIVE FILES, WHICH IS THE QUESTION THAT COST THE LAST TWO ITEMS
------------------------------------------------------------------------
`results/v_displacement_twin*.csv` is five files. The headline reproduces from
the unsuffixed one, and this producer names it rather than globbing.

**Two of the five are byte-identical.** `v_displacement_twin.csv` and
`v_displacement_twin_verbs.csv` share md5 `85031fe2` and every `n_pairs`,
though `_verbs` is the producer's suffix for a lexical-verb restriction
(`v_displacement_vector.py:404`). So either that restriction does not reach
this output or one file is a stale copy of the other. Asserted here, because a
figure citing the `_verbs` file would claim a restriction that is not in its
data.
"""
import argparse
import csv
import hashlib
import os
import statistics as st
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
FIGURES = os.path.join(M01, "figures")

#: V-5, on results/v_displacement_twin.csv
BOOKED_V5 = {"twin_mean": 0.327, "random_mean": 0.060, "n_families": 14,
             "wins": 14}
BOOKED_V5_MEDIANS = {"twin": 0.336, "random": 0.054}
#: V's global result, which the random column reproduces
BOOKED_V_GLOBAL_COS = 0.059
TWIN_C, RAND_C = "#1f4e79", "#9a9a9a"


def scene_locality():
    """queue 18: twin sites resemble each other; there is still no direction."""
    from plotnine import (aes, element_blank, element_text, geom_point,
                          geom_segment, geom_text, geom_vline, ggplot, labs,
                          scale_color_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)

    src = os.path.join(RESULTS, "v_displacement_twin.csv")
    rows = list(csv.DictReader(open(src)))
    for r in rows:
        r["twin"] = float(r["twin_cos"])
        r["rand"] = float(r["random_cos"])

    assert len(rows) == BOOKED_V5["n_families"], \
        f"families drifted: {len(rows)} vs booked {BOOKED_V5['n_families']}"
    tm, rm = st.mean(r["twin"] for r in rows), st.mean(r["rand"] for r in rows)
    assert abs(tm - BOOKED_V5["twin_mean"]) < 0.001, f"twin mean {tm:.4f}"
    assert abs(rm - BOOKED_V5["random_mean"]) < 0.001, f"random mean {rm:.4f}"
    wins = sum(1 for r in rows if r["twin"] > r["rand"])
    assert wins == BOOKED_V5["wins"], f"{wins} of {len(rows)} families, not 14"
    tmed = st.median(r["twin"] for r in rows)
    rmed = st.median(r["rand"] for r in rows)
    assert abs(tmed - BOOKED_V5_MEDIANS["twin"]) < 0.001, f"twin median {tmed:.4f}"
    assert abs(rmed - BOOKED_V5_MEDIANS["random"]) < 0.001, f"random median {rmed:.4f}"
    #: THE MEAN AND MEDIAN MUST STAY DISTINCT OR THE PANEL'S POINT ABOUT THEM
    #: IS FALSE. If they ever converge, the two-statistic caption is noise.
    assert abs(tm - tmed) > 0.005, \
        "twin mean and median have converged; the panel distinguishes them"

    #: the `_verbs` twin is byte-identical to the unsuffixed one
    other = os.path.join(RESULTS, "v_displacement_twin_verbs.csv")
    if os.path.exists(other):
        h1 = hashlib.md5(open(src, "rb").read()).hexdigest()
        h2 = hashlib.md5(open(other, "rb").read()).hexdigest()
        assert h1 == h2, \
            ("v_displacement_twin_verbs.csv now DIFFERS from the unsuffixed "
             "file; this producer's docstring says they are identical and "
             "would be wrong")

    d = pd.DataFrame(rows).sort_values("twin").reset_index(drop=True)
    d["y"] = range(len(d))
    pts = pd.concat([
        d.assign(v=d.twin, col=TWIN_C),
        d.assign(v=d["rand"], col=RAND_C)])

    p = (
        ggplot()
        + geom_vline(xintercept=BOOKED_V_GLOBAL_COS, linetype="dashed",
                     color="#b03030", size=0.45)
        + geom_segment(d, aes("rand", "y", xend="twin", yend="y"),
                       color="#c9c9c9", size=0.6)
        + geom_point(pts, aes("v", "y", color="col"), size=2.8)
        + geom_text(d, aes("twin", "y", label="family"), size=6.4, ha="left",
                    nudge_x=0.012, color="#333333")
        + scale_color_identity()
        + scale_x_continuous(limits=(0, 0.58),
                             breaks=[0, 0.059, 0.1, 0.2, 0.3, 0.4, 0.5],
                             labels=["0", "0.059", "0.1", "0.2", "0.3", "0.4", "0.5"])
        + scale_y_continuous(breaks=[], limits=(-0.8, len(d) - 0.2))
        + labs(
            title="Twin sites displace alike in all 14 families, and there is still no global direction",
            subtitle=(
                "Cosine between displacement vectors at TWIN sites (blue) against RANDOM site pairs\n"
                "(grey), one row per family, 14 families. Twin exceeds random in 14 of 14.\n"
                "THE GREY POINTS ARE V's OWN NULL AND THAT IS THE POINT OF PUTTING THEM HERE. Their\n"
                "mean is 0.060, against the 0.059 mean pairwise cosine V measures between site vectors\n"
                "generally -- so the random column reproduces the no-global-direction result, and the\n"
                "dashed line at 0.059 is where a world with no scene-locality would put every blue dot.\n"
                "WHAT THIS IS NOT. It is not a shared direction. Families agree on a CONTRAST -- twins\n"
                "resemble each other more than strangers do -- and V's finding that displacement has no\n"
                "global vector stands beside it, not against it. Nothing here is an arrow, an axis or a\n"
                "projection, because each of those would invite the reading the finding refuses.\n"
                "THE FINDING QUOTES MEANS AND THE MEDIANS DIFFER. Twin mean 0.327 against median 0.336;\n"
                "random mean 0.060 against median 0.054. A per-family panel reaches for the median by\n"
                "default and the gap is small enough to pass as rounding, so both are stated and the\n"
                "quoted one is named."),
            x="cosine between displacement vectors", y="",
            caption=(
                "Producer: meta/M01_displacement/scripts/plot_v_figs.py from\n"
                "results/v_displacement_twin.csv (producer v_displacement_vector.py).\n"
                "Asserted before drawing: 14 families; twin mean 0.327 and random mean 0.060 to 0.001;\n"
                "14 of 14 families with twin above random; both medians; and that the mean and median\n"
                "remain distinct, since the panel's point about them depends on it.\n"
                "FIVE FILES MATCH results/v_displacement_twin*.csv AND THIS NAMES ONE. The three\n"
                "residualised variants give twin 0.310-0.313 and random 0.045-0.051, all still 14 of 14,\n"
                "so the result survives residualisation with a smaller gap; the headline is the\n"
                "unresidualised file and that is what is drawn.\n"
                "AND TWO OF THE FIVE ARE BYTE-IDENTICAL: v_displacement_twin.csv and\n"
                "v_displacement_twin_verbs.csv share md5 85031fe2 and every n_pairs, though `_verbs` is\n"
                "the producer's suffix for a lexical-verb restriction (v_displacement_vector.py:404).\n"
                "Either that restriction does not reach this output or one file is a stale copy. Their\n"
                "identity is asserted here so the claim cannot go stale silently."),
        )
        + theme_minimal()
        + theme(figure_size=(12.2, 6.6),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "v5_scene_locality.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    14 families, twin mean {tm:.4f} (median {tmed:.4f}), "
          f"random mean {rm:.4f} (median {rmed:.4f}), {wins}/14")
    print(f"    random mean sits at V's global pairwise cosine {BOOKED_V_GLOBAL_COS}")
    return out


REGISTRY = {"scene_locality": scene_locality}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:16s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
