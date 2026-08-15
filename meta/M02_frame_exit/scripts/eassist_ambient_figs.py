#!/usr/bin/env python
"""Ambient control-token emission: a near-unanimous sign and a one-recipe size.

    uv run python meta/M02_frame_exit/scripts/eassist_ambient_figs.py
    uv run python meta/M02_frame_exit/scripts/eassist_ambient_figs.py --list

plot-debt M02 candidate 7. plotnine at 300 dpi, output to ../figures/, booked
numbers asserted before drawing. Case 1 by shape: reads a committed CSV and
writes only pixels.

THIS FIGURE EXISTS TO FORBID A READING
----------------------------------------
`M02_eassist_ambient.md` section 4: *"'aligned models emit control tokens 10x
more often' is a statement about Falcon3 wearing a roster's clothes, and the
pooled ratio must never travel alone."* Four models of 29 carry 68.4% of every
strict hit in the file.

So the panel's job is not to show that alignment raises the rate. It is to make
the pooled ratio unquotable without its carriers, which means every one of the
29 pairs is drawn and the four are named on the panel rather than in a caption.

AND IT IS NOT A VENDOR EFFECT, WHICH NEEDS THE MAMBA ROWS ON THE PANEL
------------------------------------------------------------------------
The obvious defensive reading of a one-family result is that the family is the
vendor. `Falcon3-Mamba-7B-Instruct` and `falcon-mamba-7b-instruct` are the same
vendor and sit at 0.35% and 0.18%. **A reader cannot rule the vendor out from a
sorted list unless the vendor's other models are findable in it**, so those two
rows are labelled explicitly even though nothing distinguishes them numerically
from the rest of the floor.

THE AXIS IS LINEAR AND EVERY VALUE IS PRINTED
-----------------------------------------------
A log axis would make the floor legible and dissolve the concentration, which is
the finding. A linear axis shows the concentration and smears 25 rows against
zero. Printing each rate beside its own lollipop keeps the floor readable
without asking the geometry to carry two jobs.
"""
import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
SRC = os.path.join(RESULTS, "eassist_ambient.csv")

#: M02_eassist_ambient.md section 4
BOOKED = {"n_pairs": 29, "falcon3_3b": 52.76, "falcon3_10b": 31.40,
          "falcon3_7b": 28.49, "falcon3_1b": 28.05, "olmo": 2.29,
          "ceiling": 2.01, "mamba_a": 0.35, "mamba_b": 0.18}
#: the sign, which is the part the finding calls defensible
BOOKED_DIR = {"aligned": 17, "tie": 11, "base": 1}
F3_RE = r"tiiuae/Falcon3-\d+B-Instruct"
F3_C, MAMBA_C, REST_C = "#b03030", "#d4762a", "#8fa8bf"


def ambient_concentration():
    """M02 candidate 7: four models of 29 carry the pooled ratio."""
    from plotnine import (aes, element_blank, element_text, geom_point,
                          geom_segment, geom_text, geom_vline, ggplot, labs,
                          scale_color_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)

    d = pd.read_csv(SRC)
    assert len(d) == BOOKED["n_pairs"], f"{len(d)} pairs, not 29"
    d["ar"] = 100 * d.aligned_strict_rate
    d["br"] = 100 * d.base_strict_rate
    d["short"] = [s.split("/")[-1] for s in d.aligned]

    f3 = d.aligned.str.match(F3_RE)
    assert int(f3.sum()) == 4, f"{int(f3.sum())} Falcon3 instruct rows, not 4"
    by = dict(zip(d.short, d.ar))
    for key, name in (("falcon3_3b", "Falcon3-3B-Instruct"),
                      ("falcon3_10b", "Falcon3-10B-Instruct"),
                      ("falcon3_7b", "Falcon3-7B-Instruct"),
                      ("falcon3_1b", "Falcon3-1B-Instruct"),
                      ("olmo", "Olmo-3-7B-Instruct-DPO")):
        assert abs(by[name] - BOOKED[key]) < 0.005, \
            f"{name}: {by[name]:.4f}% against section 4's {BOOKED[key]}%"

    #: THE CEILING IS A SELECTION AND THE FINDING DOES NOT SAY SO. "every other
    #: aligned model, max 2.01%" excludes the four Falcon3 rows AND the Olmo row
    #: listed above it. Recovered here and stated on the panel, because a
    #: reference line at 2.01% otherwise looks like a maximum over 25 rows when
    #: it is a maximum over 24.
    rest = d[~f3 & (d.aligned != "allenai/Olmo-3-7B-Instruct-DPO")]
    ceil = float(rest.ar.max())
    assert abs(ceil - BOOKED["ceiling"]) < 0.005, f"ceiling {ceil:.4f}%"
    assert len(rest) == BOOKED["n_pairs"] - 5, \
        f"the ceiling is a max over {len(rest)} rows, not 24"

    mam = d[d.aligned.str.contains("amba", case=False)]
    assert len(mam) == 2, f"{len(mam)} Mamba rows, not 2"
    assert abs(mam.ar.max() - BOOKED["mamba_a"]) < 0.005, "Mamba rate a"
    assert abs(mam.ar.min() - BOOKED["mamba_b"]) < 0.005, "Mamba rate b"
    assert (mam.ar < ceil).all(), \
        "a Mamba row now exceeds the ceiling; the not-a-vendor-effect line fails"

    dirs = d.direction.value_counts().to_dict()
    assert dirs == BOOKED_DIR, f"direction counts moved: {dirs}"
    carried = 100 * d[f3].aligned_strict.sum() / d.aligned_strict.sum()
    assert 68.0 < carried < 69.0, f"the four carry {carried:.1f}% of strict hits"

    d = d.sort_values("ar").reset_index(drop=True)
    d["y"] = range(len(d))
    d["col"] = [F3_C if re.match(F3_RE, a) else
                MAMBA_C if "amba" in a.lower() else REST_C for a in d.aligned]
    d["lab"] = [f"{v:.2f}%" for v in d.ar]
    #: floor labels are pushed clear of the 2.01% reference line rather than
    #: sitting 1.1 to the right of their own dot, which put half of them on it
    d["lx"] = [max(v + 1.1, ceil + 1.0) for v in d.ar]
    d["nx"] = -1.1

    p = (
        ggplot()
        + geom_vline(xintercept=ceil, linetype="dashed", color="#777777",
                     size=0.5)
        + geom_segment(d, aes(0, "y", xend="ar", yend="y", color="col"),
                       size=1.0, alpha=0.7)
        + geom_point(d, aes("ar", "y", color="col"), size=2.6)
        + geom_point(d[d.br > 0], aes("br", "y"), size=2.0, color="#333333",
                     shape="s")
        + geom_text(d, aes("lx", "y", label="lab"), size=6.0, ha="left",
                    color="#555555")
        + geom_text(d, aes("nx", "y", label="short"), size=6.4, ha="right",
                    color="#333333")
        + scale_color_identity()
        + scale_x_continuous(limits=(-26, 60),
                             breaks=[0, 2.01, 10, 20, 30, 40, 50],
                             labels=["0", "2.01%", "10%", "20%", "30%", "40%",
                                     "50%"])
        + scale_y_continuous(breaks=[], limits=(-0.9, len(d) - 0.1))
        + labs(
            title="The direction holds across 29 families and the magnitude is one training recipe",
            subtitle=(
                "Rate at which each ALIGNED model emits a control token into an ambient continuation,\n"
                "one row per base/aligned pair, 29 pairs. Dark squares mark the few bases with a nonzero\n"
                "rate of their own; every other base is at 0.00%.\n"
                f"FOUR MODELS OF 29 CARRY {carried:.1f}% OF EVERY STRICT HIT IN THE FILE. Falcon3-3B-Instruct\n"
                "alone is at 52.76%, and the dashed line is 2.01% -- the ceiling for everyone outside\n"
                "that family and the Olmo row above it.\n"
                "SO THE POOLED RATIO MUST NEVER TRAVEL ALONE. The finding's own sentence: \"aligned models\n"
                "emit control tokens 10x more often\" is a statement about Falcon3 wearing a roster's\n"
                "clothes. Every pair is drawn here so the pooled figure cannot be quoted without its\n"
                "carriers being visible in the same picture.\n"
                "AND IT IS NOT A VENDOR EFFECT, which is why the two Mamba rows are named. Same vendor,\n"
                "0.35% and 0.18%, indistinguishable from the floor. A reader cannot rule out \"tiiuae\n"
                "models do this\" from a sorted list unless the vendor's other models are findable in it.\n"
                f"THE DEFENSIBLE CLAIM IS THE SIGN, NOT THE SIZE. Direction across the 29 pairs: {dirs['aligned']} where\n"
                f"the aligned model is higher, {dirs['tie']} ties, {dirs['base']} where the base is. The direction is\n"
                "near-unanimous; the magnitude is one instruct recipe. Both belong in any citation and\n"
                "neither substitutes for the other.\n"
                "THE AXIS IS LINEAR AND EVERY RATE IS PRINTED. A log axis would make the floor legible\n"
                "and dissolve the concentration that is the finding, so the numbers carry the floor\n"
                "instead of the geometry."),
            x="ambient control-token rate, aligned model (%)", y="")
        + theme_minimal()
        + theme(figure_size=(12.8, 9.6),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "eassist_ambient_concentration.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {len(d)} pairs; four Falcon3 instructs carry {carried:.1f}% of strict hits")
    print(f"    ceiling outside them and Olmo: {ceil:.4f}% over {len(rest)} rows")
    print(f"    direction: {dirs}")
    print(f"    Mamba rows named: {', '.join(mam.short)} at "
          f"{mam.ar.max():.2f}% and {mam.ar.min():.2f}%")
    return out


REGISTRY = {"ambient_concentration": ambient_concentration}


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
