#!/usr/bin/env python
"""Figures for `propagation.md`: how much of an imposition reaches the chain.

    uv run python meta/M06_generation/scripts/m06_propagation_figs.py
    uv run python meta/M06_generation/scripts/m06_propagation_figs.py slope
    uv run python meta/M06_generation/scripts/m06_propagation_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.

THE FIGURE HAS TO CARRY TWO CLAIMS THAT PULL OPPOSITE WAYS
----------------------------------------------------------
H2: the slope is positive and not marginal -- 37 of 40 pairs, p 2e-08.
H1: the slope is TINY -- about 1.2% of the imposition gets through, so
roughly 99% is absorbed.

A single axis serves one of these and defeats the other. Cropped to the
data the effect looks substantial; drawn against the imposition it
looks like nothing and the reader cannot see that it is reliable. So
the panel is drawn at BOTH scales side by side, same points twice: at
the scale of the imposition, where the whole result sits against the
axis, and zoomed, where the per-pair spread and the sign counts are
legible. Neither panel alone is honest.

WHAT IS NOT DRAWN, AND WHY
--------------------------
H3 is the finding's interpretive payoff: the same slope for a
SELF-SAMPLED opening is 0.016 to 0.024 nats per nat, so an imposed
improbable word propagates no more than one the model chose itself.
**That reference is NOT on this figure, and its VALUE is not quoted on
it either.** `propagation.md:72` sources the reference to
`opening_matched`'s within-prompt ANCOVA, and plot-debt:220 records the
OPENING-MATCHED FAMILY as UNPLOTTABLE, withdrawn at construction level
([5811]), "nothing from it may be drawn in any form" -- and a value
printed in a subtitle is on the figure.

The first draft of this file got that half-right and half-wrong: it
omitted the reference from the panel geometry and then quoted the
numbers in the subtitle while explaining they were fenced. Registrar's
tightening at [5934] is the rule now applied here -- a panel should not
NAME an absent leg, because naming it puts it in the reader's head
anyway. So the panel states its own scope and stops. This docstring
carries the full account, which is where it belongs: the docstring is
read by whoever edits the producer, the subtitle by whoever reads the
result.

Worth naming because the number does not look withdrawn where a
plotter meets it: `propagation.json` carries
`undisturbed_reference: [0.016, 0.024]` as a bare pair of floats with
no marker of its provenance or its status. The withdrawal notice lives
in a different finding and in the queue; the value has travelled into
this artifact without it. Referred rather than resolved here.

ON THE HEADLINE PERCENTAGE
--------------------------
The doc says "b / ln 2 ~ 0.013 nats", about 1.3%. Recomputed from the
committed per-pair slopes the median gives 1.20% aligned and 1.05%
base; the mean gives 1.67% and 1.18%. 1.3% is a loose prose
approximation rather than a booked value and nothing turns on it -- the
claim is "roughly 99% absorbed" either way -- but this panel quotes
what it computes.
"""
import argparse
import os
import sys
from math import comb, log

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
CELLS = os.path.join(RESULTS, "propagation_cells.parquet")

LN2 = log(2)

#: Booked in results/propagation.json, per-pair slopes in nats-per-bit.
BOOKED = {
    ("all arm words", "aligned"): (0.00830, 37, 3),
    ("all arm words", "base"): (0.00729, 36, 4),
    ("single-token only", "aligned"): (0.00808, 38, 2),
    ("single-token only", "base"): (0.00679, 32, 8),
}
BOOKED_DIFF = {"all arm words": (0.00388, 26, 14, 0.0807),
               "single-token only": (0.00287, 27, 13, 0.0385)}
SCALES = {"at the scale of the imposition": 1.0,
          "zoomed (same points)": 0.045}


def _sign_p(up, dn):
    lo = min(up, dn)
    return min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)


def slope():
    """Propagation: the damage is real, robust, and almost nothing.

    One point per model pair, at both scales. Position is the share of
    the opening's improbability that reaches the continuation, which is
    the fitted slope in nats-per-bit divided by ln 2.
    """
    from plotnine import (aes, element_blank, element_text, facet_grid,
                          geom_jitter, geom_point, geom_text, geom_vline,
                          ggplot, labs, scale_color_manual,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    d = pd.read_parquet(CELLS)
    d["prop"] = d.slope / LN2

    stats = {}
    for key, g in d.groupby(["variant", "role"]):
        s = g.slope.values
        stats[key] = (float(np.median(s)), int((s > 0).sum()), int((s < 0).sum()))
    for key, (m, up, dn) in BOOKED.items():
        gm, gup, gdn = stats[key]
        assert round(gm, 5) == m and (gup, gdn) == (up, dn), \
            f"{key} drifted: {round(gm, 5)} {gup}/{gdn} vs booked {m} {up}/{dn}"

    #: the aligned-base contrast, asserted so the fence below cannot go
    #: stale without this refusing
    piv = d.pivot_table(index=["variant", "pair"], columns="role", values="slope")
    for variant, (m, up, dn, p) in BOOKED_DIFF.items():
        ds = (piv.loc[variant, "aligned"] - piv.loc[variant, "base"]).dropna()
        gm = round(float(np.median(ds)), 5)
        gup, gdn = int((ds > 0).sum()), int((ds < 0).sum())
        assert (gm, gup, gdn) == (m, up, dn), \
            f"{variant} diff drifted: {gm} {gup}/{gdn} vs booked {m} {up}/{dn}"
        assert round(_sign_p(gup, gdn), 4) == round(p, 4), \
            f"{variant} diff p drifted: {round(_sign_p(gup, gdn), 4)} vs {p}"

    rows = []
    for name in SCALES:
        f = d.copy()
        f["scale"] = name
        rows.append(f)
    df = pd.concat(rows, ignore_index=True)

    ann = []
    for name in SCALES:
        for (variant, role), (m, up, dn) in stats.items():
            ann.append({"scale": name, "variant": variant, "role": role,
                        "prop": m / LN2, "up": up, "dn": dn,
                        "txt": f"{role}   {m / LN2:.2%}   {dn} of {up + dn} pairs up"
                               if False else
                               f"{role}   {m / LN2:.2%}   {up}/{dn} pairs positive"})
    a = pd.DataFrame(ann)
    a = a[a.scale == "zoomed (same points)"].copy()
    #: numeric y throughout. Mapping the discrete `role` to y while also
    #: applying scale_y_continuous raises "Discrete value supplied to
    #: continuous scale", and the annotations need fractional offsets
    #: from the strips anyway.
    a["ypos"] = a.role.map({"base": 1.0, "aligned": 2.0})
    a["ylab"] = a.ypos + 0.34

    df["ypos"] = df.role.map({"base": 1.0, "aligned": 2.0})
    ref = pd.DataFrame([{"scale": "at the scale of the imposition", "variant": v,
                         "x": 1.0, "txt": "100% = the whole imposition\nreaches the continuation"}
                        for v in d.variant.unique()])

    for f in (df, a, ref):
        f["variant"] = pd.Categorical(
            f.variant, categories=["all arm words", "single-token only"], ordered=True)
        f["scale"] = pd.Categorical(f.scale, categories=list(SCALES), ordered=True)


    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.4)
        + geom_vline(ref, aes(xintercept="x"), color="#b03030", linetype="dashed",
                     size=0.5)
        + geom_text(ref, aes("x", 2.45, label="txt"), size=6.2, ha="right",
                    va="center", color="#b03030", nudge_x=-0.02, lineheight=1.2)
        + geom_jitter(df, aes("prop", "ypos", color="role"), height=0.13, width=0,
                      size=1.5, alpha=0.55)
        + geom_point(a, aes("prop", "ypos", color="role"), size=4.2, shape="D")
        + geom_text(a, aes(0.0435, "ylab", label="txt", color="role"), size=6.5,
                    ha="right", va="center")
        + scale_color_manual(values={"base": "#8a8a8a", "aligned": "#1f4e79"})
        + scale_x_continuous(labels=lambda bs: [f"{b:.0%}" for b in bs])
        + scale_y_continuous(limits=(0.55, 2.75))
        + facet_grid("variant ~ scale", scales="free_x")
        + labs(
            title="Forcing an improbable word does damage the chain, and about 99% of it is absorbed",
            subtitle=(
                "One point per model pair (40 pairs, 601,324 rows, 9,423 fitted cells). Position is the "
                "share of the opening's improbability that reaches the continuation: the fitted slope in "
                "nats-per-bit, divided by ln 2.\n"
                "SAME POINTS AT BOTH SCALES. Left, against the imposition itself, where the entire result "
                "sits on the axis. Right, zoomed, where the per-pair spread and the sign counts are legible. "
                "Neither panel alone is honest.\n"
                "THE ROLE DIFFERENCE IS NOT AN ALIGNMENT EFFECT and is not drawn as one: aligned minus base "
                "is +0.0039 (p 0.081) pooled and +0.0029 (p 0.039) single-token, the two variants disagree "
                "across 0.05, and the difference is worth 0.003 nats-per-bit, a third of a percent of "
                "propagation.\n"
                "SCOPE: this panel measures propagation under an IMPOSED word only. It makes no comparison "
                "to undisturbed generation and none should be read into it.\n"
                "Single pass, ungraded, [5503] applies."),
            x="share of the opening's improbability reaching the continuation",
            y="",
            caption=("Producer: meta/M06_generation/scripts/m06_propagation_figs.py from "
                     "results/propagation_cells.parquet (producer m06_propagation.py), post-repair at "
                     "22aee418.\n"
                     "The finding's \"~1.3%\" is a prose approximation; recomputed from the committed "
                     "per-pair slopes the medians are 1.20% aligned and 1.05% base. Nothing turns on it: "
                     "the claim is roughly 99% absorbed either way."),
        )
        + theme_minimal()
        + theme(figure_size=(13.0, 6.6),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.2, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                strip_text=element_text(size=8.0, weight="bold"),
                legend_position="none",
                panel_spacing=0.05)
    )
    out = os.path.join(FIGURES, "propagation_slope.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for key in sorted(stats, key=str):
        m, up, dn = stats[key]
        print(f"    {key[0]:18s} {key[1]:8s} {m:+.5f} nats/bit = {m / LN2:6.2%}  {up}/{dn}")
    return out


FIGURES_REGISTRY = {"slope": slope}


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
