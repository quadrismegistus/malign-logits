#!/usr/bin/env python
"""Figures for Findings N (N_mass_migration.md), one function per figure.

    uv run python meta/M01_displacement/scripts/plot_n_figs.py            # all
    uv run python meta/M01_displacement/scripts/plot_n_figs.py n_clusters # one or more by name
    uv run python meta/M01_displacement/scripts/plot_n_figs.py --list

Per-letter plotting convention (RH, 2026-08-14, plot-debt regime): one
script per letter in scripts/plot_<letter>_figs.py, a FIGURES registry
mapping name -> function, all figures drawn by default, plotnine at 300
dpi, output to ../figures/. Every figure embeds its slice in the
subtitle.

N's own fence, from the finding's closing line, is the design
constraint here: THE CLUSTER IS THE UNIT (34 base checkpoints), and the
82,775 cells are NOT independent observations. So the paper-facing
figure puts one mark per cluster and never one per cell; cell count
rides on the mark as area, never as position.

WHY THE PRIMARY FIGURE PLOTS A SHARE AND NOT THE PER-CLUSTER z
--------------------------------------------------------------
`result_n_primary.json` carries a per-cluster z, and 33 of its 34
values are the identical float 8.326501160015283. That is not a
measured agreement, it is a ceiling: `n_primary.py::_ppf` inverts the
normal CDF by bisection on `math.erf`, and once the one-sided binomial
p falls below about 1e-16 the CDF evaluates to exactly 0.0 in double
precision and the bisection converges to the same point for every
smaller p. Verified: _ppf returns 8.326501160015283 for p = 1e-17
through p = 1e-300 alike.

A dot plot of that column would stack 33 of 34 dots on one x value and
read as extraordinary uniformity across clusters. The uniformity is
float64. The share of cells running in the registered direction is a
real quantity with real spread (0.690 to 0.993), so that is the
position channel. `n_z_ceiling` below draws the ceiling itself as a
METHOD figure, so the reason is visible rather than only asserted.

NOT A NEW FINDING. The saturation was booked as correction [4134] on
2026-08-04 and then dropped from N_mass_migration.md by the 2026-08-12
self-contained rewrite; it was rediscovered here from the artifact,
blind to [4134], and restored to the doc at 8beecd00 ([5899]). The
general form, in lacan's words at [5898]: a statistic bounded by its
own arithmetic, quoted as if it were an estimate, is a defect even when
the number is correct. Quote the counts. The drawing rule that follows
is the one this script obeys: before a quantity gets a POSITION
channel, ask what its maximum is and whether anything is sitting on it.

Each figure function verifies the finding's booked numbers from the
artifact before drawing and refuses (with a named reason) if they do
not reproduce.
"""
import argparse
import collections
import json
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
FIGURES = os.path.join(M01, "figures")

PRIMARY = os.path.join(RESULTS, "result_n_primary.json")

#: §4 booked values, from N_mass_migration.md and the artifact's own blocks.
BOOKED = {
    "analysed": 82775,
    "clusters": 34,
    "edges": 44,
    "stimuli_en": 2199,
    "n_negative": 75230,
    "n_positive": 7545,
    "n_tie": 0,
    "sign_split_negative": 0.9088492902446391,
    "stouffer_Z": 47.912946087493516,
    "z_ceiling": 8.326501160015283,
}


def _load():
    with open(PRIMARY) as fh:
        return json.load(fh)


def _cluster_table(doc):
    """One row per CLUSTER (base checkpoint), the finding's declared unit.

    Recomputed from `cells` rather than read off a summary block, so the
    figure's own arithmetic is what gets asserted against the booked
    numbers. Ties (tail_excess exactly 0.0) are excluded from the
    denominator per §4's TIES clause, which is why `share` is computed
    over neg + pos and not over the row count.
    """
    agg = collections.defaultdict(lambda: {"neg": 0, "pos": 0, "tie": 0})
    for c in doc["cells"]:
        v = c["tail_excess_corrected"]
        a = agg[c["base"]]
        if v < 0:
            a["neg"] += 1
        elif v > 0:
            a["pos"] += 1
        else:
            a["tie"] += 1
    rows = []
    for base, a in agg.items():
        n = a["neg"] + a["pos"]
        rows.append({
            "cluster": base,
            "short": base.split("/")[-1],
            "neg": a["neg"], "pos": a["pos"], "tie": a["tie"],
            "n": n,
            "share": a["neg"] / n if n else float("nan"),
        })
    d = pd.DataFrame(rows).sort_values("share").reset_index(drop=True)
    return d


def _assert_booked(doc, d):
    pop, corr = doc["_population"], doc["corrected"]
    assert pop["analysed"] == BOOKED["analysed"], \
        f"cells drifted: {pop['analysed']} vs booked {BOOKED['analysed']}"
    assert pop["clusters"] == BOOKED["clusters"] == len(d), \
        f"clusters drifted: {pop['clusters']}/{len(d)} vs booked 34"
    assert pop["edges"] == BOOKED["edges"], \
        f"edges drifted: {pop['edges']} vs booked {BOOKED['edges']}"
    assert pop["stimuli_en"] == BOOKED["stimuli_en"], \
        f"stimuli drifted: {pop['stimuli_en']} vs booked {BOOKED['stimuli_en']}"
    neg, pos = int(d.neg.sum()), int(d.pos.sum())
    assert (neg, pos) == (BOOKED["n_negative"], BOOKED["n_positive"]), \
        f"sign counts drifted: {neg}/{pos} vs booked 75230/7545"
    share = neg / (neg + pos)
    assert abs(share - BOOKED["sign_split_negative"]) < 1e-12, \
        f"sign split drifted: {share!r} vs booked {BOOKED['sign_split_negative']!r}"
    assert abs(corr["sign_split_negative"] - share) < 1e-12, \
        "recomputed split disagrees with the artifact's own corrected block"
    #: the finding's headline is 34/34 AGREE; refuse to draw if one crosses.
    assert (d.share > 0.5).all(), \
        f"34/34 no longer holds: {int((d.share <= 0.5).sum())} cluster(s) at or below 0.5"
    return share


def n_clusters():
    """N: all 34 clusters run in the substitution direction (the anchor).

    Position is the share of non-tied cells with tail_excess < 0, the
    registered direction. Area is the cluster's cell count, which spans
    181 (pythia-2.8b) to 14,585 (Llama-3.1-8B) and is exactly what the
    "cells are not independent" fence is about: the big cluster is one
    vote, and the figure has to show that it is one mark.

    The x axis is held open to the 0.5 null so that "all 34 agree" is
    read against something. Cropping to the data would make a unanimous
    result look like an ordinary spread.
    """
    from plotnine import (aes, element_text, geom_point, geom_text,
                          geom_vline, ggplot, labs, scale_size_area,
                          scale_x_continuous, theme, theme_minimal)

    doc = _load()
    d = _cluster_table(doc)
    share = _assert_booked(doc, d)

    raw_share = doc["raw"]["sign_split_negative"]
    d["short"] = pd.Categorical(d["short"], categories=d["short"].tolist(),
                                ordered=True)
    d["n_label"] = d.n.map(lambda v: f"{int(v):,}")

    sub = (
        "Registration N clause 1, ARM A, ENGLISH ONLY. "
        f"{BOOKED['stimuli_en']:,} stimuli x {BOOKED['edges']} base-to-aligned edges = "
        f"{BOOKED['analysed']:,} cells, grouped into {BOOKED['clusters']} clusters.\n"
        "THE CLUSTER IS THE UNIT: one mark per base checkpoint. The cells are not independent "
        "observations, so cell count is area, never position.\n"
        f"Field tail_excess_corrected (after the §4.1 adversarial push); pooled {share:.1%}. "
        f"Uncorrected tail_excess_raw gives {raw_share:.1%}, so the headline does not turn on the choice.\n"
        "Ties excluded from the denominator per §4's TIES clause."
    )

    p = (
        ggplot(d, aes("share", "short"))
        + geom_vline(xintercept=0.5, linetype="dashed", color="#b03030", size=0.5)
        + geom_vline(xintercept=share, linetype="solid", color="#7f7f7f", size=0.4)
        + geom_point(aes(size="n"), color="#1f4e79", alpha=0.85)
        + geom_text(aes(label="n_label"), size=6, color="#666666",
                    nudge_x=0.018, ha="left")
        + scale_size_area(max_size=9, breaks=[200, 2000, 8000, 14000],
                          labels=lambda bs: [f"{int(b):,}" for b in bs])
        + scale_x_continuous(limits=(0.47, 1.06),
                             breaks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                             labels=lambda bs: [f"{b:.0%}" for b in bs])
        + labs(
            title="All 34 clusters run in the substitution direction",
            subtitle=sub,
            x="share of cells with tail_excess < 0  (substitution direction)",
            y="cluster (base checkpoint)",
            size="cells in cluster",
            caption=("Dashed red = 0.5, the no-preference null. Grey = pooled share. "
                     "Numbers at right are cells per cluster.\n"
                     "Producer: meta/M01_displacement/scripts/plot_n_figs.py from "
                     "results/result_n_primary.json (producer n_primary.py)."),
        )
        + theme_minimal()
        + theme(figure_size=(10.5, 8.6),
                plot_title=element_text(size=13, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.4, color="#444444", ha="left"),
                plot_caption=element_text(size=6.6, color="#666666", ha="left"),
                axis_text_y=element_text(size=7.5),
                axis_text_x=element_text(size=8))
    )
    out = os.path.join(FIGURES, "n_clusters_dotplot.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {len(d)} clusters, pooled {share:.4%}, "
          f"range {d.share.min():.4f} ({d.short.iloc[0]}) to "
          f"{d.share.max():.4f} ({d.short.iloc[-1]})")
    return out


def n_z_ceiling():
    """METHOD figure: why the anchor plots a share and not the per-cluster z.

    33 of 34 cluster z values are the identical float 8.326501160015283.
    `_ppf` inverts the normal CDF by bisection on math.erf, and below
    about p = 1e-16 that CDF is exactly 0.0 in double precision, so
    every smaller p converges to the same point. The Stouffer Z is
    therefore a floor for a NUMERICAL reason in addition to the
    conservative-clustering reason the finding names.

    INTERNAL / method. This documents a property of the instrument, not
    a property of language models, and should not travel as a result.
    """
    from plotnine import (aes, element_text, geom_point, geom_vline, ggplot,
                          labs, scale_x_continuous, theme, theme_minimal)

    doc = _load()
    clusters = doc["corrected"]["clusters"]
    ceiling = BOOKED["z_ceiling"]

    #: reimplement _ppf exactly as n_primary.py defines it, and show it
    #: saturating. If this ever stops saturating the figure is wrong.
    def _ppf(q):
        if q <= 0:
            return -40.0
        if q >= 1:
            return 40.0
        lo, hi = -40.0, 40.0
        for _ in range(300):
            mid = (lo + hi) / 2
            if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    assert abs(abs(_ppf(1e-300)) - ceiling) < 1e-12, \
        "_ppf no longer saturates at the booked ceiling; this figure is stale"
    assert abs(abs(_ppf(1e-17)) - ceiling) < 1e-12, \
        "_ppf(1e-17) no longer equals _ppf(1e-300); the ceiling claim is wrong"

    at_ceiling = sum(1 for v in clusters.values() if abs(v - ceiling) < 1e-12)
    assert at_ceiling == 33, \
        f"{at_ceiling} clusters at the ceiling, booked 33"

    d = pd.DataFrame([{"cluster": k, "short": k.split("/")[-1], "z": v,
                       "at_ceiling": abs(v - ceiling) < 1e-12}
                      for k, v in clusters.items()]).sort_values("z")
    d["short"] = pd.Categorical(d["short"], categories=d["short"].tolist(),
                                ordered=True)
    d["status"] = d.at_ceiling.map({True: "at the float64 ceiling",
                                    False: "below the ceiling"})

    p = (
        ggplot(d, aes("z", "short", color="status"))
        + geom_vline(xintercept=ceiling, linetype="dashed", color="#b03030",
                     size=0.5)
        + geom_point(size=3, alpha=0.85)
        + scale_x_continuous(limits=(4.0, 9.0))
        + labs(
            title="Do not plot the per-cluster z: 33 of 34 values are one float",
            subtitle=(
                f"Every cluster z at {ceiling} is the saturation point of "
                "n_primary.py::_ppf, which inverts the normal CDF by bisection on math.erf.\n"
                "Below about p = 1e-16 that CDF evaluates to exactly 0.0 in double precision, so "
                "p = 1e-17 and p = 1e-300 return the identical z.\n"
                "The apparent agreement across clusters is float64, not evidence. The Stouffer Z of "
                f"{BOOKED['stouffer_Z']:.2f} is a floor for this reason as well as for the conservative-clustering reason.\n"
                "Booked as correction [4134] on 2026-08-04; dropped from N_mass_migration.md in the 2026-08-12 rewrite; "
                "rediscovered from the artifact while drawing this figure and restored to the doc at 8beecd00.\n"
                "The underlying p-values are exact and the verdict is untouched.\n"
                "METHOD FIGURE, INTERNAL: a property of the instrument, not of the models."),
            x="cluster z as stored in result_n_primary.json",
            y="cluster (base checkpoint)",
            color="",
            caption=("Producer: meta/M01_displacement/scripts/plot_n_figs.py. "
                     "Ceiling reproduced by re-running _ppf at p = 1e-17 and p = 1e-300."),
        )
        + theme_minimal()
        + theme(figure_size=(10.5, 8.6),
                plot_title=element_text(size=13, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.4, color="#444444", ha="left"),
                plot_caption=element_text(size=6.6, color="#666666", ha="left"),
                axis_text_y=element_text(size=7.5),
                legend_position="top")
    )
    out = os.path.join(FIGURES, "n_z_ceiling_method.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {at_ceiling}/34 clusters at {ceiling}")
    return out


FIGURES_REGISTRY = {
    "n_clusters": n_clusters,
    "n_z_ceiling": n_z_ceiling,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="figure names; default all")
    ap.add_argument("--list", action="store_true", help="list figure names")
    a = ap.parse_args()

    if a.list:
        for k, fn in FIGURES_REGISTRY.items():
            print(f"  {k:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    names = a.names or list(FIGURES_REGISTRY)
    unknown = [n for n in names if n not in FIGURES_REGISTRY]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(FIGURES_REGISTRY)}", file=sys.stderr)
        return 2

    os.makedirs(FIGURES, exist_ok=True)
    for n in names:
        print(f"{n}:")
        FIGURES_REGISTRY[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
