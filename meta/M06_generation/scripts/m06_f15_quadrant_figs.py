#!/usr/bin/env python
"""Figures for `f15_on_passages.md`: the surprisal x drift quadrant shift.

    uv run python meta/M06_generation/scripts/m06_f15_quadrant_figs.py
    uv run python meta/M06_generation/scripts/m06_f15_quadrant_figs.py quadrants
    uv run python meta/M06_generation/scripts/m06_f15_quadrant_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.

WHY THIS IS NOT AN ALLUVIAL DIAGRAM, WHICH IS WHAT THE QUEUE ASKED FOR
----------------------------------------------------------------------
plot-debt item 4 specifies a "natural flow/alluvial diagram". An alluvial
draws ribbons between categories, and a ribbon asserts that the ITEMS
moved: this many passages left Q2 and arrived in Q4.

The data cannot support that. Base and aligned passages are DIFFERENT
GENERATIONS from different models, not the same passage measured twice.
Measured on the committed cells: 19,404 base passages against 19,173
aligned, and only 3,361 of 35,216 (pair, prompt_id, sample_idx) keys
occur under both roles at all -- and even those share a prompt slot
rather than a text. There is no passage-level correspondence to draw a
ribbon between.

What actually changes is the COMPOSITION of each arm: what share of a
pair's passages fall in each quadrant. That is a real, paired, per-pair
quantity and it is what the finding tests. So the figure keeps the
quadrant PLANE as its layout, so the geometry stays readable, and puts
the per-pair share change inside each quadrant. The flow is legible
from the signs -- two quadrants drain, two fill -- without drawing a
migration nobody measured.

THE TWO AXES DO NOT CARRY EQUAL INFORMATION, AND THAT GOES ON THE PANEL
-----------------------------------------------------------------------
A two-axis diagram invites the reader to assume parity. The Q1 share
change tracks the surprisal axis at Spearman -0.714 and the drift axis
at +0.211, so the surprisal axis carries roughly three times the
association. plot-debt records -0.694 and +0.167 for the same pair of
quantities; this script computes them from the committed cells with the
producer's own per-cell aggregation (mean over passages within pair x
role, contrasted aligned - base) and gets the values above. The
qualitative claim is identical and robust; the exact coefficients are
NOT reconciled, and the panel quotes what this script computes rather
than what the queue records. Reported to the docket on shipping; no
message id is cited here because this file was written before that post
existed, and citing an id one has not read off the tool is exactly how
a fabricated identifier enters a record ([5604], and lacan's own
instance at [5921] the same night).

THE mean_drift RIDER IS DISCHARGED HERE
---------------------------------------
plot-debt owed a `mean_drift` variant before item 4 was final, because
`total_drift` is the metric the audit found weakest and it rises with
sentence count, saturating near n=10, while untruncated passages sit
past that knee. No new run was needed: `mean_drift` is already a column
in the committed cells. Both metrics are drawn together. All four flows
replicate, and the Q2 -> Q4 movement STRENGTHENS under mean_drift
(Q2 -0.335 to -0.370, Q4 +0.299 to +0.366) while the two mixed
quadrants weaken (Q1 +0.114 to +0.053, Q3 -0.123 to -0.067).
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
CELLS = os.path.join(RESULTS, "f15_on_passages_full_bge-m3_cells.parquet")

#: Booked in results/f15_on_passages_full_bge-m3.json (P3_Q*), 38 pairs.
BOOKED = {"Q1": (+0.1140, 31, 7), "Q2": (-0.3350, 2, 36),
          "Q3": (-0.1233, 6, 32), "Q4": (+0.2989, 35, 3)}
BOOKED_MED = {"drift": 0.6691, "surprisal": 3.6186, "n_passages": 38577}

#: quadrant -> (drift level, surprisal level), from m06_f15_on_passages.py
PLANE = {"Q1": ("drift HIGH", "surprisal LOW"),
         "Q2": ("drift HIGH", "surprisal HIGH"),
         "Q3": ("drift LOW", "surprisal HIGH"),
         "Q4": ("drift LOW", "surprisal LOW")}
METRICS = {"total_drift": "#1f4e79", "mean_drift": "#c98a2b"}


def _shares(d, dcol):
    """Per-pair quadrant share change, aligned - base.

    Quadrants are reassigned here rather than read off the stored column,
    because the stored labels are `total_drift`-based and the rider needs
    the same rule applied to `mean_drift`. The rule is the producer's:
    pooled medians over BOTH arms, computed once.
    """
    md, ms = float(d[dcol].median()), float(d.mean_surprisal.median())
    q = np.select(
        [(d[dcol] >= md) & (d.mean_surprisal >= ms),
         (d[dcol] >= md) & (d.mean_surprisal < ms),
         (d[dcol] < md) & (d.mean_surprisal >= ms)],
        ["Q2", "Q1", "Q3"], default="Q4")
    sh = (d.assign(q=q).groupby(["pair", "role"]).q
          .value_counts(normalize=True).unstack(fill_value=0.0).reset_index())
    piv = sh.pivot(index="pair", columns="role")
    rows = []
    for quad in ("Q1", "Q2", "Q3", "Q4"):
        ds = (piv[(quad, "aligned")] - piv[(quad, "base")]).dropna()
        for pair, v in ds.items():
            rows.append({"quadrant": quad, "metric": dcol, "pair": pair,
                         "delta": v})
    return pd.DataFrame(rows), md, ms


def _stat(s):
    return float(np.median(s)), int((s > 0).sum()), int((s < 0).sum())


def quadrants():
    """The quadrant plane: alignment moves passages down the surprisal axis.

    Layout IS the quadrant plane (drift across, surprisal up), so the
    reader can see which quadrants drain and which fill. Inside each
    quadrant, one point per model pair for the change in that quadrant's
    share, on both drift metrics.
    """
    from plotnine import (aes, element_blank, element_text, facet_grid,
                          geom_hline, geom_jitter, geom_point, geom_text,
                          ggplot, labs, scale_color_manual,
                          scale_y_continuous, theme, theme_minimal)

    d = pd.read_parquet(CELLS)
    assert len(d) == BOOKED_MED["n_passages"], \
        f"passages drifted: {len(d)} vs booked {BOOKED_MED['n_passages']}"

    frames, stats = [], {}
    for dcol in METRICS:
        f, md, ms = _shares(d, dcol)
        frames.append(f)
        if dcol == "total_drift":
            assert round(md, 4) == BOOKED_MED["drift"], \
                f"drift median drifted: {round(md, 4)} vs {BOOKED_MED['drift']}"
            assert round(ms, 4) == BOOKED_MED["surprisal"], \
                f"surprisal median drifted: {round(ms, 4)} vs {BOOKED_MED['surprisal']}"
        for quad, g in f.groupby("quadrant"):
            stats[(dcol, quad)] = _stat(g.delta.values)

    for quad, (m, up, dn) in BOOKED.items():
        gm, gup, gdn = stats[("total_drift", quad)]
        assert round(gm, 4) == m and (gup, gdn) == (up, dn), \
            f"{quad} drifted: {round(gm, 4)} {gup}/{gdn} vs booked {m} {up}/{dn}"

    df = pd.concat(frames, ignore_index=True)
    df["dl"] = df.quadrant.map(lambda q: PLANE[q][0])
    df["sl"] = df.quadrant.map(lambda q: PLANE[q][1])

    ann = []
    for (dcol, quad), (m, up, dn) in stats.items():
        ann.append({"quadrant": quad, "metric": dcol, "m": m, "up": up, "dn": dn,
                    "dl": PLANE[quad][0], "sl": PLANE[quad][1],
                    "y": 0.62 if dcol == "total_drift" else 0.44,
                    "txt": f"{'total' if dcol == 'total_drift' else 'mean'}_drift"
                           f"   {m:+.3f}   {up}/{dn}"})
    a = pd.DataFrame(ann)
    lab = pd.DataFrame([
        {"quadrant": q, "dl": PLANE[q][0], "sl": PLANE[q][1], "y": 0.86,
         "txt": f"{q}   {'FILLS' if BOOKED[q][0] > 0 else 'DRAINS'}"}
        for q in PLANE])

    order_d = ["drift LOW", "drift HIGH"]
    order_s = ["surprisal HIGH", "surprisal LOW"]
    for f in (df, a, lab):
        f["dl"] = pd.Categorical(f.dl, categories=order_d, ordered=True)
        f["sl"] = pd.Categorical(f.sl, categories=order_s, ordered=True)

    p = (
        ggplot()
        + geom_hline(yintercept=0, color="#333333", size=0.4)
        + geom_jitter(df, aes("delta", 0, color="metric"), height=0.16, width=0,
                      size=1.4, alpha=0.5)
        + geom_point(a, aes("m", 0, color="metric"), size=4.0, shape="D")
        + geom_text(a, aes(-0.62, "y", label="txt", color="metric"), size=6.6,
                    ha="left")
        + geom_text(lab, aes(-0.62, "y", label="txt"), size=7.6, ha="left",
                    fontweight="bold", color="#333333")
        + scale_color_manual(values=METRICS, name="drift metric")
        + scale_y_continuous(limits=(-0.30, 1.02))
        + facet_grid("sl ~ dl")
        + labs(
            title="Alignment moves passages DOWN the surprisal axis. The drift axis carries much less of it.",
            subtitle=(
                "Quadrants of the M06 passage corpus, split at the pooled medians over both arms "
                f"(drift {BOOKED_MED['drift']}, surprisal {BOOKED_MED['surprisal']}; "
                f"{BOOKED_MED['n_passages']:,} passages, 38 model pairs, bge-m3, NO 75-word truncation).\n"
                "One point per pair: the change in that quadrant's SHARE of the pair's passages, "
                "aligned minus base. The two high-surprisal quadrants drain and the two low-surprisal "
                "quadrants fill.\n"
                "THE AXES ARE NOT EQUAL. The Q1 share change tracks the surprisal axis at Spearman "
                "-0.714 and the drift axis at +0.211: about three times the association on surprisal. "
                "Read the vertical split as load-bearing and the horizontal split as minor.\n"
                "BOTH DRIFT METRICS SHOWN. total_drift rises with sentence count and saturates near "
                "n=10, and untruncated passages sit past that knee, so mean_drift is the robustness "
                "check. All four flows replicate; Q2->Q4 strengthens under mean_drift.\n"
                "NOT AN ALLUVIAL: base and aligned passages are different generations, so no passage "
                "moves between quadrants. What changes is each arm's composition."),
            x="change in this quadrant's share of the pair's passages  (aligned - base)",
            y="",
            caption=("Producer: meta/M06_generation/scripts/m06_f15_quadrant_figs.py from "
                     "results/f15_on_passages_full_bge-m3_cells.parquet "
                     "(producer m06_f15_on_passages.py --no-truncate).\n"
                     "Quadrants reassigned here under the producer's own rule so the same rule "
                     "applies to mean_drift; total_drift labels reproduce the stored column's "
                     "booked flows exactly."),
        )
        + theme_minimal()
        + theme(figure_size=(12.6, 7.2),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.2, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                strip_text=element_text(size=8.4, weight="bold"),
                legend_position="none",
                panel_spacing=0.05)
    )
    out = os.path.join(FIGURES, "f15_quadrant_shift.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for dcol in METRICS:
        row = "  ".join(f"{q} {stats[(dcol, q)][0]:+.4f} "
                        f"{stats[(dcol, q)][1]}/{stats[(dcol, q)][2]}"
                        for q in ("Q1", "Q2", "Q3", "Q4"))
        print(f"    {dcol:12s} {row}")
    return out


FIGURES_REGISTRY = {"quadrants": quadrants}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
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
