#!/usr/bin/env python
"""Verse capacity figures (registry, plot_*_figs convention).

    uv run python meta/M05_emergence/scripts/verse_capacity_figs.py            # all
    uv run python meta/M05_emergence/scripts/verse_capacity_figs.py vc_olmo    # one

vc_olmo: rhyme capacity across the FULL OLMo-3 ladder (base | SFT | DPO |
RLVR), colored by era — Victorian-and-earlier (pre-1900) against modern
(1900+). fig15b's idiom (m05_capacity_prob.py): ordinal training
position, shaded post-training regions, solid = the target quantity,
dashed = its control. Here solid is called-slot rime-class mass (minus
the partner word itself — pull, not copy) and dashed is the
depth-matched null {mid4, near} ([5751]/[5753]: the only across-slot
contrast valid raw). Rhymed poems only; the unrhymed floor is flat zero
and stays off this panel (verse_capacity.py prints it).

Source: verse_capacity_rungs.parquet (producer verse_capacity.py).
Closure decomposition not available (rider rides the un-ingested .f16).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

FIGDIR = "meta/M05_emergence/figures"
RUNGS = "meta/M05_emergence/results/verse_capacity_rungs.parquet"

VIOLET, ORANGE = "#4a3aa7", "#eb6834"   # house palette (m05_capacity_prob)
INK, INK2 = "#0b0b0b", "#52514e"
ERA_LAB = {"pre-1900": "Victorian & earlier (pre-1900)",
           "1900+": "modern (1900+)"}
ERA_COL = {"pre-1900": VIOLET, "1900+": ORANGE}


def olmo_position(model):
    """(section, section_order, sort_key) over the full ladder."""
    m = re.search(r"1025-7B@stage(\d+)-step(\d+)", model)
    if m:
        return ("BASE", 0, (int(m.group(1)), int(m.group(2))))
    if model.endswith("Olmo-3-1025-7B"):
        return ("BASE", 0, (9, 0))                     # base endpoint
    m = re.search(r"Think-SFT@step(\d+)", model)
    if m:
        return ("SFT", 1, (0, int(m.group(1))))
    if model.endswith("Think-SFT"):
        return ("SFT", 1, (9, 0))
    if "Think-DPO" in model:
        return ("DPO", 2, (0, 0))
    m = re.search(r"Think@step_?(\d+)", model)
    if m:
        return ("RLVR", 3, (0, int(m.group(1))))
    if model.endswith("Think"):
        return ("RLVR", 3, (9, 0))
    return (None, 9, (9, 9))


def vc_olmo():
    s = pd.read_parquet(RUNGS)
    s = s[(s.ladder == "olmo") & s.rhymed].copy()
    pos = s.model.map(olmo_position)
    s["section"] = [p[0] for p in pos]
    s["order"] = [(p[1],) + p[2] for p in pos]
    s = s[s.section.notna()].sort_values("order")
    ladder = (s[["model", "section", "order"]].drop_duplicates("model")
              .sort_values("order").reset_index(drop=True))
    ladder["x"] = ladder.index
    s = s.merge(ladder[["model", "x"]], on="model")
    s["pull"] = s.called_mean - s.copy_called_mean

    long = pd.concat([
        s.assign(y=s.pull, kind="called slot (pull)"),
        s.assign(y=s.null_mean, kind="depth-matched null"),
    ])
    long["era_lab"] = long.era.map(ERA_LAB)

    bounds = {sec: (g.x.min(), g.x.max())
              for sec, g in ladder.groupby("section")}
    cens = s[s.section == "BASE"].censored_called_mean.mean()

    from plotnine import (aes, annotate, element_blank, element_line,
                          element_text, geom_line, ggplot, labs,
                          scale_color_manual, scale_linetype_manual,
                          theme, theme_minimal, ylim)
    p = (ggplot(long, aes("x", "y", color="era_lab", linetype="kind",
                          group="era_lab + kind"))
         + annotate("rect", xmin=bounds["SFT"][0] - 0.5,
                    xmax=bounds["SFT"][1] + 0.5, ymin=-0.02, ymax=0.72,
                    fill="#efece4", alpha=0.6)
         + annotate("rect", xmin=bounds["DPO"][0] - 0.5,
                    xmax=bounds["DPO"][1] + 0.5, ymin=-0.02, ymax=0.72,
                    fill="#e7e2d5", alpha=0.6)
         + annotate("rect", xmin=bounds["RLVR"][0] - 0.5,
                    xmax=bounds["RLVR"][1] + 0.5, ymin=-0.02, ymax=0.72,
                    fill="#efece4", alpha=0.6)
         + geom_line(size=1.1)
         + scale_color_manual(list(ERA_COL.values()),
                              limits=list(ERA_LAB.values()))
         + scale_linetype_manual(["solid", "dashed"],
                                 limits=["called slot (pull)",
                                         "depth-matched null"])
         + ylim(-0.02, 0.72)
         + labs(x="training position (base | SFT | DPO | RLVR), ordinal",
                y="rime-class mass at slot (copy excluded)",
                title="Rhyme capacity across the full OLMo-3 ladder, "
                      "by era of the poem",
                subtitle=("Solid: called-slot class pull (partner word "
                          "excluded). Dashed: depth-matched null {mid4, "
                          "near} — the only raw-valid contrast "
                          "([5751]/[5753]).\nRhymed poems only; the "
                          "unrhymed floor is zero throughout. Mean over "
                          "poems per rung; mean censored share at called "
                          f"slots {cens:.2f} (theta=0.001).\nClosure "
                          "decomposition awaits the .f16 tier."),
                color="", linetype="")
         + theme_minimal(base_size=11)
         + theme(panel_grid_minor=element_blank(),
                 panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                 text=element_text(color=INK),
                 plot_title=element_text(size=13, weight="bold"),
                 plot_subtitle=element_text(size=8, color=INK2),
                 legend_position="bottom",
                 figure_size=(11, 6.2)))
    for sec in ("BASE", "SFT", "DPO", "RLVR"):
        lo, hi = bounds[sec]
        p = p + annotate("text", x=(lo + hi) / 2, y=0.70, label=sec,
                         color=INK2, size=9)
    out = os.path.join(FIGDIR, "fig24_verse_capacity_olmo_era.png")
    p.save(out, dpi=300, verbose=False)
    print(f"wrote {out}")


REGISTRY = {"vc_olmo": vc_olmo}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(REGISTRY)):
        if k not in REGISTRY:
            sys.exit(f"unknown figure {k!r}; have: {list(REGISTRY)}")
        REGISTRY[k]()
