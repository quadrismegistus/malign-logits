#!/usr/bin/env python
"""All capacities on one page, per ladder (RH, 2026-08-14).

    uv run python meta/M05_emergence/scripts/m05_capacities_overview.py            # both
    uv run python meta/M05_emergence/scripts/m05_capacities_overview.py olmo

One line per capacity family across the full ladder, from
capacities_by_rung.parquet (aggregate_capacities.py) and nothing else.
Per-family measure (each is a probability/mass/share in [0,1], stated
in the legend label):

  reference / reasoning / discourse / packages   mean p(target)
  poetic                                         mean p(target)
  verse rhyme                                    called-slot class pull
                                                 (copy excluded), eras
                                                 averaged equally
  sense                                          natural-reading share

PANEL is not a capacity family and stays off. verse_unrhymed is a
control (~0 throughout) and stays off. The measures are not the same
instrument — the page is for SHAPE comparison (onsets, plateaus, what
post-training does), not for reading one family's level against
another's; the subtitle says so.

fig26_olmo_all_capacities.png / fig27_pythia_all_capacities.png
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

FIGDIR = "meta/M05_emergence/figures"
AGG = "meta/M05_emergence/results/capacities_by_rung.parquet"
INK, INK2 = "#0b0b0b", "#52514e"

# house palette for the battery (m05_capacity_prob), extensions for the rest
COLORS = {"reference (facts)": "#2a78d6", "reasoning": "#1baf7a",
          "discourse tracking": "#eb6834", "semantic packages": "#4a3aa7",
          "poetic (hold the word)": "#c2477f",
          "verse rhyme (pull)": "#8a6d00",
          "sense (natural share)": "#4f7d70",
          "syntax (strict licit share)": "#8f5fbf"}
LABEL = {"capacity_reference": "reference (facts)",
         "capacity_reasoning": "reasoning",
         "capacity_discourse": "discourse tracking",
         "capacity_packages": "semantic packages",
         "poetic": "poetic (hold the word)"}


def build(ladder):
    d = pd.read_parquet(AGG)
    d = d[d.ladder == ladder]
    rows = []
    b = d[d.family.isin(LABEL) & (d.measure == "mean_p_target")]
    for r in b.itertuples():
        rows.append(dict(ckpt_idx=r.ckpt_idx, fam=LABEL[r.family],
                         value=r.value, role=r.role, stage=r.stage))
    v = d[d.family.isin(["verse_rhymed_pre-1900", "verse_rhymed_1900+"])
          & (d.measure == "called_pull")]
    for ck, g in v.groupby("ckpt_idx"):
        rows.append(dict(ckpt_idx=ck, fam="verse rhyme (pull)",
                         value=g.value.mean(), role=g.role.iloc[0],
                         stage=g.stage.iloc[0]))
    s = d[(d.family == "sense") & (d.measure == "natural_share")]
    for r in s.itertuples():
        rows.append(dict(ckpt_idx=r.ckpt_idx, fam="sense (natural share)",
                         value=r.value, role=r.role, stage=r.stage))
    x = d[(d.family == "syntax") & (d.measure == "strict_licit_share")]
    for r in x.itertuples():
        rows.append(dict(ckpt_idx=r.ckpt_idx,
                         fam="syntax (strict licit share)",
                         value=r.value, role=r.role, stage=r.stage))
    return pd.DataFrame(rows).sort_values("ckpt_idx"), d


def segment(role, stage):
    """Smoothing segment: no window may cross a pretraining-stage or
    post-training boundary. base_endpoint rides with stage3 (it is the
    end of that run); each post-training phase is its own segment."""
    r, s = str(role), str(stage)
    if r == "base_step":
        return s
    if r == "base_endpoint":
        return "stage3"
    for ph in ("sft", "dpo", "rlvr"):
        if r.startswith(ph):
            return ph
    return "other"


def smooth(long, window):
    long = long.copy()
    long["seg"] = [segment(r, s) for r, s in zip(long.role, long.stage)]
    long["smooth"] = (long.sort_values("ckpt_idx")
                      .groupby(["fam", "seg"])["value"]
                      .transform(lambda v: v.rolling(window, center=True,
                                                     min_periods=1).mean()))
    return long


def sections(d):
    """ckpt bounds per ladder phase, from role."""
    r = d[["ckpt_idx", "role"]].drop_duplicates()
    phase = r.role.map(lambda x: "BASE" if str(x).startswith("base")
                       else ("SFT" if str(x).startswith("sft")
                             else ("DPO" if str(x).startswith("dpo")
                                   else ("RLVR" if str(x).startswith("rlvr")
                                         else None))))
    r = r.assign(phase=phase).dropna(subset=["phase"])
    return {p: (g.ckpt_idx.min(), g.ckpt_idx.max())
            for p, g in r.groupby("phase")}


def draw(ladder, out):
    long, d = build(ladder)
    long = smooth(long, window=9 if ladder == "pythia" else 5)
    bounds = sections(d)
    from plotnine import (aes, annotate, element_blank, element_line,
                          element_text, geom_line, ggplot, labs,
                          scale_color_manual, theme, theme_minimal, ylim)
    fams = [f for f in COLORS if f in set(long.fam)]
    long["seg_group"] = long.fam + "|" + long.seg
    p = (ggplot(long, aes("ckpt_idx", color="fam"))
         + geom_line(aes(y="value", group="fam"), size=0.45, alpha=0.28))
    # data-driven top: a fixed 0.78 cap silently CLIPPED the sense curve
    # (saturates ~0.92) out of fig27's first render — plotnine ylim drops
    # out-of-range points rather than cropping the view.
    top = float(long.value.max()) + 0.05
    for sec, fill in (("SFT", "#efece4"), ("DPO", "#e7e2d5"),
                      ("RLVR", "#efece4")):
        if sec in bounds:
            lo, hi = bounds[sec]
            p = p + annotate("rect", xmin=lo - 0.5, xmax=hi + 0.5,
                             ymin=-0.02, ymax=top, fill=fill, alpha=0.6)
    w = 9 if ladder == "pythia" else 5
    # a 1-rung segment (DPO) has a smoothed value (min_periods=1) but
    # cannot render as a LINE — draw those as points so DPO shows.
    seg_n = long.groupby("seg_group").seg_group.transform("size")
    singles = long[seg_n == 1]
    # right-edge direct labels, one per family, simple vertical repel
    xmax = int(long.ckpt_idx.max())
    last = (long.sort_values("ckpt_idx").groupby("fam", as_index=False)
            .tail(1)[["fam", "smooth"]].sort_values("smooth"))
    ys, gap = [], top * 0.034
    for y in last.smooth:
        ys.append(y if not ys else max(y, ys[-1] + gap))
    last = last.assign(x=xmax + max(1, int(xmax * 0.015)), ylab=ys)
    from plotnine import geom_point, geom_text, scale_x_continuous
    p = (p + geom_line(aes(y="smooth", group="seg_group"), size=1.1)
         + geom_point(data=singles, mapping=aes(y="smooth"), size=1.8,
                      show_legend=False)
         + geom_text(data=last,
                     mapping=aes(x="x", y="ylab", label="fam",
                                 color="fam"),
                     ha="left", size=8, fontweight="bold",
                     show_legend=False)
         + scale_x_continuous(expand=(0, 0, 0.30, 0))
         + scale_color_manual([COLORS[f] for f in fams], limits=fams)
         + ylim(-0.02, top)
         + labs(x="training position (ordinal ladder)",
                y="per-family measure (probability / mass / share)",
                title=f"All capacities across the "
                      f"{'full OLMo-3 ladder' if ladder == 'olmo' else 'Pythia ladder'}",
                subtitle=("One measure per family (legend); all in [0,1] "
                          "but NOT one instrument — compare shapes "
                          "(onsets, plateaus, post-training effects), "
                          "not levels across families.\nBold: centered "
                          f"moving average (window {w}) computed WITHIN "
                          "phase segments only (stage1/2/3, SFT, DPO, "
                          "RLVR) — no window crosses a boundary; faint: "
                          "raw rungs.\nSource: capacities_by_rung."
                          "parquet; verse = called pull, eras averaged; "
                          "unrhymed control ~0 omitted."),
                color="")
         + theme_minimal(base_size=11)
         + theme(panel_grid_minor=element_blank(),
                 panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                 text=element_text(color=INK),
                 plot_title=element_text(size=13, weight="bold"),
                 plot_subtitle=element_text(size=8, color=INK2),
                 legend_position="none",
                 figure_size=(11, 6.4)))
    for sec in bounds:
        lo, hi = bounds[sec]
        p = p + annotate("text", x=(lo + hi) / 2, y=top - 0.015,
                         label=sec, color=INK2, size=9)
    p.save(out, dpi=300, verbose=False)
    print(f"wrote {out}")


REG = {"olmo": lambda: draw("olmo",
                            os.path.join(FIGDIR,
                                         "fig26_olmo_all_capacities.png")),
       "pythia": lambda: draw("pythia",
                              os.path.join(FIGDIR,
                                           "fig27_pythia_all_capacities.png"))}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(REG)):
        if k not in REG:
            sys.exit(f"unknown target {k!r}; have {list(REG)}")
        REG[k]()
