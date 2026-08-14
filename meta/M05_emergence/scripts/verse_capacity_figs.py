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
                          "decomposition awaits the .f16 tier "
                          "(data/raw/verse_fleet, collected and NOT "
                          "ingested per [5886]; if migrated, on the "
                          "external volume behind that path)."),
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


def vc_olmo_scheme():
    """Same ladder, colored by SCHEME (AABB / ABAB / unrhymed) — the
    unrhymed arm drawn as a first-class curve: its called-slot pull is
    the COMPULSION measure (does the model impose rhyme where nothing
    calls it), so this panel shows capacity and compulsion on one page.
    Eras pooled. Computed at scheme grain from the cells parquet (the
    rung summary only carries the rhymed/unrhymed cut)."""
    import numpy as np
    d = pd.read_parquet(
        "meta/M05_emergence/results/verse_capacity_cells.parquet")
    pos = d.model.map(olmo_position)
    d = d[[p[0] is not None and "Olmo" in m
           for p, m in zip(pos, d.model)]].copy()
    pos = d.model.map(olmo_position)
    d["section"] = [p[0] for p in pos]
    d["order"] = [(p[1],) + p[2] for p in pos]
    d["pull"] = d.tclass - d.p_target_word

    rows = []
    for (model, scheme), g in d.groupby(["model", "scheme"]):
        by = {s: h.set_index("id_human") for s, h in g.groupby("slot")}
        if "called" not in by:
            continue
        called = by["called"].pull
        nulls = []
        for pm in called.index:
            m4 = by["mid4"].pull.get(pm, np.nan) if "mid4" in by else np.nan
            nr = by["near"].pull.get(pm, np.nan) if "near" in by else np.nan
            coll = (by["near"].collides.get(pm, "None")
                    if "near" in by else "None")
            nulls.append(m4 if coll not in ("None", "nan", "")
                         else np.nanmean([m4, nr]))
        rows.append(dict(model=model, scheme=scheme,
                         called=float(called.mean()),
                         null=float(np.nanmean(nulls))))
    s = pd.DataFrame(rows)
    pos = s.model.map(olmo_position)
    s["section"] = [p[0] for p in pos]
    s["order"] = [(p[1],) + p[2] for p in pos]
    ladder = (s[["model", "order", "section"]].drop_duplicates("model")
              .sort_values("order").reset_index(drop=True))
    ladder["x"] = ladder.index
    s = s.merge(ladder[["model", "x"]], on="model").sort_values("x")

    long = pd.concat([
        s.assign(y=s.called, kind="called slot (pull)"),
        s.assign(y=s.null, kind="depth-matched null"),
    ])
    SCH_COL = {"AABB": "#1baf7a", "ABAB": "#2a78d6",
               "unrhymed": "#8a8880"}
    bounds = {sec: (g.x.min(), g.x.max())
              for sec, g in ladder.groupby("section")}

    from plotnine import (aes, annotate, element_blank, element_line,
                          element_text, geom_line, ggplot, labs,
                          scale_color_manual, scale_linetype_manual,
                          theme, theme_minimal, ylim)
    p = (ggplot(long, aes("x", "y", color="scheme", linetype="kind",
                          group="scheme + kind"))
         + annotate("rect", xmin=bounds["SFT"][0] - 0.5,
                    xmax=bounds["SFT"][1] + 0.5, ymin=-0.02, ymax=0.62,
                    fill="#efece4", alpha=0.6)
         + annotate("rect", xmin=bounds["DPO"][0] - 0.5,
                    xmax=bounds["DPO"][1] + 0.5, ymin=-0.02, ymax=0.62,
                    fill="#e7e2d5", alpha=0.6)
         + annotate("rect", xmin=bounds["RLVR"][0] - 0.5,
                    xmax=bounds["RLVR"][1] + 0.5, ymin=-0.02, ymax=0.62,
                    fill="#efece4", alpha=0.6)
         + geom_line(size=1.1)
         + scale_color_manual([SCH_COL[k] for k in
                               ("AABB", "ABAB", "unrhymed")],
                              limits=["AABB", "ABAB", "unrhymed"])
         + scale_linetype_manual(["solid", "dashed"],
                                 limits=["called slot (pull)",
                                         "depth-matched null"])
         + ylim(-0.02, 0.62)
         + labs(x="training position (base | SFT | DPO | RLVR), ordinal",
                y="rime-class mass at slot (copy excluded)",
                title="Rhyme capacity vs compulsion across the OLMo-3 "
                      "ladder, by scheme",
                subtitle=("Solid: called-slot class pull (partner word "
                          "excluded); for UNRHYMED poems nothing calls "
                          "the slot, so its solid curve is the "
                          "COMPULSION measure.\nDashed: depth-matched "
                          "null {mid4, near} ([5751]/[5753]). Eras "
                          "pooled; 60 poems per scheme. AABB partner is "
                          "the adjacent line, ABAB two lines back."),
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
        p = p + annotate("text", x=(lo + hi) / 2, y=0.605, label=sec,
                         color=INK2, size=9)
    out = os.path.join(FIGDIR, "fig25_verse_capacity_olmo_scheme.png")
    p.save(out, dpi=300, verbose=False)
    print(f"wrote {out}")


def vc_errors():
    """Signal-detection split of the rhyme capacity (RH's design,
    2026-08-14): MISS = rhymed poem whose called-slot pull fails to
    clear the depth-matched null by margin m; FALSE ALARM = unrhymed
    poem whose would-be-partner class clears it. The margin is the
    honest criterion: bare exceedance is a COIN FLIP at zero signal
    (two near-zero numbers compared pairwise), which would fake a ~50%
    false-alarm rate from noise. m=0.05 primary (bold), m=0.02
    sensitivity (thin dashed). Miss = 100% at init is the correct
    reading (cannot yet). Writes fig28 (OLMo) / fig29 (Pythia) +
    results/verse_error_rates.parquet."""
    import numpy as np
    d = pd.read_parquet(
        "meta/M05_emergence/results/verse_capacity_cells.parquet")
    d["pull"] = d.tclass - d.p_target_word

    rows = []
    for model, g in d.groupby("model"):
        by = {s: h.set_index("id_human") for s, h in g.groupby("slot")}
        if "called" not in by:
            continue
        called = by["called"]
        deltas, rhymed = {}, {}
        for pm in called.index:
            m4 = by["mid4"].pull.get(pm, np.nan) if "mid4" in by else np.nan
            nr = by["near"].pull.get(pm, np.nan) if "near" in by else np.nan
            coll = (by["near"].collides.get(pm, "None")
                    if "near" in by else "None")
            null = (m4 if coll not in ("None", "nan", "")
                    else np.nanmean([m4, nr]))
            if np.isnan(null):
                continue
            deltas[pm] = called.pull[pm] - null
            rhymed[pm] = called.scheme[pm] != "unrhymed"
        dl = pd.Series(deltas)
        rh = pd.Series(rhymed)
        for m in (0.05, 0.02):
            rows.append(dict(
                model=model, margin=m,
                miss=float((dl[rh] < m).mean()),
                false_alarm=float((dl[~rh] > m).mean()),
                n_rhymed=int(rh.sum()), n_unrhymed=int((~rh).sum())))
    R = pd.DataFrame(rows)
    R.to_parquet("meta/M05_emergence/results/verse_error_rates.parquet")
    print(f"rates: {len(R):,} rows")

    from plotnine import (aes, annotate, element_blank, element_line,
                          element_text, geom_line, geom_point, geom_text,
                          ggplot, labs, scale_color_manual,
                          scale_linetype_manual, scale_x_continuous,
                          theme, theme_minimal, ylim)
    EC = {"miss (rhymed, no rhyme)": "#c0392b",
          "false alarm (unrhymed, rhymes)": "#2a78d6"}
    for ladder, fign, window in (("olmo", 28, 5), ("pythia", 29, 9)):
        if ladder == "olmo":
            sub = R[R.model.str.contains("llenai")].copy()
            pos = sub.model.map(olmo_position)
            sub["section"] = [p[0] for p in pos]
            sub["order"] = [(p[1],) + p[2] for p in pos]
            sub = sub[sub.section.notna()]
        else:
            sub = R[R.model.str.contains("pythia")].copy()
            sub["section"] = "BASE"
            sub["order"] = sub.model.map(
                lambda m: int(re.search(r"@step(\d+)$", m).group(1))
                if "@step" in m else 10**9)
        ladder_o = (sub[["model", "order", "section"]]
                    .drop_duplicates("model").sort_values("order")
                    .reset_index(drop=True))
        ladder_o["x"] = ladder_o.index
        sub = sub.merge(ladder_o[["model", "x"]], on="model")
        long = pd.concat([
            sub.assign(y=sub.miss, kind="miss (rhymed, no rhyme)"),
            sub.assign(y=sub.false_alarm,
                       kind="false alarm (unrhymed, rhymes)")])
        long["mlab"] = long.margin.map({0.05: "m=0.05", 0.02: "m=0.02"})
        long["grp"] = long.kind + long.mlab + long.section
        long = long.sort_values("x")
        long["smooth"] = (long.groupby(["kind", "mlab", "section"])["y"]
                          .transform(lambda v: v.rolling(
                              window, center=True, min_periods=1).mean()))
        bounds = {s: (g.x.min(), g.x.max())
                  for s, g in long.groupby("section")}
        xmax = int(long.x.max())
        lab = (long[long.mlab == "m=0.05"].sort_values("x")
               .groupby("kind", as_index=False).tail(1))
        p = ggplot(long, aes("x", color="kind"))
        for sec, fill in (("SFT", "#efece4"), ("DPO", "#e7e2d5"),
                          ("RLVR", "#efece4")):
            if sec in bounds:
                lo, hi = bounds[sec]
                p = p + annotate("rect", xmin=lo - 0.5, xmax=hi + 0.5,
                                 ymin=-0.02, ymax=1.05, fill=fill,
                                 alpha=0.6)
        segsz = long.groupby("grp").grp.transform("size")
        p = (p
             + geom_line(aes(y="y", group="grp", linetype="mlab"),
                         size=0.4, alpha=0.25)
             + geom_line(data=long[long.mlab == "m=0.05"],
                         mapping=aes(y="smooth", group="grp"), size=1.1)
             + geom_line(data=long[long.mlab == "m=0.02"],
                         mapping=aes(y="smooth", group="grp"), size=0.6,
                         alpha=0.6, linetype="dashed")
             + geom_point(data=long[(segsz == 1)
                                    & (long.mlab == "m=0.05")],
                          mapping=aes(y="smooth"), size=1.8,
                          show_legend=False)
             + geom_text(data=lab,
                         mapping=aes(x=xmax + max(1, int(xmax * 0.015)),
                                     y="smooth", label="kind",
                                     color="kind"),
                         ha="left", size=8, fontweight="bold",
                         show_legend=False)
             + scale_x_continuous(expand=(0, 0, 0.34, 0))
             + scale_color_manual([EC[k] for k in EC], limits=list(EC))
             + scale_linetype_manual(["solid", "dashed"],
                                     limits=["m=0.05", "m=0.02"],
                                     guide=None)
             + ylim(-0.02, 1.05)
             + labs(x="training position (ordinal ladder)",
                    y="error rate (share of poems)",
                    title=f"Rhyme error types across the "
                          f"{'OLMo-3' if ladder == 'olmo' else 'Pythia'}"
                          " ladder",
                    subtitle=("MISS: rhymed poem, called-slot pull fails "
                              "to clear the depth-matched null by m. "
                              "FALSE ALARM: unrhymed poem clears it.\n"
                              "Bold m=0.05, dashed m=0.02 (bare "
                              "exceedance is a coin flip at zero signal "
                              "— the margin is the criterion). Smoothed "
                              "within phase segments; faint = raw.\n"
                              "Miss=100% at init is the correct reading "
                              "(cannot yet). 120 rhymed / 60 unrhymed "
                              "poems per rung."),
                    color="")
             + theme_minimal(base_size=11)
             + theme(panel_grid_minor=element_blank(),
                     panel_grid_major=element_line(color="#e8e7e3",
                                                   size=0.4),
                     text=element_text(color=INK),
                     plot_title=element_text(size=13, weight="bold"),
                     plot_subtitle=element_text(size=8, color=INK2),
                     legend_position="none",
                     figure_size=(11, 6.2)))
        for sec in bounds:
            lo, hi = bounds[sec]
            p = p + annotate("text", x=(lo + hi) / 2, y=1.03, label=sec,
                             color=INK2, size=9)
        out = os.path.join(
            FIGDIR, f"fig{fign}_verse_errors_{ladder}.png")
        p.save(out, dpi=300, verbose=False)
        print(f"wrote {out}")


REGISTRY = {"vc_olmo": vc_olmo, "vc_olmo_scheme": vc_olmo_scheme,
            "vc_errors": vc_errors}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(REGISTRY)):
        if k not in REGISTRY:
            sys.exit(f"unknown figure {k!r}; have: {list(REGISTRY)}")
        REGISTRY[k]()
