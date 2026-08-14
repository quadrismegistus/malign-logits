#!/usr/bin/env python
"""The flagship acquisition figure: full ladder, ordinal training position,
all six curves (four capacities, poetic pull, syntax), phase bands.

    uv run python meta/M05_emergence/scripts/m05_acquisition_ladder.py

RH (2026-08-11): the token axis (fig17) hides alignment (no token counts
for SFT/DPO/RLVR) and flattens the syntax result; it keeps the CROSS-
LADDER job only. This figure is the per-ladder object: x = ordinal
training position (the vendor grid), OLMo's full ladder with
BASE | SFT | DPO | RLVR bands, Pythia's base arm. Each curve as share of
its own LATE-BASE value (median of the last 3 base rungs — not the mixed
base_endpoint, per lacan's provenance note), so alignment reads as
deviation from the base ceiling: above 1.0 = alignment raised it.
Coverage gate n>=10 surviving probes; rolling-median(5) display
smoothing; capacity-probe absent rate drawn with the curves ([5436]).

Outputs: figures/fig21_acquisition_ladder_{olmo,pythia}.png — the
sense-carrying version. fig18_* are the frozen pre-sense figures
(RH 2026-08-12: new filenames keep the old versions).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
FIGDIR = "meta/M05_emergence/figures"

CURVES = {"pythia": "data/pythia_curves.parquet",
          "olmo": "data/m05_curves.parquet"}
FAMS = {"CAPACITY_PACKAGES": "semantic packages",
        "CAPACITY_REFERENCE": "reference (facts)",
        "CAPACITY_REASONING": "reasoning",
        "CAPACITY_DISCOURSE": "discourse tracking"}
BLUE, ORANGE, AQUA, MAGENTA, VIOLET, GREEN, GREY = (
    "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7", "#008300",
    "#8a8987")
PAL = {"reference (facts)": BLUE, "reasoning": AQUA,
       "discourse tracking": ORANGE, "semantic packages": VIOLET,
       "poetic pull": MAGENTA, "syntax (licit share)": GREEN,
       "sense (natural share)": "#b02419",
       "probe absent rate": GREY}
INK, INK2 = "#0b0b0b", "#52514e"
EQUIV = [{"ADP", "PART"}, {"NUM", "NOUN"}, {"AUX", "VERB"}]
MIN_N = 10


def expand(classes):
    out = set(classes)
    for g in EQUIV:
        if out & g:
            out |= g
    return out


def fam_medians(df):
    med = {}
    for fam, label in list(FAMS.items()) + [("POETIC", "poetic pull")]:
        g = df[df.curve == fam]
        by = {}
        for r, gg in g.groupby("ckpt_idx"):
            piv = gg.pivot_table(index="probe", columns="word_role",
                                 values="p", aggfunc="first")
            ba = gg.groupby("probe").absent.all()
            piv = piv[~ba.reindex(piv.index, fill_value=False)]
            if len(piv) < MIN_N:
                continue
            if fam == "POETIC":
                if {"formulaic", "paraphrase"} <= set(piv.columns):
                    by[r] = float((piv.formulaic - piv.paraphrase).median())
            elif {"target", "competitor"} <= set(piv.columns):
                by[r] = float(np.log(piv.target / piv.competitor).median())
        med[label] = by
    cap = df[df.curve.str.startswith("CAPACITY")]
    absr = cap.groupby("ckpt_idx").absent.mean().to_dict()
    return med, absr


def syntax_share(ladder):
    cm = pd.read_parquet("data/m05_class_mass.parquet")
    cm = cm[(cm.ladder == ladder) & ~cm.payload_empty]
    lic = {p: expand({w["pos"] for w in v["licit"]})
           for p, v in json.load(open("data/m05_licit_sets.json"))
           ["prompts"].items()}
    recs = []
    for (r, p), g in cm.groupby(["ckpt_idx", "prompt"]):
        res = g.resolved_mass.iloc[0]
        if res <= 0:
            continue
        recs.append((r, g[g.pos_class.isin(lic.get(p, set()))].mass.sum()
                     / res))
    d = pd.DataFrame(recs, columns=["ckpt_idx", "s"])
    return d.groupby("ckpt_idx").s.median().to_dict()


def sense_share(ladder):
    """natural share of classified mass per rung (tier-3 sense verdicts);
    full ladder, same filter as this script's syntax_share."""
    sm = pd.read_parquet("data/m05_sense_mass.parquet")
    sm = sm[(sm.ladder == ladder) & ~sm.payload_empty]
    recs = []
    for (r, p), g in sm.groupby(["ckpt_idx", "prompt"]):
        m = g.set_index("band").mass.to_dict()
        cl = sum(m.get(b, 0.0) for b in
                 ("natural", "odd", "ungrammatical", "not_a_word"))
        if cl <= 0:
            continue
        recs.append((r, m.get("natural", 0.0) / cl))
    d = pd.DataFrame(recs, columns=["ckpt_idx", "s"])
    return d.groupby("ckpt_idx").s.median().to_dict()


def smooth(d):
    """rolling median(5) per (curve, segment): smoothing never crosses a
    base/SFT/DPO/RLVR boundary (RH 2026-08-12)."""
    out = []
    for _, g in d.groupby(["curve", "seg"]):
        g = g.sort_values("ckpt_idx").copy()
        g["v"] = g.v.rolling(5, center=True, min_periods=1).median()
        out.append(g)
    return pd.concat(out)


def main():
    from plotnine import (aes, annotate, coord_cartesian, element_blank,
                          element_line, element_rect, element_text,
                          geom_hline, geom_line, geom_point, geom_smooth,
                          ggplot, labs,
                          scale_color_manual, scale_fill_manual,
                          scale_x_continuous, theme, theme_minimal)
    TH = (theme_minimal(base_size=11) +
          theme(panel_grid_minor=element_blank(),
                panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                text=element_text(color=INK),
                plot_title=element_text(size=13, weight="bold"),
                plot_subtitle=element_text(size=9, color=INK2),
                legend_position="none",
                plot_background=element_rect(fill="#fcfcfb",
                                             color="#fcfcfb"),
                figure_size=(9.5, 5)))

    for ladder in ("olmo", "pythia"):
        df = pd.read_parquet(CURVES[ladder])
        med, absr = fam_medians(df)
        med["syntax (licit share)"] = syntax_share(ladder)
        med["sense (natural share)"] = sense_share(ladder)
        order = (df[["ckpt_idx", "role", "stage", "step"]]
                 .drop_duplicates().sort_values("ckpt_idx"))
        base_rungs = sorted(order[order.role == "base_step"].ckpt_idx)

        rows = []
        for label, by in med.items():
            if not by:
                continue
            late_base = [by[r] for r in base_rungs[-3:] if r in by]
            if not late_base or abs(np.median(late_base)) < 1e-9:
                continue
            ref = float(np.median(late_base))
            for r, v in by.items():
                rows.append(dict(ckpt_idx=r, curve=label, v=v / ref))
        for r, v in absr.items():
            rows.append(dict(ckpt_idx=r, curve="probe absent rate", v=v))
        SEG = {"base_step": "base", "base_endpoint": "base",
               "sft_step": "sft", "sft_endpoint": "sft",
               "dpo_endpoint": "dpo", "rlvr_step": "rlvr"}
        segmap = {r.ckpt_idx: SEG.get(r.role, "base")
                  for r in order.itertuples()}
        raw = pd.DataFrame(rows)
        raw["seg"] = raw.ckpt_idx.map(segmap)
        raw = raw[(raw.v > -0.15) & (raw.v < 1.45)]
        d = smooth(raw)
        d = d[(d.v > -0.15) & (d.v < 1.45)]
        d["grp"] = d.curve + "|" + d.seg

        ends = (d.sort_values("ckpt_idx").groupby("curve")
                .agg(x=("ckpt_idx", "last"), y=("v", "last")))
        labels_y, used = {}, []
        for cname, r in ends.sort_values("y", ascending=False).iterrows():
            yy = float(r.y)
            while any(abs(yy - u) < 0.055 for u in used):
                yy -= 0.058
            used.append(yy)
            labels_y[cname] = yy
        xmax = d.ckpt_idx.max()

        extras = []
        if ladder == "olmo":
            bounds = {}
            for ph, roles in [("BASE", ("base_step", "base_endpoint")),
                              ("SFT", ("sft_step", "sft_endpoint")),
                              ("DPO", ("dpo_endpoint",)),
                              ("RLVR", ("rlvr_step",))]:
                s = order[order.role.isin(roles)]
                if len(s):
                    bounds[ph] = (s.ckpt_idx.min() - .5,
                                  s.ckpt_idx.max() + .5)
            #: all rects first, then all labels — a later rect must not
            #: overpaint an earlier band's label
            for i, (k, v) in enumerate(bounds.items()):
                extras.append(annotate("rect", xmin=v[0], xmax=v[1],
                                       ymin=-np.inf, ymax=np.inf,
                                       fill=("#f2efe6" if i % 2
                                             else "#fcfcfb"),
                                       color="none"))
            for k, v in bounds.items():
                extras.append(annotate("text", x=float(np.mean(v)),
                                       y=1.42 if k != "RLVR" else 1.36,
                                       label=k, size=8, color=INK2))
            title = ("Acquisition and alignment on one axis — OLMo-3, "
                     "full ladder")
            xlab = "training position (base | SFT | DPO | RLVR)"
        else:
            steps = order.set_index("ckpt_idx").step
            blind = [r for r in base_rungs if (steps.get(r) or 0) < 1000]
            if blind:
                extras.append(annotate("rect", xmin=-.5,
                                       xmax=max(blind) + .5, ymin=-np.inf,
                                       ymax=np.inf, fill="#f2efe6",
                                       color="none"))
                extras.append(annotate("text", x=max(blind) + 2, y=1.4,
                                       label="below OLMo's first rung",
                                       color=INK2, size=8, ha="left"))
            title = "Acquisition on the Pythia-6.9b base ladder"
            xlab = "pretraining rung (vendor grid, log-spaced early)"

        LOESS = "--loess" in sys.argv
        SPAN = float(os.environ.get("M05_LOESS_SPAN", "0.5"))
        SE = os.environ.get("M05_LOESS_SE", "1") != "0"
        SE_A = float(os.environ.get("M05_LOESS_SE_ALPHA", "0.15"))
        if LOESS:
            raw["grp"] = raw.curve + "|" + raw.seg
            nseg = raw.groupby("grp").ckpt_idx.transform("nunique")
            big, small = raw[nseg >= 12], raw[nseg < 12]
            smooth_layers = [
                geom_smooth(big, aes(group="grp", fill="curve"),
                            method="loess", span=SPAN, se=SE,
                            alpha=SE_A, size=0.8),
                scale_fill_manual(PAL)]
            if len(small):
                smooth_layers.append(
                    geom_line(small, aes(group="grp"), size=0.8))
        else:
            smooth_layers = [geom_line(d, aes(group="grp"), size=0.8)]
        p = (ggplot(d, aes("ckpt_idx", "v", color="curve"))
             + extras
             + geom_hline(yintercept=1.0, color="#c9c8c2", size=0.4)
             + geom_point(raw, aes("ckpt_idx", "v", color="curve"),
                          alpha=0.35, size=1.0, stroke=0)
             + smooth_layers
             + coord_cartesian(ylim=(-0.12, 1.45))
             + scale_color_manual(PAL)
             + scale_x_continuous(expand=(0.02, 0, 0.28, 0))
             + sum([[annotate("text", x=xmax + 2, y=yy, label=c,
                              color=PAL[c], size=8, ha="left")]
                    for c, yy in labels_y.items()], [])
             + labs(title=title,
                    subtitle="Each curve as share of its own LATE-BASE "
                             "value (median of last 3 base rungs) — above "
                             "1.0 means alignment raised it.\nCoverage "
                             "gate n>=10. Points: exact rung values; "
                             + (f"lines: loess (span {SPAN}"
                                + (f", SE band alpha {SE_A}" if SE
                                   else ", no SE") + ") WITHIN each phase"
                                if LOESS else
                                "lines: rolling-median(5) WITHIN each "
                                "phase")
                             + ",\nnever smoothed across base/SFT/DPO/"
                             "RLVR; segments too short for a fit drawn "
                             "raw.\nGrey: probe absent rate ([5436]). "
                             "Raw curves: fig14-16.",
                    x=xlab, y="share of own late-base value")
             + TH)
        fign = "fig23" if LOESS else "fig21"
        p.save(f"{FIGDIR}/{fign}_acquisition_ladder_{ladder}.png", dpi=300,
               verbose=False)
        print(f"wrote {FIGDIR}/{fign}_acquisition_ladder_{ladder}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
