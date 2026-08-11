#!/usr/bin/env python
"""The acquisition-ordering figure on the TOKEN axis: capacities, poetic
pull and the syntax curve on one 0-1 scale, per ladder.

    uv run python meta/M05_emergence/scripts/m05_acquisition_tokens.py

RH's ask (2026-08-11): the overall capacity graphs with poetic pull AND
syntax added, x = tokens seen by that step. The units differ (log-odds /
p-difference / licit share), so every curve is shown as SHARE OF ITS OWN
BASE-FINAL VALUE — a display normalization for ordering, labelled as
such; the raw curves live in fig14/fig15/fig16 and the parquets.

Token conversion per [5434]: Pythia 2,097,152 tokens/step (documented
constant batch); OLMo 4,194,304 tokens/step (inferred from round totals,
UNVERIFIED constant batch — labelled on the figure). Base arms only,
OLMo stage1 only, step 0 dropped (no place on a log axis; it is the
floor, not a rung). Per [5436]'s obligation the capacity-probe ABSENT
RATE is drawn beside the curves (thin grey, same 0-1 axis): early rungs
are coverage, not capacity. Ladders in separate figures, never pooled.

Outputs (new filenames): figures/fig17_acquisition_tokens_pythia.png,
figures/fig17_acquisition_tokens_olmo.png.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
FIGDIR = "meta/M05_emergence/figures"

TOK = {"pythia": 2_097_152, "olmo": 4_194_304}
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
       "probe absent rate": GREY}
INK, INK2 = "#0b0b0b", "#52514e"
FORMAT_BAND = {"PUNCT", "X", "SYM"}
EQUIV = [{"ADP", "PART"}, {"NUM", "NOUN"}, {"AUX", "VERB"}]


def expand(classes):
    out = set(classes)
    for g in EQUIV:
        if out & g:
            out |= g
    return out


def fam_medians(base, min_n=10):
    """per-family per-rung median contrast + capacity absent rate.
    Rungs with fewer than min_n surviving probes are dropped — the same
    coverage gate as the onsets; normalized junk at n=2 is not a curve."""
    med, rows_abs = {}, {}
    for fam, label in list(FAMS.items()) + [("POETIC", "poetic pull")]:
        g = base[base.curve == fam]
        by = {}
        for r, gg in g.groupby("ckpt_idx"):
            piv = gg.pivot_table(index="probe", columns="word_role",
                                 values="p", aggfunc="first")
            ba = gg.groupby("probe").absent.all()
            piv = piv[~ba.reindex(piv.index, fill_value=False)]
            if len(piv) < min_n:
                continue
            if fam == "POETIC":
                if {"formulaic", "paraphrase"} <= set(piv.columns):
                    by[r] = float((piv.formulaic - piv.paraphrase).median())
            elif {"target", "competitor"} <= set(piv.columns):
                by[r] = float(np.log(piv.target / piv.competitor).median())
        med[label] = by
    cap = base[base.curve.str.startswith("CAPACITY")]
    rows_abs = cap.groupby("ckpt_idx").absent.mean().to_dict()
    return med, rows_abs


def smooth(d):
    """centered rolling median (5) per curve, display only."""
    out = []
    for c, g in d.groupby("curve"):
        g = g.sort_values("tokens").copy()
        g["v"] = (g.v.rolling(5, center=True, min_periods=1).median())
        out.append(g)
    return pd.concat(out)


def syntax_share(ladder):
    """strict licit share per rung, deepseek coder (haiku is parallel)."""
    import json
    cm = pd.read_parquet("data/m05_class_mass.parquet")
    cm = cm[(cm.ladder == ladder) & (cm.role == "base_step")
            & ~cm.payload_empty]
    if ladder == "olmo":
        cm = cm[cm.stage == "stage1"]
    lic = {p: expand({w["pos"] for w in v["licit"]})
           for p, v in json.load(open("data/m05_licit_sets.json"))
           ["prompts"].items()}
    recs = []
    for (r, p), g in cm.groupby(["ckpt_idx", "prompt"]):
        res = g.resolved_mass.iloc[0]
        if res <= 0:
            continue
        s = g[g.pos_class.isin(lic.get(p, set()))].mass.sum() / res
        recs.append((r, s))
    d = pd.DataFrame(recs, columns=["ckpt_idx", "s"])
    return d.groupby("ckpt_idx").s.median().to_dict()


def main():
    from plotnine import (aes, annotate, element_blank, element_line,
                          element_rect, element_text, geom_line, ggplot,
                          labs, scale_color_manual, scale_x_continuous,
                          theme, theme_minimal)
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

    for ladder in ("pythia", "olmo"):
        df = pd.read_parquet(CURVES[ladder])
        base = df[df.role == "base_step"]
        if ladder == "olmo":
            base = base[base.stage == "stage1"]
        steps = (base[["ckpt_idx", "step"]].drop_duplicates()
                 .set_index("ckpt_idx").step)
        med, absr = fam_medians(base)
        med["syntax (licit share)"] = syntax_share(ladder)

        rows = []
        for label, by in med.items():
            if not by:
                continue
            final = by[max(by)]
            if abs(final) < 1e-9:
                continue
            for r, v in by.items():
                st = steps.get(r, 0)
                if not st or st <= 0:
                    continue  # step 0 is the floor, not a rung
                rows.append(dict(ckpt_idx=r, curve=label, v=v / final))
        for r, v in absr.items():
            st = steps.get(r, 0)
            if st and st > 0:
                rows.append(dict(ckpt_idx=r, curve="probe absent rate",
                                 v=v))
        d = pd.DataFrame(rows).rename(columns={"ckpt_idx": "tokens"})
        d = smooth(d)
        d = d.rename(columns={"tokens": "ckpt_idx"})
        d = d[(d.v > -0.15) & (d.v < 1.35)]  # clip residual display noise

        #: ordinal vendor grid on x (log-spaced early, so no empty
        #: decades), break labels in token units
        def tok_label(r):
            t = steps.get(r, 0) * TOK[ladder]
            return (f"{t / 1e9:.0f}B" if t >= 1e9 else f"{t / 1e6:.0f}M")

        cand = sorted(set(d.ckpt_idx))
        brk = [cand[i] for i in
               sorted({0, len(cand) // 5, 2 * len(cand) // 5,
                       3 * len(cand) // 5, 4 * len(cand) // 5,
                       len(cand) - 1})]
        xmax = max(cand)
        #: labels anchored to each line's own endpoint, minimal nudges
        ends = (d.sort_values("ckpt_idx").groupby("curve")
                .agg(x=("ckpt_idx", "last"), y=("v", "last")))
        labels_y, used = {}, []
        for cname, r in ends.sort_values("y", ascending=False).iterrows():
            yy = float(r.y)
            while any(abs(yy - u) < 0.055 for u in used):
                yy -= 0.058
            used.append(yy)
            labels_y[cname] = yy
        note = ("token axis: 2,097,152 tokens/step, documented"
                if ladder == "pythia" else
                "token axis: 4,194,304 tokens/step, INFERRED constant "
                "batch, unverified ([5434])")
        p = (ggplot(d, aes("ckpt_idx", "v", color="curve"))
             + geom_line(size=0.8)
             + scale_color_manual(PAL)
             + scale_x_continuous(breaks=brk,
                                  labels=[tok_label(b) for b in brk],
                                  expand=(0.02, 0, 0.28, 0))
             + sum([[annotate("text", x=xmax + 2, y=yy, label=c,
                              color=PAL[c], size=8, ha="left")]
                    for c, yy in labels_y.items()], [])
             + labs(title=f"What installs when, on tokens seen — "
                          f"{'Pythia-6.9b' if ladder == 'pythia' else 'OLMo-3 (stage1 base arm)'}",
                    subtitle="Each curve as share of its own base-final "
                             "value; rolling-median(5) display smoothing; "
                             "rungs under 10 surviving probes dropped.\n"
                             "Raw curves in fig14-16. Syntax = strict "
                             "licit share (deepseek; haiku parallel). "
                             "Grey: probe absent rate, with the curves "
                             "per [5436].\n" + note,
                    x="tokens seen (vendor checkpoint grid — spacing is "
                      "ordinal, log-like early)",
                    y="share of own base-final value")
             + TH)
        p.save(f"{FIGDIR}/fig17_acquisition_tokens_{ladder}.png", dpi=300,
               verbose=False)
        print(f"wrote {FIGDIR}/fig17_acquisition_tokens_{ladder}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
