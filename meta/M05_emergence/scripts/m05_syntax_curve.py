#!/usr/bin/env python
"""The syntax curve, both ladders, both coders — plan
meta/M05_emergence/plans/syntax_curve.md.

    uv run python meta/M05_emergence/scripts/m05_syntax_curve.py

Inputs, all frozen: data/m05_class_mass.parquet (mass by pos_class per
cell), data/m05_licit_sets.json (deepseek-v4-flash), data/
m05_licit_sets_haiku.json (claude-haiku-4-5). Curve = share of RESOLVED
mass on licit classes, per checkpoint, median over prompts; STRICT (licit)
and PERMISSIVE (licit+marginal) variants; format band (PUNCT/X/SYM) never
counts against grammar; convention equivalences ADP=PART, NUM=NOUN,
AUX=VERB applied to the coder's sets. payload_empty cells censored.
Coverage (share of prompts with a resolved cell) is drawn WITH the curve
per [5434]'s discipline: the columns travel together or not at all.

Figures: fig16_syntax_curve_olmo.png (full ladder, phase bands),
fig16_syntax_curve_pythia.png (pretraining, sub-1000 window shaded).
Numbers: results/syntax_curve.json (per-coder onsets on the strict curve,
m05_onsets criterion, base arms).
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
OUTJ = "meta/M05_emergence/results/syntax_curve.json"
FORMAT_BAND = {"PUNCT", "X", "SYM"}
EQUIV = [{"ADP", "PART"}, {"NUM", "NOUN"}, {"AUX", "VERB"}]
CODERS = {"deepseek-v4-flash": "data/m05_licit_sets.json",
          "claude-haiku-4-5": "data/m05_licit_sets_haiku.json"}
RNG = np.random.default_rng(11)


def expand(classes):
    out = set(classes)
    for g in EQUIV:
        if out & g:
            out |= g
    return out


def licit_tables():
    tabs = {}
    for coder, path in CODERS.items():
        d = json.load(open(path))["prompts"]
        tabs[coder] = {
            p: (expand({w["pos"] for w in v["licit"]}),
                expand({w["pos"] for w in v["licit"]}
                       | {w["pos"] for w in v["marginal"]}))
            for p, v in d.items()}
    return tabs


def boot_lo(vals, n=2000):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return np.nan
    meds = np.median(RNG.choice(vals, (n, len(vals))), axis=1)
    return float(np.percentile(meds, 2.5))


def main():
    from plotnine import (aes, annotate, element_blank, element_line,
                          element_rect, element_text, geom_line, geom_rect,
                          ggplot, labs, scale_color_manual,
                          scale_linetype_manual, scale_x_continuous, theme,
                          theme_minimal)
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8987"
    INK, INK2 = "#0b0b0b", "#52514e"
    TH = (theme_minimal(base_size=11) +
          theme(panel_grid_minor=element_blank(),
                panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                text=element_text(color=INK),
                plot_title=element_text(size=13, weight="bold"),
                plot_subtitle=element_text(size=9, color=INK2),
                legend_position="none",
                plot_background=element_rect(fill="#fcfcfb",
                                             color="#fcfcfb"),
                figure_size=(9, 5)))
    PAL = {"deepseek-v4-flash": BLUE, "claude-haiku-4-5": ORANGE,
           "coverage": GREY}

    cm = pd.read_parquet("data/m05_class_mass.parquet")
    tabs = licit_tables()
    report = {}

    for ladder in ("olmo", "pythia"):
        sub = cm[cm.ladder == ladder]
        n_prompts = sub.prompt.nunique()
        cells = (sub[~sub.payload_empty]
                 .groupby(["ckpt_idx", "prompt", "resolved_mass"]))
        # per-cell licit share per coder/variant
        recs = []
        for (ck, p, res), g in cells:
            if res <= 0:
                continue
            fmt = g[g.pos_class.isin(FORMAT_BAND)].mass.sum()
            denom = res
            for coder, tab in tabs.items():
                strict_set, perm_set = tab.get(p, (set(), set()))
                lic_s = g[g.pos_class.isin(strict_set)].mass.sum()
                lic_p = g[g.pos_class.isin(perm_set)].mass.sum()
                recs.append(dict(ckpt_idx=ck, prompt=p, coder=coder,
                                 strict=lic_s / denom, perm=lic_p / denom,
                                 fmt=fmt / denom))
        d = pd.DataFrame(recs)
        med = (d.groupby(["ckpt_idx", "coder"], as_index=False)
               .agg(strict=("strict", "median"), perm=("perm", "median")))
        cov = (sub[~sub.payload_empty].groupby("ckpt_idx").prompt.nunique()
               / n_prompts).rename("coverage").reset_index()
        order = (sub[["ckpt_idx", "role", "stage", "step"]]
                 .drop_duplicates().sort_values("ckpt_idx"))

        # onsets on the strict curve, base arm, per coder
        report[ladder] = {}
        base_idx = set(order[order.role == "base_step"].ckpt_idx)
        for coder in tabs:
            per_rung = {ck: g.strict.values for ck, g in
                        d[(d.coder == coder)
                          & d.ckpt_idx.isin(base_idx)].groupby("ckpt_idx")}
            rungs = sorted(per_rung)
            above = {r: boot_lo(per_rung[r]) > 0.5 for r in rungs}
            onset = None
            for i, r in enumerate(rungs):
                if above[r] and all(above[q] for q in rungs[i:]):
                    onset = r
                    break
            row = order[order.ckpt_idx == onset]
            report[ladder][coder] = dict(
                onset_rung=onset,
                onset_step=(None if row.empty else
                            f"{row.iloc[0].stage}-{int(row.iloc[0].step)}"
                            if row.iloc[0].stage else
                            str(int(row.iloc[0].step))),
                criterion="CI(median strict licit share) > 0.5, persistent",
                final_strict=float(
                    med[(med.coder == coder)
                        & (med.ckpt_idx == med.ckpt_idx.max())].strict.iloc[0]))
            print(f"{ladder} {coder}: majority-licit onset "
                  f"{report[ladder][coder]['onset_step']} "
                  f"(final strict {report[ladder][coder]['final_strict']:.2f})")

        # ---- figure ------------------------------------------------------
        lines = med.melt(["ckpt_idx", "coder"], ["strict", "perm"],
                         "variant", "share")
        lines["key"] = lines.coder
        covl = cov.rename(columns={"coverage": "share"})
        covl["key"] = "coverage"
        covl["variant"] = "strict"
        alll = pd.concat([lines, covl[["ckpt_idx", "key", "variant",
                                       "share"]].assign(coder="coverage")])
        extras = []
        if ladder == "olmo":
            bounds = {}
            for ph, roles in [("BASE", ("base_step", "base_endpoint")),
                              ("SFT", ("sft_step", "sft_endpoint")),
                              ("DPO", ("dpo_endpoint",)),
                              ("RLVR", ("rlvr_step",))]:
                s = order[order.role.isin(roles)]
                if len(s):
                    bounds[ph] = (s.ckpt_idx.min() - .5, s.ckpt_idx.max() + .5)
            for i, (k, v) in enumerate(bounds.items()):
                extras.append(annotate(
                    "rect", xmin=v[0], xmax=v[1], ymin=-np.inf, ymax=np.inf,
                    fill=("#f2efe6" if i % 2 else "#fcfcfb"), color="none"))
                extras.append(annotate(
                    "text", x=float(np.mean(v)),
                    y=1.03 if k != "RLVR" else 0.99, label=k, size=8,
                    color=INK2))
            title = ("The syntax curve on the full OLMo-3 ladder, "
                     "two coder families")
        else:
            steps = order.set_index("ckpt_idx").step
            blind = [r for r in steps.index if (steps[r] or 0) < 1000
                     and r in set(order[order.role == "base_step"].ckpt_idx)]
            if blind:
                extras.append(annotate("rect", xmin=-.5,
                                       xmax=max(blind) + .5, ymin=-np.inf,
                                       ymax=np.inf, fill="#f2efe6",
                                       color="none"))
            title = ("The syntax curve on the Pythia-6.9b ladder, "
                     "two coder families")
        p = (ggplot(alll, aes("ckpt_idx", "share", color="coder",
                              linetype="variant"))
             + extras
             + geom_line(size=0.8)
             + scale_color_manual(PAL)
             + scale_linetype_manual({"strict": "solid", "perm": "dotted"})
             + labs(title=title,
                    subtitle="Median share of RESOLVED mass on licit "
                             "classes; solid strict, dotted permissive; "
                             "blue deepseek-v4-flash, orange "
                             "claude-haiku-4-5.\nGrey: coverage (share of "
                             "prompts with a resolved cell) — reads WITH "
                             "the curve per [5434]; early rungs are "
                             "coverage, not grammar.",
                    x=("training position (base | SFT | DPO | RLVR)"
                       if ladder == "olmo" else
                       "pretraining rung (vendor grid, log-spaced early)"),
                    y="share")
             + TH)
        p.save(f"{FIGDIR}/fig16_syntax_curve_{ladder}.png", dpi=300,
               verbose=False)
        print(f"wrote {FIGDIR}/fig16_syntax_curve_{ladder}.png")

    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    with open(OUTJ, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {OUTJ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
