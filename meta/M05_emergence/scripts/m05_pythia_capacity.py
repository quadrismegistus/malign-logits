#!/usr/bin/env python
"""Pythia ladder: capacity acquisition (packages, reference, reasoning,
discourse, poetic) on the 155-rung Pythia-6.9b population.

    uv run python meta/M05_emergence/scripts/m05_pythia_capacity.py

A SEPARATE STUDY, never pooled with M05's OLMo population ([5425](b),
[5430]): different lab, tokenizer, corpus. Same battery by declaration
(data/m05_battery.json, the axis under test is TIME), same instruments as
m05_onsets.py's base-arm block and m05_figures.py's fig2 -- the onset
criterion (first rung whose bootstrap CI of the median log ratio sits above
zero and stays there to the end of the arm) and the POST-HOC half-of-own-
ceiling milestone are copied, not reinvented, so the two ladders are read by
one rule. Comparisons with OLMo are cross-ladder and labelled so.

What Pythia adds that OLMo cannot: eleven rungs below step 1000 (log-spaced
0,1,2,4,8,...,512), the window where [5430] found the eight-fold rise and
where OLMo's release schedule is blind. The figure shades that window.

Reads data/pythia_curves.parquet (m05_curves.py --population
data/pythia_population.json). Writes results/pythia_capacity_onsets.json
and figures/fig14_pythia_capacity.png.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

CURVES = "data/pythia_curves.parquet"
OUT = "meta/M05_emergence/results/pythia_capacity_onsets.json"
FIGDIR = "meta/M05_emergence/figures"

RNG = np.random.default_rng(11)


def boot_ci(vals, n=2000):
    """95% bootstrap CI of the median (same as m05_onsets.boot_ci)."""
    vals = np.asarray(vals)
    if len(vals) == 0:
        return (np.nan, np.nan)
    meds = np.median(RNG.choice(vals, (n, len(vals))), axis=1)
    return (float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


FAMS = [("CAPACITY_PACKAGES", "semantic packages"),
        ("CAPACITY_REFERENCE", "reference (facts)"),
        ("CAPACITY_REASONING", "reasoning"),
        ("CAPACITY_DISCOURSE", "discourse tracking"),
        ("POETIC", "poetic pull")]


def main():
    df = pd.read_parquet(CURVES)
    base = df[df.role == "base_step"]
    steps = (base[["ckpt_idx", "step"]].drop_duplicates()
             .set_index("ckpt_idx").step)
    print(f"{base.ckpt_idx.nunique()} base rungs, steps "
          f"{int(steps.min())}..{int(steps.max())}")

    # per-family per-rung value vectors (one value per surviving probe)
    fam_vals, fam_med = {}, {}
    for fam, label in FAMS:
        g = base[base.curve == fam]
        vals_by_rung = {}
        for r, gg in g.groupby("ckpt_idx"):
            piv = gg.pivot_table(index="probe", columns="word_role",
                                 values="p", aggfunc="first")
            both_absent = gg.groupby("probe").absent.all()
            piv = piv[~both_absent.reindex(piv.index, fill_value=False)]
            if fam == "POETIC":
                if {"formulaic", "paraphrase"} <= set(piv.columns):
                    vals_by_rung[r] = (piv.formulaic - piv.paraphrase).values
            else:
                if {"target", "competitor"} <= set(piv.columns):
                    vals_by_rung[r] = np.log(piv.target
                                             / piv.competitor).values
        fam_vals[label] = vals_by_rung
        fam_med[label] = {r: float(np.median(v))
                          for r, v in vals_by_rung.items()}

    # ---- onsets, same criterion as m05_onsets.py base-arm block ----------
    report = {"_population": "data/pythia_population.json (separate study, "
                             "never pooled with M05 OLMo -- [5425](b)/[5430])",
              "_criterion": "first rung with bootstrap CI(median) > 0, "
                            "persisting to end of base arm"}
    #: Pythia's early rungs are absent-dominated ([5430]: ~5 words per cell
    #: at step0), so a sign-only onset can fire on two surviving pairs at
    #: imputation scale. Report BOTH: the shared criterion verbatim, and a
    #: coverage-gated variant (>= MIN_N probes surviving at the rung) so a
    #: reader can see when the onset is carried by a handful of cells.
    MIN_N = 10
    order_rows = []
    for fam, label in FAMS:
        vals_by_rung = fam_vals[label]
        rungs = sorted(vals_by_rung)
        above = {r: boot_ci(vals_by_rung[r])[0] > 0 for r in rungs}

        def first_persistent(rs):
            for i, r in enumerate(rs):
                if above[r] and all(above[q] for q in rs[i:]):
                    return r
            return None

        onset = first_persistent(rungs)
        gated = first_persistent(
            [r for r in rungs if len(vals_by_rung[r]) >= MIN_N])
        n_at = (len(vals_by_rung[onset]) if onset is not None else 0)
        st = int(steps.get(onset)) if onset is not None else None
        stg = int(steps.get(gated)) if gated is not None else None
        g = base[base.curve == fam]
        cens = float(g.groupby("probe").absent.all().mean())
        order_rows.append((label, onset, st, cens, n_at, gated, stg))
        print(f"  {label:18} onset rung "
              f"{onset if onset is not None else '--':>4} "
              f"(step {st if st is not None else 'NONE'}, n={n_at})   "
              f"gated n>={MIN_N}: step {stg if stg is not None else 'NONE'}"
              f"   both-absent probes {cens:.0%}")
    order_rows.sort(key=lambda x: (x[5] is None, x[5]))
    print(f"\nORDER OF ACQUISITION (coverage-gated n>={MIN_N}, "
          "earliest first):")
    for i, (label, *_, stg) in enumerate(order_rows, 1):
        print(f"  {i}. {label} (step {stg})")
    report["base_order"] = [
        dict(family=l, onset_rung=o, onset_step=s, both_absent=c,
             n_probes_at_onset=n, onset_rung_gated=gr, onset_step_gated=gs)
        for l, o, s, c, n, gr, gs in order_rows]
    report["_gate"] = f"gated onset requires >= {MIN_N} surviving probes"

    # ---- POST-HOC half-of-own-ceiling milestone (fig2's metric) ----------
    halfmax = {}
    for _, label in FAMS:
        med = fam_med[label]
        if not med:
            continue
        final = med[max(med)]
        target = final / 2
        hit = min((r for r, v in med.items() if v >= target), default=None)
        halfmax[label] = dict(
            half_max_rung=hit,
            half_max_step=(int(steps.get(hit)) if hit is not None else None),
            base_final=final)
        print(f"  half-max (POST-HOC) {label:18} step "
              f"{halfmax[label]['half_max_step']}")
    report["half_max_post_hoc"] = halfmax

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\nwrote {OUT}")

    # ---- figure -----------------------------------------------------------
    from plotnine import (aes, annotate, element_blank, element_line,
                          element_rect, element_text, geom_hline, geom_line,
                          geom_rect, ggplot, labs, scale_color_manual,
                          scale_x_continuous, theme, theme_minimal)
    BLUE, ORANGE, AQUA, MAGENTA, VIOLET = (
        "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7")
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

    rows = []
    for _, label in FAMS:
        for r, v in fam_med[label].items():
            rows.append(dict(ckpt_idx=r, family=label, v=v))
    cap = pd.DataFrame(rows)
    finals = cap[cap.ckpt_idx == cap.ckpt_idx.max()].set_index("family").v
    pal = {"reference (facts)": BLUE, "reasoning": AQUA,
           "discourse tracking": ORANGE, "semantic packages": VIOLET,
           "poetic pull": MAGENTA}
    #: the window OLMo cannot see: rungs with step < 1000
    blind = sorted(r for r in steps.index if steps[r] < 1000)
    blind_hi = max(blind) + 0.5 if blind else 0
    brk = [0, 11, 20, 38, 83, 153]
    brk = [b for b in brk if b in steps.index]

    def steplab(i):
        s = int(steps.get(i, 0))
        return f"{s//1000}k" if s >= 1000 else str(s)

    p = (ggplot(cap, aes("ckpt_idx", "v", color="family"))
         + geom_rect(xmin=-0.5, xmax=blind_hi, ymin=-np.inf, ymax=np.inf,
                     fill="#f2efe6", color="none")
         + geom_hline(yintercept=0, color="#c9c8c2", size=0.4)
         + geom_line(size=0.9)
         + scale_color_manual(pal)
         + sum([[annotate("text", x=cap.ckpt_idx.max() + 1.5,
                          y=finals.get(f, 0), label=f, color=c, size=8,
                          ha="left")] for f, c in pal.items()], [])
         + annotate("text", x=blind_hi + 1, y=float(cap.v.max()) * 0.97,
                    label="shaded: below OLMo's first rung (steps 0-512)",
                    color=INK2, size=8, ha="left")
         + scale_x_continuous(breaks=brk, labels=[steplab(i) for i in brk],
                              expand=(0.02, 0, 0.30, 0))
         + labs(title="Capacity acquisition on the Pythia-6.9b ladder "
                      "(separate study, cross-ladder comparisons only)",
                subtitle="Median log p(correct)/p(competitor) per family "
                         "(poetic: p difference), 154 pretraining rungs.\n"
                         "Rung axis is the vendor grid, log-spaced early, "
                         "not linear time; the shaded window is "
                         "absent-dominated (few battery words above theta).",
                x="pretraining step (vendor grid)",
                y="log odds, correct vs competitor")
         + TH)
    p.save(f"{FIGDIR}/fig14_pythia_capacity.png", dpi=300, verbose=False)
    print(f"wrote {FIGDIR}/fig14_pythia_capacity.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
