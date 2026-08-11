#!/usr/bin/env python
"""Average p(correct) over training time — the capacity FLOOR, both ladders.

    uv run python meta/M05_emergence/scripts/m05_capacity_prob.py

RH's question (2026-08-11): the onset/ratio analyses contrast target
against competitor; this plots the ABSOLUTE mean p(target) (solid) and
mean p(competitor) (dashed) per family across training. It generalises the
poetic family's registered floor principle ([5379]: "cannot yet" and
"chose not to" separate on the floor and collide in the difference) to all
capacity families. No argmax anywhere: p is the stored probability from
true_word_probs; an absent word enters at theta/2 = 0.0005 with its
absent flag counted (mean over ALL probes, so early rungs are honest
near-zeros, not survivor means).

Two figures, two populations, never pooled ([5425](b)/[5430]):
  fig15_pythia_capacity_prob.png  -- Pythia base arm (154 rungs)
  fig15b_olmo_capacity_prob.png   -- OLMo FULL ladder (base|SFT|DPO|RLVR),
                                     so alignment's effect on p(correct)
                                     is visible for the first time here.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
FIGDIR = "meta/M05_emergence/figures"

FAMS = {"CAPACITY_PACKAGES": "semantic packages",
        "CAPACITY_REFERENCE": "reference (facts)",
        "CAPACITY_REASONING": "reasoning",
        "CAPACITY_DISCOURSE": "discourse tracking"}
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
PAL = {"reference (facts)": BLUE, "reasoning": AQUA,
       "discourse tracking": ORANGE, "semantic packages": VIOLET}
INK, INK2 = "#0b0b0b", "#52514e"


def mean_p(df):
    """per (family, rung, word_role) mean p over ALL probes (absent=theta/2)."""
    cap = df[df.curve.isin(FAMS)].copy()
    cap["family"] = cap.curve.map(FAMS)
    g = (cap.groupby(["family", "ckpt_idx", "word_role"], as_index=False)
         .agg(p=("p", "mean"), absent=("absent", "mean")))
    return g


def theme():
    from plotnine import (element_blank, element_line, element_rect,
                          element_text, theme, theme_minimal)
    return (theme_minimal(base_size=11) +
            theme(panel_grid_minor=element_blank(),
                  panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                  text=element_text(color=INK),
                  plot_title=element_text(size=13, weight="bold"),
                  plot_subtitle=element_text(size=9, color=INK2),
                  legend_position="none",
                  plot_background=element_rect(fill="#fcfcfb",
                                               color="#fcfcfb"),
                  figure_size=(9, 5)))


def main():
    from plotnine import (aes, annotate, geom_line, geom_rect, ggplot, labs,
                          scale_color_manual, scale_linetype_manual,
                          scale_x_continuous)

    # ---------- Pythia ----------------------------------------------------
    dfp = pd.read_parquet("data/pythia_curves.parquet")
    dfp = dfp[dfp.role == "base_step"]
    steps = (dfp[["ckpt_idx", "step"]].drop_duplicates()
             .set_index("ckpt_idx").step)
    gp = mean_p(dfp)
    finals = gp[(gp.ckpt_idx == gp.ckpt_idx.max())
                & (gp.word_role == "target")].set_index("family").p
    blind_hi = max(r for r in steps.index if steps[r] < 1000) + 0.5
    brk = [b for b in [0, 11, 20, 38, 83, 153] if b in steps.index]

    def lab(i):
        s = int(steps.get(i, 0))
        return f"{s//1000}k" if s >= 1000 else str(s)

    p = (ggplot(gp, aes("ckpt_idx", "p", color="family",
                        linetype="word_role"))
         + geom_rect(xmin=-0.5, xmax=blind_hi, ymin=-np.inf, ymax=np.inf,
                     fill="#f2efe6", color="none")
         + geom_line(size=0.8)
         + scale_color_manual(PAL)
         + scale_linetype_manual({"target": "solid", "competitor": "dashed"})
         + sum([[annotate("text", x=gp.ckpt_idx.max() + 1.5,
                          y=float(finals.get(f, 0)), label=f, color=c,
                          size=8, ha="left")] for f, c in PAL.items()], [])
         + annotate("text", x=blind_hi + 1, y=float(gp.p.max()) * 0.97,
                    label="shaded: below OLMo's first rung", color=INK2,
                    size=8, ha="left")
         + scale_x_continuous(breaks=brk, labels=[lab(i) for i in brk],
                              expand=(0.02, 0, 0.30, 0))
         + labs(title="Mean p(correct) on the Pythia-6.9b ladder "
                      "(solid: target; dashed: competitor)",
                subtitle="Mean stored probability over ALL probes per "
                         "family (absent words at theta/2, so early rungs "
                         "are honest near-zeros).\nSeparate population; "
                         "cross-ladder comparisons only.",
                x="pretraining step (vendor grid)", y="mean p(word)")
         + theme())
    p.save(f"{FIGDIR}/fig15_pythia_capacity_prob.png", dpi=300,
           verbose=False)
    print(f"wrote {FIGDIR}/fig15_pythia_capacity_prob.png")

    # ---------- OLMo, FULL ladder -----------------------------------------
    dfo = pd.read_parquet("data/m05_curves.parquet")
    ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
                  "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
    order = (dfo[["ckpt_idx", "model", "role", "stage", "step"]]
             .drop_duplicates().sort_values("ckpt_idx"))
    go = mean_p(dfo)
    finals = go[(go.ckpt_idx == go.ckpt_idx.max())
                & (go.word_role == "target")].set_index("family").p
    bounds = {}
    for ph, roles in [("BASE", ("base_step", "base_endpoint")),
                      ("SFT", ("sft_step", "sft_endpoint")),
                      ("DPO", ("dpo_endpoint",)),
                      ("RLVR", ("rlvr_step",))]:
        sub = order[order.role.isin(roles)]
        if len(sub):
            bounds[ph] = (sub.ckpt_idx.min() - 0.5, sub.ckpt_idx.max() + 0.5)
    bands = sum([[annotate("rect", xmin=lo, xmax=hi, ymin=-np.inf,
                           ymax=np.inf,
                           fill=("#f2efe6" if i % 2 else "#fcfcfb"),
                           color="none")]
                 for i, (lo, hi) in enumerate(bounds.values())], [])
    #: stagger the two narrow right-hand phase labels so they don't collide
    labels = sum([[annotate("text", x=float(np.mean(v)),
                            y=float(go.p.max()) * (1.04 if k != "RLVR"
                                                   else 0.99),
                            label=k, size=8, color=INK2)]
                  for k, v in bounds.items()], [])
    p2 = (ggplot(go, aes("ckpt_idx", "p", color="family",
                         linetype="word_role"))
          + bands + geom_line(size=0.8) + labels
          + scale_color_manual(PAL)
          + scale_linetype_manual({"target": "solid",
                                   "competitor": "dashed"})
          + sum([[annotate("text", x=go.ckpt_idx.max() + 1.0,
                           y=float(finals.get(f, 0)), label=f, color=c,
                           size=8, ha="left")] for f, c in PAL.items()], [])
          + scale_x_continuous(expand=(0.02, 0, 0.30, 0))
          + labs(title="Mean p(correct) across the full OLMo-3 ladder "
                       "(solid: target; dashed: competitor)",
                 subtitle="Base pretraining, then SFT / DPO / RLVR. Mean "
                          "stored probability over ALL probes per family\n"
                          "(absent at theta/2). Ordinal training position, "
                          "not linear time.",
                 x="training position (base | SFT | DPO | RLVR)",
                 y="mean p(word)")
          + theme())
    p2.save(f"{FIGDIR}/fig15b_olmo_capacity_prob.png", dpi=300,
            verbose=False)
    print(f"wrote {FIGDIR}/fig15b_olmo_capacity_prob.png")

    # numbers for the record
    for name, g in (("PYTHIA", gp), ("OLMO", go)):
        tail = g[g.ckpt_idx == g.ckpt_idx.max()]
        print(f"\n{name} final rung, mean p:")
        for _, r in tail.iterrows():
            print(f"  {r.family:20} {r.word_role:10} {r.p:.4f} "
                  f"(absent {r.absent:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
