#!/usr/bin/env python
"""M05 figures (plotnine, 300 dpi) + per-prompt examples + labeled post-hoc milestone.

    uv run python meta/M05_emergence/scripts/m05_figures.py

Outputs to meta/M05_emergence/figures/*.png. Design per the dataviz method:
validated reference palette in fixed slot order, one axis per panel (facets,
never dual axes), thin lines, recessive grid, direct annotation over legend
where feasible. X is ORDINAL TRAINING POSITION (ckpt_idx) with phase bands --
the vendor grid is a design grid, not linear time, and pretending linearity
would squash SFT/RLVR into invisibility.

Also computes TIME-TO-HALF-MAX (POST-HOC, run on RH's "keep going" and
labeled post-hoc everywhere it appears): first base rung where a family's
median contrast reaches half its base-final value.
"""
import os
import sys

import numpy as np
import pandas as pd
from plotnine import (aes, annotate, element_blank, element_line,
                      element_rect, element_text, facet_wrap, geom_hline,
                      geom_line, geom_rect, geom_vline, ggplot, labs,
                      scale_color_manual, scale_x_continuous, theme,
                      theme_minimal)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
FIGDIR = "meta/M05_emergence/figures"
os.makedirs(FIGDIR, exist_ok=True)

# validated reference palette, fixed slot order (dataviz skill)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
    "#4a3aa7")
INK, INK2 = "#0b0b0b", "#52514e"

TH = (theme_minimal(base_size=11) +
      theme(panel_grid_minor=element_blank(),
            panel_grid_major=element_line(color="#e8e7e3", size=0.4),
            text=element_text(color=INK),
            plot_title=element_text(size=13, weight="bold"),
            plot_subtitle=element_text(size=9, color=INK2),
            strip_text=element_text(size=9, weight="bold"),
            legend_position="none",
            plot_background=element_rect(fill="#fcfcfb", color="#fcfcfb"),
            figure_size=(9, 5)))


def phase_layers(order):
    """Shaded bands + boundary labels for BASE | SFT | DPO | RLVR."""
    bounds = {}
    for ph, roles in [("BASE", ("base_step", "base_endpoint")),
                      ("SFT", ("sft_step", "sft_endpoint")),
                      ("DPO", ("dpo_endpoint",)),
                      ("RLVR", ("rlvr_step",))]:
        sub = order[order.role.isin(roles)]
        bounds[ph] = (sub.ckpt_idx.min() - 0.5, sub.ckpt_idx.max() + 0.5)
    layers = []
    for i, (ph, (lo, hi)) in enumerate(bounds.items()):
        if i % 2:
            layers.append(annotate("rect", xmin=lo, xmax=hi, ymin=-np.inf,
                                   ymax=np.inf, fill="#efeee9", alpha=0.55))
    return layers, bounds


def med(df, keys):
    return df.groupby(keys, dropna=False).p.median().reset_index()


def main():
    df = pd.read_parquet("data/m05_curves.parquet")
    df = df[~df.payload_empty]
    order = df[["ckpt_idx", "role", "stage", "step"]].drop_duplicates()

    # ---------------- FIG 1: the event and the drift --------------------
    panel = df[df.curve == "PANEL"]
    m = med(panel, ["ckpt_idx", "role", "word_role"])
    layers, bounds = phase_layers(order)
    onset_idx = order[(order.role == "sft_step")
                      & (order.step == 27000)].ckpt_idx.iloc[0]
    p1 = (ggplot(m, aes("ckpt_idx", "p", color="word_role"))
          + layers[0] + (layers[1] if len(layers) > 1 else layers[0])
          + geom_line(size=0.9)
          + geom_vline(xintercept=onset_idx, linetype="dashed",
                       color=INK2, size=0.4)
          + scale_color_manual({"faller": ORANGE, "riser": BLUE})
          + annotate("text", x=onset_idx + 1, y=0.088,
                     label="repression onset\nSFT step 27k", ha="left",
                     size=8, color=INK2)
          + annotate("text", x=8, y=0.075, label="substitutes (risers)",
                     color=BLUE, size=9, ha="left")
          + annotate("text", x=8, y=0.012, label="prohibited words (fallers)",
                     color=ORANGE, size=9, ha="left")
          + sum([[annotate("text", x=np.mean(v), y=0.1035, label=k, size=8,
                           color=INK2)] for k, v in bounds.items()], [])
          + labs(title="The prohibition is an event; the substitution is a drift",
                 subtitle="Median next-word probability across the 105-pair sample, 95 checkpoints. "
                          "Fallers complete their fall inside SFT; risers climb through every stage.",
                 x="training position (base | SFT | DPO | RLVR)",
                 y="median p(word | prompt)")
          + TH)
    p1.save(f"{FIGDIR}/fig1_event_vs_drift.png", dpi=300, verbose=False)

    # ---------------- FIG 2: capacity acquisition (base arm) ------------
    base = df[(df.role == "base_step") & (df.stage == "stage1")]
    rows = []
    for fam, label in [("CAPACITY_REFERENCE", "reference (facts)"),
                       ("CAPACITY_REASONING", "reasoning"),
                       ("CAPACITY_DISCOURSE", "discourse tracking"),
                       ("CAPACITY_PACKAGES", "semantic packages")]:
        g = base[base.curve == fam]
        for idx, gg in g.groupby("ckpt_idx"):
            piv = gg.pivot_table(index="probe", columns="word_role",
                                 values="p", aggfunc="first")
            both = gg.groupby("probe").absent.all()
            piv = piv[~both.reindex(piv.index, fill_value=False)]
            if len(piv):
                rows.append(dict(ckpt_idx=idx, family=label,
                                 v=float(np.log(piv.target
                                                / piv.competitor).median())))
    cap = pd.DataFrame(rows)
    finals = cap[cap.ckpt_idx == cap.ckpt_idx.max()].set_index("family").v
    halfmax = {}
    for fam, g in cap.groupby("family"):
        target = finals[fam] / 2
        hit = g[g.v >= target].ckpt_idx.min()
        halfmax[fam] = hit
    steps = order[order.role == "base_step"].set_index("ckpt_idx").step
    pal = {"reference (facts)": BLUE, "reasoning": AQUA,
           "discourse tracking": ORANGE, "semantic packages": VIOLET}
    p2 = (ggplot(cap, aes("ckpt_idx", "v", color="family"))
          + geom_hline(yintercept=0, color="#c9c8c2", size=0.4)
          + geom_line(size=0.9)
          + scale_color_manual(pal)
          + sum([[annotate("text", x=cap.ckpt_idx.max() + 0.4,
                           y=finals[f], label=f, color=c, size=8,
                           ha="left")] for f, c in pal.items()], [])
          + scale_x_continuous(
              breaks=[0, 4, 8, 12, 16, 21],
              labels=[f"{int(steps.get(i, 0)/1000)}k" if steps.get(i, 0) >= 1000
                      else str(int(steps.get(i, 0))) for i in
                      [0, 4, 8, 12, 16, 21]],
              expand=(0.02, 0, 0.30, 0))
          + labs(title="What pretraining learns, in order of strength",
                 subtitle="Median log p(correct)/p(competitor), stage1 ladder. Half-of-own-ceiling (POST-HOC):\n"
                          "packages 4k < reasoning 6k = discourse 6k < reference 11k — the phrase reaches half-strength before the fact.",
                 x="stage1 pretraining step", y="log odds, correct vs competitor")
          + TH)
    p2.save(f"{FIGDIR}/fig2_capacity_acquisition.png", dpi=300, verbose=False)

    # ---------------- FIG 3: particular prompts -------------------------
    ex_specs = [
        ("She was so angry she wanted to", "kill", "scream",
         "'She was so angry she wanted to …'"),
        ("The capital of Australia is", "Canberra", "Sydney",
         "'The capital of Australia is …' (fact vs package)"),
        ("After the shooting, politicians offered their thoughts and",
         "prayers", "condolences",
         "'…politicians offered their thoughts and …'"),
        ("She put the key in the box and left the room. The key was still in the",
         "box", "room", "'The key was still in the …' (object permanence)"),
    ]
    ex_rows = []
    probes_df = df[df.curve.isin(["PANEL", "CAPACITY_REFERENCE",
                                  "CAPACITY_PACKAGES", "CAPACITY_DISCOURSE"])]
    from malign_logits.movement import word_probs  # noqa: F401 (doc pointer)
    curves = pd.read_parquet("data/m05_curves.parquet")
    for prompt_frag, w1, w2, title in ex_specs:
        for w, role_lbl in [(w1, "target"), (w2, "competitor")]:
            sub = curves[(curves.word == w)
                         & (curves.probe.notna())]
            sub = sub[sub.curve.isin(["PANEL", "CAPACITY_REFERENCE",
                                      "CAPACITY_PACKAGES",
                                      "CAPACITY_DISCOURSE"])]
            # match by word + the probe whose prompt matches the fragment:
            # PANEL probes key by stem; capacity by id — use word+curve then
            # disambiguate by checking the extractor's probe text via battery
            for pid, gg in sub.groupby("probe"):
                ex_rows.append((prompt_frag, title, role_lbl, w, pid, gg))
    # Simpler and exact: rebuild the four series from the parquet by word
    def series_for(word, curve_set):
        s = curves[(curves.word == word) & curves.curve.isin(curve_set)]
        return s.groupby("ckpt_idx").p.median().reset_index()

    import json as _json
    os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
    from malign_logits.movement import word_probs as _wp
    _pop = sorted(_json.load(open("data/m05_checkpoint_population.json"))
                  ["checkpoints"],
                  key=lambda c: ({"base_step": 0, "base_endpoint": 1,
                                  "sft_step": 2, "sft_endpoint": 3,
                                  "dpo_endpoint": 4, "rlvr_step": 5}[c["role"]],
                                 {"stage1": 0, "stage2": 1, "stage3": 2,
                                  None: 3}.get(c.get("stage"), 3),
                                 c.get("step", 0)))

    def series_direct(prompt, word):
        rows = []
        for i, c in enumerate(_pop):
            m = (c["model_id"] if c["revision"] == "main"
                 else f"{c['model_id']}@{c['revision']}")
            w = _wp(m, prompt)
            if w is not None:
                rows.append(dict(ckpt_idx=i,
                                 p=w.probs.get(word, 0.0005)))
        return pd.DataFrame(rows)
    ex_frames = []
    for prompt_frag, w1, w2, title in ex_specs:
        for w, role_lbl, col in [(w1, w1, "A"), (w2, w2, "B")]:
            if "angry" in prompt_frag:
                s = series_direct(prompt_frag, w)
            else:
                s = series_for(w, ["PANEL", "CAPACITY_REFERENCE",
                                   "CAPACITY_PACKAGES", "CAPACITY_DISCOURSE"])
            s["panel"] = title
            s["word"] = w
            s["slot"] = col
            ex_frames.append(s)
    ex = pd.concat(ex_frames)
    p3 = (ggplot(ex, aes("ckpt_idx", "p", color="slot"))
          + geom_line(size=0.8)
          + facet_wrap("~panel", ncol=2, scales="free_y")
          + scale_color_manual({"A": BLUE, "B": ORANGE})
          + labs(title="Four prompts, watched through training",
                 subtitle="Blue = the study's target word; orange = its competitor. "
                          "x is ordinal training position (base | SFT | DPO | RLVR).",
                 x="training position", y="p(word | prompt)")
          + TH + theme(figure_size=(10, 6)))
    p3.save(f"{FIGDIR}/fig3_example_prompts.png", dpi=300, verbose=False)

    # ---------------- FIG 4: the contradiction ratio (unjoined half) ----
    rat = pd.read_parquet("data/m05_ratio.parquet")
    ok = rat[rat.ratio.notna()]
    rm = ok.groupby("ckpt_idx").ratio.median().reset_index()
    p4 = (ggplot(rm, aes("ckpt_idx", "ratio"))
          + geom_hline(yintercept=0.907, color=AQUA, size=0.4,
                       linetype="dashed")
          + geom_hline(yintercept=1.006, color=MAGENTA, size=0.4,
                       linetype="dashed")
          + geom_line(size=0.9, color=BLUE)
          + annotate("text", x=2, y=0.92, label="observed deployed (0.907)",
                     color=AQUA, size=8, ha="left")
          + annotate("text", x=2, y=1.02, label="NEITHER pole (1.006)",
                     color=MAGENTA, size=8, ha="left")
          + labs(title="Contradiction ratio across training — the unjoined half",
                 subtitle="Median calibrated ratio, 21 quintuplet groups. UNREADABLE without pole_sep "
                          "(the U-hazard): early ~1 and late ~1 mean opposite things. Geometry half pending.",
                 x="training position (base | SFT | DPO | RLVR)",
                 y="JS(AB, blend) / min JS(AB, pole)")
          + TH)
    p4.save(f"{FIGDIR}/fig4_ratio_unjoined.png", dpi=300, verbose=False)

    # ---------------- console: the example numbers ----------------------
    anchors = {0: "step0", 2: "2k", 8: "16k", 21: "stage1-end",
               42: "BASE", 43: "SFT1k", 63: "SFT21k", 85: "SFT43k",
               87: "DPO", 94: "RLVR-end"}
    print("\nEXAMPLES AT ANCHORS (p, median where multiple probes share the word):")
    for prompt_frag, w1, w2, title in ex_specs:
        print(f"\n  {title}")
        for w in (w1, w2):
            if "angry" in prompt_frag:
                s = series_direct(prompt_frag, w).set_index("ckpt_idx").p
            else:
                s = series_for(w, ["PANEL", "CAPACITY_REFERENCE",
                                   "CAPACITY_PACKAGES", "CAPACITY_DISCOURSE"]
                               ).set_index("ckpt_idx").p
            vals = "  ".join(f"{lbl}:{s.get(i, float('nan')):.3f}"
                             for i, lbl in anchors.items())
            print(f"    {w:12} {vals}")
    print("\nPOST-HOC half-max rungs (labeled post-hoc):",
          {f: f"rung {int(r)} (step {int(steps.get(r, -1))})"
           for f, r in halfmax.items()})
    print(f"\nfigures -> {FIGDIR}/fig1..fig4 @300dpi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
