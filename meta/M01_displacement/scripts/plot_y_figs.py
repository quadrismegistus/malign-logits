#!/usr/bin/env python
"""Y figures, per-letter registry convention (like plot_t_figs.py).

    uv run python meta/M01_displacement/scripts/plot_y_figs.py            # all
    uv run python meta/M01_displacement/scripts/plot_y_figs.py y_dissoc   # one
    uv run python meta/M01_displacement/scripts/plot_y_figs.py --list

y_dissoc: the dissociation scatter (Y_superego.md section 5). One point
per pair (32, pass A): x = assistant shift, y = superego shift, the
declared composites from y_dissociation.py (which reproduces the
registered r = -0.544 to the digit and asserts on it). The
anticorrelation IS the finding: alignment is not one operation, and the
pairs that gain the assistant are not the pairs that gain the moral
content. Axes in percentage points for reading; r computed on the raw
fractions upstream, identical under linear scaling.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGURES = "meta/M01_displacement/figures"
DISSOC = "meta/M01_displacement/results/y_dissociation.csv"

LABELED = {
    "LLM360/AmberSafe": "AmberSafe",
    "lomahony/eleuther-pythia6.9b-hh-dpo": "pythia-6.9b-hh-dpo",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    "llm-jp/llm-jp-3-7.2b-instruct3": "llm-jp-3",
    "tiiuae/falcon-mamba-7b-instruct": "falcon-mamba",
    "microsoft/phi-4-reasoning": "phi-4-reasoning",
    "Qwen/Qwen3-8B": "Qwen3-8B",
}


def y_dissoc():
    d = pd.read_csv(DISSOC, index_col=0)
    x = 100 * d.assistant
    y = 100 * d.superego
    r = np.corrcoef(d.superego, d.assistant)[0, 1]

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    lab = d.index.map(lambda p: p.split(">")[0] in
                      {k.split(">")[0] for k in LABELED} or p in LABELED)
    names = {p: n for k, n in LABELED.items()
             for p in d.index if p.startswith(k) or k in p}
    ax.scatter(x, y, s=42, c="#666666", alpha=0.75, zorder=3)
    for p in d.index:
        if p in names:
            ax.scatter(x[p], y[p], s=64, c="#2e6da4", zorder=4)
            ax.annotate(names[p], (x[p], y[p]),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=8.5, fontweight="bold", color="#2e6da4")
    ax.set_xlabel("assistant shift (pp): mean Δ of assistant_refusal, "
                  "<meta> presence, frame_exit")
    ax.set_ylabel("superego shift (pp): mean Δ of guilt_or_shame, "
                  "moralisation_in_scene, consent_hesitation")
    ax.set_title("Alignment is not one operation (Y §5)\n"
                 f"32 pairs, pass A; r = {r:+.3f} (registered −0.544; "
                 "producer y_dissociation.py, recovered 2026-08-14)\n"
                 "sign robust across definitional variants (−0.32..−0.59); "
                 "magnitude definition-dependent:\nquote only the declared "
                 "form", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGURES, "y_dissociation_scatter.png")
    fig.savefig(out)
    print(f"wrote {out}")



def y3_filter_predicts_the_flat_ones():
    """Y_diegetic 3-4: the filter account predicts the two that do not move.

    ADDED BY THE dario SEAT, 2026-08-14. Purely additive to this registry.

    Four measures, one axis, ordered so the argument runs top to bottom.
    The FILTER account of alignment -- a gate at the output that blocks,
    deflects or declines -- predicts EXIT rises and sexual_scene falls.
    Both are flat. What moves is the composition INSIDE the scene.

    So the nulls are the content here, and they are drawn first and
    given the same visual weight as the effects, with the prediction
    they falsify written on them. A figure that showed only the two
    moving measures would be the same finding with its argument removed.

    TWO POPULATIONS, AND THE SPLIT IS FORCED BY THE CONSTRUCT. The flat
    pair is over all pass-A parsed passages; the moving pair is
    restricted to those where a sexual scene occurred. That is not a
    choice: `sexual_scene` cannot be measured conditional on itself, and
    conditioning is what makes the within-scene effect airtight rather
    than a composition artifact. Each panel names its own population.

    THE PRODUCER'S STATISTIC, WHICH IS NOT THE OBVIOUS ONE. `y_diegetic.py`
    filters to `pass == "A" and parsed`, requires MIN_N = 20 passages per
    arm before a pair contributes, reports base and aligned as the MEAN
    of per-pair rates, and the delta as the MEDIAN of per-pair deltas.
    Guessing any of those gives four wrong numbers: an unfiltered
    per-pair mean returns -8.86pp for CLEAN_SCENE against the booked
    -6.12pp, and moves two of the four sign counts.
    """
    import numpy as np
    from plotnine import (aes, element_blank, element_text, facet_grid,
                          geom_jitter, geom_point, geom_text, geom_vline,
                          ggplot, labs, scale_color_identity,
                          scale_y_continuous, theme, theme_minimal)

    MIN_N = 20
    #: module does os.chdir(ROOT) at import, so repo-relative paths
    #: are correct here and RESULTS does not exist in this file.
    d = pd.read_parquet("meta/M01_displacement/results/y_passages.parquet")
    d = d[(d["pass"].astype(str) == "A") & (d.parsed.astype(bool))].copy()
    d["sex"] = d.sexual_scene.astype(str) == "YES"
    for c in ("EXIT", "CLEAN_SCENE", "SUPEREGO_IN_SCENE"):
        d[c] = d[c].astype(bool)

    SPEC = [
        ("EXIT", False, "EXIT  (refusal or frame exit)", "all passages",
         26.48, 27.80, 1.03, 17, "FILTER ACCOUNT PREDICTS THIS RISES"),
        ("sex", False, "sexual_scene", "all passages",
         53.85, 50.01, -0.22, 16, "FILTER ACCOUNT PREDICTS THIS FALLS"),
        ("CLEAN_SCENE", True, "CLEAN_SCENE", "given a sexual scene occurred",
         84.72, 76.68, -6.12, 27, ""),
        ("SUPEREGO_IN_SCENE", True, "SUPEREGO_IN_SCENE",
         "given a sexual scene occurred", 15.18, 21.60, 4.30, 24, ""),
    ]

    rows, ann = [], []
    for col, cond, label, pop, bb, ba, bd, bsign, pred in SPEC:
        f = d[d.sex] if cond else d
        g = (f.groupby(["pair", "arm"], observed=True)[col]
             .agg(["mean", "size"]).unstack())
        ok = (g[("size", "base")] >= MIN_N) & (g[("size", "aligned")] >= MIN_N)
        g = g[ok]
        s = ((g[("mean", "aligned")] - g[("mean", "base")]).dropna() * 100)
        base_pc = float(g[("mean", "base")].mean() * 100)
        algn_pc = float(g[("mean", "aligned")].mean() * 100)
        med = float(np.median(s))
        sign = int((s > 0).sum()) if bd > 0 else int((s < 0).sum())
        assert round(base_pc, 2) == bb and round(algn_pc, 2) == ba, \
            f"{col} rates drifted: {round(base_pc,2)}/{round(algn_pc,2)} vs {bb}/{ba}"
        assert round(med, 2) == bd, f"{col} delta drifted: {round(med,2)} vs {bd}"
        assert sign == bsign, f"{col} sign drifted: {sign} vs {bsign}"
        assert len(s) == 32, f"{col} pairs drifted: {len(s)} vs 32"
        flat = pred != ""
        for v in s:
            rows.append({"m": label, "pop": pop, "delta": v,
                         "col": "#9a9a9a" if flat else "#1f4e79"})
        ann.append({"m": label, "pop": pop, "delta": med,
                    "col": "#9a9a9a" if flat else "#1f4e79",
                    "txt": f"{bb:.2f}% -> {ba:.2f}%    median {bd:+.2f}pp    "
                           f"{sign} of 32 pairs",
                    "pred": pred})
    df, a = pd.DataFrame(rows), pd.DataFrame(ann)
    order = [s[2] for s in SPEC]
    for f in (df, a):
        f["m"] = pd.Categorical(f.m, categories=order, ordered=True)

    pr = a[a.pred != ""].copy()

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.45)
        + geom_jitter(df, aes("delta", 0, color="col"), height=0.20, width=0,
                      size=1.7, alpha=0.6)
        + geom_point(a, aes("delta", 0, color="col"), size=4.4, shape="D")
        + geom_text(a, aes("delta", 0.60, label="txt", color="col"), size=6.4,
                    ha="center", va="bottom")
        + geom_text(pr, aes(0, -0.52, label="pred"), size=6.4, ha="center",
                    va="top", color="#b03030")
        + scale_color_identity()
        + scale_y_continuous(limits=(-0.95, 1.05))
        + facet_grid("m ~ .")
        + labs(
            title="The filter account predicts the two things that do not move",
            subtitle=(
                "32 model pairs, pass-A parsed passages. One point per pair: the change in that "
                "measure's rate, aligned minus base, in percentage points.\n"
                "A FILTER at the output would block, deflect or decline, so it predicts EXIT rises and "
                "sexual_scene falls. Both are flat: 17 of 32 and 16 of 32, which is a coin flip twice.\n"
                "What moves is the composition INSIDE the scene. Given that the model writes the sex it "
                "is a fifth less likely to write it clean, on 27 of 32 pairs -- the strongest sign "
                "agreement in this corpus.\n"
                "THE TWO POPULATIONS ARE FORCED, NOT CHOSEN: sexual_scene cannot be measured "
                "conditional on itself, so the flat pair is over all passages and the moving pair is "
                "conditional on the scene occurring.\n"
                "Rates are the MEAN of per-pair rates and deltas the MEDIAN of per-pair deltas, per "
                "y_diegetic.py, with MIN_N = 20 passages per arm before a pair contributes. One coder."),
            x="change in rate, aligned minus base (percentage points)",
            y="",
            caption=("Producer: meta/M01_displacement/scripts/plot_y_figs.py from "
                     "results/y_passages.parquet (producer y_diegetic.py).\n"
                     "All twelve booked values asserted before drawing: base and aligned rates, median "
                     "delta and sign count for each of the four measures."),
        )
        + theme_minimal()
        + theme(figure_size=(11.4, 7.2),
                plot_title=element_text(size=13, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.1, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                strip_text=element_text(size=7.6, weight="bold"),
                panel_spacing=0.05)
    )
    out = os.path.join(FIGURES, "y3_filter_predicts_the_flat_ones.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    return out


REGISTRY = {
    "y3_filter": y3_filter_predicts_the_flat_ones,"y_dissoc": y_dissoc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=list(REGISTRY))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k in REGISTRY:
            print(k)
        return
    for k in (args.figs or list(REGISTRY)):
        if k not in REGISTRY:
            sys.exit(f"unknown figure {k!r}; use --list")
        REGISTRY[k]()


if __name__ == "__main__":
    main()
