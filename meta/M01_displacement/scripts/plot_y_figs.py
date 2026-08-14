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
import json
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


#: Y_superego §4's named values, on the population the finding prints:
#: <guilt> SPAN, coding PASS A. Neither the field nor the span-over-all-passes
#: reproduces these -- span/all gives AmberSafe +11.44 and a median of +0.51.
#: keyed by a fragment of the MODEL ID, not by section 4's prose name for it.
#: The doc writes "pythia-6.9b-hh-dpo"; the id is
#: `lomahony/eleuther-pythia6.9b-hh-dpo`, with no hyphen after "pythia", so a
#: prose-name match finds nothing. The assert caught it rather than the panel
#: labelling the wrong pair.
BOOKED_Y4 = {"AmberSafe": 15.4, "gemma-2-9b-it": 7.0, "llm-jp-3": 6.4,
             "hh-dpo": 6.2, "median": 0.8, "n_pairs": 32}
#: §4 says "four negative pairs including both Mamba architectures". Ten pairs
#: are below zero; exactly four are below -1pp and both Mambas are among those
#: four. The threshold is recovered from the doc's parenthetical and is
#: declared nowhere in it, so the panel states it rather than implying it.
BOOKED_Y4_THRESHOLD = -1.0
BOOKED_Y4_BELOW = 4


def y4_heterogeneity():
    """queue 17: the superego shift is heterogeneous, and that is the object."""
    from plotnine import (aes, element_blank, element_text, geom_hline,
                          geom_point, geom_segment, geom_text, geom_vline,
                          ggplot, labs, scale_color_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    src = "meta/M01_displacement/results/y_guilt_heterogeneity.json"
    d = json.load(open(src))
    rows = sorted(d["pairs"], key=lambda r: r["delta_pp"])

    assert d["n_pairs"] == BOOKED_Y4["n_pairs"] == len(rows), \
        f"pairs drifted: {d['n_pairs']} vs booked {BOOKED_Y4['n_pairs']}"
    assert d["pass"] == "A", f"population is pass {d['pass']!r}, not A"
    by = {r["aligned_model"].split("/")[-1]: r["delta_pp"] for r in rows}
    for name, booked in BOOKED_Y4.items():
        if name in ("median", "n_pairs"):
            continue
        hit = [v for k, v in by.items() if name.lower() in k.lower()]
        assert len(hit) == 1, f"{name}: {len(hit)} matches among the aligned models"
        assert abs(hit[0] - booked) < 0.1, \
            f"{name}: {hit[0]:+.2f}pp against section 4's {booked:+.1f}pp"
    assert abs(d["median_delta_pp"] - BOOKED_Y4["median"]) < 0.05, \
        f"median drifted: {d['median_delta_pp']:.3f} vs {BOOKED_Y4['median']}"
    #: THE SENTENCE THE PANEL HAS TO NOT BREAK. "Four negative pairs including
    #: both Mamba architectures" is true at a -1pp threshold and false at zero,
    #: where ten pairs are negative. Both halves asserted.
    below = [r for r in rows if r["delta_pp"] < BOOKED_Y4_THRESHOLD]
    assert len(below) == BOOKED_Y4_BELOW, \
        f"{len(below)} pairs below {BOOKED_Y4_THRESHOLD}pp, not {BOOKED_Y4_BELOW}"
    mambas = [r for r in below if "mamba" in r["aligned_model"].lower()]
    assert len(mambas) == 2, \
        (f"{len(mambas)} Mamba architectures below the threshold, not 2; the "
         "panel says both are among the four")

    df = pd.DataFrame(rows)
    df["y"] = range(len(df))
    df["short"] = [r["aligned_model"].split("/")[-1] for r in rows]
    df["col"] = ["#b03030" if v < BOOKED_Y4_THRESHOLD else
                 "#c9c9c9" if v < 1.0 else "#1f4e79" for v in df.delta_pp]
    #: name only the tails: the four below the threshold and the four §4 names
    named = set(df.nlargest(4, "delta_pp").short) | set(
        r["aligned_model"].split("/")[-1] for r in below)
    lab = df[df.short.isin(named)].copy()
    lab["lx"] = lab.delta_pp + [0.45 if v > 0 else -0.45 for v in lab.delta_pp]
    lab["ha"] = ["left" if v > 0 else "right" for v in lab.delta_pp]

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.5)
        + geom_vline(xintercept=BOOKED_Y4_THRESHOLD, linetype="dashed",
                     color="#b03030", size=0.4)
        + geom_segment(df, aes(0, "y", xend="delta_pp", yend="y", color="col"),
                       size=0.55, alpha=0.55)
        + geom_point(df, aes("delta_pp", "y", color="col"), size=2.6)
        + geom_text(lab, aes("lx", "y", label="short", ha="ha"), size=6.4,
                    color="#333333")
        + scale_color_identity()
        + scale_x_continuous(limits=(-9.5, 21),
                             breaks=[-4, -2, -1, 0, 2, 4, 6, 8, 10, 12, 14, 16])
        + scale_y_continuous(breaks=[], limits=(-0.8, len(df) - 0.2))
        + labs(
            title="The superego shift is heterogeneous, and the heterogeneity is the finding",
            subtitle=(
                "Change in the rate at which a passage carries a `<guilt>` span, aligned minus base,\n"
                "one dot per model pair over 32 pairs. Coding pass A, which is the population section 4\n"
                "prints; the span over all passes gives AmberSafe +11.4 and a median of +0.5, and the\n"
                "broader `guilt_or_shame` FIELD gives +12.3 -- neither is this figure.\n"
                "NO SUMMARY LINE IS DRAWN ACROSS THESE DOTS, and that is deliberate. The finding's own\n"
                "sentence is that heterogeneity is the object, so a mean over a distribution whose\n"
                "SPREAD is the claim would invite exactly the reading it exists to refuse. The median\n"
                "is +0.8pp and it is stated here rather than drawn.\n"
                "A 20-POINT SPREAD ON A SUB-POINT MEDIAN. AmberSafe moves +15.5pp; the middle of the\n"
                "roster moves by less than a point in either direction.\n"
                "THE DASHED LINE AT -1pp IS RECOVERED, NOT DECLARED. Section 4 says four negative pairs\n"
                "including both Mamba architectures. TEN pairs are below zero; exactly four are below\n"
                "-1pp, and both Mambas are among those four. The sentence is true at that threshold and\n"
                "false at zero, so the threshold is drawn and every pair is shown with its own value\n"
                "rather than the four being filtered out and presented as the negatives."),
            x="change in `<guilt>` span rate, aligned minus base (percentage points)",
            y="",
            caption=(
                "Producer: meta/M01_displacement/scripts/plot_y_figs.py from\n"
                "results/y_guilt_heterogeneity.json (producer y_guilt_heterogeneity.py, lacan,\n"
                "af23eef8), which exists because its own source is 143 MB and gitignored: a figure\n"
                "drawn from that directly would inherit an input no other seat can fetch.\n"
                "Asserted before drawing: 32 pairs; the population is pass A; each of section 4's four\n"
                "named pairs within 0.1pp; the median within 0.05pp; exactly four pairs below -1pp; and\n"
                "that both Mamba architectures are among those four.\n"
                "THE POPULATION IS THE WHOLE DIFFICULTY AND IT WAS NOWHERE STATED. Field, span-over-all\n"
                "and span-on-pass-A give three different answers with the same median to a decimal --\n"
                "+12.3, +11.4 and +15.5 for AmberSafe -- so a reader reproducing section 4 from the\n"
                "obvious artifact lands on a number that is wrong in the tails and right in the middle."),
        )
        + theme_minimal()
        + theme(figure_size=(12.4, 7.4),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "y4_superego_heterogeneity.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {len(rows)} pairs, pass {d['pass']}, median {d['median_delta_pp']:+.2f}pp, "
          f"spread {d['spread_pp']:.1f}")
    print(f"    {d['n_negative']} below zero, {len(below)} below {BOOKED_Y4_THRESHOLD}pp, "
          f"{len(mambas)} of them Mamba")
    print(f"    no summary line drawn; median stated in the subtitle")
    return out


REGISTRY = {
    "y3_filter": y3_filter_predicts_the_flat_ones,"y_dissoc": y_dissoc,
    "y4_heterogeneity": y4_heterogeneity}


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
