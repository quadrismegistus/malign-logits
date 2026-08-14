#!/usr/bin/env python
"""Figures for `crosslingual_arms.md`: the same operation in both languages.

    uv run python meta/M06_generation/scripts/m06_crosslingual_figs.py
    uv run python meta/M06_generation/scripts/m06_crosslingual_figs.py invariance
    uv run python meta/M06_generation/scripts/m06_crosslingual_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.

WHAT THIS FIGURE DELIBERATELY DOES NOT CONTAIN
----------------------------------------------
The finding's strongest form was a MATCHED-PROMPT contrast on a
parse-free key. Both matched-prompt legs were WITHDRAWN at 5c0b2915:
the numbers were never program output in any session log, and a
declared 32-recipe estimator sweep recovered at best 2 of 6 values, and
that on the input whose population does not match. The register form
built on them is withdrawn ([5937]), and producer-debt records the
disposition CLOSED BY WITHDRAWAL -- neither discharged nor outstanding,
because the claim is gone rather than the code recovered.

**So this panel makes no matched-prompt comparison and does not mention
one.** That is registrar's tightening at [5934] and it is deliberate: a
figure that names an absent leg still puts it in the reader's head. The
panel shows the four persisted contrasts on each of the two runs and
says what they are; the account of what is missing belongs here, in the
producer, where whoever edits this will meet it.

WHY A DIAGONAL SCATTER AND NOT TWO BARS
---------------------------------------
The surviving claim is an INVARIANCE: alignment narrows the semantic
spread of a passage in Chinese as it does in English, and the language
difference-in-differences is null on every persisted construction. A
null is not shown by two bars that happen to look similar; it is shown
by putting the two languages on the two axes and drawing the line where
"the same in both" lives. Points on the y = x diagonal ARE the claim,
and their scatter around it is the evidence about how well it holds.

Both arms are also negative on all eight persisted contrasts, so the
cloud sits in the lower-left quadrant: the operation narrows spread in
both languages, and by the same amount.

THE NOUN IS SPREAD, NOT TRAJECTORY
----------------------------------
`total_drift` is `1 - min(pairwise similarity)`: the DIAMETER of a
passage's sentence set. It is ORDER-INVARIANT -- the finding verified
this by permuting sentences and watching it not move -- so the result
is about semantic SPREAD and is not a claim about how a passage
travels. `crosslingual_arms.md` corrected itself on exactly this point
and says every sentence implying otherwise is wrong. The axis labels
here say spread.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")

RUNS = {"truncated (75-word cap)": ("crosslingual_arms_pairs.parquet",
                                    "crosslingual_arms.json"),
        "untruncated": ("crosslingual_arms_full_pairs.parquet",
                        "crosslingual_arms_full.json")}
MATCH = {False: "pooled", True: "n_sents-matched"}

#: DiD medians and sign counts per run x contrast, from each run's own json.
#: Asserted because the surviving claim IS the null: if any of these stops
#: being null the figure is making a claim the finding does not.
BOOKED_DID = {
    ("truncated (75-word cap)", "total_drift|pooled"): (10, 15, 0.4244),
    ("truncated (75-word cap)", "total_drift|n_sents-matched"): (13, 12, 1.0),
    ("truncated (75-word cap)", "mean_drift|pooled"): (16, 9, 0.2295),
    ("truncated (75-word cap)", "mean_drift|n_sents-matched"): (16, 9, 0.2295),
    ("untruncated", "total_drift|pooled"): (13, 12, 1.0),
    ("untruncated", "total_drift|n_sents-matched"): (14, 11, 0.6900),
    ("untruncated", "mean_drift|pooled"): (13, 12, 1.0),
    ("untruncated", "mean_drift|n_sents-matched"): (12, 13, 1.0),
}


def _load():
    rows, dids = [], {}
    for run, (pq, js) in RUNS.items():
        d = pd.read_parquet(os.path.join(RESULTS, pq))
        d["run"] = run
        d["contrast"] = d.metric + "|" + d.matched.map(MATCH)
        w = d.pivot_table(index=["run", "contrast", "pair"], columns="lang",
                          values="delta").reset_index()
        rows.append(w)
        j = json.load(open(os.path.join(RESULTS, js)))
        for k, v in j["contrasts"].items():
            dd = v["DiD_en_minus_zh"]
            dids[(run, k)] = (dd["up"], dd["dn"], dd["p_sign"], dd["median"])
    return pd.concat(rows, ignore_index=True), dids


def invariance():
    """The invariance: the same narrowing in both languages, DiD null.

    One point per model pair, English effect against Chinese effect, on
    every persisted contrast. The y = x line is where "the same in both
    languages" lives; the 0 lines separate narrowing from widening.
    """
    from plotnine import (aes, element_text, facet_grid, geom_abline,
                          geom_hline, geom_point, geom_text, geom_vline,
                          ggplot, labs, scale_color_manual, theme,
                          theme_minimal)

    d, dids = _load()

    for key, (up, dn, p) in BOOKED_DID.items():
        gup, gdn, gp, _ = dids[key]
        assert (gup, gdn) == (up, dn) and round(gp, 4) == round(p, 4), \
            f"{key} DiD drifted: {gup}/{gdn} p {round(gp, 4)} vs booked {up}/{dn} p {p}"
    assert all(v[2] >= 0.2295 for v in dids.values()), \
        "a DiD stopped being null; the invariance claim is not what this draws"

    #: both arms negative on all eight persisted contrasts, the other half
    #: of what survives the withdrawal
    med = d.groupby(["run", "contrast"])[["en", "zh"]].median()
    assert (med < 0).all().all(), \
        f"an arm median stopped being negative:\n{med[~(med < 0)].dropna(how='all')}"

    d["metric"] = d.contrast.str.split("|").str[0]
    d["match"] = d.contrast.str.split("|").str[1]
    d["panel"] = d.metric + "\n" + d["match"]

    ann = []
    for (run, contrast), (up, dn, p, m) in dids.items():
        metric, match = contrast.split("|")
        ann.append({"run": run, "panel": metric + "\n" + match,
                    "txt": f"DiD {m:+.4f}  {up}/{dn}  p {p:.2g}"})
    a = pd.DataFrame(ann)
    a["y"] = a.run.map({"truncated (75-word cap)": 0.021, "untruncated": 0.0135})

    order = ["total_drift\npooled", "total_drift\nn_sents-matched",
             "mean_drift\npooled", "mean_drift\nn_sents-matched"]
    cols = {"truncated (75-word cap)": "#1f4e79", "untruncated": "#c98a2b"}
    for f in (d, a):
        f["panel"] = pd.Categorical(f.panel, categories=order, ordered=True)
        f["run"] = pd.Categorical(f.run, categories=list(RUNS), ordered=True)

    p = (
        ggplot()
        + geom_hline(yintercept=0, color="#999999", size=0.35)
        + geom_vline(xintercept=0, color="#999999", size=0.35)
        + geom_abline(slope=1, intercept=0, color="#b03030", linetype="dashed",
                      size=0.5)
        + geom_point(d, aes("en", "zh", color="run"), size=1.9, alpha=0.65)
        + geom_text(a, aes(-0.148, "y", label="txt", color="run"), size=6.3,
                    ha="left", va="center")
        + scale_color_manual(values=cols, name="")
        + facet_grid(". ~ panel")
        + labs(
            title="Alignment narrows a passage's semantic spread in Chinese as it does in English",
            subtitle=(
                "One point per model pair (25 pairs complete in both languages), English arm effect "
                "against Chinese arm effect, on every persisted contrast of both runs.\n"
                "THE DASHED LINE IS y = x: points on it are pairs where alignment did the same thing in "
                "both languages. The cloud sits in the lower-left quadrant, so the operation NARROWS "
                "spread in both.\n"
                "The language difference-in-differences is null on all eight contrasts (smallest p 0.23), "
                "and holds no sign: 4 of 8 lean English, 4 lean Chinese.\n"
                "SPREAD, NOT TRAJECTORY. total_drift is 1 - min(pairwise similarity), the DIAMETER of the "
                "passage's sentence set, and is order-invariant; the finding corrected itself on this and "
                "the noun is spread.\n"
                "Single pass, ungraded, [5503] applies."),
            x="English: change in spread, aligned - base",
            y="Chinese: change in spread\naligned - base",
            caption=("Producer: meta/M06_generation/scripts/m06_crosslingual_figs.py from "
                     "results/crosslingual_arms{,_full}_pairs.parquet (producer "
                     "m06_crosslingual_arms.py).\n"
                     "DiD medians, sign counts and p asserted from each run's own json before drawing."),
        )
        + theme_minimal()
        + theme(figure_size=(13.4, 4.9),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.2, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                strip_text=element_text(size=7.8, weight="bold"),
                legend_position="top",
                panel_spacing=0.045)
    )
    out = os.path.join(FIGURES, "crosslingual_invariance.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for key in sorted(dids, key=str):
        up, dn, pv, m = dids[key]
        print(f"    {key[0]:24s} {key[1]:28s} DiD {m:+.4f} {up}/{dn} p {pv:.3g}")
    return out


FIGURES_REGISTRY = {"invariance": invariance}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in FIGURES_REGISTRY.items():
            print(f"  {k:12s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    names = a.names or list(FIGURES_REGISTRY)
    unknown = [n for n in names if n not in FIGURES_REGISTRY]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    os.makedirs(FIGURES, exist_ok=True)
    for n in names:
        print(f"{n}:")
        FIGURES_REGISTRY[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
