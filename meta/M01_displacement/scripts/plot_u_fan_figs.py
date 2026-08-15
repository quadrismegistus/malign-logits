#!/usr/bin/env python
"""Figure 4: one ablation suite, two instruments, two answers.

    uv run python meta/M01_displacement/scripts/plot_u_fan_figs.py
    uv run python meta/M01_displacement/scripts/plot_u_fan_figs.py two_instruments
    uv run python meta/M01_displacement/scripts/plot_u_fan_figs.py --list

Plotting regime: plotnine at 300 dpi, output to ../figures/, slice in the
subtitle, booked-number asserts before drawing. Case 1 by shape -- reads two
committed artifacts and writes only pixels, so re-running it is a re-render.

THE Y-AXIS IS THE ARGUMENT
--------------------------
A 0-100 benchmark score and a divergence of ~0.065 have no common unit, and
plotting them at their own scales side by side would let the axis ranges do
the arguing. Both panels are therefore percent-of-full-mix against a dashed
100% reference, with the absolute values printed on the bars. Proportional
comparison is the legitimate frame when units differ; printing the absolutes
means nobody has to take on trust that the scaling was not chosen to flatter.

Bars start at zero. A truncated bar axis would turn panel A's 80% into a
collapse and panel B's flat row into whatever the crop implied.

PANEL A IS TWO BARS AND THAT IS A LIMIT, NOT A FINDING
--------------------------------------------------------
The behavioral scores are not measured in this campaign and were not in this
repository before 2026-08-14; see `results/u_tulu_published_safety.json` for
what is known about them and what is merely attributed. Two arms were
supplied. **Whether the published table reports the other three is unverified
from here**, so the panel draws two and says the rest were not available to
it, rather than implying a full grid or asserting the values do not exist.

PANEL B ASSERTS A NULL, SO IT CARRIES ITS RESOLUTION
------------------------------------------------------
`U_ladder.md` §4 concludes the four ablations are interchangeable. A flat row
of four bars with no interval invites "flat at what resolution?", and the
campaign's standard is no null without its minimum detectable effect. The
error bars are paired bootstrap intervals on the RATIO (10,000 resamples,
`u_fan_ci.py`), and the caption carries the paired MDE.

AND THE NULL IS NOT CLEAN, WHICH THE PANEL SAYS
-------------------------------------------------
Two of the six pairwise comparisons are significant: `no-persona` sits below
both `no-safety` and `no-wildchat`. **The finding's actual claim survives** --
`no-math` against `no-safety` is not significant, so removing safety data
still costs what removing maths data costs -- but "the four are
interchangeable" is too strong and the figure should not launder it. The
differences are under 2% of the full-mix effect against panel A's 20%.
"""
import argparse
import csv
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
FIGURES = os.path.join(M01, "figures")

#: Booked in results/t_fans.csv, quoted in findings/U_ladder.md section 4.
#: The panel plots these, so the finding and the figure cannot disagree.
BOOKED_JS = {"full": 0.0651458275316188,
             "no-math": 0.05763676859277951,
             "no-persona": 0.057236900446879214,
             "no-safety": 0.05831617631514225,
             "no-wildchat": 0.05843756496427438}
BOOKED_JACCARD_FULL_NOSAFETY = 0.53385598620959
#: Faller Jaccard over the whole data fan, results/t_fans_jaccard.csv. The
#: SECOND axis, and the one this figure does not draw: panel B measures HOW
#: MUCH probability moves, this measures WHICH WORDS move, and the two break
#: "interchangeable" by different arms. Every pair involving no-wildchat sits
#: below every pair that does not, with no overlap.
JACC_WITH_WILDCHAT = (0.2940085059082473, 0.3402192716421085)
JACC_WITHOUT_WILDCHAT = (0.4855350182418757, 0.562845938083337)

#: display order: full first, then the ablation whose two readings disagree
ORDER = [("full", "Full mix"), ("no-safety", "−Safety"),
         ("no-math", "−Math"), ("no-persona", "−Persona"),
         ("no-wildchat", "−WildChat")]

GREY, ACCENT = "#9a9a9a", "#6a3d9a"


def two_instruments():
    """Fig 4: the Tulu 3 data ablations scored behaviorally and distributionally."""
    from plotnine import (aes, element_blank, element_text, facet_wrap,
                          geom_col, geom_errorbar, geom_hline, geom_text,
                          ggplot, labs, scale_fill_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    ci = json.load(open(os.path.join(RESULTS, "u_fan_ci.json")))
    beh = json.load(open(os.path.join(RESULTS, "u_tulu_published_safety.json")))

    #: the artifacts must still say what the panel says
    for k, v in BOOKED_JS.items():
        got = ci["booked_means_2026_08_06"][k]
        assert abs(got - v) < 1e-12, f"{k} booked JS drifted: {got} vs {v}"
    for k in BOOKED_JS:
        pb = 100 * BOOKED_JS[k] / BOOKED_JS["full"]
        pn = ci["per_arm"][k]["pct_of_full"]
        assert abs(pb - pn) < 0.1, \
            (f"{k}: booked percent-of-full {pb:.2f} vs recomputed {pn:.2f}; "
             "the bars are drawn from the booked ratio and the intervals from "
             "the recomputation, so these must agree or the panel is mixed")
    assert set(beh["scores"]) == {"full", "no-safety"}, \
        f"behavioral arms changed: {sorted(beh['scores'])}"
    assert beh["scores"]["full"] == 93.1 and beh["scores"]["no-safety"] == 74.7, \
        f"behavioral scores drifted: {beh['scores']}"
    jr = [r for r in csv.DictReader(open(os.path.join(RESULTS, "t_fans_jaccard.csv")))
          if r["fan"] == "data"]
    wc = [float(r["faller_jaccard"]) for r in jr if "no-wildchat" in (r["a"], r["b"])]
    ow = [float(r["faller_jaccard"]) for r in jr if "no-wildchat" not in (r["a"], r["b"])]
    assert (round(min(wc), 6), round(max(wc), 6)) == \
        (round(JACC_WITH_WILDCHAT[0], 6), round(JACC_WITH_WILDCHAT[1], 6)), \
        f"wildchat Jaccard range drifted: {min(wc)}-{max(wc)}"
    assert max(wc) < min(ow), \
        (f"the two Jaccard groups now OVERLAP ({max(wc)} vs {min(ow)}); the caption "
         "claims no overlap and would be wrong")
    n_sig = sum(1 for v in ci["paired"].values() if v["significant"])
    assert n_sig == 2, \
        (f"{n_sig} significant pairwise differences, not 2; the subtitle names "
         "which two and would be wrong")

    pa, pb_ = "A. BEHAVIORAL\nsafety score reported by the model's authors", \
              "B. DISTRIBUTIONAL\nJS against the base model, measured here"
    rows = []
    for i, (key, lab) in enumerate(ORDER):
        if key in beh["scores"]:
            v = beh["scores"][key]
            rows.append({"panel": pa, "x": i, "lab": lab,
                         "pct": 100 * v / beh["scores"]["full"],
                         "absval": f"{v:.1f}", "lo": None, "hi": None,
                         "fill": ACCENT if key == "no-safety" else GREY,
                         "missing": ""})
        else:
            rows.append({"panel": pa, "x": i, "lab": lab, "pct": 0.0,
                         "absval": "", "lo": None, "hi": None, "fill": GREY,
                         "missing": "not supplied\nto this figure"})
        a = ci["per_arm"][key]
        rows.append({"panel": pb_, "x": i, "lab": lab,
                     "pct": 100 * BOOKED_JS[key] / BOOKED_JS["full"],
                     "absval": f"{BOOKED_JS[key]:.4f}",
                     "lo": a["pct_ci_lo"], "hi": a["pct_ci_hi"],
                     "fill": ACCENT if key == "no-safety" else GREY,
                     "missing": ""})
    d = pd.DataFrame(rows)
    d["panel"] = pd.Categorical(d.panel, categories=[pa, pb_], ordered=True)
    bars = d[d.absval != ""]
    errs = d[d.lo.notna()]
    gaps = d[d.missing != ""]

    mde = ci["mde_paired_pct_of_full"]
    biggest = ci["max_abs_paired_diff_pct_of_full"]

    p = (
        ggplot()
        + geom_hline(yintercept=100, linetype="dashed", color="#555555", size=0.45)
        + geom_col(bars, aes("x", "pct", fill="fill"), width=0.66)
        + geom_errorbar(errs, aes("x", ymin="lo", ymax="hi"), width=0.18,
                        size=0.5, color="#333333")
        + geom_text(bars, aes("x", "pct", label="absval"), va="bottom",
                    nudge_y=2.2, size=7.2, color="#222222")
        + geom_text(gaps, aes("x", 6, label="missing"), size=6.2,
                    color="#8a8a8a", lineheight=1.2)
        + scale_fill_identity()
        + scale_x_continuous(breaks=list(range(len(ORDER))),
                             labels=[l for _, l in ORDER])
        + scale_y_continuous(limits=(0, 112),
                             breaks=[0, 25, 50, 75, 100],
                             labels=["0%", "25%", "50%", "75%", "100%"])
        + facet_wrap("~panel", nrow=1)
        + labs(
            #: THE TITLE COUNTED THE INSTRUMENTS AND THE COUNT WENT STALE.
            #: "two instruments, two answers" was true when drawn and stopped
            #: being true when X section 4b landed. A figure that says TWO is
            #: not merely omitting a third, it is DENYING it (registrar,
            #: [6190]) -- so the title no longer enumerates, and says which
            #: instruments are ON THIS PANEL rather than how many exist.
            title="One ablation suite, two instruments on this panel and two more that disagree with the right-hand one",
            subtitle=(
                "The Tulu 3 SFT data ablations. One base (Llama-3.1-8B), one recipe, five training sets,\n"
                "everything held fixed but the corpus. Both panels are percent of the full data mix, with\n"
                "the absolute values printed, because a 0-100 benchmark score and a divergence of 0.065\n"
                "have no common unit and raw side-by-side axes would let the scaling do the arguing.\n"
                "LEFT: the authors' behavioral safety score. Removing safety data costs a fifth of it.\n"
                "These values are NOT measured here and were not in this repository before 2026-08-14;\n"
                "two arms were supplied and whether the published table reports the other three is\n"
                "unverified from here, so the panel declares them missing rather than absent.\n"
                "RIGHT: the distributional displacement measured on the same released checkpoints.\n"
                "Every ablation costs about 10 percent, and removing safety costs what removing maths\n"
                "costs. Bars are the booked 2026-08-06 values over 2,182 cells; error bars are paired\n"
                "bootstrap intervals on the ratio from a 2,174-cell recomputation -- the prompt catalogue\n"
                "data/prompt_categorisation.json moved on 08-10, four days after the fan was measured,\n"
                "and is read at run time by a module unchanged since 07-30 -- the two agreeing on every\n"
                "ratio to better than 0.03 points.\n"
                "THE NULL IS NOT CLEAN AND IS NOT LAUNDERED HERE: 2 of 6 pairwise comparisons are\n"
                "significant, both involving −Persona, at under 2 percent of the full-mix effect.\n"
                "TWO OTHER INSTRUMENTS DISAGREE WITH THE RIGHT-HAND PANEL, AND THEY AGREE WITH EACH\n"
                "OTHER. X section 1 (K-norms, 2,583 prompts) puts −Safety at 23% of the full mix's\n"
                "reduction, sign p 0.028, against −Math at p 0.13; X section 4b (projected displacement,\n"
                "39 never-scored items, run 2026-08-15) makes −Safety the only arm excluding zero and\n"
                "−Math null. Built independently, they land on the same quarter for safety, 23% and\n"
                "27.3%, while disagreeing with each other about −WildChat.\n"
                "SO THE DISAGREEMENT HAS A LOCATION: it is safety-against-maths. This panel has them\n"
                "equal at 89.5% and 88.5%; both of those instruments separate them.\n"
                "NEITHER IS DRAWN HERE, BECAUSE A THIRD COLUMN WOULD ASSERT COMMENSURABILITY -- that\n"
                "these are measurements of one thing. They are not: JS over the union support with the\n"
                "residual retained is not a signed projection onto a pole axis that excludes the\n"
                "residual entirely, and neither is a K-weighted probability mass.\n"
                "AND THIS PANEL CLAIMS NO EVIDENTIAL ADVANTAGE OVER THEM. 2,182 prompts against 39\n"
                "items is a difference in WITHIN-ARM PRECISION and not in replication: the unit that\n"
                "licenses a causal claim about removing a corpus is the TRAINING RUN, and there are\n"
                "four of them here exactly as there. The 1-in-4 arm-level ceiling X states for itself\n"
                "sits on this panel too."),
            x="", y="percent of the full data mix",
            caption=(
                "Producers: meta/M01_displacement/scripts/plot_u_fan_figs.py from results/u_fan_ci.json\n"
                "(producer u_fan_ci.py, 10,000 paired bootstrap resamples, seed 20260814) and\n"
                "results/u_tulu_published_safety.json. JS is word-level over the union support with the\n"
                "residual retained, base -> arm, on ACTIVE English prompts deduplicated on the string.\n"
                f"MINIMUM DETECTABLE EFFECT, paired: {mde:.2f}% of the full-mix effect (widest 95% "
                f"half-width over the six\npairwise comparisons). Largest observed difference among the "
                f"ablations: {biggest:.2f}%.\n"
                "So the right panel is flat to within about 2 percent of the effect, and that is the\n"
                "resolution at which 'interchangeable' is being claimed.\n"
                "Asserted before drawing: the five booked JS values, that booked and recomputed\n"
                "percent-of-full agree within 0.1 points, the two behavioral scores, and that exactly\n"
                "two pairwise comparisons are significant.\n"
                "NOT movement_cells.js_total, which is the movement decomposition's total, a different\n"
                "quantity returning 2,199 cells and means ~18% lower.\n"
                f"THE MOVED WORDS MOSTLY STAY THE SAME: faller Jaccard between the full mix and −Safety "
                f"is {BOOKED_JACCARD_FULL_NOSAFETY:.3f},\nagainst 0.02 to 0.04 between rungs of the "
                "training ladder. Change the corpus and you get the same operation\non the same words; "
                "change the rung and you do not.\n"
                "BUT 'INTERCHANGEABLE' FAILS TWICE, BY DIFFERENT ARMS, ON DIFFERENT QUESTIONS, and this "
                "panel shows only one\nof them. HOW MUCH probability moves is what is drawn: flat to "
                f"~{mde:.1f}%, and only −Persona separates. WHICH WORDS move is\nnot drawn, and there "
                f"−WildChat alone is categorical: every pair involving it sits at "
                f"{JACC_WITH_WILDCHAT[0]:.3f}–{JACC_WITH_WILDCHAT[1]:.3f}\nand every pair without it at "
                f"{JACC_WITHOUT_WILDCHAT[0]:.3f}–{JACC_WITHOUT_WILDCHAT[1]:.3f}, with no overlap. "
                "A reader meeting only this panel would\nconclude −WildChat is unremarkable, and on the "
                "other axis it is the one arm that is not."),
        )
        + theme_minimal()
        + theme(figure_size=(13.6, 7.6),
                plot_title=element_text(size=11, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                strip_text=element_text(size=8.4, weight="bold"),
                axis_text_x=element_text(size=8.2),
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
                panel_spacing=0.06)
    )
    out = os.path.join(FIGURES, "u_fan_two_instruments.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    behavioral: {beh['scores']['full']} -> {beh['scores']['no-safety']} "
          f"= {100 * beh['scores']['no-safety'] / beh['scores']['full']:.1f}% of full")
    print(f"    distributional: four ablations at "
          f"{min(ci['per_arm'][k]['pct_of_full'] for k, _ in ORDER[1:]):.1f}-"
          f"{max(ci['per_arm'][k]['pct_of_full'] for k, _ in ORDER[1:]):.1f}% of full; "
          f"MDE {mde:.2f}%, {n_sig} of 6 pairs significant")
    print(f"    faller Jaccard full vs no-safety: {BOOKED_JACCARD_FULL_NOSAFETY:.3f} "
          "(prose, not a third panel)")
    return out


REGISTRY = {"two_instruments": two_instruments}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:18s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    names = a.names or list(REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    os.makedirs(FIGURES, exist_ok=True)
    for n in names:
        print(f"{n}:")
        REGISTRY[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
