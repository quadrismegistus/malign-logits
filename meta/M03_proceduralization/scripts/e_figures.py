#!/usr/bin/env python
"""Figures for Findings E (E_lexical_arm_contrast.md).

    uv run python meta/M03_proceduralization/scripts/e_figures.py
    uv run python meta/M03_proceduralization/scripts/e_figures.py survivors
    uv run python meta/M03_proceduralization/scripts/e_figures.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.
Naming follows this folder's descriptive style (`f_figures.py` ->
arm_verbs.png, dominance_ladder.png), not a numbered or per-letter scheme.

THE GEOMETRY IS THE ARGUMENT
----------------------------
E's claim is "degree, not kind": the same operation applied to both
speakers, harder to one. Put the individual arm's delta on x and the
institutional arm's on y and that claim becomes a location:

    quadrant 1 (both +)   rises in both   -- DEGREE
    quadrant 3 (both -)   falls in both   -- DEGREE
    quadrants 2 and 4     one rises, the other falls  -- KIND

So "degree not kind" is "almost everything is in Q1 or Q3", and the
distance from the y = x diagonal is how much more the operation bit on
one speaker than the other. Nothing has to be asserted that the reader
cannot see.

TWO POPULATIONS, AND THE SHORTLIST CONFLATES THEM
-------------------------------------------------
plot-debt's shortlist item 5 asks for "324 verbs on indiv-vs-inst axes
... 65 Bonferroni survivors labelled". Those are different populations.
The 65 survivors come from the **702 words** tested (present in >= 40
lineages), of which only 58 are verbs; the 324 is the count of VERBS
measured in both arms, which is the population behind the Pearson 0.909
in section 3.

`b_word_delta_by_word.csv` has 702 rows and **no part-of-speech column**,
so the 324-verb population cannot be drawn from the named artifact at
all. This figure draws the 702 and says so. Referred rather than
silently resolved.

DO NOT CONFUSE THIS FILE WITH `c_word_delta_by_word.csv`
--------------------------------------------------------
plot-debt carries a standing fence: `c_word_delta_by_word.csv` must NOT
be plotted as institutional-vs-narrative vocabulary, because B_C section
6 and D section 7 record that axis as form-confounded. This figure uses
`b_word_delta_by_word.csv`, a different file with a different contrast
(individual vs institutional SPEAKER). The names differ by one
character and the fence applies to only one of them.

THE FOUR REVERSALS ARE COLOURED AND ARE NOT CLAIMED
---------------------------------------------------
Four of the 65 survivors sit in the off-diagonal quadrants. The
shortlist says "four reversals coloured", and colouring four points out
of 702 invites a reader to take them as established reversals. **They
are not tested as reversals.** Bonferroni survival here is on the
difference `d` between arms, not on the sign flip; and section 3 records
ZERO SIGNIFICANT REVERSALS across the 324 verbs measured in both arms.
The panel says both things where the four points are.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
SRC = os.path.join(RESULTS, "b_word_delta_by_word.csv")

#: Booked in E_lexical_arm_contrast.md sections 2 and 3.
BOOKED = {
    "tested": 702,
    "p05": (276, 170, 106), "p01": (189, 124, 65), "bonf": (65, 43, 22),
    "patterns": {"falls in both": 35, "rises in both": 24,
                 "rises individual, falls institutional": 3,
                 "falls individual, rises institutional": 1,
                 "flat on one side": 2},
}


def _pattern(a, b):
    if a == 0 or b == 0:
        return "flat on one side"
    if a < 0 and b < 0:
        return "falls in both"
    if a > 0 and b > 0:
        return "rises in both"
    if a > 0 and b < 0:
        return "rises individual, falls institutional"
    return "falls individual, rises institutional"


def survivors():
    """E sections 3-4: degree, not kind, read off the geometry.

    702 words on individual-vs-institutional axes. The y = x diagonal is
    "the operation bit equally"; the quadrants separate degree from kind.
    """
    from plotnine import (aes, element_text, geom_abline, geom_hline,
                          geom_point, geom_text, geom_vline, ggplot, labs,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    d = pd.read_csv(SRC)
    assert len(d) == BOOKED["tested"], \
        f"words tested drifted: {len(d)} vs booked {BOOKED['tested']}"
    #: OUTWARD CLAIM, MADE TESTED (lacan's rule, [5978]). The docstring says
    #: the shortlist's 324-verb population cannot be drawn from this artifact
    #: because it carries no part-of-speech column. That was prose about a
    #: file, checkable in the run and unchecked. It is now an assert: if a POS
    #: column ever appears, the reason this figure draws 702 words stops being
    #: true and the producer says so instead of the docstring quietly lying.
    pos_like = [c for c in d.columns
                if c.lower() in {"pos", "tag", "upos", "xpos", "part_of_speech"}]
    assert not pos_like, (
        f"a part-of-speech column appeared ({pos_like}); the docstring's reason "
        "for drawing 702 words rather than the shortlist's 324 verbs is now stale")
    thr = 0.05 / len(d)

    for key, t in (("p05", 0.05), ("p01", 0.01), ("bonf", thr)):
        s = d[d.p < t]
        got = (len(s), int((s.median_d > 0).sum()), int((s.median_d < 0).sum()))
        assert got == BOOKED[key], f"{key} drifted: {got} vs booked {BOOKED[key]}"

    d["surv"] = d.p < thr
    d["pattern"] = [_pattern(a, b) for a, b in
                    zip(d.median_delta_indiv, d.median_delta_inst)]
    counts = d[d.surv].pattern.value_counts().to_dict()
    assert counts == BOOKED["patterns"], \
        f"survivor patterns drifted: {counts} vs booked {BOOKED['patterns']}"

    rev = d[d.surv & d.pattern.str.contains("individual, ")]
    assert len(rev) == 4, f"reversals drifted: {len(rev)} vs booked 4"

    d["state"] = np.where(~d.surv, "tested, not a survivor",
                          np.where(d.pattern.str.contains("individual, "),
                                   "survivor, REVERSES", "survivor"))
    #: LIMITS CHOSEN SO NO SURVIVOR IS CUT. The full range is driven by a
    #: handful of extreme non-survivors (|max| 0.01729) which crush the 65
    #: survivors into about a seventh of the panel and make the quadrant
    #: argument unreadable. At +/-0.004 every one of the 65 is inside and
    #: exactly 8 of 702 fall outside, all of them grey. The count is on the
    #: panel; a limit that dropped a survivor would be a different figure.
    lim = 0.004
    n_out = int(((d.median_delta_indiv.abs() > lim)
                 | (d.median_delta_inst.abs() > lim)).sum())
    n_out_surv = int(((d.surv) & ((d.median_delta_indiv.abs() > lim)
                                  | (d.median_delta_inst.abs() > lim))).sum())
    assert n_out_surv == 0, f"{n_out_surv} survivor(s) outside the limits"

    p = (
        ggplot()
        + geom_abline(slope=1, intercept=0, color="#b03030", linetype="dashed",
                      size=0.5)
        + geom_hline(yintercept=0, color="#999999", size=0.35)
        + geom_vline(xintercept=0, color="#999999", size=0.35)
        + geom_point(d[d.state == "tested, not a survivor"],
                     aes("median_delta_indiv", "median_delta_inst"),
                     color="#c9c9c9", size=1.0, alpha=0.55)
        + geom_point(d[d.state == "survivor"],
                     aes("median_delta_indiv", "median_delta_inst"),
                     color="#1f4e79", size=2.1, alpha=0.85)
        + geom_point(rev, aes("median_delta_indiv", "median_delta_inst"),
                     color="#b03030", size=3.4)
        + geom_text(rev, aes("median_delta_indiv", "median_delta_inst",
                             label="word"), size=6.4, color="#b03030",
                    nudge_y=lim * 0.045, ha="center")
        + scale_x_continuous(limits=(-lim, lim))
        + scale_y_continuous(limits=(-lim, lim))
        + labs(
            title="Degree, not kind: the same operation on both speakers, harder to one",
            subtitle=(
                "702 words present in at least 40 lineages. x is the word's median delta in the "
                "INDIVIDUAL arm, y in the INSTITUTIONAL arm.\n"
                "Blue are the 65 Bonferroni survivors (p < 7.12e-05 over the whole vocabulary); "
                "grey are tested and not surviving.\n"
                "THE QUADRANTS SEPARATE DEGREE FROM KIND. Both-positive or both-negative means the "
                "operation went the same way on both speakers and differed only in how far;\n"
                "the off-diagonal quadrants mean it went opposite ways. 59 of the 65 survivors are in "
                "the same-direction quadrants (35 fall in both, 24 rise in both).\n"
                "THE FOUR RED POINTS ARE NOT TESTED AS REVERSALS. Bonferroni survival is on the "
                "DIFFERENCE between arms, not on the sign flip, and the finding records\n"
                "ZERO SIGNIFICANT REVERSALS across the 324 verbs measured in both arms. They are "
                "named because the finding names them, not because they are established.\n"
                "READ THE PATTERN, NOT THE SIGN: d > 0 arises either because a word rises more in the "
                "institutional arm or because it falls less there.\n"
                f"Axes bounded at +/-{lim}: {n_out} of 702 fall outside, NONE of them a survivor. "
                "Source b_word_delta_by_word.csv, NOT c_word_delta_by_word.csv (fenced as "
                "form-confounded)."),
            x="median delta, INDIVIDUAL arm",
            y="median delta, INSTITUTIONAL arm",
            caption=("Producer: meta/M03_proceduralization/scripts/e_figures.py from "
                     "results/b_word_delta_by_word.csv.\n"
                     "Asserted before drawing: 702 tested; 276/189/65 at p<0.05, p<0.01 and "
                     "Bonferroni with their institutional/individual splits; and the survivors' full "
                     "pattern breakdown 35/24/3/1/2.\n"
                     "The shortlist's \"324 verbs\" is a different population and the source carries "
                     "no part-of-speech column, so the 702 tested words are drawn instead."),
        )
        + theme_minimal()
        + theme(figure_size=(11.4, 8.6),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.1, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"))
    )
    out = os.path.join(FIGURES, "e_survivor_scatter.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    702 tested, 65 survivors (43 inst / 22 indiv), "
          f"{len(rev)} reversals: {', '.join(sorted(rev.word))}")
    print(f"    axes bounded at +/-{lim}: {n_out} of 702 outside, "
          f"{n_out_surv} survivors outside")
    return out


FIGURES_REGISTRY = {"survivors": survivors}


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
