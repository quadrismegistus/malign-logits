#!/usr/bin/env python
"""Figures for the attention-back finding (attention_back_cross_own.md).

    uv run python meta/M04_syntagmatic/scripts/attn_figures.py
    uv run python meta/M04_syntagmatic/scripts/attn_figures.py cross_own
    uv run python meta/M04_syntagmatic/scripts/attn_figures.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to ../figures/,
slice in the subtitle, booked-number asserts before drawing. Case 1 by shape:
reads committed JSON and writes only pixels.

WHAT THIS FOLDER'S FIGURES ARE ALLOWED TO SAY
---------------------------------------------
plot-debt M04 candidate 8 calls section 3 "the one surviving fact", and the
qualifier is load-bearing. Section 3 also reported an ordering FALLER <
NONMOVER < RISER, monotone in alignment status, and **section 3's own
retraction withdraws it**: the three words' base probabilities are 0.062,
0.089 and 0.201, so on this cell "ordered by alignment status" and "ordered by
how probable the word was" are the same ordering. A second cell where the
faller is 33x more probable than the riser refuses BOTH accounts.

So this panel draws the faller-against-non-mover contrast and the mode
comparison, and it does not draw the three-way ordering at all. A figure is
where a retracted result goes to be revived, because the picture outlives the
paragraph that withdrew it.

THE BAND IS THE UNCERTAINTY OF THE MEDIAN, AND THE FIRST VERSION GOT IT WRONG
------------------------------------------------------------------------------
`cross` returns p = 0.0006 at j=7 on a median of -0.0004. Both are true and
the small p is not evidence of an effect, so the panel needs to show the
medians beside something that makes their size legible.

The first draft drew the interquartile range ACROSS HEADS, reasoning that the
spread is why a tiny median can be significant. That spread is about 0.028 in
`own` where the medians live inside 0.011, so the bands set the y axis and
flattened both lines onto zero: **the uncertainty display destroyed the one
contrast the figure exists to show, and only the rendered image said so.**

What belongs beside a p value is the interval of the quantity that p tests,
which is the median. Bands are now 95% bootstrap intervals for the median over
the 480 heads, and the head-level spread is stated as a number in the subtitle
instead. A dispersion the axis cannot hold gets said, not drawn.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")

#: Section 3's table: faller minus non-mover, paired across 480 heads,
#: norm-weighted, median and Wilcoxon p at six positions.
BOOKED = {
    "cross": {0: (+0.0011, 0.018), 1: (-0.0007, 0.009), 3: (+0.0004, 0.086),
              7: (-0.0004, 0.0006), 15: (+0.0001, 0.41), 31: (-0.0000, 0.33)},
    "own": {0: (-0.0110, 1.1e-19), 1: (-0.0071, 6.4e-12), 3: (-0.0105, 5.3e-21),
            7: (-0.0039, 9.8e-13), 15: (+0.0007, 2.8e-08), 31: (+0.0006, 6.8e-12)},
}
#: The cell. Not a corpus: one prompt, one pair, three words.
BOOKED_CELL = {"pair": "HuggingFaceTB/SmolLM2-360M>HuggingFaceTB/SmolLM2-360M-Instruct",
               "prompt": "sexual_explicit_1", "layers": 32, "heads": 15,
               "window": 32, "n": 24}
FALLER, NONMOVER = "penis", "thumb"
OWN_C, CROSS_C = "#1f4e79", "#9a9a9a"
#: half of the last printed digit: the doc prints four decimals
TOL = 5.1e-05
N_BOOT = 2000


def _diff(mode):
    """(480 heads x 32 positions) faller-minus-non-mover, and the cell's facts."""
    src = os.path.join(RESULTS, f"attn_delta_smollm2_e1_{mode}.json")
    #: NAMED, NEVER GLOBBED. attn_delta_smollm2_e1_cross_w200.json is a
    #: 200-token-window sibling; a glob would make the window a lottery.
    d = json.load(open(src))
    assert d["mode"] == mode, f"{src} says mode {d['mode']!r}"
    for k, v in BOOKED_CELL.items():
        assert d[k] == v, f"{mode}: {k} is {d[k]!r}, not the booked {v!r}"
    f = np.array(d["words"][FALLER]["nw"]["D"])
    n = np.array(d["words"][NONMOVER]["nw"]["D"])
    assert f.shape == (d["layers"], d["heads"], d["window"]), f"shape {f.shape}"
    return (f - n).reshape(-1, f.shape[-1]), d


def cross_own():
    """M04 candidate 8: the two modes disagree, and only one of them is a result."""
    from scipy.stats import wilcoxon
    from plotnine import (aes, element_text, geom_hline, geom_line, geom_point,
                          geom_ribbon, geom_text, ggplot, labs,
                          scale_color_identity, scale_fill_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)
    import pandas as pd

    frames, cells, iqr = {}, {}, {}
    for mode in ("cross", "own"):
        diff, d = _diff(mode)
        cells[mode] = d
        assert diff.shape[0] == BOOKED_CELL["layers"] * BOOKED_CELL["heads"] == 480, \
            f"{mode}: {diff.shape[0]} heads, section 3 pairs across 480"
        #: EVERY BOOKED CELL RE-DERIVED, MEDIAN AND p. The p values matter as
        #: much as the medians here: the panel's argument is that `cross` has
        #: small p AND no magnitude, so a figure that reproduced the medians
        #: while the p values had moved would be making a claim it had not
        #: checked.
        for j, (m, p) in BOOKED[mode].items():
            got = float(np.median(diff[:, j]))
            assert abs(got - m) <= TOL, \
                f"{mode} j={j}: median {got:+.5f} vs section 3's {m:+.4f}"
            _, pp = wilcoxon(diff[:, j])
            assert abs(np.log10(pp) - np.log10(p)) < 0.05, \
                f"{mode} j={j}: p {pp:.2g} vs section 3's {p:.2g}"
        #: THE BAND IS THE UNCERTAINTY OF THE MEDIAN, NOT THE SPREAD OF THE
        #: HEADS, and the first version of this figure got that wrong. An IQR
        #: across heads spans about 0.05 where the medians live inside 0.011,
        #: so drawing it flattened both lines onto zero and destroyed the one
        #: contrast the panel exists to show. The interval that belongs beside
        #: a p value is the interval of the quantity the p value tests.
        rng = np.random.default_rng(0)
        idx = rng.integers(0, diff.shape[0], size=(N_BOOT, diff.shape[0]))
        boot = np.median(diff[idx, :], axis=1)
        frames[mode] = pd.DataFrame({
            "j": np.arange(diff.shape[1]),
            "med": np.median(diff, axis=0),
            "lo": np.percentile(boot, 2.5, axis=0),
            "hi": np.percentile(boot, 97.5, axis=0),
            "mode": mode,
            "col": OWN_C if mode == "own" else CROSS_C})
        #: the head-level spread is stated as a number instead, since it is the
        #: reason the interval is narrow and cannot share the axis
        iqr[mode] = float(np.median(np.percentile(diff, 75, axis=0)
                                    - np.percentile(diff, 25, axis=0)))

    #: THE PANEL'S CLAIM, MADE TESTED. "cross is noise around a null" is drawn
    #: as a flat line and has to be true of the DATA, not only of the six
    #: printed positions: its sign alternates and its whole range is inside a
    #: band where `own` spends most of its length.
    cr, ow = frames["cross"], frames["own"]
    assert cr.med.abs().max() < 0.002, \
        f"cross now reaches {cr.med.abs().max():.4f}; the panel calls it flat"
    assert abs(ow.med.min()) > 5 * cr.med.abs().max(), \
        "own no longer dwarfs cross; the panel's whole contrast is that it does"
    signs = np.sign(cr.med.values)
    flips = int((signs[:-1] * signs[1:] < 0).sum())
    assert flips >= 5, \
        f"cross changes sign only {flips} times; the subtitle calls it alternating"

    d = pd.concat(frames.values())
    marks = pd.concat([
        frames[m].loc[frames[m].j.isin(BOOKED[m])].assign(mode=m)
        for m in ("cross", "own")])
    #: labels at the right end of each line rather than annotations over them:
    #: the first version put two p-value callouts through the lines they
    #: described, which the text audit cannot see because geom_text is not
    #: measured at all
    #: LABELS GO WHERE THE LINES ARE FURTHEST APART, WHICH IS THE LEFT END.
    #: At j=31 the two medians are +0.0006 and -0.0000, so right-end labels
    #: printed on top of each other -- the lines converge exactly where the
    #: reader's eye leaves the panel.
    note = pd.DataFrame([
        {"j": 0.4, "y": -0.0126, "t": "own", "c": OWN_C},
        {"j": 0.4, "y": +0.0024, "t": "cross", "c": "#6f6f6f"}])

    p = (
        ggplot()
        + geom_hline(yintercept=0, color="#333333", size=0.5)
        + geom_ribbon(d, aes("j", ymin="lo", ymax="hi", fill="col"), alpha=0.20)
        + geom_line(d, aes("j", "med", color="col"), size=1.0)
        + geom_point(marks, aes("j", "med", color="col"), size=2.4)
        + geom_text(note, aes("j", "y", label="t", color="c"), size=6.6,
                    ha="left")
        + scale_color_identity()
        + scale_fill_identity()
        + scale_x_continuous(breaks=[0, 1, 3, 7, 15, 31])
        + scale_y_continuous(breaks=[-0.010, -0.005, 0.0, 0.005],
                             labels=["-0.010", "-0.005", "0", "+0.005"])
        + labs(
            title="The same contrast, measured two ways: the effect is in what the aligned model WRITES, not in how it reads",
            subtitle=(
                "Attention paid back to a forced word, aligned minus base, faller MINUS non-mover, median\n"
                "over 480 heads (32 layers x 15) at each position after the word. Dark = OWN, where each\n"
                "arm writes its own continuation; grey = CROSS, where both models read one text. Bands are\n"
                "95% bootstrap intervals FOR THE MEDIAN; dots mark the six positions section 3 tabulates.\n"
                "THE ONLY DIFFERENCE BETWEEN THE MODES IS WHETHER EACH ARM WRITES ITS OWN CONTINUATION,\n"
                "so a result present in `own` and absent in `cross` is not about how the aligned model\n"
                "attends to a given text. `own` binds LESS to a demoted word early and MORE late.\n"
                "SMALL p IS NOT EVIDENCE OF AN EFFECT HERE. `cross` returns p = 0.0006 at j=7 on a\n"
                "median of -0.0004, with the sign alternating position to position. The test asks\n"
                "whether the median differs from zero and answers yes about a quantity that is nearly\n"
                "zero. Read the axis, not the p.\n"
                "THE BANDS ARE THE UNCERTAINTY OF THE MEDIAN AND NOT THE SPREAD OF THE HEADS, whose\n"
                f"interquartile range is about {iqr['own']:.3f} in `own` and {iqr['cross']:.3f} in `cross` --\n"
                "tens of times the medians drawn here. That spread cannot share this axis without\n"
                "flattening both lines onto zero, so it is stated rather than drawn.\n"
                "THE THREE-WAY ORDERING IS RETRACTED AND IS NOT DRAWN. Section 3 reported FALLER <\n"
                "NONMOVER < RISER, monotone in alignment status, and its own retraction withdraws it:\n"
                "the three words' base probabilities are 0.062, 0.089 and 0.201, so on this cell that\n"
                "ordering is indistinguishable from ordering by how probable the word was. A second cell\n"
                "whose faller is 33x more probable than its riser refuses BOTH accounts.\n"
                "ONE CELL. One prompt (`sexual_explicit_1`), one pair (SmolLM2-360M to its Instruct),\n"
                "three words, 24 samples, a 32-token window. Nothing here is a corpus result."),
            x="position after the forced word (j)",
            y="attention-back, aligned minus base:  faller minus non-mover",
            caption=(
                "Producer: meta/M04_syntagmatic/scripts/attn_figures.py from\n"
                "results/attn_delta_smollm2_e1_{cross,own}.json (producer attn_delta.py).\n"
                "plot-debt M04 candidate 8.\n"
                "Asserted before drawing: the pair, prompt, 32 layers, 15 heads, 32-token window and\n"
                "n=24 in BOTH files; 480 paired heads; and all twelve of section 3's cells, each median\n"
                "within 5.1e-05 and each p within 0.05 in log10. The p values are asserted because the\n"
                "panel's argument is that `cross` has small p and no magnitude.\n"
                "Three further asserts guard what the drawing says rather than what it plots: that\n"
                "`cross` stays inside 0.002, that `own` exceeds it at least fivefold, and that `cross`\n"
                "changes sign at least five times across the window.\n"
                "The file is NAMED, not globbed: attn_delta_smollm2_e1_cross_w200.json is a 200-token\n"
                "window sibling, and a glob would make the window a lottery."),
        )
        + theme_minimal()
        + theme(figure_size=(12.6, 7.6),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                plot_caption=element_text(size=6.3, color="#666666", ha="left",
                                          lineheight=1.45))
    )
    out = os.path.join(FIGURES, "attn_cross_own.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    cell: {cells['own']['prompt']}, {cells['own']['pair'].split('>')[0]}")
    print(f"    cross |median| max {cr.med.abs().max():.5f}, {flips} sign changes")
    print(f"    own    median min {ow.med.min():+.5f}, max {ow.med.max():+.5f}")
    print(f"    12 of 12 booked cells reproduced (median and Wilcoxon p)")
    print(f"    three-way ordering NOT drawn: retracted in section 3")
    return out


REGISTRY = {"cross_own": cross_own}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:16s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
