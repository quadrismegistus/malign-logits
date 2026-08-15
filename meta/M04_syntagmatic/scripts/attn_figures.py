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

So this panel does not draw the three-way ordering at all. A figure is where a
retracted result goes to be revived, because the picture outlives the paragraph
that withdrew it.

AND THE RETRACTION IS NOT THE ONLY THING SECTION 3 LOST
--------------------------------------------------------
Reading forward past the retraction, on registrar's prompt at [6204], two more
limits reach this panel and neither is in the queue entry:

- **Section 3b supersedes the instrument.** D_norm, a scale-free log ratio of
  ratios, "is the right quantity and the earlier tables should be read as
  superseded by it wherever they disagree." Section 3's table, which this panel
  draws, is norm-weighted D. It is drawn anyway because the cross/own
  comparison is defined in it and not in D_norm -- so the panel makes no claim
  about which word moved more, only about the two modes.
- **Section 3c refutes the contrast on this axis.** Over 28 cells the
  faller-minus-non-mover median is +0.0174 with 14 of 28 negative, p = 0.171:
  it does not order by alignment status. The cell drawn here gives -0.0393 and
  the SAME PAIR on `sexual_explicit_3` gives +0.1885 at p = 2e-32.

Section 4 is what licenses the panel: "what survives is the `cross`/`own` split
itself... a fact about the design rather than about alignment." The height of
the dark line is not a result; the existence of the gap between the modes is.
Both fences are recomputed by `_sweep()` before drawing rather than quoted.

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


#: Section 3c's 28-cell sweep, which is the fence this panel has to carry.
BOOKED_SWEEP = {"cells": 28, "fn_med": +0.0174, "fn_neg": 14, "fn_p": 0.171,
                "e1": -0.0393, "e3": +0.1885}


def _sweep():
    """Section 3c re-derived: the contrast on this panel's axis does not replicate.

    THE PANEL DRAWS ONE CELL AND SECTION 3c RAN 28. Without this, a reader takes
    the dark line's depth for a result; with it, the depth is one draw from a
    set whose median sits on the other side of zero. The numbers are DERIVED
    here rather than quoted, because a fence quoted from a document is a fence
    that goes stale silently -- which is the whole reason this figure exists.
    """
    import numpy as np
    from scipy.stats import wilcoxon

    cells = json.load(open(os.path.join(RESULTS, "attn_norm_sweep_full.json")))
    assert len(cells) == BOOKED_SWEEP["cells"], \
        f"sweep has {len(cells)} cells, section 3c reports 28"
    v = np.array([c["f_minus_n"] for c in cells])
    med, neg = float(np.median(v)), int((v < 0).sum())
    _, p = wilcoxon(v)
    assert abs(med - BOOKED_SWEEP["fn_med"]) < 5e-5, f"sweep median {med:+.4f}"
    assert neg == BOOKED_SWEEP["fn_neg"], f"{neg} of 28 negative, not 14"
    assert abs(p - BOOKED_SWEEP["fn_p"]) < 5e-4, f"sweep p {p:.4f}"
    #: THE NON-REPLICATION INSIDE ONE PAIR, which is the sharpest form of it:
    #: same models, same instrument, opposite sign, both overwhelming.
    by = {c["prompt"]: c for c in cells if "SmolLM2" in c["pair"]}
    e1, e3 = by["sexual_explicit_1"], by["sexual_explicit_3"]
    assert abs(e1["f_minus_n"] - BOOKED_SWEEP["e1"]) < 5e-5, "explicit_1 moved"
    assert abs(e3["f_minus_n"] - BOOKED_SWEEP["e3"]) < 5e-5, "explicit_3 moved"
    assert e1["f_minus_n"] * e3["f_minus_n"] < 0, \
        ("the two prompts of this pair no longer disagree in SIGN; the panel "
         "says they do and that is section 3c's sharpest evidence")
    return {"med": med, "neg": neg, "p": float(p), "e1": e1["f_minus_n"],
            "e3": e3["f_minus_n"], "p_e3": e3["p_fn"]}


def cross_own():
    """M04 candidate 8: the two modes disagree, and only one of them is a result."""
    from scipy.stats import wilcoxon
    from plotnine import (aes, element_text, geom_hline, geom_line, geom_point,
                          geom_ribbon, geom_text, ggplot, labs,
                          scale_color_identity, scale_fill_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)
    import pandas as pd

    SWEEP = _sweep()
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
            title="An effect that appears only when each arm writes its own continuation, and vanishes when both read one text",
            subtitle=(
                "Attention paid back to a forced word, aligned minus base, faller MINUS non-mover, median\n"
                "over 480 heads (32 layers x 15) at each position after the word. Dark = OWN, where each\n"
                "arm writes its own continuation; grey = CROSS, where both models read one text. Bands are\n"
                "95% bootstrap intervals FOR THE MEDIAN; dots mark the six positions section 3 tabulates.\n"
                "WHAT SURVIVES IS THE SPLIT BETWEEN THE MODES, AND ONLY THAT (section 4). The only\n"
                "difference between them is whether each arm writes its own continuation, so an effect\n"
                "present in `own` and absent in `cross` lives in what gets WRITTEN, not in how a given\n"
                "text is attended to. That is a fact about the design rather than about alignment.\n"
                "WHAT DOES NOT SURVIVE IS THE HEIGHT OF THE DARK LINE. This same contrast was run over\n"
                f"28 cells with the better instrument: {SWEEP['neg']} of 28 negative, median {SWEEP['med']:+.4f} on the WRONG\n"
                f"side, p = {SWEEP['p']:.3f}. It does not order by alignment status. The cell drawn here gives\n"
                f"{SWEEP['e1']:+.4f} and the SAME PAIR on `sexual_explicit_3` gives {SWEEP['e3']:+.4f} at p = {SWEEP['p_e3']:.0e} --\n"
                "opposite sign, both overwhelming. One cell is one cell.\n"
                "AND THIS PANEL'S INSTRUMENT IS THE SUPERSEDED ONE. Section 3b establishes D_norm, a\n"
                "scale-free log ratio of ratios, as the right quantity and says the earlier tables\n"
                "should be read as superseded by it wherever they disagree. Norm-weighted D is drawn\n"
                "here because the cross/own comparison is defined in it and not in D_norm, so this\n"
                "panel makes NO claim about which word moved more. Only about the two modes.\n"
                "SMALL p IS NOT EVIDENCE FOR EITHER MODE. `cross` returns p = 0.0006 at j=7 on a median\n"
                "of -0.0004, sign alternating position to position; `own` returns p = 1.1e-19. Section\n"
                "3c's verdict covers both: per-cell p values over 250 to 480 massively correlated heads\n"
                "run to p = 0 in both directions and are not evidence about anything. The CELL is the\n"
                "unit, and at the cell level there are 28 of them. Read the axis, not the p.\n"
                "THE BANDS ARE THE UNCERTAINTY OF THE MEDIAN AND NOT THE SPREAD OF THE HEADS, whose\n"
                f"interquartile range is about {iqr['own']:.3f} in `own` and {iqr['cross']:.3f} in `cross` --\n"
                "tens of times the medians drawn here. That spread cannot share this axis without\n"
                "flattening both lines onto zero, so it is stated rather than drawn.\n"
                "THE THREE-WAY ORDERING IS RETRACTED AND IS NOT DRAWN. Section 3 reported FALLER <\n"
                "NONMOVER < RISER, monotone in alignment status, and its own retraction withdraws it:\n"
                "the three words' base probabilities are 0.062, 0.089 and 0.201, so on this cell that\n"
                "ordering is indistinguishable from ordering by how probable the word was. A second cell\n"
                "whose faller is 33x more probable than its riser refuses BOTH accounts.\n"
                "SLICE. One prompt (`sexual_explicit_1`), one pair (SmolLM2-360M to its Instruct),\n"
                "three words, 24 samples, a 32-token window, 480 heads."),
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
                "window sibling, and a glob would make the window a lottery.\n"
                "THE REFUTATION IS DERIVED, NOT QUOTED. Section 3c's 28-cell sweep is recomputed from\n"
                "results/attn_norm_sweep_full.json before drawing: 28 cells, 14 negative, median and\n"
                "Wilcoxon p, both prompts of the SmolLM2 pair, and that their signs still disagree. A\n"
                "fence copied out of a document goes stale silently, which is what this figure is for."),
        )
        + theme_minimal()
        + theme(figure_size=(12.6, 9.2),
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




#: Section 3c's three contrasts. Predicted: FALLER below, NONMOVER and RISER
#: together -- so the first two were predicted NEGATIVE and the third NULL.
BOOKED_3C = {
    "f_minus_n": {"med": +0.0174, "neg": 14, "p": 0.171,
                  "label": "FALLER  -  NONMOVER", "pred": "predicted BELOW zero"},
    "f_minus_r": {"med": -0.0278, "neg": 16, "p": 0.194,
                  "label": "FALLER  -  RISER", "pred": "predicted BELOW zero"},
    "n_minus_r": {"med": -0.0478, "neg": 18, "p": 0.0247,
                  "label": "NONMOVER  -  RISER", "pred": "predicted NEAR zero"},
}
ROWS = ["f_minus_n", "f_minus_r", "n_minus_r"]
STRIP_X_LIM = (-1.04, 0.58)
DOT_C, MED_C, HI_C = "#9a9a9a", "#2b2b2b", "#b03030"


def sweep_strips():
    """M04 candidate 7: a null that prose cannot make credible, drawn."""
    from scipy.stats import wilcoxon
    from plotnine import (aes, element_blank, element_text, geom_point,
                          geom_segment, geom_text, geom_vline, ggplot, labs,
                          scale_color_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)
    import pandas as pd

    #: THE ENTRY NAMES BOTH FILES AND THE CHOICE IS PROVABLY IMMATERIAL. The
    #: small one carries the summary fields; `_full` adds per-head arrays and
    #: is 230x larger. Rather than pick and hope, the summaries are asserted
    #: equal and the small one is read.
    small = json.load(open(os.path.join(RESULTS, "attn_norm_sweep.json")))
    full = json.load(open(os.path.join(RESULTS, "attn_norm_sweep_full.json")))
    assert len(small) == len(full) == BOOKED_SWEEP["cells"], "cell count"
    for a, b in zip(small, full):
        assert (a["pair"], a["prompt"]) == (b["pair"], b["prompt"]), "cell order"
        for k in ROWS:
            assert a[k] == b[k], \
                f"attn_norm_sweep.json and _full disagree on {k}; the caption "
    cells = small

    rows, pts = [], []
    for i, key in enumerate(ROWS):
        b = BOOKED_3C[key]
        v = np.array([c[key] for c in cells])
        med, neg = float(np.median(v)), int((v < 0).sum())
        _, p = wilcoxon(v)
        assert abs(med - b["med"]) < 5e-5, f"{key} median {med:+.4f} vs {b['med']}"
        assert neg == b["neg"], f"{key}: {neg} of 28 negative, not {b['neg']}"
        assert abs(p - b["p"]) < 5e-4, f"{key}: p {p:.4f} vs {b['p']}"
        y = len(ROWS) - 1 - i
        rows.append({"y": y, "label": b["label"], "pred": b["pred"],
                     "med": med, "neg": neg, "p": p})
        for c in cells:
            hi = "SmolLM2" in c["pair"] and c["prompt"] in (
                "sexual_explicit_1", "sexual_explicit_3")
            pts.append({"y": y, "v": c[key], "col": HI_C if hi else DOT_C,
                        "sz": 2.6 if hi else 1.9, "key": key,
                        "prompt": c["prompt"], "pair": c["pair"]})

    #: THE ONLY NOMINALLY SIGNIFICANT CONTRAST IS THE ONE PREDICTED TO BE NULL,
    #: which is the section's argument in one sentence and is asserted so the
    #: panel cannot outlive it.
    by = {r["label"]: r for r in rows}
    sig = [r for r in rows if r["p"] < 0.05]
    assert len(sig) == 1 and sig[0]["label"] == BOOKED_3C["n_minus_r"]["label"], \
        ("the set of contrasts under p<0.05 has changed; the panel says the "
         "only one is NONMOVER-RISER, which the prediction put near zero")
    assert by[BOOKED_3C["f_minus_n"]["label"]]["med"] > 0, \
        ("FALLER-NONMOVER no longer has its median on the wrong side of zero; "
         "the panel's headline says it does")

    d_pts, d_rows = pd.DataFrame(pts), pd.DataFrame(rows)
    #: NOTHING MAY BE CLIPPED: an off-panel cell would silently subtract one
    #: dot from a panel whose whole evidence is how the 28 dots are spread.
    assert d_pts.v.min() > STRIP_X_LIM[0] and d_pts.v.max() < STRIP_X_LIM[1], \
        (f"cells span {d_pts.v.min():+.4f}..{d_pts.v.max():+.4f} against an "
         f"axis of {STRIP_X_LIM}")
    #: precomputed rather than expressed inside aes(): plotnine evaluates aes
    #: strings, so an expression there works until it silently does not, and a
    #: mapped `size` would build a scale and rescale the very values it shows
    d_rows["y0"], d_rows["y1"] = d_rows.y - 0.24, d_rows.y + 0.24
    d_rows["lab_y"], d_rows["pred_y"] = d_rows.y + 0.20, d_rows.y - 0.04
    d_rows["stat_y"] = d_rows.y - 0.26
    d_rows["stat"] = [f"{r.neg} of 28 below zero     Wilcoxon p {r.p:.3f}"
                      for r in d_rows.itertuples()]
    d_rows["lx"] = -1.02
    plain = d_pts[d_pts.col == DOT_C]
    hilit = d_pts[d_pts.col == HI_C]
    #: the two prompts of ONE pair, opposite in sign, on the primary contrast
    pair_lab = d_pts[(d_pts.key == "f_minus_n") & (d_pts.col == HI_C)]
    e = {r.prompt: r.v for r in pair_lab.itertuples()}
    assert e["sexual_explicit_1"] * e["sexual_explicit_3"] < 0, "signs agree now"
    #: EACH LABEL ANCHORED TO ITS OWN DOT. The first version placed the left
    #: one at a fixed offset of -0.30, which put it over unrelated cells and
    #: made it read as labelling them.
    ann = pd.DataFrame([
        {"y": 2.34, "v": e["sexual_explicit_3"] + 0.012, "ha": "left",
         "t": f"same pair, other prompt: {e['sexual_explicit_3']:+.4f}", "c": HI_C},
        {"y": 2.34, "v": e["sexual_explicit_1"] - 0.012, "ha": "right",
         "t": f"the cell every earlier table quotes: {e['sexual_explicit_1']:+.4f}",
         "c": HI_C}])

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.6)
        + geom_point(plain, aes("v", "y"), color=DOT_C, size=1.9, alpha=0.75)
        + geom_point(hilit, aes("v", "y"), color=HI_C, size=2.8)
        + geom_segment(d_rows, aes("med", "y0", xend="med", yend="y1"),
                       color=MED_C, size=1.5)
        + geom_text(d_rows, aes("lx", "lab_y", label="label"), size=7.6,
                    ha="left", color="#222222")
        + geom_text(d_rows, aes("lx", "pred_y", label="pred"), size=6.6,
                    ha="left", color="#777777")
        + geom_text(d_rows, aes("lx", "stat_y", label="stat"), size=6.6,
                    ha="left", color="#444444")
        + geom_text(ann, aes("v", "y", label="t", color="c", ha="ha"), size=6.4)
        + scale_color_identity()
        #: LIMITS SET FROM THE DATA, NOT FROM A ROUND NUMBER. The first version
        #: ran to +/-1.0 while the cells span -0.47 to +0.51, so half the panel
        #: was empty and the evidence sat compressed in the middle third.
        + scale_x_continuous(limits=STRIP_X_LIM,
                             breaks=[-0.4, -0.2, 0, 0.2, 0.4])
        + scale_y_continuous(breaks=[], limits=(-0.55, len(ROWS) - 0.35))
        + labs(
            title="The one contrast that separates is the one the prediction said would not",
            subtitle=(
                "Each dot is one of 28 cells (6 model pairs x 5 prompts, arms auto-selected by a rule\n"
                "committed BEFORE the sweep returned, so the prediction has a timestamp preceding its own\n"
                "test). x is the paired difference in D_norm between two words in that cell; the heavy\n"
                "tick is the median over cells.\n"
                "THE PREDICTION WAS: FALLER BELOW, NONMOVER AND RISER TOGETHER. It fails on both halves.\n"
                "FALLER-NONMOVER is a coin flip with its median on the WRONG side of zero. The only\n"
                "contrast reaching p < 0.05 is NONMOVER-RISER, which the prediction put near zero.\n"
                "NO PER-CELL p VALUE IS DRAWN, AND THAT IS DELIBERATE. Section 3c's own verdict is that\n"
                "these run to p = 0 and p = 2e-32 in BOTH directions because each is computed across 250\n"
                "to 480 massively correlated heads, and that they are not evidence about anything. The\n"
                "CELL is the unit; encoding cell-level significance here would smuggle back the quantity\n"
                "the section exists to disqualify. The spread of the dots is the evidence.\n"
                "THE TWO RED DOTS ARE ONE PAIR ON TWO PROMPTS, and they are the sharpest single fact in\n"
                "this figure. Same models, same instrument, opposite sign, both overwhelming on their own\n"
                "cell-level tests. The left one is the cell every table in sections 2, 3 and 3b quotes.\n"
                "WHAT THIS DOES NOT SAY. Not that alignment does nothing to attention-back: section 3e\n"
                "finds a pair-level shift that is real and large, which this instrument divides out by\n"
                "construction. What is refuted is that the effect ORDERS BY ALIGNMENT STATUS."),
            x="paired difference in D_norm, per cell   (log ratio of ratios, scale-free)",
            y="",
            caption=(
                "Producer: meta/M04_syntagmatic/scripts/attn_figures.py from\n"
                "results/attn_norm_sweep.json (producer attn_norm_sweep.py). plot-debt M04 candidate 7.\n"
                "The entry names attn_norm_sweep{,_full}.json and the choice is provably immaterial: the\n"
                "two agree cell for cell on all three contrasts, asserted here before drawing. `_full`\n"
                "adds per-head arrays and is 230x larger, so the small file is read.\n"
                "Asserted before drawing: 28 cells in both files and in the same order; each contrast's\n"
                "median, count below zero and Wilcoxon p against section 3c; that exactly one contrast\n"
                "falls under p<0.05 and that it is NONMOVER-RISER; that FALLER-NONMOVER still has its\n"
                "median on the wrong side of zero; and that the two SmolLM2 prompts still disagree in\n"
                "sign. Each of those is a sentence on the panel, and none of them is quoted.\n"
                "D_norm is section 3b's instrument and supersedes the raw and norm-weighted tables of\n"
                "sections 2 and 3 wherever they disagree."),
        )
        + theme_minimal()
        + theme(figure_size=(13.0, 7.4),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                plot_caption=element_text(size=6.3, color="#666666", ha="left",
                                          lineheight=1.45),
                legend_position="none",
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "attn_sweep_refutation.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for r in rows:
        print(f"    {r['label']:<22} median {r['med']:+.4f}  {r['neg']}/28 below 0  p {r['p']:.4f}")
    print(f"    one pair, two prompts: {e['sexual_explicit_1']:+.4f} and "
          f"{e['sexual_explicit_3']:+.4f}")
    print(f"    no per-cell p encoded, per section 3c")
    return out




#: Section 3e. Keyed on the BASE side of the pair string, which is how the
#: document names them; the aligned side reads SmolLM2-360M-Instruct and would
#: match nothing here.
BOOKED_3E = {
    "stabilityai/stablelm-2-1_6b": (+0.2179, 0.0187),
    "tiiuae/Falcon3-1B-Base": (+0.1883, 0.1017),
    "allenai/OLMo-2-0425-1B": (-0.2157, 0.1343),
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": (+0.0594, 0.1474),
    "HuggingFaceTB/SmolLM2-360M": (+0.0386, 0.0666),
    "Qwen/Qwen2.5-0.5B": (-0.0197, 0.1439),
}
#: pooled over all 28 cells, and the decomposition that explains why
BOOKED_POOLED = {"med": +0.0531, "pos": 17, "p": 0.227}
BOOKED_DECOMP = {"total": 0.1015, "baseline": 0.1633, "residual": 0.0951}
#: THE TWO 0.0019s. Kruskal-Wallis across the six pairs HOLDS. A modal-sign
#: count over the same 28 values gives the SAME p to four decimals against a
#: naive 0.5 null and does NOT hold, because the modal sign is defined by the
#: majority: under random signs a 5-prompt pair already agrees 3.44 times.
BOOKED_KW_P = 0.0019
BOOKED_SIGN = {"agree": 22, "n": 28, "naive_p": 0.0019, "correct_exp": 19.24,
               "correct_p": 0.077}
PAIR_C, HI_POS, HI_NEG = "#9a9a9a", "#1a7a6a", "#b03030"
X_LIM = (-0.71, 0.40)
#: KEY ON THE ID, LABEL WITH THE DOCUMENT'S NAME. Section 3e writes
#: "TinyLlama-1.1B"; the id carries a further 26 characters of checkpoint
#: detail that would run the row label into the data. Both are kept so the
#: assert cannot pass on a display string.
DISPLAY = {"TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": "TinyLlama-1.1B"}


def baseline_strips():
    """M04 candidate 6: the pooled null is cancellation, not absence."""
    from scipy.stats import binomtest, kruskal, wilcoxon
    from plotnine import (aes, element_blank, element_text, geom_point,
                          geom_segment, geom_text, geom_vline, ggplot, labs,
                          scale_color_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)
    import pandas as pd

    cells = json.load(open(os.path.join(RESULTS, "attn_norm_sweep_full.json")))
    assert len(cells) == BOOKED_SWEEP["cells"], f"{len(cells)} cells, not 28"

    #: the baseline shift: log of the ratio of each arm's OWN undisturbed
    #: attention-back, median over that cell's heads
    shift, by_pair = [], {}
    for x in cells:
        ub, ua = np.array(x["U"]["base"]), np.array(x["U"]["aligned"])
        v = float(np.median(np.log(ua / ub)))
        shift.append(v)
        by_pair.setdefault(x["pair"].split(">")[0], []).append(v)
    b = np.array(shift)

    assert abs(np.median(b) - BOOKED_POOLED["med"]) < 5e-5, "pooled median"
    assert int((b > 0).sum()) == BOOKED_POOLED["pos"], "pooled positive count"
    _, p_pool = wilcoxon(b)
    assert abs(p_pool - BOOKED_POOLED["p"]) < 5e-4, f"pooled p {p_pool:.4f}"

    assert set(by_pair) == set(BOOKED_3E), \
        f"pairs drifted: {sorted(set(by_pair) ^ set(BOOKED_3E))}"
    rows = []
    for pair, vals in by_pair.items():
        v = np.array(vals)
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        bm, bi = BOOKED_3E[pair]
        assert abs(med - bm) < 5e-5, f"{pair}: median {med:+.4f} vs {bm}"
        assert abs(iqr - bi) < 5e-5, f"{pair}: IQR {iqr:.4f} vs {bi}"
        rows.append({"pair": DISPLAY.get(pair, pair.split("/")[-1]),
                     "med": med, "iqr": iqr,
                     "n": len(v), "vals": v})

    _, p_kw = kruskal(*by_pair.values())
    assert abs(p_kw - BOOKED_KW_P) < 5e-5, f"Kruskal-Wallis p {p_kw:.4f}"

    #: THE COLLISION, PINNED. Two tests over the same 28 numbers print the same
    #: p to four decimals and only one is admissible. If they ever separate,
    #: the panel's three lines distinguishing them become noise and should go.
    agree = sum(int(max((np.array(v) > 0).sum(), (np.array(v) < 0).sum()))
                for v in by_pair.values())
    assert agree == BOOKED_SIGN["agree"], f"modal-sign agreement {agree}, not 22"
    p_naive = binomtest(agree, BOOKED_SIGN["n"], 0.5,
                        alternative="greater").pvalue
    assert abs(p_naive - BOOKED_SIGN["naive_p"]) < 5e-5, f"naive p {p_naive:.4f}"
    assert round(p_naive, 4) == round(p_kw, 4), \
        ("the Kruskal-Wallis p and the discredited sign-test p no longer agree "
         "to four decimals; the panel spends three lines separating them")
    #: and the correct null for a majority-defined statistic
    exp = sum(np.mean([max(k, len(v) - k) for k in
                       np.random.default_rng(0).binomial(len(v), 0.5, 40000)])
              for v in by_pair.values())
    assert abs(exp - BOOKED_SIGN["correct_exp"]) < 0.05, f"null expectation {exp:.2f}"

    #: the decomposition: the BASELINE is larger than the TOTAL it sits inside
    tot, res = [], []
    for x in cells:
        ub, ua = np.array(x["U"]["base"]), np.array(x["U"]["aligned"])
        for w in ("FALLER", "NONMOVER", "RISER"):
            lb = np.array(x["levels"][w + "_base"])
            la = np.array(x["levels"][w + "_aligned"])
            tot.append(float(np.median(np.log(la / lb))))
            res.append(abs(x["d_norm"][w]))
    d_tot, d_bas, d_res = (float(np.median(np.abs(tot))),
                           float(np.median(np.abs(b))), float(np.median(res)))
    for name, got, bk in (("total", d_tot, BOOKED_DECOMP["total"]),
                          ("baseline", d_bas, BOOKED_DECOMP["baseline"]),
                          ("residual", d_res, BOOKED_DECOMP["residual"])):
        assert abs(got - bk) < 5e-5, f"|{name}| {got:.4f} vs section 3e's {bk}"
    assert d_bas > d_tot, \
        "the baseline is no longer larger than the total; that is the panel's point"

    #: NOTHING MAY BE CLIPPED. A point outside the limits vanishes silently,
    #: and a missing dot on a panel whose argument is the SPREAD of the dots
    #: would subtract evidence without raising anything.
    assert float(b.min()) > X_LIM[0] and float(b.max()) < X_LIM[1], \
        (f"data spans {b.min():+.4f}..{b.max():+.4f} and the axis is {X_LIM}; "
         "a cell would be drawn off-panel")

    d_rows = pd.DataFrame(rows).sort_values("med").reset_index(drop=True)
    d_rows["y"] = range(len(d_rows))
    d_rows["col"] = [HI_NEG if m < -0.15 else HI_POS if m > 0.15 else "#2b2b2b"
                     for m in d_rows.med]
    d_rows["y0"], d_rows["y1"] = d_rows.y - 0.22, d_rows.y + 0.22
    d_rows["lx"] = -0.70
    d_rows["lab_y"], d_rows["stat_y"] = d_rows.y + 0.17, d_rows.y - 0.13
    d_rows["stat"] = [f"n={r.n}   median {r.med:+.4f}   IQR {r.iqr:.4f}"
                      for r in d_rows.itertuples()]
    pts = pd.DataFrame([{"y": r.y, "v": v, "col": r.col}
                        for r in d_rows.itertuples() for v in r.vals])

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.6)
        + geom_vline(xintercept=float(np.median(b)), linetype="dashed",
                     color="#777777", size=0.5)
        + geom_point(pts, aes("v", "y", color="col"), size=2.2, alpha=0.8)
        + geom_segment(d_rows, aes("med", "y0", xend="med", yend="y1"),
                       color="#2b2b2b", size=1.6)
        + geom_text(d_rows, aes("lx", "lab_y", label="pair"), size=7.4,
                    ha="left", color="#222222")
        + geom_text(d_rows, aes("lx", "stat_y", label="stat"), size=6.4,
                    ha="left", color="#666666")
        + geom_text(pd.DataFrame([{"x": float(np.median(b)) + 0.012, "y": 5.62,
                                   "t": f"pooled median {np.median(b):+.4f}   "
                                        f"Wilcoxon p {p_pool:.3f}   NULL"}]),
                    aes("x", "y", label="t"), size=6.6, ha="left",
                    color="#777777")
        + scale_color_identity()
        #: THE LEFT GUTTER IS SIZED AGAINST THE LEFTMOST DATUM. I first wrote
        #: -0.243 here, read off an earlier render rather than measured: the
        #: true minimum is OLMo on `sexual_explicit_1` at -0.4976, twice as far
        #: out. A number taken from a picture is a guess with a decimal point.
        + scale_x_continuous(limits=X_LIM,
                             breaks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        + scale_y_continuous(breaks=[], limits=(-0.6, len(d_rows) - 0.15))
        + labs(
            title="The pooled test is null because the pairs disagree in sign, not because nothing moves",
            subtitle=(
                "Change in each model's attention-back to its OWN undisturbed continuation, aligned minus\n"
                "base, as a log ratio. One dot per prompt, grouped by model pair, heavy tick at the pair\n"
                "median. This is the quantity D_norm divides out.\n"
                "POOLED ACROSS ALL 28 CELLS THE SHIFT IS NULL: median +0.0531, 17 of 28 positive,\n"
                "Wilcoxon p 0.227. PER PAIR IT IS NOT: Kruskal-Wallis across the six pairs, p 0.0019.\n"
                "Which pair you are in predicts the shift. The dashed line is the pooled median and it\n"
                "sits between two pairs that move by a fifth in opposite directions.\n"
                "THE NULL IS CANCELLATION, NOT ABSENCE. stablelm at +0.2179 and OLMo at -0.2157 very\n"
                "nearly annihilate. Three pairs shift by around a fifth and three barely move.\n"
                "THE INSTRUMENT HID IT FIRST AND THE POOLED TEST HID IT SECOND. D_norm divides each arm\n"
                "by this baseline, so any effect moving forced and self-chosen words TOGETHER is zero by\n"
                "construction -- and that is the most likely shape for a general effect. The median\n"
                "|baseline| is 0.1633 against a median |total| of 0.1015: the part divided out is LARGER\n"
                "than the total it sits inside.\n"
                "TWO TESTS ON THIS PANEL'S 28 NUMBERS PRINT p = 0.0019 AND ONLY ONE IS ADMISSIBLE. The\n"
                "Kruskal-Wallis holds. A modal-sign count -- 22 of 28 prompts agreeing with their pair's\n"
                "majority sign -- gives the identical p against a naive one-sided 0.5 null and does NOT,\n"
                "because the modal sign IS the majority: under random signs a 5-prompt pair already\n"
                "agrees 3.44 times, the correct null expects 19.24 of 28, and p is 0.077. Section 3e\n"
                "reports it only because it was run first and got it wrong.\n"
                "LIMITS. 6 pairs and 5 sexual prompts, but 28 cells rather than 30: two pairs contribute\n"
                "four. U is measured on undisturbed generations from these prompts only. Nothing here\n"
                "says the shift is general rather than a property of this domain, and nothing predicts\n"
                "its direction."),
            x="log ratio of aligned to base attention-back on each model's own undisturbed continuation",
            y="",
            caption=(
                "Producer: meta/M04_syntagmatic/scripts/attn_figures.py from\n"
                "results/attn_norm_sweep_full.json (producers attn_decompose.py, attn_norm_sweep.py).\n"
                "plot-debt M04 candidate 6. The `_full` file is required here and not interchangeable\n"
                "with attn_norm_sweep.json: only it stores U and the per-head levels, which is why\n"
                "section 3e needed a re-run.\n"
                "Asserted before drawing: 28 cells; the pooled median, positive count and Wilcoxon p;\n"
                "all six pair medians AND their IQRs; the Kruskal-Wallis p; the decomposition medians\n"
                "for total, baseline and residual, and that the baseline still exceeds the total.\n"
                "Two asserts guard the coincidence rather than a value: that the modal-sign count is\n"
                "still 22 of 28 and that its naive p still equals the Kruskal-Wallis p to four decimals.\n"
                "If those ever separate, the panel's five lines distinguishing them are noise and should\n"
                "be cut. The corrected null expectation of 19.24 is recomputed by simulation.\n"
                "Pairs are keyed on the BASE side of the pair string, which is how section 3e names\n"
                "them; the aligned side reads SmolLM2-360M-Instruct and would match nothing."),
        )
        + theme_minimal()
        + theme(figure_size=(13.0, 8.4),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                plot_caption=element_text(size=6.3, color="#666666", ha="left",
                                          lineheight=1.45),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "attn_baseline_by_pair.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    pooled median {np.median(b):+.4f}, {int((b>0).sum())}/28 positive, "
          f"Wilcoxon p {p_pool:.3f} (null)")
    print(f"    Kruskal-Wallis across 6 pairs p {p_kw:.4f}")
    for r in d_rows.itertuples():
        print(f"    {r.pair:<46} {r.n} prompts  median {r.med:+.4f}  IQR {r.iqr:.4f}")
    print(f"    |baseline| {d_bas:.4f} > |total| {d_tot:.4f}, |residual| {d_res:.4f}")
    print(f"    sign-test trap pinned: {agree}/28, naive p {p_naive:.4f} == KW "
          f"{p_kw:.4f}; correct null {exp:.2f} -> p {BOOKED_SIGN['correct_p']}")
    return out


REGISTRY = {"cross_own": cross_own, "sweep_strips": sweep_strips,
            "baseline_strips": baseline_strips}


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
