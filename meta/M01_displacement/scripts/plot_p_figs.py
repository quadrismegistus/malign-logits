#!/usr/bin/env python
"""Figures for findings P (P_unnamed_axis.md). P had none of record.

    uv run python meta/M01_displacement/scripts/plot_p_figs.py
    uv run python meta/M01_displacement/scripts/plot_p_figs.py headroom
    uv run python meta/M01_displacement/scripts/plot_p_figs.py --list

Plot-debt item 15(1). plotnine at 300 dpi, output to ../figures/, slice in
the subtitle, booked-number asserts before drawing. Case 1 by shape: reads
committed artifacts and writes only pixels.

THE LADDER IS MADE ENTIRELY OF NONDETERMINISTIC ROWS AND THE FIGURE SAYS SO
----------------------------------------------------------------------------
All three word-level rungs are `trees` rows, and P states the consequence in
bold: *EVERY `trees` ROW IN THIS DOCUMENT CARRIES A SPREAD OF ROUGHLY 0.003
AND SHOULD NOT BE COMPARED TO A LINEAR ROW AT THE THIRD DECIMAL.*
`HistGradientBoosting` parallelises through OpenMP and its histogram
construction is thread-order dependent, so the seed does not fix it.

On a headroom of 0.1207 a spread of 0.003 is **2.5 percentage points**, which
is most of the distance between the two embeddings. So the bars carry it. A
three-bar ladder drawn as points would invite exactly the GloVe-beats-bge
reading the producer's own docstring forbids, and P's claim does not need it:
the finding is *norms against embeddings*, not GloVe against bge.

WHY THE COMMITTED ARTIFACTS DO NOT EQUAL THE BOOKED VALUES, WHICH IS NOT A DEFECT
----------------------------------------------------------------------------------
    rung        booked      committed draw     both within the declared spread
    norms      +0.0083         +0.0074         yes
    bge        +0.0208         +0.0208         exact
    GloVe      +0.0229         +0.0256         yes; committed is draw 1 of 5

The bars plot the BOOKED values, so the figure and the finding cannot
disagree, and the asserts check the committed draws against them **within the
declared spread** rather than to the digit. Asserting equality here would be
asserting that a nondeterministic producer is deterministic.

GloVe's booked value is a mean over five runs recorded only in prose; those
five plus one measured on 2026-08-14 are persisted in
`results/k/predict_embed_en_glove_draws.json` as a transcription. **Do not
quote 21%** is P's fence, and the observed range is drawn so the fence is
geometry rather than a caption note.

SITE-DELTA IS A DIFFERENT DENOMINATOR AND GETS ITS OWN PANEL
-------------------------------------------------------------
Plot-debt item 15 asks for one ladder of *norms 7%, bge 17%, GloVe 18-21%,
site-delta 68-82%*. The first three are shares of **+0.1207** over 2,760
words; site-delta's are shares of **+0.1638** over 4,064 words, computed on
its own rows. P corrected precisely this: *the oracle here is computed on
these rows, not read from section 2, after a first version compared a
4,064-word population against a 2,760-word ceiling.* Pooling them would
reproduce a corrected error as an image, so the panels are separate and each
names its own ceiling.

THE 87% IS NOT A FOURTH BAR
----------------------------
ICC(1) = 0.131, so 87% of the fall/rise variance is WITHIN a word across its
sites. That is a share of VARIANCE, not a share of headroom, and putting it on
a percent-of-headroom axis would be a third incommensurable unit on one scale.
It is why the ceiling sits where it does, so it is stated as the reason the
axis ends where it ends.
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
K = os.path.join(RESULTS, "k")
FIGURES = os.path.join(M01, "figures")

#: P section 3's table. The bars plot these.
BOOKED = {"norms": 0.0083, "bge": 0.0208, "glove": 0.0229}
#: P section 2, the denominator for those three.
BOOKED_HEADROOM = 0.1207
#: P's declared spread on every `trees` row.
TREES_SPREAD = 0.003
#: P section 5's own ceiling, on its own population.
BOOKED_DELTA = {"headroom": 0.1638, "word": 0.1111, "prompt": 0.1345,
                "n_words": 4064, "oracle": 0.6517, "ceiling": 0.6821}
BOOKED_ICC = 0.131

GREY, ACCENT, DELTA_C = "#9a9a9a", "#1a7a6a", "#b8860b"


def _increment(path, arm, model="trees"):
    """Increment over the arm's OWN shuffle, pooled AUC.

    The two artifact families nest differently -- the norms file under
    `results` keyed by arm name, the embedding files under `components`
    keyed by PCA k -- so the container is resolved rather than assumed.
    """
    v = json.load(open(os.path.join(K, path)))
    box = v.get("results") or v.get("components")
    assert box is not None, f"{path} has neither `results` nor `components`"
    assert arm in box, f"{path}: no arm {arm!r} (have {sorted(box)[:6]})"
    a = box[arm]
    return a["real"][model][0] - a["shuffled"][model][0]


def headroom():
    """P 15(1): what share of the word-level ceiling each instrument recovers."""
    from plotnine import (aes, element_blank, element_text, facet_wrap,
                          geom_col, geom_errorbar, geom_text, ggplot, labs,
                          scale_fill_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)

    ceil = json.load(open(os.path.join(K, "ceiling_en_verbs.json")))
    H = ceil["oracle_auc"] - ceil["p_base_auc"]
    assert round(H, 4) == BOOKED_HEADROOM, \
        f"headroom drifted: {round(H, 4)} vs booked {BOOKED_HEADROOM}"
    assert ceil["n_words"] == 2760 and round(ceil["icc"], 3) == BOOKED_ICC, \
        f"ceiling population drifted: {ceil['n_words']} words, ICC {ceil['icc']}"

    #: WITHIN THE DECLARED SPREAD, NOT TO THE DIGIT. Every rung is a `trees`
    #: row and P states they carry ~0.003 of thread-nondeterminism, so an
    #: equality assert here would assert that a nondeterministic producer is
    #: deterministic and would fail on any honest re-run.
    got = {"norms": _increment("predict_verbs_en.json", "CODER ONLY, all verbs"),
           #: P quotes bge's BEST sweep row, which sits at k=100, while the
           #: GloVe row it prints beside it is k=50. Taking the max here
           #: reproduces what the finding books rather than a k the finding
           #: never claimed for bge.
           "bge": max(_increment("predict_embed_en_bge.json", k, m)
                      for k in json.load(open(os.path.join(
                          K, "predict_embed_en_bge.json")))["components"]
                      for m in ("logistic", "trees")),
           "glove": _increment("predict_embed_en_glove.json", "50")}
    for name, booked in BOOKED.items():
        assert abs(got[name] - booked) <= TREES_SPREAD, \
            (f"{name}: committed draw {got[name]:+.4f} is {abs(got[name] - booked):.4f} "
             f"from the booked {booked:+.4f}, beyond the declared {TREES_SPREAD} "
             "trees spread -- that is drift rather than nondeterminism")

    dr = json.load(open(os.path.join(K, "predict_embed_en_glove_draws.json")))
    lo_g = 100 * dr["combined_six_draws"]["min"] / H
    hi_g = 100 * dr["combined_six_draws"]["max"] / H

    d = json.load(open(os.path.join(K, "delta_predict_en.json")))
    assert d["n_words"] == BOOKED_DELTA["n_words"], \
        f"site-delta population drifted: {d['n_words']}"
    assert round(d["oracle_headroom"], 4) == BOOKED_DELTA["headroom"], \
        f"site-delta ceiling drifted: {d['oracle_headroom']}"
    for k in ("word", "prompt"):
        assert round(d[k]["gain_delta"], 4) == BOOKED_DELTA[k], \
            f"site-delta {k} gain drifted: {d[k]['gain_delta']}"
    assert d["oracle_headroom"] > H, \
        "site-delta's ceiling is no longer distinct from section 2's; the two " \
        "panels exist because the denominators differ"

    pa = ("A. WORD-LEVEL INSTRUMENTS\nshare of the +0.1207 ceiling, 2,760 verbs")
    pb = ("B. THE SITE-DELTA, A DIFFERENT POPULATION\n"
          "share of ITS OWN +0.1638 ceiling, 4,064 words")
    sp = 100 * TREES_SPREAD / H

    rows = [
        {"panel": pa, "x": 0, "lab": "18 rated norms", "pct": 100 * BOOKED["norms"] / H,
         "lo": 100 * BOOKED["norms"] / H - sp, "hi": 100 * BOOKED["norms"] / H + sp,
         "val": f"+{BOOKED['norms']:.4f}", "fill": GREY},
        {"panel": pa, "x": 1, "lab": "bge-m3, 1024d", "pct": 100 * BOOKED["bge"] / H,
         "lo": 100 * BOOKED["bge"] / H - sp, "hi": 100 * BOOKED["bge"] / H + sp,
         "val": f"+{BOOKED['bge']:.4f}", "fill": ACCENT},
        {"panel": pa, "x": 2, "lab": "GloVe, 300d", "pct": 100 * BOOKED["glove"] / H,
         "lo": lo_g, "hi": hi_g, "val": f"+{BOOKED['glove']:.4f}", "fill": ACCENT},
        {"panel": pb, "x": 0, "lab": "site delta\nheld out by WORD",
         "pct": 100 * BOOKED_DELTA["word"] / BOOKED_DELTA["headroom"],
         "lo": None, "hi": None, "val": f"+{BOOKED_DELTA['word']:.4f}", "fill": DELTA_C},
        {"panel": pb, "x": 1, "lab": "site delta\nheld out by PROMPT",
         "pct": 100 * BOOKED_DELTA["prompt"] / BOOKED_DELTA["headroom"],
         "lo": None, "hi": None, "val": f"+{BOOKED_DELTA['prompt']:.4f}", "fill": DELTA_C},
    ]
    df = pd.DataFrame(rows)
    df["panel"] = pd.Categorical(df.panel, categories=[pa, pb], ordered=True)
    err = df[df.lo.notna()]

    note = pd.DataFrame([
        {"panel": pa, "x": 1.0, "y": 55,
         "t": "the two encoders agree about nothing else\n"
              "and land within 0.005 of each other:\n"
              "GloVe near-isotropic (0.037), bge not (0.529)"},
        {"panel": pb, "x": 0.5, "y": 95,
         "t": "CONTAINS WORD IDENTITY, so this is mostly\n"
              "evidence of an excellent WORD feature —\n"
              "and it does NOT beat the oracle\n"
              "(0.6517 against 0.6821), so the ceiling holds"}])
    note["panel"] = pd.Categorical(note.panel, categories=[pa, pb], ordered=True)

    p = (
        ggplot()
        + geom_col(df, aes("x", "pct", fill="fill"), width=0.6)
        + geom_errorbar(err, aes("x", ymin="lo", ymax="hi"), width=0.16,
                        size=0.5, color="#333333")
        + geom_text(df, aes("x", "pct", label="val"), va="bottom", nudge_y=2.6,
                    size=7.0, color="#222222")
        + geom_text(note, aes("x", "y", label="t"), size=6.1, color="#666666",
                    lineheight=1.25)
        + scale_fill_identity()
        #: LABELS AS GEOMETRY, NOT AS AN AXIS SCALE. With `scales="free_x"`
        #: each facet numbers its own bars from 0, and a positional labeller
        #: cannot see which facet it is in -- the first version applied panel
        #: A's instrument names to panel B's site-delta bars, so the right
        #: panel was labelled with the left panel's instruments. A per-row
        #: label drawn from the data cannot make that mistake.
        + geom_text(df, aes("x", -5, label="lab"), size=7.2, color="#333333",
                    lineheight=1.15)
        + scale_x_continuous(breaks=[])
        + scale_y_continuous(limits=(-10, 100), breaks=[0, 25, 50, 75, 100],
                             labels=["0%", "25%", "50%", "75%", "100%"])
        + facet_wrap("~panel", nrow=1, scales="free_x")
        + labs(
            title="No word-level instrument recovers much of the ceiling, and the one that does is not a word-level instrument",
            subtitle=(
                "P's prediction study. A word's own cells predict the other half of its cells, which is the\n"
                "best any function of the word alone can reach: AUC 0.7025 against 0.5818 for base probability,\n"
                "so +0.1207 of headroom over 2,760 English verbs.\n"
                "LEFT: what each instrument beats ITS OWN SHUFFLE by, as a share of that ceiling. The eighteen\n"
                "rated norms buy 7%; two encoders that agree about nothing else both buy about three times as\n"
                "much and land within 0.005 of each other.\n"
                "EVERY BAR HERE IS A `trees` ROW AND P DECLARES THEM THREAD-NONDETERMINISTIC AT ABOUT 0.003,\n"
                "which is 2.5 points of this axis. The error bars are that spread; GloVe's is its OBSERVED\n"
                "range over six draws. So the two encoders are NOT separable and the figure does not separate\n"
                "them: the finding is norms against embeddings, not GloVe against bge.\n"
                "RIGHT: A DIFFERENT POPULATION AND A DIFFERENT CEILING, drawn apart for that reason. Its\n"
                "+0.1638 headroom is computed on its own 4,064 words, after an earlier version of the finding\n"
                "compared this population against the ceiling on the left.\n"
                "THE AXIS ENDS AT 100% OF A LOW CEILING. ICC(1) = 0.131, so 87% of the fall/rise variance is\n"
                "WITHIN a word across the sites it appears at, and is unreachable by any feature constant per\n"
                "word. Movement is a property of the word AT A SITE."),
            x="", y="percent of that panel's own ceiling",
            caption=(
                "Producer: meta/M01_displacement/scripts/plot_p_figs.py from results/k/{ceiling_en_verbs,\n"
                "predict_verbs_en, predict_embed_en_bge, predict_embed_en_glove,\n"
                "predict_embed_en_glove_draws, delta_predict_en}.json.\n"
                "THE BARS PLOT P'S BOOKED VALUES so the figure and the finding cannot disagree. The asserts\n"
                "check each committed draw against its booked value WITHIN the declared 0.003 trees spread,\n"
                "not to the digit: asserting equality would assert that a nondeterministic producer is\n"
                "deterministic. Committed draws are norms +0.0074, bge +0.0208, GloVe +0.0256.\n"
                "GLOVE'S BOOKED VALUE IS A MEAN OVER FIVE RUNS recorded only in prose; those five and one\n"
                "measured 2026-08-14 are transcribed in predict_embed_en_glove_draws.json. P's fence is DO NOT\n"
                "QUOTE 21%, and the committed artifact is that draw, so the bar carries the observed range\n"
                "instead of any single number.\n"
                "Both quoted embedding shares are the best of ten sweep rows chosen after seeing them, which\n"
                "k_predict_embed.py:207 states is an upper bound rather than an estimate; GloVe's best sits at\n"
                "k=50 and bge's at k=100, so they are not compared at the same k."),
        )
        + theme_minimal()
        + theme(figure_size=(13.2, 8.0),
                plot_title=element_text(size=11, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                strip_text=element_text(size=8.2, weight="bold"),
                axis_text_x=element_blank(),
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
                panel_spacing=0.07)
    )
    out = os.path.join(FIGURES, "p_headroom_ladder.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    ceiling +{H:.4f} over {ceil['n_words']} verbs, ICC {ceil['icc']:.3f}")
    for n in ("norms", "bge", "glove"):
        print(f"    {n:<6} booked {BOOKED[n]:+.4f} = {100 * BOOKED[n] / H:4.1f}%  "
              f"committed draw {got[n]:+.4f}  (within {TREES_SPREAD})")
    print(f"    GloVe observed range over six draws: {lo_g:.1f}-{hi_g:.1f}%")
    print(f"    site-delta on its own +{BOOKED_DELTA['headroom']:.4f}: "
          f"{100 * BOOKED_DELTA['word'] / BOOKED_DELTA['headroom']:.0f}% and "
          f"{100 * BOOKED_DELTA['prompt'] / BOOKED_DELTA['headroom']:.0f}%")
    return out


REGISTRY = {"headroom": headroom}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:10s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
