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
import collections
import json
import os
import statistics as st
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


#: P section 7b's table, mean z over the four instruments. Verified to be a
#: MEAN rather than any single instrument: motion's four average to +2.8275
#: against a booked +2.83, and every row below agrees within 0.08.
BOOKED_7B = {
    "wordnet:contact": 8.77,
    "usas:matter_objects_and_handling": 4.69,
    "wordnet:consumption": 3.73,
    "wordnet:motion": 2.83,
    "usas:body_health_and_consumption": 2.70,
    "wordnet:communication": -5.02,
    "usas:inquiry_discovery_and_education": -4.90,
    "wordnet:cognition": -4.40,
    "usas:cognition_mental": -4.37,
    "wordnet:perception": -3.58,
    "usas:evaluation_modality": -3.44,
}
SHORT = {
    "wordnet:contact": "contact",
    "usas:matter_objects_and_handling": "matter, objects, handling",
    "wordnet:consumption": "consumption",
    "wordnet:motion": "motion",
    "usas:body_health_and_consumption": "body, health",
    "wordnet:communication": "communication",
    "usas:inquiry_discovery_and_education": "inquiry, education",
    "wordnet:cognition": "cognition",
    "usas:cognition_mental": "cognition (2nd source)",
    "wordnet:perception": "PERCEPTION",
    "usas:evaluation_modality": "evaluation, modality",
}
#: P states 112 fields / 448 tests / 109 surviving. The committed artifact is a
#: WIDER run and does not reproduce those counts; the z-structure does.
BOOKED_COUNTS = {"doc_fields": 112, "doc_tests": 448, "doc_q05": 109}

#: Red and blue are this campaign's faller/riser grammar, and here the two
#: poles ARE fall and rise, so reusing them is the convention rather than a
#: collision. The wedge gets a third colour because it is the one row whose
#: argument is that it sits on the wrong side of a story.
FALL_C, RISE_C, WEDGE_C = "#b03030", "#1f4e79", "#e07b39"


def field_poles():
    """P 15(3): which semantic fields fall and which rise, and the wedge."""
    from plotnine import (aes, element_blank, element_text, geom_point,
                          geom_segment, geom_text, geom_vline, ggplot,
                          labs, scale_color_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)

    d = json.load(open(os.path.join(K, "field_poles_en.json")))
    by = collections.defaultdict(list)
    for x in d["tests"]:
        by[x["field"]].append(x)

    assert d["instruments"] == ["armAUC", "axisGloVe", "axisBGE", "delta"], \
        f"instrument set changed: {d['instruments']}"
    #: THE COUNTS DO NOT REPRODUCE AND THAT IS DECLARED, NOT ASSERTED AWAY.
    #: P section 7b reports 112 fields over 448 tests with 109 surviving; this
    #: artifact is a wider run at 175/700/152. BH runs over the whole test set,
    #: so a 448-test correction gives different q than a 700-test one and the
    #: doc's counts cannot be recovered by filtering this file. The Z-STRUCTURE
    #: does reproduce, within 0.08 on every row, and that is what is drawn.
    assert (d["n_fields"], d["n_tests"]) != (BOOKED_COUNTS["doc_fields"],
                                             BOOKED_COUNTS["doc_tests"]), \
        ("this artifact now matches the doc's field count -- the caption says "
         "it does not and would be wrong")

    rows, drift = [], []
    for f, booked in BOOKED_7B.items():
        xs = by.get(f, [])
        assert len(xs) == 4, f"{f}: {len(xs)} instrument rows, expected 4"
        m = st.mean(x["z"] for x in xs)
        assert abs(m - booked) <= 0.10, \
            (f"{f}: mean z {m:+.3f} against booked {booked:+.2f}, beyond the "
             "0.10 that a resampled size-matched null explains")
        assert len({x["z"] > 0 for x in xs}) == 1, \
            f"{f}: the four instruments no longer agree in sign"
        drift.append(abs(m - booked))
        wedge = f == "wordnet:perception"
        rows.append({"field": f, "lab": SHORT[f], "z": m, "n": xs[0]["n"],
                     "nsig": sum(1 for x in xs if x["q"] < 0.05), "wedge": wedge,
                     "fill": WEDGE_C if wedge else (FALL_C if m > 0 else RISE_C)})

    #: the wedge carries the argument and its support is the weakest drawn
    w = next(r for r in rows if r["wedge"])
    assert w["nsig"] == 3, \
        (f"perception now survives FDR on {w['nsig']} of 4 instruments, not 3; "
         "the panel says three and names which one fails")
    arm = next(x for x in by["wordnet:perception"] if x["instrument"] == "armAUC")
    assert arm["q"] > 0.05, "armAUC now survives on perception; the caption is wrong"

    df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True)
    df["y"] = range(len(df))
    df["nlab"] = [f"{r.nsig}/4" for r in df.itertuples()]
    #: LABEL BEYOND THE FURTHEST DOT, NOT BEYOND THE BAR. The per-instrument
    #: points routinely sit outside the mean -- contact's bar ends at 8.8 and
    #: its instruments reach 10.4 -- so anchoring on the bar end puts the text
    #: on top of them. The first version anchored on the bar AND inverted the
    #: alignment, which printed every label inside its own bar.
    ext = {f: [x["z"] for x in by[f]] for f in BOOKED_7B}
    df["ha"] = ["right" if v < 0 else "left" for v in df.z]
    df["nx"] = [(min(ext[f]) - 0.35) if v < 0 else (max(ext[f]) + 0.35)
                for f, v in zip(df.field, df.z)]
    #: the FDR count sits just inside the bar, on the bar's own side of zero
    df["cx"] = [(v + 0.55) if v < 0 else (v - 0.55) for v in df.z]
    pts = pd.DataFrame([{"z": x["z"], "y": int(df.index[df.field == f][0])}
                        for f in BOOKED_7B for x in by[f]])

    smallest = min(abs(v) for v in BOOKED_7B.values())
    n_omitted = sum(1 for f, xs in by.items()
                    if len(xs) == 4 and f not in BOOKED_7B
                    and abs(st.mean(x["z"] for x in xs)) >= smallest)

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.5)
        #: HORIZONTAL BARS AS SEGMENTS. This plotnine has no `orientation`
        #: parameter on geom_col, and coord_flip would rotate the text
        #: anchors with the panel. A thick segment from zero is the same
        #: mark with none of that.
        + geom_segment(df, aes(0, "y", xend="z", yend="y", color="fill"),
                       size=9.5)
        + geom_point(pts, aes("z", "y"), size=1.4, color="#2b2b2b", alpha=0.7)
        + geom_text(df, aes("nx", "y", label="lab", ha="ha"), size=7.4,
                    color="#222222")
        + geom_text(df, aes("cx", "y", label="nlab"), size=5.8,
                    color="#ffffff")
        + scale_color_identity()
        + scale_x_continuous(limits=(-12.0, 14.0),
                             breaks=[-8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        + scale_y_continuous(breaks=[], limits=(-0.8, len(df) - 0.2))
        + labs(
            title="Perception rises with cognition, and a concrete-to-abstract axis cannot produce that row",
            subtitle=(
                "Semantic fields ranked by how their words move under alignment. Each bar is the mean z\n"
                "over FOUR INSTRUMENTS -- arm AUC, the GloVe axis, the bge axis, and the site delta --\n"
                "against size-matched nulls, and the four dots on each bar are those instruments, drawn\n"
                "beside the mean rather than behind it.\n"
                "RIGHT, RED: fields whose words FALL. Contact, handling, consumption, motion, the body:\n"
                "the vocabulary of immediate physical doing.\n"
                "LEFT, BLUE: fields whose words RISE. Communication, inquiry, cognition from two\n"
                "independent sources, evaluation: mental and institutional predicates.\n"
                "THE WEDGE IS PERCEPTION, IN ORANGE. Perception verbs are concrete -- they are done with\n"
                "the body -- and they rise with cognition. A pure concrete-to-abstract reading has no way\n"
                "to put that row on the left, which is why the working name for the rise pole is\n"
                "INTERIORITY and the description surviving elimination is ENACTED -> REPRESENTED.\n"
                "THE FIGURE IS NOT KIND TO ITS OWN WEDGE. The number inside each bar is how many of the\n"
                "four instruments survive FDR, and perception is 3 of 4: arm AUC gives it z = -1.33 at\n"
                "q = 0.36 while the other three sit at q = 0.0049. All four agree in SIGN, which is what\n"
                "the finding claims, and they do not all agree in significance, which it does not."),
            x="mean z against a size-matched null   (negative = rises under alignment)",
            y="",
            caption=(
                "Producer: meta/M01_displacement/scripts/plot_p_figs.py from results/k/field_poles_en.json.\n"
                "Asserted before drawing: the four named instruments, sign agreement across all four on\n"
                "every field drawn, each mean z within 0.10 of P section 7b's booked value, and that\n"
                "perception survives on exactly 3 of 4 with arm AUC the failure.\n"
                "THE COUNTS DO NOT REPRODUCE AND THE Z-STRUCTURE DOES. P section 7b reports 112 fields\n"
                "over 448 tests with 109 surviving q<0.05 against about 22 expected. This committed\n"
                "artifact is a wider run: 175 fields, 700 tests, 152 surviving. Benjamini-Hochberg runs\n"
                "over the whole test set, so a 448-test correction gives different q values than a\n"
                "700-test one and the doc's counts cannot be recovered by filtering this file. The\n"
                "109/448 quoted in the finding belongs to a run no committed artifact holds.\n"
                "Every mean z agrees within 0.08, which is what a resampled size-matched null would give.\n"
                f"THE ELEVEN FIELDS ARE THE ONES SECTION 7b NAMES, AND THAT IS A SELECTION: {n_omitted}\n"
                "other fields have a larger mean |z| than the smallest drawn here and are not shown,\n"
                "almost all of them bare USAS codes rather than interpretable names. Which eleven is a\n"
                "reading; that the four instruments agree in sign on all of them is a measurement."),
        )
        + theme_minimal()
        + theme(figure_size=(12.8, 7.6),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "p_field_poles.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    11 fields, max |mean z - booked| = {max(drift):.3f}")
    print(f"    artifact {d['n_fields']} fields / {d['n_tests']} tests / "
          f"{d['n_q05']} q<0.05; doc books {BOOKED_COUNTS['doc_fields']}/"
          f"{BOOKED_COUNTS['doc_tests']}/{BOOKED_COUNTS['doc_q05']}")
    print(f"    perception (the wedge): {w['nsig']}/4 survive FDR, "
          f"armAUC q={arm['q']:.4f}")
    print(f"    {n_omitted} unnamed fields outrank the smallest drawn, declared")
    return out


#: Booked in plot-debt 15(4) and verified against the artifact.
BOOKED_AUC = {"n_words": 4106, "n_clearing": 900, "pct_clearing": 21.9,
              "threshold": 0.15}
#: Irregular past forms, declared BEFORE the tails were inspected so the class
#: contrast is a test rather than a description of what was seen. A regular
#: `-ed` test is the WRONG instrument here and reverses the result: the rise
#: tail is full of regular past participles (provided, examined, assessed)
#: while the fall tail's past tense is almost entirely irregular, which is
#: what high-frequency bodily verbs look like in English.
IRREGULAR_PAST = {
    "went", "told", "threw", "wrote", "said", "was", "were", "had", "got",
    "gave", "took", "came", "saw", "knew", "made", "put", "let", "felt",
    "left", "kept", "held", "found", "thought", "brought", "caught", "sat",
    "stood", "ran", "began", "broke", "drove", "fell", "heard", "hit", "lay",
    "led", "lost", "met", "paid", "read", "rode", "rose", "sold", "sent",
    "shot", "shut", "sang", "spoke", "spent", "struck", "swore", "tore",
    "woke", "wore", "won", "beat", "bit", "blew", "burnt", "chose", "dug",
    "drew", "ate", "flew", "forgot", "froze", "grew", "hung", "hid", "knelt",
    "laid", "lit", "meant", "rang", "sank", "slept", "slid", "smelt", "stole",
    "stuck", "stung", "swam", "swung", "taught", "understood", "wound",
}
TAIL = 0.35            #: fall tail at or below; rise tail at or above 1 - TAIL
HIST_C, TAIL_FALL_C, TAIL_RISE_C = "#c9c9c9", "#b03030", "#1f4e79"


def arm_auc():
    """P 15(4): how much of the vocabulary separates the arms at all."""
    import csv
    from plotnine import (aes, element_blank, element_text, geom_histogram,
                          geom_text, geom_vline, ggplot, labs,
                          scale_fill_identity, scale_x_continuous, theme,
                          theme_minimal)

    src = os.path.join(K, "word_auc_en.tsv")
    rows = list(csv.DictReader(open(src), delimiter="\t"))
    for r in rows:
        r["a"] = float(r["auc"])

    assert len(rows) == BOOKED_AUC["n_words"], \
        f"population drifted: {len(rows)} vs booked {BOOKED_AUC['n_words']}"
    clearing = [r for r in rows if abs(r["a"] - 0.5) >= BOOKED_AUC["threshold"]]
    assert len(clearing) == BOOKED_AUC["n_clearing"], \
        (f"words clearing |{BOOKED_AUC['threshold']}| drifted: {len(clearing)} "
         f"vs booked {BOOKED_AUC['n_clearing']}")
    pct = 100 * len(clearing) / len(rows)
    assert abs(pct - BOOKED_AUC["pct_clearing"]) < 0.1, \
        f"share drifted: {pct:.1f}% vs booked {BOOKED_AUC['pct_clearing']}%"

    #: THE TAIL CHARACTERISATION IS MEASURED, NOT DESCRIBED. §3c fences the
    #: vocabulary itself -- "a figure quoting one as the finding quotes the
    #: sampling noise along with it" -- so no word list appears on this panel.
    #: What appears is a class contrast that can be tested: irregular past
    #: forms concentrate in the fall tail by an order of magnitude.
    v = [r for r in rows if r["tag"] == "verb"]
    fall = [r for r in v if r["a"] <= TAIL]
    rise = [r for r in v if r["a"] >= 1 - TAIL]
    mid = [r for r in v if TAIL < r["a"] < 1 - TAIL]

    def irr(g):
        return 100 * sum(1 for r in g if r["word"] in IRREGULAR_PAST) / len(g)

    i_fall, i_mid, i_rise = irr(fall), irr(mid), irr(rise)
    assert i_fall > 5 * i_mid and i_fall > 5 * i_rise, \
        (f"the irregular-past concentration collapsed: fall {i_fall:.1f}% "
         f"mid {i_mid:.1f}% rise {i_rise:.1f}%; the panel states it as an "
         "order-of-magnitude contrast")

    d = pd.DataFrame({"auc": [r["a"] for r in rows]})
    d["fill"] = [TAIL_FALL_C if a <= 0.5 - BOOKED_AUC["threshold"]
                 else TAIL_RISE_C if a >= 0.5 + BOOKED_AUC["threshold"]
                 else HIST_C for a in d.auc]

    note = pd.DataFrame([
        #: y=300 put the first line against the panel ceiling and clipped it.
        #: The tallest bin is ~170, so 250 clears the data and the frame both.
        {"x": 0.20, "y": 250,
         "t": f"FALLS UNDER ALIGNMENT\n{len(fall)} verbs at or below {TAIL}\n"
              f"irregular past tense: {i_fall:.1f}%\n"
              f"against {i_mid:.1f}% in the middle"},
        {"x": 0.80, "y": 250,
         "t": f"RISES UNDER ALIGNMENT\n{len(rise)} verbs at or above {1 - TAIL}\n"
              f"irregular past tense: {i_rise:.1f}%\n"
              "regular -ed forms instead"}])

    p = (
        ggplot()
        + geom_histogram(d, aes("auc", fill="fill"), bins=70, colour=None)
        + geom_vline(xintercept=[0.35, 0.65], linetype="dashed",
                     color="#555555", size=0.4)
        + geom_vline(xintercept=0.5, color="#333333", size=0.4)
        + geom_text(note, aes("x", "y", label="t"), size=6.6, color="#333333",
                    lineheight=1.3)
        + scale_fill_identity()
        + scale_x_continuous(breaks=[0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
        + labs(
            title="Most of the vocabulary does not separate the arms: 78% of 4,106 words sit inside the null band",
            subtitle=(
                "Per-word arm AUC: how well a single word's probability tells an aligned checkpoint from\n"
                "its base. 0.5 is no information. The coloured tails are the 900 words clearing 0.15 in\n"
                "either direction -- 21.9% of the vocabulary -- and the grey majority is the finding:\n"
                "alignment is not a broad relabelling of the lexicon.\n"
                "THE TAILS ARE CHARACTERISED, NOT LISTED, AND THAT IS THIS SECTION'S OWN FENCE. P section\n"
                "3c: a specific hundred-word list is an unstable SAMPLE of a real direction, unstable\n"
                "because tails are, so a figure quoting one as the finding quotes the sampling noise\n"
                "along with it. No word appears on this panel.\n"
                "WHAT APPEARS INSTEAD IS A CLASS CONTRAST THAT CAN BE TESTED. Irregular past-tense forms\n"
                "are 13 times as concentrated in the fall tail as in the rise tail, which is what the\n"
                "high-frequency bodily verbs of English look like morphologically.\n"
                "A REGULAR -ed TEST REVERSES THIS AND IS THE WRONG INSTRUMENT: the rise tail is full of\n"
                "regular past participles, so `-ed` share climbs with AUC while irregular past collapses.\n"
                "The suffix is a name; the tense is the relation."),
            x="arm AUC per word   (0.5 = the word carries no information about the arm)",
            y="words",
            caption=(
                "Producer: meta/M01_displacement/scripts/plot_p_figs.py from results/k/word_auc_en.tsv.\n"
                "Asserted before drawing: 4,106 words, 900 clearing |0.15| at 21.9%, and that irregular\n"
                "past forms are at least five times as concentrated in the fall tail as in either the\n"
                "middle or the rise tail.\n"
                "The irregular-past list is declared in the producer BEFORE the tails were inspected, so\n"
                "the class contrast is a test rather than a description of what was seen. Verb tails only\n"
                "for that contrast (2,150 of the 4,106 are verbs); the histogram is all four tags.\n"
                "Tail boundaries at 0.35 and 0.65 are the panel's, not the finding's; the 0.15 band is\n"
                "the queue entry's and is what the 21.9% counts."),
        )
        + theme_minimal()
        + theme(figure_size=(12.4, 6.8),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank())
    )
    out = os.path.join(FIGURES, "p_arm_auc_distribution.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {len(rows)} words, {len(clearing)} clearing |0.15| = {pct:.1f}%")
    print(f"    irregular past: fall {i_fall:.1f}%  middle {i_mid:.1f}%  "
          f"rise {i_rise:.1f}%  ({i_fall / i_rise:.0f}x fall-vs-rise)")
    print(f"    no word list on the panel, per section 3c")
    return out


#: P section 7's ledger. R2 is the share of the AXIS's word-level variance a
#: component explains (`k_length.py:13`), so these are not slices of one pie.
BOOKED_7 = {"register": 0.1994, "brysbaert": 0.1183, "coder_conc": 0.0921,
            "length_bge": 0.1386, "length_glove": 0.0921, "coder_reg": 0.0470}
#: measured in confound_en.json; the reason the bars are not stacked
BOOKED_RHO_REG_CONC = 0.493
#: THE [5606] WELD. §7: register-after-frequency leads concreteness by 0.046 on
#: this axis and by 0.013 on the refitted verb-eliciting one, so "the largest
#: single named component" is a much weaker claim on the population where the
#: question is best posed and "should not be quoted without this row". The
#: entry requires it welded on, so the register bar carries it as a second mark
#: rather than a footnote.
BOOKED_RESID = 0.1641


def ledger():
    """P 15(2): what the direction is made of, and none of it is most of it."""
    import csv
    from plotnine import (aes, element_blank, element_text, geom_point,
                          geom_segment, geom_text, ggplot, labs,
                          scale_color_identity, scale_x_continuous,
                          scale_y_continuous, theme, theme_minimal)

    reg = json.load(open(os.path.join(K, "register_en.json")))["index_table"]
    con = json.load(open(os.path.join(K, "concreteness_en.json")))["measures"]
    dec = json.load(open(os.path.join(K, "register_decomp_en.json")))["measures"]
    lbg = json.load(open(os.path.join(K, "length_en_bge.json")))
    lgl = json.load(open(os.path.join(K, "length_en_glove.json")))
    cnf = json.load(open(os.path.join(K, "confound_en.json")))

    #: every row §7 prints, against the artifact that now holds it
    checks = [
        ("register", reg["SUBTLEX_over_coca_acad"]["r2_axis"]),
        ("brysbaert", con["Brysbaert Conc.M"]["r2_axis"]),
        ("coder_conc", con["coder concreteness"]["r2_axis"]),
        ("length_bge", lbg["r2_length"]),
        ("length_glove", lgl["r2_length"]),
        ("coder_reg", dec["coder register_level"]["r2_axis"]),
    ]
    for name, got in checks:
        assert abs(got - BOOKED_7[name]) < 5e-5, \
            f"{name}: artifact {got:.6f} vs section 7's {BOOKED_7[name]}"
    #: THE COINCIDENCE THAT COST TWO SEATS AN HOUR, PINNED SO IT CANNOT
    #: SILENTLY BECOME A DUPLICATION. coder concreteness and length/glove both
    #: print 0.0921 and are 3.9e-05 apart; I read that as one row carrying the
    #: other's value and was wrong ([6169]). If they ever become equal, that
    #: IS the defect I claimed and this refuses to draw.
    gap = abs(con["coder concreteness"]["r2_axis"] - lgl["r2_length"])
    assert 1e-6 < gap < 1e-4, \
        (f"coder-concreteness and length/glove R2 are {gap:.2e} apart; §7 "
         "prints both as 0.0921 and they are distinct quantities that happen "
         "to round together")

    resid = dec["SUBTLEX resid on freq"]["r2_axis"]
    assert abs(resid - BOOKED_RESID) < 5e-5, \
        f"frequency-residualised register drifted: {resid:.6f} vs {BOOKED_RESID}"
    #: the weld's whole point: the lead over concreteness narrows once
    #: frequency comes out, and the panel must not show only the raw row
    lead_raw = BOOKED_7["register"] - BOOKED_7["brysbaert"]
    lead_res = resid - BOOKED_7["brysbaert"]
    assert lead_res < lead_raw, \
        "residualising frequency no longer narrows register's lead; the panel says it does"

    rho = cnf["bundle_rho"]["register index"]["concreteness"]
    assert abs(rho - BOOKED_RHO_REG_CONC) < 0.001, \
        f"register-concreteness rho drifted: {rho:.4f}"

    #: each component with EVERY independent measure of it, because the
    #: agreement between measures is the evidence the component is real
    comps = [
        ("register\n(4 genre indices)",
         [v["r2_axis"] for v in reg.values()], "#1a7a6a", False),
        ("concreteness\n(2 measures)",
         [con["Brysbaert Conc.M"]["r2_axis"], con["coder concreteness"]["r2_axis"]],
         "#1a7a6a", False),
        ("word length\n(2 encoders)",
         [lbg["r2_length"], lgl["r2_length"]], "#9a9a9a", False),
        ("coder register_level\n(1 scale)",
         [dec["coder register_level"]["r2_axis"]], "#9a9a9a", False),
    ]
    rows, pts = [], []
    for i, (lab, vals, col, unemitted) in enumerate(comps):
        #: ANCHOR PAST THE FURTHEST DOT, NOT PAST THE BAR. Identical to the
        #: field-poles fix and to the rule in this seat's own notes, which I
        #: wrote down and then did not apply here: the per-measure dots sit
        #: outside the mean, so a bar-anchored label lands on top of them.
        rows.append({"y": len(comps) - 1 - i, "lab": lab,
                     "m": sum(vals) / len(vals), "col": col, "lx": max(vals),
                     "n": len(vals), "unemitted": unemitted})
        for v in vals:
            pts.append({"y": len(comps) - 1 - i, "r2": v})
    df, pd_pts = pd.DataFrame(rows), pd.DataFrame(pts)

    p = (
        ggplot()
        + geom_segment(df, aes(0, "y", xend="m", yend="y", color="col"), size=11)
        + geom_point(pd_pts, aes("r2", "y"), size=2.0, color="#2b2b2b", alpha=0.8)
        #: THE WELD, ON THE BAR IT QUALIFIES
        + geom_point(pd.DataFrame([{"r2": resid, "y": len(comps) - 1}]),
                     aes("r2", "y"), size=4.2, color="#b03030", shape="D")
        + geom_text(pd.DataFrame([{"r2": resid, "y": len(comps) - 1}]),
                    aes("r2", "y", label=f'"after frequency: {resid:.3f}"'),
                    #: -0.30 left the label inside the bar it annotates; the segment
                    #: is 11pt so it needs clearing properly.
                    size=6.4, color="#b03030", va="top", nudge_y=-0.46)
        + geom_text(df, aes("lx", "y", label="lab"), size=7.2, ha="left",
                    nudge_x=0.009, color="#222222", lineheight=1.15)
        + scale_color_identity()
        + scale_x_continuous(limits=(0, 0.34),
                             breaks=[0, 0.05, 0.10, 0.15, 0.20],
                             labels=["0%", "5%", "10%", "15%", "20%"])
        + scale_y_continuous(breaks=[], limits=(-0.7, len(comps) - 0.3))
        + labs(
            title="No named component explains most of the direction, and they cannot be added up",
            subtitle=(
                "Each bar is the share of the AXIS's word-level variance that one named component\n"
                "explains, with every independent measure of that component drawn as a dot. The\n"
                "largest is register at 19.9%.\n"
                "THESE ARE NOT SLICES AND THERE IS NO REMAINDER REGION ON THIS PANEL. Register and\n"
                f"concreteness correlate at rho {BOOKED_RHO_REG_CONC}, and section 7b states that\n"
                "interiority and abstraction are colinear BY CONSTRUCTION -- mental events score low\n"
                "on concreteness by instruction. So 'removing register costs half the prediction' and\n"
                "'removing concreteness costs a quarter' are two COUNTERFACTUALS, not two shares:\n"
                "they overlap, they can sum past one, and an empty wedge labelled UNNAMED would be a\n"
                "partition claim this finding explicitly denies.\n"
                "WHAT THE MULTIPLE DOTS ARE FOR. Four genre indices for register spanning 16.8-19.9%,\n"
                "two concreteness measures at 9.2% and 11.8%, two encoders for length. Components\n"
                "whose independent measures agree are real; the agreement is the evidence, not the\n"
                "single number, and it is the same argument the headroom ladder makes for GloVe\n"
                "against bge.\n"
                "LENGTH IS RULED OUT DESPITE ITS SIZE. Projecting it out rotates the axis by under 13\n"
                "degrees and costs nothing predictively, which is why it is grey: a component can\n"
                "explain variance in the axis and carry none of its prediction.\n"
                "AND REGISTER'S LEAD IS MOSTLY FREQUENCY, WHICH IS WHY THE RED MARKER IS ON ITS BAR.\n"
                f"Residualised on word frequency it falls to {resid:.3f}, and its lead over concreteness\n"
                f"narrows from {lead_raw:.3f} to {lead_res:.3f} -- to 0.013 on the refitted verb-eliciting\n"
                "axis, where section 7 says the question is best posed. 'The largest single named\n"
                "component' is a much weaker claim than the raw bar alone would suggest, and section 7\n"
                "says it should not be quoted without this row."),
            x="share of the axis's word-level variance explained (R2)", y="",
            caption=(
                "Producer: meta/M01_displacement/scripts/plot_p_figs.py from results/k/{register_en,\n"
                "concreteness_en, length_en_bge, length_en_glove, confound_en}.json.\n"
                "Asserted before drawing: ALL SIX of section 7's rows against the artifacts that hold\n"
                "them, each within 5e-05; the frequency-residualised row and that it narrows register's\n"
                "lead; and the register-concreteness rho.\n"
                "ALL SIX ROWS OF SECTION 7 NOW REPRODUCE. Three of them did not this morning: two\n"
                "producers wrote nothing at all and a third wrote an artifact not holding the cited\n"
                "quantity. They were emitted on 2026-08-14 in response to this figure.\n"

                "One assert here guards a coincidence rather than a value: coder concreteness and\n"
                "length/glove both print 0.0921 and are 3.9e-05 apart. I reported that as one row\n"
                "carrying the other's value and was wrong; the guard now refuses to draw if they ever\n"
                "become equal, which is the defect I claimed."),
        )
        + theme_minimal()
        + theme(figure_size=(12.4, 5.8),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "p_named_components.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for r in rows:
        tag = "  PRINTED ONLY" if r["unemitted"] else ""
        print(f"    {r['lab'].splitlines()[0]:<22} R2 {r['m']:.4f} over {r['n']} measure(s){tag}")
    print(f"    register x concreteness rho {rho:.3f} -- not stacked, no remainder region")
    return out


REGISTRY = {"headroom": headroom, "field_poles": field_poles,
            "arm_auc": arm_auc, "ledger": ledger}


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
