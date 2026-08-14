#!/usr/bin/env python
"""Figures for Findings H (H_norm_acquisition.md): the norm signature.

    uv run python meta/M05_emergence/scripts/m05_norm_figs.py
    uv run python meta/M05_emergence/scripts/m05_norm_figs.py stage_matrix
    uv run python meta/M05_emergence/scripts/m05_norm_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.
Naming follows THIS folder: numbered `figNN_` outputs from per-purpose
`m05_*` scripts (registrar, [5889]), continuing from fig29.

WHY THE STAGE MATRIX PLOTS A SIGN SPLIT AND NOT A MEDIAN
---------------------------------------------------------
The obvious encoding is the per-cell median increment. It is the wrong
one here and the artifact says so if you read the whole row.

Ties dominate several scales. `vulgarity` has 252 to 339 tied prompts of
581, so its median is exactly 0.0000 in four of five transitions while
its sign test runs to p 1.3e-25 (RLVR-DPO, 42 up against 199 down). A
median-coloured matrix would show those cells as "no change" and the
strongest tie-dominated result in the table would be invisible.

So position is the share of NON-TIED prompts moving up, centred on 0.5,
which is what the finding's own sign test measures; and mark AREA is the
non-tied count, so a cell resting on 241 movers is visibly lighter
evidence than one resting on 570. Ties are not hidden, they are the
thing that shrinks the mark.

This is the campaign's quote-the-counts rule ([5899]) applied to a
matrix: the statistic that survives the ties is the count, and the
median is the summary that does not.

AND THE COLUMNS DO NOT COMBINE, WHICH THE LAYOUT INVITES
--------------------------------------------------------
Three stage transitions sit beside two NETs, so the panel invites the
reader to add the first three and check the fourth. The finding forbids
it in a pre-agreed rider: NETs are their own paired per-prompt
contrasts, summed stage medians do not equal the median NET, and any
reading that adds rows 1-3 to predict row 4 is wrong by construction.

The first version of this figure omitted that rider. It was added after
@malign generalised the same defect from a different operator at
[5994]: the mean is linear and commutes with any linear combination of
its inputs, the median commutes with none, so a median-aggregated table
admits exactly one safe operation -- reading a row. The share-of-
non-tied encoding used here does not add either, so the rider applies to
this panel's own quantity and not only to the medians it replaced.

THE PYTHIA PANEL STARTS WHERE THE INSTRUMENT DOES
-------------------------------------------------
The finding's pre-agreed fence: below step 8 the instrument resolves a
median 0.5-0.7% of next-word mass and `k_rated_mass_share` sits at 0.63,
so every composition number below step ~64 is a reading of a sliver and
is NOT QUOTABLE. The document's own words are "the panel starts where
the instrument does".

That fence is load-bearing rather than decorative, and the shape of the
data is why: at step 0 the median concreteness reads 2.65, ABOVE the
1.08 floor it occupies once coverage reaches 1.0. Drawn without the
fence the curve opens with a dramatic collapse from 2.65 to 1.00 that a
reader would take for the earliest and largest event in pretraining. It
is the instrument reporting on a sliver. The pre-fence region is drawn
greyed and behind a boundary, and the curve of record starts at 128.

MEDIANS, BECAUSE THE SCALES ARE RANKS
-------------------------------------
Every number here is a median over prompts. The k_ riders that travel
with this finding say RANKS NOT LEVELS, medians compare and differences
do not scale. A mean over prompts gives 1.23 at step 128 against the
booked 1.08 and 3.04 at the final rung against 2.87, so the choice is
not cosmetic. Third time in one session that the median/mean grain has
decided whether a reconstruction matched; noted here so the next reader
of THIS folder meets it in the file they are editing.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
INCR = os.path.join(RESULTS, "norm_acquisition_increments.json")
MASS = os.path.join(ROOT, "data", "m05_norm_mass.parquet")

TRANS = ["SFT-base", "DPO-SFT", "RLVR-DPO", "NET base->DPO", "NET base->RLVR"]
TLABEL = {"SFT-base": "SFT\ninstalls", "DPO-SFT": "DPO\nbuys back",
          "RLVR-DPO": "RLVR\nre-suppresses",
          "NET base->DPO": "NET\nbase to DPO", "NET base->RLVR": "NET\nbase to RLVR"}
SCALES = ["transgressiveness", "bodily_harm", "concreteness", "charge",
          "valence", "register_level", "vulgarity"]

#: Booked in H_norm_acquisition.md, from the increments artifact.
BOOKED = {
    ("DPO-SFT", "concreteness"): (451, 111),      # the table's largest effect
    ("DPO-SFT", "register_level"): (268, 272),    # untouched, p 0.897
    ("RLVR-DPO", "register_level"): (333, 204),   # up, p 2.9e-08
    ("NET base->DPO", "concreteness"): (285, 286),  # the DEAD HEAT
    ("NET base->RLVR", "transgressiveness"): (120, 332),
    ("SFT-base", "transgressiveness"): (120, 332),
}
BOOKED_PYTHIA = {"concreteness": (1.08, 2.87), "charge": (1.01, 1.27),
                 "transgressiveness": (1.000, 1.024)}
FENCE_STEP = 128


def _incr():
    j = json.load(open(INCR))
    rows = []
    for t in TRANS:
        for s in SCALES:
            v = j[f"{t}:{s}"]
            nz = v["up"] + v["dn"]
            rows.append({"trans": t, "scale": s, "up": v["up"], "dn": v["dn"],
                         "ties": v["ties"], "nonzero": nz,
                         "share_up": v["up"] / nz, "median": v["median"],
                         "p": v["p_sign_ties_excluded"]})
    return pd.DataFrame(rows)


def stage_matrix():
    """Findings H: two modules, opposite signs, at checkpoint grain.

    Seven K scales by five stage transitions. Position is the share of
    non-tied prompts moving up; area is the non-tied count.
    """
    from plotnine import (aes, element_text, facet_grid, geom_point,
                          geom_text, geom_vline, ggplot, labs,
                          scale_color_manual, scale_size_area,
                          scale_x_continuous, theme, theme_minimal)

    d = _incr()
    for (t, s), (up, dn) in BOOKED.items():
        r = d[(d.trans == t) & (d.scale == s)].iloc[0]
        assert (int(r.up), int(r.dn)) == (up, dn), \
            f"{t}:{s} drifted: {int(r.up)}/{int(r.dn)} vs booked {up}/{dn}"

    d["dir"] = np.where(d.p >= 0.05, "not significant",
                        np.where(d.share_up > 0.5, "moves UP", "moves DOWN"))
    d["lab"] = d.apply(lambda r: f"{int(r.up)}/{int(r.dn)}", axis=1)
    d["tl"] = pd.Categorical(d.trans.map(TLABEL),
                             categories=[TLABEL[t] for t in TRANS], ordered=True)
    d["scale"] = pd.Categorical(d.scale, categories=SCALES[::-1], ordered=True)

    p = (
        ggplot(d, aes("share_up", "scale"))
        + geom_vline(xintercept=0.5, color="#b03030", linetype="dashed", size=0.45)
        + geom_point(aes(size="nonzero", color="dir"), alpha=0.85)
        + geom_text(aes(label="lab"), size=5.4, color="#444444", nudge_y=0.34)
        + scale_color_manual(values={"moves UP": "#1f4e79", "moves DOWN": "#b03030",
                                     "not significant": "#c9c9c9"}, name="")
        + scale_size_area(max_size=7.5, breaks=[250, 400, 570],
                          labels=lambda bs: [f"{int(b)}" for b in bs],
                          name="non-tied prompts")
        + scale_x_continuous(limits=(0.02, 0.98), breaks=[0.25, 0.5, 0.75],
                             labels=["25%", "50%", "75%"])
        + facet_grid(". ~ tl")
        + labs(
            title="The norm signature: SFT installs it, DPO buys part of it back, RLVR re-suppresses",
            subtitle=(
                "Seven K scales by five stage transitions, 581 prompts on the OLMo ladder "
                "(145,741 cells). Position is the share of NON-TIED prompts moving up; the dashed "
                "line at 50% is no preference.\n"
                "AREA IS THE NON-TIED COUNT, so a cell resting on few movers is visibly lighter "
                "evidence. Ties are not hidden: they shrink the mark. Numbers above each point are "
                "up/down.\n"
                "Position is a sign split rather than a median because ties dominate several scales: "
                "vulgarity's median is exactly 0.0000 in four of five transitions while its sign test "
                "reaches p 1.3e-25.\n"
                "THE DISSOCIATION: DPO reverses the SFT sign on four scales while register_level, the "
                "scale SFT raised, is untouched (268/272, p 0.90). What SFT installs and what DPO "
                "adjusts are different axes.\n"
                "THE THREE STAGE COLUMNS DO NOT ADD TO THE TWO NET COLUMNS. NETs are their own paired "
                "per-prompt contrasts, and any reading that combines SFT, DPO and RLVR to predict a NET "
                "is wrong by construction (the finding's own pre-agreed rider).\n"
                "RANKS NOT LEVELS: medians compare, differences do not scale. register_level is "
                "descriptor-only, construct NOT ESTABLISHED; vulgarity is sparse; CHARGE IS NOT AROUSAL. "
                "Single registrar pass."),
            x="share of non-tied prompts moving UP",
            y="",
            caption=("Producer: meta/M05_emergence/scripts/m05_norm_figs.py from "
                     "results/norm_acquisition_increments.json (producer m05_norm_acquisition.py).\n"
                     "Grey marks are cells whose sign test does not clear 0.05, drawn so the matrix "
                     "shows where nothing happened as well as where something did."),
        )
        + theme_minimal()
        + theme(figure_size=(13.2, 5.4),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                strip_text=element_text(size=8.0, weight="bold"),
                axis_text_y=element_text(size=7.6),
                legend_position="right",
                panel_spacing=0.045)
    )
    out = os.path.join(FIGURES, "fig30_norm_stage_matrix.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    dh = d[(d.trans == "NET base->DPO") & (d.scale == "concreteness")].iloc[0]
    print(f"    dead heat check: NET base->DPO concreteness {int(dh.up)}/{int(dh.dn)} "
          f"median {dh['median']:+.4f} p {dh.p:.3g}")
    return out


def pythia_curve():
    """The Pythia differentiation curve, with the coverage fence drawn.

    Median over prompts per rung. The pre-fence region is greyed rather
    than dropped, because the reader should see WHY it is fenced: the
    curve opens above the floor it later occupies.
    """
    from plotnine import (aes, annotate, element_text, geom_line, geom_point,
                          geom_rect, geom_text, geom_vline, ggplot, labs,
                          scale_color_manual, scale_x_log10, theme,
                          theme_minimal)

    d = pd.read_parquet(MASS)
    #: base_step ONLY. On THIS ladder the final rung carries the same model
    #: twice, once as base_step and once as base_endpoint, so pooling the roles
    #: averages the last point with a duplicate of itself: concreteness 2.8648
    #: pooled against 2.8654 on the ladder, which is the difference between
    #: 2.86 and the booked 2.87 once rounded. This is a pretraining-STEP curve
    #: and the endpoint is a different role, not another rung. Found by the
    #: assert.
    #:
    #: DO NOT GENERALISE THE REASON TO THE OLMO LADDER. It has the identical
    #: surface — one model_id under both roles, 584 prompts each — and is NOT a
    #: duplicate. Two discriminators, cheapest first:
    #:
    #:   1. THE STEP FIELD, no threshold needed (dario, [5947]). Pythia's
    #:      endpoint sits AT the final rung's step, 143000 == 143000. OLMo's
    #:      endpoint is step 0 against a final rung at 1413814. The cases
    #:      differ in a stored field before any value is compared. Not
    #:      sufficient alone — two different checkpoints could share a step —
    #:      but it is free and unambiguous, so run it first.
    #:   2. THE VALUES, which need their grain named or they mean nothing:
    #:      max over 584 prompts of |base_step_final − base_endpoint|, on the
    #:      RESOLVED_MASS column: pythia 6.5e-3, olmo 4.1e-1 (62x).
    #:      The SAME comparison over the seven dist_mean_k_* scale columns:
    #:      pythia 4.5e-2, olmo 3.89 (87x). Both reproduce exactly; they are
    #:      different COLUMNS, not different summaries, and they differ ~8x on
    #:      both sides. A seat reading '6.5e-3' and computing it over the
    #:      scales gets 0.045 and concludes the opposite, so the column is
    #:      part of the number.
    #:
    #: pythia-6.9b is one set of weights scored twice; Olmo-3-1025-7B's
    #: endpoint is the released base against a stage1 rung. Same repo name,
    #: different weights — dropping it as a duplicate would delete the
    #: released checkpoint. The discriminator is whether the VALUES agree to
    #: nondeterminism, never whether the LABELS collide.
    d = d[(d.ladder == "pythia") & (d.role == "base_step")]
    g = (d.groupby("step")
         .agg(**{s: (f"dist_mean_k_{s}", "median") for s in BOOKED_PYTHIA},
              cov=("k_rated_mass_share", "median"))
         .reset_index())

    for s, (at128, final) in BOOKED_PYTHIA.items():
        v128 = float(g.loc[g.step == FENCE_STEP, s].iloc[0])
        vfin = float(g[s].iloc[-1])
        assert round(v128, 2) == round(at128, 2), \
            f"{s} at step {FENCE_STEP} drifted: {round(v128, 3)} vs booked {at128}"
        assert round(vfin, 2) == round(final, 2), \
            f"{s} at final rung drifted: {round(vfin, 3)} vs booked {final}"

    long = g.melt(id_vars=["step", "cov"], value_vars=list(BOOKED_PYTHIA),
                  var_name="scale", value_name="v")
    long["step_p"] = long.step.clip(lower=1)
    fenced = long[long.step < FENCE_STEP]
    kept = long[long.step >= FENCE_STEP]

    rect = pd.DataFrame([{"xmin": 0.8, "xmax": FENCE_STEP,
                          "ymin": 0.9, "ymax": 3.1}])
    cols = {"concreteness": "#1f4e79", "charge": "#c98a2b",
            "transgressiveness": "#7a5195"}

    p = (
        ggplot()
        + geom_rect(rect, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    fill="#999999", alpha=0.16)
        + geom_line(fenced, aes("step_p", "v", color="scale"), size=0.5, alpha=0.30,
                    linetype="dotted")
        + geom_line(kept, aes("step_p", "v", color="scale"), size=0.9)
        + geom_point(kept, aes("step_p", "v", color="scale"), size=0.9, alpha=0.7)
        + geom_vline(xintercept=FENCE_STEP, color="#b03030", size=0.5)
        + annotate("text", x=110, y=3.05, ha="right", va="top", size=6.6,
                   color="#666666",
                   label=("NOT QUOTABLE below step ~64\n"
                          "the instrument resolves 0.5-0.7% of next-word mass\n"
                          "and k_rated_mass_share sits at 0.63"))
        + annotate("text", x=150, y=3.05, ha="left", va="top", size=6.6,
                   color="#b03030", label="the panel starts where the instrument does")
        + scale_color_manual(values=cols, name="")
        + scale_x_log10()
        + labs(
            title="From the function-word floor, pretraining differentiates the norm composition",
            subtitle=(
                "Pythia ladder, base_step rungs only, median over 584 prompts per rung, of the "
                "mass-weighted mean K rating of the resolved next-word distribution.\n"
                "From step 128, where rated coverage reaches ~1.0, the composition starts at the "
                "function-word floor and climbs: concreteness 1.08 to 2.87, charge 1.01 to 1.27, "
                "transgressiveness 1.000 to 1.024 (late, small).\n"
                "THE GREY REGION IS FENCED AND DRAWN SO THE READER CAN SEE WHY: at step 0 the median "
                "concreteness reads 2.65, ABOVE the floor it later occupies.\n"
                "That apparent early collapse is the instrument reporting on a sliver, not an event.\n"
                "RANKS NOT LEVELS: medians compare, differences do not scale. One rung (step 8) is "
                "short 2 of 584 prompts. Valence is omitted: its median is pinned at the scale midpoint "
                "throughout and the median is the wrong summary for it."),
            x="pretraining step (log scale)",
            y="median K rating of the resolved distribution",
            caption=("Producer: meta/M05_emergence/scripts/m05_norm_figs.py from "
                     "data/m05_norm_mass.parquet (producer m05_norm_acquisition.py).\n"
                     "Step 0 is drawn at 1 so it survives the log axis; the fenced region is dotted "
                     "and greyed rather than dropped."),
        )
        + theme_minimal()
        + theme(figure_size=(11.6, 5.6),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                legend_position="right")
    )
    out = os.path.join(FIGURES, "fig31_pythia_norm_curve.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for s in BOOKED_PYTHIA:
        print(f"    {s:18s} step {FENCE_STEP} {float(g.loc[g.step == FENCE_STEP, s].iloc[0]):.4f}"
              f"  ->  final {float(g[s].iloc[-1]):.4f}")
    return out


FIGURES_REGISTRY = {"stage_matrix": stage_matrix, "pythia_curve": pythia_curve}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in FIGURES_REGISTRY.items():
            print(f"  {k:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
