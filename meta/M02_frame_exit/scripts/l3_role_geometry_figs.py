#!/usr/bin/env python
"""t and resid by role: the shadow separates the roles and the geometry does not.

    uv run python meta/M02_frame_exit/scripts/l3_role_geometry_figs.py
    uv run python meta/M02_frame_exit/scripts/l3_role_geometry_figs.py --list

plot-debt M02 candidate 3. plotnine at 300 dpi, output to ../figures/. Case 1
by shape: reads a committed parquet and writes only pixels.

EVERY VALUE HERE IS MEASURED AND NONE IS QUOTED, BY RULING
------------------------------------------------------------
`pole_axis_t_is_not_superposition.md` prints a four-row table of t and resid by
role. **Two of its four t values do not reproduce from this artifact and the
table has no producer** -- established at [6233] and corroborated independently
at [6234]:

    role           booked t   measured t
    both             0.453      0.4514    matches
    control_a        0.793      0.7939    matches
    control_b        0.128      0.1180    MISS
    both_matched     0.412      0.4331    MISS

Two roles landing on the digit and two not is not the shape of a wrong
aggregation; that would move all four. The resid thresholds split the same way
(> 1.0 reproduces, > 2.0 does not), locating the difference in the tail. The
staleness explanation was checked and refuted: the artifact at the commit where
the numbers entered gives the same means to three decimals.

Registrar's ruling at [6235]: **draw it with measured values labelled as
measured, and do not put the table's numbers on the panel**, because a figure
carrying unreproducible numbers would launder them. Where the drawn values
disagree with the document, the disagreement is a finding and goes on the panel.

THE PRODUCER HOLDS A FENCE ITS OWN DOCUMENT NEVER STATES
----------------------------------------------------------
`l3_geometry.py:234` prints `CONTROL COVERAGE -- TWO n IN ONE FRAME. NEVER
REPORT ONE`, and scopes the BOTH-vs-control contrast to CONTROLS_SCORED while
the base-vs-aligned contrast uses both strata. **A panel putting all four roles
on one axis IS the control contrast**, so it runs on CONTROLS_SCORED and both n
are declared. That rule exists only in a print statement.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
SRC = os.path.join(RESULTS, "l3_geometry_union.parquet")

ROLES = ["control_b", "both", "both_matched", "control_a"]
#: measured on CONTROLS_SCORED, which is the population the producer's fence
#: assigns to a role contrast. Booked here as THIS PANEL'S measurements.
MEASURED = {"control_b": (0.1180, 0.8456), "both": (0.4514, 0.9630),
            "both_matched": (0.4331, 0.7540), "control_a": (0.7939, 0.8693)}
#: the two n the fence demands
BOOKED_N = {"scored_pairs": 44, "all_pairs": 52, "scored_rows": 361998,
            "undefined": 9522, "t_hidden": 1410, "r_hidden": 2290}
WINDOWS = {"t": (-1.0, 2.0), "r": (0.0, 3.0)}
#: the two of four the document's table gets right, and the two it does not
DOC_T = {"both": 0.453, "control_a": 0.793, "control_b": 0.128,
         "both_matched": 0.412}
DOC_AGREES = {"both", "control_a"}
PANELS = ["1.  t   -- shadow on the pole axis",
          "2.  resid   -- distance off that axis"]
ROLE_C = {"control_b": "#4a7ab5", "both": "#b03030",
          "both_matched": "#d4762a", "control_a": "#1a7a6a"}


def role_geometry():
    """M02 candidate 3: both and neither cast the same shadow."""
    from plotnine import (aes, element_text, facet_wrap, geom_hline,
                          geom_point, geom_text, geom_violin, ggplot, labs,
                          scale_color_identity, scale_fill_identity, theme,
                          theme_minimal)

    d = pd.read_parquet(SRC)
    assert d.groupby(["base", "aligned"]).ngroups == BOOKED_N["all_pairs"], \
        "the full roster is no longer 52 pairs"
    #: THE ROLE CONTRAST RUNS ON CONTROLS_SCORED, per l3_geometry.py:234
    s = d[d.stratum == "CONTROLS_SCORED"]
    n_pairs = s.groupby(["base", "aligned"]).ngroups
    assert n_pairs == BOOKED_N["scored_pairs"], f"{n_pairs} scored pairs, not 44"
    assert len(s) == BOOKED_N["scored_rows"], f"{len(s)} rows"
    import numpy as np
    assert set(s.role.unique()) == set(ROLES), "the four roles have changed"

    #: THE UNDEFINED CELLS ARE A DEGENERACY, NOT MISSING DATA, and plotnine only
    #: said "stat_ydensity : Removed 19044 rows containing non-finite values".
    #: Every one is at LAYER 0 with pole_sep EXACTLY zero: at the embedding the
    #: two poles have the same representation, so there is no axis to project
    #: onto and t is 0/0. They are excluded by construction rather than by
    #: choice, and the panel must not claim the raw row count as its n.
    fin = s[np.isfinite(s.t) & np.isfinite(s.resid)]
    nan = s[~np.isfinite(s.t)]
    assert len(nan) == BOOKED_N["undefined"], f"{len(nan)} undefined cells"
    assert (~np.isfinite(s.t)).equals(~np.isfinite(s.resid)), \
        "t and resid no longer go undefined on the same rows"
    assert list(nan.layer.unique()) == [0], \
        f"undefined cells are no longer confined to layer 0: {sorted(nan.layer.unique())[:5]}"
    assert (nan.pole_sep == 0).all(), \
        "an undefined cell has a nonzero pole separation; the degeneracy is not 0/0"
    s = fin

    #: every drawn value re-derived; these are THIS PANEL'S numbers
    for r, (bt, br) in MEASURED.items():
        g = s[s.role == r]
        assert abs(g.t.mean() - bt) < 5e-4, f"{r}: t {g.t.mean():.4f} vs {bt}"
        assert abs(g.resid.mean() - br) < 5e-4, f"{r}: resid {g.resid.mean():.4f}"

    #: THE DISAGREEMENT IS ASSERTED, not described. If the document's table ever
    #: starts reproducing, this panel's third paragraph is wrong and it should
    #: refuse rather than keep saying so.
    for r, v in DOC_T.items():
        near = abs(s[s.role == r].t.mean() - v) < 5e-3
        assert near == (r in DOC_AGREES), \
            (f"{r}: agreement with the document's table has changed "
             f"(measured {s[s.role == r].t.mean():.4f}, table {v})")

    #: the claim the two panels exist to make, as a test
    t_spread = max(MEASURED[r][0] for r in ROLES) - min(MEASURED[r][0] for r in ROLES)
    r_spread = max(MEASURED[r][1] for r in ROLES) - min(MEASURED[r][1] for r in ROLES)
    assert t_spread > 3 * r_spread, \
        (f"t spread {t_spread:.3f} is no longer several times the resid spread "
         f"{r_spread:.3f}; the panel's whole contrast is that it is")

    #: THE OUTLIERS SET THE AXIS AND ERASE THE RESULT IF LEFT IN. t is an
    #: unbounded projection and runs -42.6 to +67.8; drawn whole, every violin
    #: collapses to a flat line at zero and the panel says nothing. So each
    #: metric is windowed EXPLICITLY -- subset here, not dropped by a scale --
    #: and the hidden count is declared. The means below are still taken over
    #: ALL finite cells, outliers included, so the printed number is not the
    #: window's mean.
    t_hidden = int(((s.t < WINDOWS["t"][0]) | (s.t > WINDOWS["t"][1])).sum())
    r_hidden = int(((s.resid < WINDOWS["r"][0]) | (s.resid > WINDOWS["r"][1])).sum())
    assert (t_hidden, r_hidden) == (BOOKED_N["t_hidden"], BOOKED_N["r_hidden"]), \
        f"windowed-out counts moved: t {t_hidden}, resid {r_hidden}"
    assert t_hidden / len(s) < 0.01 and r_hidden / len(s) < 0.01, \
        "a window now hides more than 1% of cells; declare it or widen it"

    tw = s[(s.t >= WINDOWS["t"][0]) & (s.t <= WINDOWS["t"][1])]
    rw = s[(s.resid >= WINDOWS["r"][0]) & (s.resid <= WINDOWS["r"][1])]
    long = pd.concat([
        tw.assign(metric=PANELS[0], value=tw.t),
        rw.assign(metric=PANELS[1], value=rw.resid)])
    long["role"] = pd.Categorical(long.role, categories=ROLES, ordered=True)
    long["fill"] = [ROLE_C[r] for r in long.role]

    marks = pd.DataFrame(
        [{"metric": PANELS[0], "yint": v, "lt": "dashed"} for v in (0.0, 0.5, 1.0)]
        + [{"metric": PANELS[1], "yint": 1.0, "lt": "solid"}])
    pts = pd.DataFrame([{"metric": PANELS[0], "role": r, "value": MEASURED[r][0],
                         "lab": f"{MEASURED[r][0]:.3f}"} for r in ROLES]
                       + [{"metric": PANELS[1], "role": r, "value": MEASURED[r][1],
                           "lab": f"{MEASURED[r][1]:.3f}"} for r in ROLES])
    pts["role"] = pd.Categorical(pts.role, categories=ROLES, ordered=True)

    p = (
        ggplot()
        + geom_hline(marks, aes(yintercept="yint", linetype="lt"),
                     color="#888888", size=0.45)
        + geom_violin(long, aes("role", "value", fill="fill"), alpha=0.45,
                      color="#666666", size=0.3, width=0.86)
        + geom_point(pts, aes("role", "value"), size=2.6, color="#1a1a1a")
        + geom_text(pts, aes("role", "value", label="lab"), size=6.6,
                    color="#1a1a1a", nudge_x=0.34)
        + facet_wrap("metric", ncol=2, scales="free_y")
        + scale_fill_identity()
        + scale_color_identity()
        + labs(
            title="The shadow tells the roles apart and the geometry does not: both and neither land in the same place",
            subtitle=(
                f"Distributions over {len(s):,} cells from {n_pairs} model pairs. LEFT: t, the projection of each\n"
                "representation onto the line joining the two poles -- 0 is at pole B, 1 at pole A, and the\n"
                "dashed 0.5 is the midpoint. RIGHT: resid, the off-axis distance as a multiple of the whole\n"
                "pole separation. Black dots are the means, printed.\n"
                "t SEPARATES THE ROLES AND resid DOES NOT, which is why both panels are here and why\n"
                "neither is a result on its own. The same-side conjunctions sit at their poles (0.118 and\n"
                "0.794) and the contradiction sits between (0.451). But every role is about as far off the\n"
                "axis as every other -- 0.754 to 0.963 -- so BOTH is not a point between the poles. It is a\n"
                "point roughly a full pole-separation away in some other direction, whose shadow happens\n"
                "to fall near the middle. The solid line at resid = 1.0 is where the off-axis part equals\n"
                "the entire distance between the poles.\n"
                "SO AN INTERMEDIATE t CANNOT BE READ AS SUPERPOSITION. Three states produce one: a genuine\n"
                "mixture of the poles, a distinct third representation whose projection lands midway, and\n"
                "NEUTRALIZATION where neither pole is strongly represented. The first and third are\n"
                "opposite readings of this finding and t cannot separate them. Both and neither cast the\n"
                "same shadow.\n"
                "EVERY NUMBER ON THIS PANEL IS MEASURED HERE AND NONE IS QUOTED. The document's own table\n"
                "gives t = 0.128 for control_b and 0.412 for both_matched; this artifact gives 0.118 and\n"
                "0.433, and that table has no producer. Its other two rows reproduce. The disagreement is\n"
                "drawn rather than smoothed, and is asserted so this sentence cannot outlive it.\n"
                f"THE AXES ARE WINDOWED AND THE MEANS ARE NOT. t is an unbounded projection running\n"
                f"-42.6 to +67.8; drawn whole it collapses every violin to a flat line at zero. The left\n"
                f"panel shows [-1, 2] and hides {t_hidden:,} cells ({100*t_hidden/len(s):.2f}%); the right shows [0, 3] and hides\n"
                f"{r_hidden:,} ({100*r_hidden/len(s):.2f}%). The printed means are over ALL finite cells, outliers included.\n"
                f"{BOOKED_N['undefined']:,} FURTHER CELLS ARE UNDEFINED AND EXCLUDED BY CONSTRUCTION, not by choice.\n"
                "Every one is at LAYER 0 with a pole separation of exactly zero: at the embedding the two\n"
                "poles have the same representation, so there is no axis to project onto and t is 0/0.\n"
                "plotnine reports that only as a count of non-finite rows.\n"
                f"TWO n IN ONE FRAME, WHICH THE PRODUCER SHOUTS AND THE DOCUMENT DOES NOT SAY. A four-role\n"
                f"panel is the BOTH-versus-control contrast, so it runs on CONTROLS_SCORED: {n_pairs} pairs. The\n"
                f"base-versus-aligned contrast uses both strata and has {BOOKED_N['all_pairs']}. Neither n describes the other."),
            x="", y="", linetype="")
        + theme_minimal()
        + theme(figure_size=(12.6, 8.4),
                plot_title=element_text(size=11.0, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                strip_text=element_text(size=8.5, weight="bold"),
                legend_position="none",
                axis_text_x=element_text(size=7.6))
    )
    out = os.path.join(FIGURES, "l3_role_geometry.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    CONTROLS_SCORED: {n_pairs} pairs, {len(s):,} cells "
          f"(all strata: {BOOKED_N['all_pairs']} pairs)")
    for r in ROLES:
        tag = "" if r in DOC_AGREES else f"   <- document's table says t={DOC_T[r]}"
        print(f"    {r:<13} t {MEASURED[r][0]:.4f}   resid {MEASURED[r][1]:.4f}{tag}")
    print(f"    t spread {t_spread:.3f} against resid spread {r_spread:.3f}")
    return out


REGISTRY = {"role_geometry": role_geometry}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:20s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
