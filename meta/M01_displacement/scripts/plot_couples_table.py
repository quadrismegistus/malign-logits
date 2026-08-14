#!/usr/bin/env python
"""The mutual-best couples table: displacement is almost never reciprocal.

    uv run python meta/M01_displacement/scripts/plot_couples_table.py
    uv run python meta/M01_displacement/scripts/plot_couples_table.py couples
    uv run python meta/M01_displacement/scripts/plot_couples_table.py --list

plot-debt 13c(d). Add-beside, as `plot_chain_exhibit.py` is: the network
panels are graphviz and this is plotnine, so it is a separate producer
reading the same two artifacts. `FRAG` is imported from
`plot_displacement_network` rather than restated, so no two figures in
this folder can disagree about what counts as a word.

THE ENTRY ASKS FOR A TABLE THAT HAS ONE ROW IN IT
-------------------------------------------------
plot-debt 13c(d) says "mutual-best couples table" and books no numbers,
so the definition is mine and is declared here rather than assumed.

The strict reading is: A's highest-lift receiver is B, and B's highest-
lift receiver is A. Measured on the committed artifact, **that set has
exactly one member in each population** -- `frowned <-> saw` in the full
set, `scream <-> see` under the verb restriction. A one-row table is not
a table, and drawing it alone would present a singleton as though it
were a class.

So the figure draws the RECIPROCAL couples, the pairs where both
directions are displacement-coupled and split-half certified, and marks
which of them is mutual-best. 16 couples in the full population, 11
under the verb restriction. The strict count of one is on the panel,
because it is the finding rather than a shortfall.

RECIPROCITY IS THE EXCEPTION AND THE RATE IS THE POINT
------------------------------------------------------
32 of 1,818 edges participate in a reciprocal couple: **1.8%**. Of 487
words that fall at all, one has a partner that names it back as its own
strongest receiver. Displacement is overwhelmingly a one-way relation,
and the couples table is the measurement that establishes it rather
than a gallery of couples.

THE SEGMENT LENGTH IS THE ASYMMETRY, WHICH IS WHY BOTH LIFTS ARE DRAWN
----------------------------------------------------------------------
A couple is not thereby balanced. `see -> scream` is 10.27x while
`scream -> see` is 2.77x, a ratio of 0.27: the same two words, one
strong direction and a weak return. Drawing a couple as a single mark
would collapse exactly that. Each row is therefore two points on one
lift axis with a segment between them, ordered by the symmetry ratio,
so the panel reads from balanced at the top to lopsided at the bottom
and the segment length is the quantity.

FOUR COUPLE MEMBERS ARE FUNCTION WORDS THAT `FRAG` DOES NOT CATCH
-----------------------------------------------------------------
`the`, `not`, `just`, `so` and `instead` appear in the full population's
couples. `FRAG` filters fragments, not function words, and these are
whole words, so they pass it correctly. They are marked on the panel as
what they are, and the verb restriction removes all of them: this is the
clearest single case in the folder for why the verb population exists,
so both populations are drawn together rather than in two figures.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
FIGURES = os.path.join(M01, "figures")

sys.path.insert(0, HERE)
from plot_displacement_network import FRAG  # noqa: E402

#: Populations, both asserted against plot-debt 13c's booked figures.
POPS = {
    "full": ("pair_cascade_replicated.parquet", (1818, 670)),
    "verbs": ("pair_cascade_replicated_verbs.parquet", (795, 419)),
}

#: Measured on the committed artifacts and asserted before drawing. If any
#: of these move, the panel's prose is wrong and it refuses to draw.
BOOKED = {
    "recip_full": 16, "recip_verbs": 11,
    "mutual_best_full": ("frowned", "saw"),
    "mutual_best_verbs": ("scream", "see"),
    "fallers_full": 487,
}

#: Whole function words, so `FRAG` passes them correctly. Named, not filtered.
FUNCTION = {"the", "not", "just", "so", "instead"}


def _edges(pop):
    """Coupled, replicated, defragged edges for one population."""
    path, booked = POPS[pop]
    d = pd.read_parquet(os.path.join(RESULTS, path))
    e = d[d.displacement_coupled & d.replicated]
    e = e[~e.F.isin(FRAG) & ~e.R.isin(FRAG)].copy()
    got = (len(e), len(set(e.F) | set(e.R)))
    assert got == booked, f"{pop} population drifted: {got} vs booked {booked}"
    return e


def _couples(e):
    """Reciprocal couples, with both lifts and each direction's top-ness."""
    lift = {(r.F, r.R): r.lift_full for r in e.itertuples()}
    best = e.loc[e.groupby("F").lift_full.idxmax()].set_index("F").R.to_dict()
    out = []
    for a, b in sorted({tuple(sorted((f, r))) for f, r in lift if (r, f) in lift}):
        ab, ba = lift[(a, b)], lift[(b, a)]
        out.append({"a": a, "b": b, "ab": ab, "ba": ba,
                    "sym": min(ab, ba) / max(ab, ba),
                    "a_top": best.get(a) == b, "b_top": best.get(b) == a})
    return pd.DataFrame(out)


def couples():
    """13c(d): reciprocal couples, and the single mutual-best one."""
    from plotnine import (aes, element_blank, element_text, geom_point,
                          geom_segment, geom_text, ggplot, labs,
                          scale_color_manual, scale_x_log10,
                          scale_y_continuous, theme, theme_minimal)

    full, verb = _edges("full"), _edges("verbs")
    c, cv = _couples(full), _couples(verb)

    assert len(c) == BOOKED["recip_full"], \
        f"reciprocal couples drifted: {len(c)} vs booked {BOOKED['recip_full']}"
    assert len(cv) == BOOKED["recip_verbs"], \
        f"verb couples drifted: {len(cv)} vs booked {BOOKED['recip_verbs']}"
    assert full.F.nunique() == BOOKED["fallers_full"], \
        f"fallers drifted: {full.F.nunique()} vs {BOOKED['fallers_full']}"
    for pop, tbl in (("full", c), ("verbs", cv)):
        mb = tbl[tbl.a_top & tbl.b_top]
        assert len(mb) == 1, \
            (f"{pop}: mutual-best is no longer a singleton ({len(mb)}); the "
             "panel's central claim is that it is one")
        assert (mb.iloc[0].a, mb.iloc[0].b) == BOOKED[f"mutual_best_{pop}"], \
            (f"{pop} mutual-best changed: {(mb.iloc[0].a, mb.iloc[0].b)} vs "
             f"booked {BOOKED[f'mutual_best_{pop}']}")

    surviving = {(r.a, r.b) for r in cv.itertuples()}
    c = c.sort_values("sym", ascending=False).reset_index(drop=True)
    #: NUMERIC y with labelled breaks. A discrete scale here raises
    #: "Discrete value supplied to continuous scale" as soon as anything
    #: is placed in data space beside it, which the annotations are.
    c["ypos"] = range(len(c) - 1, -1, -1)
    c["survives"] = [(r.a, r.b) in surviving for r in c.itertuples()]
    c["is_fn"] = [bool({r.a, r.b} & FUNCTION) for r in c.itertuples()]
    c["mb"] = c.a_top & c.b_top

    #: The panel is drawn on the FULL population, so `mb` is full-population
    #: mutual-best. The verb population's mutual-best is a DIFFERENT couple
    #: and it is present here unmarked, which would read as an omission
    #: against a subtitle that names it. Marked for the population it holds in.
    mb_verbs = BOOKED["mutual_best_verbs"]

    def _lab(r):
        s = f"{r.a} <-> {r.b}"
        if r.mb:
            s += "   MUTUAL-BEST"
        if (r.a, r.b) == mb_verbs:
            s += "   MUTUAL-BEST under the verb restriction"
        if r.is_fn:
            s += "   function word"
        return s

    c["lab"] = [_lab(r) for r in c.itertuples()]
    c["kind"] = [
        "drops out under the verb restriction" if not r.survives
        else "survives the verb restriction" for r in c.itertuples()]

    #: one row becomes two marks: the two directions of the same couple
    pts = pd.concat([
        c.assign(lift=c.ab, dirn=[f"{r.a} -> {r.b}" for r in c.itertuples()]),
        c.assign(lift=c.ba, dirn=[f"{r.b} -> {r.a}" for r in c.itertuples()]),
    ])
    c["lo"] = c[["ab", "ba"]].min(axis=1)
    c["hi"] = c[["ab", "ba"]].max(axis=1)
    #: the stronger direction is labelled; the weaker is the bare point
    strong = c.assign(
        s=[f"{r.a} -> {r.b}" if r.ab >= r.ba else f"{r.b} -> {r.a}"
           for r in c.itertuples()])

    colors = {"survives the verb restriction": "#1f4e79",
              "drops out under the verb restriction": "#b03030"}
    p = (
        ggplot()
        + geom_segment(c, aes("lo", "ypos", xend="hi", yend="ypos",
                              color="kind"), size=0.7, alpha=0.5)
        + geom_point(pts, aes("lift", "ypos", color="kind"), size=2.2)
        + geom_text(strong, aes("hi", "ypos", label="s", color="kind"),
                    size=5.6, ha="left", nudge_x=0.02, va="center")
        + geom_text(c, aes("lo", "ypos", label="sym"), size=5.4, ha="right",
                    va="center", color="#777777", nudge_x=-0.02,
                    format_string="{:.2f}")
        + scale_color_manual(values=colors, name="")
        + scale_x_log10(breaks=[1.5, 2, 3, 5, 10],
                        labels=["1.5x", "2x", "3x", "5x", "10x"],
                        limits=(1.3, 26))
        + scale_y_continuous(breaks=list(c.ypos), labels=list(c.lab),
                             limits=(-0.7, len(c) - 0.3))
        + labs(
            #: TWO rows carry a MUTUAL-BEST mark, one per population, so a title
            #: saying "one of them" would be contradicted by the panel under it.
            title="Displacement is almost never reciprocal: 16 couples in 1,818 edges, and one mutual-best couple per population",
            #: LINE LENGTH IS A CONSTRAINT, NOT A PREFERENCE. plotnine does not
            #: wrap a subtitle and does not widen the canvas for it: a line
            #: longer than the figure is silently CUT at the edge, mid-word, and
            #: the loss appears only in the rendered PNG. Two lines of this
            #: subtitle were truncated on the first render. Keep every line
            #: under ~130 characters at this figure width and read the image.
            subtitle=(
                "A COUPLE is a word pair where BOTH directions are displacement-coupled and split-half certified:\n"
                "16 in the full population drawn here, 11 under the verb restriction.\n"
                "32 of 1,818 edges participate in one, which is 1.8%. Of the 487 words that fall at all, exactly ONE\n"
                "has a partner naming it back as its own strongest receiver.\n"
                "MUTUAL-BEST is the strict reading of this entry: A's top receiver is B and B's top receiver is A.\n"
                "It has ONE member in each population, and not the same one. That it is a singleton is the finding.\n"
                "A COUPLE IS NOT A BALANCE. Each row is the same two words in both directions on one log axis,\n"
                "ordered by the symmetry ratio at the left, so the SEGMENT LENGTH IS THE ASYMMETRY:\n"
                "`see -> scream` is 10.27x against 2.77x the other way, one strong direction and a weak return.\n"
                "RED COUPLES DROP OUT UNDER THE VERB RESTRICTION, and five turn on `the`, `not`, `just`, `so`\n"
                "or `instead`. FRAG filters fragments, not function words, and these are whole words that pass it."),
            x="shrunken full-data lift, log scale (each couple appears twice, once per direction)",
            y="",
            caption=("Producer: meta/M01_displacement/scripts/plot_couples_table.py from "
                     "results/pair_cascade_replicated{,_verbs}.parquet; FRAG imported from "
                     "plot_displacement_network so no two figures here disagree about a word.\n"
                     "Asserted before drawing: both populations at 1,818/670 and 795/419, 16 and 11 "
                     "reciprocal couples, 487 fallers, and that mutual-best is a singleton in each "
                     "population naming frowned/saw and scream/see.\n"
                     "Lifts are NOT comparable across populations: the verb restriction compresses them to "
                     "roughly half. Every number drawn here is from the full population."),
        )
        + theme_minimal()
        + theme(figure_size=(12.4, 7.4),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_text(size=7.0),
                legend_title=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                legend_position="bottom")
    )
    out = os.path.join(FIGURES, "displacement_couples_table.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {len(c)} couples full / {len(cv)} verbs; "
          f"{2 * len(c)}/{len(full)} edges reciprocal "
          f"({200 * len(c) / len(full):.1f}%); "
          f"mutual-best {BOOKED['mutual_best_full']} and "
          f"{BOOKED['mutual_best_verbs']}")
    return out


REGISTRY = {"couples": couples}


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
