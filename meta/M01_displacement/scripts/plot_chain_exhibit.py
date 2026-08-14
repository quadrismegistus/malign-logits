#!/usr/bin/env python
"""The chain exhibit: two-hop displacement chains, and two that fail.

    uv run python meta/M01_displacement/scripts/plot_chain_exhibit.py
    uv run python meta/M01_displacement/scripts/plot_chain_exhibit.py strip
    uv run python meta/M01_displacement/scripts/plot_chain_exhibit.py --list

plot-debt 13c(b). Add-beside rather than add-inside: the basin panels are
graphviz and this is plotnine, so it is a separate producer reading the
same two artifacts. The `FRAG` fragment list is imported from
`plot_displacement_network` rather than restated, so the two figures
cannot disagree about what a word is.

THE ENTRY NAMES TWO EXHIBITS AND BOTH OF THEM FAIL
--------------------------------------------------
plot-debt 13c(b) says, of the chain strip: *NB fired->aimed->pointed is
FRAME taxonomy (certified, co-rising), not displacement-coupled, and
kill->shout->hum dies at shout->hum under the verb restriction: quote
accordingly.*

Verified here before drawing, and both hold exactly:

    fired->aimed     taxonomy=frame, displacement_coupled=False, lift 2.05
    aimed->pointed   taxonomy=frame, displacement_coupled=False, lift 3.40
    kill->shout      coupled, survives the verb restriction (6.30 -> 2.42)
    shout->hum       coupled at 4.58 in the full set, ABSENT under verbs

So the two chains a reader is most likely to have in mind are the two
that do not qualify, for two DIFFERENT reasons. That is the figure. A
strip of surviving chains alone would answer a question nobody asked;
the exhibit is which chains hold and what the two failure modes look
like, drawn together.

FRAME AND DISPLACEMENT ARE BOTH REAL AND ARE NOT THE SAME CLAIM
---------------------------------------------------------------
`fired->aimed->pointed` is not noise. Both links are REPLICATED and
split-half certified; the taxonomy is `frame`, meaning the two words
co-rise because they belong to one frame, not because one displaces
onto the other. Drawing it as a displacement chain would convert a
certified result into a different certified result's clothing, which is
worse than drawing something unsupported.

SELECTION IS EDITORIAL AND IS DECLARED, AS THE BASIN GROUPING IS
-----------------------------------------------------------------
1,433 two-hop chains survive both links under the verb restriction after
fragment filtering. Six are drawn. The rule is stated on the panel: the
`see -> scream` convergence because four distinct feeders route through
one intermediate, then the next highest by weakest-link lift excluding
chains whose intermediate or terminal is an auxiliary. **Which six is a
reading. That 1,433 exist is a measurement.** Both halves on the panel,
the same treatment the basin panels now carry.
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

#: auxiliaries and light verbs: excluded as chain INTERMEDIATES and
#: TERMINALS only. They are real edges and stay in the counts; a chain
#: whose middle is `had` exhibits the auxiliary, not a displacement.
LIGHT = {"had", "have", "has", "did", "do", "does", "went", "go", "goes",
         "be", "been", "was", "were", "is", "are", "get", "got", "make",
         "made", "take", "took", "say", "said", "tell", "told", "try",
         "see", "come", "came", "put", "let", "know", "knew"}

#: Booked from the plot-debt entry's own cautions, asserted before drawing.
BOOKED_FAIL = {
    ("fired", "aimed"): ("frame", False),
    ("aimed", "pointed"): ("frame", False),
    ("shout", "hum"): None,      #: absent under the verb restriction
}


#: Populations after the FRAG filter, booked here so a change to the SHARED
#: `FRAG` definition cannot silently move this figure's basis. This producer
#: imports FRAG from `plot_displacement_network` so the folder's figures
#: cannot disagree about what a word is -- which is the right coupling and
#: is also a SIBLING dependency: a file in this folder that this one names
#: and whose edits are invisible in this file's own git history. The other
#: two dependents (`plot_couples_table`, `plot_graph_structure`) already
#: assert their populations; this one guarded only with `n_chains > 1000`,
#: a threshold rather than a booked value, so a FRAG change could have moved
#: the population without tripping anything. Measured: FRAG has never
#: changed. This asserts that it stays that way.
BOOKED_POP = {"pair_cascade_replicated.parquet": (1818, 670),
              "pair_cascade_replicated_verbs.parquet": (795, 419)}


def _edges(path):
    d = pd.read_parquet(os.path.join(RESULTS, path))
    e = d[d.displacement_coupled & d.replicated][["F", "R", "lift_full"]]
    e = e[~e.F.isin(FRAG) & ~e.R.isin(FRAG)]
    got = (len(e), len(set(e.F) | set(e.R)))
    assert got == BOOKED_POP[path], \
        (f"{path} population drifted: {got} vs booked {BOOKED_POP[path]} -- "
         "check whether FRAG moved in plot_displacement_network")
    return d, e


def strip():
    """Six surviving two-hop chains, beside the two the entry says fail."""
    from plotnine import (aes, element_blank, element_text, geom_segment,
                          geom_text, ggplot, labs, scale_color_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    full_all, full = _edges("pair_cascade_replicated.parquet")
    verb_all, verb = _edges("pair_cascade_replicated_verbs.parquet")

    #: the entry's cautions, verified rather than trusted
    for (f, r), expect in BOOKED_FAIL.items():
        row = full_all[(full_all.F == f) & (full_all.R == r)]
        if expect is None:
            assert len(row) == 1, f"{f}->{r} missing from the full set"
            assert len(verb_all[(verb_all.F == f) & (verb_all.R == r)]) == 0, \
                f"{f}->{r} is no longer absent under the verb restriction"
        else:
            tax, coup = expect
            assert row.iloc[0].taxonomy == tax and bool(row.iloc[0].displacement_coupled) == coup, \
                (f"{f}->{r} changed: taxonomy={row.iloc[0].taxonomy} "
                 f"coupled={bool(row.iloc[0].displacement_coupled)} vs booked {expect}")

    ch = verb.merge(verb, left_on="R", right_on="F", suffixes=("1", "2"))
    ch = ch[ch.F1 != ch.R2].copy()
    ch["weak"] = ch[["lift_full1", "lift_full2"]].min(axis=1)
    n_chains = len(ch)
    assert n_chains > 1000, f"chain population collapsed: {n_chains}"

    conv = ch[(ch.R1 == "see") & (ch.R2 == "scream")].nlargest(4, "weak")
    rest = ch[~ch.R1.isin(LIGHT) & ~ch.R2.isin(LIGHT) & ~ch.F1.isin(LIGHT)]
    rest = rest.nlargest(2, "weak")
    pick = pd.concat([conv, rest]).reset_index(drop=True)

    rows = []
    for i, r in pick.iterrows():
        rows.append({"y": i, "kind": "holds", "col": "#1f4e79",
                     "w1": r.F1, "w2": r.R1, "w3": r.R2,
                     "l1": f"{r.lift_full1:.1f}x", "l2": f"{r.lift_full2:.1f}x",
                     "note": ""})
    base = len(rows) + 0.8
    rows.append({"y": base, "kind": "fails", "col": "#b03030",
                 "w1": "fired", "w2": "aimed", "w3": "pointed",
                 "l1": "2.1x", "l2": "3.4x",
                 "note": "both links are FRAME taxonomy, not displacement-coupled:\n"
                         "certified and co-rising, which is a different claim"})
    rows.append({"y": base + 1.25, "kind": "fails", "col": "#b03030",
                 "w1": "kill", "w2": "shout", "w3": "hum",
                 "l1": "2.4x", "l2": "absent",
                 "note": "second link is ABSENT under the verb restriction:\n"
                         "the chain dies at shout->hum"})
    d = pd.DataFrame(rows)
    d["y"] = d.y.max() - d.y   #: top-to-bottom

    seg = pd.concat([
        d.assign(x=0.30, xend=0.92, mid=0.61, lab=d.l1),
        d.assign(x=1.30, xend=1.92, mid=1.61, lab=d.l2),
    ])
    seg.loc[seg.lab == "absent", "col"] = "#c9c9c9"

    p = (
        ggplot()
        + geom_segment(seg, aes("x", "y", xend="xend", yend="y", color="col"),
                       size=0.7, arrow=None, alpha=0.85)
        + geom_text(seg, aes("mid", "y", label="lab", color="col"), size=6.2,
                    va="bottom", nudge_y=0.10)
        + geom_text(d, aes(0.15, "y", label="w1", color="col"), size=8.5,
                    ha="right")
        + geom_text(d, aes(1.11, "y", label="w2", color="col"), size=8.5,
                    ha="center")
        + geom_text(d, aes(2.07, "y", label="w3", color="col"), size=8.5,
                    ha="left")
        + geom_text(d[d.note != ""], aes(2.55, "y", label="note"), size=6.0,
                    ha="left", color="#b03030", lineheight=1.2)
        + scale_color_identity()
        + scale_x_continuous(limits=(-0.55, 5.4))
        + scale_y_continuous(limits=(-0.7, d.y.max() + 0.7))
        + labs(
            title="Two-hop displacement chains, and the two that do not qualify",
            subtitle=(
                f"Blue: chains where BOTH links are displacement-coupled, split-half certified, and "
                f"survive the verb restriction. {n_chains:,} such chains exist after fragment "
                "filtering; six are drawn.\n"
                "Red: the two exhibits plot-debt names, and they fail for DIFFERENT reasons. "
                "`fired -> aimed -> pointed` is replicated and certified but its taxonomy is FRAME: "
                "the words co-rise because they share a frame, which is a different claim from one "
                "displacing onto the other.\n"
                "`kill -> shout -> hum` holds at its first link and its second is absent once the "
                "population is restricted to verbs, so the chain dies mid-way.\n"
                "WHICH SIX IS A READING; THAT 1,433 EXIST IS A MEASUREMENT.\n"
                "The convergence into `scream` is drawn because four distinct feeders route through "
                "one intermediate; the other two are the highest remaining by weakest link, "
                "excluding chains whose middle or end is an auxiliary.\n"
                "Labels are shrunken full-data lift on the verb-restricted population, where lifts "
                "compress to roughly half their unrestricted values."),
            x="", y="",
            caption=("Producer: meta/M01_displacement/scripts/plot_chain_exhibit.py from "
                     "results/pair_cascade_replicated{,_verbs}.parquet; FRAG imported from "
                     "plot_displacement_network so the two figures cannot disagree about a word.\n"
                     "Asserted before drawing: fired->aimed and aimed->pointed are taxonomy=frame and "
                     "not displacement-coupled; shout->hum is present in the full set and absent under "
                     "the verb restriction."),
        )
        + theme_minimal()
        + theme(figure_size=(12.2, 5.6),
                plot_title=element_text(size=13, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.1, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text=element_blank(), axis_ticks_major=element_blank(),
                panel_grid=element_blank())
    )
    out = os.path.join(FIGURES, "displacement_chain_exhibit.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {n_chains:,} surviving two-hop chains; drew {len(pick)} + 2 failures")
    return out


REGISTRY = {"strip": strip}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:8s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
