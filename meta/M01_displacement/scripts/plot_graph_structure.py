#!/usr/bin/env python
"""The graph-structure pass: components, depth, and the blob that hides depth.

    uv run python meta/M01_displacement/scripts/plot_graph_structure.py
    uv run python meta/M01_displacement/scripts/plot_graph_structure.py depth
    uv run python meta/M01_displacement/scripts/plot_graph_structure.py --list

plot-debt 13c(e), which closes 13c. Add-beside, as `plot_chain_exhibit.py`
and `plot_couples_table.py` are: plotnine over the same two artifacts, with
`FRAG` imported from `plot_displacement_network` so no figure in this folder
can disagree with another about what counts as a word.

Writes `results/network_structure.json` as well as the figure. The numbers
below are a measurement and belong in an artifact a reader can query, not
only in pixels and a caption.

THE RESULT IS BACKWARDS AND THE MECHANISM IS THE FINDING
---------------------------------------------------------
The verb restriction removes more than half the edges, 1,818 down to 795.
The graph it leaves is DEEPER: longest path through the condensation goes
from **10 to 14**.

That is not a paradox and it is not noise. The full graph contains ONE
non-trivial strongly connected component with **141 members** -- 21% of
the vocabulary, mutually reachable, every word in it able to reach every
other by some chain of displacements. Condensation collapses it to a
single point, and a path through that point pays one step to cross 141
words. The depth of the full graph is measured across a blob that has
swallowed the middle of the network.

**42 of the 141 are words the verb restriction removes**: `not`, `just`,
`instead`, `no`, `all`, `if`, `in`, `on`, `once`, `he`, `is`, `could`,
`how`, `later`, `long`, `before`, `after` and the like. Take them out and
the blob does not shrink, it SHATTERS: the 99 survivors sit in components
of 16, 7, 2, 2 and 2. The chain that was one step becomes many, and the
graph is deeper because it is now legible rather than because anything
was added.

So the honest statement about depth is that **the full graph's depth of 10
is not comparable to the verb graph's 14**, and the reason is not sampling
or power. One of them is measured across a function-word short circuit.

WHY THE FIGURE SPLITS EACH BAR RATHER THAN JUST PLOTTING DEPTH
---------------------------------------------------------------
A depth profile alone would show two curves of different length and invite
exactly the comparison the paragraph above forbids. Each level's bar is
therefore split by whether a word sits in a multi-word component or alone,
so the 141-word block at level 6 of the full panel is visible AS the thing
that makes the two depths incommensurable. The blob is not an annotation
on the finding; it is the finding, and it has to be in the geometry.

WHAT IS STABLE ACROSS BOTH POPULATIONS
---------------------------------------
The terminus. Both longest paths end `... -> cry -> understand -> need`,
and under the verb restriction the run into it is `scream, see -> cry ->
understand -> need`. The deep end of the network is the same in both
populations; it is the middle that the function words fuse.
"""
import argparse
import json
import os
import sys

import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")
FIGURES = os.path.join(M01, "figures")

sys.path.insert(0, HERE)
from plot_displacement_network import FRAG  # noqa: E402

POPS = {
    "full": "pair_cascade_replicated.parquet",
    "verbs": "pair_cascade_replicated_verbs.parquet",
}

#: Measured on the committed artifacts. Every one of these is asserted
#: before drawing: if the graph moves, the panel's prose is wrong.
BOOKED = {
    "full": {"nodes": 670, "edges": 1818, "wcc": 9, "giant": 654,
             "scc": 530, "nontrivial": [141], "depth": 10,
             "sinks": 183, "sources": 225},
    "verbs": {"nodes": 419, "edges": 795, "wcc": 12, "giant": 394,
              "scc": 395, "nontrivial": [16, 7, 2, 2, 2], "depth": 14,
              "sinks": 133, "sources": 167},
    "blob_dropped_by_restriction": 42,
    "terminus": ["cry", "understand", "need"],
}


def _graph(pop):
    d = pd.read_parquet(os.path.join(RESULTS, POPS[pop]))
    e = d[d.displacement_coupled & d.replicated]
    e = e[~e.F.isin(FRAG) & ~e.R.isin(FRAG)]
    g = nx.DiGraph()
    g.add_edges_from((r.F, r.R) for r in e.itertuples())
    return g


def _structure(pop, g):
    """Components, condensation levels and depth for one population."""
    b = BOOKED[pop]
    wcc = sorted((len(c) for c in nx.weakly_connected_components(g)), reverse=True)
    sccs = list(nx.strongly_connected_components(g))
    nontrivial = sorted((len(c) for c in sccs if len(c) > 1), reverse=True)
    c = nx.condensation(g)
    assert nx.is_directed_acyclic_graph(c), "condensation is not a DAG"

    got = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
           "wcc": len(wcc), "giant": wcc[0], "scc": len(sccs),
           "nontrivial": nontrivial,
           "depth": nx.dag_longest_path_length(c),
           "sinks": sum(1 for n in g if g.out_degree(n) == 0),
           "sources": sum(1 for n in g if g.in_degree(n) == 0)}
    for k, v in b.items():
        assert got[k] == v, f"{pop} {k} drifted: {got[k]} vs booked {v}"

    #: level = longest path from any source, which is the depth coordinate
    #: `dag_longest_path_length` reports. Computed over the CONDENSATION, so
    #: a multi-word component occupies one level however many words it holds.
    lvl = {}
    for n in nx.topological_sort(c):
        lvl[n] = max((lvl[p] + 1 for p in c.predecessors(n)), default=0)

    rows = []
    for n, l in lvl.items():
        size = len(c.nodes[n]["members"])
        rows.append({"level": l, "words": size,
                     "grouping": ("in a multi-word component"
                                  if size > 1 else "alone at its level")})
    prof = (pd.DataFrame(rows).groupby(["level", "grouping"], as_index=False)
            .words.sum())
    prof["pop"] = pop
    got["longest_path"] = [sorted(c.nodes[n]["members"])
                           for n in nx.dag_longest_path(c)]
    return got, prof


def depth():
    """13c(e): depth profile, split by component, both populations."""
    from plotnine import (aes, element_blank, element_text, facet_grid,
                          geom_col, geom_text, ggplot, labs,
                          scale_fill_manual, scale_x_continuous, theme,
                          theme_minimal)

    gf, gv = _graph("full"), _graph("verbs")
    sf, pf = _structure("full", gf)
    sv, pv = _structure("verbs", gv)

    #: the mechanism, asserted rather than asserted in prose: the blob is
    #: one component, and the restriction removes 42 of its members
    blob = max(nx.strongly_connected_components(gf), key=len)
    dropped = blob - set(gv.nodes)
    assert len(dropped) == BOOKED["blob_dropped_by_restriction"], \
        (f"words the restriction removes from the 141-blob drifted: "
         f"{len(dropped)} vs booked {BOOKED['blob_dropped_by_restriction']}")
    surviving = sorted((len(c) for c in nx.strongly_connected_components(gv)
                        if len(c) > 1), reverse=True)
    assert surviving == BOOKED["verbs"]["nontrivial"], \
        f"the shattered blob drifted: {surviving}"
    for s in (sf, sv):
        assert [w for grp in s["longest_path"][-3:] for w in grp] == \
            BOOKED["terminus"], \
            f"the shared terminus changed: {s['longest_path'][-3:]}"

    art = os.path.join(RESULTS, "network_structure.json")
    with open(art, "w") as fh:
        json.dump({"_about": (
            "Graph structure of the displacement network, both populations. "
            "Nodes are words, edges are displacement-coupled split-half-"
            "certified pairs after the FRAG fragment filter. `depth` is the "
            "longest path through the CONDENSATION, so a strongly connected "
            "component of any size costs one step. THE TWO DEPTHS ARE NOT "
            "COMPARABLE: the full graph contains one component of 141 words, "
            "42 of them removed by the verb restriction, and a path crossing "
            "it pays one step for 21% of the vocabulary. "
            "Producer: meta/M01_displacement/scripts/plot_graph_structure.py"),
            "full": sf, "verbs": sv,
            "blob_members_dropped_by_verb_restriction": sorted(dropped)},
            fh, indent=2)

    d = pd.concat([pf, pv], ignore_index=True)
    order = ["full population (1,818 edges): depth 10",
             "verbs only (795 edges): depth 14"]
    d["panel"] = d["pop"].map({"full": order[0], "verbs": order[1]})
    d["panel"] = pd.Categorical(d.panel, categories=order, ordered=True)

    note = pd.DataFrame([
        {"panel": order[0], "level": 6, "words": 142,
         "lab": "ONE COMPONENT OF 141 WORDS, 21% of the vocabulary,\n"
                "crossed in a single step. 42 of its members are words\n"
                "the verb restriction removes."},
        {"panel": order[1], "level": 6, "words": 18,
         "lab": "the same blob, shattered: 16 and 7 and three 2s"},
    ])
    note["panel"] = pd.Categorical(note.panel, categories=order, ordered=True)

    p = (
        ggplot(d, aes("level", "words", fill="grouping"))
        + geom_col(width=0.78)
        + geom_text(note, aes("level", "words", label="lab"), inherit_aes=False,
                    size=6.2, ha="left", va="bottom", nudge_x=0.45,
                    nudge_y=4, color="#b03030", lineheight=1.25)
        + scale_fill_manual(values={"alone at its level": "#9fb8cc",
                                    "in a multi-word component": "#b03030"},
                            name="")
        + scale_x_continuous(breaks=range(0, 15))
        #: SHARED AXES, NOT free. The whole point is that one panel runs
        #: four levels further than the other, and a free x would put both
        #: depths at the same width and delete the comparison.
        + facet_grid("panel ~ .")
        + labs(
            title="Removing more than half the edges makes the network DEEPER, because a function-word blob was hiding the depth",
            subtitle=(
                "Words per level of the condensation DAG. A level is one step of the longest path, and a\n"
                "strongly connected component costs ONE step however many words it holds.\n"
                "FULL: 670 words, 1,818 edges, 9 weakly connected components with 654 in the giant one.\n"
                "530 components, of which exactly ONE is non-trivial and it holds 141 words at level 6.\n"
                "VERBS: 419 words, 795 edges, and that single blob is gone: 16, 7, 2, 2, 2 instead.\n"
                "42 of the 141 are words the restriction removes -- `not`, `just`, `instead`, `if`, `in`,\n"
                "`on`, `no`, `all`, `he`, `is`, `could`, `how`, `later`, `long`, `before`, `after`.\n"
                "SO THE TWO DEPTHS ARE NOT COMPARABLE, and the reason is not power or sampling: the full\n"
                "graph's depth of 10 is measured across a short circuit that fuses a fifth of the vocabulary.\n"
                "WHAT IS STABLE IS THE TERMINUS. Both longest paths end `cry -> understand -> need`, and\n"
                "under the restriction the run into it is `scream, see -> cry -> understand -> need`.\n"
                "The deep end of the network is the same in both; it is the middle the function words fuse."),
            x="level of the condensation DAG (one step of the longest path)",
            y="words at that level",
            #: THE TRUNCATION TRAP APPLIES TO THE CAPTION TOO, and it caught
            #: this figure there after the subtitle had been fixed for it.
            #: An assert list is exactly the long unbroken line most at risk,
            #: and losing its tail deletes the part naming what was checked.
            #: Wrap every element that carries prose, not just the subtitle.
            caption=("Producer: meta/M01_displacement/scripts/plot_graph_structure.py from\n"
                     "results/pair_cascade_replicated{,_verbs}.parquet, also writing\n"
                     "results/network_structure.json; FRAG imported from plot_displacement_network.\n"
                     "Asserted before drawing: both populations at 670/1,818 and 419/795; 9 and 12 weak\n"
                     "components with giants of 654 and 394; 530 and 395 strong components; non-trivial\n"
                     "sizes [141] and [16,7,2,2,2]; depths 10 and 14; 183/133 pure sinks and 225/167 pure\n"
                     "sources; that 42 blob members are dropped by the restriction; and that both\n"
                     "longest paths end cry -> understand -> need."),
        )
        + theme_minimal()
        + theme(figure_size=(12.4, 8.2),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                strip_text=element_text(size=8.2, weight="bold"),
                legend_title=element_blank(),
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
                legend_position="bottom")
    )
    out = os.path.join(FIGURES, "displacement_graph_structure.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"  wrote {art}")
    print(f"    full: depth {sf['depth']}, one component of {sf['nontrivial'][0]}; "
          f"verbs: depth {sv['depth']}, components {sv['nontrivial']}; "
          f"{len(dropped)} blob members dropped by the restriction")
    return out


REGISTRY = {"depth": depth}


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
