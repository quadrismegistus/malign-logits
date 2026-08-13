#!/usr/bin/env python
"""The displacement network as viz: DOT core + basin watershed panels.

    uv run python meta/M01_displacement/scripts/plot_displacement_network.py            # both
    uv run python meta/M01_displacement/scripts/plot_displacement_network.py core       # DOT export only
    uv run python meta/M01_displacement/scripts/plot_displacement_network.py procedure  # one basin panel

Source: pair_cascade_replicated.parquet (plan_pair_cascade.md), the
DISPLACEMENT-COUPLED subnetwork only — every edge split-half certified,
increment replicated in both halves. Two objects:

CORE (working map, RH's maps/ idiom): top-2 receivers per faller at
lift >= 3, fragments filtered — ~135 edges / ~137 words. Labelled edges
(shrunken full-data lift), basin sinks clustered. Browsable, annotatable;
promote pieces to figures as the paper wants them.

BASIN PANELS (paper figure candidates): per basin, its sinks as
terminals + top feeders by lift (2 hops). BASIN MEMBERSHIP IS CURATED —
an editorial layer, declared here, not a computed property: the sink
LISTS came from the purity>=0.9, in-degree>=5 computation (2026-08-14);
their grouping into procedure/epistemic/expression/stasis is a reading.
The figure caption must say both halves of that sentence.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

FIGURES = "meta/M01_displacement/figures"
PARQUET = "meta/M01_displacement/results/pair_cascade_replicated.parquet"
FRAG = {"re", "don", "ll", "ve", "t", "s", "d", "m", "didn", "wasn",
        "couldn"}

# curated basin grouping over COMPUTED sinks (purity>=.9, in>=5) — a
# reading, declared as such
BASINS = {
    "procedure": ["reach", "prepare", "adjust", "continue", "approach",
                  "examined", "followed", "replaced", "slid", "promptly",
                  "first"],
    "epistemic": ["understand", "believe", "realize", "question",
                  "regret"],
    "expression": ["shout", "scream", "gasped", "laughed", "shook",
                   "frowned"],
    "stasis": ["remain", "shut", "fled"],
}
BASIN_COLOR = {"procedure": "#d0e6f5", "epistemic": "#e8dff5",
               "expression": "#fde2cf", "stasis": "#e2e2e2"}


def load():
    d = pd.read_parquet(PARQUET)
    d = d[~d.F.isin(FRAG) & ~d.R.isin(FRAG)]
    return d[d.taxonomy == "displacement-coupled"].copy()


def render(dot_path):
    for fmt in ("svg", "png"):
        out = dot_path.rsplit(".", 1)[0] + "." + fmt
        subprocess.run(["dot", "-T" + fmt, dot_path, "-o", out],
                       check=True)
    print(f"rendered {dot_path} -> .svg/.png")


def core():
    dc = load()
    top2 = (dc.sort_values("lift_full", ascending=False)
              .groupby("F").head(2))
    top2 = top2[top2.lift_full >= 3]
    sink_of = {w: b for b, ws in BASINS.items() for w in ws}
    lines = [
        "// Displacement network core — top-2 receivers per faller,",
        "// lift >= 3, displacement-coupled edges only (all split-half",
        "// certified; plan_pair_cascade.md). Edge label = shrunken",
        "// full-data presence lift. Basin clusters are curated over",
        "// computed sinks — a reading, not a property.",
        "digraph displacement {",
        '  rankdir=LR; node [shape=box, style="rounded,filled",',
        '    fillcolor=white, fontname="Helvetica", fontsize=11];',
        '  edge [fontsize=8, color="#777777", fontcolor="#555555"];',
    ]
    for b, ws in BASINS.items():
        present = [w for w in ws
                   if w in set(top2.R) or w in set(top2.F)]
        if not present:
            continue
        lines.append(f'  subgraph cluster_{b} {{ label="{b}"; '
                     f'style=filled; color="{BASIN_COLOR[b]}";')
        for w in present:
            lines.append(f'    "{w}";')
        lines.append("  }")
    for t in top2.itertuples():
        lines.append(f'  "{t.F}" -> "{t.R}" '
                     f'[label="{t.lift_full:.0f}x"];')
    lines.append("}")
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "displacement_network_core.dot")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(top2)} edges, "
          f"{len(set(top2.F) | set(top2.R))} words")
    render(path)


def basin_panel(basin):
    dc = load()
    sinks = BASINS[basin]
    d1 = (dc[dc.R.isin(sinks)]
          .sort_values("lift_full", ascending=False)
          .groupby("R").head(4))
    feeders = set(d1.F)
    d2 = (dc[dc.R.isin(feeders) & (dc.lift_full >= 2.5)]
          .sort_values("lift_full", ascending=False)
          .groupby("R").head(2))
    E = pd.concat([d1, d2]).drop_duplicates(["F", "R"])
    lines = [
        f"// {basin} basin watershed — sinks (computed: purity>=0.9,",
        "// in>=5) + top-6 feeders each + top-2 second-hop feeders.",
        "// All edges displacement-coupled and split-half certified.",
        "digraph basin {",
        '  rankdir=LR; node [shape=box, style="rounded,filled",',
        '    fillcolor=white, fontname="Helvetica", fontsize=12];',
        '  edge [fontsize=8, color="#777777", fontcolor="#555555"];',
    ]
    for w in sinks:
        if w in set(E.R):
            lines.append(f'  "{w}" [fillcolor="{BASIN_COLOR[basin]}", '
                         f'penwidth=2, fontsize=14];')
    for t in E.itertuples():
        lines.append(f'  "{t.F}" -> "{t.R}" '
                     f'[label="{t.lift_full:.0f}x"];')
    lines.append("}")
    path = os.path.join(FIGURES, f"displacement_basin_{basin}.dot")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(E)} edges")
    render(path)


if __name__ == "__main__":
    args = sys.argv[1:] or ["core", "procedure"]
    for a in args:
        if a == "core":
            core()
        elif a in BASINS:
            basin_panel(a)
        else:
            sys.exit(f"unknown target {a!r}; use core|" +
                     "|".join(BASINS))
