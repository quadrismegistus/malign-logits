#!/usr/bin/env python
"""Figures for `self_surprisal.md` (the S3/S4 forced-arm contrasts).

    uv run python meta/M06_generation/scripts/m06_self_surprisal_figs.py
    uv run python meta/M06_generation/scripts/m06_self_surprisal_figs.py diagonal
    uv run python meta/M06_generation/scripts/m06_self_surprisal_figs.py --list

Plotting regime (RH, 2026-08-14): plotnine at 300 dpi, output to
../figures/, slice in the subtitle, booked-number asserts before drawing.
Script naming follows M06's `m06_*` convention.

FIRST FIGURE PRODUCER IN M06. The folder's inventory on 2026-08-14 found
ten findings, all headline numbers undrawn, and zero figure-producing
scripts against M01's six, M05's fifteen and one each in M02/M03/M04.

WHAT THE FIGURE HAS TO RESIST
-----------------------------
The result is a 2x2 whose two significant cells fall on the DIAGONAL:
the base is soothed by the vocabulary it promoted (fallen words), the
aligned model by the vocabulary it promoted (risen words). That is a
genuinely pleasing shape and it is most of the risk.

The two halves are NOT equally established, and a clean 2x2 invites the
reader to take them as one symmetric fact:

    S4 (ROSE vs FLAT)   DiD pair -0.0150 p 0.0166   cell -0.0161 p 0.0013
    S3 (FELL vs FLAT)   DiD pair +0.0133 p 0.636    cell +0.0115 p 0.018

S4's difference-in-differences is non-null at BOTH grains. S3's is NULL
at the pair grain, which the finding calls the conservative unit, and
reaches significance only at the cell grain. So the mirror is one
established arm-specific effect and one half whose arm-specificity is
not established at the conservative unit. The panel says this on itself
rather than in a caption, because the caption is what gets dropped when
a figure travels.

The TITLE gets the same treatment for the same reason, and this file's
first version failed it. It opened "Each arm is soothed by the
vocabulary it promoted, and only one half of that is established",
which leads with the mirror and qualifies afterwards -- in the one line
most likely to be quoted without the panel. lacan amended the finding's
own title on the identical ground at 64dc3803 ([5917]); this title now
leads with the established half and names the open one.

Position is the per-pair median delta; sign counts ride as text. The
sign test is what the finding reports, so the counts are the evidence
and the median is the effect size, not the other way round.
"""
import argparse
import os
import sys
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
CELLS = os.path.join(RESULTS, "self_surprisal_cells.parquet")

CONTRASTS = {"S3": ("faller", "matched"), "S4": ("riser_matched", "matched")}
TITLES = {"S3": "S3   FELL vs FLAT\n(forced a fallen word)",
          "S4": "S4   ROSE vs FLAT\n(forced a risen word)"}

#: Booked in self_surprisal.md, "The result" table, PAIR grain.
BOOKED = {
    ("S3", "aligned"): (-0.0053, 15, 25), ("S3", "base"): (-0.0199, 8, 31),
    ("S4", "aligned"): (-0.0077, 13, 27), ("S4", "base"): (+0.0012, 21, 18),
    ("S3", "DiD"): (+0.0133, 22, 18), ("S4", "DiD"): (-0.0150, 12, 28),
}
#: which cells the finding reports as significant, i.e. the diagonal
SIG = {("S3", "base"), ("S4", "aligned")}


def _sign_test(ds):
    """The finding's own test, reimplemented from m06_self_surprisal.py.

    Zeros are KEPT for the median and excluded from the counts. Dropping
    them from the median instead shifts S3 base to -0.0208 against the
    booked -0.0199, which is how this was found: the medians disagreed
    while every sign count and p matched exactly.
    """
    ds = np.asarray(ds, float)
    up = int((ds > 0).sum())
    dn = int((ds < 0).sum())
    lo = min(up, dn)
    p = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
    return float(np.median(ds)), up, dn, p


def _per_pair():
    """Per-pair median deltas for each contrast x role, plus the DiD."""
    d = pd.read_parquet(CELLS)
    w = d.pivot_table(index=["pair", "role", "prompt"], columns="arm",
                      values="self_surprisal")
    rows, stats = [], {}
    for tag, (a, b) in CONTRASTS.items():
        legs = {}
        for role in ("aligned", "base"):
            sub = w.xs(role, level="role")
            s = (sub[a] - sub[b]).dropna()
            legs[role] = s
            pm = s.groupby(level="pair").median()
            stats[(tag, role)] = _sign_test(pm.values)
            for pair, v in pm.items():
                rows.append({"tag": tag, "role": role, "pair": pair, "delta": v})
        j = pd.concat({"a": legs["aligned"], "b": legs["base"]}, axis=1).dropna()
        stats[(tag, "DiD")] = _sign_test(
            (j.a - j.b).groupby(level="pair").median().values)
    return pd.DataFrame(rows), stats


def diagonal():
    """The S3/S4 diagonal: each arm soothed by the vocabulary it promoted.

    2x2 of arm by word class over the per-pair median deltas. The two
    significant cells fall on the diagonal; the panel marks which they
    are and states, on itself, that the two halves of the mirror do not
    have the same support at the conservative unit.
    """
    from plotnine import (aes, element_blank, element_text, facet_grid,
                          geom_jitter, geom_point, geom_text, geom_vline,
                          ggplot, labs, scale_color_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    d, stats = _per_pair()

    for key, (m, up, dn) in BOOKED.items():
        gm, gup, gdn, _ = stats[key]
        assert round(gm, 4) == m and (gup, gdn) == (up, dn), \
            f"{key} drifted: {round(gm, 4)} {gup}/{gdn} vs booked {m} {up}/{dn}"

    ann = []
    for (tag, role), (m, up, dn, p) in stats.items():
        if role == "DiD":
            continue
        sig = (tag, role) in SIG
        ann.append({
            "tag": tag, "role": role, "x": m, "sig": sig,
            "col": "#b03030" if sig else "#9a9a9a",
            "txt": (f"median {m:+.4f}\n{dn} of {up + dn} pairs down\n"
                    f"sign p {p:.3g}" + ("   SIGNIFICANT" if sig else "")),
        })
    a = pd.DataFrame(ann)
    d = d.merge(a[["tag", "role", "col", "sig"]], on=["tag", "role"])

    rl = {"base": "BASE arm", "aligned": "ALIGNED arm"}
    for f in (d, a):
        f["role"] = pd.Categorical(f.role.map(rl),
                                   categories=[rl["base"], rl["aligned"]],
                                   ordered=True)
        f["tag"] = pd.Categorical(f.tag.map(TITLES),
                                  categories=[TITLES["S3"], TITLES["S4"]],
                                  ordered=True)

    did = "   ".join(
        f"{t} DiD {stats[(t, 'DiD')][0]:+.4f} (pair p {stats[(t, 'DiD')][3]:.3g})"
        for t in ("S3", "S4"))

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.4)
        + geom_jitter(d, aes("delta", 0, color="col"), height=0.22, width=0,
                      size=1.5, alpha=0.55)
        + geom_point(a, aes("x", 0, color="col"), size=4.5, shape="D")
        + geom_text(a, aes(0, 0.42, label="txt", color="col"), size=6.8,
                    ha="center", va="bottom", lineheight=1.25)
        + scale_color_identity()
        + facet_grid("role ~ tag")
        #: full data range, no truncation: the two extreme pairs
        #: (recurrentgemma-9b +0.197, Llama-3.1-8B -0.137, both S3 aligned)
        #: are inside the panel rather than cut and declared. With 158
        #: points the cost of showing them is a little compression; the
        #: cost of cutting them is a reader who cannot see the spread
        #: that makes S3 aligned non-significant.
        + scale_x_continuous(limits=(-0.21, 0.21),
                             breaks=[-0.2, -0.1, 0, 0.1, 0.2])
        #: explicit y range. Without it the scale auto-fits the jitter's
        #: +/-0.22 and the annotation at 0.78 is silently CLIPPED to its
        #: last line, so each panel showed only "sign p ..." and lost the
        #: median and the sign counts. A clipped annotation looks like a
        #: design choice, which is why this is pinned rather than tuned.
        + scale_y_continuous(limits=(-0.42, 1.32))
        + labs(
            title="The ALIGNED model is soothed by the vocabulary it promoted. The mirror half is not established.",
            subtitle=(
                "Self-surprisal on the model's OWN continuation (A|A) when a word is forced into the prompt, "
                "against the flat non-mover control.\n"
                "NEGATIVE = LESS surprised by its own continuation. One point per model pair (n = 39-40), "
                "each the median over that pair's prompts; diamond is the median of those.\n"
                "The two SIGNIFICANT cells fall on the DIAGONAL: the base is soothed by fallen words, the "
                "aligned model by risen words.\n"
                f"BUT THE HALVES ARE NOT EQUALLY SUPPORTED.   {did}\n"
                "S4's difference-in-differences is non-null at both grains; S3's is NULL at the pair grain, "
                "the conservative unit, and reaches significance only at the cell grain.\n"
                "Pair grain throughout, sign test, two declared DiDs (multiplicity uncorrected). "
                "PRODUCER LAYER SINGLE-PASS per [5503]: the parquet's own construction has not been "
                "independently regenerated."),
            x="change in self-surprisal vs the flat control  (nats; negative = less surprised)",
            y="",
            caption=("Producer: meta/M06_generation/scripts/m06_self_surprisal_figs.py from "
                     "results/self_surprisal_cells.parquet (producer m06_self_surprisal.py).\n"
                     "The typicality attack the finding named as its own weakest point was run and "
                     "does not land: the contrast is uncorrelated with the base-probability gap "
                     "(Spearman -0.030 base, -0.024 aligned) and the DiD holds its sign in every "
                     "tertile of that gap."),
        )
        + theme_minimal()
        + theme(figure_size=(12.0, 6.2),
                plot_title=element_text(size=12.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.2, color="#444444", ha="left"),
                plot_caption=element_text(size=6.3, color="#666666", ha="left"),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
                strip_text=element_text(size=8.2, weight="bold"),
                panel_spacing=0.05)
    )
    out = os.path.join(FIGURES, "self_surprisal_diagonal.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for k in sorted(stats, key=str):
        m, up, dn, pv = stats[k]
        print(f"    {k[0]} {k[1]:8s} {m:+.4f}  {up}/{dn}  p {pv:.4g}"
              + ("   <- diagonal" if k in SIG else ""))
    return out


FIGURES_REGISTRY = {"diagonal": diagonal}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in FIGURES_REGISTRY.items():
            print(f"  {k:12s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
