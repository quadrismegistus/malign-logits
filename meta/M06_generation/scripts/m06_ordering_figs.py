#!/usr/bin/env python
"""Alignment compresses every passage; its sequence moves both ways.

    uv run python meta/M06_generation/scripts/m06_ordering_figs.py
    uv run python meta/M06_generation/scripts/m06_ordering_figs.py --list

Drawn on RH's request, 2026-08-15, after `drift_metric_audit.md` recommended
`ordering` as the replacement for `directedness`. plotnine at 300 dpi, output
to ../figures/. Case 1 by shape: reads committed parquets and writes pixels.

WHY THIS METRIC AND NOT THE OTHERS
------------------------------------
The audit found every other member of the drift family unable to carry a
sequence claim: `total_drift` is ORDER-INVARIANT (a diameter, identical after
shuffling), `directedness` is sentence count (Spearman -0.923, R^2 0.795
against 1.681/n), and `path_length == (n_sents - 1) * mean_drift` at
correlation 1.000000, so it is not independent of `mean_drift` at all.

    ordering = mean(successive distances) - mean(all pairwise distances)

Under a RANDOM ordering of a passage's own sentences the expected successive
distance is exactly the mean of all pairwise distances, so this is zero in
expectation for a shuffled passage. Composition and sentence count are held
fixed BY CONSTRUCTION -- same sentences, same n, only the order differs.

THE NULL IS DRAWN WITH ITS SPREAD, WHICH IS THE POINT OF DRAWING IT
---------------------------------------------------------------------
The arm contrast on `ordering` is null: 11 up / 14 down in English, p 0.69.
**A null on a centred statistic licenses "no consistent direction" and never
"no effect"** (registrar [6216], from lacan's own English case). So the 25
per-pair deltas are drawn individually rather than summarised, and their range
is stated: pairs move up to +/-0.02 in both directions, which is a substantial
fraction of the -0.034 baseline ordering effect itself.

The per-pair deltas WERE computed and discarded; lacan emitted them at
`e4333807` on this figure's account, so this producer reads `arms.<metric>.<lang>
.per_pair` from the committed artifact instead of replaying the upstream loop.

**A vector emitted beside a statistic silently claims to be that statistic's
population** (lacan, [6252]): `sign_test` drops non-finite values internally, so
a per-pair dict built by zipping names to deltas can carry pairs the `n` beside
it never counted. That assert lives upstream now; it is mirrored here on the
consumer side, because a figure that draws a range from the vector and prints an
`n` from the summary is exactly where the mismatch would surface as a wrong
number rather than an error.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
FIGURES = os.path.join(CAMP, "figures")
ROOT = os.path.dirname(os.path.dirname(CAMP))
PRODUCER = os.path.join(HERE, "m06_crosslingual_ordering.py")
BOOKED = os.path.join(RESULTS, "crosslingual_ordering_full.json")

#: descriptive block of crosslingual_ordering_full.json
BOOKED_DESC = {"en": (-0.03384894849735671, 16002),
               "zh": (-0.026590351265557196, 13862)}
ROWS = [("mean_drift", "en"), ("mean_drift", "zh"),
        ("ordering", "en"), ("ordering", "zh")]
LABEL = {"mean_drift": "mean successive distance",
         "ordering": "ordering  (successive - all pairwise)"}
C_DRIFT, C_ORD = "#b03030", "#1f4e79"


def _load():
    """Per-pair deltas straight from the artifact, checked against their own n."""
    spec = importlib.util.spec_from_file_location("m06_ord_src", PRODUCER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    book = json.load(open(BOOKED))

    out = {}
    for metric, l in ROWS:
        arm = book["arms"][metric][l]
        pp = arm["per_pair"]
        #: THE VECTOR MUST BE THE STATISTIC'S POPULATION, not merely beside it.
        assert len(pp) == arm["n_pairs"], \
            (f"{metric}/{l}: {len(pp)} per-pair entries against n_pairs "
             f"{arm['n_pairs']}; the drawn range and the printed n would "
             "describe different sets")
        #: SORTED BY PAIR NAME, so the drawn order is a property of the data
        #: and not of dict insertion. geom_jitter assigns offsets by ROW
        #: POSITION, so an upstream reordering that changes no value still
        #: changes every pixel -- caught by the PNG growing 633 bytes on a
        #: refactor whose printed values were identical to the digit.
        v = np.array([pp[k] for k in sorted(pp)], dtype=float)
        assert np.isfinite(v).all(), f"{metric}/{l}: a per-pair delta is non-finite"
        #: and it must REPRODUCE the summary printed next to it
        r = m.sign_test(list(v))
        for k in ("median", "up", "dn", "n_pairs"):
            assert abs(r[k] - arm[k]) < 1e-9, \
                f"{metric}/{l} {k}: per_pair gives {r[k]}, artifact says {arm[k]}"
        out[(metric, l)] = v

    #: the descriptive effect is cross-checked against the CELLS, so the panel's
    #: baseline claim does not rest on the same file as its arm contrast
    df = pd.concat([pd.read_parquet(os.path.join(
        RESULTS, f"crosslingual_drift_{l}_full_cells.parquet"))
        for l in ("zh", "en")])
    df["ordering"] = df["mean_drift"] - df["mean_pairwise"]
    return out, df, m


def ordering_dissociation():
    """One metric is unanimous and the other has no consistent direction."""
    from plotnine import (aes, element_blank, element_text, geom_hline,
                          geom_jitter, geom_point, geom_segment, geom_text,
                          geom_vline, ggplot, labs, scale_color_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    deltas, df, m = _load()
    book = json.load(open(BOOKED))

    #: the descriptive effect, which is what the arm null is measured against
    for l, (om, nc) in BOOKED_DESC.items():
        g = df[df.lang == l]
        assert len(g) == nc, f"{l}: {len(g)} cells, not {nc}"
        assert abs(g["ordering"].mean() - om) < 5e-9, \
            f"{l}: ordering mean {g['ordering'].mean():.6f} vs booked {om:.6f}"
        #: ORDERING IS NEGATIVE, which is the metric working: successive
        #: sentences are CLOSER than random pairs of the same sentences.
        assert g["ordering"].mean() < 0, f"{l}: ordering is no longer negative"

    #: all four booked arm summaries, re-derived through the producer's own rule
    for (metric, l), ds in deltas.items():
        r = m.sign_test(list(ds))
        bk = book["arms"][metric][l]
        for k in ("median", "up", "dn", "n_pairs"):
            assert abs(r[k] - bk[k]) < 1e-9, \
                f"{metric}/{l} {k}: {r[k]} vs booked {bk[k]}"

    dr_en, or_en = deltas[("mean_drift", "en")], deltas[("ordering", "en")]
    #: THE DISSOCIATION, AS A TEST. One metric is unanimous and the other is not.
    assert (dr_en < 0).all(), "mean_drift is no longer negative in every en pair"
    assert 0 < (or_en > 0).sum() < len(or_en), \
        "ordering no longer moves in both directions in en"
    #: and the null's spread is comparable to the effect it is a null about
    or_range = or_en.max() - or_en.min()
    assert or_range > 0.5 * abs(BOOKED_DESC["en"][0]), \
        ("the ordering deltas no longer span a substantial fraction of the "
         "baseline ordering effect; the panel's spread argument depends on it")

    #: OFFSETS COMPUTED, NOT JITTERED. geom_jitter(random_state=0) does not
    #: pin the positions: two renders of identical sorted data differed on
    #: 51,612 pixels. The figure's dot placement was a property of the run
    #: rather than of the data, which is the same defect as an artifact built
    #: through a read with no ORDER BY. A seeded generator over the sorted
    #: vector makes the panel reproducible.
    rng = np.random.default_rng(0)
    rows = []
    for i, (metric, l) in enumerate(ROWS):
        v = deltas[(metric, l)]
        y = len(ROWS) - 1 - i
        col = C_DRIFT if metric == "mean_drift" else C_ORD
        off = rng.uniform(-0.055, 0.055, len(v))
        for x, o in zip(v, off):
            rows.append({"y": y + float(o), "x": float(x), "col": col})
    d = pd.DataFrame(rows)

    stats = []
    for i, (metric, l) in enumerate(ROWS):
        v = deltas[(metric, l)]
        r = m.sign_test(list(v))
        y = len(ROWS) - 1 - i
        stats.append({
            "y": y, "med": float(np.median(v)),
            "lab": f"{LABEL[metric]}   [{l}]",
            "stat": (f"{r['up']} up / {r['dn']} down    p {r['p_sign']:.3g}"
                     f"    median {r['median']:+.4f}"
                     f"    range {v.min():+.4f} to {v.max():+.4f}"),
            "col": C_DRIFT if metric == "mean_drift" else C_ORD})
    st = pd.DataFrame(stats)
    st["y0"], st["y1"] = st.y - 0.13, st.y + 0.13
    st["lx"] = -0.163
    #: TEXT ABOVE THE STRIP, NOT BESIDE IT. The dots span the full axis, so any
    #: left-aligned label sits on the leftmost pair -- the zh mean_drift stat
    #: line ran through its own -0.112 dot.
    st["lab_y"], st["stat_y"] = st.y + 0.40, st.y + 0.26

    xlim = (-0.165, 0.045)
    assert d.x.min() > xlim[0] and d.x.max() < xlim[1], \
        f"a pair falls outside the axis: {d.x.min():+.4f}..{d.x.max():+.4f}"

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.6)
        + geom_point(d, aes("x", "y", color="col"), size=2.4, alpha=0.75)
        + geom_segment(st, aes("med", "y0", xend="med", yend="y1"),
                       color="#1a1a1a", size=1.5)
        + geom_text(st, aes("lx", "lab_y", label="lab", color="col"), size=7.4,
                    ha="left")
        + geom_text(st, aes("lx", "stat_y", label="stat"), size=6.3,
                    ha="left", color="#666666")
        + scale_color_identity()
        + scale_x_continuous(limits=xlim,
                             breaks=[-0.15, -0.10, -0.05, 0, 0.025])
        + scale_y_continuous(breaks=[], limits=(-0.45, len(ROWS) - 0.40))
        + labs(
            title="Alignment compresses every passage, and moves its sequence in both directions",
            subtitle=(
                "One dot per base/aligned pair, 25 pairs usable in both languages, matched on prompt\n"
                "before differencing. Heavy tick is the median. Negative means alignment reduces.\n"
                "THE TOP TWO ROWS ARE UNANIMOUS AND THE BOTTOM TWO ARE NOT. Mean successive distance\n"
                "falls in 25 of 25 English pairs and 24 of 25 Chinese. The ordering contrast moves in\n"
                "both directions in both languages and does not clear a sign test.\n"
                "ORDERING IS THE ONLY MEMBER OF THIS FAMILY THAT CAN CARRY A SEQUENCE CLAIM. It is\n"
                "mean(successive distances) minus mean(all pairwise distances), which is zero in\n"
                "expectation for a shuffled passage, so composition and sentence count are held fixed BY\n"
                "CONSTRUCTION. `total_drift` is order-invariant, `directedness` is sentence count\n"
                "(R2 0.795 against 1.681/n), and `path_length` equals (n-1) x mean_drift at correlation\n"
                "1.000000 -- none of them can distinguish an ordered passage from a shuffled one.\n"
                "THE BOTTOM ROWS ARE A NULL WITH A SPREAD, NOT AN ABSENCE. English pairs run -0.0209 to\n"
                "+0.0120 and Chinese -0.0228 to +0.0133. The baseline ordering effect these are a null\n"
                "ABOUT is -0.034 in English, so individual pairs move by more than half of it in each\n"
                "direction while the median sits near zero. What is established is no consistent\n"
                "DIRECTION, and nothing here says alignment leaves sequence untouched pair by pair.\n"
                "AND THE BASELINE EFFECT IS REAL: ordering is negative in both languages (-0.034 English,\n"
                "-0.027 Chinese, ~30,000 passages), meaning successive sentences are reliably CLOSER than\n"
                "random pairs drawn from the same passage. The metric detects sequence structure; what it\n"
                "does not detect is alignment changing it."),
            x="change under alignment, aligned minus base", y="")
        + theme_minimal()
        + theme(figure_size=(12.8, 6.8),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "m06_ordering_dissociation.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for s in stats:
        print(f"    {s['lab']:<44} {s['stat']}")
    return out


REGISTRY = {"ordering_dissociation": ordering_dissociation}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:24s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
