#!/usr/bin/env python
"""A-R1: the paired per-site onset lag, and why p = 0.97 is not simultaneity.

    uv run python meta/M05_emergence/scripts/m05_onset_figs.py
    uv run python meta/M05_emergence/scripts/m05_onset_figs.py --list

plot-debt M05 candidate 3. plotnine at 300 dpi, output to ../figures/, booked
numbers asserted before drawing. Folder convention: per-purpose `m05_*.py`
script, numbered `figNN_` output.

THE ARTIFACT HOLDS THE SUMMARY AND NOT THE DISTRIBUTION
--------------------------------------------------------
`data/m05_onsets.json` stores `primary_sft.paired` as five numbers: 44 sites
with both onsets, median lag 0, Wilcoxon p 0.972, 34 sites that never
persistently fall, 41 that never persistently rise. **The per-site lags
themselves are computed and discarded** -- `m05_onsets.py` builds `diffs`,
takes its median, runs the Wilcoxon on it, and writes neither.

So the histogram this entry asks for cannot be drawn from the committed
artifact. It is NOT ladder debt: the producer and its input
(`data/m05_curves.parquet`, committed) are both present, so the number stays
auditable and reproducible and only the distribution is unmaterialised. The
response is the one M03 candidate 7 established -- **import the producer's own
function rather than reimplement it**, since a reimplementation of a
threshold-free onset rule would silently re-choose it. `onset_persistent_sign`
is imported here; only the selection loop around it is replayed, and the five
booked summary values are asserted against the artifact afterwards, so a drift
in either the rule or the replay is caught.

WHY THE DISTRIBUTION IS THE POINT AND THE SUMMARY IS MISLEADING ALONE
----------------------------------------------------------------------
Median lag 0 with p = 0.97 reads as "the two onsets coincide". They almost
never do: **only 5 of the 44 sites have a lag of exactly zero**, and the
range runs from -39,000 to +39,000 steps, 21 sites negative against 18
positive. The null is SYMMETRY, not simultaneity -- large lags in both
directions that cancel. That distinction is the entire content of the entry's
"makes p = .97 legible", and no summary statistic carries it.
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
FIGURES = os.path.join(CAMP, "figures")
ROOT = os.path.dirname(os.path.dirname(CAMP))
CURVES = os.path.join(ROOT, "data", "m05_curves.parquet")
ONSETS = os.path.join(ROOT, "data", "m05_onsets.json")
PRODUCER = os.path.join(HERE, "m05_onsets.py")

#: primary_sft.paired, verbatim from the artifact
BOOKED = {"n_sites_both": 44, "median_lag_steps": 0.0,
          "wilcoxon_p": 0.9721596276935432, "sites_never_fall": 34,
          "sites_never_rise": 41}
#: the 105-pair sample the finding names, and the split this figure derives
BOOKED_POP = {"probes": 105, "neither": 14, "zero_lag": 5,
              "negative": 21, "positive": 18}
FALL_C, RISE_C, BAR_C = "#b03030", "#1f4e79", "#9a9a9a"


def _lags():
    """Replay the producer's per-site loop with its own onset rule imported."""
    spec = importlib.util.spec_from_file_location("m05_onsets_src", PRODUCER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    df = pd.read_parquet(CURVES)
    df = df[~df.payload_empty]
    panel = {r: g for r, g in df[df.curve == "PANEL"].groupby("word_role")}
    onsets = {}
    for role, direction in (("faller", "down"), ("riser", "up")):
        g = panel[role]
        bm = g[g.role == "base_endpoint"].set_index("probe").p
        per = {}
        for probe, t in g[g.role == "sft_step"].groupby("probe"):
            if probe not in bm.index:
                continue
            per[probe] = m.onset_persistent_sign(
                sorted(zip(t.step, t.p)), bm[probe], direction)
        onsets[role] = per
    return onsets


def onset_lag():
    """M05 candidate 3: the lag null is symmetry, not simultaneity."""
    from scipy.stats import wilcoxon
    from plotnine import (aes, element_blank, element_text, geom_col,
                          geom_histogram,
                          geom_text, geom_vline, ggplot, labs,
                          scale_fill_identity, scale_x_continuous,
                          scale_y_continuous, theme,
                          theme_minimal)

    onsets = _lags()
    f, r = onsets["faller"], onsets["riser"]
    both = [k for k in f if f[k] is not None and k in r and r[k] is not None]
    lags = np.array([r[k] - f[k] for k in both], dtype=float)
    never_fall = [k for k in f if f[k] is None]
    never_rise = [k for k in r if r[k] is None]
    neither = set(never_fall) & set(never_rise)
    _, p = wilcoxon(lags)

    #: THE REPLAY IS CHECKED AGAINST THE COMMITTED SUMMARY, all five values.
    #: This is what makes importing the rule safe: if either the rule or this
    #: loop drifts, the artifact disagrees and the figure refuses.
    book = json.load(open(ONSETS))["primary_sft"]["paired"]
    assert book == BOOKED, f"artifact's paired block changed: {book}"
    assert len(lags) == BOOKED["n_sites_both"], f"{len(lags)} sites, not 44"
    assert float(np.median(lags)) == BOOKED["median_lag_steps"], "median lag"
    assert abs(p - BOOKED["wilcoxon_p"]) < 1e-9, f"Wilcoxon p {p:.9f}"
    assert len(never_fall) == BOOKED["sites_never_fall"], "never-fall count"
    assert len(never_rise) == BOOKED["sites_never_rise"], "never-rise count"

    #: THE POPULATION CLOSES, which is how the flanking bars are kept honest:
    #: 34 and 41 are NOT disjoint and must not read as though they were.
    n_probes = len(set(f) | set(r))
    assert n_probes == BOOKED_POP["probes"], f"{n_probes} probes, not 105"
    assert len(neither) == BOOKED_POP["neither"], f"{len(neither)} neither"
    assert len(lags) + len(never_fall) + len(never_rise) - len(neither) \
        == n_probes, "the site accounting does not close"

    #: the claim the panel is built on: the median is zero and almost nothing is
    n_zero = int((lags == 0).sum())
    n_neg, n_pos = int((lags < 0).sum()), int((lags > 0).sum())
    assert (n_zero, n_neg, n_pos) == (BOOKED_POP["zero_lag"],
                                      BOOKED_POP["negative"],
                                      BOOKED_POP["positive"]), \
        f"lag signs moved: {n_zero} zero, {n_neg} negative, {n_pos} positive"
    assert n_zero < len(lags) / 4, \
        ("most sites now have zero lag; the panel's whole point is that the "
         "median is zero and the sites are not")

    d = pd.DataFrame({"lag": lags / 1000.0})
    #: A PLOTNINE WARNING THAT READS AS DATA LOSS AND IS NOT. This figure emits
    #: "geom_histogram : Removed 2 rows containing missing values." Those are
    #: two BOUNDARY BINS from stat_bin, not two sites: the count is exactly 2
    #: at n = 44, 400 and 4,000, and it disappears entirely when the x scale
    #: carries no explicit limits. Since this panel's argument is the spread of
    #: 44 sites, a genuine loss of two would matter, so the containment is
    #: asserted rather than trusted to that reasoning.
    assert not np.isnan(lags).any(), "a lag is NaN"
    span = float(np.abs(lags).max()) / 1000.0
    #: flanking bars live OUTSIDE the lag axis, because a site with no onset
    #: has no lag; they are placed past a marked break and never inside it
    gap, bw = span + 8, 5.0
    inside = int(((d.lag >= -(gap + bw)) & (d.lag <= gap + bw)).sum())
    assert inside == len(lags), \
        f"{len(lags) - inside} of {len(lags)} lags fall outside the x limits"
    bars = pd.DataFrame([
        {"x": -gap, "n": len(never_fall), "fill": FALL_C,
         "t": f"{len(never_fall)} sites\nnever fall"},
        {"x": +gap, "n": len(never_rise), "fill": RISE_C,
         "t": f"{len(never_rise)} sites\nnever rise"}])

    p_fig = (
        ggplot()
        + geom_col(bars, aes("x", "n", fill="fill"), width=bw, alpha=0.55)
        + geom_text(bars, aes("x", "n", label="t"), size=6.4, va="bottom",
                    nudge_y=0.6, color="#333333", lineheight=1.2)
        + geom_histogram(d, aes("lag"), bins=27, fill=BAR_C, colour="white",
                         size=0.25)
        + geom_vline(xintercept=0, color="#333333", size=0.6)
        + geom_vline(xintercept=[-(gap - bw), (gap - bw)], linetype="dotted",
                     color="#999999", size=0.5)
        + scale_fill_identity()
        #: HEADROOM FOR THE BAR LABELS, which are two lines and sit ABOVE the
        #: taller bar. Without it the 41-site bar's label loses its first line
        #: to the panel edge and reads "never rise" with the count gone --
        #: the truncation class again, and geom_text is invisible to both
        #: modes of figure_text_audit, so only the image shows it.
        + scale_y_continuous(limits=(0, 50), breaks=[0, 10, 20, 30, 40])
        + scale_x_continuous(
            limits=(-gap - bw, gap + bw),
            breaks=[-gap, -30, -20, -10, 0, 10, 20, 30, gap],
            labels=["no\nonset", "-30k", "-20k", "-10k", "0", "+10k", "+20k",
                    "+30k", "no\nonset"])
        + labs(
            title="The two onsets do not coincide: their lags are large in both directions and cancel",
            subtitle=(
                "Per-site lag between a substitute's rise-onset and its prohibited word's fall-onset,\n"
                "over the SFT arm at 43 rungs. Positive means the rise came later. Grey bars are the 44\n"
                "sites where both onsets exist; the flanking coloured bars are sites where one never\n"
                "happens, so they have no lag and sit outside the axis past the dotted breaks.\n"
                f"THE TEST IS NULL: median lag 0 steps, Wilcoxon p = {p:.2f}, n = {len(lags)}.\n"
                "THAT NULL IS SYMMETRY AND NOT SIMULTANEITY, which is the whole reason to draw it.\n"
                f"Only {n_zero} of the {len(lags)} sites have a lag of exactly zero. The lags run from -39,000 to\n"
                f"+39,000 steps, {n_neg} negative against {n_pos} positive. The onsets almost never coincide; the\n"
                "distribution is wide and balanced, and a median of zero is what balance looks like.\n"
                "WHAT IT KILLS. F04 reported repression preceding displacement as a lag, measured on a\n"
                "10-checkpoint grid. At site grain over 43 rungs there is no fall-then-rise sequence.\n"
                "The finding's replacement is a difference in KIND rather than in timing: the prohibition\n"
                "completes inside SFT as a detectable event, while the substitution keeps accumulating\n"
                "through DPO and RLVR and never resolves into an onset at all.\n"
                f"THE FLANKING BARS OVERLAP AND ARE NOT A PARTITION. {len(neither)} sites are in BOTH -- they never\n"
                f"fall and never rise. The {n_probes} probes of the 105-pair sample split as {len(lags)} with both\n"
                f"onsets, {len(never_fall) - len(neither)} that only never fall, {len(never_rise) - len(neither)} that only never rise, and {len(neither)} with neither.\n"
                f"THE TEST SEES {len(lags)} OF {n_probes} SITES. The other {n_probes - len(lags)} are excluded by construction, because a\n"
                "lag needs two onsets. That is not a filter this figure applied; it is what the statistic\n"
                "is, and it is why the flanking bars are on the panel rather than in a footnote."),
            x="lag in SFT steps  (rise-onset minus fall-onset)",
            y="sites",
            caption=(
                "Producer: meta/M05_emergence/scripts/m05_onset_figs.py from data/m05_curves.parquet,\n"
                "with `onset_persistent_sign` IMPORTED from meta/M05_emergence/scripts/m05_onsets.py.\n"
                "plot-debt M05 candidate 3.\n"
                "THE COMMITTED ARTIFACT HOLDS THE SUMMARY AND NOT THE DISTRIBUTION. data/m05_onsets.json\n"
                "stores five numbers for this result; m05_onsets.py builds the per-site lags, takes their\n"
                "median, runs the Wilcoxon and writes neither. The histogram cannot be drawn from it.\n"
                "This is not ladder debt -- producer and input are both committed, so the number stays\n"
                "auditable and reproducible and only the distribution is unmaterialised -- and the\n"
                "response is to import the producer's onset rule rather than reimplement it, since a\n"
                "reimplementation of a threshold-free rule would silently re-choose the threshold.\n"
                "Asserted before drawing: the artifact's whole `paired` block unchanged; all five of its\n"
                "values re-derived by the replay, the Wilcoxon p to 1e-09; that the 105-probe accounting\n"
                "closes with the 14 overlapping sites counted once; the lag sign split; and that zero-lag\n"
                "sites remain a minority, since the panel's argument depends on it."),
        )
        + theme_minimal()
        + theme(figure_size=(12.6, 7.6),
                plot_title=element_text(size=11.5, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                plot_caption=element_text(size=6.3, color="#666666", ha="left",
                                          lineheight=1.45),
                panel_grid_minor_x=element_blank())
    )
    out = os.path.join(FIGURES, "fig33_onset_lag_paired.png")
    p_fig.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    print(f"    {len(lags)} sites with both onsets, median lag "
          f"{np.median(lags):+.0f}, Wilcoxon p {p:.4f}")
    print(f"    zero lag {n_zero}, negative {n_neg}, positive {n_pos}, "
          f"range {lags.min():+.0f}..{lags.max():+.0f}")
    print(f"    {n_probes} probes = {len(lags)} both + "
          f"{len(never_fall) - len(neither)} never-fall-only + "
          f"{len(never_rise) - len(neither)} never-rise-only + {len(neither)} neither")
    return out


REGISTRY = {"onset_lag": onset_lag}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:16s} {(fn.__doc__ or '').strip().splitlines()[0]}")
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
