"""Plan K figures: where in the joint rating space the movement lives, and which
words are actually in each region.

    uv run --with plotnine,adjustText python meta/M01_displacement/scripts/k_figs.py

    figures/k_<a>_x_<b>.png           7x7 tiles, mean net, word COUNT on the tile
    figures/k_<a>_x_<b>_strict.png    same at the >=200-cell threshold
    figures/k_<a>_x_<b>_words.png     7x7 tiles, the WORDS themselves in the tile
    figures/k_<a>_x_<b>_biplot.png    jittered scatter, labelled sample

COLOUR: BLUE RISES, RED FALLS. Set once in FILL_RISE / FILL_FALL below, so the
convention cannot drift between panels.

WHY BOTH A COUNT PANEL AND A WORD PANEL. The count panel is the measurement --
every word in the tile contributes to the mean and the n is stated. The word
panel is an exhibit: it can only show a handful per tile, so it is a sample and
must be read as one. Keeping them as separate files stops the exhibit from being
mistaken for the evidence.

THE WORDS SHOWN ARE THE MOST-MEASURED, NOT THE BIGGEST MOVERS. Selecting the
extreme movers would be selection on the outcome -- the panel would then show a
strong effect in every tile including tiles whose mean is zero. Ranking by cell
count selects on PRECISION, which is orthogonal to the thing being displayed.
Each word is still coloured by its own net, so the spread inside a tile is
visible rather than smoothed into the tile mean.

THE THRESHOLD IS 20 CELLS, NOT 200. At 200 only 2,315 words survive and the
charged corners hold one to five words each while the flat corner holds 1,279,
so the eye reads a saturated tile that is one word. At 20 the population is
7,741, the median word has 79 cells, and charge>=5 goes from 66 words to 284.
The strict version is the robustness check: a corner that changes sign between
the two is a corner to disbelieve.

SPARSE TILES ARE GREY, NOT FAINT. Below MIN_WORDS the mean is an anecdote, and
alpha alone reads as a weak version of the colour rather than as no evidence.

ONE VOTE PER WORD, NOT PER CELL, AND THE CHOICE CHANGES THE SIGN. Word-weighted
the population mean net is +0.063; cell-weighted it is -0.020. Weighting by
cells lets a few very high-frequency words carry a tile, and the pseudo-
replication in this campaign is at the word level -- but the reader should know
that the conservative choice here is also the one that understates the falling.
"""
import collections, json, os, sys
import numpy as np
import pandas as pd
from plotnine import *

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import k_analysis as A, k_population as KP

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
FIG = os.path.join(ROOT, "meta/M01_displacement/figures")
FILL_RISE = "#2166ac"   #: blue, alignment pushes the word UP
FILL_FALL = "#b2182b"   #: red, alignment pushes the word DOWN
MIN_WORDS = 5           #: fewer words in a tile than this and it is drawn grey
PER_TILE = 7            #: words shown per tile in the word panels
LABELS = 90             #: labelled points on a biplot
SEED = 20260812
PAIRS = [("charge", "transgressiveness"), ("charge", "bodily_harm"),
         ("charge", "concreteness"), ("charge", "valence"),
         ("charge", "vulgarity"), ("charge", "register_level"),
         ("bodily_harm", "transgressiveness"),
         ("concreteness", "transgressiveness"),
         ("concreteness", "bodily_harm")]


def population():
    """word -> [net summed over its tokens' cells, cells] for the English N=50 set."""
    R = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    ep = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                     for b, a in KP.reps("en"))
    rows = A.q("""SELECT word, countIf(cls='rise')-countIf(cls='fall') net, count() n FROM (
      SELECT m.word word, m.cls cls,
        row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_base DESC) rb,
        row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_aligned DESC) ra
      FROM %s.movement m INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
        WHERE status='ACTIVE' AND language='en') p ON m.prompt=p.prompt
      WHERE m.rule='canonical' AND (%s)) WHERE rb<=50 OR ra<=50 GROUP BY word"""
              % (A.DB, A.DB, ep))
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        u = t2u.get(r["word"])
        if u in R:
            agg[u][0] += r["net"]; agg[u][1] += r["n"]
    return R, agg


def _scale(lim, name="mean net\n(rise - fall)"):
    #: LOW is the falling end, so low takes the FALL colour. Stated because the
    #: convention was reversed once and nothing in the code objected.
    return scale_fill_gradient2(low=FILL_FALL, mid="#f7f7f7", high=FILL_RISE,
                                midpoint=0, limits=(-lim, lim),
                                na_value="#e8e8e8", name=name)


def _sub(n_words, min_cells, extra=""):
    return ("%s English words seen in >=%d cells, one vote per word. %s\n"
            "Blue = alignment pushes the word UP, red = pushes it DOWN."
            % (f"{n_words:,}", min_cells, extra))


def tiles(R, agg, a, b, min_cells, suffix):
    """The measurement panel: every word counts, the tile carries its n."""
    W = {u: v[0] / v[1] for u, v in agg.items() if v[1] >= min_cells}
    d = collections.defaultdict(list)
    for u, net in W.items():
        d[(R[u][a], R[u][b])].append(net)
    df = pd.DataFrame([{"a": g, "b": s, "net": float(np.mean(v)), "n": len(v),
                        "thin": len(v) < MIN_WORDS} for (g, s), v in d.items()])
    #: the scale is set by the tiles the reader is meant to believe, so one
    #: n=1 corner cannot compress every populated tile to white
    solid = df[~df["thin"]]
    lim = float(np.nanmax(np.abs(solid["net"]))) if len(solid) else 1.0
    df["fill"] = np.where(df["thin"], np.nan, df["net"].clip(-lim, lim))
    p = (ggplot(df, aes("factor(a)", "factor(b)"))
         + geom_tile(aes(fill="fill"), colour="white", size=.6)
         + geom_text(aes(label="n"), size=6, colour="#404040")
         + _scale(lim)
         + labs(x=a.replace("_", " ") + "  (1-7)", y=b.replace("_", " ") + "  (1-7)",
                title="Net movement by %s x %s" % (a.replace("_", " "), b.replace("_", " ")),
                subtitle=_sub(len(W), min_cells,
                              "Number on tile is words in it; grey = fewer than %d, "
                              "mean not shown." % MIN_WORDS))
         + theme_minimal()
         + theme(figure_size=(6.4, 4.8), plot_title=element_text(size=11, weight="bold"),
                 plot_subtitle=element_text(size=7, colour="#595959"),
                 axis_title=element_text(size=9), panel_grid=element_blank()))
    f = os.path.join(FIG, "k_%s_x_%s%s.png" % (a, b, suffix))
    p.save(f, dpi=300, verbose=False)
    return os.path.basename(f)


def words(R, agg, a, b, min_cells):
    """The exhibit panel: the most-measured words in each tile, each in its own
    colour, so the reader can see what the region is MADE OF."""
    W = {u: (v[0] / v[1], v[1]) for u, v in agg.items() if v[1] >= min_cells}
    d = collections.defaultdict(list)
    for u, (net, cells) in W.items():
        d[(R[u][a], R[u][b])].append((cells, net, u))
    rows, tile = [], []
    for (g, s), v in d.items():
        v.sort(reverse=True)                       #: most-measured first
        show = v[:PER_TILE]
        for i, (cells, net, u) in enumerate(show):
            #: stack inside the tile; 0.5 is the half-width of a unit tile
            rows.append({"a": g, "y": s + 0.36 - 0.72 * (i + .5) / len(show),
                         "word": u, "net": net})
        tile.append({"a": g, "b": s, "n": len(v),
                     "more": "+%d" % (len(v) - len(show)) if len(v) > len(show) else ""})
    df, td = pd.DataFrame(rows), pd.DataFrame(tile)
    #: THE 70TH PERCENTILE, NOT THE 95TH, and only in this panel. Thin coloured
    #: text washes out where a filled tile would still read, so the scale
    #: saturates sooner here than in the measurement panels. That makes small
    #: movements look larger: the exhibit shows DIRECTION, the tile panel shows
    #: magnitude, and the two are deliberately not on the same scale.
    lim = float(np.percentile(np.abs(df["net"]), 70)) or 1.0
    df["col"] = df["net"].clip(-lim, lim)
    #: BOTH AXES CONTINUOUS. The words are placed at fractional y inside a tile,
    #: so a discrete y scale (factor(b)) cannot map them -- mixing the two raises
    #: "Unordered Categoricals can only compare equality".
    p = (ggplot()
         + geom_tile(td, aes("a", "b"), fill="#fafafa", colour="#dddddd",
                     size=.5, width=1, height=1)
         + geom_text(df, aes("a", "y", label="word", colour="col"), size=5.0)
         + geom_text(td, aes("a", "b + 0.42", label="more"), size=4, colour="#999999")
         + scale_colour_gradient2(low=FILL_FALL, mid="#9a9a9a", high=FILL_RISE,
                                  midpoint=0, limits=(-lim, lim),
                                  name="net, this word\n(clipped at the 70th pct)")
         + scale_x_continuous(breaks=range(1, 8), limits=(0.5, 7.5))
         + scale_y_continuous(breaks=range(1, 8), limits=(0.5, 7.6))
         + labs(x=a.replace("_", " ") + "  (1-7)", y=b.replace("_", " ") + "  (1-7)",
                title="Which words are in each region: %s x %s"
                      % (a.replace("_", " "), b.replace("_", " ")),
                subtitle=_sub(len(W), min_cells,
                              "Up to %d words per tile, chosen by MOST CELLS MEASURED "
                              "(not by size of movement); +N is how many are not shown."
                              % PER_TILE))
         + theme_minimal()
         + theme(figure_size=(13, 9.5), plot_title=element_text(size=13, weight="bold"),
                 plot_subtitle=element_text(size=8, colour="#595959"),
                 axis_title=element_text(size=10), panel_grid=element_blank()))
    f = os.path.join(FIG, "k_%s_x_%s_words.png" % (a, b))
    p.save(f, dpi=300, verbose=False)
    return os.path.basename(f)


def biplot(R, agg, a, b, min_cells):
    """Every word as a jittered point; a stratified sample labelled.

    The labelled sample is the most-measured word in each (a, b, sign-of-net)
    stratum, so both directions appear in every region that has both -- labelling
    the top |net| would show only the extremes and hide that most tiles contain
    words moving each way.
    """
    W = {u: (v[0] / v[1], v[1]) for u, v in agg.items() if v[1] >= min_cells}
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame([{"a": R[u][a], "b": R[u][b], "net": n, "cells": c, "word": u}
                       for u, (n, c) in W.items()])
    df["x"] = df["a"] + rng.uniform(-.38, .38, len(df))
    df["y"] = df["b"] + rng.uniform(-.38, .38, len(df))
    lim = float(np.percentile(np.abs(df["net"]), 95)) or 1.0
    df["col"] = df["net"].clip(-lim, lim)
    df["sign"] = np.where(df["net"] >= 0, "up", "down")
    pick = (df.sort_values("cells", ascending=False)
              .groupby(["a", "b", "sign"], as_index=False).head(1)
              .sort_values("cells", ascending=False).head(LABELS))
    p = (ggplot(df, aes("x", "y"))
         + geom_point(aes(colour="col", size="np.log10(cells)"), alpha=.55, stroke=0)
         + geom_text(pick, aes("x", "y", label="word"), size=5.2, colour="#111111",
                     adjust_text={"expand_points": (1.6, 1.8), "arrowprops":
                                  {"arrowstyle": "-", "color": "#999999", "lw": .4}})
         + scale_colour_gradient2(low=FILL_FALL, mid="#cccccc", high=FILL_RISE,
                                  midpoint=0, limits=(-lim, lim), name="net")
         + scale_size_continuous(range=(.6, 3.4), name="log10 cells")
         + scale_x_continuous(breaks=range(1, 8)) + scale_y_continuous(breaks=range(1, 8))
         + labs(x=a.replace("_", " ") + "  (1-7, jittered)",
                y=b.replace("_", " ") + "  (1-7, jittered)",
                title="Every word: %s x %s" % (a.replace("_", " "), b.replace("_", " ")),
                subtitle=_sub(len(W), min_cells,
                              "Ratings are integers; positions are jittered. Labels are the "
                              "most-measured riser and faller in each cell, %d shown." % LABELS)
                + "\nPoint size is log10 of cells measured.")
         + theme_minimal()
         + theme(figure_size=(11, 8.5), plot_title=element_text(size=12, weight="bold"),
                 plot_subtitle=element_text(size=8, colour="#595959"),
                 axis_title=element_text(size=10)))
    f = os.path.join(FIG, "k_%s_x_%s_biplot.png" % (a, b))
    p.save(f, dpi=300, verbose=False)
    return os.path.basename(f)


def main():
    R, agg = population()
    for min_cells, suffix in ((20, ""), (200, "_strict")):
        print("\n--- tiles, words seen in >=%d cells" % min_cells)
        for a, b in PAIRS:
            print("  %s" % tiles(R, agg, a, b, min_cells, suffix))
    print("\n--- word panels and biplots, >=20 cells")
    for a, b in PAIRS:
        print("  %s" % words(R, agg, a, b, 20))
        print("  %s" % biplot(R, agg, a, b, 20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
