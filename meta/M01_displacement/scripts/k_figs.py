"""Quadrant figures for Plan K: where in the joint rating space the falling lives.

    uv run --with plotnine python meta/M01_displacement/scripts/k_figs.py
    -> meta/M01_displacement/figures/k_<a>_x_<b>.png          300 dpi, n>=20 cells
    -> meta/M01_displacement/figures/k_<a>_x_<b>_strict.png   300 dpi, n>=200 cells

Each panel is the 7x7 grid of two 1-7 rating scales, filled by the MEAN NET
MOVEMENT of the words in that tile (rises minus falls over the cells the word
appears in, one vote per word) and annotated with how many words are in it.
Blue falls, red rises.

A GRID, NOT A SCATTER, because the scales are integers 1-7: a scatter is 49
overplotted columns. The grid is the honest shape of the data, and it shows what
a correlation cannot -- WHERE in the joint space the movement is, and whether the
two scales interact or merely add.

THE THRESHOLD IS 20 CELLS, NOT 200. At 200 only 2,315 words survive and the
charged corners hold one to five words each while the flat corner holds 1,279 --
so the eye reads a fully saturated tile that is one word. At 20 cells the
population is 7,741 words, the median word has 79 cells, and charge>=5 goes from
66 words to 284. The strict version is written alongside as the robustness check;
if a corner changes sign between them, believe neither.

SPARSE TILES ARE DRAWN GREY, NOT FAINT. Below MIN_WORDS the mean is an anecdote,
and alpha alone still reads as a weak version of the colour rather than as an
absence of evidence. The n stays on the tile either way.

ONE VOTE PER WORD, not per cell. Weighting by cells would let a handful of
high-frequency words carry a tile, and the whole pseudo-replication problem in
this campaign is at the word level.
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
MIN_WORDS = 5          #: fewer words than this in a tile and it is drawn grey
PAIRS = [("charge", "transgressiveness"), ("charge", "bodily_harm"),
         ("charge", "concreteness"), ("charge", "valence"),
         ("charge", "vulgarity"), ("charge", "register_level"),
         ("bodily_harm", "transgressiveness"),
         ("concreteness", "transgressiveness"),
         ("concreteness", "bodily_harm")]


def population():
    """word -> (net summed over its tokens' cells, cells) for the English N=50 set."""
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


def panel(R, agg, a, b, min_cells, suffix):
    W = {u: v[0] / v[1] for u, v in agg.items() if v[1] >= min_cells}
    d = collections.defaultdict(list)
    for u, net in W.items():
        d[(R[u][a], R[u][b])].append(net)
    df = pd.DataFrame([{"a": g, "b": s, "net": float(np.mean(v)), "n": len(v),
                        "thin": len(v) < MIN_WORDS} for (g, s), v in d.items()])
    #: the scale is set by the tiles the reader is meant to believe, so one
    #: n=1 outlier in a corner cannot compress every populated tile to white
    solid = df[~df["thin"]]
    lim = float(np.nanmax(np.abs(solid["net"]))) if len(solid) else 1.0
    df["fill"] = np.where(df["thin"], np.nan, df["net"].clip(-lim, lim))
    p = (ggplot(df, aes("factor(a)", "factor(b)"))
         + geom_tile(aes(fill="fill"), colour="white", size=.6)
         + geom_text(aes(label="n"), size=6, colour="#404040")
         + scale_fill_gradient2(low="#2166ac", mid="#f7f7f7", high="#b2182b",
                                midpoint=0, limits=(-lim, lim),
                                na_value="#e8e8e8", name="mean net\n(rise - fall)")
         + labs(x=a.replace("_", " ") + "  (1-7)", y=b.replace("_", " ") + "  (1-7)",
                title="Net movement by %s x %s" % (a.replace("_", " "), b.replace("_", " ")),
                subtitle="%s English words seen in >=%d cells, one vote per word; number on "
                         "tile is words in it.\nGrey = fewer than %d words, mean not shown. "
                         "Blue = alignment pushes down, red = up."
                         % (f"{len(W):,}", min_cells, MIN_WORDS))
         + theme_minimal()
         + theme(figure_size=(6.4, 4.8), plot_title=element_text(size=11, weight="bold"),
                 plot_subtitle=element_text(size=7, colour="#595959"),
                 axis_title=element_text(size=9),
                 panel_grid=element_blank()))
    f = os.path.join(FIG, "k_%s_x_%s%s.png" % (a, b, suffix))
    p.save(f, dpi=300, verbose=False)
    return os.path.relpath(f, ROOT), len(solid), len(df)


def main():
    R, agg = population()
    for min_cells, suffix in ((20, ""), (200, "_strict")):
        print("\n--- words seen in >=%d cells" % min_cells)
        for a, b in PAIRS:
            f, ns, nt = panel(R, agg, a, b, min_cells, suffix)
            print("  %-52s  %2d of %2d tiles above the grey floor" % (f.split("/")[-1], ns, nt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
