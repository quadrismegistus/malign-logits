"""The figure: coder intimacy against alignment's movement, split by frame.

    uv run python x_plot_intimacy.py

One panel. x is how intimate a coder judged the garment, y is whether alignment
moved onto it or off it. Two series, the female and male frames of the same
scene, so the gender gap is the distance between the two fitted lines rather
than a number in a table.

**Why D and not one of the others.** Four instruments agree at rho -0.43 to
-0.63 and the choice barely matters; D is used because its two model runs agree
at +0.888 AND it is the scene-derived scale, so its dimension was named by the
coders rather than imposed by us. The two D runs are averaged because they
agree; every other scale is in `x_coder_words.csv` for anyone who wants to
redraw this against them.

**y is net count, and four other outcome variables give the same picture** --
magnitude, rate, and each frame alone. See `x_coder_grid.csv`. The choice of
outcome was made silently on the first pass and is now reported in full.

Words are labelled where they earn it: the extremes of each axis and the items
the argument turns on. Labelling all 105 would be unreadable and labelling only
the convenient ones would be a lie by selection, so the rule is stated in the
code and applied mechanically.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)

#: named in the prose and therefore labelled whatever their position, so the
#: figure cannot be accused of showing only the words that behaved
ARGUED = ["underwear", "bra", "panties", "glasses", "helmet", "pants", "socks",
          "hijab", "stockings", "clothes", "seatbelt", "makeup"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    D = pd.read_csv(os.path.join(CAMP, "results", "x_coder_words.csv"))
    D["intimacy"] = D[["D_opus", "D_sonnet"]].mean(axis=1)
    D = D.dropna(subset=["intimacy"])

    fig, ax = plt.subplots(figsize=(11, 7))
    series = [("net_count_her", "#c2185b", "female frame  —  “She slowly took off her ___”"),
              ("net_count_his", "#1565c0", "male frame  —  “He slowly took off his ___”")]
    rng = np.random.default_rng(7)
    for col, colour, label in series:
        sub = D.dropna(subset=[col])
        jit = rng.uniform(-0.22, 0.22, len(sub))
        ax.scatter(sub["intimacy"], sub[col] + jit, s=26, alpha=0.5,
                   color=colour, edgecolors="none", zorder=2)
        r, p = stats.spearmanr(sub["intimacy"], sub[col])
        b, a = np.polyfit(sub["intimacy"], sub[col], 1)
        xs = np.linspace(0, 100, 50)
        ax.plot(xs, a + b * xs, color=colour, lw=2.4, zorder=3,
                label="%s\n        Spearman ρ = %+.2f   (p = %.0e)" % (label, r, p))

    ax.axhline(0, color="#999", lw=0.8, zorder=1)
    #: LABELS ARE THE FEMALE FRAME ONLY -- both series are plotted but one set of
    #: labels is readable and two is not. Said in the axis label so the reader is
    #: not left to infer which points the words attach to.
    lab = D[(D["intimacy"] >= 72) | (D["net_count_her"].abs() >= 8) |
            (D["word"].isin(ARGUED))].dropna(subset=["net_count_her"])
    lab = lab.sort_values("intimacy")
    used = []
    for _, r in lab.iterrows():
        x, y = r["intimacy"], r["net_count_her"]
        dy = 3
        #: nudge off any label already placed within a small box, so the
        #: left-hand cluster does not overprint itself
        for ux, uy in used:
            if abs(ux - x) < 7 and abs(uy - (y + dy / 8.0)) < 1.1:
                dy += 11
        used.append((x, y + dy / 8.0))
        ax.annotate(r["word"], (x, y), fontsize=8.5, color="#333",
                    xytext=(4, dy), textcoords="offset points", zorder=4)

    ax.set_xlabel("how intimate is this garment to remove   —   coder score, 0–100",
                  fontsize=11)
    ax.set_ylabel("alignment moves ONTO it   ←→   moves OFF it\n"
                  "(pairs where the word rose, minus pairs where it fell)\n"
                  "labelled points are the female frame", fontsize=11)
    ax.set_title("Alignment withdraws from the intimate garment and moves to the peripheral one\n"
                 "105 words, 36 base→aligned pairs per frame; coder scale named by the coders, "
                 "blind to which words moved",
                 fontsize=12.5, loc="left", pad=14)
    ax.legend(loc="lower left", fontsize=9.5, frameon=False, borderpad=1.1, labelspacing=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-4, 116)
    fig.tight_layout()

    for ext in ("png", "svg"):
        p = os.path.join(CAMP, "figures", "x_intimacy_vs_movement." + ext)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=180)
        print("wrote %s" % p)


if __name__ == "__main__":
    main()
