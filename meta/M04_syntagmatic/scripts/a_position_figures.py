"""M04/A — position-resolved surprisal curves, by arm and by scoring term.

WHAT IS PLOTTED. Mean surprisal (-logprob, nats) at each SENTENCE POSITION, for
the four scoring terms, faceted by arm. Five arms: the UNDISTURBED arm
(`forced_word=''`, 238,400 sequences) plus the four forced arms of the frozen
table.

**POSITIONS ARE ALIGNED ON THE SENTENCE, NOT ON THE ARRAY INDEX.** The forced
word is not itself scored -- verified: `logprobs[0]` varies across samples of one
forced word while log q is fixed, which it could not do if index 0 were the
forced token. So a forced arm's index 0 is the undisturbed arm's index
`n_forced_tokens`. Plotting raw indices against each other would compare
sentence position 2 with sentence position 1, and later positions are more
predictable because context has accumulated -- the exact defect A's file records
as making its forced-vs-undisturbed positive control INVALID BY DESIGN. Here the
offset is added rather than ignored, so the undisturbed arm is a legible
reference curve; it is still not a control, for the other reason A gives (the
undisturbed arm has committed to nothing and the forced arms have committed to a
word, and the entropy drop from committing to ANYTHING is definitional).

UNIT. Mean per pair at each position, then MEDIAN ACROSS THE 42 PAIRS. Absolute
logprob level differs by model family, so pooling raw rows would let the
loudest-scaled pairs draw the curve.

TERMS. role x scorer, in A's notation: first letter is who WROTE the text,
second is who SCORED it. A|A is the aligned model reading its own continuation.
"""

import json, os, sys, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import a_matched_control as A
import a_dose_response as R

FIGDIR = "meta/M04_syntagmatic/figures"
CACHE = "meta/M04_syntagmatic/results/a_position_curves.json"
MAXPOS = 256
MINPOS = 2      # position 1 is dropped: the undisturbed arm spikes to 4.4-5.4 nats
                # there (the cost of the first token after a bare prompt) and that
                # one point sets the y-range for every panel, flattening the 0.05
                # differences the figures exist to show. Dropped, never averaged
                # away -- it is a real value, and hiding it inside a smoother
                # would let it bias the first window silently.
ROLL = 5        # centred rolling mean over positions, DISCLOSED ON EVERY FIGURE.

#: Okabe-Ito, colourblind-safe, and the hue CARRIES THE DIRECTION rather than
#: just separating categories: greys are the two references, warm is the word
#: alignment pushed DOWN, cool is the two it pushed UP. A qualitative brewer
#: palette gave `undisturbed` and `riser (high q)` two similar greens, which is
#: precisely the pair a reader needs to tell apart in the A|A panel.
ARM_COLOURS = {"undisturbed": "#999999", "non-mover": "#333333",
               "faller": "#D55E00", "riser (matched q)": "#0072B2",
               "riser (high q)": "#56B4E9"}

#: Terms need their OWN palette, and finding that out cost a silent defect worth
#: recording: handing `scale_colour_manual` a dict with none of the mapped levels
#: in it does not raise -- plotnine drew all four terms in identical grey, wrote
#: the file, and printed a legend with four entries. A figure that renders is not
#: a figure that encodes. Hue = who WROTE the text, shade = who SCORED it.
TERM_COLOURS = {"A|A": "#D55E00", "A|B": "#E69F00",
                "B|A": "#0072B2", "B|B": "#56B4E9"}
ARMS = (("faller", "faller"), ("matched", "non-mover"),
        ("riser_matched", "riser (matched q)"), ("riser", "riser (high q)"))


def esc(s):
    return s.replace("'", "''")


def collect():
    tab = R.arms_table()
    bypair = {}
    for c in tab:
        bypair.setdefault(c["pair"], []).append(c)
    out = []
    pairs = A.pairs_present()
    for n, pair in enumerate(pairs, 1):
        base_m, aln_m = pair.split(">", 1)
        cells = bypair.get(pair, [])
        if not cells:
            continue
        branches = []
        for col, label in ARMS:
            tup = ["('%s','%s')" % (esc(c["prompt"]), esc(c[col]))
                   for c in cells if c.get(col)]
            if tup:
                branches.append("(s.prompt, s.forced_word) IN (%s), '%s'"
                                % (",".join(tup), label))
        arm_expr = ("multiIf(s.forced_word = '', 'undisturbed', "
                    + ", ".join(branches) + ", 'other')")
        sql = (
            "SELECT arm, role, scorer, pos, avg(lp) AS m, count() AS n FROM ("
            "  SELECT %s AS arm, s.role AS role, g.scorer AS scorer, "
            "         s.n_forced_tokens AS nft, g.logprobs AS lps "
            "  FROM (SELECT corpus,model,prompt,forced_word,sample_idx,role,n_forced_tokens "
            "        FROM %s.gen_sequences FINAL WHERE corpus='%s' AND pair='%s') AS s "
            "  INNER JOIN (SELECT corpus,model,prompt,forced_word,sample_idx,scorer,logprobs,scorable "
            "              FROM %s.gen_scores FINAL WHERE corpus='%s') AS g "
            "    ON s.corpus=g.corpus AND s.model=g.model AND s.prompt=g.prompt "
            "       AND s.forced_word=g.forced_word AND s.sample_idx=g.sample_idx "
            "  WHERE g.scorable = 1"
            ") ARRAY JOIN arrayEnumerate(lps) AS idx, lps AS lp "
            "WHERE arm != 'other' AND idx + nft <= %d "
            "GROUP BY arm, role, scorer, idx + nft AS pos"
            % (arm_expr, A.DB, A.CORPUS, esc(pair), A.DB, A.CORPUS, MAXPOS))
        try:
            rows = A.rows(sql)
        except Exception as e:
            print("  %2d/%d %-44s QUERY FAILED %s" % (n, len(pairs), pair[:44], str(e)[:60]))
            continue
        for arm, role, scorer, pos, m, cnt in rows:
            sc = "A" if scorer == aln_m else ("B" if scorer == base_m else None)
            if sc is None:
                continue
            term = {"aligned": "A", "base": "B"}[role] + "|" + sc
            out.append({"pair": pair, "arm": arm, "term": term,
                        "pos": int(pos), "mean_lp": float(m), "n": int(cnt)})
        print("  %2d/%d %-44s %6d rows" % (n, len(pairs), pair[:44], len(rows)))
    json.dump(out, open(CACHE, "w"))
    print("  cached %d rows -> %s" % (len(out), CACHE))
    return out


def _smooth(g, col, roll=None):
    """Centred rolling mean within each (arm, term) series, after aggregation.

    Smoothing AFTER the median-over-pairs, never before: rolling within a pair
    first would blur each pair's curve and then the median would be taken over
    already-blurred inputs, which changes what the median is of. The window is
    named on every figure -- an undisclosed smoother is a claim about noise the
    reader cannot check.
    """
    roll = ROLL if roll is None else roll
    if roll <= 1:
        return g
    g = g.sort_values(["arm", "term", "pos"]).copy()
    g[col] = (g.groupby(["arm", "term"], observed=True)[col]
                .transform(lambda v: v.rolling(roll, center=True, min_periods=1).mean()))
    return g


def figures(rows):
    import pandas as pd
    from plotnine import (ggplot, aes, geom_line, geom_hline, facet_wrap, labs,
                          theme_bw, theme, element_text, scale_x_continuous, scale_y_continuous,
                          scale_colour_manual, guides, guide_legend)
    df = pd.DataFrame(rows)
    df["surprisal"] = -df["mean_lp"]
    # median across pairs — absolute level is family-specific
    g = (df.groupby(["arm", "term", "pos"], as_index=False)
           .agg(surprisal=("surprisal", "median"), pairs=("pair", "nunique")))
    g = g[(g["pairs"] >= 30) & (g["pos"] >= MINPOS)]
    order = ["undisturbed", "non-mover", "faller", "riser (matched q)", "riser (high q)"]
    g["arm"] = pd.Categorical(g["arm"], categories=[a for a in order if a in set(g["arm"])],
                              ordered=True)
    g = _smooth(g, "surprisal")
    os.makedirs(FIGDIR, exist_ok=True)

    p = (ggplot(g, aes("pos", "surprisal", colour="term"))
         + geom_line(size=0.5)
         + facet_wrap("~arm", ncol=3)
         + scale_x_continuous(limits=(MINPOS, 64))
         + scale_colour_manual(values=TERM_COLOURS)
         + labs(x="sentence position (token index, forced word at 0)",
                y="surprisal  (nats, median over 42 pairs)",
                colour="text|scorer",
                title="Surprisal by position, arm and scoring term",
                subtitle="first letter = who wrote it, second = who scored it; "
                         "positions aligned on the sentence, not the array. "
                         "Position 1 dropped; %d-position centred rolling mean." % ROLL)
         + theme_bw()
         + theme(figure_size=(11, 6), plot_title=element_text(size=11, weight="bold"),
                 plot_subtitle=element_text(size=8), strip_text=element_text(size=8)))
    p.save(os.path.join(FIGDIR, "A_position_surprisal.png"), dpi=300, verbose=False)

    # full range, log-ish view of the tail
    p2 = (p + scale_x_continuous(limits=(MINPOS, MAXPOS))
            + labs(subtitle="full 256-token window"))
    p2.save(os.path.join(FIGDIR, "A_position_surprisal_full.png"), dpi=300, verbose=False)

    # CONTRAST: PAIRED — per-pair delta at each position, THEN median over pairs.
    # Differencing the medians is not the paired statistic and is far noisier.
    # The undisturbed arm is EXCLUDED here: its position-1 spike (+2.6 nats) is
    # the definitional cost of committing to any word at all, which A's file
    # gives as one of two reasons forced-vs-undisturbed is invalid as a control.
    # Left in the level plot as a reference curve; wrong to difference against.
    base = df[df["arm"] == "non-mover"][["pair", "term", "pos", "surprisal"]]
    base = base.rename(columns={"surprisal": "ref"})
    m = df[~df["arm"].isin(["non-mover", "undisturbed"])].merge(
        base, on=["pair", "term", "pos"], how="inner")
    m["delta"] = m["surprisal"] - m["ref"]
    dd = (m.groupby(["arm", "term", "pos"], as_index=False)
            .agg(delta=("delta", "median"), pairs=("pair", "nunique")))
    dd = dd[(dd["pairs"] >= 30) & (dd["pos"] >= MINPOS)]
    dd["arm"] = pd.Categorical(dd["arm"], categories=[a for a in order if a in set(dd["arm"])],
                               ordered=True)
    dd = _smooth(dd, "delta")
    for tag, xmax, ymax in (("", 64, 0.20), ("_full", MAXPOS, 0.20)):
        p3 = (ggplot(dd, aes("pos", "delta", colour="term"))
              + geom_hline(yintercept=0, size=0.3, colour="#666666")
              + geom_line(size=0.45)
              + facet_wrap("~arm", ncol=3)
              + scale_x_continuous(limits=(MINPOS, xmax))
              + scale_y_continuous(limits=(-ymax, ymax))
              + scale_colour_manual(values=TERM_COLOURS)
              + labs(x="sentence position",
                     y="surprisal minus the non-mover arm (nats)",
                     colour="text|scorer",
                     title="Each arm against its matched non-mover, paired by pair",
                     subtitle="above 0 = MORE surprising than a word alignment left alone. "
                              "All three arms sit at the faller's aligned probability; "
                              "only the direction of movement differs. "
                              "Position 1 dropped; %d-position centred rolling mean." % ROLL)
              + theme_bw()
              + theme(figure_size=(11, 4.2), plot_title=element_text(size=11, weight="bold"),
                      plot_subtitle=element_text(size=8), strip_text=element_text(size=8)))
        p3.save(os.path.join(FIGDIR, "A_position_contrast%s.png" % tag), dpi=300, verbose=False)
    figures_by_term(df, order)
    figures_by_term(df, order, roll=1, suffix='_raw')
    print("  wrote figures to %s" % FIGDIR)


def figures_by_term(df, order, roll=None, suffix=''):
    """The transpose: FACET BY TERM, COLOUR BY ARM.

    The arm-faceted figures put the four terms side by side and make the LEVEL
    ordering legible (A|A < A|B < B|B < B|A everywhere). They are the wrong
    layout for the actual question, which is how the arms differ WITHIN a term —
    differences of 0.03-0.07 nats that the level ordering dwarfs. Facetting by
    term puts the comparison inside the panel where the eye can do it.

    `coord_cartesian` ZOOMS rather than clipping, so the undisturbed arm's
    position-1 spike leaves the frame without its rows being dropped from the
    line. A `scale_y_continuous(limits=)` would silently delete them and draw a
    gap the reader would read as missing data.
    """
    import pandas as pd
    from plotnine import (ggplot, aes, geom_line, geom_hline, facet_wrap, labs,
                          theme_bw, theme, element_text, scale_x_continuous,
                          scale_colour_manual, coord_cartesian)
    g = (df.groupby(["arm", "term", "pos"], as_index=False)
           .agg(surprisal=("surprisal", "median"), pairs=("pair", "nunique")))
    g = g[(g["pairs"] >= 30) & (g["pos"] >= MINPOS)]
    g["arm"] = pd.Categorical(g["arm"], categories=[a for a in order if a in set(g["arm"])],
                              ordered=True)
    g = _smooth(g, "surprisal", roll)
    for tag, xmax in (("", 64), ("_full", MAXPOS)):
        p = (ggplot(g, aes("pos", "surprisal", colour="arm"))
             + geom_line(size=0.45)
             + facet_wrap("~term", ncol=2, scales="free_y")
             + scale_x_continuous(limits=(MINPOS, xmax))
             + scale_colour_manual(values=ARM_COLOURS)
             + labs(x="sentence position (forced word at 0, not itself scored)",
                    y="surprisal  (nats, median over 42 pairs)", colour="arm",
                    title="Surprisal by position: one panel per scoring term",
                    subtitle="text|scorer — first letter wrote it, second scored it. "
                             "Free y per panel: the between-term level gap is ~1 nat, "
                             "the between-arm differences ~0.05. "
                             + ("Position 1 dropped; NO SMOOTHING." if (roll or ROLL) <= 1 else
                              "Position 1 dropped; %d-position centred rolling mean." % (roll or ROLL)))
             + theme_bw()
             + theme(figure_size=(11, 6.5), plot_title=element_text(size=11, weight="bold"),
                     plot_subtitle=element_text(size=8), strip_text=element_text(size=9)))
        p.save(os.path.join(FIGDIR, "A_term_facets%s%s.png" % (suffix, tag)), dpi=300, verbose=False)

    base = df[df["arm"] == "non-mover"][["pair", "term", "pos", "surprisal"]]
    base = base.rename(columns={"surprisal": "ref"})
    m = df[df["arm"] != "non-mover"].merge(base, on=["pair", "term", "pos"], how="inner")
    m["delta"] = m["surprisal"] - m["ref"]
    dd = (m.groupby(["arm", "term", "pos"], as_index=False)
            .agg(delta=("delta", "median"), pairs=("pair", "nunique")))
    dd = dd[(dd["pairs"] >= 30) & (dd["pos"] >= MINPOS)]
    dd["arm"] = pd.Categorical(dd["arm"], categories=[a for a in order if a in set(dd["arm"])],
                               ordered=True)
    dd = _smooth(dd, "delta", roll)
    for tag, xmax in (("", 64), ("_full", MAXPOS)):
        p = (ggplot(dd, aes("pos", "delta", colour="arm"))
             + geom_hline(yintercept=0, size=0.3, colour="#666666")
             + geom_line(size=0.45)
             + facet_wrap("~term", ncol=2)
             + scale_x_continuous(limits=(MINPOS, xmax))
             + coord_cartesian(ylim=(-0.15, 0.15))
             + scale_colour_manual(values=ARM_COLOURS)
             + labs(x="sentence position",
                    y="surprisal minus the non-mover arm (nats)", colour="arm",
                    title="Against the matched non-mover: one panel per scoring term",
                    subtitle="above 0 = MORE surprising than a word alignment left alone. "
                             "undisturbed leaves the frame at positions 1-2 (+2.6 nats, the "
                             "definitional cost of committing to any word) — zoomed, not clipped.")
             + theme_bw()
             + theme(figure_size=(11, 6.5), plot_title=element_text(size=11, weight="bold"),
                     plot_subtitle=element_text(size=8), strip_text=element_text(size=9)))
        p.save(os.path.join(FIGDIR, "A_term_facets_contrast%s%s.png" % (suffix, tag)), dpi=300, verbose=False)


if __name__ == "__main__":
    if "--figures-only" in sys.argv and os.path.exists(CACHE):
        rows = json.load(open(CACHE))
        print("  reusing %d cached rows" % len(rows))
    else:
        rows = collect()
    figures(rows)
