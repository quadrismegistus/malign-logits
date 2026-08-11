"""M03 figures: the arm contrast, as verbs and as fields, across the ladder.

    uv run python f_figures.py --data      # compute once, cache to CSV
    uv run python f_figures.py --all       # every figure from the cache
    uv run python f_figures.py --verbs --button --dominance --diverging

Replaces four throwaway scripts. Everything is 300 dpi and every figure is
regenerable from the cached CSVs without touching the store.

## THE MOVING AVERAGE DOES NOT CROSS A PHASE BOUNDARY

The 95 rungs are three separate training runs laid end to end -- 42 pretraining
checkpoints, 43 SFT steps, 7 RLVR steps -- plus three singleton endpoints. A
centred rolling mean over the whole axis would average the base endpoint
together with the first SFT rungs and turn a STEP into a RAMP, which is exactly
the shape the analysis is trying to distinguish (`d_ladder.py` found the arms
separate within SFT's first 3,000 steps; `e_general_vs_institutional.py` found
submissive mass jumping 2.7 pretraining sd at the first SFT rung).

So the window is applied WITHIN `base_step`, `sft_step` and `rlvr_step`
separately, and the three endpoints are never smoothed at all. The raw series
is drawn faintly underneath so the smoothing can be seen rather than trusted.

The rungs are also NOT evenly spaced in training steps -- pretraining runs
0, 1k, 2k ... 1.41M -- so a k-rung window is a window in CHECKPOINT INDEX and
not in training time. Stated because the x-axis invites the other reading.

## WHAT EACH FIGURE IS

    diverging   which verbs separate the arms at the aligned endpoint, with
                the base value beside each so created differences are
                distinguishable from inherited ones
    verbs       the three most institutional and three most individual verbs
                across the ladder, both arms
    button      the four verbs alignment REMOVES and four it INSTALLS
    dominance   Warriner dominance across the ladder, arms against registrar's
                105-pair general corpus

## THE VERB FILTER IS THE SLOT, NOT THE LEXICON

Every prompt ends `I should ___`, so the next word is a lexical verb by
construction. Filtering IN by BYU's `vv*` tag drops `contact`, `file`,
`document`, `appeal`, `report`, `review` and `request`, which are tagged `nn1`
on frequency grounds and are verbs here. So the filter removes what CAN follow
a modal without being lexical -- auxiliaries, modals, adverbs, the negator --
and keeps everything else.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meta", "M05_emergence", "scripts"))

FIG = os.path.join(CAMP, "figures")
RES = os.path.join(CAMP, "results")
D_RUNGS = os.path.join(RES, "f_verb_rungs.csv")
D_ENDS = os.path.join(RES, "f_verb_endpoints.csv")
DPI = 300
WINDOW = 5

BASE = "allenai/Olmo-3-1025-7B"
ALIGNED = "allenai/Olmo-3-7B-Think-DPO"

TOP_VERBS = ["explain", "ensure", "note", "contact", "consider", "file"]
BUTTON = ["quit", "sue", "warn", "remind", "contact", "appeal", "ensure", "explain"]
TRACKED = sorted(set(TOP_VERBS) | set(BUTTON))

#: what can follow a modal without being a lexical verb
AUX = ("vb", "vh", "vd", "vm")
EXTRA = {"not", "n't", "never", "also", "still", "always", "probably", "perhaps",
         "maybe", "just", "now", "first", "then", "only", "really", "simply",
         "otherwise", "instead", "rather", "like"}


def keep(w, byu):
    k = w.strip()
    if len(k) < 2 or not k[0].isalpha():
        return False
    if k.lower() in EXTRA:
        return False
    e = byu.get(k)
    return not (e and (e[1].startswith(AUX) or e[1].startswith("rr")))


def build():
    """Compute both caches. The only step that touches the store."""
    import pandas as pd
    from d_ladder import labels
    from malign_logits.movement import word_probs
    from malign_logits import fields
    import m05_field_flow_fine as FF

    byu = fields._byu()
    L = {t: v for t, v in labels().items() if v[2] == "m03_slice"}
    print("M03 kernel prompts: %d" % len(L))

    rows = []
    for idx, mid, role in FF.population():
        acc = collections.defaultdict(lambda: collections.defaultdict(float))
        n = collections.Counter()
        for t, (arm, scen, s) in L.items():
            wp = word_probs(mid, t)
            if wp is None:
                continue
            n[arm] += 1
            for w in TRACKED:
                acc[arm][w] += wp.probs.get(w, 0.0)
        for arm in ("inst", "indiv"):
            if not n[arm]:
                continue
            for w in TRACKED:
                rows.append(dict(ckpt_idx=idx, role=role, arm=arm, word=w,
                                 mass=acc[arm][w] / n[arm]))
        print("  [%2d] %-40s %s" % (idx, mid.split("/")[-1][:38], role), flush=True)
    pd.DataFrame(rows).to_csv(D_RUNGS, index=False)
    print("wrote %s" % os.path.relpath(D_RUNGS, ROOT))

    #: the endpoint table, every verb rather than the tracked few
    out = {}
    for lab, mid in (("base", BASE), ("aligned", ALIGNED)):
        acc = collections.defaultdict(lambda: collections.defaultdict(float))
        n = collections.Counter()
        for t, (arm, scen, s) in L.items():
            wp = word_probs(mid, t)
            if wp is None:
                continue
            n[arm] += 1
            for w, p in wp.probs.items():
                if keep(w, byu):
                    acc[arm][w] += p
        out[lab] = {a: {w: v / n[a] for w, v in d.items()} for a, d in acc.items()}
    words = set()
    for lab in out:
        for a in out[lab]:
            words |= set(out[lab][a])
    pd.DataFrame([
        dict(word=w,
             base=out["base"]["inst"].get(w, 0) - out["base"]["indiv"].get(w, 0),
             aligned=out["aligned"]["inst"].get(w, 0) - out["aligned"]["indiv"].get(w, 0))
        for w in words]).sort_values("aligned", ascending=False).to_csv(D_ENDS, index=False)
    print("wrote %s" % os.path.relpath(D_ENDS, ROOT))


def smooth(df, value="mass", by=("word", "arm")):
    """Centred rolling mean of WINDOW, WITHIN each multi-rung phase.

    `base_endpoint`, `sft_endpoint` and `dpo_endpoint` are single checkpoints
    and are passed through untouched -- a window over a group of one is the
    value itself, but making that explicit stops a future edit from folding
    them into a neighbouring phase.
    """
    import pandas as pd
    MULTI = {"base_step", "sft_step", "rlvr_step"}
    out = []
    for keys, g in df.groupby(list(by) + ["role"], observed=True):
        g = g.sort_values("ckpt_idx").copy()
        if keys[-1] in MULTI and len(g) >= 3:
            g["smoothed"] = (g[value].rolling(WINDOW, center=True,
                                              min_periods=1).mean())
        else:
            g["smoothed"] = g[value]
        out.append(g)
    return pd.concat(out).sort_values("ckpt_idx")


def _theme(w, h):
    from plotnine import (element_text, theme, theme_minimal)
    return (theme_minimal()
            + theme(figure_size=(w, h), legend_position="top",
                    legend_title=element_text(size=0),
                    plot_title=element_text(size=13, weight="bold"),
                    plot_subtitle=element_text(size=8.5, colour="#52514e"),
                    strip_text=element_text(size=10, weight="bold"),
                    axis_title=element_text(size=8)))


XLAB = ("checkpoint (0-41 pretraining | 42 base end | 43-86 SFT | 87 DPO | 88-94 RLVR)")
ARMCOL = {"institutional": "#1f4e79", "individual": "#b3391c"}
BOUNDS = [42.5, 86.5, 87.5]


def ladder_figure(words, order, title, subtitle, out, ncol, labeller=None,
                  size=(13, 6.8)):
    import pandas as pd
    from plotnine import (aes, facet_wrap, geom_line, geom_vline, ggplot, labs,
                          scale_colour_manual)
    D = pd.read_csv(D_RUNGS)
    D = D[D.word.isin(words)].copy()
    D["arm"] = D.arm.map({"inst": "institutional", "indiv": "individual"})
    D = smooth(D)
    D["facet"] = pd.Categorical([labeller(w) if labeller else w for w in D.word],
                                categories=[labeller(w) if labeller else w
                                            for w in order], ordered=True)
    p = (ggplot(D, aes("ckpt_idx", colour="arm"))
         + geom_vline(xintercept=BOUNDS, colour="#c4c4c4", linetype="dashed", size=.4)
         + geom_line(aes(y="mass"), alpha=.22, size=.45)
         + geom_line(aes(y="smoothed"), size=.85)
         + facet_wrap("~facet", ncol=ncol, scales="free_y")
         + scale_colour_manual(values=ARMCOL)
         + labs(x=XLAB, y="mean next-word probability over 18 scenarios",
                title=title, subtitle=subtitle)
         + _theme(*size))
    p.save(out, dpi=DPI, verbose=False)
    print("wrote %s (%d dpi)" % (os.path.relpath(out, ROOT), DPI))


def fig_verbs():
    ladder_figure(
        TOP_VERBS, TOP_VERBS,
        "The three most institutional and three most individual verbs",
        ("M03 kernel, 18 scenarios x 2 arms. Faint line raw, solid line a %d-rung "
         "centred mean applied WITHIN each phase\nso the base-to-SFT step is not "
         "smoothed into a ramp." % WINDOW),
        os.path.join(FIG, "arm_verbs_ladder.png"), 3, size=(12, 6.5))


def fig_button():
    lose = set(BUTTON[:4])
    ladder_figure(
        BUTTON, BUTTON,
        "Exit and threat removed, channels and explanation installed",
        ("M03 kernel, 18 scenarios x 2 arms, independent y-axes. Faint line raw, "
         "solid a %d-rung centred mean within phase.\nTop row: removed. Bottom "
         "row: installed." % WINDOW),
        os.path.join(FIG, "button_verbs.png"), 4,
        labeller=lambda w: "%s  (%s)" % (w, "removed" if w in lose else "installed"))


def fig_diverging(n=15):
    import pandas as pd
    from plotnine import (aes, coord_flip, geom_hline, geom_point, geom_segment,
                          ggplot, labs, scale_colour_manual, scale_shape_manual)
    D = pd.read_csv(D_ENDS)
    top = pd.concat([D.head(n), D.tail(n)]).sort_values("aligned")
    top["word"] = pd.Categorical(top.word, categories=top.word, ordered=True)
    M = top.melt(id_vars=["word"], value_vars=["base", "aligned"],
                 var_name="model", value_name="diff")
    M["model"] = pd.Categorical(M.model, categories=["base", "aligned"], ordered=True)
    p = (ggplot(M, aes("word", "diff"))
         + geom_hline(yintercept=0, colour="#0b0b0b", size=.4)
         + geom_segment(data=top, mapping=aes(x="word", xend="word",
                                              y="base", yend="aligned"),
                        colour="#cccccc", size=1.6)
         + geom_point(aes(colour="model", shape="model"), size=2.8)
         + coord_flip()
         + scale_colour_manual(values={"base": "#9a9a9a", "aligned": "#1f4e79"})
         + scale_shape_manual(values={"base": "o", "aligned": "D"})
         + labs(x="", y="mass(institutional) - mass(individual), pooled over 18 scenarios",
                title="Lexical verbs separating the two speakers",
                subtitle=("Olmo-3-1025-7B to Olmo-3-7B-Think-DPO. Every prompt ends "
                          '"I should ___", so the slot is a verb by construction:\n'
                          "auxiliaries, modals, adverbs and the negator are removed, and "
                          "nn1-tagged verbs like contact/file/appeal are KEPT."))
         + _theme(10, 8.5))
    p.save(os.path.join(FIG, "arm_verbs.png"), dpi=DPI, verbose=False)
    print("wrote %s (%d dpi)" % (os.path.relpath(os.path.join(FIG, "arm_verbs.png"), ROOT), DPI))


def fig_dominance():
    import pandas as pd
    from plotnine import (aes, facet_wrap, geom_line, geom_vline, ggplot, labs,
                          scale_colour_manual)
    A = pd.read_parquet(os.path.join(RES, "e_field_flow_arms.parquet"))
    G = pd.read_parquet(os.path.join(ROOT, "data", "m05_field_flow_fine.parquet"))
    FIELDS = ["NORM: dominance=dominant", "NORM: dominance=submissive",
              "NORM: dominance=neutral"]
    role = A[["ckpt_idx", "role"]].drop_duplicates()
    rows = []
    for f in FIELDS:
        g = G[G.field == f].groupby("ckpt_idx").mass.median().rename("mass").reset_index()
        g["corpus"] = "general (105 transgressive/neutral)"; g["word"] = f
        rows.append(g)
        for arm, lab in (("inst", "institutional arm (18)"),
                         ("indiv", "individual arm (18)")):
            a = (A[(A.field == f) & (A.arm == arm)].groupby("ckpt_idx").mass
                 .median().rename("mass").reset_index())
            a["corpus"] = lab; a["word"] = f
            rows.append(a)
    D = pd.concat(rows).merge(role, on="ckpt_idx", how="left")
    D = smooth(D, by=("word", "corpus"))
    D["facet"] = D.word.str.replace("NORM: dominance=", "dominance = ", regex=False)
    p = (ggplot(D, aes("ckpt_idx", colour="corpus"))
         + geom_vline(xintercept=BOUNDS, colour="#c4c4c4", linetype="dashed", size=.4)
         + geom_line(aes(y="mass"), alpha=.20, size=.45)
         + geom_line(aes(y="smoothed"), size=.8)
         + facet_wrap("~facet", ncol=1, scales="free_y")
         + scale_colour_manual(values={"general (105 transgressive/neutral)": "#8a8a8a",
                                       "institutional arm (18)": "#1f4e79",
                                       "individual arm (18)": "#b3391c"})
         + labs(x=XLAB, y="field mass (median over prompts)",
                title="Dominance across the ladder, by corpus",
                subtitle=("Warriner dominance, trichotomised at the lexicon's own tertiles. "
                          "Faint raw, solid a %d-rung centred\nmean within phase. The bins are "
                          "thirds of English: `have` and `say` are top-tertile, `check` and "
                          "`file` bottom." % WINDOW))
         + _theme(11, 8.5))
    p.save(os.path.join(FIG, "dominance_ladder.png"), dpi=DPI, verbose=False)
    print("wrote %s (%d dpi)" % (os.path.relpath(os.path.join(FIG, "dominance_ladder.png"), ROOT), DPI))


def main():
    ap = argparse.ArgumentParser()
    for f in ("data", "all", "verbs", "button", "diverging", "dominance"):
        ap.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if a.data:
        return build()
    if not os.path.exists(D_RUNGS):
        raise SystemExit("no cache; run --data first")
    if a.all or a.diverging:
        fig_diverging()
    if a.all or a.verbs:
        fig_verbs()
    if a.all or a.button:
        fig_button()
    if a.all or a.dominance:
        fig_dominance()
    if not any((a.all, a.verbs, a.button, a.diverging, a.dominance)):
        ap.print_help()


if __name__ == "__main__":
    main()
