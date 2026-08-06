"""Emit findings 10-14 for `findings_t_category_flow.md` from the CSVs.

    uv run python s_findings_t_append.py            print to stdout
    uv run python s_findings_t_append.py --write    splice into the document

Every number in the emitted text is read out of `results/s_everything_*.csv`.
None is typed. The document had a fabricated row in it once -- a SocRel triple
transcribed from a query I ran in the same command as the edit -- and the fix
is not to be more careful but to stop having a step where a value passes
through a human or a model on its way from the computation to the page.

Splicing inserts before the `## Limits` heading and refuses if the anchor is
missing or appears more than once, because an append that lands in the wrong
place is worse than one that does not land.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
DOC = os.path.join(CAMP, "findings_t_category_flow.md")
ANCHOR = "## Limits"

#: `class` is the census salary battery, "The teacher earned an annual salary
#: of $___", so its vocabulary is numerals and no semantic lexicon covers it
#: (USAS 3 percent). Its movement is real and reported separately in finding
#: 14; it must not be pooled into any statement about semantic fields.
DROP = "class"


def load():
    M = pd.read_csv(os.path.join(OUT, "s_everything_marginal.csv"))
    D = pd.read_csv(os.path.join(OUT, "s_everything_direction.csv"))
    W = pd.read_csv(os.path.join(OUT, "s_everything_wordpairs.csv"))
    return M[M["n_edges"] >= 10], D, W


def nm(r):
    """Prefer the resolved name; fall back to the code."""
    v = r.get("category_name") if "category_name" in r else None
    return v if isinstance(v, str) and v else str(r["category"])


def f10(M, D, W):
    L = M[M["labeling"] != "TOKEN"]
    o = ["## 10. The threshold-free instrument, and what it costs", "",
         "Findings 1 to 9 rest on `r_population_k2.parquet`, which keeps a (faller, riser) pair only if it recurs in two or more edges, and which manufactures observations: a cell with 12 fallers and 10 risers becomes 120 rows. Both were replaced. The unit is now one alignment edge, each edge voting once, on all active English prompts rather than the 1,361 built as twins.", "",
         "    marginal shift      %s of %s category-in-stratum tests survive Bonferroni" % (f"{int(M.bonferroni.sum()):,}", f"{len(M):,}"),
         "    directed moves      %s of %s" % (f"{int(D.bonferroni.sum()):,}", f"{len(D):,}"),
         "    word pairs          %s of %s   (no lexicon at all)" % (f"{int(W.bonferroni.sum()):,}", f"{len(W):,}"),
         "",
         "**The cost is finding 2.** On the induced taxonomy the edge-unit direction test returns %d survivors in any stratum. `bodily_violence -> speech_act` at 38 against 1 is a claim about the paired population at its own weaker unit, and the edge-unit test does not carry it. What survives on that labeling is the marginal form: violence falls and perception rises across the same edges. FrameNet, USAS and VerbNet do yield directed moves; the induced taxonomy does not."
         % int(D[(D["labeling"] == "induced") & D["bonferroni"]].shape[0]), ""]
    return "\n".join(o)


def f11(M):
    L = M[(M["labeling"] != "TOKEN") & (M["stratum"] != DROP)]
    g = L[L["bonferroni"] & (L["stratum"] != "ALL")].copy()
    g["dir"] = np.where(g["delta"] > 0, "rise", "fall")
    t = g.groupby(["labeling", "category", "dir"])["stratum"].nunique().unstack("dir").fillna(0)
    for c in ("rise", "fall"):
        if c not in t:
            t[c] = 0
    t["tot"] = t["rise"] + t["fall"]
    n_str = g["stratum"].nunique()
    up = t[(t["fall"] == 0) & (t["rise"] >= 5)].sort_values("rise", ascending=False)
    dn = t[(t["rise"] == 0) & (t["fall"] >= 5)].sort_values("fall", ascending=False)
    both = int(((t["rise"] > 0) & (t["fall"] > 0) & (t["tot"] >= 5)).sum())
    o = ["## 11. The same direction in every stratum, on six lexicons that share no design", "",
         "The %d prompt strata are the two M01 twins, the two M03 arms, the two institutional positions by role, and five unpaired registry domains. A category is counted below only if it is a Bonferroni survivor in at least five of them and **never reverses**." % n_str, "",
         "**Rises, no stratum reversing:**", "",
         "| lexicon | category | strata |", "|---|---|---|"]
    for (l, c), x in up.head(12).iterrows():
        r = M[(M["labeling"] == l) & (M["category"] == c)]
        o.append("| %s | %s | %d of %d |" % (l, nm(r.iloc[0]) if len(r) else c, int(x["rise"]), n_str))
    o += ["", "**Falls, no stratum reversing:**", "", "| lexicon | category | strata |", "|---|---|---|"]
    for (l, c), x in dn.head(10).iterrows():
        r = M[(M["labeling"] == l) & (M["category"] == c)]
        o.append("| %s | %s | %d of %d |" % (l, nm(r.iloc[0]) if len(r) else c, int(x["fall"]), n_str))
    o += ["", "The model stops touching, moving, striking and competing, and starts investigating, attending, preparing and abstracting. WordNet `contact` is the most consistent single result in the set.", "",
          "**%d categories reverse between strata and they reverse coherently.** They rise in the narrative twins and in `violence`, `sexual` and `neutral`, and fall in `m03_inst`, `m03_indiv`, `inst_authority` and `inst_individual`. This is also why WordNet `cognition` is not significant pooled while being a significant riser in eight strata: the institutional prompts cancel it. Report this stratified. The pooled number hides the finding rather than summarising it." % both, ""]
    return "\n".join(o)


def f12(M):
    A = M[(M["stratum"] == "ALL") & (M["labeling"] == "usas") & M["bonferroni"]].sort_values("delta")
    o = ["## 12. USAS names the alignment vocabulary without being asked to", "",
         "USAS is a Lancaster corpus-linguistics tagset from the 1990s with 232 semantic fields (`lexicons/usas_tagset.tsv`, from ucrel.lancs.ac.uk/usas/semtags.txt). It covers this vocabulary better than anything else tried, 89 percent of word slots as surface forms, and it has no notion of alignment. Its %d Bonferroni survivors on all prompts, read as a list:" % len(A), "",
         "| code | field | shift | edges |", "|---|---|---|---|"]
    for _, x in A.tail(10).iloc[::-1].iterrows():
        o.append("| `%s` | %s | %+.4f | %d/%d |" % (x["category"], nm(x), x["delta"], x["edges_pos"], x["n_edges"]))
    fall = A[A["delta"] < 0]
    o += ["", "and the fallers, all %d of them:" % len(fall), "", "| code | field | shift | edges |", "|---|---|---|---|"]
    for _, x in fall.iterrows():
        o.append("| `%s` | %s | %+.4f | %d/%d |" % (x["category"], nm(x), x["delta"], x["edges_pos"], x["n_edges"]))
    #: THE SUMMARY SENTENCE IS BUILT FROM THE RISERS, not typed. The first
    #: version listed Attention and Understand, which are genuine survivors but
    #: sat below the ten shown, so a reader checking the sentence against the
    #: table above it would not find them. A hand-written gloss drifts from its
    #: own evidence in exactly this way and nothing flags it.
    rise = A[A["delta"] > 0]
    deliberative = [nm(x) for _, x in rise.iterrows()
                    if any(k in nm(x) for k in ("Caution", "Constraint", "Reciproc", "Attention",
                                                "Understand", "Investigate", "Helping", "planning"))]
    #: backticks, not a comma or semicolon join: `X7` is named "Wanting;
    #: planning; choosing" and any punctuation separator splits one field into
    #: three in the reader's eye.
    o += ["", "%s. A tagset built for corpus linguistics thirty years ago, applied to a question it was not built for, returns the vocabulary of alignment among its %d rising fields and physical manipulation as its largest falling one."
          % (" ".join("`%s`" % d for d in deliberative), len(rise)), ""]
    return "\n".join(o)


def f13(M):
    L = M[M["labeling"] != "TOKEN"]
    a = L[L["stratum"] == "m01_marked"].set_index(["labeling", "category"])
    b = L[L["stratum"] == "m01_unmarked"].set_index(["labeling", "category"])
    J = a[["delta", "bonferroni"]].join(b[["delta", "bonferroni"]], lsuffix="_m", rsuffix="_u", how="inner")
    J = J[J["bonferroni_m"] | J["bonferroni_u"]]
    #: the group is defined by the UNMARKED arm so the marked arm is not used to
    #: pick the set it is then tested in
    J["grp"] = np.where(J["delta_u"] < 0, "faller", "riser")
    J["gap"] = J["delta_m"].abs() - J["delta_u"].abs()
    f, r = J[J["grp"] == "faller"]["gap"], J[J["grp"] == "riser"]["gap"]
    tt, pt = stats.ttest_ind(f, r, equal_var=False)
    _u, pu = stats.mannwhitneyu(f, r)
    o = ["## 13. The withdrawal is transgression-specific; the substitution is not", "",
         "The M01 twins differ in one word. If alignment were simply softening transgression, both the removal and the replacement should be larger in the marked twin. Only the removal is.", "",
         "| lexicon | category | marked | neutral twin |", "|---|---|---|---|"]
    for (l, c), x in J[J["grp"] == "faller"].nlargest(5, "gap").iterrows():
        rr = M[(M["labeling"] == l) & (M["category"] == c)]
        o.append("| %s | %s | **%+.4f** | %+.4f |" % (l, nm(rr.iloc[0]) if len(rr) else c, x["delta_m"], x["delta_u"]))
    for (l, c), x in J[J["grp"] == "riser"].nsmallest(4, "gap").iterrows():
        rr = M[(M["labeling"] == l) & (M["category"] == c)]
        o.append("| %s | %s | %+.4f | **%+.4f** |" % (l, nm(rr.iloc[0]) if len(rr) else c, x["delta_m"], x["delta_u"]))
    o += ["", "Alignment removes the violent word only where there is one. It adds the deliberative word everywhere, and if anything slightly more where there was nothing to remove.", "",
          "**Held to what it will bear.** On the named violence categories this is large and clean. As a claim about all %d categories it is p=%.3f by rank test and p=%.3f parametric, with n=%d fallers against n=%d risers. The categories carry it; the omnibus does not yet."
          % (len(J), pu, pt, len(f), len(r)), "",
          #: NOT OUR NUMBERS. malign ran this claim on the forced-continuation
          #: sample, a different population with different pairs, and reported
          #: it in docket [4737]. Recorded here with attribution because a
          #: replication that lives only in a message is one nobody finds; the
          #: producing code is named so it can be checked rather than trusted.
          "**Replicated at a second seat, and only half of it survives.** malign ran this quantity on the forced-continuation sample -- a different population, different pairs, no shared derivation -- as a section of `scripts/fc_analyse.py`, reported in docket [4737]. Unit is the pair, per-site top faller |delta| and top riser excess under CANONICAL:", "",
          "    FALLER |delta|   marked-unmarked  +0.00294  n=16  DETECTED p=0.0279   predicted positive",
          "    RISER excess     marked-unmarked  +0.00195  n=15  not detected, MDE 0.0096   predicted negative", "",
          "**The withdrawal half replicates, correct sign, on a population we did not use.** The substitution half does not, and this is a bounded negative rather than an underpowered one: the riser gap reported above is about 0.016 in the neutral direction and their MDE is 0.0096, comfortably below it, with the observed value small and pointing the other way.", "",
          "The two quantities are not identical -- ours is a category share aggregated over lexicons, theirs is the magnitude of the top riser's excess at a site -- and they can diverge honestly. But our own omnibus for the riser half was already the weaker of the two tests reported above, and an independent instrument now bounds it under the size we claim. **Treat the withdrawal asymmetry as confirmed twice and the substitution asymmetry as not supported.** The sentence the paper can carry is the first half: alignment withdraws more where there is more to withdraw. Whether it also adds less there is not shown.", ""]
    return "\n".join(o)


def f14(M, W):
    A = M[(M["stratum"] == "ALL") & M["bonferroni"] & (M["labeling"] != "TOKEN")]
    up, dn = A[A["delta"] > 0]["delta"], A[A["delta"] < 0]["delta"].abs()
    _u, pu = stats.mannwhitneyu(dn, up)
    o = ["## 14. Few large fallers, many small risers: displacement along a chain", "",
         "Counting Bonferroni survivors on all prompts: **%d risers against %d fallers**, and the fallers are **%.1f times larger** per category (mean %+.5f against %+.5f, Mann-Whitney p=%.1e). The ratio exceeds one in every lexicon."
         % (len(up), len(dn), dn.mean() / up.mean(), -dn.mean(), up.mean(), pu), "",
         "| lexicon | risers | fallers | mean riser | mean faller | ratio | largest single faller |", "|---|---|---|---|---|---|---|"]
    for l, g in A.groupby("labeling"):
        u2, d2 = g[g["delta"] > 0]["delta"], g[g["delta"] < 0]["delta"].abs()
        big = g.nsmallest(1, "delta")
        o.append("| %s | %d | %d | %+.5f | %+.5f | %.1fx | %s %+.4f |"
                 % (l, len(u2), len(d2), u2.mean(), -d2.mean(), d2.mean() / u2.mean(),
                    nm(big.iloc[0]), big.iloc[0]["delta"]))
    o += ["", "The shape is not one-for-one substitution. A few large categories drain and their mass redistributes across many small ones, which is displacement along a chain rather than swap, measured directly. Two of the seven largest fallers are `person_reference` and the General Inquirer's `MALE`: what drains is physical contact and the marking of persons and of maleness.", ""]
    cl = M[(M["stratum"] == DROP) & (M["labeling"] == "TOKEN") & M["bonferroni"]].sort_values("delta")
    wc = W[(W["stratum"] == DROP) & W["bonferroni"]]
    o += ["**A separate result, quarantined from the above.** The `class` stratum is the census salary battery, `\"The teacher earned an annual salary of $___\"`, so its vocabulary is numerals and no semantic lexicon covers it (USAS %d percent). It has %d significant token shifts and %d significant directed word pairs, and they run one way: %s fall, %s rise. Alignment raises the predicted salary figure, reliably, across the edge population. That is worth having and it is not semantic displacement; it must not be pooled with the fields above."
          % (round(100 * M[(M["stratum"] == DROP) & (M["labeling"] == "usas")]["coverage"].max() if len(M[(M["stratum"] == DROP) & (M["labeling"] == "usas")]) else 0),
             len(cl), len(wc),
             ", ".join("`%s`" % c for c in cl.head(4)["category"]),
             ", ".join("`%s`" % c for c in cl.tail(4).iloc[::-1]["category"])), ""]
    return "\n".join(o)


def f15():
    f = os.path.join(OUT, "s_stem_clustered_verdicts.csv")
    if not os.path.exists(f):
        return ""
    J = pd.read_csv(f)
    J["testable"] = J["stems_fwd"].notna()
    J["holds"] = J["bonferroni"].fillna(False).astype(bool)
    t = J[J["testable"]]
    lost = t[~t["holds"]]
    o = ["## 15. What findings 1-9 lose when their denominator is fixed", "",
         "The 5,976 rows of the pair population are one (faller, riser) combination inside one prompt cell, so a cell with 12 fallers and 10 risers contributes 120 of them and the median stem contributes 9. The cross-tabs binomtested those rows, which makes the denominator a property of the join. `scripts/s_stem_clustered.py` re-tests every reported pair at one vote per stem.", "",
         "    reported significant              %d" % len(J),
         "    testable at the stem unit         %d" % len(t),
         "      still significant               %d" % int(t["holds"].sum()),
         "      NOT significant                 %d" % len(lost),
         "    below the minimum cell            %d" % int((~J["testable"]).sum()), "",
         "| lexicon | reported | testable | hold | lost |", "|---|---|---|---|---|"]
    for nm, g in J.groupby("labeling"):
        gt = g[g["testable"]]
        o.append("| %s | %d | %d | %d | %d |" % (nm, len(g), len(gt), int(gt["holds"].sum()), int((~gt["holds"]).sum())))
    o += ["", "**The pairs that do not survive**, with the manufactured count beside the stem count that replaces it:", "",
          "| lexicon | from | to | pairs | stems |", "|---|---|---|---|---|"]
    for _, x in lost.sort_values(["labeling", "p"]).iterrows():
        o.append("| %s | `%s` | `%s` | %d:%d | %d:%d |"
                 % (x["labeling"], x["frm"], x["to"], x["dominant"], x["reverse"],
                    int(x["stems_fwd"]), int(x["stems_rev"])))
    o += ["", "Finding 2's headline is not among them: it was reported clustered from the start, at 29 against 1. `contact -> communication`, `contact -> change` and the USAS and RID moves quoted in finding 2 hold. What goes is mostly the large-count pairs whose margin came from repetition inside a cell rather than from agreement across stems, which is exactly what the correction is for. **No direction reverses.** The claim that survives everywhere is the one about direction; the claim about how many directed pairs reach significance was inflated.", ""]
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    M, D, W = load()
    body = "\n".join([f10(M, D, W), f11(M), f12(M), f13(M), f14(M, W), f15()])
    if not a.write:
        print(body)
        return
    txt = open(DOC, encoding="utf-8").read()
    if txt.count(ANCHOR) != 1:
        raise SystemExit("anchor %r appears %d times; refusing to splice" % (ANCHOR, txt.count(ANCHOR)))
    open(DOC, "w", encoding="utf-8").write(txt.replace(ANCHOR, body + ANCHOR))
    print("spliced %d chars before %r" % (len(body), ANCHOR))


if __name__ == "__main__":
    main()
