"""Does the residualised displacement axis order T's categories the way T does?

    uv run --with lemminflect python v_axis_vs_fields.py

Findings V.5's axis, with log frequency residualised out (`v_displacement_vector
--verbs --resid`), runs from bodily action to cognition:

    from  put, sat, got, turn, lay, spit, putting, shut, puke, threw, rub, pull
    to    considered, appreciate, discovered, consider, explore, understand,
          examine, encourage, determine, noticed, realized, identified

That is a reading of 36 words out of 1,312, which is an anecdote. Findings T
made the same claim quantitatively -- alignment moves language off contact,
motion and force onto perception, cognition and speech, on six lexicons sharing
no design -- so the two can be put against each other.

THE TEST. Per lexicon, per category: T's marginal delta (category share among
risers minus among fallers, averaged over 43 edges) against the mean position of
that category's verbs on the axis. Spearman over categories.

  UNIT: the category. Words within a category share one delta, so a word-level
  correlation is pseudo-replicated and inflated; it is printed as description
  only, never as the test.

  SIGN. delta > 0 means the category rises. proj > 0 is the riser side of
  mean(risers) - mean(fallers). Both are riser-positive, so AGREEMENT IS A
  POSITIVE RHO. A negative rho is not a weak result, it is a contradiction
  between the two instruments and calls for a sign audit before anything else.

  OUTCOMES, all three.
    rho > 0 and clears its MDE -- the geometry reproduces T's ordering with no
      lexicon in it. Two instrument families, no shared design.
    rho ~ 0 -- no agreement. The pole reading is a geometric statement standing
      on its own and must not be described as confirming T.
    rho < 0 -- the instruments point opposite ways. Audit the sign convention
      in both before reporting anything.

  AND THE COMPARISON THAT MATTERS: the raw axis is scored identically. If
  residualisation IMPROVES agreement, frequency was noise on a real semantic
  axis. If it degrades it, the raw agreement was partly frequency doing the
  work. Either way the pair of numbers is the result, not the resid one alone.

MDE is reported for every lexicon because the small ones cannot detect much:
at n=15 categories Spearman needs |rho| > 0.51 to clear p<0.05, so a real but
moderate agreement there would read as nothing.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
sys.path.insert(0, HERE)

MARG = os.path.join(OUT, "s_everything_marginal.csv")
#: PRIMARY is the reported minimum: it retains the most categories and the full
#: table is printed, so the disagreements are visible. The higher minimums are a
#: sensitivity, and they read as improvement for a reason that is not agreement
#: improving -- the categories they drop are the small disagreeing ones.
MINIMUMS = (3, 5, 10, 20)
PRIMARY = 5


def labelers():
    import s_category_crosstab as C
    import s_lexicon_crosstab as L
    ind = C.induced_labels()
    return {
        "induced": lambda t: {w: ind[w] for w in t if w in ind},
        "wordnet": C.wordnet_labels,
        "verbnet": L.verbnet_labels,
        "usas": L.usas_labels,
        "framenet": L.framenet_labels,
        "rid": L.rid_labels,
    }


def flatten(lab):
    """label maps are word -> str or word -> iterable; emit (word, category).

    The lexicon labellers return (map, n_multi) or (map, n_multi, procs) while
    the category ones return a bare dict, so unwrap before reading.
    """
    if isinstance(lab, tuple):
        lab = lab[0]
    out = []
    for w, v in lab.items():
        if v is None:
            continue
        if isinstance(v, str):
            out.append((w, v))
        else:
            out.extend((w, c) for c in v if c)
    return pd.DataFrame(out, columns=["word", "category"])


def mde_spearman(n, alpha=0.05):
    """smallest |rho| reaching alpha two-tailed at this n, via the t approximation."""
    if n < 5:
        return float("nan")
    tcrit = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(tcrit / np.sqrt(n - 2 + tcrit ** 2))


def main():
    M = pd.read_csv(MARG)
    M = M[M["stratum"] == "ALL"][["labeling", "category", "delta", "n_edges", "p"]]

    #: THREE ARMS. `resid` and `raw` are V.6's, over 1,312 types. `fq_conc`
    #: also removes concreteness and lives on 1,301 types (11 unscored even
    #: after lemma fallback); its same-population freq-only control agrees with
    #: V.6 at Spearman 0.9993, so the 11 are not a confound and the arms are
    #: comparable as they stand.
    FILES = {"resid": "_resid", "raw": "", "fq_conc": "_resid_freq_conc"}
    axes = {}
    for tag, suf in FILES.items():
        f = os.path.join(OUT, "v_axis_projection_verbs%s.csv" % suf)
        if os.path.exists(f):
            axes[tag] = pd.read_csv(f).set_index("word")["proj"]
    print("axis projections: " + ", ".join("%s %d" % (t, len(v)) for t, v in axes.items()) + "\n")

    #: label once per lexicon and reuse across type minimums. verbnet and
    #: framenet each load an nltk corpus, so relabelling per minimum would
    #: quadruple the runtime for identical output.
    LAB = {}
    for name, fn in labelers().items():
        if not len(M[M["labeling"] == name]):
            print("  %-9s no marginals under this labeling, skipped" % name)
            continue
        lab = flatten(fn(list(axes["resid"].index)))
        if not len(lab):
            print("  %-9s labeller returned nothing for these verbs, skipped" % name)
            continue
        LAB[name] = lab

    rows, tables = [], {}
    for mt in MINIMUMS:
        for name, lab in LAB.items():
            sub = M[M["labeling"] == name]
            rec = {"lexicon": name, "min_types": mt}
            for tag, proj in axes.items():
                L = lab[lab["word"].isin(proj.index)].copy()
                L["proj"] = L["word"].map(proj)
                agg = L.groupby("category").agg(mean_proj=("proj", "mean"), n_types=("word", "nunique"))
                agg = agg[agg["n_types"] >= mt]
                J = sub.merge(agg, left_on="category", right_index=True)
                #: A ZERO JOIN IS A KEY MISMATCH, NOT AN ABSENCE. The `gi:` vs
                #: `gi_primary:` prefix already produced one silent all-zero result
                #: in this campaign, so both cardinalities are printed whenever the
                #: overlap is poor rather than letting it read as "too few".
                if tag == "resid" and mt == PRIMARY:
                    hit = len(set(agg.index) & set(sub["category"]))
                    if hit < 0.5 * min(len(agg), sub["category"].nunique()):
                        print("  %-9s JOIN CHECK: %d labeller categories over the type minimum, "
                              "%d in the marginals, %d matched -- suspect a key mismatch"
                              % (name, len(agg), sub["category"].nunique(), hit))
                if len(J) < 5:
                    rec["n_%s" % tag] = len(J)
                    continue
                rho, p = stats.spearmanr(J["delta"], J["mean_proj"])
                rec.update({"n_%s" % tag: len(J), "rho_%s" % tag: rho, "p_%s" % tag: p,
                            "mde_%s" % tag: mde_spearman(len(J))})
                if tag == "resid" and mt == PRIMARY:
                    tables[name] = J.sort_values("mean_proj")
            rows.append(rec)

    ALL = pd.DataFrame(rows)
    ALL.to_csv(os.path.join(OUT, "v_axis_vs_fields.csv"), index=False)
    R = ALL[ALL["min_types"] == PRIMARY]

    print("=" * 94)
    print("SPEARMAN: T's marginal delta vs mean position on the displacement axis")
    print("  unit = category, >=%d verb types each. positive rho = the two agree." % PRIMARY)
    print("=" * 94)
    star = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print("  %-9s %5s | %-20s | %-20s | %-20s"
          % ("lexicon", "cats", "resid FREQ", "resid FREQ+CONC", "raw"))
    for _, r in R.iterrows():
        if "rho_resid" not in r or pd.isna(r.get("rho_resid")):
            print("  %-9s %5s | too few categories over the type minimum" % (r["lexicon"], r.get("n_resid", 0)))
            continue
        cell = lambda t: ("rho %+.3f p %.4f %-3s" % (r["rho_%s" % t], r["p_%s" % t], star(r["p_%s" % t]))
                          if ("rho_%s" % t) in r and pd.notna(r.get("rho_%s" % t)) else "%-20s" % "not available")
        print("  %-9s %5d | %s | %s | %s"
              % (r["lexicon"], r["n_resid"], cell("resid"), cell("fq_conc"), cell("raw")))
    ok = R.dropna(subset=["rho_resid"])
    if len(ok):
        extra = (", freq+conc %+.3f" % ok["rho_fq_conc"].median()) if "rho_fq_conc" in ok else ""
        print("\n  %d of %d lexicons positive on the freq-residualised axis; median rho %+.3f (raw %+.3f%s)"
              % (int((ok["rho_resid"] > 0).sum()), len(ok), ok["rho_resid"].median(),
                 ok["rho_raw"].median(), extra))
        if "rho_fq_conc" in ok:
            print("  freq+conc arm: %d of %d positive. CONCRETENESS IS NOT THE AXIS if this holds --"
                  % (int((ok["rho_fq_conc"] > 0).sum()), len(ok.dropna(subset=["rho_fq_conc"]))))
            print("  the ordering agrees with the freq-only arm at Spearman 0.878 over 1,301 shared types.")

    print("\n  SENSITIVITY to the type minimum. The rise is NOT agreement improving: at the")
    print("  higher minimums only a handful of categories survive and the ones dropped are the")
    print("  small disagreeing ones, so the exceptions leave and the median follows.")
    print("  %9s %6s %10s %10s %8s" % ("min_types", "lexes", "med resid", "med raw", "cats"))
    for mt, g in ALL.groupby("min_types"):
        h = g.dropna(subset=["rho_resid"])
        if not len(h):
            continue
        print("  %9d %6s %+10.3f %+10.3f %8d"
              % (mt, "%d/%d+" % (int((h["rho_resid"] > 0).sum()), len(h)),
                 h["rho_resid"].median(), h["rho_raw"].median(), int(h["n_resid"].sum())))

    for name in ("wordnet", "induced"):
        if name not in tables:
            continue
        print("\n" + "=" * 94)
        print("%s, every category: where it sits on the axis and which way T says it moves" % name.upper())
        print("=" * 94)
        t = tables[name]
        print("  %-26s %9s %9s %7s %6s" % ("category", "mean_proj", "T delta", "T p", "types"))
        for _, r in t.iterrows():
            print("  %-26s %+9.4f %+9.4f %7.4f %6d"
                  % (str(r["category"])[:26], r["mean_proj"], r["delta"], r["p"], r["n_types"]))

    print("\nwrote v_axis_vs_fields.csv")


if __name__ == "__main__":
    main()
