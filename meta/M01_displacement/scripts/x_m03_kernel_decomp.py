"""Is T 17's institutional result a POSITION effect, or is it person and modal?

    uv run python x_m03_kernel_decomp.py

T 17b-bis reports signed valence rising for the individual (+0.1065, p=1.1e-03)
and flat for the institution (+0.0059, p=0.83), and calls the sign
position-specific. **That contrast pools a crossed design into two cells.** The
M03 speaker kernel is 18 scenarios x 14 cells, and the 14 are

    POSITION  indiv | inst          the scene rewritten from the other side
    PERSON    I | we
    MODAL     absent | final | final_ought | medial

**THE MODAL LEVEL NAMES ARE A MISNOMER AND IT MATTERS.** Nothing is medial. Read
off the stems, uniform across all 36 prompts at every level:

    absent        "... and I"                 site is a FINITE VERB slot
    final         "... and I should"          site is a bare infinitive
    final_ought   "... and I ought to"        site is a bare infinitive
    medial        "... and I should probably" bare infinitive after a HEDGE

So `absent` is a DIFFERENT GRAMMATICAL SITE from the other three and any
absent-vs-modal difference is partly a fact about English. `final` against
`medial` is not: same site, same modal, one added word. That contrast is the
clean one and it gets its own section.

`x_m03_pov_fields.py` and `x_m03_affect.py` both match on `m03_([NC]\\d+)_(indiv|
inst)_` and DISCARD the rest of the id, which is where person and modal live. So
"individual" in T 17 is an average over four ways of saying I and three of saying
we, and M03's own attribution constraint -- that DOMAIN x MODAL x PERSON x
SPEECH-ACT are four entangled variables -- is exactly what the kernel was built
to separate and has not yet been used to separate.

**THE WITHIN-POSITION CONTRASTS ARE MINIMAL PAIRS AND THE POSITION CONTRAST IS
NOT.** `m03_N1_indiv_I_final` and `m03_N1_indiv_I_absent` are one scene with one
thing changed. `m03_N1_indiv_*` and `m03_N1_inst_*` are two scenes. T says so
about the position contrast and is right; the consequence nobody drew is that
person and modal are cleaner than the contrast already published, not dirtier.

**PERSON AND MODAL ARE NOT FULLY CROSSED.** `final_ought` exists for `I` only --
7 forms per position, not 8. Any modal comparison spanning all four levels is
therefore within `I`, and any person comparison is restricted to the three shared
levels. Run otherwise, a person effect and the ought-form are the same column.

UNIT: the edge, and the statistic is `x_m03_affect.test` reused verbatim -- riser
mean minus faller mean per edge with >=5 of each, one-sample t across edges with
>=10 of them. Section 0 reproduces T's published pair before anything is split;
if it does not reproduce, nothing below is worth reading.

**AROUSAL IS THE BUILT-IN CONTROL.** T finds arousal falling in BOTH positions
(-0.1761 and -0.1459, both p<1e-6), so it is the campaign's example of a
non-position-specific effect. A 14-way split of a 2-cell design will produce
scatter whatever is there; if arousal scatters as widely as valence across the
cells, the valence pattern is what the split does and not what the kernel finds.

COVERAGE. The walk holds 10 of the kernel's 18 scenarios -- C1-C4, N1-N3, N5-N7.
U1-U8 (112 prompts) are in the twp store and not in the movement population, so
this is the same 140 prompts T analysed, cut 14 ways instead of 2.

EXPLORATORY. T's pooled pair is the confirmatory contrast and this does not
replace it. Fourteen cells and three contrast families are a lot of tests on one
population; every p below is uncorrected and the Bonferroni divisor is printed
with the tables so the reader can apply it.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

FORMS = ["I_absent", "I_final", "I_final_ought", "I_medial",
         "we_absent", "we_final", "we_medial"]
SHARED = ["absent", "final", "medial"]          #: the levels `we` also has


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats
    import m01_norms as N

    norms, _, _ = N.load_norms(verify=True)
    V = dict(norms[("en", "valence", "primary")])
    A = dict(norms[("en", "arousal", "primary")])

    W = pd.read_parquet(os.path.join(CAMP, "results", "movement_words.parquet"))
    D = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    pid = {r["prompt"]: str(r.get("prompt_id") or "")
           for r in D if r.get("status") == "ACTIVE" and r.get("prompt")}
    dom = {str(r.get("prompt_id") or ""): r.get("domain") for r in D}
    W["pid"] = W.prompt.map(pid)

    #: the full id, not the prefix. This is the whole point of the script.
    pat = re.compile(r"^m03_([NC]\d+)_(indiv|inst)_(I|we)_(absent|final|final_ought|medial)$")
    ex = W.pid.fillna("").str.extract(pat)
    m = W[ex[0].notna()].copy()
    m["scen"], m["pov"], m["person"], m["modal"] = (ex[0].dropna(), ex[1].dropna(),
                                                    ex[2].dropna(), ex[3].dropna())
    m["form"] = m.person + "_" + m.modal
    m["v"], m["a"] = m.word.map(V), m.word.map(A)

    #: the prefix regex the two published producers use, to show this selects the
    #: same rows they do and differs only in keeping the tail.
    old = W[W.pid.fillna("").str.match(r"m03_([NC]\d+)_(indiv|inst)_")]
    print("rows: %d under the full-id regex, %d under the published prefix regex%s"
          % (len(m), len(old), "  SAME" if len(m) == len(old) else "  *** DIFFERENT ***"))
    print("valence coverage %.0f%% of moved words, %d edges, %d prompts, %d scenarios"
          % (100 * m.v.notna().mean(), m.edge.nunique(), m.pid.nunique(), m.scen.nunique()))
    print("scenarios: %s" % " ".join(sorted(set(m.scen))))

    def per_edge(sub, col):
        """riser mean minus faller mean, per edge. `x_m03_affect.test`, unpooled."""
        out = {}
        for e, g in sub.groupby("edge"):
            x = g[g.role == "riser"][col].dropna()
            y = g[g.role == "faller"][col].dropna()
            if len(x) >= 5 and len(y) >= 5:
                out[e] = x.mean() - y.mean()
        return pd.Series(out, dtype=float)

    def one(sub, col):
        d = per_edge(sub, col)
        if len(d) < 10:
            return None
        return len(d), float(d.mean()), float(stats.ttest_1samp(d.values, 0)[1])

    print("\n" + "=" * 88)
    print("0. REPRODUCE T 17b-bis BEFORE SPLITTING ANYTHING")
    print("=" * 88)
    print("   %-12s %-34s %s" % ("", "this script", "T 17b-bis as published"))
    TPUB = {("indiv", "v"): (+0.1065, 1.1e-03), ("inst", "v"): (+0.0059, 0.83),
            ("indiv", "a"): (-0.1761, 7.6e-07), ("inst", "a"): (-0.1459, 1.7e-07)}
    ok = True
    for col, nm in (("v", "signed valence"), ("a", "arousal")):
        for pov in ("indiv", "inst"):
            r = one(m[m.pov == pov], col)
            pub = TPUB[(pov, col)]
            hit = r is not None and abs(r[1] - pub[0]) < 5e-4
            ok = ok and hit
            print("   %-14s %-6s n=%-3d %+.4f  p %.1e     %+.4f  p %.1e   %s"
                  % (nm, pov, r[0], r[1], r[2], pub[0], pub[1], "ok" if hit else "*** MISMATCH ***"))
    if not ok:
        raise SystemExit("published pair did not reproduce; nothing below is readable")

    print("\n" + "=" * 88)
    print("1. THE FOURTEEN CELLS. Arousal is the control column, not a second result.")
    print("=" * 88)
    print("   %-6s %-14s %5s %20s %20s" % ("pov", "form", "edges", "signed valence", "arousal"))
    cells = {}
    for pov in ("indiv", "inst"):
        for f in FORMS:
            sub = m[(m.pov == pov) & (m.form == f)]
            rv, ra = one(sub, "v"), one(sub, "a")
            cells[(pov, f)] = (rv, ra)
            fmt = lambda r: ("%+.4f p %.1e%s" % (r[1], r[2], "*" if r[2] < 0.05 else " ")
                             if r else "        -         ")
            print("   %-6s %-14s %5s %20s %20s"
                  % (pov, f, rv[0] if rv else "-", fmt(rv), fmt(ra)))
    print("   * = p < 0.05 uncorrected; 28 tests in this table, Bonferroni alpha = %.2e" % (0.05 / 28))

    #: SPREAD. If the split scatters arousal as much as valence, the split is
    #: doing the work. Compared on the same 14 cells, same edges, same estimator.
    sv = np.array([c[0][1] for c in cells.values() if c[0]])
    sa = np.array([c[1][1] for c in cells.values() if c[1]])
    print("\n   cell-to-cell spread across the 14        valence sd %.4f   arousal sd %.4f"
          % (sv.std(ddof=1), sa.std(ddof=1)))
    print("   range                                    valence %+.4f..%+.4f   arousal %+.4f..%+.4f"
          % (sv.min(), sv.max(), sa.min(), sa.max()))

    print("\n" + "=" * 88)
    print("2. DOES THE POSITION GAP HOLD AT EVERY SPEAKER FORM?")
    print("=" * 88)
    print("   T's claim is indiv positive, inst flat. Paired across the edges both")
    print("   cells share -- T ran two independent one-samples and did not test the gap.\n")
    print("   AND arousal is run the same way, because a gap that appears on both norms")
    print("   is an affect gap and only a gap specific to valence is a gap about SIGN.\n")
    print("   %-14s %5s %11s %11s %11s %10s   %11s %10s"
          % ("form", "edges", "indiv", "inst", "gap", "p(paired)", "arousal gap", "p"))
    gaps, agaps = [], []
    for f in FORMS:
        row = {}
        for col in ("v", "a"):
            a = per_edge(m[(m.pov == "indiv") & (m.form == f)], col)
            b = per_edge(m[(m.pov == "inst") & (m.form == f)], col)
            k = a.index.intersection(b.index)
            row[col] = (None if len(k) < 10 else
                        (len(k), a[k].mean(), b[k].mean(), (a[k] - b[k]).mean(),
                         float(stats.ttest_rel(a[k].values, b[k].values)[1])))
        rv, ra = row["v"], row["a"]
        if rv is None:
            print("   %-14s   (fewer than 10 shared edges, not tested)" % f)
            continue
        gaps.append(rv[3])
        if ra:
            agaps.append(ra[3])
        print("   %-14s %5d %+11.4f %+11.4f %+11.4f %10.1e%s  %+11.4f %10.1e%s"
              % (f, rv[0], rv[1], rv[2], rv[3], rv[4], " *" if rv[4] < 0.05 else "  ",
                 ra[3] if ra else float("nan"), ra[4] if ra else float("nan"),
                 " *" if ra and ra[4] < 0.05 else "  "))
    if gaps:
        g, ag = np.array(gaps), np.array(agaps)
        sg = stats.binomtest(int((g > 0).sum()), len(g), 0.5).pvalue
        sa = stats.binomtest(int((ag > 0).sum()), len(ag), 0.5).pvalue
        print("\n   VALENCE gap positive at %d of %d forms, mean %+.4f, sd %.4f, sign test p %.4f"
              % (int((g > 0).sum()), len(g), g.mean(), g.std(ddof=1), sg))
        print("   AROUSAL gap positive at %d of %d forms, mean %+.4f, sd %.4f, sign test p %.4f"
              % (int((ag > 0).sum()), len(ag), ag.mean(), ag.std(ddof=1), sa))
        print("   * = p < 0.05 uncorrected, %d tests per norm, Bonferroni alpha = %.2e"
              % (len(gaps), 0.05 / len(gaps)))

    print("\n" + "=" * 88)
    print("3. PERSON. I against we, within position, on the three SHARED modal levels.")
    print("=" * 88)
    print("   `final_ought` is excluded here because `we` does not have it -- including")
    print("   it would put the ought-form and the person contrast in one column.\n")
    print("   %-6s %-10s %5s %11s %11s %11s %10s"
          % ("pov", "modal", "edges", "I", "we", "I - we", "p(paired)"))
    for pov in ("indiv", "inst"):
        for md in SHARED:
            a = per_edge(m[(m.pov == pov) & (m.person == "I") & (m.modal == md)], "v")
            b = per_edge(m[(m.pov == pov) & (m.person == "we") & (m.modal == md)], "v")
            k = a.index.intersection(b.index)
            if len(k) < 10:
                print("   %-6s %-10s %5d   (not tested)" % (pov, md, len(k)))
                continue
            p = float(stats.ttest_rel(a[k].values, b[k].values)[1])
            print("   %-6s %-10s %5d %+11.4f %+11.4f %+11.4f %10.1e%s"
                  % (pov, md, len(k), a[k].mean(), b[k].mean(), (a[k] - b[k]).mean(), p,
                     "  *" if p < 0.05 else ""))

    print("\n" + "=" * 88)
    print("4. MODAL. Within person=I, where all four levels exist. Each against `absent`.")
    print("=" * 88)
    print("   absent = bare pronoun, a FINITE VERB slot. The other three are bare")
    print("   infinitives. So this table is partly a fact about English; section 4b is not.\n")
    print("   %-6s %-14s %5s %11s %11s %11s %10s"
          % ("pov", "modal", "edges", "that modal", "absent", "difference", "p(paired)"))
    for pov in ("indiv", "inst"):
        base = per_edge(m[(m.pov == pov) & (m.person == "I") & (m.modal == "absent")], "v")
        for md in ("final", "final_ought", "medial"):
            a = per_edge(m[(m.pov == pov) & (m.person == "I") & (m.modal == md)], "v")
            k = a.index.intersection(base.index)
            if len(k) < 10:
                print("   %-6s %-14s %5d   (not tested)" % (pov, md, len(k)))
                continue
            p = float(stats.ttest_rel(a[k].values, base[k].values)[1])
            print("   %-6s %-14s %5d %+11.4f %+11.4f %+11.4f %10.1e%s"
                  % (pov, md, len(k), a[k].mean(), base[k].mean(), (a[k] - base[k]).mean(), p,
                     "  *" if p < 0.05 else ""))

    print("\n" + "=" * 88)
    print("4b. THE HEDGE. 'I should' against 'I should probably'. One word.")
    print("=" * 88)
    print("   Same scene, same position, same person, same modal, same grammatical")
    print("   site. The only difference in the stem is the word `probably`. Fully")
    print("   crossed 2x2, and arousal is run beside it as the same control.\n")
    print("   %-6s %-6s %5s %11s %11s %11s %10s   %11s %10s"
          % ("pov", "person", "edges", "medial", "final", "difference", "p(paired)",
             "arousal", "p"))
    hedge = []
    for pov in ("indiv", "inst"):
        for pr in ("I", "we"):
            cell = []
            for col in ("v", "a"):
                a = per_edge(m[(m.pov == pov) & (m.person == pr) & (m.modal == "medial")], col)
                b = per_edge(m[(m.pov == pov) & (m.person == pr) & (m.modal == "final")], col)
                k = a.index.intersection(b.index)
                cell.append(None if len(k) < 10 else
                            (len(k), a[k].mean(), b[k].mean(), (a[k] - b[k]).mean(),
                             float(stats.ttest_rel(a[k].values, b[k].values)[1])))
            rv, ra = cell
            if rv is None:
                print("   %-6s %-6s   (not tested)" % (pov, pr))
                continue
            hedge.append(rv[3])
            print("   %-6s %-6s %5d %+11.4f %+11.4f %+11.4f %10.1e%s  %+11.4f %10.1e%s"
                  % (pov, pr, rv[0], rv[1], rv[2], rv[3], rv[4], " *" if rv[4] < 0.05 else "  ",
                     ra[3] if ra else float("nan"), ra[4] if ra else float("nan"),
                     " *" if ra and ra[4] < 0.05 else "  "))
    if hedge:
        h = np.array(hedge)
        print("\n   positive at %d of %d cells, mean %+.4f, range %+.4f..%+.4f"
              % (int((h > 0).sum()), len(h), h.mean(), h.min(), h.max()))
        print("   for scale: the POSITION gap over the same edges averages %+.4f" % np.array(gaps).mean())

    print("\n" + "=" * 88)
    print("5. THE ATTRIBUTION. Variance of the 14 cell means, decomposed by factor.")
    print("=" * 88)
    print("   Per-edge values, ordinary least squares on the three factors with no")
    print("   interaction, `final_ought` DROPPED so the design is balanced 2x2x3.")
    print("   Reported as partial eta-squared: the share of the variance the pooled")
    print("   contrast attributes to POSITION that each factor actually holds.\n")
    rows = []
    for pov in ("indiv", "inst"):
        for pr in ("I", "we"):
            for md in SHARED:
                d = per_edge(m[(m.pov == pov) & (m.person == pr) & (m.modal == md)], "v")
                for e, val in d.items():
                    rows.append({"edge": e, "pov": pov, "person": pr, "modal": md, "v": val})
    R = pd.DataFrame(rows)
    print("   %d edge-cell observations over %d edges, %d cells"
          % (len(R), R.edge.nunique(), len(R.groupby(["pov", "person", "modal"]))))
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        def fitted(frame, title):
            fit = smf.ols("v ~ C(pov) + C(person) + C(modal) + C(edge)", data=frame).fit()
            aov = sm.stats.anova_lm(fit, typ=2)
            ss_err = float(aov.loc["Residual", "sum_sq"])
            print("\n   %s" % title)
            print("   %-14s %12s %8s %10s %12s" % ("factor", "sum_sq", "df", "F", "partial eta2"))
            for f in ("C(pov)", "C(person)", "C(modal)"):
                ss = float(aov.loc[f, "sum_sq"])
                print("   %-14s %12.4f %8.0f %10.3f %12.4f"
                      % (f, ss, aov.loc[f, "df"], aov.loc[f, "F"], ss / (ss + ss_err)))

        fitted(R, "ALL THREE MODAL LEVELS -- `modal` here includes the finite/infinitive "
                  "site\n   difference, so its share is an upper bound on anything about modality.")
        #: the same decomposition with `absent` dropped. Every cell is now a bare
        #: infinitive after a modal, so the factor is the HEDGE and nothing else.
        fitted(R[R.modal != "absent"],
               "MODAL-BEARING FORMS ONLY (final vs medial) -- one grammatical site "
               "throughout,\n   so `modal` is the single word `probably`.")
        print("\n   (edge is in both models as a blocking factor; its row is omitted)")
    except ImportError:
        print("\n   statsmodels not installed; run with --with statsmodels for section 5.")

    print("\n" + "=" * 88)
    print("6. WHAT THE HEDGE ACTUALLY MOVES. Read the words, not the coefficient.")
    print("=" * 88)
    print("   A +0.21 shift in a norm mean is not yet a claim about language. These are")
    print("   the words whose riser-minus-faller COUNT changes most between `I should`")
    print("   and `I should probably`, pooled over both positions and both persons.\n")
    net = {}
    for md in ("final", "medial"):
        sub = m[m.modal == md]
        c = (sub[sub.role == "riser"].word.value_counts()
             .subtract(sub[sub.role == "faller"].word.value_counts(), fill_value=0))
        net[md] = c
    d = (net["medial"].subtract(net["final"], fill_value=0)).sort_values()
    vv = pd.Series(V)
    for lab, sl in (("MORE PROMOTED once the speaker hedges", d.tail(14)[::-1]),
                    ("MORE DEMOTED once the speaker hedges", d.head(14))):
        print("   %s" % lab)
        for w, x in sl.items():
            print("      %-18s net %+7.0f    valence %s"
                  % (w, x, ("%+.2f" % vv[w]) if w in vv.index else "  (not in norms)"))
        print("")
    #: the counts above are unweighted by valence, so this states the direction
    #: the norm actually saw rather than leaving the reader to infer it.
    for lab, sl in (("promoted", d.tail(40)), ("demoted", d.head(40))):
        got = [vv[w] for w in sl.index if w in vv.index]
        print("   mean valence of the 40 most-%s words: %+.3f  (n=%d in the norm table)"
              % (lab, float(np.mean(got)), len(got)))

    #: `probably` IS THE HEDGE WORD AND IT IS IN THE SLOT IN ONE ARM ONLY. After
    #: "I should" the model can emit it and alignment does; after "I should
    #: probably" it cannot. That is the campaign's decoy problem inside the
    #: contrast, so state its size rather than leaving a reader to find it.
    pw = float(d.get("probably", 0.0))
    print("\n   `probably` net %+.0f -- it is AVAILABLE IN THE SLOT in the `final` arm only."
          % pw)
    print("   It is absent from the norm table, so it contributes NOTHING to the +0.21")
    print("   directly; what cannot be ruled out cheaply is the mass it displaces.")

    #: POOLED OVER FOUR CELLS. Check the four agree before reading the pooled list.
    print("\n   THE FOUR CELLS SEPARATELY, Spearman over the net-change vectors")
    per = {}
    for pov in ("indiv", "inst"):
        for pr in ("I", "we"):
            s = m[(m.pov == pov) & (m.person == pr)]
            c = {}
            for md in ("final", "medial"):
                t = s[s.modal == md]
                c[md] = (t[t.role == "riser"].word.value_counts()
                         .subtract(t[t.role == "faller"].word.value_counts(), fill_value=0))
            per[(pov, pr)] = c["medial"].subtract(c["final"], fill_value=0)
    keys = list(per)
    P = pd.DataFrame(per).fillna(0.0)
    print("   %-14s %s" % ("", "  ".join("%s/%s" % k for k in keys)))
    for i, a in enumerate(keys):
        cells = []
        for b in keys:
            cells.append("      -   " if a == b
                         else "%+9.3f" % stats.spearmanr(P[a], P[b])[0])
        print("   %-14s %s" % ("%s/%s" % a, "  ".join(cells)))
    top = set(d.tail(14).index) | set(d.head(14).index)
    agree = sum(1 for w in top if len({np.sign(per[k].get(w, 0.0)) for k in keys}) == 1)
    print("   of the 28 words listed above, %d have the SAME SIGN in all four cells" % agree)

    out = os.path.join(CAMP, "results", "x_m03_kernel_decomp.csv")
    R.to_csv(out, index=False)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
