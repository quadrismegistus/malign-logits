"""Where does the mass go? Category of faller against category of riser.

    uv run python s_category_crosstab.py

THE TEST IS SYMMETRY, WHICH IS SHARPER THAN A CHI-SQUARE. If the direction of
substitution carries no information, the table of faller-category against
riser-category must equal its own transpose: the number of X-to-Y moves equals
the number of Y-to-X moves. Bowker's test asks exactly that, and each off
-diagonal pair (i,j) gets its own McNemar test for which direction dominates.

This needs no permutation and no null model. Asymmetry IS the effect.

THREE LABELINGS ARE RUN AND ALL THREE ARE REPORTED, because each is wrong in a
different way and agreement between them is the only thing that would not be an
artifact of one scheme:

    induced     16 categories proposed by an Opus agent from the 685 types
                shuffled, with no role or count information and no statement of
                what contrast they would be used for
    wordnet     15 verb supersenses, external and fixed, so they cannot have
                been fitted -- but too coarse to separate `whispered` from
                `said`, which is the distinction this corpus turns on
    general_inquirer   multi-label, so NOT a partition and NOT cross-tabbed;
                it gets per-category rate tests instead, which is the analysis
                its structure actually supports

SENSITIVITY ON THE ONE BOUNDARY THAT MATTERS. The induced taxonomy resolved
target-ambiguous force verbs conservatively toward `object_handling`, which its
own author flagged as under-populating `bodily_violence` (29 types). That is the
boundary this campaign's finding runs across, so the table is recomputed with
every type whose hard-case entry names `bodily_violence` as a competing reading
moved into it. If the answer survives both drawings the boundary was not
load-bearing; if it does not, the labeling is carrying the result.
"""

import collections
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
LEX = os.path.join(CAMP, "lexicons")
OUT = os.path.join(CAMP, "results")
POP = os.path.join(os.path.dirname(os.path.dirname(CAMP)), "data", "r_population_k2.parquet")

MIN_CELL = 10


def induced_labels():
    d = pd.read_csv(os.path.join(LEX, "m01_token_labels.csv"))
    return dict(zip(d.token.str.lower(), d.category))


def violence_block():
    """Types whose hard-case entry names bodily_violence as a live competitor."""
    txt = open(os.path.join(LEX, "m01_hard_cases.md")).read()
    out = set()
    for line in txt.splitlines():
        if not line.strip().startswith("- **"):
            continue
        if "bodily_violence" not in line:
            continue
        m = re.match(r"- \*\*(.+?)\*\*", line.strip())
        if m:
            for w in re.split(r"\s*/\s*", m.group(1)):
                out.add(w.strip().lower())
    return out


def wordnet_labels(tokens):
    W = json.load(open(os.path.join(LEX, "m01_token_lexicon.json")))["tokens"]
    return {t: (W[t]["wn_supersense"] or "unassigned") for t in tokens if t in W}


def bowker(T):
    """Bowker's test of symmetry. Under the null n_ij == n_ji for all i<j."""
    k = T.shape[0]
    stat = 0.0
    df = 0
    for i in range(k):
        for j in range(i + 1, k):
            s = T[i, j] + T[j, i]
            if s > 0:
                stat += (T[i, j] - T[j, i]) ** 2 / s
                df += 1
    return stat, df, (1 - stats.chi2.cdf(stat, df) if df else np.nan)


def crosstab(P, lab, name):
    f = P.faller.str.lower().str.strip().map(lab)
    r = P.riser.str.lower().str.strip().map(lab)
    ok = f.notna() & r.notna()
    cats = sorted(set(f[ok]) | set(r[ok]))
    idx = {c: i for i, c in enumerate(cats)}
    T = np.zeros((len(cats), len(cats)))
    for a, b in zip(f[ok], r[ok]):
        T[idx[a], idx[b]] += 1
    stat, df, p = bowker(T)
    print("\n" + "=" * 78)
    print("%s -- %d of %d pairs labeled on both sides, %d categories"
          % (name.upper(), int(ok.sum()), len(P), len(cats)))
    print("=" * 78)
    print("Bowker test of symmetry: chi2=%.1f, df=%d, p=%.3e" % (stat, df, p))
    print("  (symmetry is the null: direction carries no information)")

    rows = []
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if i >= j:
                continue
            n_ij, n_ji = T[i, j], T[j, i]
            if n_ij + n_ji < MIN_CELL:
                continue
            #: exact binomial, which is McNemar without the approximation
            pv = stats.binomtest(int(max(n_ij, n_ji)), int(n_ij + n_ji), 0.5).pvalue
            rows.append(dict(frm=a if n_ij > n_ji else b, to=b if n_ij > n_ji else a,
                             dominant=int(max(n_ij, n_ji)), reverse=int(min(n_ij, n_ji)),
                             n=int(n_ij + n_ji), p=pv))
    D = pd.DataFrame(rows).sort_values("p")
    m = len(D)
    D["bonferroni"] = D.p < 0.05 / max(m, 1)
    print("\n%d directed pairs with n>=%d; Bonferroni alpha=%.5f\n"
          % (m, MIN_CELL, 0.05 / max(m, 1)))
    print("  %-24s %-24s %7s %7s %10s" % ("FROM", "TO", "n", "reverse", "p"))
    for _, x in D[D.bonferroni].head(14).iterrows():
        print("  %-24s %-24s %7d %7d %10.2e" % (x.frm, x.to, x.dominant, x.reverse, x.p))
    if not D.bonferroni.any():
        print("  none survive correction")
    D.insert(0, "labeling", name)
    return D, T, cats


def gi_rates(P):
    """GI is multi-label, so it is NOT a partition and gets rate tests, not a
    cross-tab. For each category: how often does a faller carry it, against how
    often a riser does, paired within the stem."""
    L = json.load(open(os.path.join(LEX, "m01_token_lexicon.json")))["tokens"]
    cats = collections.Counter()
    for v in L.values():
        cats.update(v["gi_categories"])
    keep = [c for c, n in cats.items() if n >= 20]
    rows = []
    for c in keep:
        has = {t: (c in v["gi_categories"]) for t, v in L.items()}
        f = P.faller.str.lower().str.strip().map(has).fillna(False)
        r = P.riser.str.lower().str.strip().map(has).fillna(False)
        d = pd.DataFrame({"stem": P.stem, "f": f.astype(int), "r": r.astype(int)})
        g = d.groupby("stem").mean()
        diff = (g.r - g.f).values
        if diff.std() == 0:
            continue
        t, p = stats.ttest_1samp(diff, 0)
        rows.append(dict(category=c, faller_rate=f.mean(), riser_rate=r.mean(),
                         diff=float(diff.mean()), p=float(p), n_words=cats[c]))
    D = pd.DataFrame(rows).sort_values("p")
    m = len(D)
    D["rank"] = range(1, m + 1)
    D["bonferroni"] = D.p < 0.05 / m
    print("\n" + "=" * 78)
    print("GENERAL INQUIRER -- %d categories with >=20 words, rate tests not a cross-tab" % m)
    print("=" * 78)
    print("  %-14s %9s %9s %9s %11s" % ("category", "faller", "riser", "diff", "p"))
    for _, x in D[D.bonferroni].sort_values("diff", key=abs, ascending=False).head(16).iterrows():
        print("  %-14s %9.3f %9.3f %+9.3f %11.2e" % (x.category, x.faller_rate, x.riser_rate, x["diff"], x.p))
    print("\n  %d of %d survive Bonferroni at alpha=%.5f" % (D.bonferroni.sum(), m, 0.05 / m))
    return D


def main():
    P = pd.read_parquet(POP)
    print("population: %d pairs, %d stems, %d cells" % (len(P), P.stem.nunique(),
                                                        P.groupby(["stem", "member"]).ngroups))
    base = induced_labels()
    blk = violence_block()
    alt = dict(base)
    moved = []
    for w in blk:
        if w in alt and alt[w] != "bodily_violence":
            moved.append((w, alt[w]))
            alt[w] = "bodily_violence"
    print("\nSENSITIVITY BLOCK: %d types named as bodily_violence competitors, %d moved"
          % (len(blk), len(moved)))
    print("  %s" % ", ".join("%s(%s)" % (w, c[:12]) for w, c in sorted(moved)[:14]))

    all_d = []
    for lab, nm in [(base, "induced"), (alt, "induced_violence_wide"),
                    (wordnet_labels(set(P.faller.str.lower()) | set(P.riser.str.lower())), "wordnet")]:
        D, T, cats = crosstab(P, lab, nm)
        all_d.append(D)
        if nm == "induced":
            pd.DataFrame(T, index=cats, columns=cats).to_csv(os.path.join(OUT, "s_crosstab_induced.csv"))
    G = gi_rates(P)

    pd.concat(all_d, ignore_index=True).to_csv(os.path.join(OUT, "s_crosstab_pairs.csv"), index=False)
    G.to_csv(os.path.join(OUT, "s_crosstab_gi.csv"), index=False)
    print("\nwrote s_crosstab_induced.csv, s_crosstab_pairs.csv, s_crosstab_gi.csv")


if __name__ == "__main__":
    main()
