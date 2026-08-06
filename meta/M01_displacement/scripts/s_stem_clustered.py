"""Re-test findings 1-9's directed pairs at one vote per stem.

    uv run --with lemminflect python s_stem_clustered.py

WHY. `data/r_population_k2.parquet` has 5,976 rows and they are not 5,976
observations. Each is one (faller, riser) combination inside one prompt cell,
so a cell with 12 fallers and 10 risers contributes 120 of them, and the median
STEM contributes 9. The cross-tabs in `s_category_crosstab.py` and
`s_lexicon_crosstab.py` binomtest those rows directly, which makes the
denominator a property of the join rather than a chosen unit.

Finding 2 already reported its headline clustered by stem, which is why it
survives here. The four lexicons added later did not get the same treatment.

THE TEST. For each directed category pair (a, b), count the STEMS in which
a -> b occurs at all, against the stems in which b -> a occurs at all, and
binomtest that. One stem, one vote, however many pairs it contributed.

This is deliberately not the edge unit used in findings 10-14. That analysis
walks the store fresh and needs no pair population; this one asks the narrower
question of what findings 1-9 look like when their own denominator is fixed,
so it stays on their data.
"""

import collections
import json
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
LEX = os.path.join(CAMP, "lexicons")
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
POP = os.path.join(ROOT, "data", "r_population_k2.parquet")
MIN_CELL = 10
sys.path.insert(0, HERE)


def labelings(toks):
    """IMPORT the labelers, never re-derive them.

    The first version built the WordNet dict here from
    `wordnet_verb_supersenses.json`, whose top level is `{meta, words}` -- so it
    read the two literal strings "meta" and "words" as the entire vocabulary and
    covered nothing. It reported all 18 WordNet pairs as falling below the
    minimum cell, which reads as a finding about sparsity rather than as a
    lookup under the wrong key.

    The labeling under audit lives in `s_category_crosstab.wordnet_labels`, uses
    a different file (`m01_token_lexicon.json`, field `wn_supersense`) and has
    an `unassigned` fallback. An audit that re-derives its own labeler is not
    auditing the thing it names.
    """
    import s_category_crosstab as C
    import s_lexicon_crosstab as X
    IL = pd.read_csv(os.path.join(LEX, "m01_token_labels.csv"))
    return {
        "induced": dict(zip(IL["token"].str.lower(), IL["category"])),
        "wordnet": C.wordnet_labels(set(toks)),
        "usas": X.usas_labels(toks)[0],
        "verbnet": X.verbnet_labels(toks)[0],
        "framenet": X.framenet_labels(toks)[0],
        "rid": X.rid_labels(toks)[0],
    }


def main():
    P = pd.read_parquet(POP)
    F = P["faller"].str.lower().str.strip()
    R = P["riser"].str.lower().str.strip()
    print("%d pairs, %d stems, median %.0f pairs per stem, max %d"
          % (len(P), P["stem"].nunique(), P.groupby("stem").size().median(),
             P.groupby("stem").size().max()))
    labs = labelings(sorted(set(F) | set(R)))

    rows = []
    for nm, lab in labs.items():
        f, r = F.map(lab), R.map(lab)
        ok = f.notna() & r.notna()
        Q = pd.DataFrame({"s": P["stem"][ok].values, "a": f[ok].values, "b": r[ok].values})
        Q = Q[Q["a"] != Q["b"]]
        #: one row per (stem, directed pair): the stem votes once no matter how
        #: many manufactured pairs it produced
        S = Q.drop_duplicates(["s", "a", "b"])
        c = collections.Counter(zip(S["a"], S["b"]))
        cp = collections.Counter(zip(Q["a"], Q["b"]))
        seen = set()
        for (a, b), n in c.items():
            if (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            m = c.get((b, a), 0)
            if n + m < MIN_CELL:
                continue
            fwd = n >= m
            rows.append(dict(labeling=nm, frm=a if fwd else b, to=b if fwd else a,
                             stems_fwd=max(n, m), stems_rev=min(n, m),
                             pairs_fwd=cp.get((a, b) if fwd else (b, a), 0),
                             pairs_rev=cp.get((b, a) if fwd else (a, b), 0),
                             p=stats.binomtest(max(n, m), n + m, 0.5).pvalue))
    D = pd.DataFrame(rows)
    for nm, g in D.groupby("labeling"):
        D.loc[g.index, "bonferroni"] = g["p"] < 0.05 / len(g)
    D["bonferroni"] = D["bonferroni"].astype(bool)

    #: join back onto what was reported, to name what is LOST rather than only
    #: count what survives
    prev = []
    for f in ("s_lexicon_crosstab.csv", "s_crosstab_pairs.csv"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            prev.append(pd.read_csv(p))
    L = pd.concat(prev, ignore_index=True)
    L = L[L["labeling"].isin(labs) & L["bonferroni"]]
    J = L[["labeling", "frm", "to", "dominant", "reverse"]].merge(
        D[["labeling", "frm", "to", "stems_fwd", "stems_rev", "p", "bonferroni"]],
        on=["labeling", "frm", "to"], how="left")
    #: NOT `fillna(False)` on the bool column and then `~`: fillna makes it
    #: object dtype and `~True` on a Python bool is -2, so the mask silently
    #: becomes an integer column selector. It raised here rather than lying,
    #: but the same shape elsewhere would not.
    J["testable"] = J["stems_fwd"].notna()
    J["holds"] = J["bonferroni"].fillna(False).astype(bool)
    t = J[J["testable"]]
    print("\nreported as significant: %d" % len(J))
    print("  testable at the stem unit (>= %d stems): %d" % (MIN_CELL, len(t)))
    print("     hold: %d      lost: %d" % (int(t["holds"].sum()), int((~t["holds"]).sum())))
    print("  fell below the minimum cell: %d" % int((~J["testable"]).sum()))
    print("\nby labeling:")
    for nm, g in J.groupby("labeling"):
        gt = g[g["testable"]]
        print("  %-9s reported %3d, testable %3d, hold %3d, lost %3d, uncountable %3d"
              % (nm, len(g), len(gt), int(gt["holds"].sum()),
                 int((~gt["holds"]).sum()), int((~g["testable"]).sum())))
    D.to_csv(os.path.join(OUT, "s_stem_clustered.csv"), index=False)
    J.to_csv(os.path.join(OUT, "s_stem_clustered_verdicts.csv"), index=False)
    print("\nwrote s_stem_clustered.csv (%d rows), s_stem_clustered_verdicts.csv (%d)"
          % (len(D), len(J)))


if __name__ == "__main__":
    main()
