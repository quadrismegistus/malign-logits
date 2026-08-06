"""Pairwise association structure between annotation fields, and whether
alignment changes it. Exploratory, seeded, reproducible.

    uv run python s_pairwise_eda.py

WHAT IT TESTS. For every valid pair of binary annotation outcomes, the log odds
ratio of co-occurrence in FR minus the same in RF, permuted by flipping FR/RF
labels within stem. Model-free, so it does not inherit the scale-dependence that
made the logit markedness interactions disagree with the rate-scale ones.

WITHIN-FIELD PAIRS ARE EXCLUDED. `generic x continues` came back at log OR
-11.95 on the first run, which is not an association: they are two levels of one
field and cannot co-occur. Six such pairs are dropped, and their exclusion is
the difference between a structure plot and a plot of the schema.

CELL COUNTS ARE PRINTED BESIDE EVERY RESULT because a 0.5 continuity correction
manufactures a large log OR out of an empty cell. On the first run
`diff_reg x subst` survived Bonferroni in the UNMARKED stratum at +1.92 with a
JOINT COUNT OF ZERO in all four cells: significant by permutation, and entirely
the correction. Significance and a trustworthy effect size come apart when the
table is sparse, and only the counts show it.

MULTIPLICITY IS HANDLED BY THE CORRECTION, ONCE. Bonferroni over all 117 tests
is applied here; it is not to be invoked again in prose afterwards as though it
had not been. Under the global null 117 * 0.00043 = 0.05 false positives are
expected, so the survivors are survivors.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")
SRC = os.path.join(OUT, "s_stage2_real_long.parquet")
DST = os.path.join(OUT, "s_pairwise_eda.csv")

SEED = 20260806
NPERM = 5000

#: Levels of one field cannot co-occur; pairs within a group are dropped.
GROUP = {"mild": "pitch", "strong": "pitch", "same_pitch": "pitch",
         "generic": "reg", "continues": "reg", "diff_reg": "reg",
         "punish": "-", "speech": "-", "bare": "-", "subst": "-"}


def fields(L):
    return {"mild": L.pitch == "B_MILDER", "strong": L.pitch == "B_STRONGER",
            "same_pitch": L.pitch == "SAME_PITCH",
            "generic": L.register == "B_GENERIC",
            "continues": L.register == "B_CONTINUES",
            "diff_reg": L.register == "B_DIFFERENT_REGISTER",
            "punish": L.more_transgressive == "YES",
            "speech": L.becomes_speech == "YES",
            "bare": L.bare_verb == "YES", "subst": L.substitutable == "YES"}


def lor(c):
    return np.log((c[..., 0] + .5) * (c[..., 3] + .5) / ((c[..., 1] + .5) * (c[..., 2] + .5)))


def stratum(L, mask, label, pairs, rng):
    sub = L[mask]
    F = fields(sub)
    fr = (sub.order == "FR").values
    stems = sorted(sub.stem.unique())
    si = sub.stem.map({s: i for i, s in enumerate(stems)}).values
    flips = rng.rand(NPERM, len(stems)) < 0.5
    rows = []
    for x, y in pairs:
        X, Y = F[x].values, F[y].values
        cells = np.stack([(X & Y), (X & ~Y), (~X & Y), (~X & ~Y)], 1).astype(float)
        frc = np.zeros((len(stems), 4)); rfc = np.zeros((len(stems), 4))
        np.add.at(frc, si[fr], cells[fr]); np.add.at(rfc, si[~fr], cells[~fr])
        base, tot, delta = frc.sum(0), frc.sum(0) + rfc.sum(0), rfc - frc
        A = base + flips @ delta
        obs = lor(base) - lor(tot - base)
        null = lor(A) - lor(tot - A)
        #: the smallest of the three informative cells, in either order
        mn = int(min(frc.sum(0)[:3].min(), rfc.sum(0)[:3].min()))
        rows.append(dict(stratum=label, pair="%s x %s" % (x, y), overall=lor(tot),
                         fr=lor(base), rf=lor(tot - base), diff=obs, min_cell=mn,
                         p=(1 + np.sum(np.abs(null) >= abs(obs))) / (NPERM + 1)))
    return rows


def main():
    L = pd.read_parquet(SRC)
    pairs = [(x, y) for x, y in itertools.combinations(fields(L), 2)
             if not (GROUP[x] == GROUP[y] != "-")]
    rng = np.random.RandomState(SEED)
    rows = (stratum(L, pd.Series(True, index=L.index), "ALL", pairs, rng)
            + stratum(L, L.member == "MARKED", "MARKED", pairs, rng)
            + stratum(L, L.member == "UNMARKED", "UNMARKED", pairs, rng))
    D = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(D)
    D["rank"] = D.index + 1
    D["bonferroni"] = D.p < 0.05 / m
    D["bh"] = (D.p <= (D["rank"] / m) * 0.05)[::-1].cummax()[::-1]
    D["sparse_cells"] = D.min_cell < 10
    D.to_csv(DST, index=False)

    print("%d tests = %d pairs x 3 strata. seed=%d, %d draws, p floor %.4f."
          % (m, len(pairs), SEED, NPERM, 1 / (NPERM + 1)))
    print("Bonferroni alpha = %.5f. Expected false positives under the global"
          " null: %.2f\n" % (0.05 / m, m * 0.05 / m))
    keep = D[D.bonferroni]
    print("SURVIVE BONFERRONI: %d, of which %d have a cell under 10 and are NOT"
          " reportable as effect sizes\n" % (len(keep), keep.sparse_cells.sum()))
    print("%-22s %-9s %8s %8s %8s %7s %s"
          % ("pair", "stratum", "overall", "FR", "RF", "min n", "diff"))
    print("-" * 78)
    for _, r in keep.sort_values("diff", key=abs, ascending=False).iterrows():
        print("%-22s %-9s %+8.2f %+8.2f %+8.2f %7d %+8.2f%s"
              % (r.pair, r.stratum, r.overall, r.fr, r.rf, r.min_cell, r["diff"],
                 "   SPARSE, artifact of the correction" if r.sparse_cells else ""))
    print("\nwrote %s" % os.path.basename(DST))


if __name__ == "__main__":
    main()
