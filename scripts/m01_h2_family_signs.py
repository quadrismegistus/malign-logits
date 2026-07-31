"""H2's per-family sign test — is "alignment de-extremifies valence" a claim
about ALIGNMENT AS SUCH, or about this roster's aggregate?

RH's commission, docket [1791].

WHY IT EXISTS. H2's family ICC of -0.002 licenses POOLING the families. It does
NOT license "families show the same effect": an ICC is a VARIANCE RATIO, and with
prompt variance large, families can differ severalfold in magnitude and still
return an ICC near zero. C1's mean/median gap demonstrates exactly that on
adjacent data (mean 0.0298, median family 0.0052, [1783]). The stronger reading
was ratified twice and was wrong both times.

    .venv/bin/python scripts/m01_h2_family_signs.py

WHAT THIS IS AND IS NOT ([1791].5). A DESCRIPTIVE READOUT on a confirmed finding.
H2's verdict does not move on it. What moves is the SCOPE the finding may be
stated at.

EVERY STATISTIC IS IMPORTED FROM THE FROZEN PRODUCER, NOT REIMPLEMENTED --
`collect`, `value_of`, `A_and_terms`, the global arousal fits, the displacing
stratum's definition. A second hand-rolled copy of the residualisation is exactly
the defect the campaign has spent itself removing. The only thing new here is the
per-family grouping and the sign test over it.
"""

import os
import sys
import math
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import m01_registration_b as B  # noqa: E402
import m01_registration_c3 as C3  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "m01_h2_family_signs.csv")

DIM, VARIANT = "valence", "extremity"     # H2
PREDICTED = "up"                          # A_|v| POSITIVE


def sign_test_one_sided(k, n):
    """P(X >= k) under Binomial(n, 0.5). Exact."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    N, C = B._instrument()
    prompts, models, _h, drift = C.frozen_population()
    if drift:
        sys.exit(f"POPULATION DRIFT: {drift}. Refusing to measure.")
    edges, _ = C.operation_edges(models)
    norms, _f, _ = N.load_norms()
    cells, diag, n_moved, n_disp, n_ctrl = C3.collect(prompts, edges, norms, N, C)

    # §C0 global fit, exactly as the producer does it: over the WHOLE qualifying
    # set, never within family and never within cell.
    flat = [z for c in cells for z in c["z"]]
    Ar = [z["arousal"] for z in flat]
    coef = C3.fit(Ar, [abs(z[DIM] - C3.ORIGIN_Z) for z in flat], quad=True)

    disp = [c for c in cells if c["stratum"] == "displacing"]
    print("H2 PER-FAMILY SIGN TEST — descriptive readout, not a new test of H2")
    print(f"  stratum DISPLACING: {len(disp)} cells of {len(cells)} qualifying")
    print(f"  arousal fit is GLOBAL over {len(flat)} rated words "
          f"(b0={coef[0]:+.4f} b1={coef[1]:+.4f} b2={coef[2]:+.4f})\n")

    rows = []
    for c in disp:
        vals = [C3.value_of(z, DIM, VARIANT, coef) for z in c["z"]]
        t = C3.A_and_terms(vals, c["w"], c["roles"])
        if t is None:
            continue
        rows.append(dict(family=c["family"], prompt=c["prompt"], A=t["A"]))
    d = pd.DataFrame(rows)

    # [1791].4: NO FAMILY IS DROPPED FOR THINNESS WITHOUT THE DROP BEING PRINTED.
    counts = d.groupby("family").size()
    thin = counts[counts < B.MIN_CELLS_TO_REPORT]
    print(f"  {len(counts)} families present, {len(d)} cells")
    if len(thin):
        print(f"  FAMILIES BELOW THE {B.MIN_CELLS_TO_REPORT}-CELL FLOOR "
              f"(reported, and reported BOTH ways below):")
        for f, n in thin.items():
            print(f"      {f:<22} {n} cells")
    else:
        print(f"  no family below the {B.MIN_CELLS_TO_REPORT}-cell floor")

    per = d.groupby("family").A.agg(["mean", "median", "size", "std"])
    per = per.sort_values("mean", ascending=False)
    print(f"\n  {'family':<24}{'cells':>7}{'mean A':>11}{'median A':>11}{'SD':>10}")
    for f, r in per.iterrows():
        mark = "  <- below floor" if f in thin.index else ""
        print(f"  {f:<24}{int(r['size']):>7}{r['mean']:>11.4f}"
              f"{r['median']:>11.4f}{r['std']:>10.4f}{mark}")

    print("\n" + "=" * 70)
    print("THE COUNT — [1791].1: count and total, never a proportion")
    print("=" * 70)
    for label, sub in (("ALL families", per),
                       (f"floor-passing only (>= {B.MIN_CELLS_TO_REPORT} cells)",
                        per[~per.index.isin(thin.index)])):
        k = int((sub["mean"] > 0).sum())
        n = len(sub)
        p = sign_test_one_sided(k, n)
        print(f"  {label:<44} {k} of {n}   exact one-sided p = {p:.4g}")

    print("\n" + "=" * 70)
    print("THE MAGNITUDES — [1791].2: signs alone cannot tell the two patterns apart")
    print("=" * 70)
    m, md = per["mean"].mean(), per["mean"].median()
    print(f"  pooled A (mean over cells, the producer's statistic) : "
          f"{d.A.mean():+.4f}")
    print(f"  mean of the per-family means                        : {m:+.4f}")
    print(f"  MEDIAN family                                       : {md:+.4f}")
    print(f"  ratio mean/median                                   : "
          f"{m/md if md else float('nan'):.2f}x")
    print(f"  per-family range                                    : "
          f"{per['mean'].min():+.4f} to {per['mean'].max():+.4f}")
    # [1791].3: the raw spread beside any variance quantity, stratification declared
    print(f"\n  STRATIFICATION: A is per CELL; families group cells; the arousal fit")
    print(f"  is GLOBAL. Raw spread beside the summary, per [1787]:")
    print(f"    SD of per-family mean A (BETWEEN families) : {per['mean'].std():.4f}")
    print(f"    median WITHIN-family SD of cell A          : {per['std'].median():.4f}")
    print(f"    ratio between/within                       : "
          f"{per['mean'].std()/per['std'].median():.2f}")

    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
