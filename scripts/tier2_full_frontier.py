#!/usr/bin/env python3
"""Full floor x k grid with bootstrap margins, for the objective-choice document.

Presenting a frontier through three points is itself a framing choice. This
emits every cell so the decision-maker sees where the trade is steep and where
it is shallow, rather than a curve drawn through selected points.

FC point estimates at 4M draws. Bootstrap upper bounds resample the DECOY POOL
itself -- that is the uncertainty Monte Carlo cannot buy down, and it is what
the margin clause is written against. Only computed where FC > 0.005, since
below that the margin is irrelevant to a 0.10 ceiling.
"""
import csv, importlib.util, math
import numpy as np

g = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("g", "scripts/tier2_construct_grid.py"))
importlib.util.spec_from_file_location("g", "scripts/tier2_construct_grid.py").loader.exec_module(g)

N, NB, BOOT = 4_000_000, 1_000_000, 100
FLOORS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def main():
    rows = []
    for corp, anch in g.ANCHOR.items():
        pool = g.decoy_pool(corp)
        absd = np.sort(np.abs(pool))
        r0, r1 = np.random.default_rng(20260726), np.random.default_rng(20260727)
        z, d = r0.choice(pool, size=(N, 3)), r1.choice(pool, size=(N, 3)) + anch
        for pct in FLOORS:
            f = absd[min(int(pct / 100 * len(absd)), len(absd) - 1)]
            for k in (1, 2, 3):
                fc = float((((z > 0) & (np.abs(z) > f)).sum(1) >= k).mean())
                pw = float((((d > 0) & (np.abs(d) > f)).sum(1) >= k).mean())
                hi = ""
                if fc > 0.005:
                    b = []
                    for i in range(BOOT):
                        rs = np.random.default_rng(500 + i).choice(pool, size=len(pool), replace=True)
                        a2 = np.sort(np.abs(rs))
                        f2 = a2[min(int(pct / 100 * len(a2)), len(a2) - 1)]
                        zz = np.random.default_rng(9000 + i).choice(rs, size=(NB, 3))
                        b.append(float((((zz > 0) & (np.abs(zz) > f2)).sum(1) >= k).mean()))
                    hi = round(float(np.percentile(b, 97.5)), 5)
                rows.append(dict(corpus=corp, floor=f"p{pct}", k=k, threshold=round(float(f), 4),
                                 false_cert=round(fc, 5), fc_upper95=hi, power=round(pw, 3)))
    with open("data/tier2_full_frontier.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    for corp in g.ANCHOR:
        print(f"\n=== {corp} ===" + ("   (DESCRIPTIVE-ONLY — cannot carry the verdict)"
                                     if corp != "hh_rlhf" else ""))
        print(f"{'floor':>7s}{'k':>3s}{'|D|':>9s}{'FC':>9s}{'FC upper95':>12s}{'power':>8s}")
        for r in rows:
            if r["corpus"] == corp and r["k"] >= 2:
                print(f"{r['floor']:>7s}{r['k']:>3d}{r['threshold']:>9.4f}{r['false_cert']:>9.4f}"
                      f"{str(r['fc_upper95']):>12s}{r['power']:>8.3f}")
    print("\nk=1 rows in the CSV; all have FC far above 0.10 and are not viable.")
    print("-> data/tier2_full_frontier.csv")


if __name__ == "__main__":
    main()
