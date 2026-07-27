#!/usr/bin/env python3
"""Seeded 95% lower bounds on POWER, per cell -- the numbers the amendment pins.

WHY THIS EXISTS. The false-certification bounds have been seeded and committed
since tier2_full_frontier.py. The power bounds were not: they came from an
ad-hoc interactive run, and the same cell was reported to two seats as 0.788 and
0.782 on different days. Both are resample noise around the same quantity and
neither is wrong, but an amendment carries ONE number, and a number that moves
when nobody changed anything cannot be the one it carries.

The asymmetry was invisible while the two quantities were discussed together.
FC had a committed script because FC was the constraint under debate; power was
"just the floor check", so it never got one -- a tool doing its job silently on
one axis while the other ran bare.

WHAT IS RESAMPLED. The DECOY POOL, exactly as in the FC bounds. The alternative
distribution is the pool shifted by the anchor, so a pool resample propagates
into both the threshold and the draws; resampling only the Monte Carlo draws
would give a tight interval around the wrong thing. This is the uncertainty that
more Monte Carlo cannot buy down.

READ THE BOUNDS AS DESCRIPTION, NOT AS THE TEST. Whether the registered power
floor is evaluated on the point estimate or on the lower bound was ruled on
2026-07-27: the registration at 596213c names no estimator and its committed
script uses the point estimate, so the floor is a point-estimate constraint and
p50/3-of-3 satisfies it as registered. These bounds do not change that. They are
computed and pinned so the disclosure can state what the evidence shows
alongside what the registration required -- the gap is the disclosure's content,
not a defect to be resolved by choosing the stricter number.
"""
import csv, importlib.util
import numpy as np

_s = importlib.util.spec_from_file_location("g", "scripts/tier2_construct_grid.py")
g = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g)

N, NB, BOOT = 4_000_000, 1_000_000, 200
FLOORS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
KS = [1, 2, 3]

# Distinct from the FC-bound seeds (500+i, 9000+i) so the two bounds are not
# driven by a shared resample; same generator family, different streams.
SEED_POOL, SEED_DRAW, SEED_PT = 31_000, 47_000, 20260727


def main():
    rows = []
    for corp, anch in g.ANCHOR.items():
        pool = g.decoy_pool(corp)
        absd = np.sort(np.abs(pool))
        d = np.random.default_rng(SEED_PT).choice(pool, size=(N, 3)) + anch

        # Bootstrap replicates: resample the pool once per replicate, rebuild the
        # threshold from THAT pool, and draw the alternative from it. One pass
        # over replicates serves every cell, so the cells share replicates and
        # their bounds move together -- which is what we want when comparing them.
        reps = []
        for i in range(BOOT):
            rs = np.random.default_rng(SEED_POOL + i).choice(pool, size=len(pool), replace=True)
            a2 = np.sort(np.abs(rs))
            dd = np.random.default_rng(SEED_DRAW + i).choice(rs, size=(NB, 3)) + anch
            reps.append((a2, dd))

        for pct in FLOORS:
            f = absd[min(int(pct / 100 * len(absd)), len(absd) - 1)]
            hits_pt = ((d > 0) & (np.abs(d) > f)).sum(1)
            per_rep = []
            for a2, dd in reps:
                f2 = a2[min(int(pct / 100 * len(a2)), len(a2) - 1)]
                per_rep.append(((dd > 0) & (np.abs(dd) > f2)).sum(1))
            for k in KS:
                pw = float((hits_pt >= k).mean())
                b = [float((h >= k).mean()) for h in per_rep]
                rows.append(dict(
                    corpus=corp, floor=f"p{pct}", k=k, threshold=round(float(f), 4),
                    power=round(pw, 5),
                    power_lower95=round(float(np.percentile(b, 2.5)), 5),
                    power_upper95=round(float(np.percentile(b, 97.5)), 5)))

    with open("data/tier2_power_bounds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    for corp in g.ANCHOR:
        print(f"\n=== {corp} ===" + ("   (DESCRIPTIVE-ONLY -- cannot carry the verdict)"
                                     if corp != "hh_rlhf" else ""))
        print(f"{'floor':>7s}{'k':>3s}{'power':>9s}{'lower95':>10s}{'upper95':>10s}"
              f"{'floor 0.80':>12s}")
        for r in rows:
            if r["corpus"] == corp and r["k"] >= 2:
                pt = "pt" if r["power"] >= 0.80 else "--"
                lo = "SHOWN" if r["power_lower95"] >= 0.80 else "--"
                print(f"{r['floor']:>7s}{r['k']:>3d}{r['power']:>9.3f}"
                      f"{r['power_lower95']:>10.3f}{r['power_upper95']:>10.3f}"
                      f"{pt + '/' + lo:>12s}")
    print(f"\nBOOT={BOOT} replicates, seeds pool={SEED_POOL}+i draw={SEED_DRAW}+i "
          f"point={SEED_PT}. Re-running reproduces these exactly.")
    print("-> data/tier2_power_bounds.csv")


if __name__ == "__main__":
    main()
