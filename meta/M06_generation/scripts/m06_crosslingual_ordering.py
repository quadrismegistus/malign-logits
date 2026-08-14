"""Does alignment change a passage's PATH SHAPE, or only its extent?

    uv run python meta/M06_generation/scripts/m06_crosslingual_ordering.py [--full]
    -> results/crosslingual_ordering[_full].json

`findings/drift_metric_audit.md` retired `directedness` (it is 1.681/n, Spearman
-0.92 against sentence count) and named the replacement:

    ordering = mean(successive distances) - mean(all pairwise distances)

Under a RANDOM ordering of a passage's own sentences the expected successive
distance IS the mean of all pairwise distances, so this is a pure SEQUENCE
measure: same sentences, same count, same composition, only the order differs.
Negative means adjacent sentences are closer than a reshuffle would give, i.e.
local coherence. The audit specified it, `m06_crosslingual_drift.py` was wired
to persist `mean_pairwise` for it, and it had never been computed.

MEASURED HERE, untruncated: zh -0.0266, en -0.0339 (both languages locally
coherent, as any real text should be), and n-dependence -0.217 / -0.169 against
`total_drift`'s +0.33 to +0.42. Less n-driven, though NOT independent by
construction: the NULL is n-free, the observed deviation from it need not be.

THE CONTRAST IS RUN BESIDE `mean_drift` ON PURPOSE. A null is only worth
reporting next to a positive control drawn from the same pairs, the same matched
prompts and the same estimator; otherwise it is indistinguishable from an
instrument that cannot fire.
"""
import argparse
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
MIN_SHARED_PROMPTS = 5


def sign_test(ds):
    import numpy as np
    ds = [d for d in ds if np.isfinite(d)]
    up = sum(1 for d in ds if d > 0)
    dn = sum(1 for d in ds if d < 0)
    lo = min(up, dn)
    p = (min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
         if up + dn else 1.0)
    return {"median": float(np.median(ds)), "up": up, "dn": dn,
            "p_sign": float(p), "n_pairs": len(ds)}


def main():
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="untruncated cells and a *_full output")
    a = ap.parse_args()
    suf = "_full" if a.full else ""

    pairs = [p for p in json.load(open(os.path.join(ROOT, "data/base_aligned_pairs.json")))
             if not p.get("ambiguous")]
    df = pd.concat([pd.read_parquet(os.path.join(
        OUTD, "crosslingual_drift_%s%s_cells.parquet" % (l, suf)))
        for l in ("zh", "en")])
    df["ordering"] = df["mean_drift"] - df["mean_pairwise"]

    bylang = {l: set(df[df.lang == l].model) for l in ("zh", "en")}
    use = [p for p in pairs
           if all(p[r] in bylang[l] for r in ("base", "aligned") for l in ("zh", "en"))]
    print("population: %s cells, %d pairs usable in BOTH languages (%s)"
          % (format(len(df), ","), len(use),
             "no truncation" if a.full else "75-word truncation"))

    out = {"truncated": not a.full, "n_pairs": len(use), "descriptive": {}, "arms": {}}
    for l in ("zh", "en"):
        g = df[df.lang == l]
        out["descriptive"][l] = {
            "mean_drift": float(g["mean_drift"].mean()),
            "mean_pairwise": float(g["mean_pairwise"].mean()),
            "ordering_mean": float(g["ordering"].mean()),
            "ordering_median": float(g["ordering"].median()),
            "n_cells": int(len(g))}
        print("  %s ordering %+.5f (median %+.5f)  from mean_drift %.4f - "
              "mean_pairwise %.4f" % (l, g["ordering"].mean(), g["ordering"].median(),
                                      g["mean_drift"].mean(), g["mean_pairwise"].mean()))

    for metric in ("ordering", "mean_drift"):
        out["arms"][metric] = {}
        print("\n%s   (negative = alignment REDUCES)" % metric.upper())
        for l in ("zh", "en"):
            ds = []
            for p in use:
                b = df[(df.lang == l) & (df.model == p["base"])]
                al = df[(df.lang == l) & (df.model == p["aligned"])]
                shared = set(b.prompt) & set(al.prompt)
                if len(shared) < MIN_SHARED_PROMPTS:
                    continue
                #: matched on PROMPT before differencing, so the contrast is
                #: never between two different prompt mixes.
                bm = b[b.prompt.isin(shared)].groupby("prompt")[metric].median()
                am = al[al.prompt.isin(shared)].groupby("prompt")[metric].median()
                ds.append(float((am - bm).median()))
            r = sign_test(ds)
            out["arms"][metric][l] = r
            print("  %s  median %+.5f  %d up / %d dn  p %.3g  (pairs %d)"
                  % (l, r["median"], r["up"], r["dn"], r["p_sign"], r["n_pairs"]))

    path = os.path.join(OUTD, "crosslingual_ordering%s.json" % suf)
    json.dump(out, open(path, "w"), indent=1)
    print("\n-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
