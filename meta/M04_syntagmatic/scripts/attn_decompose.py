#!/usr/bin/env python
"""Does alignment change attention-back AT ALL? The decomposition D_norm hid.

    log[ attn(aligned,w) / attn(base,w) ]  =  log[ U_a / U_b ]  +  D_norm(w)
           TOTAL effect on the word           BASELINE shift      WORD residual

THE ERROR THIS EXISTS TO FIX. `attn_norm_sweep.py` reported only D_norm and the
contrasts between its three arms, and I summarised that as "nothing varies with
alignment status" and then as "alignment doesn't do anything to attention". The
second does not follow from the first. D_norm divides each arm by its own
undisturbed baseline, so a change that moves forced words and self-chosen words
TOGETHER is zero by construction -- and the baselines are not equal: on the two
cells where they were printed, U_aligned/U_base was +3.6% on SmolLM2 and -19.4%
on OLMo-2. A 19% shift went into the denominator and was reported as absence.

So all three terms are tested here, separately:

    BASELINE   log(U_a / U_b)          alignment's effect at a slot the model
                                       chose itself -- no forcing involved
    TOTAL      log(attn_a / attn_b)    the whole effect on a forced word
    RESIDUAL   D_norm                  what is left once the baseline is removed

THE CELL IS THE UNIT. Per-head medians within a cell, then a sign count and a
Wilcoxon across the 28 cells. Per-head p-values over 250-480 correlated heads
ran to p = 0 in both directions in this campaign and were never evidence.

    attn_decompose.py meta/M04_syntagmatic/results/attn_norm_sweep_full.json
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--by-pair", action="store_true")
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import wilcoxon

    rows = json.load(open(a.path))
    if "U" not in rows[0]:
        sys.exit("this file predates array storage; re-run attn_norm_sweep.py")
    print("%d cells from %s\n" % (len(rows), os.path.basename(a.path)))

    per = []
    for r in rows:
        Ub = np.array(r["U"]["base"], float)
        Ua = np.array(r["U"]["aligned"], float)
        base_shift = np.log(np.maximum(Ua, 1e-12) / np.maximum(Ub, 1e-12))
        rec = dict(pair=r["pair"], prompt=r["prompt"],
                   baseline=float(np.median(base_shift)))
        for lab in ("FALLER", "NONMOVER", "RISER"):
            kb, ka = "%s_base" % lab, "%s_aligned" % lab
            if kb not in r["levels"] or ka not in r["levels"]:
                continue
            lb = np.array(r["levels"][kb], float)
            la = np.array(r["levels"][ka], float)
            tot = np.log(np.maximum(la, 1e-12) / np.maximum(lb, 1e-12))
            rec["total_%s" % lab] = float(np.median(tot))
            rec["resid_%s" % lab] = float(np.median(np.array(r["d_norm_heads"][lab])))
        per.append(rec)

    def report(name, vals):
        v = np.array([x for x in vals if x == x])
        if not len(v):
            return
        try:
            p = wilcoxon(v).pvalue
        except Exception:
            p = float("nan")
        print("  %-26s median %+7.4f  (x%.3f)   %2d of %2d positive   p=%.4g"
              % (name, np.median(v), np.exp(np.median(v)),
                 int((v > 0).sum()), len(v), p))

    print("1. BASELINE -- alignment at a slot the model CHOSE, no forcing")
    report("log(U_aligned/U_base)", [r["baseline"] for r in per])
    print()
    print("2. TOTAL -- the whole effect on a forced word")
    for lab in ("FALLER", "NONMOVER", "RISER"):
        report("total, %s" % lab, [r.get("total_%s" % lab, float("nan")) for r in per])
    report("total, all three pooled",
           [r.get("total_%s" % l, float("nan")) for r in per
            for l in ("FALLER", "NONMOVER", "RISER")])
    print()
    print("3. RESIDUAL -- D_norm, what is left after the baseline is removed")
    for lab in ("FALLER", "NONMOVER", "RISER"):
        report("D_norm, %s" % lab, [r.get("resid_%s" % lab, float("nan")) for r in per])
    print()
    print("4. HOW MUCH OF THE TOTAL IS THE BASELINE")
    tot = np.array([r.get("total_%s" % l, np.nan) for r in per
                    for l in ("FALLER", "NONMOVER", "RISER")])
    bas = np.array([r["baseline"] for r in per for _ in range(3)])
    m = ~np.isnan(tot)
    print("  median |total|    %.4f" % np.median(np.abs(tot[m])))
    print("  median |baseline| %.4f" % np.median(np.abs(bas[m])))
    print("  median |residual| %.4f"
          % np.median(np.abs(np.array([r.get("resid_%s" % l, np.nan) for r in per
                                       for l in ("FALLER", "NONMOVER", "RISER")])[m])))
    if m.sum() > 2:
        print("  corr(total, baseline) across cell-words: %.3f"
              % np.corrcoef(tot[m], bas[m])[0, 1])

    if a.by_pair:
        print("\n  per pair, baseline shift:")
        seen = {}
        for r in per:
            seen.setdefault(r["pair"], []).append(r["baseline"])
        for k, v in sorted(seen.items()):
            print("    %-56s %+7.4f  (x%.3f)  n=%d"
                  % (k.split(">")[0].split("/")[-1][:56], np.median(v),
                     np.exp(np.median(v)), len(v)))


if __name__ == "__main__":
    main()
