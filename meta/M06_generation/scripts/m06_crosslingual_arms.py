"""The cross-lingual arm contrast: does alignment reduce drift in Chinese too?

    uv run python meta/M06_generation/scripts/m06_crosslingual_arms.py
    -> results/crosslingual_arms.json + crosslingual_arms_pairs.parquet

Runs plan_crosslingual_arms (committed before this file existed) on the
instrument built at a3fb226b, which deliberately computed no arm contrast.

    P1  directional  English drift falls under alignment (two prior corpora)
    Q1  open         whether it holds in Chinese
    Q2  open         the LANGUAGE DiD, the reason the design exists

THE CONFOUND, CONTROLLED RATHER THAN NOTED: `total_drift` is the DIAMETER of a
sentence set, so it grows with sentence count. If alignment changes sentence
count differently across languages the DiD inherits it mechanically. So
`n_sents` by (language, role) and per-arm survival print BEFORE any contrast,
and every contrast is reported pooled AND matched on n_sents. If they
disagree the matched one travels.
"""
import collections
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
MIN_PER_BIN = 3


def sign_test(ds):
    ds = np.asarray(ds, float)
    up = int((ds > 0).sum()); dn = int((ds < 0).sum())
    lo = min(up, dn)
    p = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
    return {"median": float(np.median(ds)), "mean": float(np.mean(ds)),
            "n": len(ds), "up": up, "dn": dn, "p_sign": p}


def main():
    import pandas as pd

    pairs = [p for p in json.load(open(os.path.join(ROOT, "data/base_aligned_pairs.json")))
             if not p.get("ambiguous")]
    df = pd.concat([pd.read_parquet(os.path.join(
        OUTD, "crosslingual_drift_%s_cells.parquet" % l)) for l in ("zh", "en")])
    models = set(df.model)
    use = [p for p in pairs if p["base"] in models and p["aligned"] in models]
    #: both languages, per the plan's population
    bylang = {l: set(df[df.lang == l].model) for l in ("zh", "en")}
    use = [p for p in use if all(p[r] in bylang[l] for r in ("base", "aligned")
                                 for l in ("zh", "en"))]
    role = {}
    for p in use:
        role[p["base"]] = (p["base"] + ">" + p["aligned"], "base")
        role[p["aligned"]] = (p["base"] + ">" + p["aligned"], "aligned")
    print("pairs complete in both languages: %d (%s)"
          % (len(use), pd.Series([p["stage"] for p in use]).value_counts().to_dict()))

    df["pair"] = [role[m][0] if m in role else None for m in df.model]
    df["role"] = [role[m][1] if m in role else None for m in df.model]
    d = df[df.pair.notna()].copy()
    print("passages in the paired population: %s of %s"
          % (format(len(d), ","), format(len(df), ",")))

    #: SURVIVAL AND THE CONFOUND, BEFORE ANY CONTRAST
    print("\nper-arm coverage and the n_sents confound, by (language, role)")
    print("  %-5s %-8s %8s %8s %10s %10s"
          % ("lang", "role", "passages", "cells", "n_sents", "length"))
    cov = {}
    for l in ("zh", "en"):
        for r in ("base", "aligned"):
            s = d[(d.lang == l) & (d.role == r)]
            cells = s.groupby(["pair", "prompt"]).ngroups
            cov["%s:%s" % (l, r)] = {"passages": len(s), "cells": cells,
                                     "n_sents_mean": float(s.n_sents.mean()),
                                     "length_mean": float(s.length.mean())}
            print("  %-5s %-8s %8s %8s %10.2f %10.1f"
                  % (l, r, format(len(s), ","), format(cells, ","),
                     s.n_sents.mean(), s.length.mean()))
    for l in ("zh", "en"):
        a, b = cov["%s:aligned" % l], cov["%s:base" % l]
        print("  %s: aligned-minus-base n_sents %+.3f (%.1f%% of base)"
              % (l, a["n_sents_mean"] - b["n_sents_mean"],
                 100 * (a["n_sents_mean"] - b["n_sents_mean"]) / b["n_sents_mean"]))

    out = {"plan": "plans/plan_crosslingual_arms.md", "n_pairs": len(use),
           "coverage": cov, "contrasts": {}}
    rows = []

    def contrast(metric, matched):
        """aligned - base per (pair, prompt), then pair medians."""
        res = {}
        for l in ("zh", "en"):
            s = d[d.lang == l]
            #: MATCHED variant pools ACROSS PROMPTS: requiring both arms at
            #: the same exact sentence count WITHIN a (pair, prompt) cell is
            #: unrunnable at ~2.3 passages per cell -- the first version
            #: returned 0 units and printed as p=1, which reads like a null.
            key = (["pair", "n_sents", "role"] if matched
                   else ["pair", "prompt", "role"])
            g = s.groupby(key)[metric].mean().unstack("role")
            g = g.dropna(subset=["aligned", "base"])
            if matched:
                #: require both arms at the SAME sentence count, >=3 passages
                cnt = s.groupby(key).size().unstack("role")
                cnt = cnt.reindex(g.index).dropna()
                keep = (cnt["aligned"] >= MIN_PER_BIN) & (cnt["base"] >= MIN_PER_BIN)
                g = g[keep.reindex(g.index).fillna(False)]
            delta = (g["aligned"] - g["base"])
            pm = delta.groupby(level="pair").median()
            if len(pm) < 8:
                res[l] = {"UNRUNNABLE": "only %d pairs survive this "
                                        "construction" % len(pm)}
                continue
            res[l] = sign_test(pm.values)
            res[l]["n_units"] = int(len(delta))
            for pr, v in pm.items():
                rows.append({"metric": metric, "matched": matched, "lang": l,
                             "pair": pr, "delta": float(v)})
        a = d[d.lang == "en"]; b = d[d.lang == "zh"]
        pe = {k: v for k, v in zip(*_pairmed(a, metric, matched))}
        pz = {k: v for k, v in zip(*_pairmed(b, metric, matched))}
        both = sorted(set(pe) & set(pz))
        if len(both) >= 8:
            res["DiD_en_minus_zh"] = sign_test([pe[k] - pz[k] for k in both])
        return res

    def _pairmed(s, metric, matched):
        key = (["pair", "n_sents", "role"] if matched
               else ["pair", "prompt", "role"])
        g = s.groupby(key)[metric].mean().unstack("role")
        g = g.dropna(subset=["aligned", "base"])
        if matched:
            cnt = s.groupby(key).size().unstack("role").reindex(g.index).dropna()
            keep = (cnt["aligned"] >= MIN_PER_BIN) & (cnt["base"] >= MIN_PER_BIN)
            g = g[keep.reindex(g.index).fillna(False)]
        pm = (g["aligned"] - g["base"]).groupby(level="pair").median()
        return list(pm.index), list(pm.values)

    for metric in ("total_drift", "mean_drift"):
        for matched in (False, True):
            tag = "%s|%s" % (metric, "n_sents-matched" if matched else "pooled")
            r = contrast(metric, matched)
            out["contrasts"][tag] = r
            print("\n%s  (negative = alignment REDUCES drift)" % tag.upper())
            for l in ("zh", "en"):
                v = r[l]
                if "UNRUNNABLE" in v:
                    print("  %-3s REFUSED: %s" % (l, v["UNRUNNABLE"]))
                    continue
                print("  %-3s median %+.4f (mean %+.4f)  %d/%d  p %.3g  "
                      "(pairs %d, units %s)"
                      % (l, v["median"], v["mean"], v["up"], v["dn"],
                         v["p_sign"], v["n"], format(v["n_units"], ",")))
            if "DiD_en_minus_zh" in r:
                v = r["DiD_en_minus_zh"]
                print("  DiD en-minus-zh  %+.4f  %d/%d  p %.3g  (pairs %d)"
                      % (v["median"], v["up"], v["dn"], v["p_sign"], v["n"]))

    pd.DataFrame(rows).to_parquet(
        os.path.join(OUTD, "crosslingual_arms_pairs.parquet"))
    p = os.path.join(OUTD, "crosslingual_arms.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
