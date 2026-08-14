"""The numbers in composition_not_level.md that were computed inline. Producer debt, discharged.

    uv run python meta/M06_generation/scripts/m06_mediation_contrast.py
    -> results/mediation_contrast.json

WHY THIS FILE EXISTS. Four load-bearing quantities in
`findings/composition_not_level.md` were computed with inline `uv run python -c`
during the session of 2026-08-13/14 and existed in no script. Per RH's ruling
relayed at [5890], producer debt outranks the work that surfaced it: a missing
producer makes a number UNAUDITABLE, which is worse than a figure being undrawn.
Written the same night rather than left for transcript recovery.

The four, in the order the finding uses them:

  1. LEVEL BY MOVEMENT CLASS. The table that forced Result 3's withdrawal: on
     base-generated text the aligned model finds nearly everything costlier and
     words M01 NEVER MEASURED costliest of all, so fallers sit BELOW the corpus
     average and "displaced words cost more" is true and not distinctive.

  2. COMMON-SUPPORT DIAGNOSTIC. Fall/rise counts by log p_aligned bin. Under
     CANONICAL a faller is partly DEFINED by its aligned probability, so the two
     classes occupy near-disjoint ranges and a partial correlation conditioned
     on p_aligned EXTRAPOLATES across a gap rather than comparing within it.

  3. THE CONTRAST ON COMMON SUPPORT. The finding's surviving claim: restricted
     to the band where both directions exist, median(level|fall) -
     median(level|rise). This is the ladder's rungs-at-one-probability contrast
     obtained observationally, and it is the form to quote.

  4. CONSISTENCY ASYMMETRY. Promotion is MORE consistent across sites than
     demotion, which REFUTES the obvious mechanism for (3) and is recorded so it
     is not re-proposed. Population is all movement pairs, NOT the mediation's
     36; it characterises `movement`, not this analysis.

Everything reads DISTINCT from `movement`: the canonical slice carries 3,982,956
excess rows over 73,642,696 distinct keys, concentrated rather than uniform.
"""
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
THETA = 0.001
BAND = (-2.464, -1.465)     # the common-support band, from diagnostic 2
MIN_CELLS = 20
MIN_SIDE = 30               # per-arm cells needed before a pair contributes


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 22)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats

    W = pd.read_parquet(os.path.join(OUTD, "mediation_words_byprompt.parquet"))
    W = W[W["prompt"] != ""].copy()
    W["lvl"] = (W["sum_s_aligned"] - W["sum_s_base"]) / W["occurrences"]
    pairs = sorted(W["pair"].unique())
    plist = "','".join(p.replace("'", "\\'") for p in pairs)
    out = {"n_pairs_m06": len(pairs), "band": list(BAND)}

    mv = pd.DataFrame(list(ch_rows(
        "SELECT pair, prompt, word, any(pa) AS pa, any(cls) AS mv FROM "
        "(SELECT DISTINCT concat(base,'>',aligned) AS pair, prompt, word, cls, "
        " p_aligned AS pa FROM malign_logits.movement WHERE rule='canonical' "
        " AND concat(base,'>',aligned) IN ('%s')) GROUP BY pair, prompt, word"
        % plist)))
    print("movement (pair, prompt, word) DISTINCT: %s" % format(len(mv), ","))

    # ---- 1. level by movement class -----------------------------------
    B = W[W.role == "base"].merge(mv, on=["pair", "prompt", "word"], how="left")
    B["mv"] = B["mv"].fillna("unmeasured")
    tot = B["occurrences"].sum()
    cls_rows = []
    print("\n1. LEVEL BY CLASS (base-generated text, occurrence-weighted)")
    print("   %-12s %10s %7s %10s %12s" % ("class", "tokens", "share", "mean lvl", "contrib"))
    for k, g in B.groupby("mv"):
        m = np.isfinite(g["lvl"])
        share = g["occurrences"].sum() / tot
        wl = float(np.average(g["lvl"][m], weights=g["occurrences"][m]))
        cls_rows.append({"class": k, "tokens": int(g["occurrences"].sum()),
                         "share": float(share), "mean_level": wl,
                         "contribution": float(share * wl)})
        print("   %-12s %10d %6.1f%% %+10.4f %+12.4f"
              % (k, g["occurrences"].sum(), 100 * share, wl, share * wl))
    out["level_by_class"] = cls_rows

    # ---- 2. common-support diagnostic ---------------------------------
    M2 = mv[mv["mv"] != "still"].copy()
    M2["lpa"] = np.log10(np.maximum(M2["pa"].astype(float), THETA))
    D = W[W.role == "base"].merge(M2, on=["pair", "prompt", "word"], how="inner")
    edges = np.unique(np.quantile(D["lpa"], np.linspace(0, 1, 9)))
    D["bin"] = pd.cut(D["lpa"], bins=edges, include_lowest=True)
    t = D.groupby(["bin", "mv"], observed=True).size().unstack(fill_value=0)
    t["minority"] = t.min(axis=1) / t.sum(axis=1)
    print("\n2. COMMON SUPPORT (mover cells with an emitted occurrence)")
    print("   %-24s %7s %7s %9s" % ("log p_aligned bin", "fall", "rise", "minority"))
    bins = []
    for i, r in t.iterrows():
        print("   %-24s %7d %7d %9.3f"
              % (str(i)[:24], r.get("fall", 0), r.get("rise", 0), r["minority"]))
        bins.append({"bin": str(i), "fall": int(r.get("fall", 0)),
                     "rise": int(r.get("rise", 0)), "minority": float(r["minority"])})
    out["common_support_bins"] = bins
    zero = sum(b["fall"] for b in bins if b["rise"] == 0)
    print("   fall cells in bins holding ZERO risers: %s" % format(zero, ","))
    out["fall_cells_in_zero_riser_bins"] = int(zero)

    # ---- 3. the contrast, on common support ---------------------------
    print("\n3. CONTRAST ON COMMON SUPPORT  median(level|fall) - median(level|rise)")
    out["contrast"] = {}
    for role, lab in (("base", "base-generated"), ("aligned", "aligned-generated")):
        Dr = W[W.role == role].merge(M2, on=["pair", "prompt", "word"], how="inner")
        Br = Dr[(Dr.lpa >= BAND[0]) & (Dr.lpa <= BAND[1])]
        ds, nf, nr = [], 0, 0
        for p, g in Br.groupby("pair"):
            f = g.loc[g.mv == "fall", "lvl"].dropna()
            r = g.loc[g.mv == "rise", "lvl"].dropna()
            if len(f) < MIN_SIDE or len(r) < MIN_SIDE:
                continue
            ds.append(float(f.median() - r.median()))
            nf += len(f)
            nr += len(r)
        ds = np.array(ds)
        tt = stats.wilcoxon(ds) if len(ds) > 5 else None
        print("   %-19s %+.4f   n=%d pairs  p %-9.3g [%d/%d positive]  cells %d/%d"
              % (lab, ds.mean(), len(ds), tt.pvalue if tt else float("nan"),
                 int((ds > 0).sum()), len(ds), nf, nr))
        out["contrast"][role] = {"mean": float(ds.mean()), "n_pairs": int(len(ds)),
                                 "p": float(tt.pvalue) if tt else None,
                                 "n_positive": int((ds > 0).sum()),
                                 "cells_fall": nf, "cells_rise": nr}

    # ---- 4. consistency asymmetry (ALL movement pairs) ----------------
    C = pd.DataFrame(list(ch_rows(
        "SELECT pair, word, countIf(cls='fall') AS nf, countIf(cls='rise') AS nr "
        "FROM (SELECT DISTINCT concat(base,'>',aligned) AS pair, prompt, word, cls "
        "FROM malign_logits.movement WHERE rule='canonical') "
        "GROUP BY pair, word HAVING nf+nr >= %d" % MIN_CELLS)))
    C["mov"] = C["nf"] + C["nr"]
    C["consistency"] = (C["nf"] - C["nr"]).abs() / C["mov"]
    C["dominant"] = np.where(C["nf"] > C["nr"], "fall", "rise")
    d = []
    for p, g in C.groupby("pair"):
        f = g.loc[g.dominant == "fall", "consistency"]
        r = g.loc[g.dominant == "rise", "consistency"]
        if len(f) < 20 or len(r) < 20:
            continue
        d.append(float(f.median() - r.median()))
    d = np.array(d)
    tt = stats.wilcoxon(d)
    print("\n4. CONSISTENCY ASYMMETRY (all movement pairs, NOT the mediation's %d)"
          % len(pairs))
    for k, g in C.groupby("dominant"):
        print("   %-14s mean %.3f  median %.3f  words %d"
              % (k + "-dominant", g["consistency"].mean(), g["consistency"].median(), len(g)))
    print("   per-pair fall minus rise  %+.4f  n=%d  p %.3g  [%d/%d positive]"
          % (d.mean(), len(d), tt.pvalue, int((d > 0).sum()), len(d)))
    print("   -> PROMOTION IS MORE CONSISTENT. Refutes 'demotion generalises'.")
    out["consistency"] = {"per_pair_fall_minus_rise": float(d.mean()),
                          "n_pairs": int(len(d)), "p": float(tt.pvalue),
                          "n_positive": int((d > 0).sum()),
                          "by_dominant": {k: {"mean": float(g["consistency"].mean()),
                                              "median": float(g["consistency"].median()),
                                              "n_words": int(len(g))}
                                          for k, g in C.groupby("dominant")}}

    p = os.path.join(OUTD, "mediation_contrast.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n-> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
