"""The ordering test: does alignment shape the CHAIN or only the SET?

    uv run python meta/M06_generation/scripts/m06_zh_ordering.py
    -> results/zh_ordering.json

Plan: `plans/plan_zh_ordering.md`, committed at f9480f7a BEFORE this producer
existed, with the statistic, both populations, the MDE and P1/P2/P3 declared.

NO NEW COMPUTE. `m06_crosslingual_drift.py` persisted `mean_pairwise` beside
`mean_drift` for exactly this test; both are in the committed per-passage
parquets and nothing is re-encoded.

THE STATISTIC IS A RATIO AND THE PLAN SAYS WHY.

    order_ratio = mean_drift / mean_pairwise

Under a random sentence order the expected successive distance IS the mean of
all pairwise distances, so **1.0 is a null by construction, not by
estimation**. Below 1 means successive sentences are closer than random pairs:
the passage is locally coherent.

`crosslingual_arms.md` proposes the DIFFERENCE, `mean_drift - mean_pairwise`.
That is not scale-free, and alignment is already established to shrink the
whole sentence set in Chinese -- so both terms shrink, their difference
shrinks with them, and the subtraction would carry the spread effect it exists
to remove. The difference is computed and reported as secondary.

THE GRAIN MIRRORS `m06_crosslingual_arms.py` EXACTLY: mean within
(pair, prompt, role), unstack, drop cells missing an arm, delta = aligned -
base, then the per-pair MEDIAN of deltas. A different grain would produce a
number that is not comparable to the finding it is meant to bear on.

**AND THE REPLICATION IS VERIFIED RATHER THAN ASSERTED.** Before reporting
anything new, this reproduces the published `total_drift` zh unmatched result
-- 21 of 25 negative, median -0.0314 -- from the same parquets. If that does
not come back the grain is wrong and the run aborts, because every new number
below would be wrong in the same way and none of them would look it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
OUT = os.path.join(OUTD, "zh_ordering.json")

#: the published value this replication must recover before anything else runs
BOOKED = {"metric": "total_drift", "lang": "zh", "matched": False,
          "neg": 21, "n": 25, "median": -0.0314, "tol": 5e-4}
FLUENT_THRESHOLDS = (0.0, 1.5, 2.0)
SCORE = {"fluent": 3, "flawed": 2, "broken": 1, "not_chinese": 0}


def sign_test(vals):
    from scipy import stats
    v = [x for x in vals if x == x]
    neg = sum(1 for x in v if x < 0)
    return {"n": len(v), "neg": neg,
            "median": float(sorted(v)[len(v) // 2]) if v else float("nan"),
            "p": float(stats.binomtest(neg, len(v), 0.5).pvalue) if v else float("nan")}


def fluency_scores():
    """model -> mean judged Chinese fluency, FIRST ratings only."""
    import collections
    import glob
    per = collections.defaultdict(list)
    for sp in sorted(glob.glob(os.path.join(OUTD, "zh_fluency_sample*.json"))):
        sfx = os.path.basename(sp)[len("zh_fluency_sample"):-len(".json")]
        vp = os.path.join(OUTD, "zh_fluency_verdicts%s.json" % sfx)
        if not os.path.exists(vp):
            continue
        truth = json.load(open(sp))["truth"]
        vd = json.load(open(vp))
        vd = vd["verdicts"] if isinstance(vd, dict) else vd
        for r in vd:
            t = truth.get(r.get("key"))
            if t and t.get("role", "new") == "new":
                per[t["model"]].append(SCORE[r["verdict"]])
    return {m: sum(x) / len(x) for m, x in per.items() if x}


def load(suf=""):
    """Read the drift cells. `suf` mirrors `m06_crosslingual_arms.py --full`.

    THE DEFAULT IS THE TRUNCATED VARIANT AND THAT IS NOT A PREFERENCE. The
    published 25-pair contrast is computed over the 75-word-truncation cells
    (26,981 passages); `_full` is the no-truncation variant, has 29,864, and
    writes to its own `*_full` outputs. Reading `_full` here returned 28 pairs
    and 24/28 at median -0.0174 against a booked 21/25 at -0.0314 -- a result
    that looks entirely plausible and is a different measurement. The
    replication gate caught it; nothing else would have.
    """
    import pandas as pd
    frames = []
    for lang in ("zh", "en"):
        p = os.path.join(OUTD,
                         "crosslingual_drift_%s%s_cells.parquet" % (lang, suf))
        frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True)

    #: `not ambiguous` is the arms producer's filter and omitting it gave 28
    #: pairs against the published 25 -- 24/28 at median -0.0221, which reads
    #: as a fine result and is a different population.
    pairs = [p for p in json.load(
        open(os.path.join(ROOT, "data/base_aligned_pairs.json")))
        if not p.get("ambiguous")]
    models = set(d.model)
    bylang = {l: set(d[d.lang == l].model) for l in ("zh", "en")}
    use = [p for p in pairs if p["base"] in models and p["aligned"] in models]
    use = [p for p in use
           if all(p[r] in bylang[l] for r in ("base", "aligned")
                  for l in ("zh", "en"))]
    role = {}
    for p in use:
        role[p["base"]] = (p["base"] + ">" + p["aligned"], "base")
        role[p["aligned"]] = (p["base"] + ">" + p["aligned"], "aligned")
    d["pair"] = [role[m][0] if m in role else None for m in d.model]
    d["role"] = [role[m][1] if m in role else None for m in d.model]
    d = d[d.pair.notna()].copy()

    #: n_sents < 2 leaves mean_pairwise undefined; those rows carry no
    #: ordering information at all and are dropped rather than imputed.
    d = d[d.mean_pairwise.notna() & (d.mean_pairwise > 0)].copy()
    d["order_ratio"] = d.mean_drift / d.mean_pairwise
    d["order_diff"] = d.mean_drift - d.mean_pairwise
    return d, use


def per_pair(d, metric, lang):
    """aligned - base per (pair, prompt), then the per-pair median."""
    s = d[d.lang == lang]
    g = s.groupby(["pair", "prompt", "role"])[metric].mean().unstack("role")
    g = g.dropna(subset=["aligned", "base"])
    delta = g["aligned"] - g["base"]
    return delta.groupby(level="pair").median()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="the no-truncation cells; NOT the published contrast")
    a = ap.parse_args()
    suf = "_full" if a.full else ""
    global OUT
    OUT = os.path.join(OUTD, "zh_ordering%s.json" % suf)

    d, use = load(suf)
    print("cells: %s%s | %d pairs | %s passages with a defined order_ratio"
          % ("crosslingual_drift_*", suf or " (75-word truncation)",
             len(use), format(len(d), ",")))

    #: ---- REPLICATION GATE ----
    chk = per_pair(d, BOOKED["metric"], BOOKED["lang"])
    st = sign_test(chk.values)
    ok = (st["neg"] == BOOKED["neg"] and st["n"] == BOOKED["n"]
          and abs(st["median"] - BOOKED["median"]) <= BOOKED["tol"])
    print("\nREPLICATION GATE: %s zh unmatched -> %d/%d negative, median %+.4f"
          % (BOOKED["metric"], st["neg"], st["n"], st["median"]))
    print("  booked: %d/%d, median %+.4f -> %s"
          % (BOOKED["neg"], BOOKED["n"], BOOKED["median"],
             "MATCH" if ok else "MISMATCH"))
    if not ok:
        raise SystemExit(
            "grain does not reproduce the published contrast; every number "
            "below would be wrong in the same way. Not proceeding.")

    out = {"plan": "plans/plan_zh_ordering.md",
           "replication_gate": {"booked": BOOKED, "observed": st},
           "P1_levels": {}, "P2_contrast": {}, "P3_comparison": {},
           "confound": {}}

    #: ---- P1: is the ratio below 1 at all? ----
    print("\nP1  order_ratio LEVELS (null = 1.0 exactly, by construction)")
    print("    %-5s %-9s %8s %9s %9s" % ("lang", "role", "n", "median", "mean"))
    for lang in ("zh", "en"):
        for r in ("base", "aligned"):
            s = d[(d.lang == lang) & (d.role == r)]["order_ratio"]
            out["P1_levels"]["%s:%s" % (lang, r)] = {
                "n": int(len(s)), "median": float(s.median()),
                "mean": float(s.mean())}
            print("    %-5s %-9s %8s %9.4f %9.4f"
                  % (lang, r, format(len(s), ","), s.median(), s.mean()))
    zh_med = out["P1_levels"]["zh:base"]["median"]
    print("    P1 %s: passages are locally %s"
          % ("HOLDS" if zh_med < 1 else "FAILS",
             "coherent" if zh_med < 1 else "ANTI-coherent"))

    #: ---- P2 / P3: the contrast, on both declared populations ----
    sc = fluency_scores()
    print("\nP2/P3  aligned - base, by metric and population")
    print("    %-12s %-5s %-14s %4s %7s %10s %10s"
          % ("metric", "lang", "population", "n", "neg", "median", "p"))
    for metric in ("order_ratio", "order_diff", "mean_drift", "total_drift"):
        for lang in ("zh", "en"):
            pm = per_pair(d, metric, lang)
            for thr in FLUENT_THRESHOLDS:
                keep = [p for p in pm.index
                        if thr == 0 or (
                            p.split(">")[0] in sc and p.split(">")[1] in sc
                            and min(sc[p.split(">")[0]],
                                    sc[p.split(">")[1]]) >= thr)]
                if len(keep) < 4:
                    continue
                st2 = sign_test(pm.loc[keep].values)
                tag = "ALL" if thr == 0 else "fluent>=%.1f" % thr
                out["P2_contrast"]["%s:%s:%s" % (metric, lang, tag)] = st2
                print("    %-12s %-5s %-14s %4d %3d/%-3d %10.4f %10.4g"
                      % (metric, lang, tag, st2["n"], st2["neg"], st2["n"],
                         st2["median"], st2["p"]))
        print()

    #: ---- the same confound test the fluency work ran on total_drift ----
    print("CONFOUND: does the fluency gap predict the ORDER gap? (zh)")
    from scipy import stats
    for metric in ("order_ratio", "order_diff"):
        pm = per_pair(d, metric, "zh")
        xs, ys = [], []
        for p, v in pm.items():
            b, a = p.split(">")
            if b in sc and a in sc and v == v:
                xs.append(sc[a] - sc[b])
                ys.append(float(v))
        if len(xs) >= 8:
            rho, pv = stats.spearmanr(xs, ys)
            out["confound"][metric] = {"n": len(xs), "spearman": float(rho),
                                       "p": float(pv)}
            print("    %-12s n=%d  spearman %+.3f  p=%.4g  -> %s"
                  % (metric, len(xs), rho, pv,
                     "tracks fluency" if pv < .05 else "independent of fluency"))

    json.dump(out, open(OUT, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
