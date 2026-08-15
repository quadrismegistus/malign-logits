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
#: bootstrap seed. Declared here rather than borrowed: the first version of
#: the interval block referenced `SEED` from a sibling producer's namespace
#: and raised NameError -- which is the right failure, since an unseeded
#: bootstrap would have produced a different CI on every run and nothing
#: would have looked wrong.
SEED = 20260814
BOOT = 20000
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

    #: ---- 5. PUT THE INTERVAL INSIDE THE COMPARISON ----
    #: malign at [6188]: *freezing a rule does not supply it an error bar*.
    #: This finding's headline compared two CHANGES under restriction
    #: (-0.0314 -> -0.0046 against -0.0090 -> -0.0087) with no uncertainty on
    #: either, having disclaimed the restricted p-values and then leaned on
    #: restricted POINT ESTIMATES from the same 6 pairs. Both quantities get
    #: a paired bootstrap: one resample of PAIRS per replicate, both metrics
    #: computed on it, so the two are correlated exactly as the data are.
    print("\n5. INTERVALS ON THE COMPARISONS THEMSELVES")
    import random as _rnd
    tab = {}
    for m in ("total_drift", "order_ratio"):
        for p, v in per_pair(d, m, "zh").items():
            if v == v:
                tab.setdefault(p, {})[m] = float(v)
    allp = [p for p, v in tab.items() if len(v) == 2]
    flu = [p for p in allp
           if p.split(">")[0] in sc and p.split(">")[1] in sc
           and min(sc[p.split(">")[0]], sc[p.split(">")[1]]) >= 2.0]

    def _med(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    B, rr = BOOT, _rnd.Random(SEED)
    chg = {"total_drift": [], "order_ratio": []}
    dif = []
    for _ in range(B):
        A = [allp[rr.randrange(len(allp))] for _ in allp]
        F = [flu[rr.randrange(len(flu))] for _ in flu]
        row = {}
        for m in chg:
            row[m] = _med([tab[p][m] for p in F]) - _med([tab[p][m] for p in A])
            chg[m].append(row[m])
        dif.append(row["total_drift"] - row["order_ratio"])

    def _ci(v):
        v = sorted(v)
        return _med(v), v[int(.025 * len(v))], v[int(.975 * len(v))]

    out["intervals"] = {}
    for m in ("total_drift", "order_ratio"):
        c, lo, hi = _ci(chg[m])
        out["intervals"]["change_%s" % m] = {"est": c, "lo": lo, "hi": hi}
        print("   change under restriction %-12s %+.4f [%+.4f, %+.4f] %s"
              % (m, c, lo, hi, "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"))
    c, lo, hi = _ci(dif)
    out["intervals"]["difference_of_changes"] = {"est": c, "lo": lo, "hi": hi}
    print("   DIFFERENCE OF CHANGES        %+.4f [%+.4f, %+.4f] -> %s"
          % (c, lo, hi, "ESTABLISHED" if lo > 0 or hi < 0 else "NOT ESTABLISHED"))

    #: the confound-correlation difference, which is the leg that survives
    gap = {p: sc[p.split(">")[1]] - sc[p.split(">")[0]] for p in allp
           if p.split(">")[0] in sc and p.split(">")[1] in sc}
    P = [p for p in allp if p in gap]
    rt = stats.spearmanr([gap[p] for p in P],
                         [tab[p]["total_drift"] for p in P]).statistic
    ro = stats.spearmanr([gap[p] for p in P],
                         [tab[p]["order_ratio"] for p in P]).statistic
    ds = []
    for _ in range(B):
        S = [P[rr.randrange(len(P))] for _ in P]
        g = [gap[p] for p in S]
        if len(set(g)) < 3:
            continue
        a = stats.spearmanr(g, [tab[p]["total_drift"] for p in S]).statistic
        b = stats.spearmanr(g, [tab[p]["order_ratio"] for p in S]).statistic
        if a == a and b == b:
            ds.append(a - b)
    c, lo, hi = _ci(ds)
    out["intervals"]["confound_rho_difference"] = {
        "total_drift_rho": float(rt), "order_ratio_rho": float(ro),
        "est": c, "lo": lo, "hi": hi, "n_pairs": len(P)}
    print("   confound rho difference (n=%d) %+.3f [%+.3f, %+.3f] -> %s"
          % (len(P), c, lo, hi,
             "ESTABLISHED" if lo > 0 or hi < 0 else "NOT ESTABLISHED"))
    print("   -> the dissociation rests on the CORRELATIONS at n=25, not on")
    print("      the restriction at n=6.")

    json.dump(out, open(OUT, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
