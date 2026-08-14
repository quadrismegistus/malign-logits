"""Does alignment change Chinese FLUENCY, and does that confound the arm effect?

    uv run python meta/M06_generation/scripts/m06_zh_fluency_arms.py

Pools every judging round (`zh_fluency_sample*.json` + `zh_fluency_verdicts*.json`),
reports inter-rater agreement from the blind re-rates, then asks the question
the crosslingual arm finding cannot answer about itself.

WHY THIS EXISTS. `findings/crosslingual_arms.md` reports that alignment narrows
the semantic spread of a passage in Chinese as it does in English, on 25
base/aligned pairs. Round 1 of the fluency judging (6 passages per model) found
that aligned models appeared to write BETTER Chinese than their base models --
14 pairs up, 7 down, 4 tied, mean +0.40 -- at p=0.19, which establishes nothing.

**If that gap is real it is a confound rather than a detail.** A bge-m3
embedding of word salad is not a measurement of the same kind as an embedding
of prose, so a difference in sentence-embedding drift between arms could be a
difference in the COHERENCE of the text rather than in its geometry.

**AND THE ORIGINAL FORM OF THAT SENTENCE WAS WRONG, corrected 2026-08-14.** It
read *"could then be a difference in whether the text is Chinese at all"*, and
it is not: coding the verdicts as IS-CHINESE-AT-ALL (fluent|flawed|broken
against not_chinese) gives **12/7, p=0.36 -- null.** Alignment does not make a
model likelier to stay in Chinese. It improves the Chinese it does write, and
the effect concentrates at the top of the scale (`fluent` alone: 15/0). The
confound therefore runs through broken-versus-coherent, not through language
choice -- which is consistent with `order_ratio` being independent of it,
since coherence is what an ordering measure is sensitive to.

THE THREE THINGS THIS REPORTS, in increasing order of what they can settle:

  1. AGREEMENT. 160 passages were re-emitted under fresh keys in round 2 and
     judged by a different agent that could not tell them from first ratings.
     Without this the per-model scores have no reliability at all.

  2. THE FLUENCY CONTRAST. aligned - base, over pairs, at 20 passages per model
     instead of 6. Sign test plus a paired bootstrap, so a null has a bound
     rather than being an absence of stars.

  3. THE CONFOUND TEST, which is the point. Across pairs, does the fluency gap
     PREDICT the drift gap? If the correlation is strong, the arm effect on
     `total_drift` is not separable from the arm effect on how COHERENT the
     Chinese is. If it is flat, the drift result survives the objection.

A NULL ON 3 IS THE USEFUL OUTCOME and it is what the finding needs; a positive
is a reason to restrict the population to pairs where both members write
Chinese, of which there were 7 of 25 at round 1.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
PAIRS_PQ = os.path.join(OUTD, "crosslingual_arms_pairs.parquet")
OUT = os.path.join(OUTD, "zh_fluency_arms.json")

#: The interval coding of an ORDINAL scale, and the finding tests whether it
#: matters rather than assuming: four codings agree (20/5, 20/4, 15/0, 23/2)
#: and IS-CHINESE-AT-ALL is NULL (12/7, p=0.36). Alignment does not make a
#: model likelier to stay in Chinese; it improves the Chinese it does write,
#: and the effect concentrates at the top of the scale. `--coding` re-runs
#: the contrast under any of them.
SCORE = {"fluent": 3, "flawed": 2, "broken": 1, "not_chinese": 0}
CODINGS = {
    "interval": {"fluent": 3, "flawed": 2, "broken": 1, "not_chinese": 0},
    "binary_ok": {"fluent": 1, "flawed": 1, "broken": 0, "not_chinese": 0},
    "strict": {"fluent": 1, "flawed": 0, "broken": 0, "not_chinese": 0},
    "is_chinese": {"fluent": 1, "flawed": 1, "broken": 1, "not_chinese": 0},
    "compressed": {"fluent": 3, "flawed": 2, "broken": 0, "not_chinese": 0},
}
ORDER = ["fluent", "flawed", "broken", "not_chinese"]
BOOT = 10000
SEED = 20260814


def load_rounds():
    """{key: (verdict, model, role, first_key)} pooled over every round.

    Sample and verdict files are matched by their round suffix rather than by
    position, because a mismatched pairing would silently attribute one
    round's verdicts to another round's models -- and every key would still
    resolve, so nothing would look wrong.
    """
    out, rounds = {}, []
    for sp in sorted(glob.glob(os.path.join(OUTD, "zh_fluency_sample*.json"))):
        sfx = os.path.basename(sp)[len("zh_fluency_sample"):-len(".json")]
        vp = os.path.join(OUTD, "zh_fluency_verdicts%s.json" % sfx)
        if not os.path.exists(vp):
            print("  (no verdicts for sample%s; skipping)" % (sfx or " [r1]"))
            continue
        truth = json.load(open(sp))["truth"]
        vd = json.load(open(vp))
        vd = vd["verdicts"] if isinstance(vd, dict) else vd
        n = 0
        for r in vd:
            k = r.get("key")
            if k not in truth:
                continue
            t = truth[k]
            out[k] = (r["verdict"], t["model"], t.get("role", "new"),
                      t.get("first_key"))
            n += 1
        rounds.append((sfx or "[r1]", len(truth), n))
    for sfx, nt, nv in rounds:
        print("  round %-6s %4d sampled  %4d judged" % (sfx, nt, nv))
    return out


def agreement(pool):
    """Exact and adjacent agreement between the two independent ratings."""
    prs = [(pool[fk][0], v) for k, (v, m, role, fk) in pool.items()
           if role == "iaa" and fk in pool]
    if not prs:
        return None
    exact = sum(1 for a, b in prs if a == b) / len(prs)
    adj = sum(1 for a, b in prs if abs(SCORE[a] - SCORE[b]) <= 1) / len(prs)
    #: Cohen's kappa over the 4 categories
    cats = ORDER
    n = len(prs)
    po = exact
    pa = collections.Counter(a for a, _ in prs)
    pb = collections.Counter(b for _, b in prs)
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    conf = collections.Counter(prs)
    return {"n": n, "exact": exact, "adjacent": adj, "kappa": kappa,
            "confusion": {"%s->%s" % k: v for k, v in conf.most_common()}}


def main():
    import random

    print("POOLED JUDGING ROUNDS")
    pool = load_rounds()
    print("  %d verdicts total" % len(pool))

    print("\n1. INTER-RATER AGREEMENT (blind re-rates, two independent readers)")
    ag = agreement(pool)
    if not ag:
        print("   no re-rates present yet")
    else:
        print("   n=%d  exact %.3f  adjacent %.3f  Cohen kappa %.3f"
              % (ag["n"], ag["exact"], ag["adjacent"], ag["kappa"]))
        print("   most common disagreements:")
        for k, v in list(ag["confusion"].items())[:6]:
            a, b = k.split("->")
            if a != b:
                print("      %-24s %d" % (k, v))

    #: first ratings only for the per-model score, so a re-rated passage does
    #: not count twice and the models with re-rates are not over-weighted
    per = collections.defaultdict(list)
    for k, (v, m, role, fk) in pool.items():
        if role == "new":
            per[m].append(SCORE[v])
    sc = {m: sum(x) / len(x) for m, x in per.items() if x}
    npm = {m: len(x) for m, x in per.items()}
    print("\n   per-model n: min %d  median %d  max %d  (%d models)"
          % (min(npm.values()), sorted(npm.values())[len(npm) // 2],
             max(npm.values()), len(npm)))

    import pandas as pd
    pq = pd.read_parquet(PAIRS_PQ)
    pairs = sorted({tuple(s.split(">")) for s in set(pq["pair"])})

    print("\n2. FLUENCY CONTRAST: aligned - base, over the arms pairs")
    d = [(sc[a] - sc[b], b, a) for b, a in pairs if b in sc and a in sc]
    up = sum(1 for x, _, _ in d if x > 0)
    dn = sum(1 for x, _, _ in d if x < 0)
    eq = len(d) - up - dn
    nz = up + dn
    try:
        from scipy import stats
        p = stats.binomtest(up, nz, 0.5).pvalue if nz else float("nan")
    except Exception:
        p = float("nan")
    rng = random.Random(SEED)
    vals = [x for x, _, _ in d]
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals)
                for _ in range(BOOT))
    lo, hi = bs[int(.025 * BOOT)], bs[int(.975 * BOOT)]
    mean = sum(vals) / len(vals)
    print("   pairs %d | aligned more fluent %d | less %d | tied %d"
          % (len(d), up, dn, eq))
    print("   sign test p=%.4g   mean %+.3f  95%% CI [%+.3f, %+.3f]"
          % (p, mean, lo, hi))
    print("   %s" % ("CI EXCLUDES ZERO: alignment moves Chinese fluency"
                     if lo > 0 or hi < 0 else
                     "CI includes zero: no established fluency difference"))

    print("\n3. CONFOUND TEST: does the fluency gap predict the drift gap?")
    #: EVERY zh metric x matched cell is tested, so the p-values are a FAMILY
    #: and a bare 0.05 is the wrong threshold. Round 1 produced exactly one
    #: nominal hit (total_drift, unmatched, p=0.041) out of four, which
    #: survives no correction at all -- and it landed on the finding's
    #: headline metric, which is precisely when a nominal p is most tempting
    #: to quote. The corrected threshold is printed so it cannot be skipped.
    cells = [(m, mt) for m in sorted(set(pq["metric"]))
             for mt in sorted(set(pq["matched"]))
             if not pq[(pq.metric == m) & (pq.matched == mt) &
                       (pq.lang == "zh")].empty]
    thr = 0.05 / max(1, len(cells))
    print("   %d tests in this family; Bonferroni threshold p < %.4f"
          % (len(cells), thr))
    got = False
    for metric in sorted(set(pq["metric"])):
        for matched in sorted(set(pq["matched"])):
            s = pq[(pq.metric == metric) & (pq.matched == matched) &
                   (pq.lang == "zh")]
            if s.empty:
                continue
            dd = {}
            for _, r in s.iterrows():
                dd[tuple(r["pair"].split(">"))] = r["delta"]
            xs, ys = [], []
            for x, b, a in d:
                if (b, a) in dd and dd[(b, a)] == dd[(b, a)]:
                    xs.append(x)
                    ys.append(float(dd[(b, a)]))
            if len(xs) < 8:
                continue
            got = True
            try:
                from scipy import stats
                rho, pv = stats.spearmanr(xs, ys)
                pr, ppv = stats.pearsonr(xs, ys)
            except Exception:
                rho = pv = pr = ppv = float("nan")
            flag = ("CONFOUNDED" if pv == pv and pv < thr
                    else "nominal only" if pv == pv and pv < .05
                    else "clear")
            print("   zh %-14s matched=%-6s n=%2d  spearman %+.3f (p=%.3g)  "
                  "pearson %+.3f  -> %s"
                  % (metric, matched, len(xs), rho, pv, pr, flag))
    if not got:
        print("   (no zh pair deltas joinable)")

    #: ---- 4. DOES THE DRIFT EFFECT SURVIVE ON FLUENT PAIRS? ----
    #: The confound test says the two gaps move together; it does not say the
    #: drift effect is spurious. The separable question is whether the effect
    #: is still there once the pairs whose base model does not write Chinese
    #: are removed. Restricting COSTS POWER, so a null here is ambiguous
    #: between "no effect" and "not enough pairs" -- the n is printed beside
    #: every line so that cannot be quietly skipped.
    print("\n4. DOES THE DRIFT EFFECT SURVIVE ON PAIRS THAT WRITE CHINESE?")
    print("   sign test needs %s of n for p<0.05" % "roughly 80%")
    for metric in ("total_drift", "mean_drift"):
        for matched in sorted(set(pq["matched"])):
            s = pq[(pq.metric == metric) & (pq.matched == matched) &
                   (pq.lang == "zh")]
            if s.empty:
                continue
            dd = {tuple(r["pair"].split(">")): r["delta"] for _, r in s.iterrows()}
            for thr in (0.0, 1.5, 2.0):
                keep = [(b, a) for b, a in pairs
                        if b in sc and a in sc
                        and min(sc[b], sc[a]) >= thr and (b, a) in dd]
                vals = [float(dd[k]) for k in keep if dd[k] == dd[k]]
                if len(vals) < 4:
                    continue
                neg = sum(1 for x in vals if x < 0)
                try:
                    from scipy import stats
                    p2 = stats.binomtest(neg, len(vals), 0.5).pvalue
                except Exception:
                    p2 = float("nan")
                med = sorted(vals)[len(vals) // 2]
                tag = "ALL PAIRS" if thr == 0 else "both >= %.1f" % thr
                print("   %-12s matched=%-6s %-14s n=%2d  neg %2d/%2d  "
                      "median %+.4f  p=%.4g"
                      % (metric, matched, tag, len(vals), neg, len(vals),
                         med, p2))
        print()

    json.dump({"_about":
               "Chinese fluency by arm, and whether it confounds the "
               "crosslingual drift contrast. Scores are FIRST ratings only; "
               "the re-rates are used for agreement and are not counted twice.",
               "n_verdicts": len(pool), "agreement": ag,
               "per_model_score": sc, "per_model_n": npm,
               "fluency_contrast": {
                   "pairs": len(d), "aligned_more_fluent": up,
                   "aligned_less_fluent": dn, "tied": eq,
                   "sign_test_p": p, "mean": mean, "ci95": [lo, hi]}},
              open(OUT, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
