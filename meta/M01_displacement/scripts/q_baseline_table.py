"""q_baseline_table.py — THE CORPUS BASELINE Q WAS FOR AND DOES NOT CONTAIN.

**WHY IT EXISTS.** RH, verbatim at [4521]: *"The entire point of this test was
to give a baseline to substitution and norms so we can answer 'is it
transgressiveness or in general' and once again we registered a test that does
not answer it."* **He is right on the facts.**

Registration Q's two columns were reported as `vs NEUTRAL TWIN` and `vs CORPUS
AT LARGE`. **The second is `nonpair_transgressive` vs `nonpair_neutral`.** Both
of Q's substitution arms are transgressive-vs-neutral; they differ in
population and matching, **not in comparison target.** Neither is a baseline.
§Q7 says so plainly — *"different KINDS of contrast"* — **so the registration
was honest and the summary layer was not.**

**AND THE BASELINE SAT INSIDE Q AS A GATE THE WHOLE TIME.** §Q6's known answer
is the pooled corpus level. It was computed, verified to 5e-5, and argued over
for two hours — **and never once compared to anything.**

WHAT THIS FILE IS, AND IS NOT:

  **DESCRIPTIVE.** No alpha, no test, no verdict language, no branches.
  RH released the pre-registration requirement for this deliverable; that is
  **not** permission for sloppy arithmetic, so every number here is
  cluster-aware and every partition rule is imported rather than retyped.
  **It is an appendix to Q, not a successor to it.**

THE FOUR RULES IT DOES KEEP:

  1 **THE BASELINE EXCLUDES TRANSGRESSIVE CELLS.** A pooled level containing
    the arm under test is contaminated by it. This is the one piece of rigor
    not relaxed.
  2 **PAIR SET AND RESIDUE ALWAYS SEPARATE.** `pair_marked` and
    `nonpair_transgressive` are both "transgressive" and they do not behave
    alike; pooling them hides the thing that explains H2.
  3 **INTERVALS CLUSTERED BY MODEL.** Cells within a checkpoint are not
    independent; a cell-level interval on 82,775 cells would be a fiction.
  4 **THE PARTITION RULE IS IMPORTED FROM `p_yield_pass.partition_map`**, the
    producer that made §Q1.3's published counts. Retyping it would make this
    table a sibling of the registration's rather than an instance of it.

`tail_excess_corrected` is READ from N's artifact (§Q2's provenance split:
the correction is defined by `n_primary.py` and re-implementing a sibling
registration's definition is the move §Q6 rejected). `departed` and
`A_|valence|` are MACHINERY quantities and are computed from the pinned
module.
"""
import argparse
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_ART = os.path.join(CAMPAIGN, "results", "result_n_primary.json")
OUT = os.path.join(ROOT, "data", "q_baseline_table.json")

MOVEMENT_REL = "malign_logits/movement.py"
MOVEMENT_PIN_COMMIT = "e7864dab"

TRANSGRESSIVE_PARTS = ("pair_marked", "nonpair_transgressive")

ORDER = ["pair_marked", "nonpair_transgressive", "pair_unmarked",
         "nonpair_neutral", "nonpair_institutional", "nonpair_literary",
         "nonpair_contradiction", "nonpair_other"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def cluster_interval(by_cluster, conf=0.95):
    """Mean and interval with the CLUSTER as the unit (rule 3).

    Cells within a checkpoint are not independent, so the cell count is not
    the sample size. Each cluster contributes one mean; the interval is over
    those. Returns (grand mean over cells, lo, hi, n_clusters).
    """
    from statistics import NormalDist
    cm = [mean(v) for v in by_cluster.values() if v]
    k = len(cm)
    allc = [x for v in by_cluster.values() for x in v]
    if k < 2:
        return mean(allc), float("nan"), float("nan"), k, mean(allc)
    m = mean(cm)
    sd = math.sqrt(sum((x - m) ** 2 for x in cm) / (k - 1))
    half = NormalDist().inv_cdf(1 - (1 - conf) / 2) * sd / math.sqrt(k)
    #: **TWO ESTIMATORS AND THE CHOICE WAS UNDECLARED ON BOTH SIDES.**
    #: `mean(allc)` is CELL-WEIGHTED (big clusters count more); `m` is the
    #: UNWEIGHTED mean of cluster means. They differ whenever clusters hold
    #: unequal cell counts, which they do. This is U2's unweighted-vs-
    #: inverse-variance choice arriving in a descriptive table. **The
    #: interval is built on the cluster means, so the estimator that MATCHES
    #: it is the unweighted one** -- returned as the point estimate, with
    #: the cell-weighted figure carried beside it rather than hidden.
    return m, m - half, m + half, k, mean(allc)


def gate_movement_blob():
    import subprocess
    wt = subprocess.run(["git", "hash-object", MOVEMENT_REL],
                        capture_output=True, text=True, cwd=ROOT).stdout.strip()
    pin = subprocess.run(["git", "rev-parse", "%s:%s" % (MOVEMENT_PIN_COMMIT, MOVEMENT_REL)],
                         capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not wt or wt != pin:
        raise SystemExit("REFUSING: movement.py blob %s != pinned %s" % (wt[:16], pin[:16]))
    return wt


def report(title, measure_rows, fh_payload, key):
    """One measure's table. Transgressive rows first, baseline last."""
    print("\n" + "=" * 78)
    print("%s" % title)
    print("=" * 78)
    print("  %-24s %7s  %11s  %-22s %s"
          % ("partition", "cells", "level(unwtd)", "95% CI (cluster)", "clus  level(cellwtd)"))
    for name in ORDER:
        if name not in measure_rows:
            continue
        m, lo, hi, k, n, cw = measure_rows[name]
        mark = " **" if name in TRANSGRESSIVE_PARTS else "   "
        print("%s%-24s %7d  %+11.6f  [%+.6f, %+.6f]  %d   %+11.6f"
              % (mark, name, n, m, lo, hi, k, cw))
    fh_payload[key] = {n: {"level_unweighted_cluster_mean": v[0],
                           "ci95": [v[1], v[2]], "n_clusters": v[3],
                           "n_cells": v[4], "level_cell_weighted": v[5]}
                       for n, v in measure_rows.items()}


def contrast(rows, a, b):
    return rows[a][0] - rows[b][0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail-only", action="store_true",
                    help="skip the machinery pass (tail_excess is read, not computed)")
    args = ap.parse_args(argv)

    blob = gate_movement_blob()
    print("movement.py blob %s == @%s" % (blob[:16], MOVEMENT_PIN_COMMIT))

    #: rule 4 -- IMPORT the partition rule from the producer of SQ1.3's counts.
    import p_yield_pass as PY
    lab = PY.partition_map()
    #: **partition_map LABELS TEXTS; IT DOES NOT DEFINE THE POPULATION.** It
    #: carries no sentinel/CJK filter, so it holds 2,579 texts against N's
    #: published 2,199. Reading `tail_excess` from N's ARTIFACT hides this --
    #: the artifact is already population-filtered -- but the MACHINERY pass
    #: iterates texts directly and reached `<<<LOGICAL:BOS>>>`, whose twp row
    #: carries a NaN and refuses by design. **The fix is the registration's
    #: own population rule (N SS3/SS3.0: sentinels out, zh out), not a skip:
    #: the cell was never in the population.**
    pop = PY.english_stimuli()
    lab = {t: v for t, v in lab.items() if t in pop}
    print("partition rule imported from p_yield_pass.partition_map; "
          "intersected with english_stimuli(): %d texts" % len(lab))
    if len(lab) != 2199:
        raise SystemExit("REFUSING: population is %d texts, not N's published "
                         "2,199" % len(lab))

    art = json.load(open(N_ART))
    cells = art["cells"]
    print("N artifact: %d analysed cells" % len(cells))

    payload = {"_what": "Corpus baseline for Q's measures, by SQ1 partition.",
               "_why": "Q contains no corpus baseline; RH [4521].",
               "_status": "DESCRIPTIVE. No alpha, no test, no verdict language.",
               "_rules": [
                   "baseline EXCLUDES transgressive cells (pair_marked, "
                   "nonpair_transgressive) -- a pooled level containing the "
                   "arm under test is contaminated by it",
                   "pair set and residue reported SEPARATELY, never pooled",
                   "intervals clustered by base checkpoint; cells within a "
                   "checkpoint are not independent",
                   "partition rule imported from p_yield_pass.partition_map",
               ],
               "_pins": {"movement.py_blob": blob}}

    # ---- SUBSTITUTION: tail_excess_corrected, READ from N's artifact -------
    sub = collections.defaultdict(lambda: collections.defaultdict(list))
    subneg = collections.defaultdict(lambda: [0, 0])
    for c in cells:
        p = lab.get(c["prompt"])
        if p is None:
            continue
        sub[p][c["base"]].append(c["tail_excess_corrected"])
        subneg[p][0] += 1
        subneg[p][1] += 1 if c["tail_excess_corrected"] < 0 else 0
    subrows = {}
    for p, byc in sub.items():
        m, lo, hi, k, cw = cluster_interval(byc)
        subrows[p] = (m, lo, hi, k, subneg[p][0], cw)
    report("SUBSTITUTION — `tail_excess_corrected`, read from N's artifact",
           subrows, payload, "substitution")
    print("\n  %-24s %7s  %s" % ("partition", "cells", "% of cells substituting"))
    for name in ORDER:
        if name in subneg:
            n, neg = subneg[name]
            print("  %-24s %7d  %5.1f%%" % (name, n, 100.0 * neg / n))
    payload["substitution_pct_negative"] = {k: 100.0 * v[1] / v[0]
                                            for k, v in subneg.items()}

    if args.tail_only:
        json.dump(payload, open(OUT, "w"), indent=1, sort_keys=True)
        print("\nwrote %s (tail only)" % OUT)
        return 0

    # ---- MAGNITUDE and NORMS: machinery quantities ------------------------
    from malign_logits.movement import CANONICAL
    import m01_concentration as CC
    import m01_norms as N
    import m01_registration_b as B

    norms, _f, _r = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")] for d in ("arousal", "valence")}

    def wmean(v, w):
        s = sum(w)
        return sum(a * b for a, b in zip(v, w)) / s if s > 0 else None

    _p, mods, _h, _d = CC.frozen_population()
    edges_raw, _drop = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    steps = {}
    for _fam, _pos, st in edges_raw:
        steps.setdefault((mid(st.pre), mid(st.post)), st)

    dep = collections.defaultdict(lambda: collections.defaultdict(list))
    nrm = collections.defaultdict(lambda: collections.defaultdict(list))
    texts = sorted(lab)
    for ei, ((b_, a_), st) in enumerate(sorted(steps.items()), 1):
        for t in texts:
            p = lab[t]
            c = st.cell(t)
            if not c.is_present:
                continue
            try:
                d_ = c.decompose(None)
            except Exception:
                continue
            if not d_:
                continue
            try:
                roles = N.cell_roles(c, CANONICAL)
            except Exception:
                continue
            if roles is None or not any(r == "faller" for _w, _wt, r in roles):
                continue
            dep[p][b_].append(float(d_["departed"]))
            wf, zf, wr, zr = [], [], [], []
            for w, wt, role in roles:
                key = N.norm_key(w, "en", fold=False)
                if N.is_function_word(key, "en"):
                    continue
                zv = {}
                for dim in ("arousal", "valence"):
                    val, _s = N.lookup(tabs[dim], key.casefold(), "en")
                    zv[dim] = val
                if any(x is None for x in zv.values()):
                    continue
                if role == "faller":
                    wf.append(wt); zf.append(abs(zv["valence"]))
                else:
                    wr.append(wt); zr.append(abs(zv["valence"]))
            if len(wf) >= B.QUALIFYING_MIN and len(wr) >= B.QUALIFYING_MIN:
                mf, mr = wmean(zf, wf), wmean(zr, wr)
                if mf is not None and mr is not None:
                    nrm[p][b_].append(mf - mr)
        print("  [%2d/%d] edges done" % (ei, len(steps)), flush=True)

    for label, store, key in (("MAGNITUDE — `departed`, machinery", dep, "magnitude"),
                              ("NORMS — `A_|valence|`, machinery", nrm, "norms")):
        rows = {}
        for p, byc in store.items():
            m, lo, hi, k, cw = cluster_interval(byc)
            rows[p] = (m, lo, hi, k, sum(len(v) for v in byc.values()), cw)
        report(label, rows, payload, key)

    # ---- THE CONTRAST RH ASKED FOR ---------------------------------------
    print("\n" + "=" * 78)
    print("IS IT TRANSGRESSIVENESS, OR IN GENERAL?")
    print("=" * 78)
    print("  BASELINE = every non-transgressive partition pooled; the two")
    print("  transgressive partitions are EXCLUDED from it (rule 1), and the")
    print("  pair set and the residue are never pooled with each other (rule 2).")
    summary = {}
    for key, store in (("substitution", sub), ("magnitude", dep), ("norms", nrm)):
        if key not in payload:
            continue
        base_c = collections.defaultdict(list)
        for p, byc in store.items():
            if p in TRANSGRESSIVE_PARTS:
                continue
            for cl, v in byc.items():
                base_c[cl].extend(v)
        bm, blo, bhi, bk, _bcw = cluster_interval(base_c)
        print("\n  %s" % key.upper())
        print("    baseline (non-transgressive)   %+.6f  [%+.6f, %+.6f]  %d clusters"
              % (bm, blo, bhi, bk))
        summary[key] = {"baseline": bm, "baseline_ci95": [blo, bhi]}
        for p in TRANSGRESSIVE_PARTS:
            if p not in store:
                continue
            m, lo, hi, k, _n = (payload[key][p]["level_unweighted_cluster_mean"], payload[key][p]["ci95"][0],
                                payload[key][p]["ci95"][1], payload[key][p]["n_clusters"],
                                payload[key][p]["n_cells"])
            print("    %-28s %+.6f  [%+.6f, %+.6f]   vs baseline %+.6f (%.2fx)"
                  % (p, m, lo, hi, m - bm, (m / bm) if bm else float("nan")))
            summary[key][p] = {"level": m, "minus_baseline": m - bm,
                               "ratio_to_baseline": (m / bm) if bm else None}
    # ---- THE PAIRED CONTRAST, WHICH IS THE CORRECT ONE -------------------
    #: **THE LEVEL INTERVALS ABOVE ARE NOT THE COMPARISON.** The SAME 34
    #: checkpoints appear in every partition, so partition levels are paired,
    #: not independent samples. Overlapping level-CIs do not mean the
    #: difference is indistinguishable -- the between-cluster variance that
    #: dominates each level CANCELS in a within-cluster difference. This is
    #: SQ2's own pairing argument applied to the baseline table.
    print("\n" + "=" * 78)
    print("PAIRED WITHIN-CLUSTER: transgressive partition MINUS its own")
    print("cluster's non-transgressive baseline. The level CIs above are NOT")
    print("this comparison; the same 34 checkpoints appear in every row, so")
    print("the between-cluster variance cancels here and not there.")
    print("=" * 78)
    #: **TWO BASELINES, BOTH REPORTED, BECAUSE THE ANSWER IS NOT ROBUST TO
    #: A CHOICE NOBODY MADE ON PURPOSE ([4527]).**
    #:
    #:   A  all non-transgressive        49,173 cells -- but **52.6% of it
    #:      is `pair_unmarked`**, the minimal pairs' own neutral twins:
    #:      sentences written FOR this experiment, matched to the marked
    #:      members on topic, length, register and syntax.
    #:   B  A minus `pair_unmarked`      23,324 cells -- removes the
    #:      constructed control condition, and is then **~50%
    #:      institutional**, which is its own composition problem.
    #:
    #: **NEITHER IS CLEAN AND THE SIGN OF `nonpair_transgressive` DIFFERS
    #: BETWEEN THEM.** The rule at [4521] said the baseline must exclude
    #: the arm under test; the contaminant is not "transgressive" but
    #: **CONSTRUCTED FOR THIS EXPERIMENT**, and under-scoping that is what
    #: put a sign-unstable sentence one post away from RH.
    BASELINES = (("A: all non-transgressive", TRANSGRESSIVE_PARTS),
                 ("B: also excluding pair_unmarked",
                  TRANSGRESSIVE_PARTS + ("pair_unmarked",)))
    paired = {}
    for key, store in (("substitution", sub), ("magnitude", dep), ("norms", nrm)):
        if key not in payload:
            continue
        print("\n  %s" % key.upper())
        paired[key] = {}
        for bname, excluded in BASELINES:
            base_c = collections.defaultdict(list)
            ncells = 0
            for p_, byc in store.items():
                if p_ in excluded:
                    continue
                for cl, v in byc.items():
                    base_c[cl].extend(v); ncells += len(v)
            print("    %s  (%d cells)" % (bname, ncells))
            paired[key][bname] = {}
            for tp in TRANSGRESSIVE_PARTS:
                if tp not in store:
                    continue
                d_ = {cl: [mean(store[tp][cl]) - mean(base_c[cl])]
                      for cl in store[tp] if store[tp][cl] and base_c.get(cl)}
                if len(d_) < 2:
                    continue
                m, lo, hi, k, _c = cluster_interval(d_)
                npos = sum(1 for v in d_.values() if v[0] > 0)
                print("      %-24s %+.6f  [%+.6f, %+.6f]  %2d/%2d pos  %s"
                      % (tp, m, lo, hi, npos, k,
                         "EXCLUDES 0" if (lo > 0 or hi < 0) else "**includes 0**"))
                paired[key][bname][tp] = {"paired_diff": m, "ci95": [lo, hi],
                                          "n_clusters": k, "n_positive": npos,
                                          "excludes_zero": (lo > 0 or hi < 0)}
            vals = [paired[key][bname][t]["paired_diff"]
                    for t in TRANSGRESSIVE_PARTS if t in paired[key][bname]]
            if len(vals) == 2:
                print("      -> the two transgressive partitions fall on %s"
                      % ("**OPPOSITE SIDES**" if vals[0] * vals[1] < 0
                         else "the SAME side"))
        #: the sign-stability check, stated per measure.
        try:
            a = paired[key][BASELINES[0][0]]["nonpair_transgressive"]["paired_diff"]
            b = paired[key][BASELINES[1][0]]["nonpair_transgressive"]["paired_diff"]
            print("      **SIGN STABLE ACROSS BASELINES: %s** (%+.6f vs %+.6f)"
                  % ("YES" if a * b > 0 else "**NO**", a, b))
            paired[key]["_sign_stable"] = bool(a * b > 0)
        except KeyError:
            pass
    if False:
        for tp in TRANSGRESSIVE_PARTS:
            if tp not in store:
                continue
            d_ = {cl: [mean(store[tp][cl]) - mean(base_c[cl])]
                  for cl in store[tp]
                  if store[tp][cl] and base_c.get(cl)}
            m, lo, hi, k, _cw = cluster_interval(d_)
            npos = sum(1 for v in d_.values() if v[0] > 0)
            print("    %-24s %+.6f  [%+.6f, %+.6f]  %d clusters, %d/%d positive"
                  % (tp, m, lo, hi, k, npos, k))
            paired[key][tp] = {"paired_diff": m, "ci95": [lo, hi],
                               "n_clusters": k, "n_positive": npos}
        if all(t in paired[key] for t in TRANSGRESSIVE_PARTS):
            a, b = (paired[key][t]["paired_diff"] for t in TRANSGRESSIVE_PARTS)
            print("    **the two transgressive partitions differ by %+.6f "
                  "and %s the baseline**"
                  % (a - b, "fall on OPPOSITE SIDES of" if a * b < 0
                     else "fall on the SAME side of"))
    #: [4526]: institutional magnitude looks 33% above transgressive on
    #: OVERLAPPING level intervals. The paired contrast is the test nobody
    #: had run, and it is one loop.
    print("\n" + "=" * 78)
    print("PAIRED WITHIN-CLUSTER, PARTITION vs PARTITION — [4526]'s lead")
    print("=" * 78)
    head = {}
    for key, store in (("substitution", sub), ("magnitude", dep), ("norms", nrm)):
        if key not in payload:
            continue
        print("\n  %s" % key.upper())
        head[key] = {}
        for a, b in (("nonpair_institutional", "pair_marked"),
                     ("nonpair_literary", "pair_marked"),
                     ("nonpair_contradiction", "pair_marked"),
                     ("nonpair_other", "pair_marked"),
                     ("pair_marked", "nonpair_transgressive"),
                     ("nonpair_institutional", "nonpair_transgressive"),
                     ("nonpair_contradiction", "nonpair_transgressive"),
                     ("nonpair_other", "nonpair_transgressive")):
            if a not in store or b not in store:
                continue
            d_ = {cl: [mean(store[a][cl]) - mean(store[b][cl])]
                  for cl in store[a]
                  if store[a][cl] and store[b].get(cl)}
            if len(d_) < 2:
                continue
            m, lo, hi, k, _cw = cluster_interval(d_)
            npos = sum(1 for v in d_.values() if v[0] > 0)
            excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
            print("    %-22s - %-22s %+.6f  [%+.6f, %+.6f]  %2d cl, %2d/%2d pos  %s"
                  % (a.replace("nonpair_", ""), b.replace("nonpair_", "").replace("pair_", ""),
                     m, lo, hi, k, npos, k, excl))
            head[key]["%s_minus_%s" % (a, b)] = {
                "diff": m, "ci95": [lo, hi], "n_clusters": k,
                "n_positive": npos, "excludes_zero": (lo > 0 or hi < 0)}
    #: **THE ONLY MATCHED NUMBER IN THIS FILE.** Q's own within-pair arms
    #: are the sole contrasts with a control behind them; every partition
    #: difference above is UNMATCHED. Recorded so the ratio is a field
    #: rather than a sentence someone computes later.
    MATCHED = {"substitution": ("H1", -0.002313),
               "magnitude": ("H5", +0.005260),
               "norms": ("H6", +0.017400)}
    print("\n" + "=" * 78)
    print("SCALE: the MATCHED within-pair effect against the UNMATCHED")
    print("differences between prompt collections. The matched number is the")
    print("only one with a control behind it; every other row is unmatched.")
    print("=" * 78)
    for key in ("substitution", "magnitude", "norms"):
        if key not in head:
            continue
        arm, mv = MATCHED[key]
        print("\n  %s   matched %s = %+.6f" % (key.upper(), arm, mv))
        for name, rec in sorted(head[key].items(),
                                key=lambda kv: -abs(kv[1]["diff"])):
            if not rec["excludes_zero"]:
                continue
            print("    %-52s %+.6f  %5.1fx"
                  % (name.replace("nonpair_", "").replace("pair_", ""),
                     rec["diff"], abs(rec["diff"] / mv)))
        head[key]["_matched_reference"] = {"arm": arm, "value": mv}
    payload["paired_partition_contrasts"] = head
    payload["paired_within_cluster"] = paired
    payload["baseline_contrast"] = summary
    payload["_limits"] = [
        "DESCRIPTIVE. No hypothesis was registered for any number here and "
        "no verdict word attaches to any of them.",
        "The partitions are not matched on length, topic or anything else; "
        "a level difference between them is not an effect of "
        "transgressiveness alone.",
        "Intervals treat the base checkpoint as the unit. Within-cluster "
        "dependence is carried; between-partition dependence (the same "
        "checkpoint appears in every row) is NOT.",
    ]
    json.dump(payload, open(OUT, "w"), indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
