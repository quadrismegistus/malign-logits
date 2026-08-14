"""M04/A — DOSE-RESPONSE: is the demotion contrast graded, and by what?

The matched control established that A's effect is about DEMOTION rather than
mere improbability, and the four-term decomposition then showed that 86% of it
is visible to the BASE model — the moving terms are the base-text ones (B|A,
B|B), not the aligned-text ones. That reframes the question from "how much does
alignment damage the chain" to "WHERE does alignment choose to act", and a
categorical answer cannot settle it. This file asks the continuous version.

FOUR TESTS, ONE QUERY PASS. The ClickHouse join is the expensive step, so
`four_terms()` runs once and every test below is arithmetic on its per-site
output.

  1. DEMOTION MAGNITUDE. RH's question, and it has never been run. F13
     correlated PARADIGMATIC SIMILARITY against syntagmatic_js -- the quality of
     the substitute, not the size of the demotion -- and its r values are
     QUARANTINED pending registered re-analysis ([399]/[400]). F14's
     content-grading is categorical. **No test in this campaign has correlated
     how far a word fell against how much the chain moved.**
  2. BASE-PROBABILITY GAP, ON THE B TERMS DIRECTLY. The earlier stratification
     defended the ALIGNED composite and left the B terms untested, and the B
     terms are exactly what a base-probability confound would inflate: the
     faller outranks its control in base probability on 99.9% of sites. This is
     the test that decides whether there is a claim here at all.
  3. DOMAIN. Seven, of which `sexual`/`violence`/`taboo` are transgressive and
     `animal`/`betrayal`/`property`/`power` are not. F14 predicts grading.
  4. K NORMS. `fields.k_rating` -- charge, concreteness, transgressiveness,
     bodily_harm, valence. **RANKS, NOT LEVELS** (fields.py:99): charge and
     concreteness move in LEVEL between instrument versions while holding their
     ORDER at r 0.88, so everything here is Spearman or a rank split and nothing
     is thresholded on an absolute value. `register_level` is NOT ESTABLISHED
     and `vulgarity` is a sparse indicator whose floor effects are not nulls --
     both are carried as description and neither may be read as evidence.

UNIT. The lineage, as everywhere in this campaign: correlate WITHIN a pair,
then sign-test the 42 pair-level rho values. A pooled correlation over ~4,000
sites would count 42 lineages as thousands of independent observations, which is
the error class the campaign has corrected three times.
"""

import json, math, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import a_matched_control as A
from malign_logits import fields

ARMS = "data/forced_arms_46reps_drmatch.json"
K_USABLE = ("charge", "concreteness", "transgressiveness", "bodily_harm", "valence")
K_DESCRIPTIVE = ("register_level", "vulgarity")


def spearman(xs, ys):
    n = len(xs)
    if n < 8:
        return None
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def arms_table():
    d = json.load(open(ARMS))
    return d["cells"] if isinstance(d, dict) and "cells" in d else d


def site_frame(k=8, comparison="faller-matched"):
    """One row per site: the outcomes, and every covariate joined onto them."""
    per, sites = A.four_terms(k, comparison, want_sites=True)
    meta = {}
    for c in arms_table():
        mq, md, fp, fq = (c.get("matched_q"), c.get("matched_delta"),
                          c.get("faller_p"), c.get("faller_q"))
        row = {"domain": c.get("domain"), "stratum": c.get("stratum"),
               "faller": c.get("faller"), "matched": c.get("matched"),
               "class_match": c.get("class_match")}
        # demotion magnitude: how far the faller fell, in log2. NEGATIVE = fell.
        row["demotion_log2"] = (math.log2(fq / fp)
                                if fp and fq and fp > 0 and fq > 0 else None)
        row["demotion_abs"] = c.get("faller_delta")
        mp = (mq - md) if (mq is not None and md is not None) else None
        row["basegap_log2"] = (math.log2(fp / mp)
                               if fp and mp and fp > 0 and mp > 0 else None)
        for sc in K_USABLE + K_DESCRIPTIVE:
            fv = fields.k_rating(c["faller"], sc) if c.get("faller") else None
            mv = fields.k_rating(c["matched"], sc) if c.get("matched") else None
            row["k_faller_" + sc] = fv
            row["k_contrast_" + sc] = (fv - mv) if (fv is not None and mv is not None) else None
        meta[(c["pair"], c["prompt"])] = row
    rows = []
    for pair, d in sites.items():
        for i, pr in enumerate(d["prompts"]):
            m = meta.get((pair, pr))
            if m is None:
                continue
            r = dict(m)
            r["pair"], r["prompt"] = pair, pr
            r["D_aligned"], r["D_base"] = d["aligned"][i], d["base"][i]
            r["TRIPLE"] = d["triple"][i]
            for t, v in d["terms"][i].items():
                r["t_" + t] = v
            rows.append(r)
    return per, rows


def by_pair_rho(rows, xcol, ycol, min_sites=8):
    """Spearman within each pair, then a sign test on the pair-level rhos."""
    bypair = {}
    for r in rows:
        if r.get(xcol) is None or r.get(ycol) is None:
            continue
        bypair.setdefault(r["pair"], []).append((r[xcol], r[ycol]))
    rhos = []
    for pair, v in bypair.items():
        if len(v) < min_sites:
            continue
        rho = spearman([a for a, _ in v], [b for _, b in v])
        if rho is not None:
            rhos.append(rho)
    if len(rhos) < 5:
        return None
    st = A.sign_test(rhos)
    return {"n_pairs": len(rhos), "median_rho": statistics.median(rhos),
            "neg": st["neg"], "pos": st["pos"], "p": st["p"]}


def by_pair_median(rows, col, subset=None):
    """Pair medians of one column, then a sign test — the campaign's unit."""
    bypair = {}
    for r in rows:
        if subset and not subset(r):
            continue
        if r.get(col) is None:
            continue
        bypair.setdefault(r["pair"], []).append(r[col])
    vals = [statistics.median(v) for v in bypair.values() if len(v) >= 3]
    if len(vals) < 5:
        return None
    st = A.sign_test(vals)
    return {"n_pairs": len(vals), "median": statistics.median(vals),
            "neg": st["neg"], "pos": st["pos"], "p": st["p"]}


def fmt(d, lab, w=30):
    if d is None:
        return "  %-*s %s" % (w, lab, "(too few pairs)")
    key = "median_rho" if "median_rho" in d else "median"
    return "  %-*s %+9.4f %5d/%-4d %4d %9s" % (
        w, lab, d[key], d["neg"], d["pos"], d["n_pairs"],
        ("%.4f" % d["p"]) if d["p"] is not None else "-")


def main(k=8):
    per, rows = site_frame(k)
    out = {"k": k, "n_sites": len(rows),
           "_fingerprint": A.fingerprint({"k": k, "arms": ARMS}),
           "k_meta": {kk: fields.k_meta().get(kk)
                      for kk in ("instrument_sha256", "model", "n_words", "built")}}
    hdr = "  %-30s %9s %10s %4s %9s" % ("", "median", "neg/pos", "prs", "p(sign)")

    print("\n=== 1. DEMOTION MAGNITUDE (never run before) ===")
    print("  Spearman WITHIN pair, then sign test on pair rhos.")
    print("  demotion_log2 is NEGATIVE for a bigger fall, so a POSITIVE rho with")
    print("  D means bigger demotion -> more negative D -> stronger effect.")
    print(hdr)
    t1 = {}
    for y in ("D_aligned", "D_base", "TRIPLE", "t_B|A", "t_A|A"):
        t1[y] = by_pair_rho(rows, "demotion_log2", y)
        print(fmt(t1[y], "rho(demotion, %s)" % y))
    out["demotion"] = t1

    print("\n=== 2. BASE-PROBABILITY GAP vs THE B TERMS (the decisive one) ===")
    print("  If the base-text rise is a base-probability artefact it scales here.")
    print(hdr)
    t2 = {}
    for y in ("t_B|A", "t_B|B", "t_A|A", "D_aligned"):
        t2[y] = by_pair_rho(rows, "basegap_log2", y)
        print(fmt(t2[y], "rho(basegap, %s)" % y))
    out["basegap"] = t2

    print("\n=== 3. DOMAIN ===")
    print(hdr)
    t3 = {}
    doms = sorted({r["domain"] for r in rows if r.get("domain")})
    for dom in doms:
        t3[dom] = by_pair_median(rows, "D_aligned", lambda r, d=dom: r["domain"] == d)
        print(fmt(t3[dom], "D_aligned | %s" % dom))
    TRANS = {"sexual", "violence", "taboo"}
    for lab, f in (("transgressive", lambda r: r["domain"] in TRANS),
                   ("non-transgressive", lambda r: r["domain"] not in TRANS)):
        t3[lab] = by_pair_median(rows, "D_aligned", f)
        print(fmt(t3[lab], "D_aligned | %s" % lab))
    out["domain"] = t3

    print("\n=== 4. K NORMS (ranks only; register_level/vulgarity descriptive) ===")
    print(hdr)
    t4 = {}
    for sc in K_USABLE + K_DESCRIPTIVE:
        tag = "" if sc in K_USABLE else "  [descr]"
        for col, nm in (("k_faller_" + sc, "faller"), ("k_contrast_" + sc, "f-m")):
            key = "%s/%s" % (sc, nm)
            t4[key] = by_pair_rho(rows, col, "D_aligned")
            print(fmt(t4[key], "rho(%s %s, D_aligned)%s" % (sc, nm, tag)))
    out["k_norms"] = t4

    dst = "meta/M04_syntagmatic/results/a_dose_response.json"
    json.dump(out, open(dst, "w"), indent=1, default=float)
    print("\n  %d sites, %d pairs -> %s" % (len(rows), len(per), dst))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)


# ── the full-range version of test 1 ────────────────────────────────

def arm_frame(k=8):
    """Per (pair, prompt, ARM): the four terms, for ALL FOUR arms.

    **Test 1 run on the faller arm alone is uninformative and the reason is
    measurable, not arguable.** Every faller fell by construction: the arm's
    demotion spans an IQR of 1.01 log2 units with a p95 of -1.05, so no faller
    fell less than 2x. Correlating an outcome against a predictor truncated to
    its own tail attenuates toward zero whatever the truth is. The four arms
    together span an IQR of 2.40 log2 units and run from -3.70 (heavy demotion)
    through 0 (the matched non-movers) to +4.06 (heavy promotion) -- a real
    gradient, and the only frame in which "does the size of the fall predict the
    size of the disruption" is a question the data can answer.
    """
    am = A.arm_map()
    out = {}
    for pair in A.pairs_present():
        base_m, aln_m = pair.split(">", 1)
        sql = ("SELECT s.prompt AS prompt, s.forced_word AS w, s.role AS role, "
               "       g.scorer AS scorer, "
               "       avg(arrayAvg(arraySlice(g.logprobs, 1, %d))) AS lp "
               "FROM (SELECT corpus, model, prompt, forced_word, sample_idx, role "
               "      FROM %s.gen_sequences FINAL "
               "      WHERE corpus='%s' AND pair='%s') AS s "
               "INNER JOIN (SELECT corpus, model, prompt, forced_word, sample_idx, "
               "                   scorer, logprobs, scorable "
               "            FROM %s.gen_scores FINAL WHERE corpus='%s') AS g "
               "  ON s.corpus=g.corpus AND s.model=g.model AND s.prompt=g.prompt "
               "     AND s.forced_word=g.forced_word AND s.sample_idx=g.sample_idx "
               "WHERE g.scorable = 1 GROUP BY prompt, w, role, scorer"
               % (k, A.DB, A.CORPUS, pair.replace("'", "''"), A.DB, A.CORPUS))
        cell = {}
        for prompt, w, role, scorer, lp in A.rows(sql):
            arm = am.get((pair, prompt, w))
            if arm is None:
                continue
            sc = "aligned" if scorer == aln_m else ("base" if scorer == base_m else None)
            if sc is None:
                continue
            cell.setdefault((prompt, arm), {})[(role, sc)] = float(lp)
        for (prompt, arm), d in cell.items():
            need = [("aligned", "aligned"), ("base", "aligned"),
                    ("aligned", "base"), ("base", "base")]
            if not all(x in d for x in need):
                continue
            out[(pair, prompt, arm)] = {
                "A|A": d[("aligned", "aligned")], "B|A": d[("base", "aligned")],
                "A|B": d[("aligned", "base")],    "B|B": d[("base", "base")],
                "D_aligned": d[("aligned", "aligned")] - d[("base", "aligned")],
                "D_base": d[("aligned", "base")] - d[("base", "base")]}
    return out


def _arm_demotion():
    """(pair, prompt, arm) -> log2(q_aligned / p_base) for every arm."""
    cols = {"faller": ("faller_q", "faller_p", None),
            "faller-matched": ("matched_q", None, "matched_delta"),
            "riser": ("riser_q", None, "riser_delta"),
            "riser-matched": ("riser_matched_q", None, "riser_matched_delta")}
    out = {}
    for c in arms_table():
        for arm, (qc, pc, dc) in cols.items():
            q = c.get(qc)
            p = c.get(pc) if pc else (None if c.get(dc) is None or q is None
                                      else q - c[dc])
            if q and p and q > 0 and p > 0:
                out[(c["pair"], c["prompt"], arm)] = math.log2(q / p)
    return out


def full_range(k=8):
    terms, dem = arm_frame(k), _arm_demotion()
    rows = []
    for key, t in terms.items():
        d = dem.get(key)
        if d is None:
            continue
        r = dict(t)
        r["pair"], r["arm"], r["demotion_log2"] = key[0], key[2], d
        rows.append(r)
    print("\n=== 1b. DEMOTION MAGNITUDE, FULL RANGE (all four arms) ===")
    print("  %d arm-sites over %d pairs. rho is Spearman WITHIN pair across all"
          % (len(rows), len({r["pair"] for r in rows})))
    print("  four arms, then a sign test on the pair-level rhos.")
    print("  %-30s %9s %10s %4s %9s" % ("", "median", "neg/pos", "prs", "p(sign)"))
    out = {}
    for y in ("D_aligned", "D_base", "B|A", "A|A"):
        out[y] = by_pair_rho(rows, "demotion_log2", y, min_sites=20)
        print(fmt(out[y], "rho(demotion, %s)" % y))
    print("\n  per-arm D_aligned (level, not slope):")
    for arm in ("faller", "faller-matched", "riser-matched", "riser"):
        sub = [r for r in rows if r["arm"] == arm]
        bp = {}
        for r in sub:
            bp.setdefault(r["pair"], []).append(r["D_aligned"])
        v = [statistics.median(x) for x in bp.values() if len(x) >= 3]
        if v:
            print("    %-16s n=%2d pairs  median D %+8.4f  (median demotion %+.2f)"
                  % (arm, len(v), statistics.median(v),
                     statistics.median([r["demotion_log2"] for r in sub])))
    dst = "meta/M04_syntagmatic/results/a_dose_response_fullrange.json"
    json.dump({"k": k, "n_arm_sites": len(rows), "rho": out,
               "_fingerprint": A.fingerprint({"k": k, "arms": ARMS})},
              open(dst, "w"), indent=1, default=float)
    print("\n  -> %s" % dst)
    return rows
