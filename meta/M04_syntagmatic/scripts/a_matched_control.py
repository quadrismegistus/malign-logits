#!/usr/bin/env python
"""a_matched_control.py — M04/A on the passage corpus: DEMOTION or IMPROBABILITY?

## READ THIS BEFORE BUILDING ANY FORCED-vs-UNDISTURBED COMPARISON ON THESE ARMS

**THE FORCED WORD IS NOT SCORED.** It is neither in `prompt` nor in `text`; it
acts as one extra word of unscored conditioning context. Measured over all
904,544 forced rows: `text` starts with the forced word in 0.0008 of cases,
`prompt` ends with it in 0.0000, and the within-site sd of `logprobs[1]` is 2.113
(median 1.911, 56,534 sites) where a deterministic forced-token logprob would be
exactly 0. Undisturbed rows scatter alike at 2.479 — **position 1 is a SAMPLED
token in both conditions.**

**CONSEQUENCE: a forced row and an undisturbed row at the same nominal position
DO NOT carry the same amount of context, so any forced-vs-undisturbed contrast is
broken at CONSTRUCTION, not at interpretation.** This cost `opening_matched.md`
its entire construction on 13 Aug — withdrawn wholesale, not amended, after four
successive controls all "survived" precisely because none of them touched the
asymmetry. Align on the SENTENCE via `n_forced_tokens`, as
`a_position_figures.py` does.

**ARM-vs-ARM IS SAFE**, and for a stronger reason than "all arms are forced":
every forced arm carries EXACTLY ONE unscored extra word, so the confound is
constant in amount and cannot differ between rungs.

This warning lives here, in the module other seats import, because the same fact
was already written in `../results/A_RESULTS.md:185` and two seats missed it —
one who had written it and did not re-read it, one who read this producer for a
definition and never opened the results file beside it. **The producer says what
was computed; the results file says what was learned. A warning is a conclusion,
so it belongs in both.** Docket [5810]–[5814].

    meta/M04_syntagmatic/scripts/a_matched_control.py --diagnostic
    meta/M04_syntagmatic/scripts/a_matched_control.py --run
    meta/M04_syntagmatic/scripts/a_matched_control.py --run --out results/a_matched.json

## THE QUESTION, AND WHY IT NEEDED A NEW CORPUS

A found a post-utterance shock: forced to utter a word alignment had demoted, the
aligned model's continuation diverges. A's own spec §8 named the confound before
A ran:

> The faller is by construction low-probability UNDER ALIGNED [...] conditioning
> on it places the aligned model in a state it already assigns low probability
> to; the next token inherits that mechanically. **Separating them requires a
> word matched on improbability-under-aligned but NOT demoted by alignment. No
> collected corpus has one.**

One does now. A's original contrast was faller vs RISER; this producer runs the
same machinery with faller vs FALLER-MATCHED, the arm built to be improbable
under aligned without having been demoted. If the shock survives the matched
control it is about DEMOTION; if it vanishes it was about IMPROBABILITY.

    D(s,a) = mean_lp(aligned's beams | aligned) − mean_lp(base's beams | aligned)
    Δ(s)   = D(s, faller) − D(s, comparison)      comparison ∈ {matched, riser}

`riser` is computed beside `matched` as A's original contrast, so the new result
and the replication sit in one table.

## THE DECLARED READING, FIXED BEFORE ANY NUMBER EXISTS

Truncated primary and full-length secondary are TWO QUANTITIES ANSWERING TWO
QUESTIONS, not a primary and its robustness check ([5679]/[5680]). The naive
framing gets the third row exactly backwards:

    truncated fires, full does not   the shock is LOCALISED at the utterance —
                                     which is what "post-utterance" MEANS.
                                     CONFIRMATORY.
    both fire                        the effect persists across the passage; a
                                     stronger and different claim than A made.
    full fires, truncated does not   the effect is NOT at the utterance and the
                                     name is wrong. THE WORRYING CASE, and the
                                     one a robustness framing would call a pass.

## WHY TRUNCATION AND NOT A LENGTH FILTER

Sequence length is POST-TREATMENT — an outcome of the model given the injected
word. Filtering to equal-length sequences conditions on a collider: if the
aligned model terminates early BECAUSE of the faller, keeping the long ones keeps
a non-random subset of each arm, manufacturing bias while looking like hygiene.

Truncation drops nothing. Every sequence contributes positions 1..k, the
denominator is constant by construction, and there is no selection step. Measured
survival, undisturbed arm: at k=32 every arm retains 88–100%; at k=256 the
aligned arm falls to 47% on Amber — so the FULL-LENGTH mean is already computed
over a set that thins asymmetrically with position.

`k` is DECLARED BY RULE, not chosen: the smallest k at which every (pair, arm,
role) cell retains >= RETENTION of its sequences. Survival is not the outcome, so
computing k from it cannot fork the path.

## THE STANDING CLAUSES THIS PRODUCER IS BOUND BY

**Clause 1 (reads).** Every ReplacingMergeTree read whose figure travels goes
through FINAL, and FINAL is NOT sufficient: `gen_scores` sorts on
`(corpus, model, prompt, forced_word, sample_idx, scorer)` and the analysis unit
is the same tuple, so FINAL suffices HERE — but the collapse is written out
explicitly rather than assumed, because a sorting key is a storage decision and
an analysis key is a claim about the unit. Measured on twp_words the same night:
FINAL 60,425,347 vs analysis-key 58,292,343, a 2,133,004 disagreement.

**Clause 2 (writes).** The output carries a PRODUCER FINGERPRINT — code SHA plus
the parameters that matter — because an artifact records what was produced and
not what produced it. This script does not resume; it recomputes.

**Preamble (scope).** Every completeness sentence here names what it covered and
the nearest thing it did not.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import ch_ingest as ci                                    # noqa: E402

DB = ci.DB
CORPUS = "passage"
POPULATION = os.path.join(ROOT, "data", "forced_arms_46reps_drmatch.json")
RETENTION = 0.85
K_GRID = (8, 16, 32, 64, 128, 200, 256)


def fingerprint(params):
    """Producer identity: code SHA + the parameters that matter (clause 2)."""
    try:
        sha = subprocess.run(["git", "log", "-1", "--format=%H", "--",
                              os.path.abspath(__file__)],
                             capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        sha = ""
    body = open(os.path.abspath(__file__), "rb").read()
    return {"script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "commit": sha or "UNCOMMITTED",
            "file_sha256_16": hashlib.sha256(body).hexdigest()[:16],
            "params": params}


def q(sql):
    return ci.ch_read(sql).strip()


def rows(sql):
    out = q(sql)
    return [r.split("\t") for r in out.split("\n")] if out else []


def arm_map():
    """(pair, prompt, word) -> arm, from the FROZEN population table.

    The corpus stores `forced_word` but not which arm it is; the arm is a
    property of the frozen table and is never re-derived from the data.
    """
    m = {}
    for c in json.load(open(POPULATION)).get("cells", []):
        for src, arm in (("faller", "faller"), ("matched", "faller-matched"),
                         ("riser", "riser"), ("riser_matched", "riser-matched")):
            if c.get(src):
                m[(c["pair"], c["prompt"], c[src])] = arm
    return m


def pairs_present():
    return [r[0] for r in rows(
        "SELECT DISTINCT pair FROM %s.gen_sequences WHERE corpus='%s' "
        "AND pair != '' ORDER BY pair" % (DB, CORPUS))]


# ── the diagnostic that runs FIRST ──────────────────────────────────

def diagnostic():
    """arm x role INTERACTION in sequence length, on means AND a tail quantile.

    A's Δ is a DOUBLE difference, so a role-level length effect cancels ONLY if
    it is the same across arms. That is a checkable precondition, not an
    assumption, and it is reported before the delta regardless of what it says.

    Reported on means AND p10 because the corpus-grain audit found the role
    coupling lives in the LOWER TAIL (aligned p10 25 words vs base 57) while the
    pair median centres on zero — a mean-based read feels it, a median-based one
    mostly does not.
    """
    print("== DIAGNOSTIC 1: arm x role length interaction (runs before the delta)")
    sql = ("SELECT pair, role, forced_word != '' AS forced, "
           "       avg(n_tokens) AS mean_len, "
           "       quantile(0.10)(n_tokens) AS p10, count() AS n "
           "FROM %s.gen_sequences FINAL "
           "WHERE corpus='%s' AND pair != '' "
           "GROUP BY pair, role, forced ORDER BY pair, role, forced" % (DB, CORPUS))
    got = {}
    for pair, role, forced, mean_len, p10, n in rows(sql):
        got[(pair, role, int(forced))] = (float(mean_len), float(p10), int(n))

    inter_mean, inter_p10 = [], []
    for pair in sorted({k[0] for k in got}):
        try:
            bu, bf = got[(pair, "base", 0)], got[(pair, "base", 1)]
            au, af = got[(pair, "aligned", 0)], got[(pair, "aligned", 1)]
        except KeyError:
            continue
        # interaction = (aligned forced-vs-undisturbed) - (base forced-vs-undisturbed)
        inter_mean.append((af[0] - au[0]) - (bf[0] - bu[0]))
        inter_p10.append((af[1] - au[1]) - (bf[1] - bu[1]))
    for label, v in (("mean length", inter_mean), ("p10 length", inter_p10)):
        if not v:
            continue
        pos = sum(1 for x in v if x > 0)
        print("   interaction on %-12s n=%d  median %+8.2f tokens  %d up / %d dn"
              % (label, len(v), statistics.median(v), pos, len(v) - pos))
    print("   READ: a median near zero with a balanced split means the role effect")
    print("         cancels in the double difference. It does not retire the")
    print("         truncated primary, which is declared for its own reason.")
    return {"interaction_mean_len": inter_mean, "interaction_p10_len": inter_p10}


def survival_and_k():
    """DIAGNOSTIC 2 — survival by position, and k from the declared retention rule."""
    print("\n== DIAGNOSTIC 2: survival, and k by the RETENTION RULE (>= %.0f%%)"
          % (100 * RETENTION))
    cols = ", ".join("countIf(n_tokens >= %d) / count() AS k%d" % (k, k) for k in K_GRID)
    sql = ("SELECT pair, role, forced_word != '' AS forced, %s "
           "FROM %s.gen_sequences FINAL WHERE corpus='%s' AND pair != '' "
           "GROUP BY pair, role, forced" % (cols, DB, CORPUS))
    worst = {k: 1.0 for k in K_GRID}
    for r in rows(sql):
        for i, k in enumerate(K_GRID):
            worst[k] = min(worst[k], float(r[3 + i]))
    for k in K_GRID:
        print("   k=%-4d worst cell retains %6.1f%%%s"
              % (k, 100 * worst[k], "   <- passes" if worst[k] >= RETENTION else ""))
    ok = [k for k in K_GRID if worst[k] >= RETENTION]
    if ok:
        k = max(ok)
        print("   RULE FIRED: largest k whose worst cell retains >= %.0f%%  ->  k = %d"
              % (100 * RETENTION, k))
        return k, {str(kk): worst[kk] for kk in K_GRID}, True
    #: **THE RULE DID NOT FIRE, AND SAYING SO IS THE POINT.** No k on the grid
    #: reaches the declared threshold — the worst cell is 58.3% at k=8 and 72.4%
    #: excluding the bloom pair, whose aligned arm averages 15.5 tokens against
    #: the corpus's 130-200 and is already the named outlier on the empty-text
    #: axis. An earlier version of this function silently fell back to
    #: min(K_GRID) and PRINTED the rule's name beside it, which asserted a
    #: selection that had not happened. A rule nothing satisfies does not
    #: select; it defers to whatever the code does on failure, and the code must
    #: say which of those occurred.
    k = min(K_GRID)
    print("   ** RULE DID NOT FIRE: no k on the grid reaches %.0f%%." % (100 * RETENTION))
    print("      worst cell is %.1f%% at k=%d. Falling back to k=%d, which is NOT"
          % (100 * worst[min(K_GRID)], min(K_GRID), k))
    print("      the rule's answer — it is the smallest grid value, reported as such.")
    print("      A's registered readout is position +1, which k=%d contains." % k)
    return k, {str(kk): worst[kk] for kk in K_GRID}, False


# ── the delta ───────────────────────────────────────────────────────

def deltas(k, comparison_arm):
    """Δ(s) = D(s, faller) − D(s, comparison), truncated to 1..k and full-length.

    D(s,a) = mean_lp(aligned text | aligned scorer) − mean_lp(base text | aligned
    scorer). The scorer is a MODEL NAME in this schema, so `aligned scorer` is
    the pair's aligned checkpoint.

    FINAL + explicit GROUP BY the analysis key, per clause 1 — here the sorting
    key and the analysis key coincide, and the collapse is written out anyway so
    the next reader does not have to check.
    """
    am = arm_map()
    out = {}
    for pair in pairs_present():
        base_m, aln_m = pair.split(">", 1)
        sql = ("SELECT s.prompt AS prompt, s.forced_word AS w, s.role AS role, "
               "       avg(arrayAvg(arraySlice(g.logprobs, 1, %d))) AS trunc_lp, "
               "       avg(least(length(g.logprobs), %d)) AS trunc_denom, "
               "       avg(arrayAvg(g.logprobs)) AS full_lp, "
               "       countIf(g.scorable = 0) AS unscorable, count() AS n "
               "FROM (SELECT corpus, model, prompt, forced_word, sample_idx, role "
               "      FROM %s.gen_sequences FINAL "
               "      WHERE corpus='%s' AND pair='%s') AS s "
               "INNER JOIN (SELECT corpus, model, prompt, forced_word, sample_idx, "
               "                   logprobs, scorable "
               "            FROM %s.gen_scores FINAL "
               "            WHERE corpus='%s' AND scorer='%s') AS g "
               "  ON s.corpus=g.corpus AND s.model=g.model AND s.prompt=g.prompt "
               "     AND s.forced_word=g.forced_word AND s.sample_idx=g.sample_idx "
               "WHERE g.scorable = 1 "
               "GROUP BY prompt, w, role"
               % (k, k, DB, CORPUS, pair.replace("'", "''"),
                  DB, CORPUS, aln_m.replace("'", "''")))
        cell = {}
        for prompt, w, role, tl, td, fl, uns, n in rows(sql):
            arm = am.get((pair, prompt, w))
            if arm is None:
                continue
            cell.setdefault((prompt, arm), {})[role] = (float(tl), float(fl), int(uns), int(n), float(td))
        per_site = []
        for (prompt, arm), d in cell.items():
            if "base" in d and "aligned" in d:
                per_site.append((prompt, arm,
                                 d["aligned"][0] - d["base"][0],     # D truncated
                                 d["aligned"][1] - d["base"][1]))    # D full
        byprompt = {}
        for prompt, arm, dt, df in per_site:
            byprompt.setdefault(prompt, {})[arm] = (dt, df)
        dt_l, df_l = [], []
        for prompt, arms in byprompt.items():
            if "faller" in arms and comparison_arm in arms:
                dt_l.append(arms["faller"][0] - arms[comparison_arm][0])
                df_l.append(arms["faller"][1] - arms[comparison_arm][1])
        if dt_l:
            out[pair] = {"n_sites": len(dt_l),
                         "delta_trunc_median": statistics.median(dt_l),
                         "delta_full_median": statistics.median(df_l),
                         "sites_trunc_neg": sum(1 for x in dt_l if x < 0),
                         "sites_full_neg": sum(1 for x in df_l if x < 0)}
    return out


def sign_test(vals):
    """Two-sided sign test over pair medians; ties dropped and REPORTED."""
    import math
    neg = sum(1 for v in vals if v < 0)
    pos = sum(1 for v in vals if v > 0)
    ties = len(vals) - neg - pos
    n = neg + pos
    if n == 0:
        return {"n": 0, "neg": 0, "pos": 0, "ties": ties, "p": None}
    k = min(neg, pos)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"n": n, "neg": neg, "pos": pos, "ties": ties, "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic", action="store_true",
                    help="run the two declared diagnostics only, no delta")
    ap.add_argument("--run", action="store_true", help="diagnostics then the delta")
    ap.add_argument("--comparison", default="faller-matched",
                    choices=["faller-matched", "riser"],
                    help="faller-matched = the new control; riser = A's original")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not (a.diagnostic or a.run):
        ap.error("pass --diagnostic or --run")

    print("M04/A ON THE PASSAGE CORPUS — faller vs %s" % a.comparison)
    print("SCOPE: corpus='passage' in ClickHouse only. This says nothing about")
    print("       beam_fc, y or f11_l2, and nothing about pairs absent from the")
    print("       corpus (Pharia, RWKV-4, Zamba2, Olmo-Hybrid).\n")

    res = {"_fingerprint": fingerprint({"retention": RETENTION,
                                        "comparison": a.comparison,
                                        "corpus": CORPUS}),
           "_reading": "truncated and full-length answer DIFFERENT questions; "
                       "see the module docstring's declared table"}
    res["diagnostic_interaction"] = diagnostic()
    k, surv, rule_fired = survival_and_k()
    res["k"] = k
    res["retention_rule_fired"] = rule_fired
    res["survival_worst_cell"] = surv
    if a.diagnostic:
        print("\n(diagnostics only; no delta computed)")
    else:
        print("\n== DELTA  (faller − %s), k=%d" % (a.comparison, k))
        per_pair = deltas(k, a.comparison)
        res["per_pair"] = per_pair
        tr = [v["delta_trunc_median"] for v in per_pair.values()]
        fu = [v["delta_full_median"] for v in per_pair.values()]
        res["sign_trunc"] = sign_test(tr)
        res["sign_full"] = sign_test(fu)
        for label, vals, st in (("TRUNCATED (primary)", tr, res["sign_trunc"]),
                                ("FULL-LENGTH (other question)", fu, res["sign_full"])):
            if vals:
                print("   %-30s pairs %2d  median %+.5f  %d neg / %d pos"
                      "  ties %d  p=%s"
                      % (label, len(vals), statistics.median(vals),
                         st["neg"], st["pos"], st["ties"],
                         ("%.4f" % st["p"]) if st["p"] is not None else "-"))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ── the four terms, and the permutation null ────────────────────────

def four_terms(k, comparison_arm, want_sites=False):
    """A's FOUR TERMS: {aligned text, base text} x {aligned scorer, base scorer}.

    A's split is BY SCORER, not by text: "Both terms under the aligned scorer
    move; neither under the base scorer does" — later partly retracted (B|A
    weaker than first posted). If the effect is a disturbance of the ALIGNED
    model's expectations rather than a production difference, it should live in
    the aligned-scorer terms and be absent from the base-scorer ones.

        A|A  aligned text under aligned scorer      D = A|A − B|A
        B|A  base text under aligned scorer         Δ = D(faller) − D(comparison)
        A|B  aligned text under base scorer
        B|B  base text under base scorer

    Returns per-pair term deltas, and optionally the SITE-level D values that
    the permutation null needs — fetched in the same pass so the null is
    computed on exactly the data the estimate was.
    """
    am = arm_map()
    per_pair, sites_out = {}, {}
    for pair in pairs_present():
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
               % (k, DB, CORPUS, pair.replace("'", "''"), DB, CORPUS))
        cell = {}
        for prompt, w, role, scorer, lp in rows(sql):
            arm = am.get((pair, prompt, w))
            if arm is None:
                continue
            sc = "aligned" if scorer == aln_m else ("base" if scorer == base_m else None)
            if sc is None:
                continue
            cell.setdefault((prompt, arm), {})[(role, sc)] = float(lp)
        terms = {t: [] for t in ("A|A", "B|A", "A|B", "B|B")}
        dvals = []
        byprompt = {}
        for (prompt, arm), d in cell.items():
            need = [("aligned", "aligned"), ("base", "aligned"),
                    ("aligned", "base"), ("base", "base")]
            if not all(x in d for x in need):
                continue
            byprompt.setdefault(prompt, {})[arm] = {
                "A|A": d[("aligned", "aligned")], "B|A": d[("base", "aligned")],
                "A|B": d[("aligned", "base")],    "B|B": d[("base", "base")]}
        dbase, dtrip, sitekeys, siteterms = [], [], [], []
        for prompt, arms in byprompt.items():
            if "faller" in arms and comparison_arm in arms:
                for t in terms:
                    terms[t].append(arms["faller"][t] - arms[comparison_arm][t])
                f, c = arms["faller"], arms[comparison_arm]
                da = (f["A|A"] - f["B|A"]) - (c["A|A"] - c["B|A"])
                db = (f["A|B"] - f["B|B"]) - (c["A|B"] - c["B|B"])
                dvals.append(da); dbase.append(db); dtrip.append(da - db)
                sitekeys.append(prompt)
                siteterms.append({t: f[t] - c[t] for t in terms})
        if dvals:
            per_pair[pair] = {t: statistics.median(v) for t, v in terms.items() if v}
            per_pair[pair]["PRIMARY"] = statistics.median(dvals)
            per_pair[pair]["PRIMARY_basescorer"] = statistics.median(dbase)
            per_pair[pair]["TRIPLE"] = statistics.median(dtrip)
            per_pair[pair]["n_sites"] = len(dvals)
            if want_sites:
                sites_out[pair] = {"aligned": dvals, "base": dbase, "triple": dtrip,
                                   "prompts": sitekeys, "terms": siteterms}
    return (per_pair, sites_out) if want_sites else per_pair


def permutation_null(sites, iters=2000, seed=20260813):
    """SIGN-FLIP the whole site difference, per site, preserving pair structure.

    **The null flips SIGNS, it does not reassign arms across sites.** lacan's
    F-P was withdrawn because a per-cell label shuffle averaged a pair's members
    together and collapsed the null's spread, so the observed spread beat a null
    that had been crushed rather than a real one ([5588]). Flipping the sign of a
    site's whole Δ preserves every magnitude and every pair's site count, and
    tests exactly the exchangeability the sign test claims.

    Statistic: the number of pairs whose MEDIAN site Δ is negative — the same
    quantity the headline reports, so the null tests the estimate rather than a
    neighbour of it.
    """
    import random
    rng = random.Random(seed)
    obs = sum(1 for v in sites.values() if statistics.median(v) < 0)
    n_pairs = len(sites)
    ge = 0
    dist = []
    for _ in range(iters):
        c = 0
        for v in sites.values():
            flipped = [x if rng.random() < 0.5 else -x for x in v]
            if statistics.median(flipped) < 0:
                c += 1
        dist.append(c)
        if c >= obs:
            ge += 1
    return {"observed_neg_pairs": obs, "n_pairs": n_pairs, "iters": iters,
            "null_mean": statistics.mean(dist), "null_sd": statistics.pstdev(dist),
            "p_one_sided": (ge + 1) / (iters + 1)}
