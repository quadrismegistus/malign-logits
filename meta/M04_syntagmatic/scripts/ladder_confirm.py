"""M04 ladder — the declared run of `plans/plan_ladder.md`.

**THIS IS USE (1) OF THE PLAN'S §7 AND NOTHING MORE: a disciplined re-analysis
of the passage corpus.** The ladder was found in this data, so a declared re-run
on it is NOT independent confirmation and no output of this script may be
described as confirmed. What it buys is the end of analytic drift: the tests,
the directions and the decision rules were written down first, and this file
executes them without further choices.

**THE DECISION RULES ARE EXECUTED, NOT NARRATED.** Each hypothesis prints
SUPPORTED / NOT SUPPORTED computed from its own stated criterion. A prose map
cannot fail a test; a rule that is evaluated can.

**EVERY NUMBER HERE DIFFERS FROM THE EXPLORATORY PASS AND THAT IS EXPECTED.**
The per-pair retention gate (85% of aligned forced sequences must reach the
window) takes n from 42 to 40, dropping `bloomz-7b1` (58.3%) and
`recurrentgemma-9b-it` (82.8%). A re-run that reproduced the old numbers exactly
would mean the gate had not been applied.

Plan: `../plans/plan_ladder.md`. Exploratory record: `../results/A_RESULTS.md`.
"""

import json, math, os, statistics, sys, random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import a_matched_control as A
import a_dose_response as R
from wordfreq import zipf_frequency

K = 8                     # analysis window, plan §2
GATE = 0.85               # per-pair retention gate, plan §2
SEED = 20260813
ITERS = 2000
OUT = "meta/M04_syntagmatic/results/ladder_confirm.json"

# cumulative windows, disjoint bands, and single offsets — one query pass
CUM = [(1, 4), (1, 8), (1, 16), (1, 32)]
BAND = [(33, 32), (65, 64), (129, 128)]
OFFS = [1, 2, 16, 32]


def eligible_pairs():
    """Plan §2: a pair enters at window K if >=85% of its ALIGNED forced
    sequences reach K tokens. Computed from `n_tokens` before any contrast is
    run, blind to D and to the ladder."""
    rows = A.rows(
        "SELECT pair, countIf(n_tokens >= %d) / count() AS r "
        "FROM %s.gen_sequences FINAL WHERE corpus='%s' AND forced_word != '' "
        "AND role='aligned' GROUP BY pair" % (K, A.DB, A.CORPUS))
    keep, drop = [], []
    for pair, r in rows:
        (keep if float(r) >= GATE else drop).append((pair, float(r)))
    return sorted(p for p, _ in keep), sorted(drop, key=lambda x: x[1])


def collect(pairs):
    """Per (pair, prompt, arm, term): every window this plan needs, one pass."""
    am = A.arm_map()
    sel = ", ".join(
        ["avg(arrayAvg(arraySlice(g.logprobs,%d,%d))) AS c%d" % (s, n, i)
         for i, (s, n) in enumerate(CUM)]
        + ["avg(arrayAvg(g.logprobs)) AS cfull"]
        + ["avg(if(length(g.logprobs) >= %d, "
           "arrayAvg(arraySlice(g.logprobs,%d,%d)), null)) AS b%d" % (s, s, n, i)
           for i, (s, n) in enumerate(BAND)]
        + ["avgIf(g.logprobs[%d], length(g.logprobs) >= %d) AS f%d" % (k, k, k)
           for k in OFFS])
    meta = {}
    for c in R.arms_table():
        for arm, col, qk, dk in (("faller", "faller", "faller_q", None),
                                 ("faller-matched", "matched", "matched_q", "matched_delta"),
                                 ("riser-matched", "riser_matched", "riser_matched_q", "riser_matched_delta"),
                                 ("riser", "riser", "riser_q", "riser_delta")):
            q, w = c.get(qk), c.get(col)
            p = c.get("faller_p") if arm == "faller" else (
                None if c.get(dk) is None or q is None else q - c[dk])
            if q and q > 0 and p and p > 0 and w:
                meta[(c["pair"], c["prompt"], arm)] = {
                    "logq": math.log2(q), "logp": math.log2(p),
                    "demotion": math.log2(q / p), "zipf": zipf_frequency(w, "en"),
                    "domain": c.get("domain")}
    out = {}
    for n, pair in enumerate(pairs, 1):
        base_m, aln_m = pair.split(">", 1)
        sql = ("SELECT s.prompt, s.forced_word, s.role, g.scorer, " + sel +
               " FROM (SELECT corpus,model,prompt,forced_word,sample_idx,role "
               "       FROM %s.gen_sequences FINAL WHERE corpus='%s' AND pair='%s') AS s "
               "INNER JOIN (SELECT corpus,model,prompt,forced_word,sample_idx,scorer,"
               "                   logprobs,scorable FROM %s.gen_scores FINAL "
               "            WHERE corpus='%s') AS g "
               " ON s.corpus=g.corpus AND s.model=g.model AND s.prompt=g.prompt "
               "    AND s.forced_word=g.forced_word AND s.sample_idx=g.sample_idx "
               "WHERE g.scorable=1 GROUP BY s.prompt,s.forced_word,s.role,g.scorer"
               % (A.DB, A.CORPUS, pair.replace("'", "''"), A.DB, A.CORPUS))
        for r in A.rows(sql):
            prompt, w, role, scorer = r[0], r[1], r[2], r[3]
            arm = am.get((pair, prompt, w))
            m = meta.get((pair, prompt, arm)) if arm else None
            #: DO NOT DROP THE SITE. `meta` needs p = q - delta > 0, which only the
            #: MOVEMENT columns (logq/logp/demotion) use -- D never touches p. The
            #: original `continue` conditioned every column on a predicate one
            #: derived column needed, and it did so ARM-ASYMMETRICALLY: the sites
            #: it removes are exactly those with p == 0 (|delta|/q = 1.000), i.e.
            #: words that rose from nothing, which are 3.4x more common in
            #: riser-matched than in faller. Filtering an arm CONTRAST unequally
            #: by arm is not a wash. Consumers of the movement fields now guard
            #: for their absence individually. Docket [5828].
            sc = "A" if scorer == aln_m else ("B" if scorer == base_m else None)
            if sc is None:
                continue
            term = {"aligned": "A", "base": "B"}[role] + "|" + sc
            d = dict(m) if m is not None else {}
            names = (["k%d" % n_ for _, n_ in CUM] + ["kfull"]
                     + ["band%d" % s for s, _ in BAND] + ["off%d" % k for k in OFFS])
            for j, nm in enumerate(names):
                try:
                    v = float(r[4 + j])
                    if v == v and v != 0.0:
                        d[nm] = v
                except (ValueError, TypeError):
                    pass
            out.setdefault((pair, prompt), {})[(arm, term)] = d
        print("  collected %2d/%d  %s" % (n, len(pairs), pair.split(">")[1][:44]))
    return out


def contrast(cells, focal, ctrl, term, col, use_D=False):
    """Pair medians of the per-site difference, then a sign test + permutation."""
    bypair = {}
    for (pair, prompt), cell in cells.items():
        def val(arm):
            if use_D:
                a, b = cell.get((arm, "A|A")), cell.get((arm, "B|A"))
                if a and b and col in a and col in b:
                    return a[col] - b[col]
                return None
            c = cell.get((arm, term))
            return c[col] if c and col in c else None
        f, c = val(focal), val(ctrl)
        if f is not None and c is not None:
            bypair.setdefault(pair, []).append(f - c)
    sites = {p: v for p, v in bypair.items() if len(v) >= 3}
    med = [statistics.median(v) for v in sites.values()]
    if len(med) < 5:
        return None
    st = A.sign_test(med)
    rng = random.Random(SEED)
    obs = sum(1 for m in med if m < 0)
    ge = sum(1 for _ in range(ITERS)
             if sum(1 for v in sites.values()
                    if statistics.median([x if rng.random() < .5 else -x for x in v]) < 0) >= obs)
    return {"median": statistics.median(med), "neg": st["neg"], "pos": st["pos"],
            "n_pairs": len(med), "p": st["p"], "p_perm_neg": (ge + 1) / (ITERS + 1),
            "p_perm_pos": 1 - ge / (ITERS + 1)}


def show(lab, d, w=30):
    if d is None:
        return "  %-*s (insufficient)" % (w, lab)
    return "  %-*s %+9.5f %4d/%-4d %4d  p=%-8s perm=%.4f" % (
        w, lab, d["median"], d["neg"], d["pos"], d["n_pairs"],
        ("%.4f" % d["p"]) if d["p"] is not None else "-",
        min(d["p_perm_neg"], d["p_perm_pos"]))


def main():
    keep, drop = eligible_pairs()
    present = set(A.pairs_present())
    keep = [p for p in keep if p in present]
    print("=" * 78)
    print("  M04 LADDER — DECLARED RUN of plans/plan_ladder.md")
    print("  DISCIPLINED RE-ANALYSIS, NOT CONFIRMATION (plan §7): the ladder was")
    print("  found in this data and no output below may be called confirmed.")
    print("=" * 78)
    print("\n  RETENTION GATE (plan §2): >=%d%% of aligned forced sequences reach k=%d"
          % (GATE * 100, K))
    for pair, r in drop:
        print("    EXCLUDED  %-46s %.1f%%" % (pair.split(">")[1][:46], r * 100))
    print("    n = %d eligible pairs (was 42 before the gate)" % len(keep))

    cells = collect(keep)
    res = {"n_pairs": len(keep), "excluded": [{"pair": p, "retention": r} for p, r in drop],
           "k": K, "gate": GATE, "_fingerprint": A.fingerprint({"k": K, "gate": GATE})}
    verdicts = {}

    print("\n" + "-" * 78)
    print("  H1 — MONOTONE IN DIRECTION (primary)")
    print("  rule: fell-flat < 0 AND rose-flat > 0, both sign p<0.05;")
    print("        AND fell-rose < 0 at p<0.01 with permutation p<0.05")
    print("-" * 78)
    h1 = {"fell_flat": contrast(cells, "faller", "faller-matched", None, "k8", use_D=True),
          "rose_flat": contrast(cells, "riser-matched", "faller-matched", None, "k8", use_D=True),
          "fell_rose": contrast(cells, "faller", "riser-matched", None, "k8", use_D=True)}
    print(show("D  fell - flat", h1["fell_flat"]))
    print(show("D  rose - flat", h1["rose_flat"]))
    print(show("D  fell - rose", h1["fell_rose"]))
    a, b, c = h1["fell_flat"], h1["rose_flat"], h1["fell_rose"]
    verdicts["H1"] = bool(a and b and c and a["median"] < 0 and a["p"] < 0.05
                          and b["median"] > 0 and b["p"] < 0.05
                          and c["median"] < 0 and c["p"] < 0.01 and c["p_perm_neg"] < 0.05)
    res["H1"] = h1

    print("\n" + "-" * 78)
    print("  H2 — THE SPLIT IS BY TEXT, NOT BY SCORER")
    print("  rule: A|A and A|B agree in sign on faller-riser, both p<0.05,")
    print("        and B|A, B|B do not both clear")
    print("-" * 78)
    h2 = {t: contrast(cells, "faller", "riser", t, "k8") for t in ("A|A", "A|B", "B|A", "B|B")}
    for t in ("A|A", "A|B", "B|A", "B|B"):
        print(show("%s   faller - riser" % t, h2[t]))
    aa, ab, ba, bb = (h2[t] for t in ("A|A", "A|B", "B|A", "B|B"))
    verdicts["H2"] = bool(aa and ab and ba and bb
                          and (aa["median"] < 0) == (ab["median"] < 0)
                          and aa["p"] < 0.05 and ab["p"] < 0.05
                          and not (ba["p"] < 0.05 and bb["p"] < 0.05))
    res["H2"] = h2

    print("\n" + "-" * 78)
    print("  H3 — THE LADDER IS THE SMALL PART")
    print("  rule: rho(log q, A|A) exceeds rho(demotion, A|A), paired over pairs")
    print("-" * 78)
    rowsAA = []
    for (pair, prompt), cell in cells.items():
        for arm in ("faller", "faller-matched", "riser-matched", "riser"):
            c = cell.get((arm, "A|A"))
            if c and "k8" in c and "logq" in c and "demotion" in c:
                rowsAA.append({"pair": pair, "y": c["k8"], "logq": c["logq"],
                               "demotion": c["demotion"], "zipf": c["zipf"],
                               "domain": c.get("domain")})
    dq, dd, diff = [], [], []
    bypair = {}
    for r in rowsAA:
        bypair.setdefault(r["pair"], []).append(r)
    for pair, v in bypair.items():
        if len(v) < 25:
            continue
        rq = R.spearman([x["logq"] for x in v], [x["y"] for x in v])
        rd = R.spearman([x["demotion"] for x in v], [x["y"] for x in v])
        if rq is None or rd is None:
            continue
        dq.append(rq); dd.append(rd); diff.append(rq - rd)
    st = A.sign_test(diff)
    print("  rho(log q,  A|A)   median %+.4f" % statistics.median(dq))
    print("  rho(demotion, A|A) median %+.4f" % statistics.median(dd))
    print("  paired difference  median %+.4f  %d-/%d+  n=%d  p=%s"
          % (statistics.median(diff), st["neg"], st["pos"], len(diff),
             ("%.4f" % st["p"]) if st["p"] is not None else "-"))
    verdicts["H3"] = bool(statistics.median(diff) > 0 and st["p"] is not None and st["p"] < 0.05)
    res["H3"] = {"rho_logq": statistics.median(dq), "rho_demotion": statistics.median(dd),
                 "paired_diff": statistics.median(diff), "neg": st["neg"],
                 "pos": st["pos"], "n_pairs": len(diff), "p": st["p"]}

    print("\n" + "-" * 78)
    print("  DECLARED NEGATIVES (plan §4) — each must hold")
    print("-" * 78)
    # N1: frequency has the OPPOSITE sign early and is null from +16
    n1 = {}
    bypair_cells = {}
    for (pair, prompt), cell in cells.items():
        for (arm, t), c in cell.items():
            if t == "A|A":
                bypair_cells.setdefault(pair, []).append(c)
    for k in OFFS:
        col = "off%d" % k
        sl = []
        for pair, cs in bypair_cells.items():
            vv = [(c["zipf"], -c[col]) for c in cs if col in c and "zipf" in c]  # surprisal = -logprob
            if len(vv) < 25:
                continue
            x = np.array([a for a, _ in vv]); y = np.array([b for _, b in vv])
            if x.std() == 0:
                continue
            sl.append(float(np.polyfit(x, y, 1)[0]))
        if len(sl) >= 5:
            st = A.sign_test(sl)
            n1[col] = {"median": statistics.median(sl), "neg": st["neg"],
                       "pos": st["pos"], "p": st["p"], "n_pairs": len(sl)}
            print("  zipf -> A|A surprisal, offset +%-3d  %+9.5f  %3d-/%3d+  p=%s"
                  % (k, statistics.median(sl), st["neg"], st["pos"],
                     ("%.4f" % st["p"]) if st["p"] is not None else "-"))
    e1, e2 = n1.get("off1"), n1.get("off16")
    verdicts["N1_not_frequency"] = bool(e1 and e1["median"] > 0 and e1["p"] < 0.05
                                        and e2 and (e2["p"] is None or e2["p"] > 0.05))
    res["N1"] = n1

    # N2: not transgression-graded — sexual must NOT be the strongest domain
    TRANS = {"sexual", "violence", "taboo"}
    n2 = {}
    for dom in sorted({r["domain"] for r in rowsAA if r.get("domain")}):
        bp = {}
        for (pair, prompt), cell in cells.items():
            f, m = cell.get(("faller", "A|A")), cell.get(("faller-matched", "A|A"))
            fb, mb = cell.get(("faller", "B|A")), cell.get(("faller-matched", "B|A"))
            if not (f and m and fb and mb) or f.get("domain") != dom:
                continue
            if all("k8" in x for x in (f, m, fb, mb)):
                bp.setdefault(pair, []).append((f["k8"] - fb["k8"]) - (m["k8"] - mb["k8"]))
        v = [statistics.median(x) for x in bp.values() if len(x) >= 3]
        if len(v) >= 5:
            st = A.sign_test(v)
            n2[dom] = {"median": statistics.median(v), "p": st["p"],
                       "neg": st["neg"], "pos": st["pos"]}
            print("  D fell-flat | domain %-14s %+9.5f  %3d-/%3d+  p=%s"
                  % (dom, statistics.median(v), st["neg"], st["pos"],
                     ("%.4f" % st["p"]) if st["p"] is not None else "-"))
    sig = {d: x for d, x in n2.items() if x["p"] is not None and x["p"] < 0.05}
    strongest = min(n2, key=lambda d: n2[d]["median"]) if n2 else None
    verdicts["N2_not_transgression_graded"] = bool(
        n2 and (n2.get("sexual", {}).get("p", 1) > 0.05) and strongest not in ("sexual",))
    res["N2"] = n2

    # N3: two-part — cumulative A|A null AND disjoint late bands negative
    n3 = {}
    for nm, lab in ([("k%d" % n_, "cumulative k=%d" % n_) for _, n_ in CUM]
                    + [("kfull", "cumulative full")]
                    + [("band%d" % s, "disjoint %d-%d" % (s, s + n_ - 1)) for s, n_ in BAND]):
        d = contrast(cells, "faller", "faller-matched", "A|A", nm)
        if d:
            n3[nm] = d
            print(show("A|A  %s" % lab, d, w=30))
    cum_null = all(n3[k]["p"] is None or n3[k]["p"] > 0.05
                   for k in ("k4", "k8", "k16", "k32", "kfull") if k in n3)
    late = [n3[k] for k in ("band65", "band129") if k in n3]
    late_neg = bool(late) and all(x["median"] > 0 and x["p"] is not None and x["p"] < 0.05
                                  for x in late)
    verdicts["N3_not_a_shock"] = bool(cum_null and late_neg)
    res["N3"] = n3
    print("     (A|A here is LOGPROB: a POSITIVE median = less surprising)")

    print("\n" + "=" * 78)
    print("  VERDICTS — computed from the plan's own criteria, not asserted")
    print("=" * 78)
    for k, v in verdicts.items():
        print("    %-32s %s" % (k, "SUPPORTED" if v else "NOT SUPPORTED"))
    res["verdicts"] = verdicts
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("\n  wrote %s" % OUT)
    print("  PLAN §7 STANDS: this is a re-analysis. Nothing here is confirmed.")


if __name__ == "__main__":
    main()
