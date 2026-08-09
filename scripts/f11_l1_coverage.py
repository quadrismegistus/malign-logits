#!/usr/bin/env python
"""f11_l1_coverage.py — N3's `unresolved` and the demotion rate, without a coder.

    scripts/f11_l1_coverage.py             report
    scripts/f11_l1_coverage.py --json      machine-readable

**WHY THIS NEEDS NO CODING.** N3 (`n3_frame_exit_registration.md` §2.3):

    in_frame   = pole1_mass + pole2_mass + IN-FRAME mass
    off_frame  = sum of P over OFF-FRAME surfaces
    unresolved = 1 - (in_frame + off_frame)

**Every CANDIDATE surface is coded into one of the four classes**, so `unresolved`
is the mass the candidate vocabulary DOES NOT REACH -- below theta, or removed by
the §2.2 filter (non-alphabetic, len < 2, zero English unigram frequency).
IN-FRAME and OFF-FRAME absorb everything else, generic verbs included. So a
POLE1+POLE2 share of ~0.10 implies NOTHING about the demotion rule ([5158] ->
[5160]): pole-share and candidate-coverage are different quantities.

**THE ROSTER IS EVERY CHECKPOINT WITH DATA, NOT A SLICE.** The scratch version of
this computed on `sorted(ckpts)[:14]` -- the first fourteen models ALPHABETICALLY
-- and its 49.0% reached a ruling that governs a spend. An arbitrary prefix is
not a population, which is the defect I had flagged about the seven edges one
post earlier and then committed myself. Cells are read wherever twp data exists
and the roster actually read is printed with the number.
"""
import argparse, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
THETA = 0.001
DEMOTE = 0.50


def eligible(w):
    """AMENDED per the pen's ruling on [5169] -> [5170]: NO SHAPE FILTER.
    Everything above theta is codeable.

    The old predicate was `len(w) >= 2 and w.isalpha() and
    word_frequency(w, "en") > 0` -- three tests of which the frequency one
    removed 0.0% of ENGLISH mass and 53.2% of Chinese, and `len >= 2` removed a
    further 46.3% of Chinese (single-character words: 在 用 对 和 为). Inherited
    from F40 by citation; F40 was English-only, where it was harmless AND inert.

    And N3 §2.2 already defines OFF-FRAME to include "punctuation-led
    continuations" and "list/format tokens" -- so the filter was deleting a
    class the registration says the CODER decides, by fiat, with no kappa.

    `unresolved` now means theta truncation and nothing else, which makes it a
    property of the instrument rather than of a design choice."""
    return bool(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--by-role", action="store_true",
                    help="coverage BY ROLE, and the differential-coverage test")
    ap.add_argument("--csv", help="write the per-cell rows here")
    a = ap.parse_args()

    from f11_quintuplet_spec import PROMPT_ROLES
    from malign_logits.cache import get_cache
    from malign_logits.registry import Registry
    cm = get_cache()

    q = json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))["quintuplets"]
    items = q.items() if isinstance(q, dict) else [(e.get("group"), e) for e in q]
    groups, control = {}, {}
    for gid, v in items:
        if not isinstance(v, dict):
            continue
        name, st = v.get("group", gid), (v.get("status") or "").upper()
        cells = {r: v.get(r) for r in PROMPT_ROLES
                 if isinstance(v.get(r), str) and v.get(r)}
        if "RETIRED" in st:
            continue
        (control if name.startswith("f11_reason") else groups)[name] = cells

    ckpts = sorted({m for p in Registry().base_aligned_pairs()
                    for m in (p["base"], p["aligned"])})
    rows, seen_ck = [], set()
    for mid in ckpts:
        for g, roles in groups.items():
            for role, prompt in roles.items():
                v = cm.get_true_word_probs(mid, prompt, theta=THETA)
                if not v or not v.get("rows"):
                    continue
                seen_ck.add(mid)
                cand = filt = 0.0
                for r in v["rows"]:
                    p = r.get("p", 0.0)
                    if p < THETA:
                        continue
                    (cand := cand) if False else None
                    if eligible(r.get("word", "")):
                        cand += p
                    else:
                        filt += p
                res = v.get("residual") or {}
                tail = sum(res.get(k, 0) or 0 for k in ("tail", "drop", "open")) \
                    if isinstance(res, dict) else 0.0
                rows.append({"model": mid, "group": g, "role": role,
                             "candidate": cand, "filtered": filt,
                             "tail": tail, "unresolved": 1.0 - cand})

    n = len(rows)
    unres = [r["unresolved"] for r in rows]
    dem = sum(1 for u in unres if u > DEMOTE)
    rep = {
        "roster_read": len(seen_ck), "roster_available": len(ckpts),
        "groups": len(groups), "cells": n,
        "candidate_mean": statistics.mean(r["candidate"] for r in rows),
        "candidate_median": statistics.median(r["candidate"] for r in rows),
        "unresolved_mean": statistics.mean(unres),
        "unresolved_median": statistics.median(unres),
        "tail_mean": statistics.mean(r["tail"] for r in rows),
        "filter_mean": statistics.mean(r["filtered"] for r in rows),
        "demoted": dem, "demoted_pct": 100.0 * dem / n if n else 0.0,
        "demote_threshold": DEMOTE, "theta": THETA,
    }
    if a.json:
        print(json.dumps(rep, indent=1)); return
    print("N3 CANDIDATE COVERAGE — no coder involved")
    print("  roster read      %d of %d checkpoints with twp data"
          % (rep["roster_read"], rep["roster_available"]))
    print("  groups           %d (status-filtered; reason/_zh held beside)" % rep["groups"])
    print("  cells            %d\n" % n)
    print("  candidate mass (in_frame + off_frame)   mean %.3f  median %.3f"
          % (rep["candidate_mean"], rep["candidate_median"]))
    print("  UNRESOLVED                              mean %.3f  median %.3f"
          % (rep["unresolved_mean"], rep["unresolved_median"]))
    print("     twp tail, below theta                mean %.3f" % rep["tail_mean"])
    print("     removed by the N3 §2.2 filter        mean %.3f" % rep["filter_mean"])
    print("\n  CELLS DEMOTED (unresolved > %.2f):  %d of %d = %.1f%%"
          % (DEMOTE, dem, n, rep["demoted_pct"]))

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("  rows -> %s" % a.csv)
    if a.by_role:
        by_role(rows)


def by_role(rows):
    """DIFFERENTIAL COVERAGE: does the demotion rule select on the contrast?

    **A DEMOTED CELL IS A CELL THE ANALYSIS DOES NOT READ.** If BOTH cells are
    demoted at a different rate than their controls, then every declared
    contrast computed on surviving cells is conditioned on an outcome-correlated
    variable -- the cells that drop out are the ones where the model's
    continuation was least reachable, which is not independent of what the
    contrast is about. The pooled 26.2% cannot see this: it is one number over
    six roles.

    The statistic is the registration's OWN shape, applied to coverage instead
    of to a coded mass:

        excess_unresolved = unresolved(BOTH) - mean(unresolved(CONTROL_A),
                                                    unresolved(CONTROL_B))

    Paired within (checkpoint, group), so the pairing removes the group and the
    checkpoint at once, and the unit is the CHECKPOINT per [5152].
    """
    import statistics
    from collections import defaultdict
    per = defaultdict(dict)
    for r in rows:
        per[(r["model"], r["group"])][r["role"]] = r
    roles = ("pole_a", "pole_b", "both", "control_a", "control_b",
             "both_matched")

    #: **THE GATE IS NOT A POOLED NUMBER, AND POOLING HIDES WHICH HALF FAILS
    #: IT.** N3 §4 withdraws the instrument if a MAJORITY of cells are demoted.
    #: Pooled, this population does not come close. Split by script, one half
    #: has ZERO demoted cells and the other has a majority -- so the pooled
    #: figure is a mixture statistic and the second time this exact shape has
    #: appeared here ("47.5% demoted" was "the battery is 48% Chinese").
    #: ADD UP THE HALVES.
    print("\n  BY LANGUAGE — the gate, at the grain that decides it")
    print("    %-6s %8s %8s %9s %s" % ("lang", "cells", "unres", "demoted",
                                       "N3 §4 gate"))
    for lab, pred in (("en", lambda g: not g.endswith("_zh")),
                      ("zh", lambda g: g.endswith("_zh"))):
        rs = [r for r in rows if pred(r["group"])]
        if not rs:
            continue
        u = [r["unresolved"] for r in rs]
        d = 100.0 * sum(1 for x in u if x > DEMOTE) / len(u)
        print("    %-6s %8d %8.3f %8.1f%% %s"
              % (lab, len(rs), statistics.mean(u), d,
                 "FAILS (majority demoted)" if d > 50 else "passes"))

    print("\n  BY ROLE — coverage is not one number")
    print("    %-14s %7s %8s %8s %9s" % ("role", "cells", "unres", "median",
                                         "demoted"))
    for role in roles:
        rs = [r for r in rows if r["role"] == role]
        if not rs:
            continue
        u = [r["unresolved"] for r in rs]
        print("    %-14s %7d %8.3f %8.3f %8.1f%%"
              % (role, len(rs), statistics.mean(u), statistics.median(u),
                 100.0 * sum(1 for x in u if x > DEMOTE) / len(u)))

    #: per checkpoint, so a checkpoint contributes ONE number to the test
    ck = defaultdict(list)
    complete = 0
    for (mid, _g), d in per.items():
        if not all(k in d for k in ("both", "control_a", "control_b")):
            continue
        complete += 1
        ck[mid].append(d["both"]["unresolved"]
                       - (d["control_a"]["unresolved"]
                          + d["control_b"]["unresolved"]) / 2.0)
    vals = [statistics.mean(v) for v in ck.values() if v]
    if not vals:
        print("\n    no (checkpoint, group) triple complete on BOTH+controls")
        return
    pos = sum(1 for v in vals if v > 0)
    print("\n  DIFFERENTIAL COVERAGE  excess_unresolved(BOTH vs controls)")
    print("    triples complete   %d" % complete)
    print("    checkpoints        %d" % len(vals))
    print("    mean excess        %+.4f" % statistics.mean(vals))
    print("    median excess      %+.4f" % statistics.median(vals))
    print("    positive           %d of %d" % (pos, len(vals)))
    try:
        from scipy.stats import wilcoxon
        st, p = wilcoxon(vals)
        print("    Wilcoxon (roster)  W=%.1f  p=%.4g" % (st, p))
    except Exception as e:                      # scipy optional
        print("    Wilcoxon           unavailable (%s)" % e)
    print("    READING: a nonzero excess means the demotion rule is not")
    print("    independent of the contrast, and the declared masses inherit it.")


if __name__ == "__main__":
    main()
