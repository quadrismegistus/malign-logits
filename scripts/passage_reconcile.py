#!/usr/bin/env python
"""passage_reconcile.py — what the passage-corpus fleet DELIVERED, against what was DECLARED

    scripts/passage_reconcile.py
    scripts/passage_reconcile.py --write-json data/passage_reconciliation.json

## WHY THIS EXISTS

The fleet's own reporting answers "did a box finish", which is a question about
the RENTAL. This answers "did the declared population arrive", which is the
question about the CORPUS, and the two come apart in four ways this run has
already produced at least one of each:

    a pair present but SHORT          (a restart that lost --eager, a crash
                                       mid-role: the file exists and is wrong)
    a pair present and DUPLICATED     (SmolLM2-360M: 3,688 rows, 1,844 distinct
                                       keys, every key exactly twice)
    a pair present in TWO box dirs    (a rebalance moved it; both copies are
                                       partial and neither is the answer)
    a pair ABSENT with no failure     (`FAILED.jsonl` was deleted on a restart,
                                       so `fail=0` means "no record", not "no
                                       failure" -- RWKV's loss survives in no log)

**Absence is counted here from the DECLARED POPULATION, never from the absence
of a failure file.** That is the whole point: a pair nobody recorded failing and
nobody collected is invisible to every other instrument in this run.

## THE UNIT, AND WHY DEDUP IS PART OF THE COUNT

A row is one (pair, role, prompt_id, word) cell. **The dedup key IS an
independence claim** (`feedback_unit_corrections`), so this script reports
distinct keys and raw rows as separate columns and never silently collapses
them: a pair at 200% of expectation with 100% distinct coverage is complete and
duplicated, which is a different repair from a pair at 100% raw with 50%
distinct, which is missing half its cells.

Expected rows per pair = 2 roles x that pair's arm-cells, reconstructed from the
FROZEN POPULATION and never from a manifest: `passage_rebalance.py` rewrites
manifests as pairs move, and 16 delivered pairs had already dropped out of every
manifest on disk. The reconstruction is validated against all 25 surviving
manifest pairs before it is used, and the script REFUSES to print a percentage if
that check fails.

## WHAT IT DOES NOT CLAIM

Row presence is not row correctness. A cell whose sequences are all unscorable
counts as delivered here and is reported in the `unscorable` column, because the
scoring contract records a dropped sequence as None and never reassigns it
(launch plan §8). Judging that is the ingest's job, not this script's.
"""
import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFESTS = os.path.join(ROOT, "data", "passage_manifests")
CORPUS = os.path.join(ROOT, "data", "raw", "passage_corpus")
POPULATION = os.path.join(ROOT, "data", "forced_arms_46reps_drmatch.json")


#: A manifest is NOT a denominator. `passage_rebalance.py` rewrites manifests as
#: pairs move, and a completed pair can drop out of every file on disk — 16 of
#: them had, so the first version of this script reported delivered pairs as
#: UNASSIGNED and could not tell a short pair from an unmeasurable one. The
#: authority is the FROZEN POPULATION, which cannot go stale
#: (`feedback_derived_values_go_stale`).
ARM_WORDS = ("faller", "matched", "riser", "riser_matched")

#: manifest stem -> the vast state file whose EXISTENCE means the box is alive.
#: Destroying a box renames its state file to `.DESTROYED-...`, so a pair flips
#: from IN_FLIGHT to ABSENT by itself — the staleness lives in the filename,
#: where the eye lands (`feedback_read_live_state_not_a_record_of_it`).
BOX_STATE = {"box0": ".vastai.passage0.json", "box0b": ".vastai.passage0b.json",
             "box1": ".vastai.passage1.json", "box2": ".vastai.passage2.json",
             "box3": ".vastai.passage3.json", "box4": ".vastai.passage4.json",
             "box5": ".vastai.passage5.json", "box6": ".vastai.passage6.json",
             "box7": ".vastai.passage7.json",
             "rescue_aquila_baichuan": ".vastai.rescue3.json",
             "kanana": ".vastai.kanana.json"}


def declared():
    """pair -> expected rows, reconstructed from the frozen population.

    An arm-cell is the undisturbed arm plus one per non-null forced word, and
    each is generated under BOTH roles. **The rule is not asserted: it is
    validated against every manifest still on disk before it is used** — it
    reproduces all 25 surviving per-pair cell counts exactly, which is what
    licenses applying it to the pairs whose manifests are gone
    (`feedback_mirror_the_producer`).
    """
    out = {}
    for c in json.load(open(POPULATION)).get("cells", []):
        key = c.get("pair")
        if not key:
            continue
        rec = out.setdefault(key, {"cells": 0, "prompts": 0})
        rec["cells"] += 1 + sum(1 for a in ARM_WORDS if c.get(a))
        rec["prompts"] += 1
    for rec in out.values():
        rec["expected"] = 2 * rec["cells"]
    return out


def assigned():
    """pair -> {manifests, live}. Provenance and liveness only, NOT the count."""
    out = {}
    for path in sorted(glob.glob(os.path.join(MANIFESTS, "*.json"))):
        cfg = json.load(open(path))
        box = os.path.basename(path)[:-5]
        live = os.path.exists(os.path.join(ROOT, BOX_STATE.get(box, "\0")))
        for p in cfg.get("pairs", []):
            key = "%s>%s" % (p["base"], p["aligned"])
            slots = p.get("prompts") or cfg.get("prompts") or []
            rec = out.setdefault(key, {"manifests": [], "live": False,
                                       "manifest_cells": set()})
            rec["manifests"].append(box)
            rec["live"] = rec["live"] or live
            rec["manifest_cells"].add(sum(len(s.get("cells") or []) for s in slots))
    return out


def validate_reconstruction(dec, asg):
    """The known-answer column, run BEFORE the reconstruction is trusted."""
    bad = []
    for key, a in asg.items():
        for got in a["manifest_cells"]:
            want = dec.get(key, {}).get("cells")
            if want is not None and got != want:
                bad.append((key, got, want))
    return bad


def collected():
    """pair -> delivery record, merged across box directories by dedup key."""
    per = {}
    for path in sorted(glob.glob(os.path.join(CORPUS, "*", "y__*.jsonl"))):
        box = os.path.basename(os.path.dirname(path))
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                per.setdefault("_MALFORMED", {"n": 0})["n"] += 1
                continue
            key = r.get("pair")
            rec = per.setdefault(key, {
                "rows": 0, "keys": set(), "boxes": collections.Counter(),
                "seqs": 0, "unscorable": 0, "roles": collections.Counter(),
                "fidelity": collections.Counter(), "blocked": 0})
            rec["rows"] += 1
            rec["keys"].add((r.get("role"), r.get("prompt_id"), r.get("word")))
            rec["boxes"][box] += 1
            rec["roles"][r.get("role")] += 1
            if r.get("cross_score_blocked"):
                rec["blocked"] += 1
            for s in r.get("sequences") or []:
                rec["seqs"] += 1
                if s.get("scored_by_base") is None or s.get("scored_by_aligned") is None:
                    rec["unscorable"] += 1
            fid = r.get("fidelity") or {}
            for role in ("base", "aligned"):
                st = (fid.get(role) or {}).get("status")
                if st:
                    rec["fidelity"][st] += 1
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-json")
    ap.add_argument("--quiet-complete", action="store_true",
                    help="list only pairs that are short, duplicated or absent")
    a = ap.parse_args()

    dec = declared()
    asg = assigned()
    got = collected()
    got.pop("_MALFORMED", None)

    bad = validate_reconstruction(dec, asg)
    if bad:
        print("RECONSTRUCTION REFUSED — the arm rule disagrees with a manifest,")
        print("so the denominator is not established and no percentage below")
        print("would mean anything:")
        for k, g, w in bad:
            print("  %-58s manifest %5d  reconstructed %5d" % (k[:58], g, w))
        return 2

    rows = []
    for key in sorted(set(dec) | set(asg) | set(got)):
        m = asg.get(key)
        g = got.get(key)
        exp = dec.get(key, {}).get("expected")
        distinct = len(g["keys"]) if g else 0
        raw = g["rows"] if g else 0
        pct = (100.0 * distinct / exp) if exp else None
        rows.append({
            "pair": key,
            "declared": key in dec,
            "assigned": bool(m),
            "live_box": bool(m and m["live"]),
            "expected": exp,
            "rows": raw,
            "distinct": distinct,
            "pct": pct,
            "dup_rows": raw - distinct,
            "boxes": sorted(g["boxes"]) if g else [],
            "sequences": g["seqs"] if g else 0,
            "unscorable": g["unscorable"] if g else 0,
            "blocked_rows": g["blocked"] if g else 0,
            "fidelity": dict(g["fidelity"]) if g else {},
            "roles": dict(g["roles"]) if g else {},
            "manifests": m["manifests"] if m else [],
        })

    def status(r):
        #: IN_FLIGHT before ABSENT: a pair on a live box has not failed, and
        #: calling it missing is the error that makes a running fleet look
        #: like a broken one.
        if r["live_box"] and (r["pct"] is None or r["pct"] < 98.0):
            return "IN_FLIGHT"
        if r["distinct"] == 0:
            return "ABSENT"
        if r["expected"] and r["pct"] < 98.0:
            return "SHORT"
        if r["dup_rows"] > 0:
            return "DUPLICATED"
        if not r["expected"]:
            return "UNDECLARED"
        return "ok"

    for r in rows:
        r["status"] = status(r)

    order = {"ABSENT": 0, "SHORT": 1, "DUPLICATED": 2, "UNDECLARED": 3,
             "IN_FLIGHT": 4, "ok": 5}
    rows.sort(key=lambda r: (order[r["status"]], r["pair"]))

    print("PASSAGE CORPUS RECONCILIATION")
    print("  declared population   %3d pairs   (%s)" % (len(dec),
                                                        os.path.basename(POPULATION)))
    print("  arm rule validated against %d manifest pairs, 0 disagreements"
          % sum(len(v["manifest_cells"]) for v in asg.values()))
    print("  assigned to a manifest %3d" % len(asg))
    print("  delivering rows        %3d" % sum(1 for r in rows if r["distinct"]))
    print()
    print("  %-52s %-11s %7s %7s %6s %6s" %
          ("pair", "status", "distinct", "expect", "pct", "dup"))
    for r in rows:
        if a.quiet_complete and r["status"] == "ok":
            continue
        print("  %-52s %-11s %7d %7s %5s%% %6d%s" % (
            r["pair"][:52], r["status"], r["distinct"],
            r["expected"] if r["expected"] is not None else "-",
            ("%.1f" % r["pct"]) if r["pct"] is not None else "  -",
            r["dup_rows"], "  on " + ",".join(r["boxes"]) if r["boxes"] else ""))

    tot = collections.Counter(r["status"] for r in rows)
    print()
    print("  " + "   ".join("%s=%d" % (k, v) for k, v in sorted(tot.items())))
    seqs = sum(r["sequences"] for r in rows)
    uns = sum(r["unscorable"] for r in rows)
    print("  sequences %s   unscorable %s (%.2f%%)   blocked rows %d"
          % (f"{seqs:,}", f"{uns:,}", 100.0 * uns / seqs if seqs else 0.0,
             sum(r["blocked_rows"] for r in rows)))
    fid = collections.Counter()
    for r in rows:
        fid.update(r["fidelity"])
    print("  fidelity verdicts  " + ("   ".join("%s=%s" % (k, f"{v:,}")
                                                for k, v in sorted(fid.items())) or "none"))
    absent = [r["pair"] for r in rows if r["status"] == "ABSENT" and r["declared"]]
    flight = [r["pair"] for r in rows if r["status"] == "IN_FLIGHT"]
    if flight:
        print()
        print("  IN FLIGHT on a live box — not missing, not yet delivered:")
        for p in flight:
            print("    %s" % p)
    if absent:
        print()
        print("  ABSENT AND DECLARED — each needs a reason in the amendment,")
        print("  and 'no failure was recorded' is not one:")
        for p in absent:
            print("    %s" % p)

    if a.write_json:
        json.dump({
            "_about": "Delivered-vs-declared for the passage corpus. Absence is "
                      "counted from the frozen population, never from the "
                      "absence of a failure record.",
            "_producer": "scripts/passage_reconcile.py",
            "_population": os.path.basename(POPULATION),
            "_dedup_key": "(pair, role, prompt_id, word)",
            "n_declared": len(dec),
            "totals": dict(tot),
            "sequences": seqs,
            "unscorable": uns,
            "pairs": rows,
        }, open(a.write_json, "w"), indent=1)
        print("\nwrote %s" % a.write_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
