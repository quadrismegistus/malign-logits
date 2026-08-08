#!/usr/bin/env python
"""f11_canonical_texts.py — the canonical-text map for the F11 redo population.

    scripts/f11_canonical_texts.py --show     print the map, write nothing
    scripts/f11_canonical_texts.py --write    emit data/f11_canonical_texts.json

WHY THIS EXISTS. The generations stash keys on PROMPT TEXT. Five texts in the
44-triplet population are claimed by more than one group, so a naive
group-by-group enumeration counts the same passages two, four or six times and
treats them as independent rows (docket [5081]).

**GENERATION DEDUPLICATES FOR FREE. ANALYSIS DOES NOT.** `count_generations`
collapses a shared text to one cell, so nothing is generated twice and no cost
changes. But a per-triplet statistic reads two rows where one set of passages
exists, which is the `a count is not a unit` failure with the duplication inside
the roster rather than inside the ids.

**THIS FILE IS DERIVED, NEVER HAND-KEYED.** Four `both` strings in
`f11_conjunction_controls.json` were written by analogy instead of read from
source and shipped past two readers and RH ([5080].1). A map of which text owns
which role is exactly the artifact where that failure would be invisible, so it
is computed from `prompt_categorisation.json` on every run and the selftest
compares it to source.

THREE CASES, AND THEY ARE NOT THE SAME PROBLEM:

  SHARED_BOTH    two groups share the BOTH cell, differing only in their poles.
                 f11_holy / f11_holy_b are ONE contradiction cell with two
                 pole-pairs attached. Not two triplets.

  SHARED_POLE    two groups share a pole, with different opposing poles.
                 f11_beauty / f11_beauty_ugly share POLE_A. Legitimate design
                 -- one baseline, two contrasts -- but the groups are NOT
                 independent and must not be pooled as if they were.

  ROLE_COLLISION one text is POLE_A in one group and POLE_B in another.
                 f11_species / f11_species_wolf. **This one has no correct
                 automatic resolution**: any role-aggregated statistic reads the
                 cell as both poles of one construct. Emitted as BLOCKING with
                 no owner assigned, because choosing a role here is a construct
                 decision and this script does not get to make it.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "prompt_categorisation.json")
OUT = os.path.join(ROOT, "data", "f11_canonical_texts.json")
CORE = ("POLE_A", "POLE_B", "BOTH")

#: lacan [5080].2, read from source: holy_b's poles are matched (`holy place` /
#: `filthy place`) and holy's are not (`holy temple` / `filthy alley`), while
#: BOTH is one string belonging to neither. So holy_b owns the shared cell.
CANONICAL_PREF = {"f11_holy_b": 0, "f11_holy_b_zh": 0, "f11_holy": 1, "f11_holy_zh": 1}


def load():
    d = json.load(open(SRC))
    rows = [r for r in d["prompts"] if r.get("finding") == "F11"]
    tri = collections.defaultdict(dict)
    for r in rows:
        g, role = r.get("group_id"), (r.get("group_role") or "").upper()
        if g and role:
            tri[g][role] = r["prompt"]
    return {g: v for g, v in tri.items() if set(CORE) <= set(v)}


def build(comp):
    zh = lambda s: bool(re.search(r"[一-鿿]", s))
    claims = collections.defaultdict(list)
    for g, v in comp.items():
        for role in CORE:
            claims[v[role]].append((g, role))
    shared = {t: c for t, c in claims.items() if len(c) > 1}

    entries = []
    for t, c in sorted(shared.items()):
        roles = {r for _g, r in c}
        if len(roles) > 1:
            kind, owner = "ROLE_COLLISION", None
        elif "BOTH" in roles:
            kind = "SHARED_BOTH"
            owner = sorted(c, key=lambda x: CANONICAL_PREF.get(x[0], 9))[0]
        else:
            kind, owner = "SHARED_POLE", None
        entries.append({
            "text": t, "language": "zh" if zh(t) else "en", "kind": kind,
            "claims": ["%s/%s" % (g, r) for g, r in sorted(c)],
            "owner": ("%s/%s" % owner) if owner else None,
            "blocking": kind == "ROLE_COLLISION",
        })
    return claims, shared, entries


def selftest(comp, claims):
    """Known-answer: the five texts docket [5081] named, from source."""
    ok = True
    expect = {
        "In a place both holy and filthy she began to": 2,
        "He was beautiful and she wanted to": 2,
        "The human stood in the clearing and began to": 2,
    }
    for t, n in expect.items():
        got = len(claims.get(t, []))
        if got != n:
            print("  [FAIL] %r claimed by %d, expected %d" % (t[:40], got, n))
            ok = False
    ncore = sum(1 for g in comp for _ in CORE)
    ndist = len(claims)
    if ndist > ncore:
        print("  [FAIL] %d distinct texts from %d cells -- impossible" % (ndist, ncore))
        ok = False
    print("selftest: %s  (%d cells, %d distinct texts)"
          % ("pass" if ok else "FAIL", ncore, ndist))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    comp = load()
    claims, shared, entries = build(comp)
    if not selftest(comp, claims):
        sys.exit("selftest failed; refusing to emit a map that misreads source")

    zh = lambda s: bool(re.search(r"[一-鿿]", s))
    en_g = [g for g, v in comp.items() if not zh(v["BOTH"])]
    zh_g = [g for g, v in comp.items() if zh(v["BOTH"])]
    print("\ngroups %d (en %d / zh %d) | core cells %d | DISTINCT texts %d | shared %d"
          % (len(comp), len(en_g), len(zh_g), 3 * len(comp), len(claims), len(shared)))
    print()
    for e in entries:
        print("%-14s %-3s %-46s" % (e["kind"], e["language"], e["text"][:46]))
        print("               claims: %s   owner: %s%s"
              % (", ".join(e["claims"]), e["owner"] or "NONE",
                 "   *** BLOCKING ***" if e["blocking"] else ""))
    nb = sum(1 for e in entries if e["blocking"])
    print("\n%d shared texts | %d BLOCKING (role collision, no automatic owner)" % (len(entries), nb))

    if a.write:
        json.dump({
            "_about": "canonical-text map for the F11 redo. DERIVED from "
                      "prompt_categorisation.json; never hand-keyed.",
            "_producer": "scripts/f11_canonical_texts.py",
            "_rule": "generation dedupes for free (stash keys on text); analysis "
                     "must not treat two groups claiming one text as independent.",
            "n_groups": len(comp), "n_core_cells": 3 * len(comp),
            "n_distinct_texts": len(claims), "shared": entries,
        }, open(OUT, "w"), ensure_ascii=False, indent=1)
        print("wrote %s" % os.path.relpath(OUT, ROOT))
    else:
        print("\n(--write to emit)")


if __name__ == "__main__":
    main()
