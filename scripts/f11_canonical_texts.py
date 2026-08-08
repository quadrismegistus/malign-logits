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


def load(allowed=("ACTIVE",)):
    """Complete groups under a GROUP-WISE status filter. Returns (kept, excluded).

    **THE FILTER IS GROUP-WISE, NOT ROW-WISE, AND THE DIFFERENCE IS FATAL**
    (lacan [5085].1). `f11_reason` has an ACTIVE BOTH cell and DISPUTED/RETIRED
    poles; row-wise `status == ACTIVE` keeps the contradiction cell and deletes
    both baselines, yielding a triplet on which `excess = rate(BOTH) -
    mean(poles)` divides by an empty set. A group enters only if EVERY row it
    needs is live.

    **AND A ROLE CAN HAVE SEVERAL ROWS.** `f11_reason` POLE_A appears twice,
    same text, once DISPUTED and once RETIRED. Building a dict keyed by role
    silently keeps whichever row iterated last -- so the group's status
    depended on file order. Statuses are collected as a SET per role and the
    role is live only if every one of them is allowed; a role whose rows carry
    DIFFERENT TEXTS is AMBIGUOUS and excludes the group by name.
    """
    d = json.load(open(SRC))
    rows = [r for r in d["prompts"] if r.get("finding") == "F11"]
    cells = collections.defaultdict(lambda: {"texts": set(), "status": set(), "rows": []})
    for r in rows:
        g, role = r.get("group_id"), (r.get("group_role") or "").upper()
        if not (g and role):
            continue
        c = cells[(g, role)]
        c["texts"].add(r["prompt"])
        c["status"].add(r.get("status"))
        c.setdefault("rows", []).append((r.get("status"), r["prompt"]))
    groups = collections.defaultdict(dict)
    for (g, role), c in cells.items():
        groups[g][role] = c
    kept, excluded = {}, {}
    for g, roles in groups.items():
        missing = [r for r in CORE if r not in roles]
        if missing:
            excluded[g] = "incomplete: missing %s" % ",".join(missing)
            continue
        #: **HAS-A-LIVE-ROW, NOT ALL-ROWS-LIVE** (registrar [5088], correcting
        #: me). My first rule required every row of a role to be in `allowed`,
        #: which conflated TWO different things: a cell with a stale duplicate
        #: record, and a cell that is dead. f11_beauty carries ACTIVE and
        #: RETIRED rows of the IDENTICAL string; f11_gender has one ACTIVE BOTH
        #: beside a retired row with different text. Both are live and my rule
        #: dropped them -- 39 groups where 41 are alive.
        #:
        #: The refusal that survives is narrower and is the real hazard: TWO
        #: LIVE ROWS carrying DIFFERENT text, where picking either is arbitrary.
        live = {r: {t for st, t in roles[r]["rows"] if st in allowed}
                for r in CORE}
        dead = [r for r in CORE if not live[r]]
        if dead:
            excluded[g] = "not live: " + ", ".join(
                "%s=%s" % (r, "/".join(sorted(x or "?" for x, _ in roles[r]["rows"])))
                for r in dead)
            continue
        amb = [r for r in CORE if len(live[r]) > 1]
        if amb:
            excluded[g] = ("AMBIGUOUS: %s has MULTIPLE LIVE texts -- picking "
                           "either is arbitrary" % ",".join(amb))
            continue
        kept[g] = {r: next(iter(live[r])) for r in CORE}
    return kept, excluded


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
    """Two columns, and the first version of this conflated them.

    **A KNOWN-ANSWER TEST WHOSE EXPECTATIONS DEPEND ON THE POPULATION MUST SAY
    WHICH POPULATION.** v1 asserted that three texts are claimed twice -- true
    of the UNFILTERED source, false under ACTIVE, where f11_beauty and
    f11_species_wolf are excluded as retired. It failed on a correct run and
    would have been "fixed" by loosening it. The duplication facts are asserted
    against SOURCE, where they are invariant; the filter is asserted separately
    against its own behaviour.
    """
    ok = True
    d = json.load(open(SRC))
    raw = collections.defaultdict(list)
    for r in [x for x in d["prompts"] if x.get("finding") == "F11"]:
        g, role = r.get("group_id"), (r.get("group_role") or "").upper()
        if g and role in CORE:
            #: **DISTINCT (group, role) CLAIMS, NOT ROWS.** f11_beauty POLE_A
            #: has two rows for one text -- one ACTIVE, one RETIRED whose
            #: prompt_id is `store_g001_BOTH`, a BOTH-named id on a pole row
            #: (registrar's [5036].1 hazard, inside the 44). Counting rows made
            #: that text look claimed three times; the claim count is two.
            if (g, role) not in raw[r["prompt"]]:
                raw[r["prompt"]].append((g, role))
    for t, n in {"In a place both holy and filthy she began to": 2,
                 "He was beautiful and she wanted to": 2,
                 "The human stood in the clearing and began to": 2}.items():
        got = len(raw.get(t, []))
        if got != n:
            print("  [FAIL] SOURCE: %r claimed by %d, expected %d" % (t[:38], got, n))
            ok = False
    #: the group-wise filter must exclude f11_reason. Row-wise would keep its
    #: ACTIVE BOTH and delete both poles ([5085].1) -- a triplet with no baseline.
    if "f11_reason" in comp:
        print("  [FAIL] f11_reason survived an ACTIVE filter: row-wise leakage")
        ok = False
    ncore = 3 * len(comp)
    if len(claims) > ncore:
        print("  [FAIL] %d distinct texts from %d cells -- impossible"
              % (len(claims), ncore))
        ok = False
    print("selftest: %s  (source duplications verified; %d cells, %d distinct)"
          % ("pass" if ok else "FAIL", ncore, len(claims)))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--status", default="ACTIVE",
                    help="comma-separated allowed statuses; the POPULATION "
                         "definition, printed on every run. ACTIVE-vs-DISPUTED "
                         "is a construct ruling and has no safe default.")
    a = ap.parse_args()

    allowed = tuple(x.strip() for x in a.status.split(","))
    comp, excluded = load(allowed)
    print("POPULATION FILTER: status in %s, applied GROUP-WISE (all rows live)"
          % (allowed,))
    print("excluded groups: %d" % len(excluded))
    for g, why in sorted(excluded.items()):
        print("   %-24s %s" % (g, why))
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
