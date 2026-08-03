#!/usr/bin/env python3
"""AT 69 SHARDS, HOW MANY LINEAGES CARRY A PAIRED M04 CELL IN >= 1 CATEGORY?

Ordered at [3789], ahead of RH's Route A word: if 25, the charter ratifies as
written; if nearer 13, its n and MDE table re-derive first, because otherwise
*"he would be ratifying a power calculation against a population that does not
exist yet."*

**MIRRORS THE PRODUCER, DOES NOT REINVENT IT.** Lineage resolution, the category
derivation (`domain + "_" + subdomain`, never `domain` alone — [2356]/[2360]),
`BATTERY_CATEGORIES` and the `(lineage, category) -> {roles}` cell shape are all
imported from `meta/M01_displacement/scripts/m04_producer.py`. The only thing
this file changes is the SOURCE OF KEYS: the producer's stage 1 scans the beam
stash, and the question is about the 69 RE-SCORED SHARDS actually on disk.

A cell is PAIRED when both roles are present for one (lineage, category) —
`source` from the beam's originating model, `judge` from the re-scoring model.
A lineage counts when it has at least one paired category.

**Two counts, two units, one word "lineages" ([3789]): beams-in-both-arms is
not paired-scored-cells.** This reports both, plus the per-category table, so
the number that reaches the charter carries its unit.

    python scripts/m04_paired_lineages.py
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROD = os.path.join(ROOT, "meta", "M01_displacement", "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, PROD)

import m04_producer as MP                                    # noqa: E402

SHARDS = os.path.join(ROOT, "data", "m04_rescore_shards", "*.json")
OUT = os.path.join(ROOT, "data", "m04_paired_lineages.json")


def categories():
    """The producer's own derivation, imported by construction."""
    def _cat(r):
        d, sd = r.get("domain"), r.get("subdomain")
        return f"{d}_{sd}" if d and sd else d
    return {r["prompt"]: _cat(r)
            for r in json.load(open(MP.CATEGORISATION))["prompts"]}


def main():
    lin = MP.Lineages()
    cats = categories()
    print(f"lineage map: {lin.n_lineages} lineages")
    print(f"categorisation: {len(cats)} prompts")
    print(f"BATTERY_CATEGORIES: {len(MP.BATTERY_CATEGORIES)}\n")

    cells = collections.defaultdict(set)          # (lineage, cat) -> {roles}
    shard_files = sorted(glob.glob(SHARDS))
    n_keys = off_cat = uncat = unmapped_src = unmapped_mod = 0
    unmapped = set()
    for g in shard_files:
        d = json.load(open(g))
        for k in d.get("probs", {}):
            parts = k.split("|")
            if len(parts) < 4:
                continue
            src, mod, pr = parts[0], parts[1], "|".join(parts[2:-1])
            n_keys += 1
            l_src, l_mod = lin.of(src), lin.of(mod)
            if l_src is None:
                unmapped_src += 1
                unmapped.add(src)
            if l_mod is None:
                unmapped_mod += 1
                unmapped.add(mod)
            cat = cats.get(pr)
            if cat is None:
                uncat += 1
                continue
            if cat not in MP.BATTERY_CATEGORIES:
                off_cat += 1
                continue
            if l_src is not None:
                cells[(l_src, cat)].add("source")
            if l_mod is not None:
                cells[(l_mod, cat)].add("judge")

    paired = {(l, c) for (l, c), r in cells.items() if r == {"source", "judge"}}
    lin_paired = sorted({l for l, _ in paired})
    lin_present = sorted({l for l, _ in cells})
    per_cat = collections.Counter(c for _, c in paired)

    print(f"shards read                    {len(shard_files):>7}")
    print(f"scored keys                    {n_keys:>7,}")
    print(f"  uncategorised prompts        {uncat:>7,}")
    print(f"  outside the nine             {off_cat:>7,}")
    print(f"  source id off the map        {unmapped_src:>7,}")
    print(f"  judge  id off the map        {unmapped_mod:>7,}")
    if unmapped:
        print(f"  unmapped identifiers ({len(unmapped)}): {sorted(unmapped)[:6]}")
    print()
    print(f"  (lineage, category) cells present      {len(cells):>5}")
    print(f"  ...of which PAIRED (source AND judge)  {len(paired):>5}")
    print()
    print(f"  LINEAGES PRESENT IN ANY ROLE           {len(lin_present):>5}")
    print(f"  **LINEAGES WITH A PAIRED CELL IN >=1 CATEGORY  {len(lin_paired):>5}**")
    print(f"     -> charter's 25 {'HOLDS' if len(lin_paired) >= 25 else 'DOES NOT HOLD'}"
          f" at 69 shards")
    print()
    print("  PAIRED LINEAGES PER CATEGORY:")
    for c in sorted(MP.BATTERY_CATEGORIES):
        print(f"    {c:<26} {per_cat.get(c, 0):>4}")
    print()
    print("  PER-LINEAGE paired-category count:")
    per_lin = collections.Counter(l for l, _ in paired)
    for l in lin_paired:
        print(f"    {l:<34} {per_lin[l]:>2} of {len(MP.BATTERY_CATEGORIES)}")
    only_one = [l for l in lin_paired if per_lin[l] == 1]
    print(f"\n  lineages paired in EXACTLY ONE category: {len(only_one)}  {only_one}")

    json.dump({"shards": len(shard_files), "scored_keys": n_keys,
               "cells": len(cells), "paired_cells": len(paired),
               "lineages_present": lin_present,
               "lineages_paired": lin_paired,
               "per_category": dict(per_cat),
               "per_lineage": dict(per_lin),
               "uncategorised": uncat, "off_category": off_cat,
               "unmapped": sorted(unmapped)}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
