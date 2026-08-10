#!/usr/bin/env python3
"""PLAN H2's PRODUCER — is alignment's effect distributed through the stack, or
concentrated in the last layers?

Plan: `registrations/plan_h2_alignment_depth.md` §5.
Run by RH locally, 2026-08-10; receipt `data/h2_depth_receipt.json`; shards
`data/h2_depth/*.canonical.jsonl` written by `scripts/twp_depth_battery.py`.

    python meta/M01_displacement/scripts/h_depth_primary.py
    python meta/M01_displacement/scripts/h_depth_primary.py --json out.json

## THE PRIMARY IS A RAW DIFFERENCE, AND THE CEILING IS A COLUMN

    d = recovery(all-but-last-2) - recovery(last-2)

computed INSIDE each cell, from that cell's own `bottom[N-2]` and `top[2]`.

**THE CEILING IS NEVER A DIVISOR.** §5 of the plan settles this and the reason
is arithmetic, not taste: `d_norm = d_raw / ceiling` does NOT cancel the shared
denominator, it SCALES it. A cell whose ceiling is 0.07 inflates 14x and a
NEGATIVE ceiling flips the sign of a difference that was never in doubt. So the
ceiling enters as a four-level CLASS and gates membership; it never touches the
quantity.

    failed   ceiling <= 0     construction failed; the cell has no scale
    low      ceiling < 0.5    a scale too small to divide by
    normal   0.5 .. 1.2
    over     ceiling > 1.2

`failed` and `low` are excluded. **The gate is reported beside the ungated
number on purpose** -- if gating moved the headline, the headline would be a
fact about the gate.

## THE UNIT IS THE PAIR, NOT THE CELL

4,318 gated cells are not n=4,318. Cells within a pair share a base model, a
tokenizer and a training recipe, and they move together. The pair-level line is
the one to quote; the cell-level line is reported because a per-pair median
hides how wide the within-pair spread is, not because it is an independent n.

## REVERSALS ARE NAMED, NOT COUNTED

§5 requires it. A count of reversals is a number that cannot be checked; a name
can be looked up, and one named pair turned out to carry a third of them.

## WHAT THIS SCRIPT DOES NOT DO

No re-derivation of `recovery`, `ceiling`, `N` or the lens permission -- those
are the battery's and are read as written. This file is the READING of §5 and
nothing else: if the shards are wrong, this script faithfully reports wrong
numbers, which is the correct division of labour between a producer and a
reader.

**TWO PAIRS OF 25 CONTRIBUTED NOTHING** and are reported as absent rather than
silently reducing n: `llm-jp/llm-jp-3-7.2b-instruct3` and
`m-a-p/neo_7b_instruct_v0.1` recorded 231 of 231 prompts as `no_row`. Cause,
established 2026-08-10 and NOT a property of those checkpoints: the shards were
built through `movement.word_probs`, which returns None for llm-jp under the
ClickHouse default while the hashstash holds 204 words for the same cell. The
twp data exists (2,590 cells on both arms). RH ruled the gap out of scope for
this reading; it is recorded here because "23 pairs" must not read as "the
design had 23".
"""

import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))

SHARDS = os.path.join(ROOT, "data", "h2_depth", "*.canonical.jsonl")
RECEIPT = os.path.join(ROOT, "data", "h2_depth_receipt.json")

#: §5. `failed` and `low` are excluded from the gated population.
CEILING_BANDS = (("failed", None, 0.0), ("low", 0.0, 0.5),
                 ("normal", 0.5, 1.2), ("over", 1.2, None))
KEEP = ("normal", "over")


def ceiling_class(c):
    if c is None or c <= 0:
        return "failed"
    if c < 0.5:
        return "low"
    return "over" if c > 1.2 else "normal"


def load_cells():
    """One record per (pair, prompt) with the primary already differenced.

    Reads `bottom[str(N-2)]` and `top["2"]` -- the keys the battery writes. A
    cell missing either is DROPPED AND COUNTED, never defaulted to zero: a
    missing recovery is not a recovery of nothing.
    """
    cells, dropped = [], 0
    for f in sorted(glob.glob(SHARDS)):
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                dropped += 1
                continue
            top, bot, N = r.get("top"), r.get("bottom"), r.get("N")
            if not top or not bot or not N:
                dropped += 1
                continue
            b, t = bot.get(str(N - 2)), top.get("2")
            if b is None or t is None:
                dropped += 1
                continue
            cells.append({"pair": r["aligned"], "base": r.get("base"),
                          "prompt": r.get("prompt"), "d": b - t,
                          "ceiling": r.get("ceiling"),
                          "cls": ceiling_class(r.get("ceiling")),
                          "L50": r.get("repr_L50"), "N": N,
                          "n_fallers": r.get("n_fallers"),
                          "n_risers": r.get("n_risers")})
    return cells, dropped


def describe(ds):
    ds = sorted(ds)
    n = len(ds)
    if not n:
        return None
    return {"n": n, "median": st.median(ds),
            "q1": ds[n // 4], "q3": ds[(3 * n) // 4],
            "n_pos": sum(1 for x in ds if x > 0),
            "frac_pos": sum(1 for x in ds if x > 0) / n,
            "min": ds[0], "max": ds[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full result object here")
    a = ap.parse_args()

    cells, dropped = load_cells()
    if not cells:
        raise SystemExit("no cells read from %s -- has the sweep run?" % SHARDS)
    gated = [c for c in cells if c["cls"] in KEEP]

    print("PLAN H2 §5 — is alignment distributed through the stack?\n")
    print("  d = recovery(all-but-last-2) - recovery(last-2), RAW, per cell")
    print("  ceiling is a CLASS, never a divisor\n")

    print("  cells read      %5d   (%d unreadable/incomplete, dropped not defaulted)"
          % (len(cells), dropped))
    print("  pairs present   %5d" % len({c["pair"] for c in cells}))
    band = collections.Counter(c["cls"] for c in cells)
    print("  ceiling classes " + "  ".join("%s %d" % (k, band.get(k, 0))
                                           for k, _, _ in CEILING_BANDS))
    print("  gated (normal+over) %5d   (%d dropped by the gate)"
          % (len(gated), len(cells) - len(gated)))

    #: **THE UNGATED LINE IS PRINTED BESIDE THE GATED ONE.** If the gate moved
    #: the headline, the headline would be a fact about the gate.
    print("\n  CELL LEVEL (not an independent n; see the pair line)")
    for label, sel in (("ungated", cells), ("ceiling-gated", gated)):
        s = describe([c["d"] for c in sel])
        print("    %-14s n=%5d  median %+.3f  IQR [%+.3f, %+.3f]  d>0 %d/%d (%.1f%%)"
              % (label, s["n"], s["median"], s["q1"], s["q3"],
                 s["n_pos"], s["n"], 100 * s["frac_pos"]))

    per = collections.defaultdict(list)
    for c in gated:
        per[c["pair"]].append(c["d"])
    meds = {p: st.median(v) for p, v in per.items()}
    s = describe(list(meds.values()))
    print("\n  PAIR LEVEL — THE UNIT TO QUOTE")
    print("    pairs n=%d   median of per-pair medians %+.3f   range [%+.3f, %+.3f]"
          % (s["n"], s["median"], s["min"], s["max"]))
    print("    pairs with median d > 0: %d/%d" % (s["n_pos"], s["n"]))

    print("\n  PER PAIR (ascending)")
    print("    %-46s %7s %6s" % ("pair", "median", "cells"))
    for p, m in sorted(meds.items(), key=lambda kv: kv[1]):
        print("    %-46s %+7.3f %6d" % (p[:46], m, len(per[p])))

    #: §5: NAMED, not counted.
    rev_pairs = sorted([p for p, m in meds.items() if m <= 0])
    rev_cells = collections.Counter(c["pair"] for c in gated if c["d"] <= 0)
    print("\n  REVERSALS, NAMED (§5)")
    if rev_pairs:
        for p in rev_pairs:
            print("    PAIR REVERSES  %-44s median %+.3f" % (p[:44], meds[p]))
    else:
        print("    no pair reverses")
    print("    reversing CELLS: %d of %d gated, across %d pairs"
          % (sum(rev_cells.values()), len(gated), len(rev_cells)))
    for p, k in rev_cells.most_common(6):
        print("      %-46s %d" % (p[:46], k))

    #: **ABSENT PAIRS ARE NAMED SO "23" NEVER READS AS THE DESIGN.**
    if os.path.exists(RECEIPT):
        rec = json.load(open(RECEIPT))
        empty = [s2["pair"] for s2 in rec.get("shards", []) if not s2.get("rows")]
        if empty:
            print("\n  CONTRIBUTED NOTHING (%d of %d designed pairs)"
                  % (len(empty), len(rec.get("shards", []))))
            for p in empty:
                print("    %s" % p)
            print("    Cause is the reader, not the checkpoint: word_probs returns")
            print("    None under the ClickHouse default where the stash holds the")
            print("    cell. Out of scope for this reading (RH, 2026-08-10).")

    if a.json:
        out = {"_about": "Plan H2 §5 primary: d = recovery(all-but-last-2) "
                         "- recovery(last-2), raw, ceiling as class",
               "_producer": "meta/M01_displacement/scripts/h_depth_primary.py",
               "cell_ungated": describe([c["d"] for c in cells]),
               "cell_gated": describe([c["d"] for c in gated]),
               "pair_level": s, "per_pair_median": meds,
               "ceiling_classes": dict(band),
               "reversing_pairs": rev_pairs,
               "reversing_cells": dict(rev_cells)}
        json.dump(out, open(a.json, "w"), indent=1)
        print("\n  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
