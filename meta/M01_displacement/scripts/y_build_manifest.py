#!/usr/bin/env python
"""Freeze exactly which passages get annotated. Build once, then run against it.

    python y_build_manifest.py            # builds, refuses if it exists
    python y_build_manifest.py --census   # describe the existing one, build nothing

WHY A MANIFEST AND NOT JUST A SEEDED SAMPLE IN THE RUNNER. A seed makes a draw
reproducible only if every input to the draw is also unchanged -- the corpus
files, the glob order, the filter thresholds, the exclusion list. All four have
moved this week. A manifest separates "which passages" from "what the coder
said", so the selection can be audited without rerunning the coder, and a
re-run can be checked against it rather than trusted.

Each row carries a sha256 of the passage text. That is what makes the pin real:
if the corpus is ever regenerated, the manifest does not silently point at
different passages, it fails to match.

REFUSES TO OVERWRITE. The refusal is the part that makes "frozen" mean
anything; a file that can be rebuilt on a whim records a decision nobody made.

POPULATION (all four filters stated here, not inherited):
    cross-scored pairs only     3 pairs blocked on vocabulary mismatch, and the
                                block is permanent -- 32000 vs 32001 cannot be
                                embedded, so those pairs can never carry a
                                surprisal beside a span
    bloomz pair excluded        1 of 1700 sequences reaches full length; there
                                is no full-length population to sample
    PASS A  tokens >= 256       uniform length, so a per-passage rate stops
                                conflating "did it happen" with "was there room"
    PASS B  11 <= tokens < 256  the refusal-by-stopping population, which pass
                                A's filter would otherwise discard: refusal
                                markers peak at 51-100 tokens (14.0% aligned)
                                and are LOWEST at the ceiling (7.4%)
    NEITHER tokens <= 10        no text to annotate; measured by token count

PASS A IS SAMPLED, PASS B IS A CENSUS, and the asymmetry is deliberate. A is
estimating content rates in a large homogeneous pool, so 20 per cell suffices.
B is measuring a rare event whose whole interest is its arm asymmetry, and its
median cell holds 2 -- demanding cell balance there would discard the
asymmetry being measured.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
OUT = os.path.join(CAMP, "registrations", "y_annotation_manifest.jsonl")

SEED = 20260808
N_PER_CELL = 20
FULL = 256
MIN_B = 11


def load_pool():
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl")))
             if "FAILED" not in f]
    rows, skipped = [], collections.Counter()
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                skipped["unparseable"] += 1
                continue
            if "sequences" not in r:
                continue
            pair = r.get("pair") or "?"
            if "bloomz" in pair:
                skipped["bloomz pair"] += len(r["sequences"]); continue
            if r.get("cross_score_blocked"):
                skipped["cross-score blocked"] += len(r["sequences"]); continue
            for i, s in enumerate(r["sequences"]):
                txt = s.get("text") or ""
                n = len(s.get("tokens") or [])
                if n >= FULL:
                    band = "A"
                elif n >= MIN_B:
                    band = "B"
                else:
                    skipped["<=10 tokens (no text)"] += 1
                    continue
                rows.append({
                    "pair": pair, "model": r.get("model"), "role": r.get("role"),
                    "prompt_id": r.get("prompt_id"), "word": r.get("word"),
                    "seq_i": i, "pass": band, "n_tokens": n, "n_chars": len(txt),
                    "sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16],
                })
    return rows, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true", help="describe the existing manifest")
    a = ap.parse_args(argv)

    if a.census:
        if not os.path.exists(OUT):
            print("no manifest at %s" % OUT); return 1
        rows = [json.loads(l) for l in open(OUT)]
        describe(rows); return 0

    if os.path.exists(OUT):
        print("REFUSED: %s already exists." % OUT)
        print("A manifest that can be rebuilt on a whim records a decision nobody made.")
        print("Delete it deliberately if the selection is genuinely being redrawn.")
        return 1

    pool, skipped = load_pool()
    print("eligible sequences: %d" % len(pool))
    for k, v in skipped.most_common():
        print("   excluded %-24s %d" % (k, v))

    rng = random.Random(SEED)
    chosen = []
    #: PASS A -- balanced within cell. The cap is min(20, min over the two
    #: arms), so a cell never contributes more base than aligned: the unit of
    #: comparison is the cell and an asymmetric n inside it would put a
    #: precision difference where the contrast is.
    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in pool:
        if r["pass"] == "A":
            cell[(r["pair"], r["prompt_id"], r["word"])][r["role"]].append(r)
    short_cells = 0
    for k in sorted(cell, key=lambda x: tuple(str(y) for y in x)):
        arms = cell[k]
        take = min(N_PER_CELL, min(len(arms.get("base", [])), len(arms.get("aligned", []))))
        if take < N_PER_CELL:
            short_cells += 1
        for role in ("base", "aligned"):
            v = sorted(arms.get(role, []), key=lambda r: r["seq_i"])
            chosen.extend(rng.sample(v, take) if take else [])
    #: PASS B -- census. Every one.
    chosen.extend(r for r in pool if r["pass"] == "B")

    for i, r in enumerate(sorted(chosen, key=lambda r: (r["pass"], r["pair"], str(r["prompt_id"]),
                                                        str(r["word"]), r["role"], r["seq_i"]))):
        r["mid"] = "y%06d" % i
    chosen.sort(key=lambda r: r["mid"])

    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in chosen)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_manifest": True, "seed": SEED, "n_per_cell": N_PER_CELL,
                             "pass_A_min_tokens": FULL, "pass_B_range": [MIN_B, FULL - 1],
                             "cells_short_of_%d" % N_PER_CELL: short_cells,
                             "rows": len(chosen), "sha256": digest},
                            sort_keys=True) + "\n")
        fh.write(blob + "\n")
    print("\nwrote %s" % OUT)
    print("   rows %d   sha256 %s" % (len(chosen), digest[:32]))
    print("   cells short of %d: %d" % (N_PER_CELL, short_cells))
    describe(chosen)
    return 0


def describe(rows):
    rows = [r for r in rows if not r.get("_manifest")]
    print("\nMANIFEST CENSUS")
    byp = collections.Counter((r["pass"], r["role"]) for r in rows)
    print("  %-8s %9s %9s %9s" % ("pass", "base", "aligned", "total"))
    for p in ("A", "B"):
        b, a = byp[(p, "base")], byp[(p, "aligned")]
        print("  %-8s %9d %9d %9d" % (p, b, a, b + a))
    print("  %-8s %9d %9d %9d" % ("TOTAL", byp[("A", "base")] + byp[("B", "base")],
                                  byp[("A", "aligned")] + byp[("B", "aligned")], len(rows)))
    print("\n  pairs %d   prompts %d   words %d   distinct passages %d"
          % (len({r["pair"] for r in rows}), len({r["prompt_id"] for r in rows}),
             len({r["word"] for r in rows}), len({r["sha256"] for r in rows})))
    ch = [r["n_chars"] for r in rows]
    print("  passage chars: median %d   mean %.0f   total %.1fM"
          % (statistics.median(ch), statistics.mean(ch), sum(ch) / 1e6))
    #: PASS A MUST BE ARM-BALANCED CELL BY CELL. Equal totals can hide a cell
    #: that leans one way cancelling one that leans the other, so it is checked
    #: per cell rather than in aggregate.
    cc = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r["pass"] == "A":
            cc[(r["pair"], r["prompt_id"], r["word"])][r["role"]] += 1
    bad = [k for k, v in cc.items() if v["base"] != v["aligned"]]
    print("  pass A cells: %d   arm-imbalanced: %d  %s"
          % (len(cc), len(bad), "" if not bad else "<-- BUG, should be zero"))
    n = collections.Counter(min(v["base"], v["aligned"]) for v in cc.values())
    print("  pass A per-cell n: %s" % dict(sorted(n.items(), reverse=True)[:6]))
    #: 529/item is OBSERVED on 200 confirmatory-length passages, not
    #: extrapolated. The earlier 847 scaled the WHOLE item by passage length
    #: when only `tagged` scales with it. Pass B is prorated by chars because
    #: its passages are shorter.
    est = (sum(529 for r in rows if r["pass"] == "A")
           + sum(529 * r["n_chars"] / 1035.0 for r in rows if r["pass"] == "B"))
    print("\n  estimated output tokens: %s  (%.1fM)" % (format(int(est), ","), est / 1e6))
    dup = len(rows) - len({r["sha256"] for r in rows})
    if dup:
        print("  rows sharing a passage hash: %d -- distinct (pair,prompt,word,seq_i)"
              % dup)
        print("     cells whose text converged, not duplicated rows. The pilot")
        print("     showed the same thing and every case was one model.")


if __name__ == "__main__":
    sys.exit(main())
