#!/usr/bin/env python3
"""STEP (2): clear the twp store and re-ingest under the rule-keyed schema.

THE ONLY DESTRUCTIVE OPERATION IN THIS SEQUENCE, so it is a script rather than
typed commands: every gate is checked in one place, in order, and the run
refuses rather than asks.

    (2a) TRANSPORT COMPLETE   ruled [2967]. An ingest against a moving
                              transport produces a count matching no posted
                              receipt -- and the receipt is the only evidence
                              the clear was safe. The dress rehearsal watched
                              the transport grow from 99 to 101 shards
                              underneath it; this is not procedural.
    (2b) RESTORE RECEIPT      ruled [2963].1. Every resident cell must be
                              re-derivable from a named source directory,
                              verified present, so the clear is a cache
                              eviction with a receipt and not a deletion with
                              a hope.
    (2c) CLEAR                the store is MOVED, not deleted. A move is
                              reversible for the cost of a move.
    (2d) FLIP + INGEST        rule_version and dict_sha become key fields.
    (2e) VERIFY               count, rule count, read-back named and unnamed.

Nothing here writes to the live store until every gate above it has passed.

    python scripts/twp_step2_ingest.py --check      # gates only, no writes
    python scripts/twp_step2_ingest.py --execute
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "data/raw/cloud_run_20260801")
SPEC = os.path.join(ROOT, "data/grid_spec.json")
#: every directory the resident cells are re-derivable from. THREE, not one:
#: twp_grid_v3 alone covers 91,421 of 93,216 and a restore reaching for the
#: obvious single directory comes back 1,795 cells short and looks complete.
RESTORE_DIRS = ["data/twp_grid_v3", "data/twp_phase32b", "data/twp_phasefalcon"]


def fail(msg):
    print(f"\nREFUSING: {msg}")
    sys.exit(1)


def gate_transport_complete(expect_models=None):
    """(2a) Every roster model has a complete shard, and nothing is still being
    written."""
    files = sorted(glob.glob(SRC + "/*.jsonl"))
    raw = json.load(open(SPEC))
    spec = raw["spec"] if isinstance(raw, dict) else raw
    roster = {e["model"] for e in spec}
    expect = max(len(e["prompts"]) for e in spec)

    counts, newest = {}, 0.0
    for f in files:
        m = os.path.basename(f)[:-6].replace("__", "/")
        counts[m] = sum(1 for _ in open(f, errors="ignore"))
        newest = max(newest, os.path.getmtime(f))

    complete = {m for m, n in counts.items() if n >= expect}
    missing = sorted(roster - complete)
    quiet_s = time.time() - newest
    print(f"(2a) transport   {len(files)} shards, {len(complete)} complete "
          f"of {len(roster)} roster, expect {expect} rows")
    print(f"     newest shard written {quiet_s/60:.1f} min ago")
    if missing:
        print(f"     INCOMPLETE ({len(missing)}):")
        for m in missing[:12]:
            print(f"       {m}  {counts.get(m, 0)}/{expect}")
        fail("transport is not complete; an ingest now yields a count no "
             "receipt can match")
    #: a shard written seconds ago means the mirror is still running
    if quiet_s < 120:
        fail(f"newest shard is {quiet_s:.0f}s old -- the mirror is still "
             f"writing. Stop the sync loop and re-check.")
    return sum(counts.values())


def gate_restore_receipt():
    """(2b) Every resident cell is re-derivable from a named source on disk."""
    from malign_logits.cache import get_cache
    cm = get_cache()
    resident = {(d.get("model"), d.get("prompt"))
                for d in cm.iter_keys("true_word_probs")}
    union, per = set(), {}
    for d in RESTORE_DIRS:
        s = set()
        for f in glob.glob(os.path.join(ROOT, d) + "/*.jsonl"):
            m = os.path.basename(f)[:-6].replace("__", "/")
            for line in open(f, errors="ignore"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                s.add((r.get("model", m), r.get("prompt")))
        per[d] = len(resident & s)
        union |= s
    lost = resident - union
    print(f"(2b) receipt     resident {len(resident):,}   unrecoverable {len(lost):,}")
    for d in RESTORE_DIRS:
        print(f"       {d:<28} restores {per[d]:>7,}")
    if lost:
        for m, p in sorted(lost)[:6]:
            print(f"       LOST {m} / {p[:48]}")
        fail(f"{len(lost):,} resident cells have no source file; this would be "
             f"a deletion, not an eviction")
    return len(resident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="gates only, no writes")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.execute):
        ap.print_help()
        return 1

    print("STEP (2) — clear and re-ingest under the rule-keyed schema\n")
    rows = gate_transport_complete()
    resident = gate_restore_receipt()
    print(f"\n     transport rows {rows:,}   resident cells {resident:,}")

    if a.check:
        print("\n--check: all gates PASS, nothing written.")
        return 0

    from malign_logits.cache import CACHE_ROOT
    live = os.path.join(CACHE_ROOT, "true_word_probs")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    retired = f"{live}.RETIRED-{stamp}-preflip-{resident}cells"
    print(f"\n(2c) clear       MOVING (not deleting):")
    print(f"       {live}")
    print(f"    -> {retired}")
    shutil.move(live, retired)
    print(f"     moved. Reversible for the cost of a move.")

    print(f"\n(2d) flip + ingest — run:")
    print(f"       python scripts/twp_ingest.py --src {SRC}")
    print(f"     after committing the SCHEMAS flip to TRUE_WORD_PROBS_WITH_RULE.")
    print(f"     The flip and this clear must land together: a flipped schema "
          f"over an unkeyed store makes every read raise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
