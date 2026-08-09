#!/usr/bin/env python
"""Compare two codings of THE SAME passages under two versions of the field.

A rate that moves between two runs can move because the field changed or
because the draw changed. This joins on the passage itself -- (pair, group,
arm, first 60 characters of the continuation) -- so only rows present in both
runs are compared and the draw is held fixed by construction.

    l2_field_delta.py OLD.jsonl NEW.jsonl
    l2_field_delta.py OLD.jsonl NEW.jsonl --field tension_named --flips
"""
import argparse
import json
import os
import sys
from collections import Counter

FIELDS = ["frame_exit", "refusal", "tension_enacted", "tension_named",
          "tension_deliberated", "degenerate"]
COMPS = ["PERFORMED", "DESCRIBED", "BOTH_MODES", "OEDIPALIZED",
         "SPLIT_PERSONS", "EXITED"]


def key(r):
    return (r.get("pair", r["model"]), r["group"], r["arm"], r["text"][:60])


def load(p):
    return {key(r): r for r in (json.loads(l) for l in open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--field", default="tension_named")
    ap.add_argument("--flips", action="store_true",
                    help="print every passage whose --field answer changed")
    a = ap.parse_args()

    O, N = load(a.old), load(a.new)
    both = sorted(set(O) & set(N))
    print("old %d rows, new %d rows, %d passages in both"
          % (len(O), len(N), len(both)))
    if not both:
        sys.exit("no overlap -- different draws?")
    print()

    #: Report the arms separately throughout. A field edit can move the two
    #: arms by different amounts, and that is the thing most worth knowing:
    #: an edit that cleans base and aligned unequally has changed the contrast
    #: as a side effect of changing the construct.
    print("  %-18s %-19s %-19s %s"
          % ("", "base  old -> new", "aligned  old -> new", "diff old -> new"))
    for name in FIELDS + COMPS:
        def has(r):
            return (r[name] == "YES") if name in FIELDS else (name in r["composites"])
        cells = {}
        for arm in ("base", "aligned"):
            ks = [k for k in both if k[2] == arm]
            if not ks:
                continue
            cells[arm] = (100 * sum(has(O[k]) for k in ks) / len(ks),
                          100 * sum(has(N[k]) for k in ks) / len(ks), len(ks))
        if len(cells) < 2:
            continue
        b, al = cells["base"], cells["aligned"]
        if b[0] == b[1] == al[0] == al[1] == 0:
            continue
        print("  %-18s %5.1f -> %5.1f  %+5.1f  %5.1f -> %5.1f  %+5.1f  %+6.1f -> %+6.1f"
              % (name, b[0], b[1], b[1] - b[0], al[0], al[1], al[1] - al[0],
                 al[0] - b[0], al[1] - b[1]))

    f = a.field
    def hasf(r):
        return (r[f] == "YES") if f in FIELDS else (f in r["composites"])
    c = Counter((hasf(O[k]), hasf(N[k])) for k in both)
    print("\n  %s transition matrix over %d passages" % (f, len(both)))
    print("    kept YES   %3d      YES -> NO  %3d   (dropped)"
          % (c[(True, True)], c[(True, False)]))
    print("    kept NO    %3d      NO -> YES  %3d   (newly caught)"
          % (c[(False, False)], c[(False, True)]))

    if a.flips:
        for k in both:
            if hasf(O[k]) == hasf(N[k]):
                continue
            o, n = O[k], N[k]
            print("\n  %s  %s -> %s  [%s %s]"
                  % (k[0].split("/")[-1][:24], "YES" if hasf(o) else "NO",
                     "YES" if hasf(n) else "NO", k[2], k[1]))
            print("    %s%s" % (o["prompt"], o["text"][:150]))
            for r, lab in ((o, "old"), (n, "new")):
                sp = r.get(f.replace("tension_", "") + "_span", "")
                if sp:
                    print("    %s span: %r" % (lab, sp[:88]))


if __name__ == "__main__":
    main()
