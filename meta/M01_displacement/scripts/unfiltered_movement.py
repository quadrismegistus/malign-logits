#!/usr/bin/env python
"""Every word's riser and faller count on the 105-stem battery. NO vv* filter.

    uv run python unfiltered_movement.py          # ~10 min, writes the counts
    uv run python unfiltered_movement.py --show

WHY UNFILTERED. Every forced-arm instrument in this campaign selects words under
CLAWS `vv*`, the battery's own eligibility frame. That is defensible and it hides
a fifth of the phenomenon: **17.6% of riser events and 23.3% of faller events are
function words or pronouns**, and `he` is the most-fallen word of any kind in the
corpus -- more than any lexical verb. Nothing else in the repo holds this view.

THE RULE IS THE LIBRARY'S, NOT A COPY. `Step(base, aligned).cell(prompt)
.movement(CANONICAL)`. Reimplementing CANONICAL in SQL would be faster and would
be a second implementation of the rule every M01/M03 instrument shares; the
renormalisation null is exactly the part that would drift.

SPEED, SINCE IT LOOKS SLOW. ~0.06 s/cell is PYTHON, not I/O: `ch_read.prefetch`
already loads every cell for a model in ONE query and caches it, so 9,748 cells
cost ~92 queries. There is no ClickHouse win available here that does not cost a
duplicated rule.
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")
sys.path.insert(0, ROOT)

BATTERY = os.path.join(ROOT, "data", "beam_sample_105_plus_anger.csv")
PAIRS = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
OUT = os.path.join(CAMP, "results", "unfiltered_movement_counts.json")

#: for the FUNC tag in --show only; nothing is filtered by it
FUNC = set("the a an and or but of to in on at for with from by he she it they "
           "i we you him her them his hers its their my your our me us was were "
           "is are be been being had has have do does did not no so then now "
           "there here that this these those as if when while all some any "
           "one two".split())


def build():
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.movement import CANONICAL
    prompts = [r["prompt"].strip() for r in csv.DictReader(open(BATTERY))]
    pairs = [l.strip().split(">") for l in open(PAIRS) if l.strip()]
    rise, fall = collections.Counter(), collections.Counter()
    #: SIGNED MASS, not events. @registrar [5472]: B_field_flow has the
    #: grammatical/function bin as the SOLE RISER by MASS SHARE while this
    #: artifact had function words as the biggest fallers by EVENT COUNT. Both
    #: can hold -- T-14's few-large-against-many-small asymmetry at function-word
    #: grain -- and only signed mass decides which regime this is.
    mass = collections.Counter()
    #: PER PROMPT, which neither @registrar's frame test nor @malign's
    #: within-prompt gender test can be done without ([5472]/[5473]).
    per_prompt = collections.defaultdict(lambda: collections.defaultdict(
        lambda: [0, 0, 0.0]))          # prompt -> word -> [rise, fall, mass]
    cells = miss = 0
    t = time.time()
    for i, (b, a) in enumerate(pairs):
        try:
            s = Step(Checkpoint(b), Checkpoint(a))
        except Exception:
            continue
        for p in prompts:
            try:
                c = s.cell(p)
                m = c.movement(CANONICAL) if c else None
            except Exception:
                m = None
            if m is None:
                miss += 1
                continue
            cells += 1
            rise.update(m.risers)
            fall.update(m.fallers)
            pp = per_prompt[p]
            for w in m.risers:
                pp[w][0] += 1
            for w in m.fallers:
                pp[w][1] += 1
            for w, dv in m.delta.items():
                mass[w] += dv
                pp[w][2] += dv
        if (i + 1) % 10 == 0:
            print("  %d/%d pairs, %s cells, %.0fs"
                  % (i + 1, len(pairs), format(cells, ","), time.time() - t),
                  flush=True)
    json.dump({"riser": dict(rise), "faller": dict(fall),
               "signed_mass": {w: round(v, 8) for w, v in mass.items()},
               "per_prompt": {pr: {w: [v[0], v[1], round(v[2], 8)]
                                   for w, v in d.items() if v[0] or v[1]}
                              for pr, d in per_prompt.items()},
               "cells": cells,
               "absent": miss, "battery": os.path.relpath(BATTERY, ROOT),
               "pairs": os.path.relpath(PAIRS, ROOT), "rule": "CANONICAL"},
              open(OUT, "w"))
    print("\n%s cells, %s absent -> %s"
          % (format(cells, ","), format(miss, ","), os.path.relpath(OUT, ROOT)))


def show(min_events=300, k=20):
    d = json.load(open(OUT))
    R, F = d["riser"], d["faller"]
    rows = sorted(((R.get(w, 0) - F.get(w, 0), w, R.get(w, 0), F.get(w, 0))
                   for w in set(R) | set(F)
                   if R.get(w, 0) + F.get(w, 0) >= min_events), reverse=True)
    print("%s cells, %d distinct words with movement\n"
          % (format(d["cells"], ","), len(set(R) | set(F))))
    print("%-16s %8s %8s %9s" % ("word", "riser", "faller", "NET"))
    print("  ---- MOST RISER-Y ----")
    for net, w, r, f in rows[:k]:
        print("%-16s %8d %8d %+9d  %s"
              % (w, r, f, net, "FUNC" if w.lower() in FUNC else ""))
    print("  ---- MOST FALLER-Y ----")
    for net, w, r, f in rows[-k:]:
        print("%-16s %8d %8d %+9d  %s"
              % (w, r, f, net, "FUNC" if w.lower() in FUNC else ""))
    tr, tf = sum(R.values()), sum(F.values())
    fr = sum(v for w, v in R.items() if w.lower() in FUNC)
    ff = sum(v for w, v in F.items() if w.lower() in FUNC)
    print("\nWHAT vv* DISCARDS")
    print("  riser events  %s, function/pronoun %s (%.1f%%)"
          % (format(tr, ","), format(fr, ","), 100 * fr / tr))
    print("  faller events %s, function/pronoun %s (%.1f%%)"
          % (format(tf, ","), format(ff, ","), 100 * ff / tf))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--min-events", type=int, default=300)
    a = ap.parse_args()
    if a.show:
        show(a.min_events)
    else:
        build()
        show(a.min_events)
