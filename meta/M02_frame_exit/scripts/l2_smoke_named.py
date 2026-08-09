#!/usr/bin/env python
"""Iterate on `tension_named` against ten passages with known answers.

WHY THIS EXISTS. On the paired 500 the field fired 48 times and roughly 40% of
those spans named a contradiction. The rest named a mood -- 'he was all
confused', 'feel bad about what she was doing' -- and one was the bare
connective 'At the same time'. The field was behaving as a distress detector.

FORCE=TRUE ON EVERY CALL, AND THAT IS NOT AN OPTIMISATION DETAIL. `prompt_version`
belongs to SequentialTask, not Task, so this task's cache key does NOT include
the system prompt or the field descriptions. The framework re-validates a cached
response and recomputes only if it no longer PARSES -- which a reworded field
description never triggers. Without force, every iteration of this loop would
return the previous iteration's answer and the tuning would look instantaneous
and be imaginary.

    l2_smoke_named.py                 # score the ten
    l2_smoke_named.py --verbose       # show the passage and the coder's span

THE SET IS A DEVELOPMENT SET AND STOPS MEASURING THE FIELD THE MOMENT THE FIELD
IS TUNED AGAINST IT. Ten items, five of each class, arms balanced within class
so tuning cannot teach an arm. A score here is a debugging signal, never a
precision estimate; the estimate needs spans this set has never seen.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

from malign_logits.tasks.code_m02_l2_treatment_v1 import (  # noqa: E402
    TreatmentV1Task, code)

FIXTURE = os.path.join(ROOT, "meta", "M02_frame_exit", "fixtures",
                       "tension_named_smoke10.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cached", action="store_true",
                    help="do NOT force; only valid when nothing in the task changed")
    args = ap.parse_args()

    doc = json.load(open(FIXTURE))
    items = doc["items"]
    task = TreatmentV1Task()

    print("tension_named smoke: %d items, %d YES / %d NO"
          % (len(items), sum(i["expected"] == "YES" for i in items),
             sum(i["expected"] == "NO" for i in items)))
    print("model %s, force=%s\n" % (args.model, not args.cached))

    hits = {"YES": [0, 0], "NO": [0, 0]}
    misses = []
    for n, it in enumerate(items, 1):
        try:
            r = code(task, it["pole_a"], it["pole_b"], it["prompt"],
                     it["continuation"], model=args.model,
                     force=not args.cached)
            d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
            got, span = d["tension_named"], d["named_span"]
        except Exception as exc:
            got, span = "ERROR", str(exc).split("\n")[0][:60]
        ok = got == it["expected"]
        hits[it["expected"]][1] += 1
        hits[it["expected"]][0] += ok
        mark = "ok " if ok else "XX "
        print("  %s%2d want %-3s got %-5s  %-8s %s"
              % (mark, n, it["expected"], got, it["arm"], it["why"]))
        if not ok:
            misses.append((n, it, got, span))
        if args.verbose or not ok:
            print("        prompt: %s" % it["prompt"])
            print("        text:   %s" % it["continuation"][:150])
            print("        was:    %r" % it["coder_span"][:80])
            if span:
                print("        now:    %r" % span[:80])

    #: Report the two classes SEPARATELY. A single accuracy number hides the only
    #: failure mode that matters here: a field tightened until it says NO to
    #: everything scores 5/10 and looks like partial progress.
    print("\n  recall  (YES kept)     %d of %d" % tuple(hits["YES"]))
    print("  specificity (NO rejected) %d of %d" % tuple(hits["NO"]))
    print("  total %d of %d" % (hits["YES"][0] + hits["NO"][0], len(items)))
    if hits["YES"][0] == 0:
        print("\n  WARNING: zero recall. A field that never fires passes the NO half")
        print("  for free and is not an improvement.")


if __name__ == "__main__":
    main()
