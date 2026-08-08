#!/usr/bin/env python
"""Does the coder still work at CONFIRMATORY passage length? Gate before spend.

    python y_length_gate.py --n 200

Everything v3 has been validated on is the pilot corpus: 423 chars at the
median. The confirmatory corpus is 1,035 -- 2.4x -- and the one field that
scales with length is `tagged`, which must reproduce the passage CHARACTER FOR
CHARACTER. That is the property most likely to break with length and the one
the schema deliberately does not enforce, so nothing downstream would raise if
it degraded: a drifted reproduction still parses, still carries every field,
and only the span offsets are quietly wrong.

MEASURED HERE, ALL THREE ON THE SAME ITEMS:

    parse rate        does it come back at all
    ROUND TRIP        stripping tags reproduces the source exactly
    tokens per item   the cost extrapolation is 847 and has never been observed

A 42.7M-token programme rests on the third number being right, and on the
second not collapsing. Both are cheap to check and neither is checkable after
the fact -- a corpus annotated with drifted spans looks exactly like a corpus
annotated correctly until someone reads a span.
"""
import argparse
import collections
import difflib
import glob
import json
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

SEED = 20260808
#: VOCABULARY ONLY. The permissive form stripped HTML the coder had faithfully
#: reproduced -- <br>, <p>, <strong>, <URL> -- from one side of the comparison
#: and not the other, making 11 of 40 SEVERE round-trips an artefact.
from malign_logits.tasks.code_y_superego_v3 import strip_tags as _strip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    #: SAMPLED FROM THE POPULATION THAT WOULD ACTUALLY BE ANNOTATED -- >=256
    #: tokens, cross-scored, bloomz out. A gate run on a convenience sample of
    #: whatever parses first would answer a question nobody asked.
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl")))
             if "FAILED" not in f]
    pool = []
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "sequences" not in r:
                continue
            if "bloomz" in (r.get("pair") or "") or r.get("cross_score_blocked"):
                continue
            for s in r["sequences"]:
                if len(s.get("tokens") or []) >= 256:
                    pool.append((r.get("prompt_id"), r.get("word"), s.get("text") or "",
                                 r.get("model"), r.get("role")))
    rng = random.Random(SEED)
    pick = rng.sample(pool, min(a.n, len(pool)))
    lens = [len(t) for _, _, t, _, _ in pick]
    print("gate sample: %d of %d eligible   passage chars: median %d  mean %.0f  max %d"
          % (len(pick), len(pool), statistics.median(lens), statistics.mean(lens), max(lens)))

    from malign_logits.tasks.code_y_superego_v3 import prepare, SuperegoV3Task
    PROMPTS = json.load(open(os.path.join(CAMP, "registrations",
                                          "registration_y_slots.json")))
    stem = {}
    for p in PROMPTS["prompts"]:
        stem[p["prompt_id"]] = p.get("text") or p.get("prompt") or ""

    items = [prepare(stem.get(pid, ""), w, txt) for pid, w, txt, _, _ in pick]
    task = SuperegoV3Task()
    errors, per_item = {}, {}
    res = task.map(items, model=a.model, num_workers=a.workers,
                   errors=errors, per_item_usage=per_item)

    ok = [r for r in res if r is not None]
    print("\nparsed %d/%d   errors %d" % (len(ok), len(res), len(errors)))
    try:
        print("usage: %s" % task.usage.summary_line())
    except Exception:
        pass

    #: ROUND TRIP, the check the schema does not make.
    exact = 0
    ratios, drift = [], []
    for (pid, w, src, mdl, role), r in zip(pick, res):
        if r is None:
            continue
        plain = _strip(r.tagged or "")
        if plain == src:
            exact += 1
            ratios.append(1.0)
            continue
        ratio = difflib.SequenceMatcher(None, src, plain).ratio()
        ratios.append(ratio)
        drift.append((ratio, len(src), len(plain), mdl, src, plain))
    print("\nROUND TRIP over %d parsed" % len(ok))
    print("  exact               %d/%d = %.1f%%" % (exact, len(ok), 100 * exact / max(1, len(ok))))
    if ratios:
        print("  similarity          median %.4f   p05 %.4f   min %.4f"
              % (statistics.median(ratios), sorted(ratios)[int(.05 * len(ratios))], min(ratios)))
    print("  inexact             %d" % len(drift))
    for ratio, ls, lp, mdl, src, plain in sorted(drift)[:3]:
        print("     %.4f  %5d -> %5d chars  %s" % (ratio, ls, lp, (mdl or "?").split("/")[-1][:26]))
        sm = difflib.SequenceMatcher(None, src, plain)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                print("        %-8s src %r -> got %r" % (op, src[i1:i2][:60], plain[j1:j2][:60]))
                break

    #: THE COST NUMBER, OBSERVED RATHER THAN EXTRAPOLATED.
    outs = [u.get("output") or u.get("output_tokens") or 0
            for u in per_item.values()] if per_item else []
    if outs:
        m = statistics.mean(outs)
        print("\nOUTPUT TOKENS PER ITEM: mean %.0f  median %.0f  p95 %.0f"
              % (m, statistics.median(outs), sorted(outs)[int(.95 * len(outs))]))
        print("  extrapolation in the plan was 847")
        for label, n in (("pass A  n=20 balanced", 42002), ("pass B  census", 20679)):
            print("    %-24s %6d items -> %5.1fM at observed (%.1fM at 847)"
                  % (label, n, n * m / 1e6, n * 847 / 1e6))
        print("    %-24s %6s    %5.1fM total at observed"
              % ("", "", (42002 * m + 20679 * m * 0.41) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
