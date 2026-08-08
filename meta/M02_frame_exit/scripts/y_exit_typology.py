#!/usr/bin/env python
"""Type the Y confirmatory exits with M02's typology, before M02's battery runs.

    python y_exit_typology.py

WHY THIS FILE EXISTS. `plan_m02_battery_foreclosure.md` records the expectation
that alignment's contribution is in the TYPE of frame exit, not its rate, and
names the alternative outcome as equally informative. Y's confirmatory
annotation already carries typed layer-1 regions on ~30k passages, so the
typology can be applied to it now rather than waiting.

WHAT Y'S TAGS CANNOT DO ALONE: `<meta>` pools E-QUIZ, E-QA, E-TASK, E-MENTION
and E-META into one region kind, and `<web>` catches E-ASSIST when a leaked
system prompt reads as boilerplate. The layer-1 result -- story share flat,
web down ~4pts, meta up ~6.5pts -- is therefore a statement about pooled exit
kinds and cannot say which type moved. This types the text inside those
regions.

REGEX TYPING, AND ITS LIMIT STATED UP FRONT. These are surface patterns over
the coder's non-story regions, not a second annotation pass. They will miss
paraphrased exits and will fire on in-scene dialogue that happens to ask a
question. The direction of that error is unknown rather than conservative, so
this is a first look that should be replaced by coded types, never a
measurement to quote.
"""
import collections
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
from malign_logits.tasks.code_y_superego_v3 import spans  # noqa: E402

IN = os.path.join(CAMP, "M01_displacement", "results", "y_confirmatory_coded.jsonl")

#: M02's typology, in the plan's own order. REFUSAL is deliberately absent:
#: the plan requires it stay a separate field, because the Y dissociation
#: (refusal is not frame exit) has to remain visible.
TYPES = [
    ("E-QUIZ",    re.compile(r"_{3,}|\([A-D]\)\s|\b[A-D]\.\s+\w|possible answers|"
                             r"multiple choice|fill in the blank|choose the correct", re.I)),
    ("E-QA",      re.compile(r"\bQ\s*:|\bA\s*:|\bquestion\s*:|\banswer\s*:|答案", re.I)),
    ("E-TASK",    re.compile(r"\btranslate\b|\brewrite\b|\bsummari[sz]e\b|your task is|"
                             r"\bexercise\b|\bcomplete the\b|\bconvert the\b", re.I)),
    ("E-ASSIST",  re.compile(r"you are an? (?:ai|helpful|language)|as an ai|"
                             r"i(?:'m| am) an ai|i cannot (?:assist|help|provide)|"
                             r"it(?:'s| is) important to (?:approach|note|remember)|"
                             r"raises (?:some )?(?:significant )?concerns", re.I)),
    ("E-MENTION", re.compile(r"words like|the (?:word|term|phrase) [\"'“]|"
                             r"\bmeans\b.{0,20}\bdefinition|refer(?:s|ring) to in the", re.I)),
    ("E-META",    re.compile(r"content warning|trigger warning|this (?:excerpt|passage|text) is|"
                             r"\bgenre\b|editor(?:'s|ial) note|\bexcerpt from\b", re.I)),
]


def sign(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def nonstory(r):
    """The text OUTSIDE <story>, which is where an exit lives by definition."""
    txt, cov = spans(r.get("tagged") or "")
    story = cov.get("story") or set()
    keep = [i for i in range(len(txt)) if i not in story]
    if not keep:
        return ""
    out, s, p = [], keep[0], keep[0]
    for i in keep[1:]:
        if i != p + 1:
            out.append(txt[s:p + 1]); s = i
        p = i
    out.append(txt[s:p + 1])
    return " ".join(out)


def main():
    rows = [json.loads(l) for l in open(IN)]
    ok = [r for r in rows if r.get("parsed") and r.get("pass") == "A"]
    print("pass A parsed %s   pairs %d\n" % (format(len(ok), ","), len({r["pair"] for r in ok})))

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    anyexit = collections.defaultdict(list)
    for r in ok:
        ns = nonstory(r)
        k = (r["pair"], r["role"])
        per["ANY EXIT (non-story present)"][k].append(1.0 if ns.strip() else 0.0)
        hit = False
        for name, pat in TYPES:
            v = 1.0 if (ns and pat.search(ns)) else 0.0
            per[name][k].append(v)
            hit = hit or bool(v)
        per["exit, UNTYPED by these patterns"][k].append(
            1.0 if (ns.strip() and not hit) else 0.0)
        #: refusal kept separate, per the plan
        per["REFUSAL (not an exit)"][k].append(
            1.0 if "<refusal>" in (r.get("tagged") or "") else 0.0)

    print("EXIT TYPE, within pair, aligned minus base")
    print("  %-32s %8s %8s %9s %8s" % ("type", "base", "aligned", "pairs+", "p"))
    print("  " + "-" * 70)
    order = ["ANY EXIT (non-story present)"] + [t for t, _ in TYPES] + \
            ["exit, UNTYPED by these patterns", "REFUSAL (not an exit)"]
    for name in order:
        v = per[name]
        d, B, A = [], [], []
        for p in {x[0] for x in v}:
            b, a = v.get((p, "base")), v.get((p, "aligned"))
            if not b or not a:
                continue
            mb, ma = statistics.mean(b), statistics.mean(a)
            d.append(ma - mb); B.append(mb); A.append(ma)
        if not d:
            continue
        pos = sum(1 for x in d if x > 0); n = len(d)
        pv = sign(max(pos, n - pos), n)
        star = " ***" if pv < 0.01 else (" *" if pv < 0.05 else "")
        print("  %-32s %7.1f%% %7.1f%% %9s %8.4f%s"
              % (name, 100 * statistics.mean(B), 100 * statistics.mean(A),
                 "%d/%d" % (pos, n), pv, star))
    return 0


if __name__ == "__main__":
    sys.exit(main())
