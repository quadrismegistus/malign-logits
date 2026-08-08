#!/usr/bin/env python
"""Layer-1 SEQUENCE: what shape does the passage take, not what is in it.

    python y_sequence.py
    python y_sequence.py --pass B

Every other Y statistic asks "how much X". This asks "in what ORDER", which is
a different question and the only one that can see a RESUMPTION -- story,
interruption, story again. A rate cannot represent that: the passage that
alternates four times and the passage with one long break have the same
<noise> coverage.

MEASURES, all per passage, contrasted within pair:

    shape         the collapsed layer-1 sequence with repeats kept
                  (story|noise|story is a different shape from story|noise)
    n_regions     how fragmented
    n_switches    transitions between region kinds
    resumes       story ... non-story ... story: the model came BACK
    opens_story   does the passage begin in the fiction
    ends_story    does it end there
    terminal      what kind of region the passage ends in -- where the text
                  LANDS, which for a frame-exit account is the outcome variable

WHY `terminal` MATTERS SEPARATELY FROM COVERAGE. `<meta>` rising and `<web>`
falling is a statement about how much of the passage each occupies. Whether the
passage ENDS in meta is a statement about whether it ever came back, and a
model that exits and returns is doing something different from one that exits
and stays.
"""
import argparse
import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
sys.path.insert(0, HERE)
from malign_logits.tasks.code_y_superego_v3 import spans, LAYER1  # noqa: E402
from y_paired_tests import wilcoxon, boot_ci  # noqa: E402

IN = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")


def l1_sequence(tagged):
    """Opening order of layer-1 regions. Adjacent duplicates collapsed, because
    `<story><story>` is a tagging artefact and not two episodes."""
    import re
    seq = []
    for m in re.finditer(r"<(/?)(%s)>" % "|".join(LAYER1), tagged or ""):
        if not m.group(1):
            if not seq or seq[-1] != m.group(2):
                seq.append(m.group(2))
    return seq


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pas", default="A", choices=("A", "B", "all"))
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(IN)]
    ok = [r for r in rows if r.get("parsed")]
    if a.pas != "all":
        ok = [r for r in ok if r.get("pass") == a.pas]
    print("rows %s   pairs %d   pass %s\n"
          % (format(len(ok), ","), len({r["pair"] for r in ok}), a.pas))

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    shape = collections.defaultdict(collections.Counter)
    for r in ok:
        seq = l1_sequence(r.get("tagged") or "")
        if not seq:
            continue
        k = (r["pair"], r["role"])
        ns = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        res = any(seq[i] == "story" and "story" in seq[i + 1:]
                  and any(z != "story" for z in seq[i + 1:seq[i + 1:].index("story") + i + 1])
                  for i in range(len(seq) - 1) if seq[i] == "story" and "story" in seq[i + 1:])
        per["n_regions"][k].append(float(len(seq)))
        per["n_switches"][k].append(float(ns))
        per["resumes (story..X..story)"][k].append(1.0 if res else 0.0)
        per["opens in story"][k].append(1.0 if seq[0] == "story" else 0.0)
        per["ends in story"][k].append(1.0 if seq[-1] == "story" else 0.0)
        per["pure story (no exit)"][k].append(1.0 if set(seq) == {"story"} else 0.0)
        per["never enters story"][k].append(1.0 if "story" not in seq else 0.0)
        for t in LAYER1:
            per["terminal=%s" % t][k].append(1.0 if seq[-1] == t else 0.0)
        shape[r["role"]]["|".join(seq[:4])] += 1

    print("SHAPE FREQUENCY (first four regions), pooled -- descriptive")
    print("  %-34s %9s %9s %8s" % ("shape", "base", "aligned", "ratio"))
    print("  " + "-" * 64)
    tb, ta = sum(shape["base"].values()), sum(shape["aligned"].values())
    allsh = set(shape["base"]) | set(shape["aligned"])
    for sh in sorted(allsh, key=lambda s: -(shape["base"][s] + shape["aligned"][s]))[:a.top]:
        b, x = shape["base"][sh], shape["aligned"][sh]
        rb, rx = 100 * b / max(1, tb), 100 * x / max(1, ta)
        print("  %-34s %8.2f%% %8.2f%% %7s" % (sh[:34], rb, rx,
              ("%.2fx" % (rx / rb)) if rb else "-"))

    print("\nSEQUENCE MEASURES, within pair, aligned minus base")
    print("  %-30s %8s %8s %8s %8s %18s" % ("measure", "base", "algn", "med d", "WILCOX", "boot 95% CI"))
    print("  " + "-" * 84)
    out = []
    for nm, v in per.items():
        d, B, A = [], [], []
        for p in {x[0] for x in v}:
            b, x = v.get((p, "base")), v.get((p, "aligned"))
            if not b or not x or len(b) < 20 or len(x) < 20:
                continue
            mb, ma = statistics.mean(b), statistics.mean(x)
            d.append(ma - mb); B.append(mb); A.append(ma)
        if len(d) < 10:
            continue
        wp, _ = wilcoxon(d)
        lo, hi = boot_ci(d)
        out.append((wp, nm, statistics.mean(B), statistics.mean(A),
                    statistics.median(d), lo, hi, len(d)))
    out.sort()
    for wp, nm, mb, ma, md, lo, hi, n in out:
        claim = " <=" if (lo > 0 or hi < 0) else ""
        unit = "" if nm.startswith("n_") else "%"
        sc = 1 if nm.startswith("n_") else 100
        print("  %-30s %7.3f%s %7.3f%s %+8.3f %8.4f  [%+6.3f,%+6.3f]%s"
              % (nm, sc * mb, unit, sc * ma, unit, sc * md, wp, sc * lo, sc * hi, claim))
    return 0


if __name__ == "__main__":
    sys.exit(main())
