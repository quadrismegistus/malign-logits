#!/usr/bin/env python
"""Does displacement move along a chain of CONTIGUITY, and how far?

RH, 2026-08-10: the campaign calls M01 "displacement", but `kill` -> `scream` is
substitution within a semantic field -- two responses to rage, related by
SIMILARITY. That is metaphor, and in the Lacanian mapping, condensation.
`bra` -> `shoes` is a different operation: those words are not similar, they are
adjacent in a BODILY SERIES. That is metonymy, and displacement in the strict
sense.

One instrument -- change in word probability at a slot -- picks up both, because
the axis distinction does not live in where the measurement is taken. **It lives
in the relation between the word that fell and the word that rose**, and twp
does not measure that.

THE RELATION IS ALREADY DECLARED IN THE Y DESIGN AND NOBODY HAD READ IT AS ONE.
The shard specs carry a `cls` per forced word, and the labels are not similarity
categories:

    explicit_1   GENITAL       DIGIT      EXTREMITY
    explicit_5   EROGENOUS     ADJACENT   DISTAL
    liminal_6/7  GENITAL_ZONE  EXTREMITY
    explicit_3   GENITAL       EUPHEMISM  GARMENT

The first three are ANATOMICAL DISTANCE -- a contiguity series, fixed before any
data. `explicit_3` is not: EUPHEMISM is a register alternative, i.e. the
metaphoric chain. So the corpus contains at least two chains and this script
keeps them apart rather than pooling them.

WHY explicit_5 IS THE CASE THAT DECIDES IT. It is the only prompt with an
INTERMEDIATE term. A monotone series (liminal_6/7: near vs far, nothing between)
cannot distinguish "moves away" from "moves one step". explicit_5 can, because
DISTAL sits beyond ADJACENT.

THE UNIT IS THE PAIR. Class medians are taken WITHIN a pair, then tested across
pairs. Sign counts are reported beside every median, because a large median with
a coin-flip sign count is a few pairs moving a lot -- which is what `legs` is.

    anatomical_series.py
    anatomical_series.py --prompt sexual_explicit_5 --words
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

#: Ordered near-to-far where the series is anatomical. `None` marks a prompt
#: whose classes are NOT a contiguity series and must not be read as one.
SERIES = {
    "sexual_explicit_1": ["GENITAL", "DIGIT", "EXTREMITY"],
    "sexual_explicit_5": ["EROGENOUS", "ADJACENT", "DISTAL"],
    "sexual_liminal_6": ["GENITAL_ZONE", "EXTREMITY"],
    "sexual_liminal_7": ["GENITAL_ZONE", "EXTREMITY"],
    "sexual_explicit_3": None,          # GENITAL / EUPHEMISM / GARMENT: register
}


def design():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "y_shard_*.json"))):
        for p in json.load(open(f)).get("prompts", []):
            out[p["prompt_id"]] = (p["prompt"],
                                   {c["word"]: c.get("cls") for c in p["cells"]
                                    if c.get("word")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--words", action="store_true", help="per-word detail")
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import wilcoxon
    from malign_logits.step import Step

    D = design()
    pairs = set()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*",
                                           "y__*.jsonl"))):
        for line in open(f):
            pairs.add(json.loads(line)["pair"])
            break
    pairs = sorted(pairs)

    pids = [a.prompt] if a.prompt else list(SERIES)
    for pid in pids:
        if pid not in D:
            continue
        ptext, cls = D[pid]
        ser = SERIES.get(pid)
        #: per pair, per class: the median Delta over that class's words
        per = defaultdict(dict)
        words = defaultdict(list)
        for pr in pairs:
            b, al = pr.split(">")
            try:
                c = Step(b, al).cell(ptext)
                if not c.is_present:
                    continue
                P, Q = c.pre.probs, c.post.probs
            except Exception:
                continue
            byc = defaultdict(list)
            for w, k in cls.items():
                if w in P or w in Q:
                    d = Q.get(w, 0.0) - P.get(w, 0.0)
                    byc[k].append(d)
                    words[(k, w)].append(d)
            for k, v in byc.items():
                per[pr][k] = float(np.median(v))

        print("\n%s  %r" % (pid, ptext))
        print("  %d pairs, series = %s"
              % (len(per), " -> ".join(ser) if ser else "NOT ANATOMICAL (register)"))
        if a.words:
            for (k, w), v in sorted(words.items()):
                neg = sum(1 for x in v if x < 0)
                print("    %-14s %-11s median %+9.5f   %2d of %2d NEGATIVE"
                      % (k, w, np.median(v), neg, len(v)))
        order = ser or sorted({k for d in per.values() for k in d})
        print("  %-16s %11s %14s" % ("class", "median", "pairs negative"))
        for k in order:
            v = [d[k] for d in per.values() if k in d]
            if not v:
                continue
            print("  %-16s %+11.5f %8d of %-4d"
                  % (k, np.median(v), sum(1 for x in v if x < 0), len(v)))

        #: THE NON-MONOTONICITY TEST, only where a middle term exists. A series
        #: with two levels cannot separate "moves away" from "moves one step",
        #: so the test is not run there and says so rather than returning a
        #: number that reads as evidence.
        if ser and len(ser) >= 3:
            near, mid, far = ser[0], ser[1], ser[-1]
            print("  ONE-STEP TEST (needs the middle term; %s has one)" % pid)
            for lab, x, y in ((("%s - %s" % (mid, near)), mid, near),
                              (("%s - %s" % (mid, far)), mid, far),
                              (("%s - %s" % (far, near)), far, near)):
                d = [per[p][x] - per[p][y] for p in per if x in per[p] and y in per[p]]
                if len(d) < 8:
                    continue
                print("    %-24s median %+9.5f  %2d of %2d positive  p=%.4g"
                      % (lab, np.median(d), sum(1 for t in d if t > 0), len(d),
                         wilcoxon(d).pvalue))
            print("    Displacement-with-a-distance predicts the first two POSITIVE")
            print("    and the third near zero: both ends fall, the middle does not.")
        elif ser:
            print("  ONE-STEP TEST NOT RUN: %s has %d levels and no middle term,"
                  % (pid, len(ser)))
            print("    so it cannot distinguish moving away from moving one step.")


if __name__ == "__main__":
    main()
