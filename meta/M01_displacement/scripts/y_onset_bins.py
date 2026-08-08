#!/usr/bin/env python
"""Where does the model break off? Onset position, binned.

    python y_onset_bins.py

Reads `results/y_pilot_coded.jsonl` and locates each coder's `refusal_onset`
quote inside the continuation it was coding, then bins the word position.

## WHY THE QUOTE AND NOT A POSITION FROM THE CODER

Models are unreliable at "which word number is this" and reliable at "copy
these words", so the coder copies and this computes. Measured on the pilot
before the field existed, using the general-purpose `evidence` field: 85% of
quotes located as exact substrings, 87% with fuzzy, 0% gave no quote.

## WHAT A POSITION DISTRIBUTION CAN AND CANNOT SETTLE

**A prompt-level filter fires at position 0, every time.** So mass away from
zero rules out "the classifier read the prompt and refused". That is the claim
this plot can support.

**It cannot rule out a CONTENT threshold** -- a trigger that fires once
accumulated explicitness crosses a line is still a mechanism, and it would also
sit away from zero. Distinguishing that needs the position of the onset against
the position of the explicit material, which is a different measurement and is
NOT done here. Named so the plot is not read as more than it is.

## THE COMPARISON THAT MATTERS IS THE BASE ARM

Base models also emit "I'm sorry" and "I can't" -- as dialogue. If the two arms
put their onsets in the same place, this is one phenomenon at two rates. If the
aligned arm is early and the base arm is late and diffuse, they are different
objects. The base arm is the null here and it is printed beside every table.
"""
import collections
import difflib
import json
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

QUOTE = re.compile(r"['\"‘’“”]([^'\"‘’“”]{6,})['\"‘’“”]")
BINS = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 10 ** 6)]
LABELS = ["0", "1-5", "6-10", "11-20", "21-40", "41-80", "81+"]


def norm(s):
    return " ".join((s or "").lower().split())


def locate(quote, text):
    """word index of `quote` in `text`, or None. Exact first, then fuzzy."""
    q, nt = norm(quote), norm(text)
    if not q or not nt:
        return None
    if q in nt:
        return len(nt[:nt.index(q)].split())
    sm = difflib.SequenceMatcher(None, nt, q).find_longest_match(0, len(nt), 0, len(q))
    #: fuzzy only when the matched run is most of the quote. A short accidental
    #: overlap would place an onset anywhere and it would look like data.
    if sm.size >= max(12, 0.6 * len(q)):
        return len(nt[:sm.a].split())
    return None


def main():
    import y_pilot_coder as Y
    rng = random.Random(Y.SEED)
    G = Y.load()
    texts, metas = Y.build_items(G, 10, rng)
    k2t = {}
    for t, m in zip(texts, metas):
        k2t[(m["pair"], m["role"], m["word"], m["seq_i"])] = \
            t.split('CONTINUATION: "', 1)[1].rsplit('"', 1)[0]

    path = os.path.join(CAMP, "results", "y_pilot_coded.jsonl")
    rows = [json.loads(l) for l in open(path)]
    has_field = any("refusal_onset" in d for d in rows)
    print("rows %d | refusal_onset present: %s" % (len(rows), has_field))
    if not has_field:
        print("  the file predates the field -- rerun y_pilot_coder.py first")
        return 1

    #: LEDGER. Every annotation that claimed an onset is accounted for as
    #: located, unlocatable, or empty. A distribution built from the located
    #: ones alone would silently be about a subset.
    led = collections.Counter()
    obs = []
    for d in rows:
        k = (d["pair"], d["role"], d["word"], d["seq_i"])
        if k not in k2t:
            led["no text"] += 1
            continue
        q = (d.get("refusal_onset") or "").strip()
        if not q:
            led["empty (no departure)"] += 1
            continue
        pos = locate(q, k2t[k])
        if pos is None:
            led["QUOTE NOT IN TEXT"] += 1
            continue
        led["located"] += 1
        obs.append({"pos": pos, "role": d["role"], "cls": d["cls"],
                    "word": d["word"], "pair": d["pair"], "coder": d["coder"],
                    "refusal": d.get("assistant_refusal") == "YES",
                    "exit": d.get("frame_exit") == "YES",
                    "n_words": len(norm(k2t[k]).split())})
    tot = sum(led.values())
    print("\nLEDGER over %d annotations" % tot)
    for k2, v in led.most_common():
        print("   %-24s %5d  %5.1f%%" % (k2, v, 100 * v / tot))

    def hist(sel, label):
        if not sel:
            print("  %-22s (none)" % label)
            return
        c = collections.Counter()
        for o in sel:
            for (lo, hi), lab in zip(BINS, LABELS):
                if lo <= o["pos"] <= hi:
                    c[lab] += 1
                    break
        n = len(sel)
        ps = [o["pos"] for o in sel]
        bar = "  ".join("%s:%d" % (l, c.get(l, 0)) for l in LABELS)
        print("  %-22s n=%-4d median %3d  mean %5.1f  max %3d" %
              (label, n, statistics.median(ps), statistics.mean(ps), max(ps)))
        print("      %s" % bar)
        print("      " + " ".join("%-6s" % l for l in LABELS))
        print("      " + " ".join("%-6s" % ("%.0f%%" % (100 * c.get(l, 0) / n)) for l in LABELS))

    print("\n" + "=" * 88)
    print("ONSET POSITION IN WORDS, all located departures")
    for role in ("base", "aligned"):
        hist([o for o in obs if o["role"] == role], role.upper())

    print("\nRESTRICTED TO CODER-CONFIRMED ASSISTANT REFUSALS")
    for role in ("base", "aligned"):
        hist([o for o in obs if o["role"] == role and o["refusal"]], role.upper())

    print("\nFRAME EXIT WITHOUT REFUSAL  (the base arm's ordinary behaviour)")
    for role in ("base", "aligned"):
        hist([o for o in obs if o["role"] == role and o["exit"] and not o["refusal"]],
             role.upper())

    print("\nALIGNED REFUSALS BY FORCED WORD")
    for w in ("cock", "penis", "fingers", "thumb", "toes", "-"):
        hist([o for o in obs if o["role"] == "aligned" and o["refusal"] and o["word"] == w], w)

    #: the position is bounded by the length of the continuation, so a late
    #: onset is partly a fact about how long the text is. Reported as a
    #: fraction too, which is scale-free.
    print("\nAS A FRACTION OF THE CONTINUATION (position / total words)")
    for role in ("base", "aligned"):
        f = [o["pos"] / o["n_words"] for o in obs
             if o["role"] == role and o["refusal"] and o["n_words"]]
        if f:
            print("  %-8s n=%-3d median %.2f  mean %.2f" %
                  (role, len(f), statistics.median(f), statistics.mean(f)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
