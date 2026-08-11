#!/usr/bin/env python
"""Which words open quoted dialogue, measured rather than listed.

    uv run python dialogue_rate.py            # write the table
    uv run python dialogue_rate.py --show     # read it back

WHY THIS EXISTS. The passage-corpus arms are lexically imbalanced: the faller
is a speech verb ~15.5% of the time against the matched control's ~7.5%, and
42% of matched pairs disagree on lexical class WITHIN the cell. That matters
because the instrument measures the surprisal of what FOLLOWS a forced word,
and a word that opens quoted dialogue puts the continuation in a different
generative regime. Forcing `said` and forcing `found` differ in far more than
movement history.

The obvious fix is to make speech verbs ineligible. **I wrote such a list from
memory and it was half wrong**, which is why this file exists instead:

    measured top inducers my list MISSED
        nodded 34.3%   sighed 26.0%   smiled 23.2%   shook 22.8%   laughed 20.7%

    on my list, not ranked by the data
        begged, exclaimed, murmured, muttered, warned, yelled

**`nodded` induces dialogue more often than `answered` and is not a speech verb
at all.** Nor are `sighed`, `smiled`, `shook`, `laughed` -- they are the gesture
verbs of speech tags (*she nodded. "Yes."*). No part-of-speech tag catches them
and no introspection reliably does either. A closed list written by the analyst
is a researcher degree of freedom wearing a registration's clothes.

THE MEASURE. For each word, the probability that a quotation mark appears
within `WINDOW` tokens after it, over generated continuations in a declared
corpus. It is tied to the mechanism at issue rather than to a word class, it is
computable before any new data exists, and anyone can recompute it.

WHAT IT IS NOT. It is not a claim about which words ARE speech verbs, and the
threshold is deliberately not set here -- a gate belongs to the analysis that
declares it, not to the producer ([5233]/[5236]). This writes a COLUMN.

CAVEAT WORTH ITS LINE: the rate is measured on `f11_l2`, a contradiction-prompt
corpus of literary continuations. A word's dialogue rate is a property of word
AND genre. Applying it to the 105-stem transgressive battery assumes the
ordering transfers; the levels certainly do not.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
OUT = os.path.join(CAMP, "results", "dialogue_rate.json")
CH = "/opt/homebrew/bin/clickhouse"

CORPUS = "f11_l2"
WINDOW = 6            #: tokens after the word in which a quote mark counts
MIN_N = 200           #: below this the rate is not estimated
QUOTE = re.compile(r'["“”]')
WORD = re.compile(r"[^a-z]")

#: THREE REGIMES PUT QUOTE MARKS IN TEXT AND ONLY ONE IS DIALOGUE. Running the
#: first version on the full 228,520 passages instead of a 60k sample put HTML
#: at the top -- `typebutton` 99.6%, `img` 76.7%, `div` 73.2%, `span` 71.7%,
#: all of them `class="..."` attributes -- and metalinguistic quotation just
#: below it (`phrase` 49.4%, `verb` 32.0%, `hypothesis` 32.5%). Neither is a
#: speech tag. The sample had hidden it by being a cleaner subset, which is the
#: inverse of the usual failure and no less misleading.
MARKUP = re.compile(r"[<>]|\bclass=|https?://|&[a-z]+;")


def sweep():
    q = ("SELECT text FROM malign_logits.gen_sequences WHERE corpus='%s' "
         "FORMAT JSONEachRow" % CORPUS)
    pr = subprocess.Popen([CH, "client", "-q", q], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 20)
    tot, quo = collections.Counter(), collections.Counter()
    n = 0
    skipped = [0]
    for line in pr.stdout:
        try:
            t = json.loads(line)["text"] or ""
        except Exception:
            continue
        if MARKUP.search(t):
            skipped[0] += 1
            continue
        n += 1
        w = t.split()
        for i, x in enumerate(w):
            k = WORD.sub("", x.lower())
            if len(k) < 3:
                continue
            tot[k] += 1
            if QUOTE.search(" ".join(w[i + 1:i + 1 + WINDOW])):
                quo[k] += 1
        if n % 50000 == 0:
            print("  ... %s passages" % format(n, ","), flush=True)
    pr.wait()
    rows = {w: {"n": tot[w], "quoted": quo[w], "rate": quo[w] / tot[w]}
            for w in tot if tot[w] >= MIN_N}
    meta = {"corpus": CORPUS, "window_tokens": WINDOW, "min_n": MIN_N,
            "passages": n, "skipped_markup": skipped[0], "words": len(rows),
            "measure": "P(quote mark within WINDOW tokens after the word)",
            "note": "a COLUMN, not a gate; no threshold is set here"}
    json.dump({"_meta": meta, "rates": rows}, open(OUT, "w"), indent=1)
    print("swept %s passages (%s skipped as markup) -> %d words -> %s"
          % (format(n, ","), format(skipped[0], ","), len(rows),
             os.path.relpath(OUT, ROOT)))


def show():
    d = json.load(open(OUT))
    r = sorted(d["rates"].items(), key=lambda kv: -kv[1]["rate"])
    m = d["_meta"]
    print("%s, %s passages, window %d tokens, %d words with n>=%d\n"
          % (m["corpus"], format(m["passages"], ","), m["window_tokens"],
             m["words"], m["min_n"]))
    print("  TOP 25 DIALOGUE INDUCERS")
    for w, v in r[:25]:
        print("    %-16s %5.1f%%  n=%s" % (w, 100 * v["rate"], format(v["n"], ",")))
    print("\n  MEDIAN RATE %.1f%%" % (100 * r[len(r) // 2][1]["rate"]))
    print("\n  the arms of the passage corpus, if the table is present:")
    p = os.path.join(ROOT, "data", "forced_arms_105_v4.json")
    if not os.path.exists(p):
        print("    (forced_arms_105_v4.json not found)")
        return
    cells = json.load(open(p))["cells"]
    rate = {w: v["rate"] for w, v in d["rates"].items()}
    for arm in ("faller", "matched", "riser_matched", "riser"):
        vals = [rate[(c.get(arm) or "").lower()] for c in cells
                if rate.get((c.get(arm) or "").lower()) is not None]
        if not vals:
            continue
        vals.sort()
        print("    %-14s n=%5d  median %5.1f%%  mean %5.1f%%  share above 20%%: %4.1f%%"
              % (arm, len(vals), 100 * vals[len(vals) // 2],
                 100 * sum(vals) / len(vals),
                 100 * sum(1 for v in vals if v > 0.20) / len(vals)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.show or os.path.exists(OUT) and not a.show:
        pass
    if a.show:
        show()
    else:
        sweep()
        show()
