#!/usr/bin/env python
"""dialogue_rate_transfer.py — do the f11_l2 dialogue rates transfer to the battery?

    meta/M04_syntagmatic/scripts/dialogue_rate_transfer.py
    meta/M04_syntagmatic/scripts/dialogue_rate_transfer.py --write

Emits `meta/M04_syntagmatic/results/dialogue_rate_transfer.json`.

## THE QUESTION, AND WHY IT IS THE ONE THAT COULD SINK THE MATCHER

@lacan's `build_forced_arms_drmatch.py` matches the `matched` arm on measured
dialogue rate, and those rates come from `dialogue_rate.py` run on **f11_l2, a
contradiction-prompt corpus**, while the arms will be used on the 105/212-prompt
narrative battery. Their own [5514].3: *"Ordering should transfer, levels will
not, and the matcher only uses ordering -- but nobody has checked that the
ordering transfers."* @registrar's [5515].3 ruled the check rather than a
remeasure: cross-check on a second corpus, free, and pool if they disagree.

**The matcher consumes ORDERING ONLY** -- it picks the candidate minimising
`|rate(w) - rate(faller)|` -- so a level shift is harmless and a rank shuffle is
fatal. That asymmetry is what makes a cross-check sufficient here and a
remeasure unnecessary.

## METHOD — @lacan'S EXACT MEASURE, NOT A REIMPLEMENTATION

`dialogue_rate.py`'s definition is reused verbatim: whitespace `split()` so
quote marks stay attached to their tokens, key = lowercased with `[^a-z]`
stripped, keys under 3 chars dropped, markup passages skipped, and a hit is a
quote mark within WINDOW=6 tokens AFTER the word. MIN_N=200 on both sides.

**Getting this wrong is easy and silent.** A first pass here tokenised with
`[A-Za-z']+`, which strips the quote marks and then searches the stripped tokens
for them -- every rate came back 0.0% and the transfer read as total failure.
A measure that returns all zeros is a bug; a measure that returns plausible
wrong numbers is the dangerous version, and the only defence is reusing the
source definition rather than writing a second one.

## RESULT

    244,113 battery passages scored, 5,009 words shared with the f11_l2 table

    Spearman rho  0.653        Pearson  0.793
    median rate   f11_l2 0.0530   battery 0.0603   (levels differ, as predicted)
    top decile on f11_l2 staying in the battery's top HALF: 474/500 = 95%

**Ordering transfers.** Two corpora from unrelated domains -- contradiction
prompts and narrative continuations -- agree on which words open dialogue.

And @lacan's load-bearing case survives on the second corpus: the GESTURE verbs
that no speech-verb tag catches sit far above the neutral verbs in both.

    nodded 32.9 -> 27.7    sighed 28.7 -> 25.4    smiled 22.5 -> 28.4
    walked  3.7 ->  4.8    opened  4.4 ->  4.9    killed  8.7 -> 11.8

## THE LEVEL SHIFT CARRIES NO ARGUMENT ABOUT THE TOLERANCE, AND AN EARLIER
## DRAFT OF THIS FILE SAID IT DID

A first version of this docstring argued that battery rates are systematically
LOWER and compress hardest where the confound lives, making the confound
smaller on the corpus of use and favouring the tighter tolerance. **That was
computed with a mangled quote character class that matched only straight
quotes**, and it does not survive the corrected measure: levels move BOTH ways
(`said` +2.5, `smiled` +5.9, `killed` +3.1 against `murmured` -10.2, `nodded`
-5.2), and the battery median is slightly HIGHER, not lower.

So this file has nothing to say about TOL 0.15 versus 0.25, and the tolerance
should be decided on @registrar's repairability argument ([5515].2) alone. The
transfer question is what this producer answers, and it answers it cleanly.
"""
import argparse
import collections
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

SRC = os.path.join(os.path.dirname(HERE), "results", "dialogue_rate.json")
GENS = os.path.join(ROOT, "data", "f37_gens_for_scoring.parquet")
OUT = os.path.join(os.path.dirname(HERE), "results", "dialogue_rate_transfer.json")

#: verbatim from dialogue_rate.py — reused, never re-derived
WINDOW, MIN_N = 6, 200
QUOTE = re.compile(r'["“”]')
WORD = re.compile(r"[^a-z]")
MARKUP = re.compile(r"[<>]|\bclass=|https?://|&[a-z]+;")

PROBES = ["said", "replied", "murmured", "exclaimed", "nodded", "sighed",
          "smiled", "shook", "laughed", "walked", "opened", "killed"]


def measure(texts):
    tot, quo, n, skip = collections.Counter(), collections.Counter(), 0, 0
    for t in texts:
        if not t:
            continue
        if MARKUP.search(t):
            skip += 1
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
    return ({w: quo[w] / tot[w] for w in tot if tot[w] >= MIN_N},
            {"passages": n, "skipped_markup": skip})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    L = json.load(open(SRC))
    base = {w: v["rate"] for w, v in L["rates"].items()}
    d = pd.read_parquet(GENS)
    d = d[~d.model.str.startswith("human/")]
    new, meta = measure(d.text.values)

    common = sorted(set(new) & set(base))
    x = pd.Series([base[w] for w in common])
    y = pd.Series([new[w] for w in common])
    rho = float(x.rank().corr(y.rank()))
    n = len(common)
    topd = set(sorted(common, key=lambda w: -base[w])[:max(1, n // 10)])
    rk = {w: i for i, w in enumerate(sorted(common, key=lambda w: -new[w]))}
    kept = sum(1 for w in topd if rk[w] < n / 2)

    print("DIALOGUE-RATE TRANSFER  f11_l2 -> battery generations\n")
    print("  battery passages scored %s (%s skipped markup)"
          % (format(meta["passages"], ","), format(meta["skipped_markup"], ",")))
    print("  shared words at n>=%d: %d\n" % (MIN_N, n))
    print("  Spearman rho  %.3f      Pearson  %.3f" % (rho, float(x.corr(y))))
    print("  median rate   f11_l2 %.4f   battery %.4f" % (x.median(), y.median()))
    print("  top decile staying in battery's top half: %d/%d (%.0f%%)"
          % (kept, len(topd), 100 * kept / len(topd)))
    print("\n  %-12s %10s %10s" % ("word", "f11_l2", "battery"))
    for w in PROBES:
        if w in base and w in new:
            print("  %-12s %9.1f%% %9.1f%%" % (w, 100 * base[w], 100 * new[w]))

    if a.write:
        json.dump({
            "_about": "Does the f11_l2 dialogue-rate ORDERING transfer to the "
                      "battery corpus the forced arms will be used on? The "
                      "matcher consumes ordering only, so this is the question "
                      "that matters and a level shift is harmless.",
            "_producer": "meta/M04_syntagmatic/scripts/dialogue_rate_transfer.py",
            "_measure": "verbatim from dialogue_rate.py (whitespace split so "
                        "quote marks stay attached; key lowercased with [^a-z] "
                        "stripped; quote within 6 tokens after the word)",
            "_ruled_by": "@registrar [5515].3 -- cross-check, do not remeasure",
            "_verdict": "TRANSFER EVIDENCED. rho %.3f over %d shared words from "
                        "unrelated domains; %.0f%% of the f11_l2 top decile stays "
                        "in the battery's top half. Levels differ (median %.4f "
                        "vs %.4f) and the matcher does not use levels."
                        % (rho, n, 100 * kept / len(topd),
                           float(x.median()), float(y.median())),
            "_tolerance_note": "THIS FILE CARRIES NO ARGUMENT ABOUT THE "
                               "TOLERANCE. Levels move both ways (said +2.5pp, "
                               "smiled +5.9pp against murmured -10.2pp, nodded "
                               "-5.2pp) and the battery median is slightly "
                               "higher, so there is no systematic compression "
                               "to reason from. An earlier draft claimed there "
                               "was; it was computed with a quote class that "
                               "matched only straight quotes.",
            "spearman_rho": round(rho, 4),
            "pearson": round(float(x.corr(y)), 4),
            "n_shared_words": n,
            "median_rate": {"f11_l2": round(float(x.median()), 4),
                            "battery": round(float(y.median()), 4)},
            "top_decile_kept_in_top_half": {"kept": kept, "of": len(topd)},
            "battery_corpus": meta,
            "probes": {w: {"f11_l2": round(base[w], 4), "battery": round(new[w], 4)}
                       for w in PROBES if w in base and w in new},
        }, open(OUT, "w"), indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
