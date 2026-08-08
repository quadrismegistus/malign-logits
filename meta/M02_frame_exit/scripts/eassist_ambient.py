#!/usr/bin/env python
"""E-ASSIST-AMBIENT: do aligned checkpoints emit assistant control tokens into
RAW continuation, unbidden?

    python eassist_ambient.py                      # declared roster, n>=200
    python eassist_ambient.py --min-n 50           # registrar's cut
    python eassist_ambient.py --csv out.csv

THE PROMPT IS THE DISCRIMINATOR, and it is the whole reason this measures
anything. Producer-side chat-wrapping and model-side leakage leave the SAME text
signature and have OPPOSITE causes (malign, docket [5047].2): a wrapped prompt
plus chat-flavoured output is the harness; a RAW prompt plus chat-flavoured
output is the model. An audit that flagged on text alone would have quarantined
62 models and destroyed the corpus.

Every prompt read here is raw. So every hit is the model reaching for the
assistant frame with nothing inviting it.

## TWO SIGNATURES, AND THEY DO NOT NEST

Reading the hits before counting them caught a bug in the first one: `As an AI`
without a word boundary matches "as an AIr conditioner", "as an aid", "as an
aircraft", which inflated base rates. `the assistant` is worse, because a story
may simply contain an assistant.

    LOOSE    the first pass, kept for comparability, ambiguous phrases included
    STRICT   control tokens and verbatim system-prompt openers ONLY

STRICT is not a subset of LOOSE: it drops the ambiguous phrases and adds control
tokens LOOSE lacked. zephyr-7b-beta reads 0.03% loose and 1.77% strict. Both are
reported so a difference is never read as a correction.

## THE ROSTER IS A DECLARED CHOICE AND IT CHANGES THE COUNTS

This project has produced 37 / 42 / 21 / 32 as "the roster n" in one evening.
So: the roster is `data/base_aligned_pairs.json`, the filter is a minimum
per-arm passage count at temp 1.0, and **both are printed in the header of every
run and written into the CSV**. At --min-n 200 it is 29 pairs; registrar's
independent check at 50 over the Registry's 52 declared pairs gave 33. Same
direction, same concentration, same top rates to the second decimal where the
rosters overlap -- corroboration on a DIFFERENT cut, which is worth more than
agreement on the same one, but the write-up carries one declared roster.

## WHAT MAY BE CLAIMED

The SIGN, roster-wide. Not the pooled magnitude: four Falcon3 Instruct models
run 27-53% while every other aligned model sits under 2.3%, so a pooled ratio is
one family wearing the roster's clothes. Ties (both arms at zero) are EXCLUDED
from the sign test rather than counted as agreement -- counting them as
agreement is what cost Y its sign test.
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

from malign_logits.cache import CacheManager   # noqa: E402

PAIRS = os.path.join(ROOT, "malign-logits", "data", "base_aligned_pairs.json")
if not os.path.exists(PAIRS):
    PAIRS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "data",
                         "base_aligned_pairs.json")

LOOSE = re.compile(
    r"<think>|<\|im_end\|>|You are a helpful|the user (?:needs|wants|asks|sent|gave)"
    r"|the assistant|<\|assistant\|>|As an AI", re.I)
STRICT = re.compile(
    r"<\|assistant\|>|<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>"
    r"|<think>|</think>|\[/?INST\]|<<SYS>>"
    r"|You are a helpful|You are an AI\b|As an AI\b(?! )"
    r"|I am unable to fulfill|I cannot assist with", re.I)


def sign_test(k, n):
    """One-sided binomial, exact. Ties are not passed in."""
    if n == 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=200,
                    help="minimum passages PER ARM for a pair to contribute")
    ap.add_argument("--csv", default=os.path.join(
        os.path.dirname(HERE), "results", "eassist_ambient.csv"))
    a = ap.parse_args(argv)

    pairs = [(p["base"], p["aligned"]) for p in json.load(open(PAIRS))]
    want = {m for b, x in pairs for m in (b, x)}
    st = CacheManager()._stash("generations")
    n = collections.Counter()
    L = collections.Counter()
    S = collections.Counter()
    for k in st.keys():
        m = k.get("model") or ""
        if m not in want or k.get("temp") != 1.0 or ":" in m:
            continue
        try:
            t = st.get(k)
        except Exception:
            continue
        if not isinstance(t, str):
            continue
        n[m] += 1
        if LOOSE.search(t):
            L[m] += 1
        if STRICT.search(t):
            S[m] += 1

    print("E-ASSIST-AMBIENT  |  roster %s  |  filter: both arms n >= %d at temp=1.0"
          % (os.path.basename(PAIRS), a.min_n))
    print("prompts are RAW throughout, so every hit is model-side\n")
    rows = [(b, x) for b, x in pairs if n[b] >= a.min_n and n[x] >= a.min_n]
    print("  %-38s %6s %7s %7s   %-38s %6s %7s %7s"
          % ("base", "n", "loose", "STRICT", "aligned", "n", "loose", "STRICT"))
    print("  " + "-" * 124)
    out = []
    for b, x in rows:
        rb, rx = S[b] / n[b], S[x] / n[x]
        print("  %-38s %6d %6.2f%% %6.2f%%   %-38s %6d %6.2f%% %6.2f%%"
              % (b[:38], n[b], 100 * L[b] / n[b], 100 * rb,
                 x[:38], n[x], 100 * L[x] / n[x], 100 * rx))
        out.append(dict(base=b, base_n=n[b], base_loose=L[b], base_strict=S[b],
                        base_strict_rate=round(rb, 6),
                        aligned=x, aligned_n=n[x], aligned_loose=L[x],
                        aligned_strict=S[x], aligned_strict_rate=round(rx, 6),
                        direction=("aligned" if rx > rb else
                                   "base" if rb > rx else "tie"),
                        roster=os.path.basename(PAIRS), min_n=a.min_n))
    up = sum(1 for r in out if r["direction"] == "aligned")
    dn = sum(1 for r in out if r["direction"] == "base")
    tie = sum(1 for r in out if r["direction"] == "tie")
    tb = sum(r["base_strict"] for r in out) / max(1, sum(r["base_n"] for r in out))
    tx = sum(r["aligned_strict"] for r in out) / max(1, sum(r["aligned_n"] for r in out))
    print("\n  STRICT, pair-level: aligned higher %d | base higher %d | tied at zero %d"
          "  (of %d pairs)" % (up, dn, tie, len(out)))
    print("  sign test over the %d pairs that MOVED: %d/%d, one-sided p = %.2g"
          % (up + dn, up, up + dn, sign_test(up, up + dn)))
    print("  pooled: base %.3f%%  aligned %.3f%%" % (100 * tb, 100 * tx))
    #: THE POOLED RATIO IS NOT THE FINDING. Print what carries it, every run,
    #: so the number cannot travel without its owner.
    top = sorted(out, key=lambda r: -r["aligned_strict_rate"])[:5]
    print("\n  the pooled ratio is carried by:")
    for r in top:
        print("     %-40s %6.2f%%   (its base %.2f%%)"
              % (r["aligned"][:40], 100 * r["aligned_strict_rate"],
                 100 * r["base_strict_rate"]))
    rest = [r for r in out if r not in top]
    if rest:
        print("     every other aligned model: max %.2f%%"
              % (100 * max(r["aligned_strict_rate"] for r in rest)))

    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("\n  wrote %s (%d pairs)" % (a.csv, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
