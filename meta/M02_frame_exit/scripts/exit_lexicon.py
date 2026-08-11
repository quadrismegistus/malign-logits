#!/usr/bin/env python
"""Build the exit lexicon that `exit-free` means, from M01's coded spans.

    uv run python exit_lexicon.py            # derive and write
    uv run python exit_lexicon.py --eval     # score it against M02's coder

WHY THIS FILE EXISTS, AND IT IS A DEBT BEING PAID. `findings/second_order_naming.md`
leads with "52,559 EXIT-FREE passages" and every headline number in it -- the
2.18x regex contrast, the 3.37x pooled Opus contrast, the pole control at
unity -- is computed on that subset. **The lexicon defining it lived only in a
session scratchpad and was never committed**, so the finding could not be
re-derived from this repository. That is the produce-before-plot failure the
campaign tracks in `meta/plot-debt.md`, committed by the seat that has spent
the day pointing at it elsewhere.

## WHY A LEXICON AND NOT `y_exit_typology.TYPES`

Measured against M02's coder on the same 50-word windows:

    y_exit_typology regexes    13.7% recall   94.1% precision
    this lexicon               51.9% recall   71.2% precision
    the union                  57.5% recall   72.8% precision

The declared regexes are a high-precision, near-blind instrument: they miss 86%
of what a reader calls a frame exit. Any null measured on them is a null about
the regex-visible exit, which is what `depth_and_exit_do_not_join.md` says in
its own limits.

## HOW IT IS DERIVED, AND THE ONE CHOICE THAT MATTERS

From `meta/M01_displacement/results/y_passages.parquet` -- 62,167 coder-parsed
passages carrying tagged span regions. Words distinctive of `<meta>` /
`<web>` / `<refusal>` spans against `<story>` spans, by log-odds.

**THE SPAN POOL IS BALANCED BY ARM AND THIS IS NOT COSMETIC.** `<refusal>`
appears in 1,468 aligned passages against 130 base ones, 10.6x. A lexicon read
off the unbalanced pool is an ALIGNED-arm vocabulary, and applying it to an
arm contrast imports the effect under test. Equal numbers are sampled from
each arm for every span kind, and the reference `<story>` pool likewise.

The mirror error is just as bad and was made first: deriving from the BASE arm
alone to be "safe" builds a base-flavoured lexicon and biases the other way.
Balanced is the answer; neither arm is privileged.

DERIVED IN M01 AND APPLIED IN M02, so it is out-of-sample for every M02 arm
contrast it filters.
"""
import argparse
import collections
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

SRC = os.path.join(ROOT, "meta", "M01_displacement", "results", "y_passages.parquet")
OUT = os.path.join(CAMP, "results", "exit_lexicon.json")

KINDS = ("meta", "web", "refusal")
REF = "story"
SEED = 4946
MIN_COUNT = 40        #: a word must appear in this many spans to be estimated
LOG_ODDS = 2.2        #: threshold against the story reference


def spans_by_arm(df, tag):
    out = {"base": [], "aligned": []}
    pat = re.compile(r"<%s>(.*?)</%s>" % (tag, tag), re.S | re.I)
    for arm, t in zip(df["arm"], df["tagged"]):
        if arm in out:
            out[arm] += pat.findall(t)
    return out


def balanced(pool, rng):
    k = min(len(pool["base"]), len(pool["aligned"]))
    return rng.sample(pool["base"], k) + rng.sample(pool["aligned"], k), k


def derive():
    import pandas as pd
    d = pd.read_parquet(SRC)
    d = d[(d["parsed"] == True) & (d["tagged"].notna())]  # noqa: E712
    print("coder-parsed passages: %s  (base %s / aligned %s)"
          % (format(len(d), ","), format((d.arm == "base").sum(), ","),
             format((d.arm == "aligned").sum(), ",")))
    rng = random.Random(SEED)
    pool, prov = [], {}
    for tag in KINDS:
        s = spans_by_arm(d, tag)
        bal, k = balanced(s, rng)
        prov[tag] = {"base": len(s["base"]), "aligned": len(s["aligned"]),
                     "balanced_each": k}
        print("  <%s> base %d aligned %d -> balanced %d+%d  (arm skew %.1fx)"
              % (tag, len(s["base"]), len(s["aligned"]), k, k,
                 (len(s["aligned"]) / len(s["base"])) if s["base"] else float("inf")))
        pool += bal
    ref, kref = balanced(spans_by_arm(d, REF), rng)
    print("  <%s> reference, balanced %d+%d" % (REF, kref, kref))

    a, r = collections.Counter(), collections.Counter()
    for m in pool:
        a.update(set(re.findall(r"[a-z']+", m.lower())))
    for m in ref:
        r.update(set(re.findall(r"[a-z']+", m.lower())))
    na, nr = len(pool), len(ref)
    words = []
    for w, c in a.items():
        if c < MIN_COUNT or len(w) < 3:
            continue
        lo = math.log(((c + 0.5) / (na + 1)) / ((r.get(w, 0) + 0.5) / (nr + 1)))
        if lo > LOG_ODDS:
            words.append((round(lo, 4), c, w))
    words.sort(reverse=True)
    meta = {"source": os.path.relpath(SRC, ROOT), "seed": SEED,
            "kinds": list(KINDS), "reference": REF,
            "min_count": MIN_COUNT, "log_odds_threshold": LOG_ODDS,
            "span_provenance": prov, "spans_pooled": na, "reference_spans": nr,
            "n_words": len(words),
            "note": "balanced by arm; derived in M01, applied in M02"}
    json.dump({"_meta": meta, "words": [w for _, _, w in words],
               "detail": [{"word": w, "log_odds": lo, "spans": c}
                          for lo, c, w in words]}, open(OUT, "w"), indent=1)
    print("\n%d words -> %s" % (len(words), os.path.relpath(OUT, ROOT)))
    print("  top 12: %s" % ", ".join(w for _, _, w in words[:12]))


def evaluate():
    """Recall and precision against M02's coded `frame_exit`, same windows."""
    import glob
    sys.path.insert(0, HERE)
    from y_exit_typology import TYPES
    words = json.load(open(OUT))["words"]
    rx = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(w) for w in words), re.I)
    seen = {}
    for f in ("l2_treatment_paired500", "l2_treatment_paired100_v2",
              "l2_treatment_n100"):
        p = os.path.join(CAMP, "results", f + ".jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            seen.setdefault((r["model"], r["group"], r.get("sample_idx"),
                             (r.get("prompt") or "")[:40]), r)
    rows = [r for r in seen.values() if r.get("arm") in ("base", "aligned")]
    old = lambda t: any(p.search(t or "") for _, p in TYPES)  # noqa: E731
    print("against M02's coded frame_exit, %d passages\n" % len(rows))
    print("  %-26s %8s %10s" % ("instrument", "recall", "precision"))
    for name, fn in (("y_exit_typology", old),
                     ("this lexicon", lambda t: bool(rx.search(t or ""))),
                     ("union", lambda t: old(t) or bool(rx.search(t or "")))):
        tp = sum(1 for r in rows if r["frame_exit"] == "YES" and fn(r["text"]))
        fn_ = sum(1 for r in rows if r["frame_exit"] == "YES" and not fn(r["text"]))
        fp = sum(1 for r in rows if r["frame_exit"] == "NO" and fn(r["text"]))
        print("  %-26s %7.1f%% %9.1f%%"
              % (name, 100 * tp / (tp + fn_), 100 * tp / (tp + fp) if tp + fp else 0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.eval:
        evaluate()
    else:
        derive()
        evaluate()
