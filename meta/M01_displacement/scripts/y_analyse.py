#!/usr/bin/env python
"""Read the confirmatory annotation. Per pair, never pooled.

    python y_analyse.py                  # everything
    python y_analyse.py --pass A         # just the full-length sample
    python y_analyse.py --tag moral      # one measure, with the pair table

THE UNIT IS THE PAIR. Every rate here is computed inside a pair and then
counted across pairs with a sign test. A pooled rate over 62,681 rows would
weight pairs by how much text they produced, and this corpus has already
produced four readings that were one model wearing a corpus-wide number.

WHAT IS PRINTED FOR EVERY MEASURE, in this order and never abridged:

    n pairs, and how many are positive        the sign test's actual input
    median within-pair difference             not the mean; one pair at 39x
                                              the median has already eaten a
                                              mean on this project
    IQR of the within-pair difference         the heterogeneity, which for a
                                              book arguing that different labs
                                              perform different operations is
                                              the finding, not the noise
    the per-pair table                        so a reader can see which pairs
                                              carry it

FIDELITY IS A COLUMN, NOT A FILTER. Rows are reported with their rt_band. Tag
RATES are computed on the coder's own reproduction and are unaffected by
round-trip drift; only span->source alignment needs the source. So drift is
shown and nothing is dropped for it here.
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
IN = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")

FIELDS = ["assistant_refusal", "frame_exit", "sexual_scene", "consummation",
          "moralisation_in_scene", "guilt_or_shame", "consent_hesitation",
          "continues_narrative", "degenerate", "noise_present"]
COMPOS = ["SUPEREGO_IN_SCENE", "EXIT", "CLEAN_SCENE", "MORAL_UTTERED"]
#: DERIVED FROM THE TASK, with one deliberate exclusion. Hand-listing this is
#: how `y_span_agreement.py` ended up searching a v3 corpus for v2's tags. The
#: only omission is <story>, which covers ~100% of every passage and whose rate
#: carries no information -- excluded by name so the omission is a decision
#: rather than a tag someone forgot to add when the vocabulary grew.
from malign_logits.tasks.code_y_superego_v3 import LAYER1, LAYER2  # noqa: E402

_NOT_A_RATE = {"story"}
TAGS = [t for t in (LAYER2 + LAYER1) if t not in _NOT_A_RATE]


def signtest(k, n):
    """One-sided p that k or more of n land positive under a fair coin."""
    if not n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def measure(rows, name, getter):
    """-> (per-pair deltas, per-pair (base,aligned) rates). Pairs with an empty
    arm are dropped and COUNTED, never silently skipped."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if not r.get("parsed"):
            continue
        by[r["pair"]][r["role"]].append(1 if getter(r) else 0)
    out, dropped = {}, 0
    for p, arms in by.items():
        b, a = arms.get("base") or [], arms.get("aligned") or []
        if not b or not a:
            dropped += 1
            continue
        out[p] = (statistics.mean(b), statistics.mean(a), len(b), len(a))
    return out, dropped


def concentration(rows, getter):
    """What share of the ALIGNED-arm events come from the top two pairs?

    THE CHECK THAT DECIDES WHETHER A RESULT SURVIVES, and it is not the p-value.
    In the pilot, <consent> looked like a clean doubling under alignment --
    1.4% to 2.5% -- until the model breakdown showed that ALL NINE aligned hits
    were AmberSafe and Tulu, two of six aligned models. A sign test over pairs
    does not see that: each pair contributes one sign, and 30 pairs sitting at
    zero with two pairs carrying everything can still read as a direction.

    So the concentration is printed beside every measure. A finding where the
    top two pairs hold most of the events is one model's behaviour with a
    corpus-wide number on it -- which has happened four times on this corpus
    already, every time with bloomz.
    """
    ev = collections.Counter()
    for r in rows:
        if r.get("parsed") and r.get("role") == "aligned" and getter(r):
            ev[r["pair"]] += 1
    tot = sum(ev.values())
    if not tot:
        return 0, 0.0, []
    top = ev.most_common(2)
    return tot, sum(c for _, c in top) / tot, top


def report(rows, label, getter, show_pairs=0):
    per, dropped = measure(rows, label, getter)
    if not per:
        print("  %-24s no data" % label)
        return
    d = {p: a - b for p, (b, a, nb, na) in per.items()}
    v = sorted(d.values())
    pos = sum(1 for x in v if x > 0)
    n = len(v)
    q1, q3 = v[n // 4], v[(3 * n) // 4]
    p = signtest(max(pos, n - pos), n)
    nev, share, top = concentration(rows, getter)
    print("  %-24s %2d/%2d pos  median %+7.4f  IQR %+7.4f..%+7.4f  sign p %s%s"
          % (label, pos, n, statistics.median(v), q1, q3,
             ("%.4f" % p) if p >= 1e-4 else "<0.0001",
             ("   [%d pairs dropped, an arm was empty]" % dropped) if dropped else ""))
    #: CONCENTRATION ON THE SAME LINE AS THE p-VALUE, so it cannot be read past.
    if nev:
        flag = "  <-- TWO PAIRS CARRY IT" if share >= 0.60 and n >= 8 else ""
        print("  %-24s   %d aligned events, top-2 pairs hold %.0f%%: %s%s"
              % ("", nev, 100 * share,
                 ", ".join("%s %d" % (pr.split(">")[-1].split("/")[-1][:18], c)
                           for pr, c in top), flag))
    if show_pairs:
        for pr, dv in sorted(d.items(), key=lambda kv: -kv[1])[:show_pairs]:
            b, a, nb, na = per[pr]
            print("       %-44s base %5.1f%%  algn %5.1f%%  %+6.2f  (n %d/%d)"
                  % (pr.split(">")[-1][:44], 100 * b, 100 * a, 100 * dv, nb, na))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pas", default=None, choices=("A", "B"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pairs", type=int, default=0, help="show the top N pairs per measure")
    a = ap.parse_args(argv)

    if not os.path.exists(IN):
        print("no results yet at %s" % IN); return 1
    rows = [json.loads(l) for l in open(IN)]
    if a.pas:
        rows = [r for r in rows if r.get("pass") == a.pas]
    ok = [r for r in rows if r.get("parsed")]
    print("rows %d   parsed %d   pairs %d   pass %s"
          % (len(rows), len(ok), len({r["pair"] for r in ok}), a.pas or "A+B"))
    band = collections.Counter(r.get("rt_band") for r in ok)
    print("round-trip: %s" % dict(band.most_common()))
    sev = [r for r in ok if r.get("rt_band") == "SEVERE"]
    if sev:
        print("  SEVERE drift on %d rows -- usable for tag rates, NOT for span->source:" % len(sev))
        for r in sev[:4]:
            print("     %s  %s  ratio %.3f" % (r["mid"], (r.get("model") or "?").split("/")[-1][:28],
                                               r.get("rt_ratio", 0)))
    print()
    if a.tag:
        print("TAG <%s>" % a.tag)
        report(ok, "<%s>" % a.tag, lambda r, t=a.tag: ("<%s>" % t) in (r.get("tagged") or ""),
               show_pairs=a.pairs or 12)
        return 0

    for title, items, get in (
            ("FIELDS", FIELDS, lambda r, f: r.get(f) == "YES"),
            ("COMPOSITES", COMPOS, lambda r, f: r.get(f) is True),
            ("TAGS", TAGS, lambda r, f: ("<%s>" % f) in (r.get("tagged") or ""))):
        print("%s -- within-pair, aligned minus base" % title)
        for f in items:
            report(ok, f if title != "TAGS" else "<%s>" % f,
                   (lambda r, f=f, g=get: g(r, f)), show_pairs=a.pairs)
        print()

    #: PASS A AND PASS B SIDE BY SIDE ON REFUSAL, which is the whole reason
    #: pass B exists: pass A's length filter keeps only refusals verbose enough
    #: to fill 256 tokens, and the marker rate peaks at 51-100.
    if not a.pas:
        print("REFUSAL, PASS A vs PASS B -- the reason B was censused")
        for p in ("A", "B"):
            sub = [r for r in ok if r.get("pass") == p]
            if sub:
                print("  pass %s (n=%d)" % (p, len(sub)))
                report(sub, "  assistant_refusal", lambda r: r.get("assistant_refusal") == "YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
