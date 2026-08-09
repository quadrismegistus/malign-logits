#!/usr/bin/env python
"""The registered primary's CODER-FREE TWIN: does the cross-model penalty
accumulate across a passage, and does it accumulate more at a contradiction?

    python l2_crossscore_slope.py

Corpus: the L2 fleet, `data/raw/f11_l2/`, 26 complete pairs ([5215]). No coding,
no coder gate, no spend -- arithmetic over rows that already exist.

## THE MEASURE, DECLARED BEFORE THE CORPUS EXISTED

Declared at [5199].1, its alignment fixed in advance at [5202] and RULED at
[5204] (slope-as-primary, not a tolerated deviation). Nothing below is chosen
after seeing the data.

    d(i) = logP_partner(tok_i) - logP_author(tok_i)      at continuation token i

NEGATIVE = the partner model finds this token less probable than the model that
wrote it. On a base-written passage the partner is the aligned model, so d is
how much alignment disprefers its parent's continuation.

    PRIMARY     per-passage SLOPE of d(i) on i, then the DISTRIBUTION of slopes,
                family-clustered, BOTH against mean(CONTROL_A, CONTROL_B).
    ANCHOR      d(1). The 20 samples of a cell share the prompt, so at position 1
                the context is byte-identical across them; d(1) is poolable and
                is the L1 quantity [5195] measured at 2% of variance.
    SECONDARY   the per-position pooled curve, DESCRIPTIVE and labelled, never
                the test.

**WHY THE SLOPE AND NOT THE POOLED CURVE.** malign's caveat at [5200].6: the 20
samples share a prompt but not a continuation, so position 40 of one is not the
same place in a story as position 40 of another, and "a flat curve and a curve
destroyed by averaging look identical." Worse: passages that depart at different
depths pool into a GENTLE RAMP, which is exactly the shape [5199] named as
confirming the passage-grain argument. The artifact does not hide the predicted
result, it manufactures it. A slope taken WITHIN a passage cannot be forged by
averaging; a pooled curve can.

**The claim under test**: slope < 0 while d(1) is small would mean the penalty
accumulates over a passage rather than sitting at the substitution -- the L2
argument. Slope flat means the move to passages was wrong, and the same data
says so.
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

RAW = os.path.join(ROOT, "data", "raw", "f11_l2")
ROLES = ("both", "control_a", "control_b", "pole_a", "pole_b")
NEG = {"f11_reason", "f11_reason_zh"}


def slug(m):
    return m.replace("/", "__")


def load_scores(model):
    """(src_model, prompt_sha) -> {sample_idx: logprobs}, from this model's file."""
    p = os.path.join(RAW, "%s.score.jsonl" % slug(model))
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = {}
        for s in r.get("scores") or []:
            lp = s.get("logprobs")
            if lp:
                d[s["sample_idx"]] = lp
        out[(r["src_model"], r["prompt_sha256_16"])] = d
    return out


def load_claims(model):
    """prompt_sha -> (lang, [(group, role)])"""
    p = os.path.join(RAW, "%s.gen.jsonl" % slug(model))
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["prompt_sha256_16"]] = (r.get("lang"),
                                      [(c.get("group"), c.get("role")) for c in (r.get("claims") or [])])
    return out


def slope(y):
    """OLS slope of y on its index. Closed form; x is 0..n-1 so sums are exact."""
    n = len(y)
    if n < 8:
        return None
    sx = n * (n - 1) / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sy = sum(y)
    sxy = sum(i * v for i, v in enumerate(y))
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "zh", "all"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from scipy import stats as sst

    #: POPULATION FROM THE RECEIPT, NOT FROM THE FILESYSTEM. `complete` is the
    #: 26 pairs malign certifies; `partial` is 2 with complete generation and
    #: INCOMPLETE scoring, which [5215].2 says explicitly are "NOT usable as
    #: pairs and must not be counted as such". Globbing for score files picks
    #: them up and gives 29 -- the exact "the roster was 26" error the receipt's
    #: taxonomy exists to prevent, made by the seat that asked for the taxonomy.
    R = json.load(open(os.path.join(ROOT, "data", "f11_l2_receipt.json")))
    ok = {(c["base"], c["aligned"]) for c in R["complete"]}
    pairs = json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))
    live = [p for p in pairs if (p["base"], p["aligned"]) in ok]
    print("receipt: complete %d, partial %d (excluded), never-started %d, dead %d, bf16-deferred %d"
          % (len(R["complete"]), len(R["partial"]), len(R["never_started"]),
             len(R["excluded_known_dead"]), len(R["deferred_to_bf16_box"])))
    if a.limit:
        live = live[:a.limit]
    print("pairs analysed: %d\n" % len(live))

    #: per (family, group, role, arm) -> list of per-passage slopes and d(1)
    S = collections.defaultdict(list)
    D1 = collections.defaultdict(list)
    CURVE = collections.defaultdict(lambda: collections.defaultdict(list))
    led = collections.Counter()

    for pr in live:
        b, al, fam = pr["base"], pr["aligned"], pr["family"]
        sc_b, sc_a = load_scores(b), load_scores(al)
        cl = load_claims(b) or load_claims(al)
        for arm, author, partner in (("base", b, al), ("aligned", al, b)):
            own = (sc_b if author == b else sc_a)
            oth = (sc_a if author == b else sc_b)
            for (src, sha), samples in own.items():
                if src != author:
                    continue                      #: self rows only, as the author
                cross = oth.get((src, sha))
                if not cross:
                    led["no partner row"] += 1
                    continue
                lang, claims = cl.get(sha, (None, []))
                if a.lang != "all" and lang != a.lang:
                    continue
                for si, own_lp in samples.items():
                    x_lp = cross.get(si)
                    if not x_lp or len(x_lp) != len(own_lp):
                        led["length mismatch"] += 1
                        continue
                    d = [x_lp[i] - own_lp[i] for i in range(len(own_lp))]
                    #: asserted, not assumed: one -inf makes a slope NaN and a
                    #: NaN propagates silently through every mean above it
                    if not all(v == v and abs(v) != float("inf") for v in d):
                        led["non-finite d"] += 1
                        continue
                    sl = slope(d)
                    if sl is None:
                        continue
                    for grp, role in claims:
                        if role not in ROLES or grp in NEG:
                            continue
                        S[(fam, grp, role, arm)].append(sl)
                        D1[(fam, grp, role, arm)].append(d[0])
                        for i in range(0, min(len(d), 256), 16):
                            CURVE[(role, arm)][i].append(d[i])
    print("cells: %s   dropped: %s\n" % (format(len(S), ","), dict(led)))

    def excess(store, label, unit):
        """BOTH minus mean(CONTROL_A, CONTROL_B), within (family, group), then
        across families. The registered contrast, on a coder-free quantity."""
        print("=" * 92)
        print("%s   %s" % (label, unit))
        print("=" * 92)
        print("  %-8s %7s %7s %10s %10s %11s %10s %6s"
              % ("arm", "cells", "fams", "BOTH", "controls", "EXCESS", "p", "sign"))
        for arm in ("base", "aligned"):
            byfam = collections.defaultdict(list)
            bo, co = [], []
            for (fam, grp, role, ar), v in store.items():
                if ar != arm or role != "both":
                    continue
                ca = store.get((fam, grp, "control_a", ar))
                cb = store.get((fam, grp, "control_b", ar))
                if not ca or not cb:
                    continue
                mb = statistics.mean(v)
                mc = (statistics.mean(ca) + statistics.mean(cb)) / 2
                byfam[fam].append(mb - mc)
                bo.append(mb)
                co.append(mc)
            fm = [statistics.mean(x) for x in byfam.values()]
            if len(fm) < 5:
                print("  %-8s (only %d families)" % (arm, len(fm)))
                continue
            t, p = sst.ttest_1samp(fm, 0)
            med = statistics.median(fm)
            print("  %-8s %7d %7d %10.4f %10.4f %+11.4f %10.2e %3d/%-2d%s"
                  % (arm, len(bo), len(fm), statistics.mean(bo), statistics.mean(co),
                     statistics.mean(fm), p,
                     sum(1 for x in fm if (x > 0) == (med > 0)), len(fm),
                     "  <=" if p < 0.05 else ""))
        print()

    excess(S, "PRIMARY: per-passage SLOPE of d(i) on i", "nats per token, per token index")
    excess(D1, "ANCHOR: d(1), the L1 quantity", "nats at the first continuation token")

    print("=" * 92)
    print("SECONDARY, DESCRIPTIVE ONLY: pooled d by position. Not the test -- see docstring.")
    print("=" * 92)
    print("  %-10s %-8s %s" % ("role", "arm", "  ".join("%4d" % i for i in range(0, 256, 32))))
    for role in ("both", "control_a", "control_b"):
        for arm in ("base", "aligned"):
            row = CURVE.get((role, arm), {})
            if not row.get(0):
                continue
            print("  %-10s %-8s %s" % (role, arm, "  ".join(
                "%+.2f" % statistics.mean(row[i]) if row.get(i) else "   . " for i in range(0, 256, 32))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
