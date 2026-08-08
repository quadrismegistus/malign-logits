#!/usr/bin/env python
"""Four ways to test the same contrast, so the cost of each is visible.

    python y_paired_tests.py

THE SIGN TEST WAS THROWING AWAY THE DATA, not the pairing. It counts how many
pairs point up and discards how far each moved, so a pair that shifts 20 points
and a pair that shifts 0.01 count the same. At 25 pairs its floor is p=0.0000001
but it needs near-unanimity to get anywhere near it, and any real effect with a
few contrary pairs stalls around p=0.1.

    SIGN        directions only. Assumption-free, and nearly powerless.
    WILCOXON    signed-rank: keeps the pairing AND the magnitudes. Same n,
                substantially more power. This is the one that should be read.
    BOOTSTRAP   percentile CI on the median within-pair delta. Says how big,
                which no p-value does.
    POOLED      every row, pairing discarded. n goes from 25 to ~31,000 and the
                CI collapses -- which is why it is here as a WARNING rather
                than an option: this corpus has produced four readings that
                were one model carrying a pooled number, and the pooled column
                cannot tell that case from a real one.

Read WILCOXON and BOOTSTRAP together. Read POOLED only against them: where it
disagrees with the paired tests, the disagreement IS the finding, because it
means pair composition is doing the work.
"""
import collections
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
from malign_logits.tasks.code_y_superego_v3 import spans, LAYER1, LAYER2  # noqa: E402

IN = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")
SEED = 20260808


def sign_p(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def wilcoxon(d):
    """Two-sided signed-rank p by normal approximation with tie correction.

    Hand-rolled because scipy is not a dependency here. Zeros are DROPPED
    (Wilcoxon's own convention) and the count is reported, since dropping them
    shrinks n and a silent shrink is how an n stops meaning what it says.
    """
    v = [x for x in d if x != 0]
    n = len(v)
    if n < 6:
        return float("nan"), n
    order = sorted(range(n), key=lambda i: abs(v[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(v[order[j + 1]]) == abs(v[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wpos = sum(r for r, x in zip(ranks, v) if x > 0)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return float("nan"), n
    z = (wpos - mu) / sd
    return math.erfc(abs(z) / math.sqrt(2)), n


def boot_ci(d, reps=4000, seed=SEED):
    rng = random.Random(seed)
    med = []
    n = len(d)
    for _ in range(reps):
        med.append(statistics.median([d[rng.randrange(n)] for _ in range(n)]))
    med.sort()
    return med[int(.025 * reps)], med[int(.975 * reps)]


def main():
    rows = [json.loads(l) for l in open(IN)]
    ok = [r for r in rows if r.get("parsed") and r.get("pass") == "A"]
    print("pass A parsed %s   pairs %d\n" % (format(len(ok), ","), len({r["pair"] for r in ok})))

    MEAS = []
    for t in list(LAYER1) + list(LAYER2):
        MEAS.append(("<%s>" % t, (lambda r, t=t: ("<%s>" % t) in (r.get("tagged") or ""))))
    for f in ("assistant_refusal", "sexual_scene", "moralisation_in_scene",
              "guilt_or_shame", "consent_hesitation", "frame_exit", "degenerate"):
        MEAS.append((f, (lambda r, f=f: r.get(f) == "YES")))

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ok:
        for name, f in MEAS:
            per[name][(r["pair"], r["role"])].append(1.0 if f(r) else 0.0)

    print("  %-22s %10s %8s %9s %20s %10s" %
          ("measure", "median d", "SIGN", "WILCOXON", "bootstrap 95% CI", "POOLED d"))
    print("  " + "-" * 86)
    out = []
    for name, _ in MEAS:
        v = per[name]
        d, allb, alla = [], [], []
        for p in {x[0] for x in v}:
            b, a = v.get((p, "base")), v.get((p, "aligned"))
            if not b or not a:
                continue
            d.append(statistics.mean(a) - statistics.mean(b))
            allb += b; alla += a
        if len(d) < 8:
            continue
        pos = sum(1 for x in d if x > 0); n = len(d)
        sp = sign_p(max(pos, n - pos), n)
        wp, wn = wilcoxon(d)
        lo, hi = boot_ci(d)
        pooled = statistics.mean(alla) - statistics.mean(allb)
        out.append((wp, name, statistics.median(d), sp, wp, lo, hi, pooled, n, wn))
    out.sort()
    for wp, name, md, sp, _, lo, hi, pooled, n, wn in out:
        star = " ***" if wp < 0.01 else (" *" if wp < 0.05 else "")
        flag = ""
        if (lo > 0) != (pooled > 0) and (lo > 0 or hi < 0):
            flag = "  <-- POOLED DISAGREES"
        print("  %-22s %+10.4f %8.3f %9.4f  [%+7.4f,%+7.4f] %+10.4f%s%s"
              % (name, md, sp, wp, lo, hi, pooled, star, flag))
    print("\n  n pairs = %d; WILCOXON drops zero-deltas, so its n can be lower." % n)
    print("  A bootstrap CI excluding 0 is the claim worth making; the p-values rank.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
