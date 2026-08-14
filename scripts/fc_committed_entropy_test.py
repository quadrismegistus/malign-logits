#!/usr/bin/env python
"""fc_committed_entropy_test.py — the ONE committed test, specified at docket
[4796] and fixed at [4798], WRITTEN BEFORE THE DATA LANDED.

    scripts/fc_committed_entropy_test.py

WHAT IS COMMITTED, verbatim from [4796].3 and [4798]:

  * runs ONCE, at 33 pairs, defined as the moment phi-4 and Olmo-32B land
  * population: every pair clearing the STANDING >=5-site floor. That floor
    antedates this question -- it already governs the damage estimates in the
    same run -- so it applies rather than being chosen here.
  * DEEPSEEK STAYS, in every cell, always. Removing the one reversed pair from
    the analysis that tests whether the effect survives is the move malign
    refused at [4795].3 and the register bars it. Its residual is READ, not
    removed.
  * the relation is CONTINUOUS: Spearman AND a linear fit, both declared in
    advance, NO BINNING. Terciles cut the data into three cells and then
    interrogate the weakest; the fit uses every pair.
  * the low-concentration question is answered by the FITTED ASYMMETRY AT ZERO
    ENTROPY DROP and its confidence interval. Intercept excluding zero => the
    effect exists where concentration does not. Including zero => the
    low-concentration regime is unresolved at final n, and says so.

WHY IT IS WRITTEN NOW. The numbers do not exist yet. A test authored after
seeing its inputs can be shaped by them without anyone intending it -- the
binning I chose at [4793] was tercile because terciles are conventional, and
that choice was never declared. This file is the declaration.

WHAT THE RESULT MUST CARRY WHEREVER IT TRAVELS:
  "on 33 of the 36-pair roster; both Falcon-H1 and both Mamba pairs absent on
   the selective_scan_cuda conflict"  ([4798], the no-silent-caps rule)
  plus the standing population qualifier: cross-model-recurrent movers, until
  pass 2 lands. Committing the analysis does not launder the population.

RIDER ([4798].1): if the SSM pairs land later, THIS 33-pair result stands as
the committed number. Any 37-pair rerun is reported BESIDE it as an extension,
never silently in its place -- replication as the control, not revision.
"""
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

MIN_SITES = 5          #: the standing floor, not chosen here
TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)


def entropy(st, prepare, mid, prompt):
    k = dict(TWP); k["model"] = mid; k["prompt"] = prompt
    try:
        v = st[k]
    except Exception:
        return None
    rows = v.get("rows") if isinstance(v, dict) else None
    if not rows:
        return None
    o, pr = prepare(rows)
    tot = sum(pr[w] for w in o) or 1.0
    return -sum((pr[w] / tot) * math.log(pr[w] / tot) for w in o if pr[w] > 0)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for j, i in enumerate(order):
            r[i] = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def linfit(xs, ys):
    """OLS slope, intercept, and the intercept's 95% CI.

    The INTERCEPT is the committed quantity: the fitted asymmetry at zero
    entropy drop, i.e. the effect where no concentration happened. Its standard
    error carries the usual sum-of-squares term, so it widens as the data get
    further from x=0 -- which is honest here, since no pair has a drop of
    exactly zero and the intercept is an extrapolation to the edge of the
    observed range rather than an interpolation.
    """
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    resid = [y - (inter + slope * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se_i = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    t = 2.045 if n >= 30 else 2.120        #: t(.975), df ~29 / ~16
    return slope, inter, se_i, (inter - t * se_i, inter + t * se_i), resid


def main():
    import fc_analyse as F
    from malign_logits.cache import get_cache
    from m05_sites import prepare
    cm = get_cache()
    st = cm._stash("true_word_probs")
    by = F.load(cm, None)

    pairs = []
    for pid, cells in sorted(by.items()):
        base, aligned = pid.split(">")
        per = {}
        for (role, arm, w, prompt), rec in cells.items():
            if arm != "undisturbed":
                continue
            sb, sa = rec.get("scored_by_base"), rec.get("scored_by_aligned")
            if not sb or not sa:
                continue
            first, second = (sb, sa) if role == "base" else (sa, sb)
            v = [x - y for r1, r2 in zip(first, second)
                 for i, (x, y) in enumerate(zip(r1, r2)) if i > 0]
            if v:
                per.setdefault(prompt, {})[role] = statistics.mean(v)
        asym = [(d["base"] - d["aligned"]) / 2 for d in per.values() if len(d) == 2]
        if len(asym) < MIN_SITES:
            continue
        drops = []
        for prompt in per:                        #: SAME SITES, both quantities
            eb = entropy(st, prepare, base, prompt)
            ea = entropy(st, prepare, aligned, prompt)
            if eb is not None and ea is not None:
                drops.append(eb - ea)
        if len(drops) < MIN_SITES:
            continue
        pairs.append((base.split("/")[-1][:24], statistics.mean(asym),
                      statistics.mean(drops)))

    A = [a for _, a, _ in pairs]
    D = [d for *_, d in pairs]
    print("THE COMMITTED TEST — [4796].3, fixed [4798], written before the data")
    print("  pairs clearing the standing >=%d-site floor: %d" % (MIN_SITES, len(pairs)))
    print()
    print("  Spearman(asymmetry, entropy drop) = %+.3f" % spearman(A, D))
    slope, inter, se, ci, resid = linfit(D, A)
    print("  linear fit: asymmetry = %+.4f %+.4f * drop" % (inter, slope))
    print()
    print("  *** THE COMMITTED QUANTITY ***")
    print("  fitted asymmetry at ZERO entropy drop = %+.4f" % inter)
    print("  95%% CI  [%+.4f, %+.4f]   (se %.4f)" % (ci[0], ci[1], se))
    excl = ci[1] < 0 or ci[0] > 0
    print()
    if excl:
        print("  CI EXCLUDES ZERO -> the effect exists where concentration does not.")
    else:
        print("  CI INCLUDES ZERO -> the low-concentration regime is UNRESOLVED at")
        print("  this n. That is the answer, not a failure of the test.")
    ds = [(n, a, d, r) for (n, a, d), r in zip(pairs, resid) if "deepseek" in n]
    if ds:
        n, a, d, r = ds[0]
        print()
        print("  DEEPSEEK, read not removed ([4798]): drop %+.3f  asym %+.4f"
              "  residual %+.4f" % (d, a, r))
        print("  it is %.1f residual-sd from the fit"
              % (abs(r) / statistics.pstdev(resid) if statistics.pstdev(resid) else 0))
    print()
    print("  TRAVELS WITH THIS NUMBER, ALWAYS:")
    print("   on %d of the 36-pair roster; both Falcon-H1 and both Mamba pairs" % len(pairs))
    print("   absent on the selective_scan_cuda conflict. Population:")
    print("   cross-model-recurrent movers, until pass 2 lands.")


if __name__ == "__main__":
    main()
