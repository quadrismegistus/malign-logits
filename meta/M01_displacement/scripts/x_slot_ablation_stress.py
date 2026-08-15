#!/usr/bin/env python3
"""Stress the 39-item result. A null or a reversal is where to START looking.

    x_slot_ablation_stress.py [--results results/x_slot_ablation_61.json]

The registered run returned: `no-safety` the only arm whose CI excludes zero on
the 39 fresh items, `no-wildchat` a zero-spanning null there while being the
largest arm on the 61 it was derived from. Four ways to try to break that, all
on data already collected, none of them a new spend.

  A  TRIMMED. Drop the k largest |d| items per arm. If one item carries the
     safety interval, it dies immediately; §4a's own robustness argument was
     exactly this test and it is applied here to the arm that now survives.
  B  BOOTSTRAP, because the t interval assumes a shape 39 items cannot show.
  C  THE CUTS §4a USED -- residual <= 0.30, and leverage above the dead
     reference. If the reversal only exists at low quality it is not a result.
  D  THE MECHANISM CLAIM. §4a said WildChat is "specifically what supplies the
     replacement": for `no-wildchat` the SUBSTITUTION term collapsed to +0.0003
     while suppression stayed. That is a sharper claim than the magnitude one
     and it is separately testable on these 39.

D IS THE ONE THAT MATTERS. Magnitude claims can reverse because an instrument
was fitted to its items; a MECHANISM claim about which half of dN moves is much
harder to produce by fitting, so if D replicates while the magnitude does not,
the honest reading changes from "the result was noise" to "the size was fitted
and the structure was not".
"""
import argparse, json, os, sys

ROOT = "/Users/rj416/github/malign-logits"
sys.path.insert(0, ROOT)
POP = os.path.join(ROOT, "meta/M01_displacement/populations/reg_slot_new_items.json")
ARMS = ["no-safety", "no-math", "no-persona", "no-wildchat"]


def mean_ci(xs, conf=0.95):
    n = len(xs)
    if n < 2:
        return float("nan"), (float("nan"), float("nan"))
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    try:
        from scipy import stats
        t = stats.t.ppf(0.5 + conf / 2.0, n - 1)
    except Exception:
        t = 2.03
    return m, (m - t * se, m + t * se)


def boot(xs, n=20000, seed=11):
    import random
    r = random.Random(seed)
    N = len(xs)
    ms = sorted(sum(r.choice(xs) for _ in range(N)) / N for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def d_rows(rows, arm):
    return [(r["arms"][arm]["dN"] - r["arms"]["full"]["dN"], r)
            for r in rows if arm in r["arms"] and "full" in r["arms"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        ROOT, "meta/M01_displacement/results/x_slot_ablation_61.json"))
    #: ALL 61 IS NOT AN OUT-OF-SAMPLE TEST AND THE LABEL SAYS SO EVERY TIME.
    #: 22 of the 61 are the items 4a's claims were derived from, so a stress
    #: result over the full set is a stability check on an estimate that
    #: contains its own source. Useful -- it is the larger n and RH asked for
    #: it -- and it can never be quoted as replication.
    ap.add_argument("--all", action="store_true",
                    help="stress ALL 61 items, not just the 39 registered")
    a = ap.parse_args()
    items = json.load(open(a.results))["items"]
    pset = {p.strip() for p in json.load(open(POP))}
    reg = [r for r in items if r["prompt"].strip() in pset]
    assert len(reg) == len(pset), "population join is %d of %d" % (len(reg), len(pset))
    if a.all:
        n_src = len(items) - len(reg)
        reg = items
        print("  stressing ALL %d items -- %d registered + %d that 4a's claims "
              "were DERIVED FROM.\n  NOT an out-of-sample test; read as stability, "
              "never as replication.\n" % (len(items), len(pset), n_src))
    else:
        print("  stressing the %d registered items\n" % len(reg))

    # ── A. TRIMMED
    print("  A. TRIMMED MEANS -- drop the k largest |d| items per arm")
    print("     %-13s %10s %10s %10s %10s" % ("arm", "k=0", "k=1", "k=2", "k=3"))
    for arm in ARMS:
        ds = sorted(d_rows(reg, arm), key=lambda x: -abs(x[0]))
        cells = []
        for k in range(4):
            xs = [d for d, _ in ds[k:]]
            m, (lo, hi) = mean_ci(xs)
            cells.append("%+.5f%s" % (m, "*" if lo > 0 or hi < 0 else " "))
        print("     %-13s %10s %10s %10s %10s" % (arm, *cells))
    print("     * = 95% CI excludes zero")

    # ── B. BOOTSTRAP
    print("\n  B. BOOTSTRAP 20,000x, percentile interval")
    for arm in ARMS:
        xs = [d for d, _ in d_rows(reg, arm)]
        m, (tlo, thi) = mean_ci(xs)
        blo, bhi = boot(xs)
        print("     %-13s mean %+.5f   t [%+.5f, %+.5f]   boot [%+.5f, %+.5f] %s"
              % (arm, m, tlo, thi, blo, bhi, "EXCLUDES 0" if blo > 0 or bhi < 0 else ""))

    # ── C. THE QUALITY CUTS
    print("\n  C. THE CUTS 4a USED")
    from malign_logits.slot_axis import LEV_DEAD
    for label, keep in [
            ("all", lambda r: True),
            ("residual <= 0.30", lambda r: r.get("resid_base", 1) <= 0.30),
            ("residual <= 0.25 + lev > dead",
             lambda r: r.get("resid_base", 1) <= 0.25 and r.get("leverage", 0) > LEV_DEAD)]:
        sub = [r for r in reg if keep(r)]
        out = []
        for arm in ARMS:
            xs = [d for d, _ in d_rows(sub, arm)]
            if len(xs) < 2:
                out.append("%s n/a" % arm); continue
            m, (lo, hi) = mean_ci(xs)
            out.append("%s %+.5f%s" % (arm.replace("no-", ""), m,
                                       "*" if lo > 0 or hi < 0 else ""))
        print("     %-30s n=%2d   %s" % (label, len(sub), "  ".join(out)))

    # ── D. THE MECHANISM
    print("\n  D. MECHANISM -- does the SUBSTITUTION collapse replicate?")
    print("     4a: full substitution -0.0131, no-wildchat substitution +0.0003")
    print("     %-13s %12s %12s %12s" % ("arm", "suppression", "substitution", "dN"))
    for arm in ["full"] + ARMS:
        sup = [r["arms"][arm]["suppression"] for r in reg if arm in r["arms"]]
        sub = [r["arms"][arm]["substitution"] for r in reg if arm in r["arms"]]
        dn = [r["arms"][arm]["dN"] for r in reg if arm in r["arms"]]
        if not sup:
            continue
        print("     %-13s %+12.5f %+12.5f %+12.5f"
              % (arm, sum(sup) / len(sup), sum(sub) / len(sub), sum(dn) / len(dn)))
    #: the paired form: per item, how much of `full`'s substitution does the
    #: arm retain? A collapse is a ratio near zero, not a smaller number.
    print("\n     paired substitution ratio, arm / full  (per item, then median)")
    for arm in ARMS:
        rs = []
        for r in reg:
            if arm in r["arms"] and "full" in r["arms"]:
                f = r["arms"]["full"]["substitution"]
                if abs(f) > 1e-6:
                    rs.append(r["arms"][arm]["substitution"] / f)
        rs.sort()
        if rs:
            print("        %-13s median %+.3f   (n=%d, IQR %+.3f..%+.3f)"
                  % (arm, rs[len(rs) // 2], len(rs),
                     rs[len(rs) // 4], rs[3 * len(rs) // 4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
