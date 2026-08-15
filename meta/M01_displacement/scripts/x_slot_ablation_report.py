#!/usr/bin/env python3
"""Report the Tulu ablation run AGAINST THE FROZEN REGISTRATION.

    x_slot_ablation_report.py [--results results/x_slot_ablation_61.json]

The registration is `meta/M01_displacement/registration_slot_ablation.md`,
frozen at 7c4d1f4b BEFORE this run. It names two claims as executable rules and
names the falsifying outcome for each. This script executes them and prints the
verdict whichever way it lands.

    d(arm) = dN(arm) - dN(full)     per item
    positive => removing that corpus SUPPRESSED LESS, i.e. that corpus was
    carrying part of the reduction.

THE REGISTERED TEST IS THE 39 NEVER-RUN ITEMS, not the 61. The other 22 are the
items both claims were DERIVED from; including them would be predicting an
observation. The 61-item numbers are printed too and are labelled as containing
their own source, per registration §1.

TWO SPECIFICATION PARSES, NAMED RATHER THAN SILENTLY RESOLVED. A producer
resolves every ambiguity merely by existing, so the ambiguous ones get said out
loud:

  (1) "the effect size WildChat shows" in claim B is read as `mean d(no-wildchat)`
      -- WildChat's own effect. The rival reading, the wildchat-MINUS-safety
      contrast, is also printed; the claim is only reported as clean if it holds
      under BOTH.
  (2) TIES ARE DROPPED from sign tests and the dropped count is reported, per
      the campaign rule. A tie is not evidence for the null and counting it as
      one inflates n.

AND THE DECLARED HETEROGENEITY IS CARRIED THROUGH. Registration §4 declares five
identity-group twins as five observations of ONE frame. Every count is therefore
reported twice: at n=39 treating them as independent, and collapsed to their
mean as one observation (n=35). If a claim's verdict differs between the two,
the claim depends on counting one frame five times and that is said.
"""
import argparse, json, math, os, re, sys

ROOT = "/Users/rj416/github/malign-logits"
sys.path.insert(0, ROOT)
POP = os.path.join(ROOT, "meta/M01_displacement/populations/reg_slot_new_items.json")
#: The five identity-group twins, declared in registration §4.
TWIN = re.compile(r"^Three (Muslims|Arabs|Jews|Black men|white men)\b")


def sign_test(diffs):
    """Two-sided exact binomial on the signs. TIES DROPPED, and counted."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - pos - neg
    n = pos + neg
    if n == 0:
        return dict(pos=0, neg=0, ties=ties, n=0, p=1.0)
    C = math.comb
    k = min(pos, neg)
    p = min(1.0, 2.0 * sum(C(n, i) for i in range(k + 1)) / (2.0 ** n))
    return dict(pos=pos, neg=neg, ties=ties, n=n, p=p)


def mean_ci(xs, conf=0.95):
    """Mean and a t interval. n is small enough that t matters."""
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), (float("nan"), float("nan"))
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    try:
        from scipy import stats
        t = stats.t.ppf(0.5 + conf / 2.0, n - 1)
    except Exception:
        t = 2.03
    return m, (m - t * se, m + t * se)


def collapse_twins(rows):
    """Five identity twins -> ONE row carrying their mean. Registration §4."""
    twins = [r for r in rows if TWIN.match(r["prompt"])]
    if len(twins) < 2:
        return rows, len(twins)
    rest = [r for r in rows if not TWIN.match(r["prompt"])]
    arms = sorted({a for r in twins for a in r["arms"]})
    merged = {"item_id": "IDENTITY-TWINS(collapsed x%d)" % len(twins),
              "prompt": "Three <group> ...", "arms": {}}
    for a in arms:
        ds = [r["arms"][a]["dN"] for r in twins if a in r["arms"]]
        merged["arms"][a] = {"dN": sum(ds) / len(ds)}
    return rest + [merged], len(twins)


def d_of(rows, arm):
    """d(arm) per item, only where BOTH that arm and `full` exist."""
    out = []
    for r in rows:
        if arm in r["arms"] and "full" in r["arms"]:
            out.append(r["arms"][arm]["dN"] - r["arms"]["full"]["dN"])
    return out


def paired(rows, a, b):
    """Per-item d(a) - d(b). Paired, so `full` cancels: dN(a) - dN(b)."""
    out = []
    for r in rows:
        if a in r["arms"] and b in r["arms"]:
            out.append(r["arms"][a]["dN"] - r["arms"][b]["dN"])
    return out


def report(rows, label, primary):
    print("\n" + "=" * 74)
    print("  %s   n=%d items" % (label, len(rows)))
    print("=" * 74)

    print("\n  d(arm) = dN(arm) - dN(full); + means that corpus CARRIED suppression")
    means = {}
    for arm in ["no-safety", "no-math", "no-persona", "no-wildchat"]:
        ds = d_of(rows, arm)
        if not ds:
            continue
        m, (lo, hi) = mean_ci(ds)
        means[arm] = m
        print("     %-12s mean %+.5f  95%% CI [%+.5f, %+.5f]  %d/%d positive"
              % (arm, m, lo, hi, sum(1 for x in ds if x > 0), len(ds)))

    # ── CLAIM A
    print("\n  CLAIM A  WildChat is at least as important as safety")
    ok_mean = means.get("no-wildchat", 0) >= means.get("no-safety", 0)
    dif = paired(rows, "no-wildchat", "no-safety")
    st = sign_test(dif)
    m, (lo, hi) = mean_ci(dif)
    print("     mean d(no-wildchat) %+.5f  >=  mean d(no-safety) %+.5f   -> %s"
          % (means.get("no-wildchat", float("nan")),
             means.get("no-safety", float("nan")), "YES" if ok_mean else "NO"))
    print("     per-item difference mean %+.5f  95%% CI [%+.5f, %+.5f]" % (m, lo, hi))
    print("     sign test  %d+ / %d-  (%d ties dropped)  p = %.4g"
          % (st["pos"], st["neg"], st["ties"], st["p"]))
    a_ok = ok_mean and st["p"] < 0.05
    print("     ==> %s" % ("SUPPORTED" if a_ok else
                           ("FALSIFIED (safety carries more)" if not ok_mean
                            else "NOT SUPPORTED (sign test p >= 0.05)")))

    # ── CLAIM B
    print("\n  CLAIM B  safety is NOT significantly more important than the others")
    b_ok = True
    for other in ["no-math", "no-persona"]:
        dd = paired(rows, "no-safety", other)
        s = sign_test(dd)
        bigger = "safety" if s["pos"] > s["neg"] else other
        sig = s["p"] < 0.05
        if sig and bigger == "safety":
            b_ok = False
        print("     vs %-11s sign %d+ / %d-  (%d ties)  p = %.4g   %s"
              % (other, s["pos"], s["neg"], s["ties"], s["p"],
                 "SIGNIFICANT, larger = %s" % bigger if sig else "n.s."))

    #: THE NULL IS REPORTED AS A BOUNDED INTERVAL. Failing to reject at n=39 is
    #: not evidence of equality -- registration §2 says so itself -- so the
    #: claim stands only if the interval EXCLUDES the effect WildChat shows.
    dd = paired(rows, "no-safety", "no-math")
    m, (lo, hi) = mean_ci(dd)
    wc_own = means.get("no-wildchat", float("nan"))
    wc_over = mean_ci(paired(rows, "no-wildchat", "no-safety"))[0]
    print("\n     BOUNDING INTERVAL  mean d(no-safety) - d(no-math) = %+.5f" % m)
    print("        95%% CI [%+.5f, %+.5f]" % (lo, hi))
    ex1 = not (lo <= wc_own <= hi)
    ex2 = not (lo <= wc_over <= hi)
    print("        parse (1) WildChat's own effect      %+.5f  -> %s"
          % (wc_own, "EXCLUDED" if ex1 else "INSIDE the interval"))
    print("        parse (2) WildChat over safety       %+.5f  -> %s"
          % (wc_over, "EXCLUDED" if ex2 else "INSIDE the interval"))
    b_final = b_ok and ex1 and ex2
    print("     ==> %s" % ("SUPPORTED under both parses" if b_final else
                           ("NOT SUPPORTED" if not b_ok else
                            "sign tests pass but the interval does NOT exclude "
                            "WildChat's effect under every parse")))
    if primary:
        print("\n  [this is the REGISTERED test set]")
    return {"claim_a": a_ok, "claim_b": b_final, "n": len(rows), "means": means}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        ROOT, "meta/M01_displacement/results/x_slot_ablation_61.json"))
    a = ap.parse_args()

    R = json.load(open(a.results))
    items = R["items"]
    pop = json.load(open(POP))
    pset = {p.strip() for p in pop}

    reg = [r for r in items if r["prompt"].strip() in pset]
    #: ASSERT THE COUNT. A join that silently finds 31 of 39 reports a real
    #: number for the wrong population, and this campaign has booked that.
    if len(reg) != len(pset):
        missing = pset - {r["prompt"].strip() for r in items}
        print("  REFUSING: the frozen population has %d prompts, the results "
              "join %d.\n  Missing from results:" % (len(pset), len(reg)))
        for m in sorted(missing):
            print("     %r" % m[:70])
        return 1

    prov = R.get("provenance", {})
    print("  results   %s" % os.path.relpath(a.results, ROOT))
    print("  items     %d in file, %d joined to the frozen population"
          % (len(items), len(reg)))
    print("  cells     %d cached, %d freshly expanded, twp rule %s"
          % (prov.get("cells_cached", -1), prov.get("cells_expanded", -1),
             prov.get("twp_rule_version", "?")))

    report(reg, "REGISTERED TEST -- the 39 never-run items", True)

    coll, ntw = collapse_twins(reg)
    if ntw >= 2:
        report(coll, "SENSITIVITY -- %d identity twins collapsed to one frame"
               % ntw, False)

    report(items, "ALL 61 -- CONTAINS THE 22 THE CLAIMS WERE DERIVED FROM "
                  "(not an out-of-sample test)", False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
