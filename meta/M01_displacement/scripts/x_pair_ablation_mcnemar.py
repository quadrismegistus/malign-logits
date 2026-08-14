import importlib.util, json, statistics as st
from scipy.stats import fisher_exact, binomtest
spec = importlib.util.spec_from_file_location("x", "x_pair_ablation_split.py")
x = importlib.util.module_from_spec(spec); spec.loader.exec_module(x)

def cells(base, aligned, met):
    sql = ("SELECT prompt, %s AS v FROM malign_logits.movement_cells "
           "WHERE base='%s' AND aligned='%s' FORMAT JSONEachRow" % (met, base, aligned))
    d = {}
    for line in x.q(sql).strip().split("\n"):
        if line.strip():
            r = json.loads(line); d[r["prompt"]] = float(r["v"])
    return d

P = x.load_pairs(); arms = x.load_arms()
pl = [(P[k]["MARKED"]["prompt"], P[k]["UNMARKED"]["prompt"]) for k in P]

for met in ["departed", "arrived", "js_total"]:
    full = cells(x.PRE, x.FULL_SFT, met)
    print("\n=== %s ===" % met)
    for arm in ["full"] + [a for a in sorted(arms) if a != "full"]:
        ab = cells(x.PRE, arms[arm], met)
        ok = [(m,u) for m,u in pl if m in ab and u in ab and m in full and u in full]
        pos = sum(1 for m,u in ok if ab[m]-ab[u] > 0)
        print("   %-9s %3d/%d reacted (MARKED>UNMARKED)  = %.1f%%" % (arm, pos, len(ok), 100*pos/len(ok)))

    print("   -- tests, each arm vs full, on the SAME pairs --")
    for arm in [a for a in sorted(arms) if a != "full"]:
        ab = cells(x.PRE, arms[arm], met)
        ok = [(m,u) for m,u in pl if m in ab and u in ab and m in full and u in full]
        fp = [(full[m]-full[u]) > 0 for m,u in ok]
        ap = [(ab[m]-ab[u]) > 0 for m,u in ok]
        a11 = sum(1 for f,g in zip(fp,ap) if f and g)
        b   = sum(1 for f,g in zip(fp,ap) if f and not g)   # full pos, arm neg
        c   = sum(1 for f,g in zip(fp,ap) if not f and g)   # full neg, arm pos
        d   = sum(1 for f,g in zip(fp,ap) if not f and not g)
        # Fisher: treats the two rows as INDEPENDENT samples (they are not)
        fo, fpv = fisher_exact([[sum(fp), len(fp)-sum(fp)], [sum(ap), len(ap)-sum(ap)]])
        # McNemar exact: the correct paired test, on the discordant cells only
        mp = binomtest(b, b+c, 0.5).pvalue if b+c else float("nan")
        print("      %-9s  b=%3d c=%3d (discordant %3d)   McNemar p=%-8.4g   Fisher p=%-8.4g  OR=%.3f"
              % (arm, b, c, b+c, mp, fpv, fo))
