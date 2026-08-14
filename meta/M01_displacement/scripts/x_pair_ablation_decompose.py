#!/usr/bin/env python3
"""JS BY ROLE on the base->ablation edges, COMPUTED rather than read.

    meta/M01_displacement/scripts/x_pair_ablation_decompose.py [--limit N]

WHY THIS EXISTS. `movement_cells` declares `js_fall` and `js_rise` and
populates NEITHER -- 0 non-zero values in 568,977 rows, checked across the whole
table. A first run printed +0.00000 for every arm, which reads as "no effect"
and is "no data". RH's original request was for the js measurements, so the
column being empty is not an answer; Cell.decompose() computes the exact
partition from the word_probs store and the request is answerable without it.

WHAT THE PARTITION BUYS OVER THE SCALAR. js_total says HOW MUCH moved.
decompose says WHERE it went, exactly -- the four parts sum to js_total:

    js_fallers   words that FELL          the repression
    js_risers    words that ROSE past the null   the selective uptake
    js_tail      the unresolved residual bin     not a lexical event at all
    js_other     moved, too little to be either  diffuse reshaping

AND `tail_share` DECIDES WHETHER THE SCALAR MEANT ANYTHING. A step whose
divergence is mostly unresolved tail is not doing lexical work, and no amount
of significance on js_total changes that.

STATISTIC. Same within-pair difference-in-differences as the scalar producer,
on the same 684 pairs, so the two tables are comparable row for row:
    within(arm) = arm(MARKED) - arm(UNMARKED);  DiD = within(arm) - within(full)
DiD < 0 = the ablated model withdrew LESS at the transgressive member.
"""
import argparse, statistics as st, sys, importlib.util
sys.path.insert(0, "/Users/rj416/github/malign-logits")
from scipy.stats import binomtest
import numpy as np
from malign_logits.step import Step
from malign_logits.checkpoint import Checkpoint

spec = importlib.util.spec_from_file_location(
    "x", "/Users/rj416/github/malign-logits/meta/M01_displacement/scripts/"
         "x_pair_ablation_split.py")
x = importlib.util.module_from_spec(spec); spec.loader.exec_module(x)

#: The four JS parts, then the mass and diagnostic quantities. `js_other` is
#: carried deliberately: it is the LARGEST part on every arm, so a table
#: reporting only fallers and risers would imply alignment's divergence is
#: mostly identifiable lexical events when most of it is sub-threshold drift.
FIELDS = ["js_total", "js_fallers", "js_risers", "js_tail", "js_other",
          "departed", "arrived", "tail_share", "selectivity", "concentration"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    P = x.load_pairs(); arms = x.load_arms()
    keys = list(P)[:a.limit] if a.limit else list(P)
    if a.limit:
        print("  --limit %d (SMOKE, not the declared population)" % len(keys))
    pre = Checkpoint(x.PRE)

    #: {arm: {role: {pair_key: decompose}}}
    D = {}
    for arm in sorted(arms):
        step = Step(pre, Checkpoint(arms[arm]))
        assert step.is_runnable, "%s not runnable" % arm
        D[arm] = {"MARKED": {}, "UNMARKED": {}}
        for k in keys:
            for role in ("MARKED", "UNMARKED"):
                c = step.cell(P[k][role]["prompt"])
                if not c.is_present:
                    continue
                d = c.decompose()
                if d:
                    D[arm][role][k] = d
        n = min(len(D[arm]["MARKED"]), len(D[arm]["UNMARKED"]))
        print("  %-9s %s cells both roles" % (arm, n))

    rows = []
    for fld in FIELDS:
        #: PER FIELD, not once. `selectivity` is arrived/departed and
        #: `concentration` is a share of `arrived`: both are None where the
        #: denominator is zero, which is a real state (a cell with no flagged
        #: fallers) and not a missing cell. Computing `both` once over all
        #: fields crashed on the first None; computing it per field keeps each
        #: field's own n, and the n is printed so a shrunken one is visible
        #: rather than silently pooled with the others.
        both = [k for k in keys
                if all(k in D[arm][r] and D[arm][r][k].get(fld) is not None
                       for arm in D for r in ("MARKED", "UNMARKED"))]
        if not both:
            print("\n=== %s ===   NO CELLS with this field defined in every arm" % fld)
            continue
        w_full = {k: D["full"]["MARKED"][k][fld] - D["full"]["UNMARKED"][k][fld]
                  for k in both}
        Dn = st.mean(w_full.values())
        print("\n=== %s ===   full SFT within-pair %+.6f  (n=%d, %d pos)"
              % (fld, Dn, len(both), sum(1 for v in w_full.values() if v > 0)))
        for arm in sorted(arms):
            if arm == "full":
                continue
            did = [(D[arm]["MARKED"][k][fld] - D[arm]["UNMARKED"][k][fld]) - w_full[k]
                   for k in both]
            neg = sum(1 for v in did if v < 0); pos = sum(1 for v in did if v > 0)
            p = binomtest(neg, neg + pos, 0.5).pvalue if neg + pos else float("nan")
            m = st.mean(did)
            se = st.stdev(did) / (len(did) ** 0.5) if len(did) > 1 else 0.0
            star = "*" if abs(m) > 1.96 * se else " "
            print("   %-9s MARKED %.5f  UNMARKED %.5f  DiD %+.6f %s  %3d-/%3d+  p=%.4g"
                  % (arm, st.mean(D[arm]["MARKED"][k][fld] for k in both),
                     st.mean(D[arm]["UNMARKED"][k][fld] for k in both),
                     m, star, neg, pos, p))
            rows.append(dict(field=fld, arm=arm, did=m, se=se, n=len(did),
                             n_neg=neg, n_pos=pos, sign_p=p, full_within=Dn))
    import pandas as pd
    out = "/Users/rj416/github/malign-logits/meta/M01_displacement/results/x_pair_ablation_decompose.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n  wrote %s (%d rows)" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
