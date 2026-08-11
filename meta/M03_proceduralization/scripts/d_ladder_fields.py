"""Plan D, the content half: semantic fields on the arms, along the ladder.

    uv run python d_ladder_fields.py --run
    uv run python d_ladder_fields.py --report

The magnitude half (`d_ladder.py`) is dead on this lineage: pooling both
collections gives an arm effect of -0.0060 bits, 14 of 30 scenarios positive,
p=0.86, with an item sd of 0.116 that is twice the difference being chased.
**That is a result about HOW FAR and says nothing about TOWARD WHAT.**

Composition is a different quantity with different properties. A JS is one
scalar per cell; a field share sums over every word that moved, so the
per-cell estimate is built from more information and may be steadier across
scenarios even where the scalar is not. It may equally not be -- this script
exists to find out, and the unit question below is the one that decides it.

## THE UNIT IS STILL THE SCENARIO, AND THAT IS THE WHOLE LESSON OF TODAY

`d_ladder.py` reported p=9.1e-14 by sign-testing 50 RUNGS, which are 50
correlated snapshots of 12 or 18 scenarios, not 50 observations. The ICC of
the paired difference across rungs is 0.85: a scenario's value is fixed from
the first alignment checkpoint to the last. **So every test here is over
SCENARIOS, and the rung axis is read for SHAPE only.**

## WHAT IS COMPUTED

Per (rung, prompt): the risers from the anchor to that rung under
`movement.CANONICAL` -- risers tested against the renormalisation null,
fallers a bare ratio rule, and that asymmetry never narrated away. Then
`fields.count` over every lexicon `fields.available()` reports, at both
granularities, with COVERAGE on every row.

Per (rung, scenario, field): d = share(inst) - share(indiv), paired.

## COVERAGE IS NOT DECORATION AND ONE SOURCE ALREADY BURNED US

On plan B's population RID produced the LARGEST field differences at 40%
coverage -- a composition over the small non-random subset RID happens to
know. I then over-corrected and discarded the result entirely, when coverage
was 0.400 against 0.429 ACROSS ARMS, near-identical, so it could not bias the
contrast. Coverage limits what a row GENERALISES to; it does not invalidate a
within-arm comparison. Both halves of that are printed.
"""
import argparse
import collections
import csv
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from d_ladder import labels, rungs, ANCHOR   # noqa: E402

OUT = os.path.join(CAMP, "results", "d_ladder_fields.csv")


def sign_test(vals):
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return n, k, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def flat(f):
    for src, v in f.items():
        if src == "norms":
            for norm, w in v.items():
                for k, n in w.get("counts", {}).items():
                    yield "norms:" + norm, k, n
        else:
            for k, n in v.get("counts", {}).items():
                yield src, k, n


def cov(f):
    for src, v in f.items():
        if src == "norms":
            for norm, w in v.items():
                yield "norms:" + norm, w.get("coverage")
        else:
            yield src, v.get("coverage")


def run():
    from malign_logits.movement import word_probs, movement, CANONICAL
    from b_twp_institutional import field_counts
    L = labels()
    R = rungs()
    base = {}
    for t in L:
        wp = word_probs(ANCHOR, t)
        if wp is not None:
            base[t] = wp
    print("anchor covers %d of %d texts; %d rungs" % (len(base), len(L), len(R)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "role_ck", "step", "stratum", "arm",
                    "scenario", "source", "field", "share", "coverage",
                    "n_risers"])
        for key, role, step, rev in R:
            got = 0
            for t, (arm, scen, stratum) in sorted(L.items()):
                if t not in base:
                    continue
                wp = word_probs(key, t)
                if wp is None:
                    continue
                #: anchor -> rung, so a riser is a word this checkpoint
                #: promotes relative to where pretraining ended.
                mv = movement(base[t].probs, wp.probs, CANONICAL,
                              residual_pre=base[t].residual,
                              residual_post=wp.residual)
                if not mv.risers:
                    continue
                fc = field_counts(mv.risers)
                tot = collections.Counter()
                for src, fld, c in flat(fc):
                    tot[src] += c
                cvg = dict(cov(fc))
                for src, fld, c in flat(fc):
                    if not tot[src]:
                        continue
                    w.writerow([key, role, step, stratum, arm, scen, src, fld,
                                "%.6g" % (c / tot[src]),
                                "" if cvg.get(src) is None else "%.4f" % cvg[src],
                                len(mv.risers)])
                    n += 1
                got += 1
            print("  %-44s %-14s %3d cells" % (key.split("/")[-1][:42], role, got),
                  flush=True)
    print("\nwrote %d rows -> %s" % (n, os.path.relpath(OUT, ROOT)))


def report(top=14):
    import pandas as pd
    D = pd.read_csv(OUT)
    AL = ("sft_step", "sft_endpoint", "dpo_endpoint", "rlvr_step")
    A = D[D.role_ck.isin(AL)]
    print("=" * 78)
    print("FIELDS ON THE ARMS, ALIGNMENT RUNGS, UNIT = SCENARIO")
    print("=" * 78)
    print("  d = share(inst) - share(indiv), paired within scenario, averaged")
    print("  over alignment rungs, then sign-tested over SCENARIOS. The rung")
    print("  axis is not an observation axis: ICC of the paired difference")
    print("  across rungs is 0.85.")
    for strat in ("f21_inst", "m03_slice", None):
        S = A if strat is None else A[A.stratum == strat]
        lab = strat or "BOTH POOLED"
        P = S.pivot_table(index=["stratum", "scenario", "source", "field"],
                          columns="arm", values="share")
        P = P.dropna().reset_index()
        P["d"] = P["inst"] - P["indiv"]
        rows = []
        for (src, fld), g in P.groupby(["source", "field"]):
            if len(g) < 8:
                continue
            n, k, p = sign_test(list(g.d))
            rows.append((st.median(g.d), n, k, p, src, fld,
                         float(A[A.source == src].coverage.median())))
        if not rows:
            print("\n  %s: nothing with >=8 scenarios" % lab)
            continue
        rows.sort(reverse=True)
        print("\n  --- %s (%d scenarios) ---" % (lab, P.scenario.nunique()))
        print("  %-34s %+9s %7s %-9s %s" % ("field", "d", "scen>0", "p", "cov"))
        for r in rows[:top // 2] + rows[-(top // 2):]:
            print("  %-34s %+9.4f  %2d/%-2d %-9.2g %.2f"
                  % ("%s/%s" % (r[4], r[5]), r[0], r[2], r[1], r[3], r[6]))
        sig = [r for r in rows if r[3] < 0.05]
        print("  fields tested %d, p<0.05 %d, expected by chance %.1f"
              % (len(rows), len(sig), 0.05 * len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.report:
        report()
    if not (a.run or a.report):
        ap.print_help()


if __name__ == "__main__":
    main()
