"""The contradiction lens: where in depth does the frame exit?

    uv run python lens_analysis.py                 # primary + secondaries
    uv run python lens_analysis.py --zh            # the Chinese half, apart
    uv run python lens_analysis.py --csv           # write the tidy tables

Plan: `meta/M02_frame_exit/plans/contradiction_lens.md`. Producer:
`lens_ratio_by_layer.py`. Reads `results/lens_group_layer.jsonl`.

THE MEASURE IS THE PLAN'S, NOT A NEW ONE (§3):

    ratio(L) = JS(AB_L, mean(A_L,B_L)) / min( JS(AB_L,A_L), JS(AB_L,B_L) )

per layer, against a per-layer null built cross-group and content-disjoint.
Parameter-free, and already calibrated on this substrate: **neutralization
1.006, resolution 4.03**. The plan killed a top-k A/B decomposition because its
verdict was a monotone function of k on 63.1% of cells -- "a three-way
classification whose verdict is a monotone function of a free parameter is not
a classification" -- so nothing here introduces a threshold either.

WHAT THE LENS ADDS IS DEPTH AND ONLY DEPTH. The question is not whether
alignment collapses superposition; the JS ratio settled that at the output. It
is WHERE:

    LATE GATE          the pole continuations rise through the stack on the
                       BOTH prompt and are cut in the last layers
    EARLY RE-ROUTING   they never rise; the computation went elsewhere

Different claims about what alignment IS -- a mask over an unchanged
computation, or a changed one -- and a late gate is cheap, reversible and
cosmetic, which is where the political-economic weight sits.

THREE CONSTRAINTS FROM THE PLAN, ENFORCED HERE RATHER THAN REMEMBERED:

  **DEPTH, NEVER LAYER INDEX.** n_layers runs 25 to 49 across this roster, so
  "layer 12" is a different fraction of the stack in every family. Every
  trajectory is interpolated onto one depth grid before anything is compared.

  **NO ABSOLUTE EARLY-LAYER CLAIM** (§5). A lens is a readout in the OUTPUT
  basis; early layers are not in that basis and the raw lens is known to be
  biased there. The base model's own trajectory on the same prompt is the
  paired reference, always -- so the primary is a DIFFERENCE and never a level.

  **THERE IS NO THETA IN THIS INSTRUMENT, AND AN EARLIER VERSION OF THIS FILE
  SAID THERE WAS.** The plan's §8a warns that theta=0.001 floors depths <= 0.25,
  where the tail runs 0.65-0.70 and an interior zero is truncation rather than
  absence. **That describes the PILOT**, which used `twp.expand_layers` at WORD
  level. The producer here is TOKEN level and `layer_probs` returns a full
  normalised softmax over the whole vocabulary -- no threshold, no tail, no
  truncation, and the string "theta" does not occur in it.

  The caveat was imported across instruments, and it had teeth: Secondary 1
  computed the late-gate result after DISCARDING THE BOTTOM HALF OF THE STACK
  on a ground that does not apply to these numbers. Re-run over the full
  stack the conclusion holds and strengthens -- 0.339 of the gap in the top
  eighth against 0.111 for an even spread, 35 of 38 lineages, p=6.7e-08,
  where the truncated version gave 0.489 against 0.200, 33 of 38, p=4.3e-06 --
  but that is luck. `READABLE` is now 0.0 and the band is the whole stack.

  The REAL early-layer caution is the one above and it is not a reason to
  truncate: a lens reads in the output basis, so no ABSOLUTE early-layer level
  is a claim. The paired difference against the model's own base arm already
  handles that, which is why the primary is a difference.

THE UNIT IS THE LINEAGE. Not the model, not the cell. A pair contributes one
observation per group, and the test is over lineages.

ZH IS REPORTED APART, NOT POOLED (§7). It is 47% of the rows and a weaker
instrument on an English-heavy roster.
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

ROWS = os.path.join(CAMP, "results", "lens_group_layer.jsonl")
PAIRS = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
OUT_TRAJ = os.path.join(CAMP, "results", "lens_trajectories.csv")
OUT_DIV = os.path.join(CAMP, "results", "lens_divergence.csv")

#: calibrated on this substrate, `contradiction_ratio_has_no_null.md`
NEUTRALIZATION = 1.006
RESOLUTION = 4.03

#: the common axis. Nine points because the shallowest model has 25 layers, so
#: a finer grid would interpolate more than it measures.
GRID = [i / 8 for i in range(9)]
READABLE = 0.0          #: the whole stack; see the header on why not 0.5


def sign_test(vals):
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return n, k, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


#: A DEGENERACY GUARD ON THE DENOMINATOR, NEVER ON THE RATIO ITSELF.
#: `min(JS(AB,A), JS(AB,B))` goes to zero when the BOTH distribution is
#: indistinguishable from one pole at that layer, and the ratio then diverges:
#: this run holds a cell at 4.5e15 (granite-3.0-8b-base / f11_gender_he, js_min
#: 1.23e-32). Six cells sit under 1e-9 and sixteen exceed |100|.
#:
#: THE GUARD DROPS THE CELL AND COUNTS IT; it does not clip the ratio. A clip
#: would invent a value at exactly the sites where the measure has none, and
#: those sites are informative -- a vanishing denominator IS the BOTH
#: distribution collapsing onto a pole.
#:
#: Medians happen to survive without this, which is why it is stated rather
#: than trusted: a summary that is robust by luck is one bad estimator away
#: from being wrong, and the mean of this column is 2.7e12.
JS_MIN_FLOOR = 1e-6
RATIO_CEILING = 100.0


def load(lang="en"):
    """{(model, group): [(depth, ratio, null_ratio)]}, sorted by depth."""
    traj = collections.defaultdict(list)
    n_all = n_lang = 0
    dropped = collections.Counter()
    for line in open(ROWS):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "ratio" not in r:
            continue
        n_all += 1
        if r["language"] != lang:
            continue
        n_lang += 1
        v = r["ratio"]
        if v is not None:
            if r.get("js_min") is not None and r["js_min"] < JS_MIN_FLOOR:
                dropped["denominator below %g" % JS_MIN_FLOOR] += 1
                v = None
            elif abs(v) > RATIO_CEILING:
                dropped["ratio above %g" % RATIO_CEILING] += 1
                v = None
        traj[(r["model"], r["group"])].append((r["depth"], v, r["null_ratio"]))
    for k in traj:
        traj[k].sort()
    if dropped:
        print("degeneracy guard dropped %d of %d %s rows:"
              % (sum(dropped.values()), n_lang, lang))
        for k, n in dropped.most_common():
            print("    %-28s %d" % (k, n))
    return traj, n_all, n_lang


def interp(points, grid):
    """Linear interpolation onto the common grid; None where unresolvable.

    A model with 25 layers and one with 49 are not comparable at any layer
    index, and taking the nearest layer would give the 49-layer model a
    finer effective grid than the 25-layer one at the same nominal depth.
    """
    xs = [(d, v) for d, v, _ in points if v is not None]
    if len(xs) < 2:
        return [None] * len(grid)
    out = []
    for g in grid:
        if g <= xs[0][0]:
            out.append(xs[0][1] if abs(g - xs[0][0]) < 1e-9 else None)
            continue
        if g >= xs[-1][0]:
            out.append(xs[-1][1] if abs(g - xs[-1][0]) < 1e-9 else None)
            continue
        for i in range(1, len(xs)):
            if xs[i][0] >= g:
                (x0, y0), (x1, y1) = xs[i - 1], xs[i]
                w = 0.0 if x1 == x0 else (g - x0) / (x1 - x0)
                out.append(y0 + w * (y1 - y0))
                break
    return out


def paired(traj):
    """{(lineage, group): (base_curve, aligned_curve)} on the common grid."""
    pairs = [l.strip().split(">") for l in open(PAIRS)]
    out = {}
    for base, aligned in pairs:
        for (m, g) in list(traj):
            if m != base:
                continue
            if (aligned, g) not in traj:
                continue
            out[(base, g)] = (interp(traj[(base, g)], GRID),
                              interp(traj[(aligned, g)], GRID))
    return out


def h_sizes(label, groups):
    sz = collections.Counter(len(v) for v in groups.values())
    print("  %-30s %d groups; sizes %s"
          % (label, len(groups), ", ".join("%dx%d" % (n, s)
                                           for s, n in sorted(sz.items()))))


def primary(P, lang):
    print("=" * 78)
    print("PRIMARY -- LATE GATE or EARLY RE-ROUTING?   (%s)" % lang)
    print("=" * 78)
    print("  d(depth) = ratio_ALIGNED - ratio_BASE, paired on the same group.")
    print("  A LEVEL IS NOT A CLAIM HERE: the lens reads in the output basis and")
    print("  is biased in early layers, so the base arm's own trajectory on the")
    print("  same prompt is the reference at every depth (plan §5).")
    print("\n  calibration on this substrate: neutralization %.3f, resolution %.2f"
          % (NEUTRALIZATION, RESOLUTION))

    bylin = collections.defaultdict(list)
    for (lin, g), (b, a) in P.items():
        bylin[lin].append((b, a))
    print()
    h_sizes("groups per lineage", bylin)
    print("  lineage pairs with both arms: %d   cells: %d"
          % (len(bylin), len(P)))

    print("\n  %6s  %9s %9s  %9s  %7s %-9s %s"
          % ("depth", "base", "aligned", "d(median)", "lins>0", "p", ""))
    rows = []
    for i, g in enumerate(GRID):
        per = collections.defaultdict(list)
        bb, aa = [], []
        for (lin, grp), (b, a) in P.items():
            if b[i] is None or a[i] is None:
                continue
            per[lin].append(a[i] - b[i])
            bb.append(b[i])
            aa.append(a[i])
        if not per:
            print("  %6.3f  %s" % (g, "no data"))
            continue
        lm = [st.median(v) for v in per.values()]
        n, k, p = sign_test(lm)
        flag = "" if g >= READABLE else "  <- theta-floored, not a claim"
        print("  %6.3f  %9.4f %9.4f  %+9.4f   %2d/%-2d %-9.2g%s"
              % (g, st.median(bb), st.median(aa), st.median(lm), k, n, p, flag))
        rows.append({"depth": g, "median_base": st.median(bb),
                     "median_aligned": st.median(aa),
                     "median_d": st.median(lm), "lineages": n,
                     "lineages_pos": k, "p": p,
                     "readable": g >= READABLE})
    return rows


def divergence(P):
    """Where the two arms part company, WITHOUT a threshold.

    A "first depth where the gap exceeds X and stays" rule would put back the
    free parameter the plan removed. Two parameter-free summaries instead:

      argmax depth   the depth at which |aligned - base| is largest for this
                     cell -- late gate predicts it piles up at 0.875-1.0,
                     early re-routing spreads it down the stack
      gap share      the fraction of the total absolute gap that falls in the
                     top quarter of the stack, which is the same question as a
                     continuous quantity rather than a vote
    """
    print("\n" + "=" * 78)
    print("SECONDARY 1 -- WHERE the arms diverge, threshold-free")
    print("=" * 78)
    am, share = collections.defaultdict(list), collections.defaultdict(list)
    for (lin, g), (b, a) in P.items():
        gaps = [(GRID[i], abs(a[i] - b[i]))
                for i in range(len(GRID))
                if b[i] is not None and a[i] is not None and GRID[i] >= READABLE]
        if len(gaps) < 3:
            continue
        tot = sum(v for _, v in gaps)
        if tot <= 0:
            continue
        am[lin].append(max(gaps, key=lambda t: t[1])[0])
        share[lin].append(sum(v for d, v in gaps if d >= 0.875) / tot)
    if not am:
        print("  no cells with enough readable depths")
        return
    print("  computed over the READABLE band (depth >= %.2f) only." % READABLE)
    print("\n  depth of the LARGEST base/aligned gap, per lineage median:")
    hist = collections.Counter(round(st.median(v), 3) for v in am.values())
    for d in sorted(hist):
        print("    %.3f  %s %d" % (d, "#" * hist[d], hist[d]))
    sh = [st.median(v) for v in share.values()]
    print("\n  share of the total gap falling in the TOP EIGHTH (depth >= 0.875):")
    print("    median over %d lineages: %.3f" % (len(sh), st.median(sh)))
    print("    an even spread over the readable band would give ~%.3f"
          % (1 / len([g for g in GRID if g >= READABLE])))
    n, k, p = sign_test([x - 1 / len([g for g in GRID if g >= READABLE]) for x in sh])
    print("    above that even share in %d of %d lineages, p = %.3g" % (k, n, p))
    print("\n  READ THIS AS SHAPE, NOT AS A DATE. The lens cannot say which layer")
    print("  'does' anything -- it says where the output-basis readout of the two")
    print("  arms stops agreeing, which is a fact about the readout as much as")
    print("  about the computation.")


def against_null(traj, lang):
    print("\n" + "=" * 78)
    print("SECONDARY 2 -- the ratio against its OWN per-layer null   (%s)" % lang)
    print("=" * 78)
    print("  'the ratio is 1.2 at depth 0.75' means nothing absolutely. The null")
    print("  is another live group's BOTH distribution scored on this group's")
    print("  poles, AT THAT LAYER.")
    by = collections.defaultdict(lambda: ([], []))
    for (m, g), pts in traj.items():
        for d, r, nr in pts:
            if r is None or nr is None:
                continue
            k = round(d * 8) / 8
            by[k][0].append(r)
            by[k][1].append(nr)
    print("\n  %6s %10s %10s %10s %8s" % ("depth", "ratio", "null", "ratio-null", "n"))
    for d in sorted(by):
        r, nr = by[d]
        flag = "" if d >= READABLE else "   <- theta-floored"
        print("  %6.3f %10.4f %10.4f %+10.4f %8d%s"
              % (d, st.median(r), st.median(nr),
                 st.median(r) - st.median(nr), len(r), flag))


def write_csv(P, rows):
    with open(OUT_TRAJ, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "group", "depth", "ratio_base", "ratio_aligned", "d"])
        for (lin, g), (b, a) in sorted(P.items()):
            for i, d in enumerate(GRID):
                if b[i] is None or a[i] is None:
                    continue
                w.writerow([lin, g, d, "%.6f" % b[i], "%.6f" % a[i],
                            "%.6f" % (a[i] - b[i])])
    with open(OUT_DIV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % v if isinstance(v, float) else v)
                        for k, v in r.items()})
    print("\nwrote %s" % os.path.relpath(OUT_TRAJ, ROOT))
    print("      %s" % os.path.relpath(OUT_DIV, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", action="store_true", help="the Chinese half, apart")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    lang = "zh" if a.zh else "en"
    traj, n_all, n_lang = load(lang)
    print("rows with a ratio: %d total, %d in %s, %d (model, group) cells\n"
          % (n_all, n_lang, lang, len(traj)))
    P = paired(traj)
    rows = primary(P, lang)
    divergence(P)
    against_null(traj, lang)
    if a.csv and rows:
        write_csv(P, rows)


if __name__ == "__main__":
    main()
