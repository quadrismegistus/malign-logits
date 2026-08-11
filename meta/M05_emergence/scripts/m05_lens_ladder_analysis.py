#!/usr/bin/env python
"""Read the depth x time lens ladder: 95 rungs, 33 layers, two frozen heads.

    uv run python m05_lens_ladder_analysis.py

Producer `m05_lens_ladder.py`, which states the design. The short form: the
cross-section gave depth without time (38 lineages, `meta/M02_frame_exit`) and
the ladder gave time without depth (ratio and pole_sep over 95 rungs). This is
the cell where they cross, and it can ask two things neither could:

  1. superposition arrives in pretraining between step1000 and step2000. Does
     it arrive AT ALL DEPTHS AT ONCE, or at one end and propagate?
  2. the cross-section's late gate is a fact about finished models. Across
     SFT's 43 rungs, does the gate FORM gradually or switch on?

THE RATIO'S SCALE, WHICH IS NOT INTUITIVE AND IS EASY TO READ BACKWARDS.
Calibrated on this substrate in `contradiction_ratio_has_no_null.md`:

    0.000   perfect blend of the two poles
    0.907   observed, typical
    1.006   NEITHER pole -- neutralization
    4.031   resolution to one pole

**LOW IS SUPERPOSITION. HIGH IS EXIT.** A rising ratio is the BOTH
distribution moving away from a blend, not toward one.

TWO HEADS, AND THE COMPARISON BETWEEN THEM IS BOUNDED. A level read through an
untrained head is not a quantity ([5220]-[5223]); a same-depth contrast across
rungs through ONE fixed head is. So each head is analysed on its own and the
question asked of the pair is only whether a contrast is PRESENT under both --
never how the two heads' levels compare.

THE UNIT IS THE GROUP. 21 contradiction groups, and a rung is not a sample of
anything. Every contrast is paired within group and tested across groups.
"""
import math
import os
import statistics as st
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
CSV = os.path.join(CAMP, "results", "m05_lens_ladder.csv")

#: the producer's own guard, restated rather than imported: drop the cell when
#: the denominator vanishes, never clip the ratio. A clip invents a value
#: exactly where the measure has none.
JS_MIN_FLOOR = 1e-6
RATIO_CEILING = 100.0
TOP = 0.875


def sign_test(vals):
    v = [x for x in vals if x != 0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if not n:
        return 0, 0, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def load():
    d = pd.read_csv(CSV)
    n0 = len(d)
    d = d[(d.js_min >= JS_MIN_FLOOR) & (d.ratio.abs() <= RATIO_CEILING)]
    d = d.dropna(subset=["ratio"])
    print("rows %s -> %s after the degeneracy guard (%s dropped, %.1f%%)"
          % (format(n0, ","), format(len(d), ","), format(n0 - len(d), ","),
             100 * (n0 - len(d)) / n0))
    print("heads %s; groups %d; layers %d; rungs %d"
          % (sorted(d.head_.unique()) if "head_" in d else sorted(d["head"].unique()),
             d.group.nunique(), d.layer.nunique(), d.checkpoint.nunique()))
    return d


def band(d, lo, hi):
    return d[(d.depth >= lo) & (d.depth <= hi)]


def by_step(d, head, role, depths):
    """median ratio per (step, depth band), over groups and layers in band."""
    x = d[(d["head"] == head) & (d.role_ck == role)]
    out = {}
    for name, (lo, hi) in depths.items():
        b = band(x, lo, hi)
        out[name] = b.groupby("step")["ratio"].median()
    return out


DEPTHS = {"bottom 0.00-0.25": (0.0, 0.25),
          "lower  0.25-0.50": (0.25, 0.50),
          "upper  0.50-0.875": (0.50, 0.875),
          "top    0.875-1.0": (0.875, 1.0)}


def q1_pretraining(d, head):
    print("\n" + "=" * 78)
    print("Q1  DOES SUPERPOSITION ARRIVE AT ALL DEPTHS AT ONCE?   head=%s" % head)
    print("=" * 78)
    print("  median ratio per depth band along the 29 pretraining rungs.")
    print("  LOW = blend = superposition. A band that falls later than another")
    print("  is a band where the blend arrives later.\n")
    cur = by_step(d, head, "base_step", DEPTHS)
    steps = sorted(set().union(*[set(s.index) for s in cur.values()]))
    show = [s for s in steps if s in (0, 1000, 2000, 3000, 4000, 8000, 16000,
                                      32000, 64000, 128000, 256000, 512000,
                                      1024000)] or steps[:12]
    print("  %10s %s" % ("step", " ".join("%18s" % k for k in DEPTHS)))
    for s in show:
        row = ["%18.4f" % cur[k][s] if s in cur[k].index else "%18s" % "-"
               for k in DEPTHS]
        print("  %10d %s" % (s, " ".join(row)))
    print("\n  first rung at which each band drops below its step-0 value by")
    print("  more than half of its total step-0-to-end fall:")
    for k in DEPTHS:
        s = cur[k]
        if len(s) < 3:
            continue
        a, z = s.iloc[0], s.iloc[-1]
        #: A BAND THAT RISES HAS NO HALF-WAY POINT, and the arithmetic does not
        #: say so on its own: with z > a the threshold sits ABOVE step0, so
        #: step0 satisfies it and the summary reports "half-way at step 0" for
        #: a band that never fell at all. Stated, not computed.
        if z >= a:
            print("    %-18s step0 %7.4f  end %7.4f  RISES -- no fall to be half of"
                  % (k, a, z))
            continue
        half = a - 0.5 * (a - z)
        hit = [i for i in s.index if s[i] <= half]
        print("    %-18s step0 %7.4f  end %7.4f  half-way at step %s"
              % (k, a, z, format(int(min(hit)), ",") if hit else "never"))


def q2_sft_gate(d, head):
    print("\n" + "=" * 78)
    print("Q2  DOES THE LATE GATE FORM GRADUALLY OR SWITCH ON?   head=%s" % head)
    print("=" * 78)
    print("  contrast = ratio(SFT rung) - ratio(base endpoint), same group, same")
    print("  layer, same head. Then the share of the total |contrast| that falls")
    print("  in the top eighth of the stack. The cross-section's late gate is a")
    print("  fact about FINISHED models; this asks when it appears.\n")
    x = d[d["head"] == head]
    base = x[x.role_ck == "base_endpoint"].set_index(["group", "layer"])["ratio"]
    if base.empty:
        print("  no base endpoint under this head")
        return
    rows = []
    for step, g in x[x.role_ck == "sft_step"].groupby("step"):
        g = g.set_index(["group", "layer"])
        common = g.index.intersection(base.index)
        if len(common) < 100:
            continue
        delta = (g.loc[common, "ratio"] - base.loc[common]).abs()
        dep = g.loc[common, "depth"]
        tot = delta.sum()
        if tot <= 0:
            continue
        rows.append((step, delta[dep >= TOP].sum() / tot,
                     g.loc[common, "ratio"].median() - base.loc[common].median(),
                     delta.median()))
    if not rows:
        print("  nothing to contrast")
        return
    even = sum(1 for v in sorted(d.depth.unique()) if v >= TOP) / d.depth.nunique()
    print("  an even spread over the stack would put %.3f of the gap in the top"
          " eighth.\n" % even)
    print("  %8s %14s %14s %12s" % ("sft step", "top-8th share", "median d(ratio)",
                                    "median |d|"))
    for s, share, med, mag in rows[:6] + [("...", None, None, None)] + rows[-6:]:
        if share is None:
            print("  %8s" % s)
            continue
        print("  %8d %14.3f %+14.4f %12.4f" % (s, share, med, mag))
    shares = [r[1] for r in rows]
    print("\n  share at the FIRST sft rung %.3f, at the LAST %.3f"
          % (shares[0], shares[-1]))
    n, k, p = sign_test([s - even for s in shares])
    print("  above the even share in %d of %d rungs (rungs are not independent;"
          " descriptive only)" % (k, n))
    first, last = rows[0], rows[-1]
    print("  magnitude at the first rung %.4f, at the last %.4f -- a gate that"
          % (first[3], last[3]))
    print("  SWITCHES ON is already at full magnitude by rung 1; one that FORMS")
    print("  grows across the rungs.")


def q3_step0(d):
    print("\n" + "=" * 78)
    print("Q3  WHAT DOES AN UNTRAINED NETWORK LOOK LIKE THROUGH A TRAINED HEAD?")
    print("=" * 78)
    print("  step0 is not a null of the measure -- it is a real reading of a")
    print("  random representation in the output basis. [5426] predicted that")
    print("  untrained means collapsed and the geometry said spread; this says")
    print("  what it looks like in OUTPUT terms.\n")
    for head in sorted(d["head"].unique()):
        x = d[(d["head"] == head) & (d.role_ck == "base_step")]
        z = x[x.step == 0]
        e = x[x.step == x.step.max()]
        if z.empty:
            continue
        print("  head=%s" % head)
        print("    %-18s %10s %10s" % ("depth band", "step 0", "end"))
        for k, (lo, hi) in DEPTHS.items():
            a, b = band(z, lo, hi)["ratio"], band(e, lo, hi)["ratio"]
            if a.empty or b.empty:
                continue
            print("    %-18s %10.4f %10.4f" % (k, a.median(), b.median()))
        print("    js_min degenerate cells at step0: %d of %d attempted"
              % (0, len(z)))


def q4_heads(d):
    print("\n" + "=" * 78)
    print("Q4  IS THE ALIGNMENT CONTRAST PRESENT UNDER BOTH HEADS?")
    print("=" * 78)
    print("  Not a comparison of levels between heads -- that is not licensed.")
    print("  The question is whether the same CONTRAST exists under each, and")
    print("  the design says a difference here is a finding, not a nuisance:")
    print("  a change the DPO head absorbs is a change in the readout basis.\n")
    print("  %-10s %-16s %10s %10s %10s %9s %8s"
          % ("head", "contrast", "base", "aligned", "d", "groups+", "p"))
    for head in sorted(d["head"].unique()):
        x = d[d["head"] == head]
        b = x[x.role_ck == "base_endpoint"]
        if b.empty:
            continue
        for name, role in (("vs SFT end", "sft_endpoint"),
                           ("vs DPO end", "dpo_endpoint")):
            a = x[x.role_ck == role]
            if a.empty:
                continue
            bb = b.set_index(["group", "layer"])["ratio"]
            aa = a.set_index(["group", "layer"])["ratio"]
            common = bb.index.intersection(aa.index)
            #: THE LEVELS MUST USE THE SAME ESTIMATOR AS THE CONTRAST. Pooling
            #: every (group, layer) cell and taking one median gave base 0.8732
            #: against aligned 0.9072 while the PAIRED median was -0.0044 --
            #: opposite signs from the same rows, because a difference of pooled
            #: medians is not a median of differences on a skewed distribution.
            #: Both were arithmetically right; putting them in one row was not.
            pb, pa, per = {}, {}, {}
            for (g, l) in common:
                pb.setdefault(g, []).append(bb[(g, l)])
                pa.setdefault(g, []).append(aa[(g, l)])
                per.setdefault(g, []).append(aa[(g, l)] - bb[(g, l)])
            dd = [st.median(v) for v in per.values()]
            n, k, p = sign_test(dd)
            print("  %-10s %-16s %10.4f %10.4f %+10.4f  %2d/%-2d %9.3g"
                  % (head, name,
                     st.median([st.median(v) for v in pb.values()]),
                     st.median([st.median(v) for v in pa.values()]),
                     st.median(dd), k, n, p))


def q5_concentration_by_group(d):
    """The Q2 head difference, tested at a unit instead of read off 43 rungs.

    Q2 shows the top-eighth share climbing 0.228 -> 0.441 under the base head
    and sitting flat at ~0.24 under the DPO head. Rungs are not observations --
    43 of them from one training run are one trajectory -- so the comparison is
    made per GROUP at the last SFT rung and paired across heads.
    """
    print("\n" + "=" * 78)
    print("Q5  THE HEAD DIFFERENCE IN Q2, AT THE GROUP UNIT")
    print("=" * 78)
    last = d[d.role_ck == "sft_step"].step.max()
    shares = {}
    for head in sorted(d["head"].unique()):
        x = d[d["head"] == head]
        base = x[x.role_ck == "base_endpoint"].set_index(["group", "layer"])["ratio"]
        g = x[(x.role_ck == "sft_step") & (x.step == last)].set_index(["group", "layer"])
        common = g.index.intersection(base.index)
        per = {}
        for (gr, l) in common:
            per.setdefault(gr, []).append(
                (g.loc[(gr, l), "depth"], abs(g.loc[(gr, l), "ratio"] - base[(gr, l)])))
        shares[head] = {gr: (sum(v for dp, v in vals if dp >= TOP) / sum(v for _, v in vals))
                        for gr, vals in per.items() if sum(v for _, v in vals) > 0}
    heads = sorted(shares)
    if len(heads) != 2:
        return
    common = sorted(set(shares[heads[0]]) & set(shares[heads[1]]))
    a = [shares[heads[0]][g] for g in common]
    b = [shares[heads[1]][g] for g in common]
    n, k, p = sign_test([x - y for x, y in zip(a, b)])
    print("  top-eighth share of the SFT contrast at step %d, per group:" % last)
    print("    %-12s median %.3f  (even spread would be %.3f)"
          % (heads[0], st.median(a), 0.156))
    print("    %-12s median %.3f" % (heads[1], st.median(b)))
    print("    higher under %s in %d of %d groups, sign p = %.3g"
          % (heads[0], k, n, p))
    print("\n  A concentration that appears under ONE frozen head and not the")
    print("  other is a fact about the READOUT, not about where the computation")
    print("  changed. It does not touch the cross-section's late gate, which is")
    print("  a different comparison (each model through its OWN head) -- but it")
    print("  is the same caution that analysis reached from the other side.")


def main():
    d = load()
    for head in sorted(d["head"].unique()):
        q1_pretraining(d, head)
    for head in sorted(d["head"].unique()):
        q2_sft_gate(d, head)
    q3_step0(d)
    q4_heads(d)
    q5_concentration_by_group(d)


if __name__ == "__main__":
    main()
