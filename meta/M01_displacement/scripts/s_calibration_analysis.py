"""Does the S instrument BEHAVE? Calibration only -- no hypothesis is tested.

WRITTEN BEFORE ANY FIELD VALUE WAS READ. The run finished first, so I checked
parse counts and nothing else before writing this. That matters here more than
usual: the whole value of a calibration pass is the thresholds, and thresholds
chosen after seeing the rates are not thresholds.

THE FOUR CHECKS AND THEIR STATED FAILURE CONDITIONS, fixed in the post that
commissioned this run:

  CEILING/FLOOR   any field above 90% or below 2% -- the two ways R died. CO_ACT
                  took 59% and scored 100% on words that never moved; METONYMY,
                  EUPHEMISM and AFFECT together took 0.5%.
  DIRECTIONALITY  a field designed to move with order whose FR-minus-RF sits
                  below the position-bias floor is not directional in practice,
                  whatever its description claims, and silently loses its
                  control.
  CHAINING        `bare_verb = YES` items answering NO or NOT_APPLICABLE on
                  everything else more than half the time means the anti-chaining
                  fix failed. Three readers independently predicted this failure
                  from revision 1's examples.
  FRAGILITY       coders splitting three ways on more than ~15% of items means
                  the field inherits R's relation-axis fragility, where one pair
                  drew four different labels from four coders.

AND ONE COMPARISON THAT IS THE POINT OF THE WHOLE REDESIGN. These 50 stems carry
R's answers already. If operations R buried in CO_ACT light up under S on the
SAME items, the "unreachable label" diagnosis is confirmed on the material that
produced it rather than argued from a rate difference across instruments.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
OUT = os.path.join(CAMPAIGN, "results")
LONG = os.path.join(OUT, "s_calibration_long.parquet")
RLONG = os.path.join(OUT, "r_decoy_compare_long.parquet")

TRI = ["bare_verb", "related", "more_transgressive", "substitutable", "act_lands",
       "internalised", "becomes_speech", "knowing_deflated", "blank_discloses"]
#: declared in the schema docstring, before this run
SYMMETRIC = ["related", "substitutable"]
DIRECTIONAL = ["act_lands", "internalised", "becomes_speech", "knowing_deflated",
               "blank_discloses", "more_transgressive"]


def main():
    L = pd.read_parquet(LONG)
    print("annotations %d   items %d   coders %d   orders %s"
          % (len(L), L.groupby(["stem", "member", "order"]).ngroups,
             L.coder.nunique(), sorted(L.order.unique())))

    print("\n=== 1. CEILING / FLOOR   (fail: YES-rate >90% or <2%) ===")
    print("  %-20s %7s %7s %7s   %s" % ("field", "YES", "NO", "N/A", "verdict"))
    for f in TRI:
        y = (L[f] == "YES").mean(); n = (L[f] == "NO").mean()
        na = (L[f] == "NOT_APPLICABLE").mean()
        v = "CEILING" if y > .90 else ("FLOOR" if y < .02 else "ok")
        print("  %-20s %6.1f%% %6.1f%% %6.1f%%   %s" % (f, 100*y, 100*n, 100*na, v))
    print("  %-20s %s" % ("pitch", dict(L.pitch.value_counts(normalize=True).round(3))))

    print("\n=== 2. DIRECTIONALITY   (per stem-member, FR minus RF) ===")
    print("  symmetric fields estimate the position-bias floor; directional")
    print("  fields must clear it.")
    def diff(col, val="YES"):
        s = L.copy(); s["_x"] = s[col] == val
        w = s.groupby(["order", "stem", "member"])._x.mean().unstack("order").dropna()
        return (w["FR"] - w["RF"]).mean(), len(w)
    floor = []
    for f in SYMMETRIC:
        d, n = diff(f); floor.append(abs(d))
        print("  %-20s %+7.3f   SYMMETRIC, expected ~0" % (f, d))
    fl = float(np.mean(floor))
    print("  position-bias floor from symmetric fields: %.3f" % fl)
    print()
    for f in DIRECTIONAL:
        d, n = diff(f)
        print("  %-20s %+7.3f   %s" % (f, d, "clears floor" if abs(d) > fl else "BELOW FLOOR"))
    for v in ["B_MILDER", "B_STRONGER"]:
        d, n = diff("pitch", v)
        print("  %-20s %+7.3f   %s" % ("pitch=" + v, d, "clears floor" if abs(d) > fl else "BELOW FLOOR"))

    print("\n=== 3. CHAINING   (fail: >50% of bare_verb=YES rows are otherwise empty) ===")
    bv = L[L.bare_verb == "YES"]
    others = [f for f in TRI if f != "bare_verb"]
    empty = bv.apply(lambda r: all(r[f] != "YES" for f in others)
                     and r.pitch == "NOT_APPLICABLE", axis=1)
    print("  bare_verb=YES annotations: %d of %d (%.1f%%)" % (len(bv), len(L), 100*len(bv)/len(L)))
    if len(bv):
        print("  of those, NOTHING else answered YES and pitch N/A: %d (%.1f%%)   %s"
              % (empty.sum(), 100*empty.mean(),
                 "CHAINING SURVIVED" if empty.mean() > .5 else "fix held"))
        print("  what DOES still fire on bare_verb=YES rows:")
        for f in others:
            r = (bv[f] == "YES").mean()
            if r > 0: print("      %-20s %5.1f%%" % (f, 100*r))

    print("\n=== 4. FRAGILITY   (fail: coders split 3 ways on >15% of items) ===")
    print("  %-20s %10s %10s" % ("field", "3-way", "2-way"))
    for f in TRI + ["pitch"]:
        g = L.groupby(["stem", "member", "order"])[f].nunique()
        print("  %-20s %9.1f%% %9.1f%%" % (f, 100*(g >= 3).mean(), 100*(g == 2).mean()))

    print("\n=== 5. DID S REACH WHAT R BURIED IN CO_ACT? ===")
    R = pd.read_parquet(RLONG)
    R = R[R.arm == "REAL"]
    rc = R.groupby(["stem", "member"]).relation.apply(lambda s: s.value_counts().index[0])
    S = L[L.order == "FR"].copy()
    S["r_label"] = [rc.get((r.stem, r.member)) for r in S.itertuples()]
    co = S[S.r_label == "CO_ACT"]
    print("  items R's majority called CO_ACT: %d of %d"
          % (co.groupby(["stem", "member"]).ngroups, S.groupby(["stem", "member"]).ngroups))
    print("  on those same items, S fires:")
    for f in ["becomes_speech", "internalised", "act_lands", "knowing_deflated",
              "blank_discloses", "substitutable", "more_transgressive"]:
        print("      %-20s %5.1f%% YES" % (f, 100*(co[f] == "YES").mean()))
    print("  R's three near-dead labels on these items, for comparison:")
    for lab in ["METONYMY", "EUPHEMISM", "AFFECT"]:
        print("      %-20s %5.2f%% of R annotations" % (lab, 100*(R.relation == lab).mean()))


if __name__ == "__main__":
    main()
