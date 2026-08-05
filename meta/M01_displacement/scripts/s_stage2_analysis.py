"""Stage 2 analysis. WRITTEN AND HASHED BEFORE THE RUN.

Registrar's condition (iii) at [4697]. Every test, denominator and threshold is
fixed here while the data does not exist, because the practice is the only thing
that made stage 1 and the reversed analysis readable afterwards.

WHAT IS TESTED, from registration_s.md:

  PRIMARY, both conditions required
    1. conjunction `register=B_CONTINUES` AND `pitch=B_MILDER`, FR-RF, positive
    2. the EXCESS of that conjunction over the product of its marginals,
       per (stem, member, order) as j - c*m, FR-RF, positive under the SAME
       sign-flip test. Registrar's objection was that condition 2 as first
       written was a bare inequality: a gate whose refusal criterion had no
       uncertainty attached. It now has one.

  DECLARED STAGE-1-DERIVED, one test
    B_MILDER within B_CONTINUES, FR-RF, positive AND greater than B_MILDER
    within B_GENERIC. Provenance travels with it in every report.

  SECONDARY, seven, nominal p with the count stated.

  SYMMETRIC, three, position bias only, never an effect.

  DECOY, the withdrawal gate.

THE DENOMINATOR RULE, fixed here because it is the one place the conditional
could be silently gamed. For a (stem, member, order) cell, the conditional's
denominator is the number of annotations in that cell whose `register` equals
the arm. A cell with ZERO annotations in that arm is UNDEFINED and dropped, and
the count of dropped cells is printed. It is not zero-filled: zero-filling would
read "no softening" where the truth is "the question never arose", and with
seven coders and a 43/54 arm split some cells will be empty by chance.

IF THE PRIMARY AND THE CONDITIONAL BOTH CONFIRM THEY ARE ONE FINDING. Printed
as one, per registration. They are the same dependence structure seen as a joint
rate and as a between-arm contrast.

THE DECOY GATE IS ONE-DIRECTIONAL AND PRE-COMMITTED. If B_GENERIC fires at the
same rate on non-movers as on risers, the deflation secondary is WITHDRAWN. A
null here cannot be reinterpreted as evidence of anything else, and the
conditional does not depend on it because it contrasts B_MILDER across arms
rather than reading an arm's rate.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
OUT = os.path.join(CAMPAIGN, "results")

REAL = os.path.join(OUT, "s_stage2_real.parquet")
DEC = {"RANDOM": os.path.join(OUT, "s_stage2_decoy_random.parquet"),
       "RANDOM_NL": os.path.join(OUT, "s_stage2_decoy_randomnl.parquet")}
RESULT = os.path.join(OUT, "result_s_stage2.json")

SEED = 20260806
NPERM = 20000
ALPHA = 0.05


def sf(x, seed=SEED, n=NPERM):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), 1.0
    obs = x.mean()
    r = np.random.RandomState(seed)
    null = (r.choice([-1.0, 1.0], size=(n, len(x))) * x).mean(axis=1)
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1)


def cellwise(L, mask):
    """Rate over coders per (order, stem, member), then FR minus RF per stem."""
    s = L.copy()
    s["_x"] = np.asarray(mask, dtype=float)
    w = s.groupby(["order", "stem", "member"])._x.mean().unstack("order").dropna()
    d = (w["FR"] - w["RF"]).values
    o, p = sf(d)
    return dict(fr=float(w["FR"].mean()), rf=float(w["RF"].mean()),
                diff=float(o), p=float(p), n=int(len(d)))


def line(lab, r, pred=None):
    v = ""
    if pred is not None:
        ok = r["p"] < ALPHA and np.sign(r["diff"]) == pred
        v = "  %s" % ("CONFIRMED" if ok else "not confirmed")
    print("  %-40s FR %.3f RF %.3f  diff %+0.3f  p=%.4f%s"
          % (lab, r["fr"], r["rf"], r["diff"], r["p"], v))


def conditional(L, arm):
    """B_MILDER within one register arm. Undefined cells DROPPED, not zeroed."""
    s = L[L.register == arm].copy()
    s["_x"] = (s.pitch == "B_MILDER").astype(float)
    w = s.groupby(["order", "stem", "member"])._x.mean().unstack("order")
    full = len(w)
    w = w.dropna()
    d = (w["FR"] - w["RF"]).values
    o, p = sf(d)
    return dict(diff=float(o), p=float(p), n=int(len(d)),
                dropped=int(full - len(w)), fr=float(w["FR"].mean()),
                rf=float(w["RF"].mean())), w


def main():
    L = pd.read_parquet(REAL.replace("s_stage2_real", "s_stage2_real_long"))
    print("stage 2: %d annotations, %d stems, %d coders"
          % (len(L), L.stem.nunique(), L.coder.nunique()))
    out = {"n": len(L), "stems": int(L.stem.nunique()), "alpha": ALPHA}

    cont = L.register == "B_CONTINUES"
    mild = L.pitch == "B_MILDER"

    print("\n=== PER-ORDER DEPENDENCE, the diagnostic of record ===")
    print("Pooled is retired as a mis-cut: order is the manipulated axis.")
    dep = {}
    for o in ["FR", "RF"]:
        m = L.order == o
        c, p_, j = cont[m].mean(), mild[m].mean(), (cont & mild)[m].mean()
        dep[o] = float(j / (c * p_)) if c * p_ else float("nan")
        print("  %s  P(cont)=%.3f P(mild)=%.3f  product=%.4f  observed=%.4f  ratio=%.2fx"
              % (o, c, p_, c * p_, j, dep[o]))
    out["dependence_per_order"] = dep

    print("\n=== PRIMARY, both conditions required ===")
    r1 = cellwise(L, cont & mild)
    line("1. conjunction", r1, pred=+1)

    #: Condition 2, per (stem, member, order): j - c*m, then FR-RF, sign-flip.
    g = L.groupby(["order", "stem", "member"])
    ex = (g.apply(lambda s: ((s.register == "B_CONTINUES") & (s.pitch == "B_MILDER")).mean()
                  - (s.register == "B_CONTINUES").mean() * (s.pitch == "B_MILDER").mean())
          .unstack("order").dropna())
    e_obs, e_p = sf((ex["FR"] - ex["RF"]).values)
    r2 = dict(fr=float(ex["FR"].mean()), rf=float(ex["RF"].mean()),
              diff=float(e_obs), p=float(e_p), n=int(len(ex)))
    line("2. excess over independence", r2, pred=+1)

    c1 = r1["p"] < ALPHA and r1["diff"] > 0
    c2 = r2["p"] < ALPHA and r2["diff"] > 0
    print("\n  PRIMARY: %s   (condition 1 %s, condition 2 %s)"
          % ("CONFIRMED" if c1 and c2 else "NOT CONFIRMED",
             "met" if c1 else "not met", "met" if c2 else "not met"))
    out["primary"] = dict(conjunction=r1, excess=r2, confirmed=bool(c1 and c2))

    print("\n=== DECLARED STAGE-1-DERIVED: the conditional ===")
    print("Provenance: found at stage 1, not predicted. Reported as derived, always.")
    rc, _ = conditional(L, "B_CONTINUES")
    rg, _ = conditional(L, "B_GENERIC")
    for lab, r in [("B_MILDER within B_CONTINUES", rc), ("B_MILDER within B_GENERIC", rg)]:
        print("  %-40s diff %+0.3f  p=%.4f  n=%d  (%d cells undefined, dropped)"
              % (lab, r["diff"], r["p"], r["n"], r["dropped"]))
    cond_ok = rc["p"] < ALPHA and rc["diff"] > 0 and rc["diff"] > rg["diff"]
    print("  CONDITIONAL: %s" % ("CONFIRMED" if cond_ok else "NOT CONFIRMED"))
    out["conditional"] = dict(within_continues=rc, within_generic=rg, confirmed=bool(cond_ok))

    if (c1 and c2) and cond_ok:
        print("\n  >> PRIMARY AND CONDITIONAL BOTH CONFIRM. THIS IS ONE FINDING.")
        print("     The same dependence seen as a joint rate and as a between-arm")
        print("     contrast. Reporting them as two would be false corroboration.")
    out["one_finding_guard"] = bool((c1 and c2) and cond_ok)

    print("\n=== SECONDARY, seven declared ===")
    SEC = [("register=B_GENERIC", L.register == "B_GENERIC", +1),
           ("register=B_DIFFERENT_REGISTER", L.register == "B_DIFFERENT_REGISTER", +1),
           ("register=B_CONTINUES", cont, None),
           ("more_transgressive", L.more_transgressive == "YES", -1),
           ("pitch=B_MILDER", mild, +1),
           ("pitch=B_STRONGER", L.pitch == "B_STRONGER", -1),
           ("becomes_speech", L.becomes_speech == "YES", +1)]
    out["secondary"] = {}
    for lab, m, pred in SEC:
        r = cellwise(L, m)
        line(lab, r, pred)
        out["secondary"][lab] = dict(r, predicted_sign=pred)
    print("  seven tests declared; that count travels with any reported p.")

    print("\n=== SYMMETRIC: position bias. Never an effect. ===")
    bias = []
    for lab, m in [("related", L.related == "YES"), ("substitutable", L.substitutable == "YES"),
                   ("bare_verb", L.bare_verb == "YES")]:
        r = cellwise(L, m)
        line(lab, r)
        bias.append(abs(r["diff"]))
    print("  mean |bias| %.3f   (R corpus 0.010, stage 1 0.008)" % np.mean(bias))
    out["position_bias"] = float(np.mean(bias))

    print("\n=== DECOY GATE: is B_GENERIC the new CO_ACT? ===")
    print("Pre-committed and one-directional. Comparison is FR only, B slot.")
    real_fr = (L[L.order == "FR"].register == "B_GENERIC").mean()
    print("  B_GENERIC on RISEN words (FR)        %.3f" % real_fr)
    out["decoy"] = {"real_fr": float(real_fr)}
    withdraw = []
    for nm, path in DEC.items():
        lp = path.replace(".parquet", "_long.parquet")
        if not os.path.exists(lp):
            print("  %-36s NOT RUN" % nm)
            continue
        D = pd.read_parquet(lp)
        d_rate = (D.register == "B_GENERIC").mean()
        gap = real_fr - d_rate
        print("  B_GENERIC on NON-MOVERS, %-11s %.3f   gap %+0.3f  (n=%d)"
              % (nm, d_rate, gap, len(D)))
        out["decoy"][nm] = dict(rate=float(d_rate), gap=float(gap), n=int(len(D)))
        withdraw.append(abs(gap) < 0.05)
    if withdraw and all(withdraw):
        print("\n  *** B_GENERIC FIRES EQUALLY ON NON-MOVERS. Per registration, the")
        print("      deflation secondary is WITHDRAWN. This was written before the run.")
    elif withdraw:
        print("\n  B_GENERIC discriminates movers from non-movers; the arm survives.")
    out["deflation_withdrawn"] = bool(withdraw and all(withdraw))

    with open(RESULT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote %s" % RESULT)


if __name__ == "__main__":
    main()
