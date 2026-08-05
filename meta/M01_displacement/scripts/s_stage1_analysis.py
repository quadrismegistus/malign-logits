"""Stage 1 analysis. WRITTEN AND COMMITTED BEFORE THE DATA LANDED.

Stage 1's job is NOT to find an effect. It is to answer a question the six smoke
items cannot: what is the base rate of the primary conjunction, and can 255
stems detect it? A joint prediction with no base rate cannot be powered, and
finding that out on the held-out set means finding it out after 7,140 calls.

THE PRIMARY, from registration_s.md, both conditions required:

  1. `register = B_CONTINUES` AND `pitch = B_MILDER` together, FR minus RF,
     positive.
  2. That conjunction exceeds the product of its two marginals.

Condition 2 is what makes it a test of displacement rather than of its parts. A
conjunction can rise entirely because one component rose; positive DEPENDENCE
between staying-in-register and going-milder is the Freudian claim. The
marginals alone are consistent with alignment simply lowering intensity
everywhere, which is what the R corpus already showed happening on both the
contiguity and the similarity axis.

WHAT STAGE 1 MAY AND MAY NOT DO. It may report that the instrument is unusable,
that the conjunction is too rare to power, or that the prediction looks wrong.
It may NOT revise the prediction. The registration was committed at 411f6640
before this ran, and a pilot that rewrites its own hypothesis is not a pilot --
that is precisely how R's +0.235 became a design input for a study it then
failed.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
OUT = os.path.join(CAMPAIGN, "results")
LONG = os.path.join(OUT, "s_stage1_50_rev3.parquet")
RESULT = os.path.join(OUT, "result_s_stage1.json")

SEED = 20260806
NPERM = 20000
N_STAGE2 = 255


def sf(x, seed=SEED, n=NPERM):
    x = np.asarray(x, float)
    obs = x.mean()
    r = np.random.RandomState(seed)
    null = (r.choice([-1.0, 1.0], size=(n, len(x))) * x).mean(axis=1)
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1), x.std(ddof=1)


def diff(L, mask):
    s = L.copy()
    s["_x"] = mask.values if hasattr(mask, "values") else mask
    w = s.groupby(["order", "stem", "member"])._x.mean().unstack("order").dropna()
    d = (w["FR"] - w["RF"]).values
    o, p, sd = sf(d)
    return dict(fr=float(w["FR"].mean()), rf=float(w["RF"].mean()),
                diff=float(o), p=float(p), sd=float(sd), n=int(len(d)))


def line(lab, r, pred=None):
    flag = ""
    if pred is not None and r["p"] < 0.05:
        flag = "  sign %s" % ("AS PREDICTED" if np.sign(r["diff"]) == pred else "AGAINST PREDICTION")
    print("  %-34s FR %.3f RF %.3f  diff %+0.3f  p=%.4f%s"
          % (lab, r["fr"], r["rf"], r["diff"], r["p"], flag))


def main():
    L = pd.read_parquet(LONG)
    print("stage 1: %d annotations, %d stems, %d coders"
          % (len(L), L.stem.nunique(), L.coder.nunique()))
    out = {"n": len(L), "stems": int(L.stem.nunique()), "coders": sorted(L.coder.unique())}

    cont = L.register == "B_CONTINUES"
    mild = L.pitch == "B_MILDER"
    both = cont & mild

    print("\n=== BASE RATES, the reason stage 1 exists ===")
    for lab, m in [("register=B_CONTINUES", cont), ("pitch=B_MILDER", mild),
                   ("CONJUNCTION", both)]:
        print("  %-34s %5.1f%% of %d annotations" % (lab, 100 * m.mean(), len(L)))
    exp_ind = cont.mean() * mild.mean()
    print("  expected if independent            %5.1f%%" % (100 * exp_ind))
    print("  observed / expected                %5.2fx   <- >1 is positive dependence"
          % (both.mean() / exp_ind if exp_ind else float("nan")))
    out["base_rates"] = dict(continues=float(cont.mean()), milder=float(mild.mean()),
                             conjunction=float(both.mean()), expected_independent=float(exp_ind))

    print("\n=== PRIMARY: the displacement conjunction ===")
    r_both = diff(L, both)
    r_cont = diff(L, cont)
    r_mild = diff(L, mild)
    line("B_CONTINUES and B_MILDER", r_both, pred=+1)
    line("  component: B_CONTINUES", r_cont)
    line("  component: B_MILDER", r_mild)
    #: Condition 2. The conjunction's FR-RF must exceed what the two marginals
    #: would produce on their own. Product of the FR rates minus product of the
    #: RF rates is the independence benchmark for the same difference.
    bench = r_cont["fr"] * r_mild["fr"] - r_cont["rf"] * r_mild["rf"]
    print("  independence benchmark             %+0.3f" % bench)
    print("  excess over benchmark              %+0.3f" % (r_both["diff"] - bench))
    c1 = r_both["diff"] > 0 and r_both["p"] < 0.05
    c2 = r_both["diff"] > bench
    print("\n  condition 1 (positive, p<0.05): %s" % ("MET" if c1 else "NOT MET"))
    print("  condition 2 (exceeds marginals): %s" % ("MET" if c2 else "NOT MET"))
    print("  PRIMARY: %s" % ("SUPPORTED at stage 1" if (c1 and c2) else "NOT SUPPORTED at stage 1"))
    out["primary"] = dict(conjunction=r_both, continues=r_cont, milder=r_mild,
                          benchmark=float(bench), cond1=bool(c1), cond2=bool(c2))

    print("\n=== SECONDARIES, seven declared ===")
    SEC = [("register=B_GENERIC", L.register == "B_GENERIC", +1),
           ("register=B_DIFFERENT_REGISTER", L.register == "B_DIFFERENT_REGISTER", +1),
           ("register=B_CONTINUES", cont, None),
           ("more_transgressive", L.more_transgressive == "YES", -1),
           ("pitch=B_MILDER", mild, +1),
           ("pitch=B_STRONGER", L.pitch == "B_STRONGER", -1),
           ("becomes_speech", L.becomes_speech == "YES", +1)]
    out["secondary"] = {}
    for lab, m, pred in SEC:
        r = diff(L, m)
        line(lab, r, pred)
        out["secondary"][lab] = dict(r, predicted_sign=pred)

    print("\n=== SYMMETRIC: position bias, no prediction, never an effect ===")
    bias = []
    for lab, m in [("related", L.related == "YES"), ("substitutable", L.substitutable == "YES"),
                   ("bare_verb", L.bare_verb == "YES")]:
        r = diff(L, m)
        line(lab, r)
        bias.append(abs(r["diff"]))
    print("  mean |bias| %.3f  (R corpus: 0.010)" % np.mean(bias))
    out["position_bias"] = float(np.mean(bias))

    print("\n=== POWER FOR STAGE 2, the actual deliverable ===")
    print("Per-stem SD of the conjunction difference at stage 1: %.4f" % r_both["sd"])
    for nm, n in [("stage 1 (observed)", r_both["n"]), ("stage 2", N_STAGE2 * 2)]:
        mde = 2.8 * r_both["sd"] / np.sqrt(n)
        print("  %-20s n=%4d  MDE at 80%% power, alpha .05 = %+0.4f" % (nm, n, mde))
    mde2 = 2.8 * r_both["sd"] / np.sqrt(N_STAGE2 * 2)
    print("\n  stage-1 conjunction effect        %+0.4f" % r_both["diff"])
    print("  stage-2 MDE                       %+0.4f" % mde2)
    if abs(r_both["diff"]) < mde2:
        print("  *** STAGE 2 CANNOT DETECT AN EFFECT THIS SIZE. Say so before running. ***")
    else:
        print("  stage 2 is powered for an effect of the stage-1 size, which is a")
        print("  SELECTED estimate and should be expected to shrink.")
    out["power"] = dict(sd=float(r_both["sd"]), mde_stage2=float(mde2),
                        stage1_effect=float(r_both["diff"]),
                        detectable=bool(abs(r_both["diff"]) >= mde2))

    print("\n=== INSTRUMENT HEALTH ===")
    tw = L.groupby(["stem", "member", "order"]).register.nunique()
    print("  register three-way splits: %.0f%% of items" % (100 * (tw >= 3).mean()))
    print("  register arm usage: %s" % L.register.value_counts().to_dict())
    print("  per-coder B_GENERIC rate:")
    for c, v in (L.groupby("coder").register.apply(lambda s: (s == "B_GENERIC").mean())
                 .sort_values(ascending=False).items()):
        print("    %-42s %5.1f%%" % (c, 100 * v))
    out["three_way_register"] = float((tw >= 3).mean())

    with open(RESULT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote %s" % RESULT)


if __name__ == "__main__":
    main()
