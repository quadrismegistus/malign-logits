"""Re-derive the liminal/explicit block from battery_results.csv.

Written because the block was carrying three defects at once: an n that its own
correction had already superseded (9 families, but tulu and tulu-no-safety share
base AND superego, so 8 distinct base->superego pairs), a CI whose interval method
was never named, and a slope that does not reproduce under any method tried.

Run it before citing any number in that block.
"""
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("data/battery_results.csv")
d["cat"] = d.label.str.rsplit("_", n=1).str[0]
JS, EN = "js_base_superego", "entropy_base"
LIM = ["sexual_liminal", "violence_liminal"]
EXP = ["sexual_explicit", "violence_explicit"]

def fm(f, cats, col):
    return d[(d.family == f) & d.cat.isin(cats)][col].mean()

ALL = sorted(d.family.unique())
# tulu and tulu-no-safety share base and superego: identical on every
# base->superego metric to 0.0e+00 across all 47 prompts. One unit, not two.
UNITS = [f for f in ALL if f != "tulu-no-safety"]

# THE THIRD ARM, ADDED 1 AUG. n=8 collapses tulu/tulu-no-safety on an EXACT
# DUPLICATE argument -- identical to 0.0e+00 -- which is the narrowest possible
# grounds. **llama, tulu and tulu-no-safety all sit on meta-llama/Llama-3.1-8B:
# three alignment recipes on ONE pretraining run.** By the campaign's own rule
# ("two recipes applied to one base are two recipes, not two implementations")
# the independent unit is the LINEAGE, and n=8 is 7.
#
# **This arm exists because the same error class cost F40 its headline interval
# the same afternoon** (39 base strings resampled as 39 lineages; 34). Run on
# THIS block it changes nothing that matters, and that is the result: the check
# is not a machine for downgrading, and a claim that survives it is stronger for
# having been asked.
import json as _json
_lm = _json.load(open("data/lineage_map_models.json"))["model_to_lineage"]
from malign_logits import MODEL_FAMILIES as _MF
LIN = {f: _lm.get(getattr(_MF.get(f), "base", None), f) for f in ALL}


def collapse(vals, fams):
    """Mean within pretraining run. Averaging, not picking: no representative."""
    by = {}
    for f, v in zip(fams, vals):
        by.setdefault(LIN[f], []).append(v)
    return np.array([np.mean(v) for v in by.values()])


for tag, fams in [("as booked (n=9, superseded)", ALL), ("corrected (n=8)", UNITS),
                  ("LINEAGE (n=7)", "LINEAGE")]:
    print(f"\n{tag}")
    lineage = fams == "LINEAGE"
    if lineage:
        fams = ALL
    for name, A, B in [("liminal - explicit", LIM, EXP), ("liminal - neutral", LIM, ["neutral"])]:
        v = np.array([fm(f, A, JS) - fm(f, B, JS) for f in fams])
        if lineage:
            v = collapse(v, fams)
        t, p = stats.ttest_1samp(v, 0)
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(),
                              scale=v.std(ddof=1) / np.sqrt(len(v)))
        rng = np.random.default_rng(0)
        bs = np.percentile([rng.choice(v, len(v), replace=True).mean()
                            for _ in range(20000)], [2.5, 97.5])
        print(f"  {name:20s} n={len(v)} mean={v.mean():+.4f} t-CI [{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"boot-CI [{bs[0]:+.4f},{bs[1]:+.4f}] t={t:+.2f} p={p:.4f} "
              f"pos={int((v > 0).sum())}/{len(v)}")

    # The entropy gap reproduces; the booked +0.0187/nat slope does not.
    obs = np.mean([fm(f, LIM, JS) - fm(f, EXP, JS) for f in fams])
    gap = np.mean([fm(f, LIM, EN) - fm(f, EXP, EN) for f in fams])
    sl = []
    for f in fams:
        g = d[d.family == f]
        g = g[np.isfinite(g[JS]) & np.isfinite(g[EN])]
        sl.append(np.polyfit(g[EN], g[JS], 1)[0])
    g = d[d.family.isin(fams)]
    g = g[np.isfinite(g[JS]) & np.isfinite(g[EN])]
    pooled = np.polyfit(g[EN], g[JS], 1)[0]
    print(f"  entropy gap {gap:.3f} nats  (booked 1.315 at n=9 -- reproduces)")
    for m, s in [("mean-of-within-family OLS", float(np.mean(sl))), ("pooled OLS", float(pooled))]:
        print(f"    {m:26s} {s:+.4f}/nat  predicts {s*gap:+.4f} of {obs:+.4f}"
              f"  = {100*s*gap/obs:.0f}% explained")
    print("    booked slope               +0.0187/nat  -- NOT reproduced by either method")
