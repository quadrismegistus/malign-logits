"""F20's "29 distinct base models, which is the unit" ARE 25 LINEAGES.
Malign, owning [2175]'s recount.

`findings/F20_generation_drift.md:30` names the base model as the unit and :52
reports quiet_drift base-higher in **28 of 29**. Against
`data/lineage_map_models.json` those 29 base strings are **25 independent
pretraining lineages**: Qwen2.5-0.5B/7B is one release at two scales, Falcon3
1B/3B/7B/10B one at four.

`docs/f20x_generation_spec.md:55` shows how it happened: it DECLARES "clusters
that must collapse" and names two -- 6x Llama-3.1-8B, 2x Olmo-3-1025-7B. **Both
are families collapsing onto a base STRING. Neither is a size ladder.** The
declaration is what made it look already-handled.

**NO AGGREGATION RULE IS CHOSEN.** Collapsing a group of rungs to one vote is a
researcher degree of freedom on a published number, so ALL THREE plausible rules
are run and reported together ([2172] standard):

    majority   the rungs vote; ties are dropped, never broken
    largest    the largest rung speaks for the release
    pooled     rates pooled across the group's rows, then compared

A conclusion holding under all three does not depend on the choice.
"""
import collections, json
import numpy as np, pandas as pd
from scipy import stats

CODE = "quiet_drift"
d = pd.read_parquet("data/f20x_codings.parquet")
d[CODE] = d.codes.apply(lambda c: CODE in (c if c is not None else []))
d["aligned"] = d.arm != "base"
m2l = json.load(open("data/lineage_map_models.json"))["model_to_lineage"]

#: per (base string): rate in base arm vs aligned arm, and the direction
rows = []
for b, g in d.groupby("base_model_id"):
    rb = g[~g.aligned][CODE].mean()
    ra = g[g.aligned][CODE].mean()
    if np.isnan(rb) or np.isnan(ra):
        continue
    rows.append(dict(base=b, lineage=m2l.get(b, b), n=len(g),
                     base_rate=rb, aligned_rate=ra, higher=rb > ra))
R = pd.DataFrame(rows)
print(f"base strings {len(R)}   LINEAGES {R.lineage.nunique()}\n")
for l, g in R.groupby("lineage"):
    if len(g) > 1:
        print(f"  {l}  ({len(g)} rungs)")
        for _, r in g.iterrows():
            print(f"      {r.base:<34} base {r.base_rate:.3f} vs "
                  f"aligned {r.aligned_rate:.3f}  {'+' if r.higher else '-'}")


def report(tag, k, n):
    p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    print(f"  {tag:<30} {k}/{n}   p = {p:.2e}")


print(f"\nAS PUBLISHED")
report("base string as unit", int(R.higher.sum()), len(R))
print(f"\nLINEAGE AS UNIT, all three rules")
# majority: ties dropped, never broken
maj = []
for l, g in R.groupby("lineage"):
    up, dn = int(g.higher.sum()), int((~g.higher).sum())
    if up != dn:
        maj.append(up > dn)
report("majority (ties dropped)", sum(maj), len(maj))
# largest rung
big = R.loc[R.groupby("lineage").n.idxmax()]
report("largest rung speaks", int(big.higher.sum()), len(big))
# pooled rates within lineage
po = []
for l, g in R.groupby("lineage"):
    sub = d[d.base_model_id.isin(g.base)]
    po.append(sub[~sub.aligned][CODE].mean() > sub[sub.aligned][CODE].mean())
report("pooled within lineage", sum(po), len(po))
