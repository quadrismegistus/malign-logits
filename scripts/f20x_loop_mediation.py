"""Q-loop mediation on the published 1P corpus.

REGISTERED AT docket [239], CORRECTED AT [241]/[245]. Read the correction before
the number: this test CANNOT kill mechanism (1).

    (1) says alignment converts CONTINUATION into ANSWERING. Docket [239]
    specified the test as a CHAIN, arm -> loop -> drift. RH's correction is that
    the answerer role was post-trained into the weights, which makes (1) a FORK:
    one disposition producing fewer self-written `Q:` turns AND fewer
    incompatible accounts as SIBLINGS. A fork returns LOW mediation with (1)
    fully intact.

WHAT THE TWO BRANCHES ARE WORTH, booked at [245] before the number existed:

    HIGH mediation  the loop carries the effect. Runs AGAINST lacan's general-
                    fixing reading (a fixing operation on the signifier should
                    appear independent of looping). Consistent with chain-(1)
                    AND with the loop being an opportunity multiplier -- a
                    passage that spawns three more questions has more surface
                    on which two accounts can fail to cohere.
    LOW  mediation  against chain-(1) and against opportunity. Leaves fork-(1)
                    and lacan's reading BOTH standing. Discriminates least.

So the discriminating branch damages the seat that registered it, which is the
reverse of what [239] declared. It gates nothing: the four-level format battery
registers independently of this result ([243], [244], [245]).

METHOD. Standardization, not a regression product-of-coefficients. For each base
model the crude arm delta is compared against the delta standardized over that
model's POOLED loop distribution -- i.e. what the delta would be if both arms
looped at the same rate. The gap between them is the part attributable to the
arm difference in looping.

    proportion mediated = (crude - standardized) / crude

CAVEAT THAT IS NOT OPTIONAL. `loop` is post-treatment (base 0.673, aligned
0.459). Standardizing fixes the COMPOSITION; the within-stratum contrast is a
controlled direct effect and still assumes no unmeasured loop->drift
confounding. Passage length is the obvious candidate -- looping passages are
longer -- so the whole thing is reported again stratified on loop x word-count
tertile. Both are reported, per the collider rule at docket [213]; the
difference between them is left visible rather than resolved.

UNIT: the distinct base model (Rule 2). CELL: terminal aligned arm only,
matching the published analysis and `f20x_crossprovider.py`.

    uv run .venv/bin/python scripts/f20x_loop_mediation.py
"""
import numpy as np
import pandas as pd
from scipy import stats

SRC = "data/f20x_codings.parquet"
ALIGNED = ("ego", "superego", "reinforced_superego")


def load():
    """Published cell selection. Derived here, never read back from disk."""
    d = pd.read_parquet(SRC)
    n_raw = len(d)
    d = d[d.family != "olmo-think"].copy()
    d["text"] = d.text.fillna("")
    d = d[d.text.str.strip().str.len() > 0]
    d["al"] = d.arm != "base"

    # Terminal aligned arm only.
    term = {}
    for f, g in d[d.al].groupby("family"):
        for s in reversed(ALIGNED):
            if s in set(g.arm):
                term[f] = s
                break
    d = d[(d.arm == "base").to_numpy()
          | np.array([term.get(r.family) == r.arm for r in d.itertuples()])]

    # `codes` is a JSON STRING in some files and an array in others -- the trap
    # booked at docket [205] and again at [248]. Derive, do not assume.
    def has_qd(c):
        if c is None:
            return False
        if isinstance(c, str):
            import json
            try:
                c = json.loads(c)
            except Exception:
                return "quiet_drift" in c
        return "quiet_drift" in list(c)

    d["drift"] = d.codes.map(has_qd)
    d["loop"] = d.text.str.contains(r"\nQ:", regex=True, na=False)
    d["words"] = d.text.str.split().str.len()

    assert len(d) <= n_raw, "selection grew the frame"
    assert d.base_model_id.nunique() == 29, d.base_model_id.nunique()
    print(f"{n_raw:,} raw -> {len(d):,} analysed | "
          f"{d.base_model_id.nunique()} base models")
    print(f"drift  base {d[~d.al].drift.mean():.4f}  "
          f"aligned {d[d.al].drift.mean():.4f}")
    print(f"loop   base {d[~d.al].loop.mean():.4f}  "
          f"aligned {d[d.al].loop.mean():.4f}\n")
    return d


def standardize(g, strata):
    """Crude and standardized base-minus-aligned delta for one base model.

    Standardized = both arms reweighted to this model's POOLED stratum mix.
    Strata missing an arm are DROPPED and their pooled weight returned, so the
    caller can report how much mass the standardization could not use rather
    than silently renormalising over it.
    """
    crude = g[~g.al].drift.mean() - g[g.al].drift.mean()
    num = 0.0
    used = 0.0
    for _, cell in g.groupby(strata, observed=True):
        b, a = cell[~cell.al], cell[cell.al]
        if len(b) == 0 or len(a) == 0:
            continue
        wt = len(cell) / len(g)
        num += wt * (b.drift.mean() - a.drift.mean())
        used += wt
    if used == 0:
        return None
    return crude, num / used, used


def run(d, strata, label):
    rows, dropped = [], []
    for bm, g in d.groupby("base_model_id"):
        r = standardize(g, strata)
        if r is None:
            dropped.append(bm)
            continue
        crude, std, used = r
        rows.append({"base_model_id": bm, "crude": crude, "std": std,
                     "mass_used": used})
    r = pd.DataFrame(rows)
    mediated = (r.crude.sum() - r["std"].sum()) / r.crude.sum()
    pos_crude = int((r.crude > 0).sum())
    pos_std = int((r["std"] > 0).sum())
    n = len(r)
    p_crude = stats.binomtest(pos_crude, n, 0.5, "greater").pvalue
    p_std = stats.binomtest(pos_std, n, 0.5, "greater").pvalue
    # Per-model mediation, so the pooled figure is not carried by one model.
    pm = ((r.crude - r["std"]) / r.crude.replace(0, np.nan)).dropna()
    print(f"--- {label}  (n={n} of 29 base models) ---")
    print(f"  crude delta        {r.crude.mean():+.4f}   {pos_crude}/{n}  "
          f"p={p_crude:.4f}")
    print(f"  standardized       {r['std'].mean():+.4f}   {pos_std}/{n}  "
          f"p={p_std:.4f}")
    print(f"  PROPORTION MEDIATED (pooled)   {mediated:+.3f}")
    print(f"  per-model mediation  median {pm.median():+.3f}  "
          f"IQR [{pm.quantile(.25):+.3f}, {pm.quantile(.75):+.3f}]")
    print(f"  stratum mass usable  min {r.mass_used.min():.3f}  "
          f"median {r.mass_used.median():.3f}")
    if dropped:
        print(f"  DROPPED (no stratum with both arms): {dropped}")
    return r


if __name__ == "__main__":
    d = load()
    d["wt"] = d.groupby("base_model_id").words.transform(
        lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
    # Three-way decomposition. Standardizing on loop x length removes BOTH, so
    # the joint figure alone cannot be read as the loop's contribution net of
    # length -- the length-only run is what makes the attribution possible.
    run(d, ["loop"], "LOOP ONLY")
    run(d, ["wt"], "LENGTH ONLY (word-count tertile)")
    run(d, ["loop", "wt"], "LOOP x LENGTH (joint)")
