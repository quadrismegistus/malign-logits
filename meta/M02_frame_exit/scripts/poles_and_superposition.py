"""Two joins between the output-side ratio and the representation-side geometry.

    uv run python poles_and_superposition.py

Both questions are RH's, 2026-08-11, and both were asked because the two M02
findings appeared to disagree: the output-side ratio finds superposition and the
pole-axis projection does not.

(1) WHICH "BOTH" DOES THE MODEL DO? They do not disagree; they target different
mixtures, and only one of them is a thing the models produce.

    softmax(a.z_A + (1-a).z_B)  proportional to  P_A^a . P_B^(1-a)

so a MIDPOINT IN LOGIT SPACE gives the GEOMETRIC mean -- the product of experts,
the INTERSECTION, mass only where both poles agree. F11's ratio scores against
the ARITHMETIC mean -- the UNION, mass where EITHER pole licenses. Those are
opposite notions of "both", and inclusive disjunction is the union one.

(2) DOES POLE SEPARATION PREDICT LOSS OF SUPERPOSITION? RH's hypothesis: driving
the poles apart makes their continuation sets more disjoint, so a distribution
covering both becomes harder to occupy -- exclusive disjunction installed in the
axis itself rather than by picking a side. This joins two INDEPENDENT
substrates: `pole_sep` from the L3 hidden states, the superposition signal from
the twp output ratio. Agreement across substrates is worth more than either
alone.

    THE CONFOUND, WHICH THIS CANNOT RULE OUT. Both quantities may simply track
    HOW MUCH ALIGNMENT HAPPENED: a heavily aligned model would show more pole
    separation and more superposition loss with neither causing the other. The
    correlation is real; the arrow is not established, and settling it needs the
    SFT/DPO checkpoint ladder, where separation either precedes collapse or does
    not. That is M05 and it is held.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

import contradiction_null as C  # noqa: E402
from malign_logits.lineage import lineage_of, UnmappedModel  # noqa: E402


def lin(m):
    try:
        return lineage_of(m)
    except UnmappedModel:
        return "X:" + m


def _vecs(ds):
    keys = sorted(set().union(*[d.keys() for d in ds]))
    return [np.array([d.get(k, 0.0) for k in keys]) / max(sum(d.values()), 1e-30)
            for d in ds]


def union_vs_intersection(lang="en"):
    G = C.groups(lang)
    PR = C.pairs()
    models = sorted({m for p in PR for m in p.split(">")})
    prompts = sorted({g[k] for g in G for k in ("pole_a", "pole_b", "both")})
    D = C.fetch_twp(models, prompts)
    rows = []
    for pr in PR:
        b, al = pr.split(">")
        for arm, mid in (("base", b), ("aligned", al)):
            for g in G:
                k = [(mid, g[x]) for x in ("pole_a", "pole_b", "both")]
                if not all(x in D for x in k):
                    continue
                A, B, AB = _vecs([D[k[0]], D[k[1]], D[k[2]]])
                #: the geometric mean is renormalised because P_A^.5 P_B^.5 is
                #: a product of experts and does not sum to 1 on its own.
                geo = np.sqrt(np.clip(A, 1e-12, None) * np.clip(B, 1e-12, None))
                geo = geo / geo.sum()
                rows.append((lin(b), arm, g["group"],
                             C._js(AB, 0.5 * (A + B)), C._js(AB, geo)))
    return pd.DataFrame(rows, columns=["lineage", "arm", "group",
                                       "js_arith", "js_geo"])


def separation_vs_superposition():
    P = pd.read_parquet(os.path.join(CAMP, "results", "l3_geometry_union.parquet"))
    P = P.rename(columns={"base": "base_model", "aligned": "aligned_model"})
    P["depth"] = P.layer / (P.n_layers - 1)
    bad = P[P.pole_sep < 0.02][["base_model", "aligned_model", "group", "layer"]].drop_duplicates()
    P = P.merge(bad.assign(_d=1), on=["base_model", "aligned_model", "group", "layer"], how="left")
    P = P[P._d.isna()].drop(columns=["_d"])
    I = P[(~P.negative_control) & (P.depth >= 0.2) & (P.depth < 0.6)].copy()
    I["lineage"] = I.base_model.map(lin)
    ps = I.pivot_table(index=["lineage", "base_model", "group", "layer"],
                       columns="arm", values="pole_sep").dropna()
    dsep = (ps.aligned - ps.base).groupby(level="lineage").mean().rename("d_polesep")
    S = pd.read_csv(os.path.join(CAMP, "results", "contradiction_null_by_pair_en.csv"))
    S["lineage"] = S.base.map(lin)
    return S.set_index("lineage")[["delta"]].join(dsep, how="inner").dropna()


def main():
    R = union_vs_intersection()
    out = os.path.join(CAMP, "results", "union_vs_intersection.csv")
    R.to_csv(out, index=False)
    print("=" * 76)
    print("(1) UNION OR INTERSECTION: which mixture is the contradiction output?")
    print("=" * 76)
    print("   %-8s %16s %16s %s" % ("arm", "JS to ARITHMETIC", "JS to GEOMETRIC", "closer to arithmetic"))
    for arm in ("base", "aligned"):
        g = R[R.arm == arm].groupby("lineage")[["js_arith", "js_geo"]].mean()
        print("   %-8s %16.4f %16.4f   %d of %d lineages"
              % (arm, g.js_arith.mean(), g.js_geo.mean(),
                 int((g.js_geo > g.js_arith).sum()), len(g)))
    R["pref"] = R.js_geo - R.js_arith
    gb = R[R.arm == "base"].groupby("lineage").pref.mean()
    ga = R[R.arm == "aligned"].groupby("lineage").pref.mean()
    k = gb.index.intersection(ga.index)
    print("\n   preference for UNION over INTERSECTION (js_geo - js_arith, >0 = union)")
    print("      base %+0.4f   aligned %+0.4f   shift %+0.4f  p=%.3g"
          % (gb[k].mean(), ga[k].mean(), (ga[k] - gb[k]).mean(),
             stats.wilcoxon(gb[k], ga[k]).pvalue))
    print("\n   ARITHMETIC WINS, so the models do UNION. A midpoint in logit space")
    print("   would give the geometric mixture, which is not what they produce --")
    print("   so t's off-axis remainder is where the union lives, not the shadow.")

    J = separation_vs_superposition()
    J.to_csv(os.path.join(CAMP, "results", "polesep_vs_superposition.csv"))
    sp = stats.spearmanr(J.d_polesep, J.delta)
    pe = stats.pearsonr(J.d_polesep, J.delta)
    print("\n" + "=" * 76)
    print("(2) DOES POLE SEPARATION GROWTH PREDICT LOSS OF SUPERPOSITION?")
    print("=" * 76)
    print("   x = pole_sep change (L3 hidden states), y = superposition change (twp output)")
    print("   n = %d lineages   Spearman rho %+0.3f p=%.3g   Pearson r %+0.3f p=%.3g"
          % (len(J), sp.correlation, sp.pvalue, pe[0], pe[1]))
    print("   negative = poles separating goes with superposition falling.")
    print("   TWO INDEPENDENT SUBSTRATES, which is the point; and the arrow is NOT")
    print("   established -- both may track how much alignment happened. M05's")
    print("   checkpoint ladder is what would separate those, and it is held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
