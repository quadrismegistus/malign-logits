"""Is there a DIRECTION of displacement, and is it the same everywhere?

    uv run --with lemminflect python v_displacement_vector.py

RH's question, and the one version of the geometric line still open. Everything
before it asked about DISTANCE -- how far the risers are from the fallers -- and
distance has now failed at five grains. **A vector asks which way instead, and
the two are independent: sets can be far apart AND consistently offset.**

That matters because findings V.4 found displacement moving AWAY from what it
replaced (own risers farther than twin risers, 14/14 families, p=0.0001). A
shared direction is entirely compatible with that, and would say the moving-away
is organised rather than arbitrary.

FIVE QUESTIONS, and the first is a control that can kill the rest.

  0. IS THE AXIS JUST WORD CLASS? Findings T says risers are more content-y than
     fallers. If so, every site's vector points along the open/closed axis and
     they align TRIVIALLY -- artefact 1 for the third time in plan V. Control:
     project the vocabulary onto the discovered axis and regress position on
     open-class membership and log frequency. **Printed before anything else.**
  1. DO SITE VECTORS ALIGN? mean(risers) - mean(fallers) per site, then mean
     pairwise cosine, against a null that shuffles which riser set pairs with
     which faller set. The null preserves every marginal and destroys only the
     pairing, which is the thing under test.
  2. WHAT IS AT THE POLES? Project the vocabulary onto the mean axis. This is
     the lexicon-free statement of what alignment moves language from and to --
     what the regional test tried to produce and could not.
  3. IS SFT PARALLEL TO DPO? Findings U.6 concluded "one operation at two
     amplitudes" from word overlap and field counts. If the rungs' vectors are
     parallel with different norms, that is the same claim geometrically and far
     more directly. If they are orthogonal, U.6 is wrong.
  4. DOES A MARKED SITE DISPLACE DIFFERENTLY FROM ITS TWIN? Same direction
     applied harder, or a direction of its own? This separates "alignment has
     one direction, pressed harder where there is transgression" from
     "transgression has its own direction".

UNIT: the family for every test. VECTORS: bare bge-m3, `v_bare_vectors.npz`.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))

CACHE = os.path.join(OUT, "v_bare_vectors.npz")
WALK = os.path.join(OUT, "t_ladder_words.parquet")
POP = os.path.join(ROOT, "data", "r_population_k2.parquet")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"
MIN_SIDE = 3
DRAWS = 2000


def claws():
    pos, rank = {}, {}
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for i, ln in enumerate(fh):
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                w, t = f[-1].strip().lower(), f[-3].strip()
                if w and w not in pos:
                    pos[w], rank[w] = t, i
    return pos, rank


def site_vectors(g, wi, X):
    """mean(risers) - mean(fallers) per prompt, unit-normalised."""
    out = {}
    for p, h in g.groupby("prompt"):
        f = [wi[w] for w in h[h["role"] == "faller"]["word"].unique() if w in wi]
        r = [wi[w] for w in h[h["role"] == "riser"]["word"].unique() if w in wi]
        if len(f) >= MIN_SIDE and len(r) >= MIN_SIDE:
            d = X[r].mean(0) - X[f].mean(0)
            nrm = np.linalg.norm(d)
            if nrm > 0:
                out[p] = (d / nrm, f, r)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    #: RESTRICT TO LEXICAL VERBS. The pooled axis came out as frequency (r=+0.46)
    #: plus class (r=+0.39), with pronouns and auxiliaries at the from-pole. CLAWS
    #: `vv*` is lexical verbs only -- it excludes BE, HAVE, DO, modals, pronouns,
    #: determiners and prepositions, so EVERY member is open-class and the class
    #: artefact cannot arise. Frequency variance survives (`put`/`go` against
    #: `administered`/`informed`), so this separates the two artefacts rather than
    #: removing both. Same restriction `s_lexicon_crosstab.run(verbs_only=True)`
    #: uses, so the two are comparable.
    ap.add_argument("--verbs", action="store_true", help="lexical verbs (CLAWS vv*) only")
    a = ap.parse_args()
    z = np.load(CACHE, allow_pickle=True)
    words = list(z["words"])
    X = z["X"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    wi = {w: i for i, w in enumerate(words)}
    pos, rank = claws()
    OPEN = ("vv", "nn", "jj", "rr")
    isopen = np.array([str(pos.get(w, "")).startswith(OPEN) for w in words])
    lrank = np.log1p(np.array([rank.get(w, 60000) for w in words]))

    W = pd.read_parquet(WALK)
    W = W[W["word"].isin(wi)]
    if a.verbs:
        vv = {w for w in words if str(pos.get(w, "")).startswith("vv")}
        W = W[W["word"].isin(vv)]
        keepi = np.array([w in vv for w in words])
        print("VERBS ONLY: %d of %d types are CLAWS vv*, %s movement rows remain\n"
              % (len(vv), len(words), f"{len(W):,}"))
    else:
        keepi = np.ones(len(words), dtype=bool)

    per_fam = {}
    for (fam, rung), g in W.groupby(["family", "rung"]):
        sv = site_vectors(g, wi, X)
        if len(sv) >= 50:
            per_fam[(fam, rung)] = sv
    rungs = sorted({r for _, r in per_fam})
    print("families x rungs with >=50 usable sites: %d   rungs: %s\n" % (len(per_fam), rungs))

    #: ---- the mean axis, pooled over base>sft, for the control and the poles
    base = [v for (f, r), sv in per_fam.items() if r == "base>sft" for v, _, _ in sv.values()]
    axis = np.mean(base, axis=0)
    axis = axis / np.linalg.norm(axis)
    proj = X @ axis

    print("=" * 90)
    print("CONTROL FIRST: is the axis just word class or frequency?")
    print("=" * 90)
    ro, po = stats.pointbiserialr(isopen[keepi].astype(int), proj[keepi]) \
        if len(set(isopen[keepi])) > 1 else (float("nan"), float("nan"))
    rf, pf = stats.pearsonr(lrank[keepi], proj[keepi])
    print("  projection ~ open-class:   r=%+.3f  p=%.2e" % (ro, po))
    print("  projection ~ log freq rank r=%+.3f  p=%.2e" % (rf, pf))
    if len(set(isopen[keepi])) > 1:
        print("  open-class mean projection %+.4f vs closed %+.4f"
              % (proj[keepi & isopen].mean(), proj[keepi & ~isopen].mean()))
    else:
        print("  open-class correlation UNDEFINED: every retained type is open-class,")
        print("  which is the point of the restriction. Frequency is the artefact left.")
    print("  -> a large |r| on class means the direction is grammatical, not semantic.")

    print("\n" + "=" * 90)
    print("1. DO SITE VECTORS ALIGN?  observed vs a pairing-shuffled null")
    print("=" * 90)
    rng = np.random.default_rng(20260806)
    rows = []
    for (fam, rung), sv in sorted(per_fam.items()):
        V = np.vstack([v for v, _, _ in sv.values()])
        obs = float((V @ V.T)[np.triu_indices(len(V), 1)].mean())
        fs = [f for _, f, _ in sv.values()]
        rs = [r for _, _, r in sv.values()]
        nulls = []
        for _ in range(20):
            perm = rng.permutation(len(rs))
            Vn = []
            for i, j in enumerate(perm):
                d = X[rs[j]].mean(0) - X[fs[i]].mean(0)
                n = np.linalg.norm(d)
                if n > 0:
                    Vn.append(d / n)
            Vn = np.vstack(Vn)
            nulls.append(float((Vn @ Vn.T)[np.triu_indices(len(Vn), 1)].mean()))
        rows.append(dict(family=fam, rung=rung, n_sites=len(sv), obs=obs,
                         null=float(np.mean(nulls)), gap=obs - float(np.mean(nulls))))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(OUT, "v_displacement_vector%s.csv" % ("_verbs" if a.verbs else "")), index=False)
    for rung, g in A.groupby("rung"):
        if len(g) < 6:
            continue
        p = stats.wilcoxon(g["obs"], g["null"]).pvalue
        print("  %-10s %2d families  observed %.4f  shuffled %.4f  gap %+.4f  %d/%d higher  p=%.4f"
              % (rung, len(g), g["obs"].mean(), g["null"].mean(), g["gap"].mean(),
                 int((g["gap"] > 0).sum()), len(g), p))
    print("  (mean pairwise cosine between site vectors; 0 = no shared direction, 1 = identical)")

    print("\n" + "=" * 90)
    print("2. THE POLES OF THE AXIS")
    print("=" * 90)
    idx = np.where(keepi)[0]
    o = idx[np.argsort(proj[idx])]
    print("  most NEGATIVE (the from-pole):  %s" % ", ".join(words[i] for i in o[:18]))
    print("  most POSITIVE (the to-pole):    %s" % ", ".join(words[i] for i in o[-18:][::-1]))

    print("\n" + "=" * 90)
    print("3. IS SFT PARALLEL TO DPO?")
    print("=" * 90)
    fam_axis = {}
    for (fam, rung), sv in per_fam.items():
        V = np.vstack([v for v, _, _ in sv.values()])
        m = V.mean(0)
        fam_axis[(fam, rung)] = m / np.linalg.norm(m)
    both = [f for f, _ in fam_axis if (f, "base>sft") in fam_axis and (f, "sft>pref") in fam_axis]
    if both:
        cos = [float(fam_axis[(f, "base>sft")] @ fam_axis[(f, "sft>pref")]) for f in both]
        print("  cosine between a family's base>sft axis and its sft>pref axis:")
        print("    mean %.4f   min %.4f   max %.4f   over %d families"
              % (np.mean(cos), np.min(cos), np.max(cos), len(cos)))
        xf = [float(fam_axis[(a, "base>sft")] @ fam_axis[(b, "base>sft")])
              for i, a in enumerate(both) for b in both[i + 1:]]
        print("    for reference, cosine between DIFFERENT families' base>sft axes: %.4f" % np.mean(xf))
        print("  -> near 1 and above the cross-family reference = one operation, two amplitudes.")

    print("\n" + "=" * 90)
    print("4. DOES A MARKED SITE DISPLACE DIFFERENTLY FROM ITS TWIN?")
    print("=" * 90)
    P = pd.read_parquet(POP).drop_duplicates("prompt").set_index("prompt")[["stem", "member"]]
    P["member"] = P["member"].str.lower()
    twin = {}
    for stem, g in P.reset_index().groupby("stem"):
        if len(g) == 2 and g["member"].nunique() == 2:
            a, b = g["prompt"].tolist()
            twin[a], twin[b] = b, a
    rows = []
    for (fam, rung), sv in per_fam.items():
        if rung != "base>sft":
            continue
        pairs = [(p, twin[p]) for p in sv if twin.get(p) in sv]
        if len(pairs) < 20:
            continue
        tw = [float(sv[a][0] @ sv[b][0]) for a, b in pairs]
        ks = list(sv)
        rn = [float(sv[ks[i]][0] @ sv[ks[j]][0])
              for i, j in zip(rng.integers(0, len(ks), 400), rng.integers(0, len(ks), 400)) if i != j]
        rows.append(dict(family=fam, n_pairs=len(pairs), twin_cos=float(np.mean(tw)),
                         random_cos=float(np.mean(rn))))
    T = pd.DataFrame(rows)
    if len(T) >= 6:
        p = stats.wilcoxon(T["twin_cos"], T["random_cos"]).pvalue
        print("  cosine between a marked site's vector and its TWIN's:  %.4f" % T["twin_cos"].mean())
        print("  cosine between two RANDOM sites' vectors:              %.4f" % T["random_cos"].mean())
        print("  %d/%d families higher for twins   p=%.4f" % (int((T["twin_cos"] > T["random_cos"]).sum()), len(T), p))
        print("  -> higher for twins = the scene sets the direction; equal = the direction is global.")
        T.to_csv(os.path.join(OUT, "v_displacement_twin.csv"), index=False)
    print("\nwrote v_displacement_vector.csv")


if __name__ == "__main__":
    main()
