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

--------------------------------------------------------------------------
`--resid`: THE FREQUENCY CONTROL, specified at findings V line 139 and run on
2026-08-06 after RH asked for it.

The verbs-only axis reads plain Anglo-Saxon action verbs (put, got, go, tell)
to formal Latinate ones (administered, determined, cautioned, concluded), and
correlates with log frequency rank at r=+0.554. Latinate words are rarer, so
"alignment shifts register" and "alignment shifts toward rarer words" predict
the SAME axis and the correlation cannot choose between them. Residualise:
regress each of the 1024 dimensions on log frequency rank across the retained
types, keep the residuals, recompute everything from those.

OUTCOME MAP, all cells, written before running.

  POWER FIRST -- cos(raw axis, residualised axis).
    ~1.00  the axis did not move; frequency lies almost entirely OUTSIDE the
           axis and residualising it out was never capable of changing the
           poles. Survival is then VACUOUS, not evidence. Report as a control
           that could not fire and stop.
    <0.95  the axis moved materially and the poles below are a real re-reading.

  THE POLES, the primary read.
    (a) still plain -> formal. Register survives frequency; it is a second
        instrument agreeing with findings T's proceduralisation, no shared
        design between them.
    (b) legible but a DIFFERENT contrast. Report that contrast as itself. Do
        not bend it back onto the register story.
    (c) not legible. The axis was frequency. The plain-to-formal reading is
        not available, and the geometric line closes with scene-locality as
        its only survivor.

  SITE ALIGNMENT, obs vs the pairing-shuffled null (raw: 0.059 vs 0.046).
    Note the null already carries the frequency gradient -- it preserves both
    marginals and destroys only the pairing -- so the GAP is expected to
    survive roughly intact while both levels fall. If the gap ALSO vanishes,
    frequency was interacting with the pairing and that is a finding.

  SCENE-LOCALITY, twin vs random (raw: 0.327 vs 0.060). Predicted to survive:
    it is a within-scene contrast and frequency is a global gradient. If it
    does NOT survive, V.5's one robust result was frequency too, which would
    be the largest negative in plan V and has to be reported as such.

  RUNG PARALLELISM (raw: own 0.238 vs cross-family 0.323). Both fall; the
    ordering is the content, and it is already recorded as cutting against U.6.

  THE POST-RESIDUAL FREQUENCY CORRELATION IS NOT A RESULT. Without row
  renormalisation it is exactly zero by construction: every column of X_res is
  orthogonal to centred lrank, so lrank . (X_res @ a) = 0 for ANY axis a. It is
  printed as an implementation check -- a non-zero value means the code is
  wrong, a zero value means nothing about the world. With renormalisation the
  identity is broken by a nonlinearity and the residual is a small real number.
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
    ap.add_argument("--resid", action="store_true",
                    help="residualise log frequency rank out of the vectors first")
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

    #: ---- THE FREQUENCY CONTROL. Regress every dimension on log frequency rank
    #: across the RETAINED types (the analysis population, so vv* under --verbs),
    #: keep the residuals. Row renormalisation afterwards matches the raw
    #: pipeline, which normalises before use; it also breaks the exact-zero
    #: identity, so the un-renormalised projection is kept for the check.
    X_raw = X
    freq_dir = None
    if a.resid:
        oov = np.array([w not in rank for w in words])
        k = keepi
        y = lrank[k] - lrank[k].mean()
        Xc = X[k] - X[k].mean(0)
        beta = (y @ Xc) / (y @ y)                      # 1024 OLS slopes
        X_res = X - np.outer(lrank - lrank[k].mean(), beta)
        ss_tot = float((Xc ** 2).sum())
        ss_exp = float((y @ y) * (beta @ beta))
        freq_dir = beta / np.linalg.norm(beta)
        X_plain = X_res                                 # identity holds on this
        nrm = np.linalg.norm(X_res, axis=1, keepdims=True)
        X = X_res / np.where(nrm > 0, nrm, 1.0)
        print("=" * 90)
        print("RESIDUALISING LOG FREQUENCY RANK OUT OF THE VECTORS")
        print("=" * 90)
        print("  fit over %d retained types, of which %d (%.1f%%) have no BYU rank"
              % (int(k.sum()), int((oov & k).sum()), 100 * (oov & k).mean() / max(k.mean(), 1e-9)))
        print("  frequency explains %.3f%% of the embedding variance over those types"
              % (100 * ss_exp / ss_tot))
        print("  (a small share here is expected and is NOT the power check --")
        print("   the axis is one direction, not the bulk of the variance)")

    per_fam, per_raw = {}, {}
    for (fam, rung), g in W.groupby(["family", "rung"]):
        sv = site_vectors(g, wi, X)
        if len(sv) >= 50:
            per_fam[(fam, rung)] = sv
            if a.resid:
                per_raw[(fam, rung)] = site_vectors(g, wi, X_raw)
    rungs = sorted({r for _, r in per_fam})
    print("families x rungs with >=50 usable sites: %d   rungs: %s\n" % (len(per_fam), rungs))

    #: ---- the mean axis, pooled over base>sft, for the control and the poles
    base = [v for (f, r), sv in per_fam.items() if r == "base>sft" for v, _, _ in sv.values()]
    axis = np.mean(base, axis=0)
    axis = axis / np.linalg.norm(axis)
    proj = X @ axis

    if a.resid:
        raw_base = [v for (f, r), sv in per_raw.items() if r == "base>sft" for v, _, _ in sv.values()]
        raw_axis = np.mean(raw_base, axis=0)
        raw_axis = raw_axis / np.linalg.norm(raw_axis)
        rot = float(raw_axis @ axis)
        load = float(raw_axis @ freq_dir)
        pj = X_plain @ axis
        rid, _ = stats.pearsonr(lrank[keepi], pj[keepi])
        print("\n  POWER CHECK -- could this control have changed anything?")
        print("    cos(raw axis, residualised axis)        %+.4f" % rot)
        print("    cos(raw axis, frequency direction)      %+.4f" % load)
        print("    -> |cos| near 1 on the first line means the axis did not move and")
        print("       whatever survives below survives VACUOUSLY. Below ~0.95 the")
        print("       control had power and the poles are a genuine re-reading.")
        print("    IMPLEMENTATION CHECK, not a result: projection ~ log freq on the")
        print("    un-renormalised residuals r=%+.6f (exactly 0 by construction;" % rid)
        print("    a non-zero value here means the residualisation is coded wrong)")

    print("\n" + "=" * 90)
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
    sfx = ("_verbs" if a.verbs else "") + ("_resid" if a.resid else "")
    A.to_csv(os.path.join(OUT, "v_displacement_vector%s.csv" % sfx), index=False)
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
    #: every retained type's position on the axis, so the pole reading can be
    #: tested on all of them rather than on the 36 printed here. Dumped from
    #: HERE rather than recomputed downstream: a second script re-deriving the
    #: axis is a second estimator, and the two drift without either being wrong.
    pd.DataFrame(dict(word=[words[i] for i in idx], proj=proj[idx],
                      pos=[pos.get(words[i], "") for i in idx],
                      log_freq_rank=lrank[idx])).to_csv(
        os.path.join(OUT, "v_axis_projection%s.csv"
                     % (("_verbs" if a.verbs else "") + ("_resid" if a.resid else ""))), index=False)

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
        T.to_csv(os.path.join(OUT, "v_displacement_twin%s.csv" % sfx), index=False)
    print("\nwrote v_displacement_vector%s.csv" % sfx)


if __name__ == "__main__":
    main()
