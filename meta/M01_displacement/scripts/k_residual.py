"""Decompose the movement axis against a register direction built WITHOUT
movement data, and read the residuals.

    uv run python meta/M01_displacement/scripts/k_residual.py

RH's proposal: regress our register scale against the GloVe register and
interpret the residuals. The point of doing it is that each instrument sees
something the other misses, and the words where they disagree name what.

WHICH GLOVE REGISTER, AND WHY NOT THE MOVEMENT AXIS. `k_axis` fits its direction
TO PREDICT FALLING, so residualising anything against it and then testing the
residual on movement is circular. The direction used here is built from a usage
guide and GloVe and never sees a probability:

    register_dir = normalise( mean over Brooke's 399 near-synonym pairs of
                              vec(formal member) - vec(informal member) )

NEAR-SYNONYMY IS DOING THE WORK. `smooch/kiss` and `shiv/knife` differ in
register and barely in topic, so the difference vectors share their register
component and cancel elsewhere. This is the standard attribute-direction
construction and it is much cleaner than a seed-centroid difference, because
Brooke's formal seeds are almost all adverbs and connectives while the informal
ones are nouns and verbs -- a centroid difference between those two lists would
be substantially a part-of-speech direction. The seed version is computed anyway
and reported beside it, so the gap between them is visible.

THE FIRST NUMBER IS THE ONE THAT MATTERS. cos(movement axis, register_dir) says
how much of the direction alignment sorts on is register, using a register
measure that has no access to the movement data. Everything after it is
interpretation.

THEN THE RESIDUALS, BOTH WAYS, because they answer different questions:

    coder register residualised on GloVe register
        what our LLM coder judges to be register that the distribution does not
        carry. If these words are coherent, the coder is seeing something real
        that embeddings miss; if they are noise, that is the IAA 0.60 showing.

    GloVe register residualised on coder register
        register the distribution carries that the coder missed.

AND THEN THE TEST THAT MAKES IT MORE THAN A WORD LIST: does either residual
still predict movement, held out by word? If the coder residual predicts, the
coder scale is contributing something beyond distributional register. If neither
does, then whatever the two share is the whole of it.
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_analysis as A
import k_predict as KP2
from k_frequency import fpm

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
B = "/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources/brooke_formality"
N_SHOW = 25
SEED = 20260812


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    EM = {w: v.astype(np.float64) for w, v in zip(z["words"], z["E"])}
    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    ax = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"])

    #: the full GloVe table, so the register direction is not limited to our verbs
    import gensim.downloader as api
    KV = api.load("glove-wiki-gigaword-300")

    def gv(w):
        w = w.strip().lower()
        return KV[w].astype(np.float64) if w in KV else None

    pairs = []
    for ln in open(os.path.join(B, "CTRWpairsfull.txt"), encoding="utf-8", errors="replace"):
        p = ln.strip().split("/")
        if len(p) == 2 and p[0].strip() and p[1].strip():
            a, b = gv(p[0]), gv(p[1])
            if a is not None and b is not None:
                pairs.append(unit(b) - unit(a))       #: formal minus informal
    RD = unit(np.mean(pairs, 0))
    print("REGISTER DIRECTION from %d near-synonym difference vectors, no movement"
          " data involved" % len(pairs))

    fs = [gv(w) for w in open(os.path.join(B, "formal_seeds_100.txt"),
                              encoding="utf-8", errors="replace") if w.strip()]
    isd = [gv(w) for w in open(os.path.join(B, "informal_seeds_100.txt"),
                               encoding="utf-8", errors="replace") if w.strip()]
    SD = unit(np.mean([unit(v) for v in fs if v is not None], 0)
              - np.mean([unit(v) for v in isd if v is not None], 0))
    print("  seed-centroid version for comparison; the two agree at cos %+.3f"
          % float(RD @ SD))

    #: 1. HOW MUCH OF THE MOVEMENT AXIS IS REGISTER?
    axu = unit(ax)
    print("\n1. cos(movement axis, register direction)   %+.3f   (pairs)"
          % float(axu @ RD))
    print("   cos(movement axis, seed direction)       %+.3f"
          % float(axu @ SD))
    print("   The movement axis points at FALLING, the register direction at")
    print("   FORMAL, so a NEGATIVE cosine is the register reading: formal rises.")

    #: 2. THE TWO REGISTER MEASURES AGAINST EACH OTHER
    V = [w for w in EM if w in rate]
    gr = {w: float(unit(EM[w]) @ RD) for w in V}      #: + = formal
    cr = {w: float(rate[w]["register_level"]) for w in V}
    x = np.array([gr[w] for w in V]); y = np.array([cr[w] for w in V])
    print("\n2. GloVe register x coder register_level   rho %+.3f  (n %d)"
          % (spearmanr(x, y).statistic, len(V)))

    def resid(a, b):
        M = np.column_stack([np.ones(len(b)), b])
        return a - M @ np.linalg.lstsq(M, a, rcond=None)[0]
    rc = resid(y, x)      #: coder beyond GloVe
    rg = resid(x, y)      #: GloVe beyond coder

    o = np.argsort(rc)
    print("\n3. CODER REGISTER BEYOND GLOVE REGISTER")
    print("   coder says FORMAL, distribution does not:")
    print("     %s" % ", ".join(V[i] for i in o[-N_SHOW:][::-1]))
    print("   coder says INFORMAL, distribution does not:")
    print("     %s" % ", ".join(V[i] for i in o[:N_SHOW]))
    o = np.argsort(rg)
    print("\n4. GLOVE REGISTER BEYOND CODER REGISTER")
    print("   distribution says FORMAL, coder does not:")
    print("     %s" % ", ".join(V[i] for i in o[-N_SHOW:][::-1]))
    print("   distribution says INFORMAL, coder does not:")
    print("     %s" % ", ".join(V[i] for i in o[:N_SHOW]))

    #: 5. DOES EITHER RESIDUAL PREDICT MOVEMENT?
    RC = dict(zip(V, rc)); RG = dict(zip(V, rg))
    rows = KP2.fetch("en", False)
    Xn, cols, yy, g, site = [], [], [], [], []
    fq = {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in RC or r["p_base"] <= 0:
            continue
        if u not in fq:
            fq[u] = fpm(u, "en", "coca_fic")
        if not fq[u]:
            continue
        Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
        cols.append([gr[u], cr[u], RC[u], RG[u]])
        yy.append(1 if r["cls"] == "fall" else 0)
        g.append(u); site.append(hash((r["prompt"], r["base"], r["aligned"])))
    Xn = np.array(Xn); C = np.array(cols); yy = np.array(yy)
    g = np.array(g, object); site = np.array(site)
    print("\n5. DOES EITHER RESIDUAL PREDICT? held out by word, %s cells, %d words"
          % (f"{len(yy):,}", len(set(g))))
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    specs = {"nuisance only": None, "+ GloVe register": 0, "+ coder register": 1,
             "+ coder BEYOND glove": 2, "+ glove BEYOND coder": 3}
    out = {}
    for name, j in specs.items():
        M = Xn if j is None else np.column_stack([Xn, C[:, j]])
        p = np.zeros(len(yy))
        for tr, te in gkf.split(M, yy, groups=g):
            sc = StandardScaler().fit(M[tr])
            p[te] = LogisticRegression(max_iter=4000).fit(
                sc.transform(M[tr]), yy[tr]).predict_proba(sc.transform(M[te]))[:, 1]
        ps, ns = KP2.per_site_auc(site, yy, p)
        out[name] = [float(roc_auc_score(yy, p)), ps]
        print("   %-24s pooled %.4f   per-site %.4f" % (name, out[name][0], ps))
    b = out["nuisance only"]
    print("\n   over the floor:")
    for k in list(specs)[1:]:
        print("     %-24s %+.4f / %+.4f" % (k, out[k][0] - b[0], out[k][1] - b[1]))

    json.dump({"cos_axis_register_pairs": float(axu @ RD),
               "cos_axis_register_seeds": float(axu @ SD),
               "cos_pairs_seeds": float(RD @ SD),
               "rho_glove_coder": float(spearmanr(x, y).statistic),
               "auc": out}, open(os.path.join(K, "residual_en.json"), "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(os.path.join(K, "residual_en.json"), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
