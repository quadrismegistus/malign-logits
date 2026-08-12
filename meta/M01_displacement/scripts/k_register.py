"""Test the register hypothesis with an INDEPENDENT instrument: corpus genre counts.

    uv run python meta/M01_displacement/scripts/k_register.py

`k_axis` finds that the GloVe direction predicting FALL runs from vernacular
speech to academic and institutional prose. That naming rests on two things that
are not independent of each other: an embedding geometry, and reading word lists.
This tests the same hypothesis with neither.

THE INSTRUMENT IS A LOG FREQUENCY RATIO BETWEEN GENRES, which is the standard
corpus-linguistic operationalisation of register and is made of counts:

    register_index(w) = log10( fpm_spoken(w) / fpm_academic(w) )

positive for words commoner in speech than in academic prose. Two versions are
computed from independent corpus families and reported side by side, because a
result that holds in COCA and not in the BNC is a fact about COCA:

    COCA   log10(coca_spok / coca_acad)
    BNC    log10(bnc_spok  / bnc_acad)
    SOAP   log10(SOAP / coca_acad)     soap-opera dialogue, the most demotic
                                       register the BYU database carries

THE CAMPAIGN ALREADY DOCUMENTED THAT THIS IS WHAT THOSE COLUMNS MEASURE.
`k_frequency.py` records `coca_fic ~ bnc_acad` at 0.48, the bottom of a range
whose top is 0.97, and concludes in its own words that the structure of the
measures is REGISTER. The ingredient was already in the repo.

WHY THIS IS A REAL TEST AND NOT A RESTATEMENT. The GloVe axis, the coder scales
and this index share no inputs. GloVe is co-occurrence in Wikipedia and Gigaword;
the coder scales are an LLM's judgements of single words; this is the ratio of
two genre counts in COCA and the BNC. If the axis is register, this index should
(1) correlate with the axis position, and (2) predict falling out of sample.

THE NUISANCE PROBLEM IS SHARPER HERE THAN ANYWHERE ELSE IN K, and it is why the
ratio matters. Overall frequency already predicts movement, and spoken words are
frequent, so raw `fpm_spok` would recover the frequency effect and be named
register. A RATIO of two frequencies divides the overall-frequency component out
by construction, and `log10(fpm_coca_fic)` stays in the model as an explicit
control on top of that.

REPORTED WHETHER OR NOT IT WORKS. If the index correlates with the axis but does
not predict, the axis is register and register does not travel. If it predicts
but does not correlate with the axis, the axis is something else that happens to
co-vary. Both are informative and neither is the headline the hypothesis wants.
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
SEED = 20260812
INDICES = {"COCA_spok_over_acad": ("coca_spok", "coca_acad"),
           "BNC_spok_over_acad": ("bnc_spok", "bnc_acad"),
           "SOAP_over_coca_acad": ("SOAP", "coca_acad")}


def register_index(u, hi, lo):
    a, b = fpm(u, "en", hi), fpm(u, "en", lo)
    if not a or not b:
        return None      #: DROPPED, never imputed. A word absent from academic
    return math.log10(a / b)   #: prose is not a word with zero academic frequency


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    ax = json.load(open(os.path.join(K, "axis_en.json")))
    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    EM = {w: v for w, v in zip(z["words"], z["E"])}
    axis = np.array(ax["axis"], np.float32)

    #: 1. DOES THE INDEX AGREE WITH THE AXIS? No movement data involved.
    print("1. DOES A GENRE-RATIO REGISTER INDEX AGREE WITH THE GLOVE AXIS?")
    print("   Spearman over the verbs both cover; the axis is embedding geometry,")
    print("   the index is corpus counts, and they share no inputs.\n")
    print("   %-24s %8s %10s   %s" % ("index", "n words", "rho w/ axis", "rho w/ coder register_level"))
    RI = {}
    for name, (hi, lo) in INDICES.items():
        vals = {u: register_index(u, hi, lo) for u in EM}
        vals = {u: v for u, v in vals.items() if v is not None}
        RI[name] = vals
        common = [u for u in vals if u in EM]
        r = spearmanr([vals[u] for u in common],
                      [float(EM[u] @ axis) for u in common]).statistic
        rl = [u for u in common if u in rate]
        r2 = spearmanr([vals[u] for u in rl],
                       [rate[u]["register_level"] for u in rl]).statistic
        print("   %-24s %8d %+10.3f   %+.3f" % (name, len(common), r, r2))

    #: 2. DOES IT PREDICT? Same protocol as k_predict: held out by word.
    best = max(RI, key=lambda k: len(RI[k]))
    vals = RI[best]
    print("\n2. DOES IT PREDICT MOVEMENT? held out by WORD, %s, %d words covered"
          % (best, len(vals)))
    rows = KP2.fetch("en", False)
    Xn, Xr, Xc, y, g, site = [], [], [], [], [], []
    fq = {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in vals or u not in rate or r["p_base"] <= 0:
            continue
        if u not in fq:
            fq[u] = fpm(u, "en", "coca_fic")
        if not fq[u]:
            continue
        Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
        Xr.append([vals[u]])
        Xc.append([float(rate[u][s]) for s in A.SCALES])
        y.append(1 if r["cls"] == "fall" else 0)
        g.append(u); site.append(hash((r["prompt"], r["base"], r["aligned"])))
    Xn = np.array(Xn, float); Xr = np.array(Xr, float); Xc = np.array(Xc, float)
    y = np.array(y); g = np.array(g, object); site = np.array(site)
    print("   %s mover cells | %d words | fall %.3f" % (f"{len(y):,}", len(set(g)), y.mean()))

    rng = np.random.default_rng(SEED)
    words = sorted(set(g))
    perm = dict(zip(words, rng.permutation([vals[w] for w in words])))
    Xrs = np.array([[perm[u]] for u in g], float)

    gkf = GroupKFold(n_splits=KP2.FOLDS)
    specs = {"nuisance": Xn,
             "+ 7 coder scales": np.hstack([Xn, Xc]),
             "+ register index": np.hstack([Xn, Xr]),
             "+ register SHUFFLED": np.hstack([Xn, Xrs]),
             "+ both": np.hstack([Xn, Xc, Xr])}
    print("\n   %-22s %10s %10s   %10s %10s"
          % ("features", "LR pooled", "LR site", "GB pooled", "GB site"))
    out = {}
    for name, M in specs.items():
        pl, pt = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in gkf.split(M, y, groups=g):
            sc = StandardScaler().fit(M[tr])
            pl[te] = LogisticRegression(max_iter=4000).fit(
                sc.transform(M[tr]), y[tr]).predict_proba(sc.transform(M[te]))[:, 1]
            pt[te] = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=.08, random_state=SEED).fit(
                M[tr], y[tr]).predict_proba(M[te])[:, 1]
        a, b = KP2.per_site_auc(site, y, pl), KP2.per_site_auc(site, y, pt)
        out[name] = {"lr": [float(roc_auc_score(y, pl)), a[0]],
                     "gb": [float(roc_auc_score(y, pt)), b[0]], "n_sites": a[1]}
        print("   %-22s %10.4f %10.4f   %10.4f %10.4f"
              % (name, out[name]["lr"][0], a[0], out[name]["gb"][0], b[0]))

    print("\n   ONE COLUMN OF CORPUS COUNTS against SEVEN RATED SCALES, over nuisance:")
    for k, lab in (("lr", "logistic"), ("gb", "trees")):
        n = out["nuisance"][k]
        print("     %-9s coder %+.4f / %+.4f      register %+.4f / %+.4f"
              % (lab,
                 out["+ 7 coder scales"][k][0] - n[0], out["+ 7 coder scales"][k][1] - n[1],
                 out["+ register index"][k][0] - n[0], out["+ register index"][k][1] - n[1]))
        s = out["+ register SHUFFLED"][k]
        print("     %-9s register over its OWN shuffle  %+.4f / %+.4f"
              % ("", out["+ register index"][k][0] - s[0],
                 out["+ register index"][k][1] - s[1]))

    p = os.path.join(K, "register_en.json")
    json.dump({"indices": {k: len(v) for k, v in RI.items()}, "used": best,
               "n_cells": int(len(y)), "n_words": len(set(g)), "auc": out},
              open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
