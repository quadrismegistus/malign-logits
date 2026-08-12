"""How much of the movement axis is word length?

    uv run python meta/M01_displacement/scripts/k_length.py en glove
    uv run python meta/M01_displacement/scripts/k_length.py zh bge

Chinese replicates the English axis on a different historical seam -- English
splits Germanic against Latinate, Chinese splits monosyllabic native words
against bisyllabic literary compounds -- which rules out etymology as the
explanation. It does NOT rule out length, because the falling pole is short in
both languages. This asks how much of the axis IS length, three ways.

    1. VARIANCE. Regress the axis projection, one scalar per word, on length and
       on the rest of the bundle. R2 is the share of the axis's word-level
       variance that each accounts for. Cheap and direct.
    2. GEOMETRY. Fit the direction in embedding space that best predicts length,
       and take its cosine with the axis. This asks whether the axis and length
       are the same direction, which is a stronger claim than sharing variance.
    3. PREDICTION. Project length out of the axis and re-run the held-out test.
       If the length-free axis still beats its shuffle, the axis is not length.
       THIS IS THE ONE THAT MATTERS -- the first two describe the vector, only
       this one says whether what remains does any work.

LENGTH IS CHARACTERS, AND IN CHINESE THAT IS THE RIGHT UNIT. A Chinese character
is roughly a syllable and a morpheme, so character count is the monosyllabic /
bisyllabic contrast directly. In English characters and syllables correlate at
0.83, so both are reported and neither is treated as the definitive measure.

THE LENGTH DIRECTION IS FITTED ON THE FULL VERB VOCABULARY, not on the words that
move, so it is not tuned to the outcome. It uses no probability data at all.

WHAT WOULD REFUTE THE AXIS. If the length-residualised projection stops beating
its own shuffle, then "the axis" is a long-word/short-word direction and every
register reading of it is decoration. That outcome is reported as flatly as the
other one.
"""
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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def syllables(w):
    import re
    w = w.lower().strip()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(n, 1)


def main(lang="en", name="glove"):
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    z = np.load(os.path.join(K, "embed_%s_%s.npz" % (lang, name)), allow_pickle=True)
    EM = {w: v.astype(np.float64) for w, v in zip(z["words"], z["E"])}
    af = "axis_%s.json" % lang if name == "glove" else "axis_%s_%s.json" % (lang, name)
    ax = unit(np.array(json.load(open(os.path.join(K, af)))["axis"]))
    rate = json.load(open(os.path.join(K, "ratings_%s.json" % lang)))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]

    W = sorted(EM)
    E = np.array([unit(EM[w]) for w in W])
    L = np.array([float(len(w.strip())) for w in W])
    S = (np.array([float(syllables(w)) for w in W]) if lang == "en" else L)
    P = E @ ax                                   #: axis projection, + = falls
    print("[%s/%s] %d verbs | length mean %.2f  sd %.2f" % (lang, name, len(W), L.mean(), L.std()))

    #: 1. VARIANCE
    def r2(y, X):
        X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return 1 - r.var() / y.var()
    conc = np.array([float(rate[w]["concreteness"]) if w in rate else np.nan for w in W])
    ok = ~np.isnan(conc)
    print("\n1. VARIANCE OF THE AXIS PROJECTION EXPLAINED (R2, word level)")
    print("   length alone                       %.4f" % r2(P, [L]))
    print("   syllables alone                    %.4f" % r2(P, [S]))
    print("   length + syllables                 %.4f" % r2(P, [L, S]))
    print("   concreteness alone                 %.4f" % r2(P[ok], [conc[ok]]))
    print("   length + syllables + concreteness  %.4f" % r2(P[ok], [L[ok], S[ok], conc[ok]]))
    print("   rho(projection, length)            %+.3f" % spearmanr(P, L).statistic)

    #: 2. GEOMETRY -- the direction in embedding space that encodes length
    ld = unit(Ridge(alpha=1.0).fit(E, L).coef_)
    print("\n2. GEOMETRY")
    print("   cos(axis, length direction)        %+.3f" % float(ax @ ld))
    print("   the length direction predicts length at R2 %.3f in sample"
          % r2(L, [E @ ld]))

    #: THE AXIS WITH LENGTH PROJECTED OUT, renormalised
    ax_free = unit(ax - (ax @ ld) * ld)
    print("   cos(axis, length-free axis)        %+.3f" % float(ax @ ax_free))
    Pf = E @ ax_free
    print("   rho(length-free projection, length) %+.3f" % spearmanr(Pf, L).statistic)

    #: 3. PREDICTION -- does what is left do any work?
    rows = KP2.fetch(lang, False)
    proj = dict(zip(W, P)); projf = dict(zip(W, Pf))
    rng = np.random.default_rng(SEED)
    sh = dict(zip(W, rng.permutation(Pf)))
    meas = "coca_fic" if lang == "en" else "SUBTLEX_CH"
    Xn, c, y, g, site, fq = [], [], [], [], [], {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in proj or r["p_base"] <= 0:
            continue
        if u not in fq:
            fq[u] = fpm(u, lang, meas)
        if not fq[u]:
            continue
        Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
        c.append([proj[u], projf[u], sh[u], float(len(u.strip()))])
        y.append(1 if r["cls"] == "fall" else 0)
        g.append(u); site.append(hash((r["prompt"], r["base"], r["aligned"])))
    Xn = np.array(Xn); C = np.array(c); y = np.array(y)
    g = np.array(g, object); site = np.array(site)
    print("\n3. PREDICTION, held out by word, %s cells, %d words"
          % (f"{len(y):,}", len(set(g))))
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    specs = {"nuisance only": None, "+ length alone": 3, "+ axis": 0,
             "+ LENGTH-FREE axis": 1, "+ length-free SHUFFLED": 2}
    out = {}
    for nm, j in specs.items():
        M = Xn if j is None else np.column_stack([Xn, C[:, j]])
        p = np.zeros(len(y))
        for tr, te in gkf.split(M, y, groups=g):
            sc = StandardScaler().fit(M[tr])
            p[te] = LogisticRegression(max_iter=4000).fit(
                sc.transform(M[tr]), y[tr]).predict_proba(sc.transform(M[te]))[:, 1]
        ps, _ = KP2.per_site_auc(site, y, p)
        out[nm] = [float(roc_auc_score(y, p)), ps]
        print("   %-26s pooled %.4f   per-site %.4f" % (nm, out[nm][0], ps))
    b = out["nuisance only"]
    print("\n   over the floor          %s" % "  ".join(
        "%s %+.4f/%+.4f" % (k.strip("+ "), out[k][0] - b[0], out[k][1] - b[1])
        for k in ("+ length alone", "+ axis")))
    f, s = out["+ LENGTH-FREE axis"], out["+ length-free SHUFFLED"]
    print("   LENGTH-FREE AXIS over its OWN shuffle   %+.4f / %+.4f"
          % (f[0] - s[0], f[1] - s[1]))

    json.dump({"lang": lang, "encoder": name,
               "r2_length": r2(P, [L]), "r2_len_syl": r2(P, [L, S]),
               "cos_axis_length_dir": float(ax @ ld),
               "cos_axis_lengthfree": float(ax @ ax_free), "auc": out},
              open(os.path.join(K, "length_%s_%s.json" % (lang, name)), "w"), indent=1)
    print("\n  -> results/k/length_%s_%s.json" % (lang, name))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en",
                  sys.argv[2] if len(sys.argv) > 2 else "glove"))
