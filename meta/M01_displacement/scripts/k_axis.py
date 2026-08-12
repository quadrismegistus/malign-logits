"""Name the direction. What is the GloVe axis along which alignment sorts words?

    uv run python meta/M01_displacement/scripts/k_axis.py en

`k_predict_embed` shows 300 unsupervised GloVe dimensions recover 21% of the
word-level headroom against the rated norms' 7%. So alignment sorts words along
something distributional semantics represents and the affective vocabulary does
not name. This tries to name it.

THE AXIS IS THE LOGISTIC COEFFICIENT VECTOR OVER THE GLOVE BLOCK, fitted with
log p_base and log frequency in the model as nuisance. Reading a direction off a
model without those controls would return the frequency axis, since frequency is
the strongest single predictor of movement and is itself strongly encoded in
GloVe geometry.

THREE THINGS HAVE TO HOLD BEFORE A DIRECTION IS WORTH NAMING, and all three are
reported whether or not they hold:

  1. STABILITY. The axis is refitted on each CV fold and the pairwise cosines
     between fold vectors are printed. A direction that changes between folds is
     not a direction; it is what a model does with noise. Anything below about
     0.9 should not be given a name.
  2. SUFFICIENCY. The one-dimensional projection is scored out of sample on its
     own. If the full 300-d model is far better, the structure is not one axis
     and naming a single direction misrepresents it.
  3. NON-IDENTITY WITH THE CONTROLS. Correlation of the axis position with log
     frequency and mean log p_base. A "semantic axis" that correlates 0.8 with
     frequency is the frequency axis wearing a new label.

NAMING IS DONE THREE WAYS, because each can mislead alone:

  poles          the verbs in OUR vocabulary at each end of the projection
  neighbours     the nearest words to the axis vector in the FULL GloVe
                 vocabulary, which is not restricted to our verbs and so is not
                 constrained to say something about our sample
  rated scales   correlation with each of the seven coder scales, which says
                 what the axis is NOT as much as what it is -- the whole point
                 is that the affective vocabulary missed it

WHAT WOULD MAKE THIS A NULL. If the poles are incoherent, the neighbours are
function words, and the scale correlations are all near zero, then the embedding
gain is real but distributed across many small directions with no single
interpretable axis. That is a legitimate outcome and should be reported as one
rather than dressed up by picking the most suggestive twenty words.
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

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
SEED = 20260812
N_SHOW = 30


def main(lang="en", name="glove"):
    """`name` selects the encoder. GloVe is English-only, so Chinese must use
    bge, and an en/zh comparison is then bge-to-bge -- never GloVe against bge,
    which would put the encoder and the language in the same contrast."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    from k_frequency import fpm

    z = np.load(os.path.join(K, "embed_%s_%s.npz" % (lang, name)), allow_pickle=True)
    EM = {w: v for w, v in zip(z["words"], z["E"])}
    rate = json.load(open(os.path.join(K, "ratings_%s.json" % lang)))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]
    rows = KP2.fetch(lang, False)

    Xn, Xe, y, g = [], [], [], []
    fq = {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in EM or r["p_base"] <= 0:
            continue
        if u not in fq:
            fq[u] = fpm(u, lang, "coca_fic" if lang == "en" else "SUBTLEX_CH")
        if not fq[u]:
            continue
        Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
        Xe.append(EM[u]); y.append(1 if r["cls"] == "fall" else 0); g.append(u)
    Xn = np.array(Xn, float); Xe = np.array(Xe, np.float32)
    y = np.array(y); g = np.array(g, object)
    print("[%s] %s mover cells | %d words | fall %.3f" % (lang, f"{len(y):,}", len(set(g)), y.mean()))

    #: 1. STABILITY -- refit per fold, compare the directions
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    axes, pred = [], np.zeros(len(y))
    for tr, te in gkf.split(Xn, y, groups=g):
        sc = StandardScaler().fit(Xn[tr])
        M = np.hstack([sc.transform(Xn[tr]), Xe[tr]])
        m = LogisticRegression(max_iter=5000, C=0.1).fit(M, y[tr])
        axes.append(m.coef_[0][2:])
        pred[te] = m.predict_proba(np.hstack([sc.transform(Xn[te]), Xe[te]]))[:, 1]
    U = np.array([a / np.linalg.norm(a) for a in axes])
    cs = [float(U[i] @ U[j]) for i in range(len(U)) for j in range(i + 1, len(U))]
    print("\n1. STABILITY  pairwise cosine between the %d fold axes: min %.3f, "
          "median %.3f, max %.3f" % (len(U), min(cs), float(np.median(cs)), max(cs)))
    print("   %s" % ("stable enough to name" if min(cs) >= 0.9 else
                     "NOT STABLE -- do not name a direction that moves between folds"))
    axis = U.mean(0); axis /= np.linalg.norm(axis)

    #: 2. SUFFICIENCY -- is one dimension most of it?
    proj_cell = Xe @ axis
    P1 = np.zeros(len(y))
    for tr, te in gkf.split(Xn, y, groups=g):
        sc = StandardScaler().fit(np.column_stack([Xn[tr, 0], Xn[tr, 1], proj_cell[tr]]))
        M = sc.transform(np.column_stack([Xn[tr, 0], Xn[tr, 1], proj_cell[tr]]))
        m = LogisticRegression(max_iter=4000).fit(M, y[tr])
        P1[te] = m.predict_proba(sc.transform(
            np.column_stack([Xn[te, 0], Xn[te, 1], proj_cell[te]])))[:, 1]
    print("\n2. SUFFICIENCY  held-out pooled AUC")
    print("   nuisance + the ONE axis      %.4f" % roc_auc_score(y, P1))
    print("   nuisance + all 300 GloVe     %.4f" % roc_auc_score(y, pred))

    #: 3. IS IT JUST FREQUENCY?
    words = sorted({u for u in g})
    proj = {u: float(EM[u] @ axis) for u in words}
    lf = {u: math.log10(fq[u]) for u in words}
    pbar = collections.defaultdict(list)
    for i, u in enumerate(g):
        pbar[u].append(Xn[i, 0])
    pb = {u: float(np.mean(v)) for u, v in pbar.items()}
    pv = np.array([proj[u] for u in words])
    print("\n3. NOT THE CONTROLS?  Spearman of axis position with")
    print("   log frequency                %+.3f" % spearmanr(pv, [lf[u] for u in words]).statistic)
    print("   mean log p_base              %+.3f" % spearmanr(pv, [pb[u] for u in words]).statistic)

    print("\n4. WHAT THE RATED SCALES SAY  (Spearman with axis position)")
    for s in A.SCALES:
        have = [u for u in words if u in rate]
        r = spearmanr([proj[u] for u in have], [rate[u][s] for u in have]).statistic
        print("   %-20s %+.3f" % (s, r))

    order = sorted(words, key=lambda u: proj[u])
    print("\n5. POLES OF THE AXIS, our verbs")
    print("   positive end = predicts FALLING")
    print("     %s" % ", ".join(order[-N_SHOW:][::-1]))
    print("   negative end = predicts RISING")
    print("     %s" % ", ".join(order[:N_SHOW]))

    #: ONLY GLOVE HAS A VOCABULARY TO LOOK THE AXIS UP IN. bge-m3 is an encoder,
    #: not a lookup table, so there is no full-vocabulary neighbour list for it
    #: and this section is skipped rather than faked from our own verb list --
    #: which would be the sample flattering itself.
    if name != "glove":
        print("\n6. NEAREST WORDS IN A FULL VOCABULARY: not available for %s, which"
              " is an encoder rather than a lookup table. Section 5 is the only"
              " naming evidence here." % name)
    else:
      print("\n6. NEAREST WORDS TO THE AXIS IN THE FULL GLOVE VOCABULARY")
      print("   (not restricted to our verbs, so not constrained to flatter the sample)")
      try:
        import gensim.downloader as api
        KV = api.load("glove-wiki-gigaword-300")
        a = axis / np.linalg.norm(axis)
        for sign, lab in ((1, "positive end (falling)"), (-1, "negative end (rising)")):
            nb = KV.similar_by_vector(sign * a.astype(np.float32), topn=25)
            print("   %-24s %s" % (lab, ", ".join(w for w, _ in nb)))
      except Exception as e:
        print("   unavailable: %s" % str(e)[:100])

    out = {"lang": lang, "encoder": name, "stability_min_cos": min(cs), "axis": axis.tolist(),
           "auc_one_axis": float(roc_auc_score(y, P1)),
           "auc_full": float(roc_auc_score(y, pred)),
           "poles_positive": order[-N_SHOW:][::-1], "poles_negative": order[:N_SHOW]}
    #: the glove/en file keeps its original name, because k_register and
    #: k_confound already read it; anything else is suffixed by encoder
    p = os.path.join(K, "axis_%s.json" % lang if name == "glove"
                     else "axis_%s_%s.json" % (lang, name))
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en",
                  sys.argv[2] if len(sys.argv) > 2 else "glove"))
