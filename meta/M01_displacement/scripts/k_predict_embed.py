"""Can a distributional embedding capture the word-level headroom the rated
norms miss?

    uv run python meta/M01_displacement/scripts/k_predict_embed.py en bge
    uv run python meta/M01_displacement/scripts/k_predict_embed.py zh bge

THE QUESTION, STATED AGAINST A MEASURED CEILING. `k_ceiling` splits each word's
own cells and uses the rate in one half to predict the other, which is the best
any function of the word alone can do. English verbs:

    oracle (word identity)          0.7027
    log p_base alone, same cells    0.5821
    headroom for any word feature  +0.1206

and `k_predict`'s eighteen rated norms buy +0.002 to +0.003 of that. So the
word carries real information and the rating instrument is not carrying it. This
asks whether a distributional representation does.

    embedding recovers most of 0.12  -> the information IS semantic and the
                                        seven affective axes are the wrong basis
    embedding also fails             -> the word-level signal is not semantic;
                                        stop building word features and build a
                                        word-by-site instrument

BOTH OUTCOMES ARE RESULTS. Neither is a failed experiment.

THE COMPARISON IS TO THE HEADROOM, NOT TO ZERO. An AUC of 0.62 sounds poor and
would be two thirds of everything available. The ceiling is low because ICC(1) is
0.131: 87% of the fall/rise variance is WITHIN a word across its sites and is
unreachable by any feature that is constant per word. The script therefore prints
the fraction of headroom recovered, not the raw AUC alone.

PCA IS FITTED INSIDE EACH FOLD. bge-m3 is 1024-d against ~2,760 verbs, and this
campaign has already produced a held-out R2 of -1.01 from 191 features on 1,041
words. Fitting the projection on all words before splitting would leak the test
words' distribution into the basis; the leak is unsupervised and usually small,
but "usually small" is not a thing to assert about a run whose headline is a
number near zero. The sweep over n_components is reported in full, because
picking the best k after seeing the scores would be selecting on the outcome --
the whole curve is the result.

ANISOTROPY IS WHY THE SWEEP MATTERS. bge-m3 puts two random bare verbs at median
cosine 0.53 (see `k_embed`), so a large share of the raw variance is a shared
direction carrying no word information. Whether the signal survives projection is
an empirical question and the low-k rows answer it.

THE SHUFFLED CONTROL PERMUTES EMBEDDING ROWS ACROSS WORDS within the eligible
set, exactly as the norms are permuted in `k_predict`, leaving every probability,
frequency, cell and site where it was.
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
COMPONENTS = (10, 25, 50, 100, 200)
SEED = 20260812

#: THIS SCRIPT IS NOT REPRODUCIBLE TO THE FOURTH DECIMAL AND THE SEED DOES NOT
#: FIX IT. HistGradientBoosting parallelises through OpenMP and its histogram
#: construction is thread-order dependent, so `random_state` controls the
#: subsampling and not the result. Five identical runs gave k=50 tree increments
#: of +0.0256, +0.0223, +0.0216, +0.0218 and +0.0231 -- a 0.0040 spread on a
#: 0.0229 mean. The LOGISTIC rows are byte-identical across the same runs, which
#: is how the cause was localised; it is NOT the ClickHouse row-order defect
#: k_ceiling had, and adding ORDER BY to the fetch changed nothing.
#:
#: SO QUOTE A RANGE FROM SEVERAL RUNS, NEVER A SINGLE TREE FIGURE. Setting
#: OMP_NUM_THREADS=1 would make it reproducible at a large cost in runtime and
#: would report one arbitrary thread schedule as though it were the answer; the
#: spread is the honest object and P section 3 carries it.
#:
#: AND THIS SCRIPT OVERWRITES ITS RESULTS FILE ON EVERY RUN. Four determinism
#: runs silently replaced the committed predict_embed_en_glove.json with their
#: own draws; git had the cited version and it was restored. Re-running to check
#: a number is not a read-only act.


def load_embedding(lang, name):
    p = os.path.join(K, "embed_%s_%s.npz" % (lang, name))
    z = np.load(p, allow_pickle=True)
    print("  embedding %s: %d words x %d dims | synonym gap %+.4f | anisotropy %.4f"
          % (name, z["E"].shape[0], z["E"].shape[1], float(z["syn_gap"]),
             float(z["anisotropy"])))
    return {w: v for w, v in zip(z["words"], z["E"])}, z["E"].shape[1]


def build(rows, lang, t2u, EM, shuffle_seed=None):
    """-> Xn (nuisance), Xe (embedding), y, groups, sites."""
    from k_frequency import fpm
    meas = "coca_fic" if lang == "en" else "SUBTLEX_CH"
    tab = EM
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        ks = sorted(EM); vs = [EM[k] for k in ks]
        rng.shuffle(vs)
        tab = dict(zip(ks, vs))
    Xn, Xe, y, g, site, fq = [], [], [], [], [], {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in EM or r["p_base"] <= 0:
            continue
        if u not in fq:
            fq[u] = fpm(u, lang, meas)
        if not fq[u]:
            continue
        Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
        Xe.append(tab[u])
        y.append(1 if r["cls"] == "fall" else 0)
        g.append(u); site.append(hash((r["prompt"], r["base"], r["aligned"])))
    return (np.array(Xn, float), np.array(Xe, np.float32), np.array(y),
            np.array(g, object), np.array(site))


def run(Xn, Xe, y, g, site, k, tag):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    pl, pt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in gkf.split(Xn, y, groups=g):
        #: PCA fitted on the TRAINING rows only -- see the module docstring
        pca = PCA(n_components=k, random_state=SEED).fit(Xe[tr])
        Mtr = np.hstack([Xn[tr], pca.transform(Xe[tr])])
        Mte = np.hstack([Xn[te], pca.transform(Xe[te])])
        sc = StandardScaler().fit(Mtr)
        m = LogisticRegression(max_iter=4000).fit(sc.transform(Mtr), y[tr])
        pl[te] = m.predict_proba(sc.transform(Mte))[:, 1]
        t = HistGradientBoostingClassifier(max_iter=300, learning_rate=.08,
                                           random_state=SEED).fit(Mtr, y[tr])
        pt[te] = t.predict_proba(Mte)[:, 1]
    out = {}
    for nm, p in (("logistic", pl), ("trees", pt)):
        ps, ns = KP2.per_site_auc(site, y, p)
        out[nm] = (float(roc_auc_score(y, p)), ps, ns)
    return out


def main(lang, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]
    EM, dim = load_embedding(lang, name)
    rows = KP2.fetch(lang, False)
    Xn, Xe, y, g, site = build(rows, lang, t2u, EM)
    print("  %s mover cells | %d words | %d sites | fall %.3f"
          % (f"{len(y):,}", len(set(g)), len(set(site)), y.mean()))

    ceil = json.load(open(os.path.join(K, "ceiling_%s_verbs.json" % lang))) \
        if os.path.exists(os.path.join(K, "ceiling_%s_verbs.json" % lang)) else None

    #: the floor, on exactly these rows
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    pn = np.zeros(len(y))
    for tr, te in gkf.split(Xn, y, groups=g):
        sc = StandardScaler().fit(Xn[tr])
        m = LogisticRegression(max_iter=4000).fit(sc.transform(Xn[tr]), y[tr])
        pn[te] = m.predict_proba(sc.transform(Xn[te]))[:, 1]
    nps, nns = KP2.per_site_auc(site, y, pn)
    floor = float(roc_auc_score(y, pn))
    print("\n  NUISANCE FLOOR on these rows      pooled %.4f  per-site %.4f (%d sites)"
          % (floor, nps, nns))
    if ceil:
        #: the ceiling's own p_base AUC, measured on ITS cells with ITS split, is
        #: the only thing the oracle is comparable to. Differencing it against
        #: this run's `floor` would subtract two numbers from two populations.
        print("  CEILING from k_ceiling            oracle %.4f  |  p_base on the "
              "ceiling's own cells %.4f  |  headroom %+.4f"
              % (ceil["oracle_auc"], ceil["p_base_auc"],
                 ceil["oracle_auc"] - ceil["p_base_auc"]))

    Xns, Xes, ys, gs, ss = build(rows, lang, t2u, EM, shuffle_seed=SEED)
    print("\n  %-6s %-10s %8s %10s   %8s %10s"
          % ("k", "model", "pooled", "per-site", "shuf pool", "shuf site"))
    res = {}
    for k in COMPONENTS:
        if k > min(dim, len(set(g)) // 4):
            print("  %-6d skipped: %d components against %d words is not a fit"
                  % (k, k, len(set(g))))
            continue
        real = run(Xn, Xe, y, g, site, k, "real")
        shuf = run(Xns, Xes, ys, gs, ss, k, "shuffled")
        for nm in ("logistic", "trees"):
            print("  %-6d %-10s %8.4f %10.4f   %8.4f %10.4f     adds %+.4f / %+.4f"
                  % (k, nm, real[nm][0], real[nm][1], shuf[nm][0], shuf[nm][1],
                     real[nm][0] - shuf[nm][0], real[nm][1] - shuf[nm][1]))
        res[k] = {"real": {n: list(v) for n, v in real.items()},
                  "shuffled": {n: list(v) for n, v in shuf.items()}}

    if ceil:
        #: THE BEST ROW IS SELECTED AFTER SEEING THE SCORES, so this fraction is
        #: an upper bound on what the embedding recovers, not an estimate of it.
        #: The full sweep above is the result; this line is a reading aid.
        head = ceil["oracle_auc"] - ceil["p_base_auc"]
        best = max((v["real"][m][0] - v["shuffled"][m][0]
                    for v in res.values() for m in ("logistic", "trees")),
                   default=float("nan"))
        print("\n  FRACTION OF THE WORD-LEVEL HEADROOM RECOVERED (upper bound: the")
        print("  best of %d sweep rows, chosen after seeing them)" % (2 * len(res)))
        print("    oracle beats p_base by                    %+.4f" % head)
        print("    best embedding beats its OWN shuffle by   %+.4f" % best)
        print("    recovered                                  %.0f%%" % (100 * best / head))
    p = os.path.join(K, "predict_embed_%s_%s.json" % (lang, name))
    json.dump({"lang": lang, "embedding": name, "dim": int(dim),
               "n_cells": int(len(y)), "n_words": len(set(g)),
               "nuisance_floor": floor, "components": res}, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en",
                  sys.argv[2] if len(sys.argv) > 2 else "bge"))
