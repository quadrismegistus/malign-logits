"""Plan K prediction: given only a word's rated properties, can we predict which
way alignment will move it -- for a word the model has never seen?

    uv run python meta/M01_displacement/scripts/k_predict.py            all prompts
    uv run python meta/M01_displacement/scripts/k_predict.py --marked   MARKED only

Everything K has done so far is a correlation or a group contrast, all in-sample.
This asks the harder question, and the answer is allowed to be no.

THE CASE IS A CELL, THE HELD-OUT UNIT IS A WORD. Outcome is fall (1) vs rise (0)
under the canonical rule; non-movers are excluded because the question is which
DIRECTION, not whether. Cross-validation is GroupKFold on the word, so no word
appears in both train and test: a model that has memorised `murder falls` scores
nothing for it. This is the only split that answers the question as posed, and it
is also immune to the word-level pseudo-replication that has bitten this campaign
repeatedly -- with a random split, 5-fold CV on 2.5M cells over 20k words would
report the training fit.

CELLS PER WORD ARE CAPPED. The falling is carried by the words appearing in the
most cells (word-weighted mean net +0.063, cell-weighted -0.020), so uncapped the
fit is dominated by a few hundred high-frequency words. The cap is applied by a
deterministic hash so the sample is reproducible and is not the word's first N
prompts, which would be a corpus-order artefact.

THE NUISANCE MODEL IS THE BASELINE, NOT ZERO. [3652] named the tautology channel:
a word's base probability is its distance to the eviction boundary, so p_base
predicts movement under ANY perturbation of the distribution, including random
noise. X_metonymy records the same thing as a -0.33 floor. The number that means
something is therefore the AUC INCREMENT of the norms OVER {p_base, frequency,
POS}, never the norms' AUC on its own.

FOUR MODELS, BECAUSE ADDITIVE AND NON-ADDITIVE ARE DIFFERENT CLAIMS:

    nuisance      p_base + frequency + POS                     the floor
    additive      + the seven norms, linear in the log-odds
    interactions  + all 21 pairwise products of the norms
    trees         HistGradientBoosting on the same features

Logistic regression does not discover interactions; it fits what you declare. The
tile figures show the falling concentrated in a CORNER of charge x harm, which is
exactly the shape an additive model cannot represent, so the tree is here as the
function class that can find it without being told where to look. **The gap
between `additive` and `trees` is the measurement of how non-additive the
structure is.**

TWO AUCs, ANSWERING DIFFERENT QUESTIONS:

    pooled      over all held-out cells. Includes between-site variation, so a
                model can score well by learning which SITES have many fallers.
    per-site    computed within each (prompt, base, aligned) cell and averaged.
                Asks: given the fifty words competing at THIS site, does the
                model rank which of them falls? All site and model-pair variance
                is removed. This is the sharper test and the one that matches the
                displacement claim.

A NEGATIVE CONTROL RUNS EVERY TIME. The ratings are shuffled across words and the
whole pipeline re-run. Shuffled AUC must land at the nuisance floor. Three tests
in this campaign have been reported before anyone checked they could fail.

POS IS OUT OF CONTEXT AND THAT IS A KNOWN DEFECT. `fields._byu()` returns the
most frequent reading of an ambiguous form, so `snarled` is tagged from the
commoner sense rather than the speech verb it is in these prompts. Same limit as
the coder ratings, and it biases toward the null.
"""
import collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import k_analysis as A, k_population as KP
from k_frequency import fpm
from malign_logits import fields as FL

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
SCALES = A.SCALES
CAP = 60           #: cells per word, sampled by hash
FOLDS = 5
SEED = 20260812
MIN_SITE = 12      #: movers needed at a site before its within-site AUC is used
POS_CLASSES = ("nn", "vv", "jj", "rr", "np", "other")


def pos_of(w):
    e = FL._byu().get(w.strip().lower())
    if e is None:
        return "other"
    for c in POS_CLASSES[:-1]:
        if e[1].startswith(c):
            return c
    return "other"


def fetch(marked):
    """One row per mover cell, capped per word by a reproducible hash order."""
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    ep = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                     for b, a in KP.reps("en"))
    role = " AND pair_role='MARKED'" if marked else ""
    return A.q("""
      SELECT word, prompt, base, aligned, cls, p_base, p_aligned FROM (
        SELECT *, row_number() OVER (PARTITION BY word
                 ORDER BY cityHash64(word, prompt, base, aligned)) rw FROM (
          SELECT m.word word, m.prompt prompt, m.base base, m.aligned aligned,
                 m.cls cls, m.p_base p_base, m.p_aligned p_aligned,
            row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                               ORDER BY m.p_base DESC) rb,
            row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                               ORDER BY m.p_aligned DESC) ra
          FROM %s.movement m
          INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                      WHERE status='ACTIVE' AND language='en'%s) p ON m.prompt=p.prompt
          WHERE m.rule='canonical' AND (%s))
        WHERE (rb<=50 OR ra<=50) AND cls IN ('fall','rise'))
      WHERE rw <= %d""" % (A.DB, A.DB, role, ep, CAP))


#: THE ELIGIBILITY GATE, and it is most of what `p_base alone` was measuring.
#: The canonical rule fires a FALL only when p_base >= 0.003 and a RISE only when
#: p_aligned >= 0.003, so a word below the floor in the base arm CANNOT be scored
#: a faller however hard alignment pushes it. Base probability therefore predicts
#: direction partly by construction -- [3652]'s tautology channel exactly.
#: `--eligible` keeps only cells above the floor in BOTH arms, where the rule
#: could have returned either answer, and that is the population in which a
#: predictor is answering the question rather than reciting the gate.
MIN_PROB = 0.003


def design(rows, R, t2u, shuffle_seed=None, eligible=False):
    """Feature matrix. `shuffle_seed` permutes the RATINGS across words, leaving
    every movement, probability, frequency and POS value exactly where it was."""
    rate = R
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        ks = list(R); vs = [R[k] for k in ks]
        rng.shuffle(vs); rate = dict(zip(ks, vs))
    unit, keep = {}, []
    for i, r in enumerate(rows):
        u = t2u.get(r["word"])
        if eligible and not (r["p_base"] >= MIN_PROB and r["p_aligned"] >= MIN_PROB):
            continue
        if u in rate and r["p_base"] > 0:
            unit[i] = u; keep.append(i)
    fr = {}
    for u in set(unit.values()):
        fr[u] = fpm(u, "en", "coca_fic")
    med = float(np.median([v for v in fr.values() if v]))
    X, y, g, site = [], [], [], []
    for i in keep:
        r = rows[i]; u = unit[i]; f = fr[u]
        #: NUISANCE FIRST, so the column block is easy to slice for the floor model
        row = [np.log10(r["p_base"]), np.log10(f if f else med), 0.0 if f else 1.0]
        p = pos_of(u)
        row += [1.0 if p == c else 0.0 for c in POS_CLASSES[:-1]]
        row += [float(rate[u][s]) for s in SCALES]
        X.append(row); y.append(1 if r["cls"] == "fall" else 0)
        g.append(u); site.append((r["prompt"], r["base"], r["aligned"]))
    return (np.array(X, float), np.array(y), np.array(g, object),
            np.array([hash(s) for s in site]))


NUIS = 3 + len(POS_CLASSES) - 1     #: p_base, log fpm, missing-flag, 5 POS dummies


def add_interactions(X):
    """All 21 pairwise products of the seven norms, centred first so the product
    is an interaction and not a proxy for the main effects."""
    N = X[:, NUIS:]
    C = N - N.mean(0)
    cols = [C[:, i] * C[:, j] for i in range(C.shape[1]) for j in range(i + 1, C.shape[1])]
    return np.column_stack([X] + cols)


def per_site_auc(site, y, p):
    """Mean AUC within (prompt, base, aligned), over sites with both classes."""
    from sklearn.metrics import roc_auc_score
    idx = collections.defaultdict(list)
    for i, s in enumerate(site):
        idx[s].append(i)
    out = []
    for s, ii in idx.items():
        if len(ii) < MIN_SITE:
            continue
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, p[ii]))
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def evaluate(X, y, g, site, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Xi = add_interactions(X)
    specs = {
        "p_base alone":  ("lr", X[:, :1]),
        "nuisance":      ("lr", X[:, :NUIS]),
        "norms only":    ("lr", X[:, NUIS:]),
        "additive":      ("lr", X),
        "interactions":  ("lr", Xi),
        "trees nuisance":("gb", X[:, :NUIS]),
        "trees":         ("gb", X),
    }
    out = {}
    gkf = GroupKFold(n_splits=FOLDS)
    for name, (kind, M) in specs.items():
        pred = np.zeros(len(y))
        for tr, te in gkf.split(M, y, groups=g):
            if kind == "lr":
                sc = StandardScaler().fit(M[tr])
                m = LogisticRegression(max_iter=2000, C=1.0)
                m.fit(sc.transform(M[tr]), y[tr])
                pred[te] = m.predict_proba(sc.transform(M[te]))[:, 1]
            else:
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.08,
                                                   max_leaf_nodes=31, random_state=SEED)
                m.fit(M[tr], y[tr])
                pred[te] = m.predict_proba(M[te])[:, 1]
        ps, ns = per_site_auc(site, y, pred)
        out[name] = (roc_auc_score(y, pred), ps, ns)
        print("    %-14s  pooled AUC %.4f   per-site AUC %.4f  (%d sites)"
              % (name, out[name][0], ps, ns))
    return out


def scalar(rows, R, t2u):
    """The continuous version: predict HOW FAR alignment moves the word, not
    which side of the rule it lands on.

    Outcome is log10(p_aligned / p_base), which is the quantity the canonical
    rule thresholds -- a fall is log ratio < log10(0.5). Two things change:
    magnitude is kept instead of discarded, and the 59% of cells the rule calls
    `still` come back into the population instead of being dropped. Dropping
    non-movers is a defensible answer to "which direction", but it means the
    binary model never sees the majority of the data and cannot be asked whether
    the norms predict a word STAYING PUT.

    Restricted to p_base >= MIN_PROB. Below the floor the ratio is a ratio of two
    numbers the instrument does not resolve, and it explodes.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    EPS = 1e-6
    keep = [r for r in rows if r["p_base"] >= MIN_PROB]
    X, y, g, site = [], [], [], []
    fr = {}
    for r in keep:
        u = t2u.get(r["word"])
        if u not in R:
            continue
        if u not in fr:
            fr[u] = fpm(u, "en", "coca_fic")
    med = float(np.median([v for v in fr.values() if v]))
    for r in keep:
        u = t2u.get(r["word"])
        if u not in R:
            continue
        f = fr[u]
        row = [np.log10(r["p_base"]), np.log10(f if f else med), 0.0 if f else 1.0]
        p = pos_of(u)
        row += [1.0 if p == c else 0.0 for c in POS_CLASSES[:-1]]
        row += [float(R[u][sc_]) for sc_ in SCALES]
        X.append(row)
        y.append(np.log10((r["p_aligned"] + EPS) / (r["p_base"] + EPS)))
        g.append(u); site.append((r["prompt"], r["base"], r["aligned"]))
    X = np.array(X, float); y = np.array(y); g = np.array(g, object)
    site = np.array([hash(s) for s in site])
    print("\n  SCALAR OUTCOME = log10(p_aligned / p_base), p_base >= %.3f" % MIN_PROB)
    print("    %s cells | %d words | %d sites | mean %+.3f sd %.3f"
          % (f"{len(y):,}", len(set(g)), len(set(site)), y.mean(), y.std()))
    gkf = GroupKFold(n_splits=FOLDS)
    specs = {"nuisance":       ("ridge", X[:, :NUIS]),
             "additive":       ("ridge", X),
             "interactions":   ("ridge", add_interactions(X)),
             "trees nuisance": ("gb",    X[:, :NUIS]),
             "trees":          ("gb",    X)}
    res = {}
    for name, (kind, M) in specs.items():
        pred = np.zeros(len(y))
        for tr, te in gkf.split(M, y, groups=g):
            if kind == "ridge":
                sc_ = StandardScaler().fit(M[tr])
                m = Ridge(alpha=1.0).fit(sc_.transform(M[tr]), y[tr])
                pred[te] = m.predict(sc_.transform(M[te]))
            else:
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.08,
                                                  random_state=SEED).fit(M[tr], y[tr])
                pred[te] = m.predict(M[te])
        idx = collections.defaultdict(list)
        for i, sv in enumerate(site):
            idx[sv].append(i)
        rhos = [spearmanr(y[ii], pred[ii]).statistic for ii in idx.values()
                if len(ii) >= MIN_SITE]
        rhos = [r for r in rhos if r == r]
        res[name] = (r2_score(y, pred), float(np.mean(rhos)), len(rhos))
        print("    %-14s  held-out R2 %+.4f   mean per-site Spearman %+.4f  (%d sites)"
              % (name, res[name][0], res[name][1], res[name][2]))
    print("    %-14s  norms add  R2 %+.4f / rho %+.4f (linear), %+.4f / %+.4f (trees)"
          % ("", res["additive"][0] - res["nuisance"][0],
             res["additive"][1] - res["nuisance"][1],
             res["trees"][0] - res["trees nuisance"][0],
             res["trees"][1] - res["trees nuisance"][1]))
    return {k: list(v) for k, v in res.items()}


def main():
    marked = "--marked" in sys.argv
    R = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    rows = fetch(marked)
    print("\n%s PROMPTS%s | %s mover cells fetched (cap %d per word)"
          % ("MARKED" if marked else "ALL",
             ", BOTH ARMS ABOVE THE %.3f FLOOR" % MIN_PROB if "--eligible" in sys.argv else "",
             f"{len(rows):,}", CAP))
    elig = "--eligible" in sys.argv
    X, y, g, site = design(rows, R, t2u, eligible=elig)
    print("  %s cells joined to a rating | %d distinct words | %d sites | fall rate %.3f"
          % (f"{len(y):,}", len(set(g)), len(set(site)), y.mean()))
    print("  features: %d nuisance (log p_base, log fpm, missing flag, %d POS) + %d norms"
          % (NUIS, len(POS_CLASSES) - 1, len(SCALES)))
    print("\n  REAL RATINGS, %d-fold GroupKFold held out by WORD" % FOLDS)
    real = evaluate(X, y, g, site, "real")
    print("\n  NEGATIVE CONTROL, ratings shuffled across words")
    Xs, ys, gs, ss = design(rows, R, t2u, shuffle_seed=SEED, eligible=elig)
    shuf = evaluate(Xs, ys, gs, ss, "shuffled")
    #: TWO INCREMENTS, AND THE SECOND IS THE ONE THAT MEANS ANYTHING.
    #: vs its OWN function class without the norms -- comparing the tree to the
    #: logistic nuisance model would credit the norms for the tree's nonlinear
    #: use of p_base, which the shuffled control shows is most of its AUC.
    #: vs the SAME model on shuffled ratings -- the only contrast in which
    #: everything except the word-meaning link is held fixed.
    print("\n  WHAT THE NORMS ADD (pooled / per-site)")
    print("    %-14s  %-19s  %s" % ("model", "over same class,", "over ITSELF on"))
    print("    %-14s  %-19s  %s" % ("", "no norms", "shuffled ratings"))
    BASE = {"additive": "nuisance", "interactions": "nuisance",
            "norms only": "nuisance", "trees": "trees nuisance"}
    for k in ("norms only", "additive", "interactions", "trees"):
        b = BASE[k]
        print("    %-14s  %+.4f / %+.4f    %+.4f / %+.4f"
              % (k, real[k][0] - real[b][0], real[k][1] - real[b][1],
                 real[k][0] - shuf[k][0], real[k][1] - shuf[k][1]))
    #: COEFFICIENTS FROM THE WHOLE SAMPLE, standardised so they are comparable,
    #: and IN-SAMPLE -- they describe the fit, they do not evidence prediction.
    #: The AUC table above is the evidence. SEs are omitted deliberately: the
    #: cells are clustered by word and by site and an unclustered SE here would
    #: be wrong by an order of magnitude in the flattering direction.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    names = (["log p_base", "log fpm", "fpm missing"]
             + ["POS " + c for c in POS_CLASSES[:-1]] + list(SCALES))
    print("\n  ADDITIVE COEFFICIENTS, standardised, in-sample (positive = predicts FALL)")
    for n, c in sorted(zip(names, m.coef_[0]), key=lambda t: -abs(t[1])):
        print("    %-20s %+.4f%s" % (n, c, "   <- nuisance" if n in names[:8] else ""))
    coefs = dict(zip(names, [float(c) for c in m.coef_[0]]))

    sc_res = scalar(rows, R, t2u) if "--scalar" in sys.argv else None

    out = os.path.join(K, "predict_%s%s.json"
                       % ("marked" if marked else "all", "_eligible" if elig else ""))
    json.dump({"_cap": CAP, "_folds": FOLDS, "_n_cells": int(len(y)),
               "_n_words": len(set(g)), "_n_sites": len(set(site)),
               "_fall_rate": float(y.mean()), "_marked": marked, "_eligible": elig,
               "coefficients_standardised_in_sample": coefs, "scalar": sc_res,
               "real": {k: list(v) for k, v in real.items()},
               "shuffled": {k: list(v) for k, v in shuf.items()}},
              open(out, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
