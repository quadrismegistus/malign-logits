"""Plan K prediction: given only a word's rated properties, can we predict which
way alignment will move it -- for a word never seen in training?

    uv run python meta/M01_displacement/scripts/k_predict.py --lang en
    uv run python meta/M01_displacement/scripts/k_predict.py --lang zh
    ... --marked      transgressive sites only
    ... --eligible    both arms above the rule's probability floor
    ... --scalar      also run the continuous outcome, on ALL cells
    ... --cv 5|20|loo cross-validation, always grouped by word (default 5)
    ... --curve       held-out AUC against training-set size

LANGUAGES ARE RUN SEPARATELY AND NEVER POOLED. Different tokenizers, different
rating instruments, different frequency norms, different external lexicons, and
Chinese has one external norm where English has eight. A pooled fit would let the
larger, better-covered language set the coefficients and then report them as
though they held for both.

LEXICAL VERBS ONLY, SO POS IS NOT A CONFOUND. English keeps CLAWS `vv*`, which
is the lexical verb series and deliberately excludes be/do/have and the modals;
Chinese keeps jieba `v*`. The previous version carried POS as dummies and that
was worse than it looked: `np` was a DEAD COLUMN (this lexicon does not tag
proper nouns np at all, so its coefficient was exactly zero), and 35% of the
vocabulary is absent from the lexicon entirely and got pooled with prepositions
and interjections. Restricting to one homogeneous class removes the question
rather than modelling it badly.

EIGHTEEN FEATURES FOR ENGLISH, IN THREE BLOCKS, and the blocks are what the
comparison is over:

    nuisance   log p_base, log fpm                            2
    coder      the seven K scales + valence/charge/concreteness
               extremity, on the campaign's own convention
               (distance from the LEXICON mean, not from the
               scale midpoint -- fields.py:292)               10
    external   Warriner valence/arousal/dominance/concreteness
               and their four extremity columns                8

Chinese has the ten coder features and ONE external norm, the Xu & Li two-
character concreteness set, sign-flipped as `k_frame` flips it.

TWO POPULATIONS, BECAUSE WARRINER COVERS 22% OF THE VERBS. It covers 48% of all
rating units, and verbs are worse served than nouns; the 48% is the number to
quote about the vocabulary and the wrong one to quote here. Imputing would leave
eight of eighteen columns mostly constant; dropping would select on frequency,
which is a nuisance correlate of the outcome. So the coder features are fitted on
ALL verbs and the full eighteen on the covered subset, and the two are reported
side by side with their n. Neither is quietly the headline.

THE CASE IS A CELL, THE HELD-OUT UNIT IS A WORD. GroupKFold on the word, so no
word is in both train and test: a model that memorised `murder falls` scores
nothing for it. This is also the only split immune to the word-level
pseudo-replication that has bitten this campaign repeatedly.

TWO OUTCOMES, AND THE BINARY ONE SELECTS ON ITSELF. The binary outcome is fall
(1) vs rise (0) under the canonical rule, which requires dropping the `still`
cells -- and the rule decides fall/rise/still by THRESHOLDING THE VERY QUANTITY
BEING PREDICTED. So the binary analysis is fitted on a population selected by the
outcome variable. That is defensible when the question is literally "which
direction", which is how it was asked, but it is not the safer choice and should
not be read as one. The scalar outcome log10(p_aligned / p_base) is defined for
every cell, needs no such restriction, and keeps magnitude: it is the version
without the selection, and where the two disagree the scalar one is the evidence.

CELLS PER WORD ARE CAPPED by a deterministic hash. The falling is carried by the
words in the most cells, so uncapped the fit is a few hundred frequent words.

THE NUISANCE MODEL IS THE BASELINE, NOT ZERO. [3652] named the tautology channel:
base probability is distance to the eviction boundary, so it predicts movement
under any perturbation including noise. The number that means anything is the
increment OVER the nuisance block, and against the SAME function class -- a tree
with norms beat a logistic without them by +0.155 in the first version, and the
shuffled control showed that was the function class, not the norms.

TWO AUCs. Pooled includes between-site variation, so a model can score by
learning which SITES have many fallers. Per-site is computed within each
(prompt, base, aligned) and averaged: given the words competing at THIS site,
does the model rank which falls? That is the displacement claim, and it is also
the evaluation a conditional logit would be fitted to maximise.

A NEGATIVE CONTROL RUNS EVERY TIME: ratings shuffled across words, whole pipeline
re-run, must land at the nuisance floor.
"""
import collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import k_analysis as A, k_population as KP
import k_frame as KF
from k_frequency import fpm
from malign_logits import fields as FL

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
SCALES = A.SCALES
EXTREMITY = ("valence", "charge", "concreteness")
CAP = 60
FOLDS = 5
SEED = 20260812
#: FOUR, NOT TWELVE. A site offers ~50 candidate words and only a handful are
#: lexical verbs, so the twelve-mover threshold that suited the whole vocabulary
#: left SEVEN usable sites here and ZERO in the shuffled control -- a per-site
#: number computed on seven sites, and a control that could not be computed at
#: all. A four-point AUC is coarse, but it is averaged over thousands of sites
#: and the coarseness is symmetric across models; seven sites was not noise, it
#: was an absent measurement reported as a number.
MIN_SITE = 4
MIN_PROB = 0.003    #: the canonical rule's floor; see --eligible
#: "5" | "20" | "loo"; set by --cv. See `_splitter` for why loo is honoured on
#: the linear models and refused on the trees.
CV = "5"


def is_verb(u, lang):
    if lang == "en":
        e = FL._byu().get(u.strip().lower())
        return bool(e) and e[1].startswith("vv")
    import jieba.posseg as pseg
    seg = list(pseg.cut(u.strip()))
    return len(seg) == 1 and seg[0].flag.startswith("v")


def fetch(lang, marked, movers_only=True):
    """`movers_only=False` keeps the `still` cells, which is REQUIRED for the
    scalar outcome and wrong for the binary one. The first version of the scalar
    analysis reused the mover-only rowset while its docstring claimed the still
    cells were back in the population; the numbers were real and the population
    described was not."""
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    ep = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                     for b, a in KP.reps(lang))
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
                      WHERE status='ACTIVE' AND language='%s'%s) p ON m.prompt=p.prompt
          WHERE m.rule='canonical' AND (%s))
        WHERE (rb<=50 OR ra<=50)%s)
      WHERE rw <= %d""" % (A.DB, A.DB, lang, role, ep,
                           " AND cls IN ('fall','rise')" if movers_only else "", CAP))


def feature_table(lang, rate):
    """unit -> {feature: value}, plus the coder and external column names.

    Coder extremity uses the campaign's convention -- distance from the LEXICON
    mean of that scale over the rated vocabulary, not from the scale midpoint,
    because a 1-7 scale's mass is not centred at 4.
    """
    NM = KF.norms_en() if lang == "en" else KF.norms_zh()
    ext = (["n_" + d for d in KF.NORM_DIMS] if lang == "en" else ["n_concreteness"])
    coder = list(SCALES) + [s + "_extremity" for s in EXTREMITY]
    mu = {s: float(np.mean([rate[u][s] for u in rate])) for s in SCALES}
    T = {}
    for u in rate:
        d = {s: float(rate[u][s]) for s in SCALES}
        for s in EXTREMITY:
            d[s + "_extremity"] = abs(rate[u][s] - mu[s])
        e = NM.get(u.strip().lower() if lang == "en" else u.strip())
        if e:
            for c in ext:
                if c in e:
                    d[c] = float(e[c])
        T[u] = d
    return T, coder, ext


def build(rows, lang, rate, t2u, T, coder, ext, want_ext, shuffle_seed=None,
          eligible=False):
    """-> X, y, groups, sites, colnames. `want_ext` restricts to words the
    external lexicon covers and appends its columns."""
    #: THE SHUFFLE MUST NOT CHANGE THE POPULATION. Permuting whole feature dicts
    #: across the full vocabulary moves the external-lexicon coverage with them,
    #: so the control kept a DIFFERENT set of words: 111 usable sites against the
    #: real run's 1,600, compared as though they were the same experiment. The
    #: eligible set is therefore fixed from the REAL table first, and values are
    #: permuted only among the words already in it.
    elig_units = {u for u in T if not want_ext or all(c in T[u] for c in ext)}
    tab = T
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        ks = sorted(elig_units); vs = [T[k] for k in ks]
        rng.shuffle(vs)
        tab = dict(T); tab.update(dict(zip(ks, vs)))
    meas = "coca_fic" if lang == "en" else "SUBTLEX_CH"
    cols = ["log_p_base", "log_fpm"] + coder + (ext if want_ext else [])
    X, y, g, site = [], [], [], []
    fq = {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in elig_units:
            continue
        if eligible and not (r["p_base"] >= MIN_PROB and r["p_aligned"] >= MIN_PROB):
            continue
        if r["p_base"] <= 0:
            continue
        if u not in fq:
            fq[u] = fpm(u, lang, meas)
        if not fq[u]:
            continue                      #: dropped, never zeroed
        d = tab[u]
        X.append([math.log10(r["p_base"]), math.log10(fq[u])]
                 + [d[c] for c in coder] + ([d[c] for c in ext] if want_ext else []))
        y.append(1 if r["cls"] == "fall" else 0)
        g.append(u); site.append((r["prompt"], r["base"], r["aligned"]))
    return (np.array(X, float), np.array(y), np.array(g, object),
            np.array([hash(s) for s in site]), cols)


def interactions(X, n_nuis):
    N = X[:, n_nuis:]
    C = N - N.mean(0)
    return np.column_stack([X] + [C[:, i] * C[:, j]
                                  for i in range(C.shape[1])
                                  for j in range(i + 1, C.shape[1])])


def per_site_auc(site, y, p):
    from sklearn.metrics import roc_auc_score
    idx = collections.defaultdict(list)
    for i, s in enumerate(site):
        idx[s].append(i)
    out = [roc_auc_score(y[ii], p[ii]) for ii in idx.values()
           if len(ii) >= MIN_SITE and y[ii].min() != y[ii].max()]
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def _fit_predict(kind, Mtr, ytr, Mte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        sc = StandardScaler().fit(Mtr)
        m = LogisticRegression(max_iter=3000).fit(sc.transform(Mtr), ytr)
        return m.predict_proba(sc.transform(Mte))[:, 1]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.08,
                                       random_state=SEED).fit(Mtr, ytr)
    return m.predict_proba(Mte)[:, 1]


def _splitter(kind, n_groups):
    """LEAVE-ONE-WORD-OUT WHERE IT IS AFFORDABLE, AND SAID PLAINLY WHERE IT IS NOT.

    LOO on the linear models is n_groups refits of a twelve-feature logistic,
    which is minutes. LOO on the boosted trees is n_groups refits of a
    300-iteration ensemble -- roughly eleven hours per model per population -- so
    the tree rows fall back to 20-fold and the fallback is PRINTED rather than
    silently substituted. 20-fold trains on 95% of words against LOO's 100%, and
    the learning curve below is what says whether that last 5% could matter.
    """
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    if CV == "loo":
        if kind == "lr":
            return LeaveOneGroupOut(), "LOO"
        return GroupKFold(n_splits=min(20, n_groups)), "20-fold (LOO infeasible)"
    return GroupKFold(n_splits=int(CV)), "%s-fold" % CV


def evaluate(X, y, g, site, n_nuis, tag):
    from sklearn.metrics import roc_auc_score
    specs = {"nuisance":       ("lr", X[:, :n_nuis]),
             "norms only":     ("lr", X[:, n_nuis:]),
             "additive":       ("lr", X),
             "interactions":   ("lr", interactions(X, n_nuis)),
             "trees nuisance": ("gb", X[:, :n_nuis]),
             "trees":          ("gb", X)}
    out = {}
    ng = len(set(g))
    for name, (kind, M) in specs.items():
        sp, how = _splitter(kind, ng)
        pred = np.zeros(len(y))
        for tr, te in sp.split(M, y, groups=g):
            pred[te] = _fit_predict(kind, M[tr], y[tr], M[te])
        ps, ns = per_site_auc(site, y, pred)
        out[name] = (float(roc_auc_score(y, pred)), ps, ns)
        print("      %-15s pooled %.4f   per-site %.4f  (%d sites)  [%s]"
              % (name, out[name][0], ps, ns, how))
    return out


def learning_curve(X, y, g, n_nuis, fracs=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8)):
    """Held-out AUC against how many TRAINING WORDS the model saw.

    This is the question LOO is really asking. LOO differs from 5-fold only in
    training on 100% of the remaining words rather than 80%, so if the curve has
    already flattened, the extra 20% cannot change the answer and the eleven
    hours of tree refits buy nothing. If instead the curve is still climbing at
    80%, the null IS a sample-size result and should be reported as one.

    The test set is a fixed 20% of words, identical at every fraction, so the
    points are comparable to each other -- resampling the test set per point
    would put the curve's own noise on both axes.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    words = np.array(sorted(set(g)))
    rng.shuffle(words)
    nte = max(1, len(words) // 5)
    te_w = set(words[:nte]); tr_pool = words[nte:]
    te = np.array([w in te_w for w in g])
    print("      learning curve: %d test words held fixed, %d in the training pool"
          % (len(te_w), len(tr_pool)))
    print("        %-10s %8s %10s %10s %10s"
          % ("train n", "words", "nuisance", "additive", "trees"))
    curve = []
    for f in fracs:
        k = max(20, int(len(tr_pool) * f))
        sel = set(tr_pool[:k])
        tr = np.array([w in sel for w in g])
        row = {"frac": f, "train_words": k}
        for name, kind, M in (("nuisance", "lr", X[:, :n_nuis]),
                              ("additive", "lr", X),
                              ("trees", "gb", X)):
            p = _fit_predict(kind, M[tr], y[tr], M[te])
            row[name] = float(roc_auc_score(y[te], p))
        print("        %-10.0f%% %8d %10.4f %10.4f %10.4f"
              % (100 * f, k, row["nuisance"], row["additive"], row["trees"]))
        curve.append(row)
    return curve


def within_site_coefs(X, y, site, cols, n_sites=2500):
    """Conditional logit stratified by site: only WITHIN-site contrasts are used,
    so every site-level and model-pair-level nuisance drops out of the likelihood
    instead of being modelled. This is the estimator that matches the per-site
    AUC, and the one the displacement claim implies -- the words at a site compete
    for the same mass.

    Fitted on a random subsample of sites because the full stratification is
    60k+ strata. Coefficients only; no SEs are printed, since the cells remain
    clustered by WORD inside each stratum and an unclustered SE here would be
    wrong in the flattering direction.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit
    rng = np.random.default_rng(SEED)
    idx = collections.defaultdict(list)
    for i, s in enumerate(site):
        idx[s].append(i)
    usable = [s for s, ii in idx.items()
              if len(ii) >= 4 and y[ii].min() != y[ii].max()]
    if not usable:
        return None
    pick = rng.permutation(usable)[:n_sites]
    keep = np.concatenate([idx[s] for s in pick])
    Xs = X[keep]; Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    try:
        r = ConditionalLogit(y[keep], Xs, groups=site[keep]).fit(disp=0)
    except Exception as e:
        print("      conditional logit failed: %s" % str(e)[:90])
        return None
    print("      within-site (conditional logit, %d strata, %s cells), "
          "positive = predicts FALL" % (len(pick), f"{len(keep):,}"))
    for n, c in sorted(zip(cols, r.params), key=lambda t: -abs(t[1])):
        print("        %-24s %+.4f" % (n, c))
    return dict(zip(cols, [float(c) for c in r.params]))


def scalar(rows, lang, rate, t2u, T, coder, ext, want_ext, label, shuffle_seed=None):
    """The continuous outcome: HOW FAR alignment moves the word, not which side
    of the rule it lands on.

    y = log10(p_aligned / p_base), the quantity the canonical rule thresholds --
    a fall is y < log10(0.5). Two things change against the binary version.
    Magnitude is kept instead of discarded, and the cells the rule calls `still`
    are in the population, so the model can be asked whether the features predict
    a word STAYING PUT. Restricted to p_base >= MIN_PROB, below which the ratio
    is a ratio of two numbers the instrument does not resolve.

    THIS MATTERS MORE THAN IT LOOKS. Binarising by the rule makes p_base appear
    to predict: the rule's fall condition is p_base-relative, so the label is
    partly a restatement of the predictor. Measured here, within a site, p_base
    predicts magnitude at Spearman +0.001 -- nothing at all -- against a per-site
    AUC of 0.66 in the binary version. The nuisance variable's apparent power in
    the binary analysis is manufactured by the binarisation.

    `shuffle_seed` permutes the feature values across words WITHIN the eligible
    set, leaving every probability, frequency and site exactly where it was. The
    binary path has always run this control; the first version of this function
    did not, and a within-site rho of +0.037 against an unmeasured null is not a
    result.
    """
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    EPS = 1e-6
    meas = "coca_fic" if lang == "en" else "SUBTLEX_CH"
    elig_units = {u for u in T if not want_ext or all(c in T[u] for c in ext)}
    tab = T
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        ks = sorted(elig_units); vs = [T[k] for k in ks]
        rng.shuffle(vs)
        tab = dict(T); tab.update(dict(zip(ks, vs)))
    cols = ["log_p_base", "log_fpm"] + coder + (ext if want_ext else [])
    X, y, g, site, fq = [], [], [], [], {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in elig_units or r["p_base"] < MIN_PROB:
            continue
        if u not in fq:
            fq[u] = fpm(u, lang, meas)
        if not fq[u]:
            continue
        d = tab[u]
        X.append([math.log10(r["p_base"]), math.log10(fq[u])]
                 + [d[c] for c in coder] + ([d[c] for c in ext] if want_ext else []))
        y.append(math.log10((r["p_aligned"] + EPS) / (r["p_base"] + EPS)))
        g.append(u); site.append((r["prompt"], r["base"], r["aligned"]))
    X = np.array(X, float); y = np.array(y); g = np.array(g, object)
    site = np.array([hash(s) for s in site])
    if len(y) < 500 or len(set(g)) < 30:
        print("    --- SCALAR %s: too thin (%d cells, %d words)" % (label, len(y), len(set(g))))
        return None
    print("    --- SCALAR %s | %s cells | %d words | %d sites | mean %+.3f sd %.3f"
          % (label, f"{len(y):,}", len(set(g)), len(set(site)), y.mean(), y.std()))
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=FOLDS)
    specs = {"nuisance":       ("ridge", X[:, :2]),
             "additive":       ("ridge", X),
             "interactions":   ("ridge", interactions(X, 2)),
             "trees nuisance": ("gb",    X[:, :2]),
             "trees":          ("gb",    X)}
    res = {}
    for name, (kind, M) in specs.items():
        pred = np.zeros(len(y))
        for tr, te in gkf.split(M, y, groups=g):
            if kind == "ridge":
                sc = StandardScaler().fit(M[tr])
                m = Ridge(alpha=1.0).fit(sc.transform(M[tr]), y[tr])
                pred[te] = m.predict(sc.transform(M[te]))
            else:
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.08,
                                                  random_state=SEED).fit(M[tr], y[tr])
                pred[te] = m.predict(M[te])
        idx = collections.defaultdict(list)
        for i, sv in enumerate(site):
            idx[sv].append(i)
        rh = [spearmanr(y[ii], pred[ii]).statistic for ii in idx.values()
              if len(ii) >= MIN_SITE]
        rh = [v for v in rh if v == v]
        res[name] = (float(r2_score(y, pred)), float(np.mean(rh)) if rh else float("nan"),
                     len(rh))
        print("      %-15s held-out R2 %+.4f   per-site Spearman %+.4f  (%d sites)"
              % (name, res[name][0], res[name][1], res[name][2]))
    print("      %-15s features add  R2 %+.4f / rho %+.4f (linear)   %+.4f / %+.4f (trees)"
          % ("", res["additive"][0] - res["nuisance"][0],
             res["additive"][1] - res["nuisance"][1],
             res["trees"][0] - res["trees nuisance"][0],
             res["trees"][1] - res["trees nuisance"][1]))
    return {k: list(v) for k, v in res.items()}


def main():
    global CV
    lang = "zh" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "zh" else "en"
    marked = "--marked" in sys.argv
    elig = "--eligible" in sys.argv
    if "--cv" in sys.argv:
        CV = sys.argv[sys.argv.index("--cv") + 1]
        if CV not in ("5", "20", "loo"):
            print("--cv takes 5, 20 or loo"); return 1
    rate = json.load(open(os.path.join(K, "ratings_%s.json" % lang)))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]
    verbs = {u for u in rate if is_verb(u, lang)}
    rate = {u: v for u, v in rate.items() if u in verbs}
    print("\n[%s]%s%s  LEXICAL VERBS ONLY: %d of the rated vocabulary"
          % (lang, " MARKED" if marked else "", " ELIGIBLE" if elig else "", len(rate)))
    T, coder, ext = feature_table(lang, rate)
    ncov = sum(1 for u in T if all(c in T[u] for c in ext))
    print("  features: 2 nuisance + %d coder + %d external (%s); external covers "
          "%d of %d verbs (%.0f%%)"
          % (len(coder), len(ext), ", ".join(ext[:3]) + ("..." if len(ext) > 3 else ""),
             ncov, len(T), 100 * ncov / max(len(T), 1)))
    rows = fetch(lang, marked)
    all_rows = [None]      #: lazily fetched, and only when --scalar is passed
    print("  %s mover cells fetched (cap %d per word)" % (f"{len(rows):,}", CAP))

    res = {}
    for want_ext, label in ((False, "CODER ONLY, all verbs"),
                            (True, "CODER + EXTERNAL, covered verbs")):
        X, y, g, site, cols = build(rows, lang, rate, t2u, T, coder, ext,
                                    want_ext, eligible=elig)
        if len(y) < 500 or len(set(g)) < 30:
            print("\n  --- %s: too thin (%d cells, %d words), skipped"
                  % (label, len(y), len(set(g)))); continue
        print("\n  --- %s | %s cells | %d words | %d sites | %d features | fall %.3f"
              % (label, f"{len(y):,}", len(set(g)), len(set(site)), X.shape[1], y.mean()))
        print("    real ratings")
        real = evaluate(X, y, g, site, 2, label)
        Xs, ys, gs, ss, _ = build(rows, lang, rate, t2u, T, coder, ext, want_ext,
                                  shuffle_seed=SEED, eligible=elig)
        print("    ratings shuffled across words")
        shuf = evaluate(Xs, ys, gs, ss, 2, label)
        print("    WHAT THE FEATURES ADD (pooled / per-site)")
        BASE = {"norms only": "nuisance", "additive": "nuisance",
                "interactions": "nuisance", "trees": "trees nuisance"}
        for k in ("norms only", "additive", "interactions", "trees"):
            b = BASE[k]
            print("      %-15s over same class %+.4f / %+.4f   over shuffled %+.4f / %+.4f"
                  % (k, real[k][0] - real[b][0], real[k][1] - real[b][1],
                     real[k][0] - shuf[k][0], real[k][1] - shuf[k][1]))
        cf = within_site_coefs(X, y, site, cols)
        lc = learning_curve(X, y, g, 2) if "--curve" in sys.argv else None
        sc_res = None
        if "--scalar" in sys.argv:
            if all_rows[0] is None:
                all_rows[0] = fetch(lang, marked, movers_only=False)
                print("    scalar population: %s cells including `still` "
                      "(binary used %s movers)" % (f"{len(all_rows[0]):,}", f"{len(rows):,}"))
            sc_res = {"real": scalar(all_rows[0], lang, rate, t2u, T, coder, ext,
                                     want_ext, label),
                      "shuffled": scalar(all_rows[0], lang, rate, t2u, T, coder, ext,
                                         want_ext, label + " [SHUFFLED]",
                                         shuffle_seed=SEED)}
        res[label] = {"n_cells": int(len(y)), "n_words": len(set(g)),
                      "n_sites": len(set(site)), "fall_rate": float(y.mean()),
                      "cols": cols, "real": {k: list(v) for k, v in real.items()},
                      "shuffled": {k: list(v) for k, v in shuf.items()},
                      "within_site_coefs": cf, "scalar": sc_res,
                      "learning_curve": lc, "cv": CV}
    out = os.path.join(K, "predict_verbs_%s%s%s%s.json"
                       % (lang, "_marked" if marked else "", "_eligible" if elig else "",
                          "_scalar" if "--scalar" in sys.argv else ""))
    json.dump({"_lang": lang, "_cap": CAP, "_folds": FOLDS, "_verbs_only": True,
               "_marked": marked, "_eligible": elig, "results": res},
              open(out, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
