"""Plan K prediction: given only a word's rated properties, can we predict which
way alignment will move it -- for a word never seen in training?

    uv run python meta/M01_displacement/scripts/k_predict.py --lang en
    uv run python meta/M01_displacement/scripts/k_predict.py --lang zh
    ... --marked      transgressive sites only
    ... --eligible    both arms above the rule's probability floor
    ... --scalar      also run the continuous outcome

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

TWO POPULATIONS, BECAUSE WARRINER COVERS 48% OF THE VERBS. Imputing would leave
eight of eighteen columns half-constant; dropping would select on frequency,
which is a nuisance correlate of the outcome. So the coder features are fitted on
ALL verbs and the full eighteen on the covered subset, and the two are reported
side by side with their n. Neither is quietly the headline.

THE CASE IS A CELL, THE HELD-OUT UNIT IS A WORD. Outcome is fall (1) vs rise (0)
under the canonical rule. GroupKFold on the word, so no word is in both train and
test: a model that memorised `murder falls` scores nothing for it. This is also
the only split immune to the word-level pseudo-replication that has bitten this
campaign repeatedly.

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


def is_verb(u, lang):
    if lang == "en":
        e = FL._byu().get(u.strip().lower())
        return bool(e) and e[1].startswith("vv")
    import jieba.posseg as pseg
    seg = list(pseg.cut(u.strip()))
    return len(seg) == 1 and seg[0].flag.startswith("v")


def fetch(lang, marked):
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
        WHERE (rb<=50 OR ra<=50) AND cls IN ('fall','rise'))
      WHERE rw <= %d""" % (A.DB, A.DB, lang, role, ep, CAP))


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


def evaluate(X, y, g, site, n_nuis, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    specs = {"nuisance":       ("lr", X[:, :n_nuis]),
             "norms only":     ("lr", X[:, n_nuis:]),
             "additive":       ("lr", X),
             "interactions":   ("lr", interactions(X, n_nuis)),
             "trees nuisance": ("gb", X[:, :n_nuis]),
             "trees":          ("gb", X)}
    out = {}
    gkf = GroupKFold(n_splits=FOLDS)
    for name, (kind, M) in specs.items():
        pred = np.zeros(len(y))
        for tr, te in gkf.split(M, y, groups=g):
            if kind == "lr":
                sc = StandardScaler().fit(M[tr])
                m = LogisticRegression(max_iter=3000).fit(sc.transform(M[tr]), y[tr])
                pred[te] = m.predict_proba(sc.transform(M[te]))[:, 1]
            else:
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.08,
                                                   random_state=SEED).fit(M[tr], y[tr])
                pred[te] = m.predict_proba(M[te])[:, 1]
        ps, ns = per_site_auc(site, y, pred)
        out[name] = (float(roc_auc_score(y, pred)), ps, ns)
        print("      %-15s pooled %.4f   per-site %.4f  (%d sites)"
              % (name, out[name][0], ps, ns))
    return out


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


def main():
    lang = "zh" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "zh" else "en"
    marked = "--marked" in sys.argv
    elig = "--eligible" in sys.argv
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
        res[label] = {"n_cells": int(len(y)), "n_words": len(set(g)),
                      "n_sites": len(set(site)), "fall_rate": float(y.mean()),
                      "cols": cols, "real": {k: list(v) for k, v in real.items()},
                      "shuffled": {k: list(v) for k, v in shuf.items()},
                      "within_site_coefs": cf}
    out = os.path.join(K, "predict_verbs_%s%s%s.json"
                       % (lang, "_marked" if marked else "", "_eligible" if elig else ""))
    json.dump({"_lang": lang, "_cap": CAP, "_folds": FOLDS, "_verbs_only": True,
               "_marked": marked, "_eligible": elig, "results": res},
              open(out, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
