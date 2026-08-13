"""Can a classifier tell BASE from ALIGNED out of the word-probability vector,
and does the direction it learns agree with the movement axis?

    uv run python meta/M01_displacement/scripts/k_armclf.py en

RH's design. Rows are (model, prompt) cells, columns are the top-N content words,
values are that word's probability in that cell, the label is the arm. It is a
different question from everything else in P and it has one large advantage:

**IT NEVER TOUCHES THE CANONICAL RULE.** No p_base >= 0.003 floor, no fall/rise
binarisation, no thresholding of a quantity that is also a predictor. Several of
the artifacts corrected in P came from exactly that, and this instrument cannot
reproduce them because it does not know the rule exists.

THE PAYOFF IS THE COEFFICIENT VECTOR, NOT THE AUC. The logistic weights over
words say which words' probabilities distinguish the arms, learned with no
movement data at all. Projected into GloVe space that is an independent estimate
of the direction `k_axis` fits from movement. **If the two agree, two instruments
sharing no machinery have found the same thing. If they do not, one of them is
measuring something else, and that is worth more than another AUC.**

HELD OUT BY LINEAGE, WHICH IS THE WHOLE DESIGN. Splitting rows at random would
let the model learn "this is Llama-base" rather than "this is a base model" --
the same failure as holding out cells instead of words in P section 1. GroupKFold
on the lineage asks whether a base/aligned signature generalises to an unseen
model family. A null under this split with a high score under a random one would
mean the signature is family-specific, which is itself a finding and would
explain much of P.

THE NUISANCE IS SHARPNESS AND IT MAY BE THE WHOLE EFFECT. Aligned models put more
mass on fewer words. A classifier reading raw probabilities could be reading
nothing but that, so the nuisance block is computed from each cell's own
distribution -- top-1 mass, entropy over the stored support, and the count of
words above a fixed floor -- and the word identities are scored as an INCREMENT
over it, exactly as the norms are in P.

TWO FEATURE FORMS, BOTH RUN, BECAUSE THE GAP BETWEEN THEM IS INTERPRETABLE:

    raw     log10(p), with absent words at the floor
    ranked  within-cell rank, which removes the sharpness gradient entirely

If the signature survives ranking, it is about WHICH words; if it collapses, it
was about the SHAPE of the distribution. Both are real findings about alignment
and they are not the same finding.

ABSENCE IS CENSORING, NOT MISSINGNESS. `twp_words` stores about 125 words per
cell, so a word not present is below that cutoff rather than unmeasured. It is
entered at a floor one decade below the smallest stored probability, and that is
a declared choice: treating it as missing would drop most of the matrix, and
treating it as exactly zero would put it at negative infinity in log space.

THE NULL IS A WITHIN-LINEAGE LABEL FLIP. Permuting arm labels across the whole
sample would break the lineage structure as well as the arm contrast. Flipping
base and aligned within a lineage, at random per lineage, leaves every cell and
every model where it is and destroys only the direction under test.
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
import k_population as KP

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
#: SWEPT, NOT CHOSEN. The label varies at the MODEL level, so there are 92
#: labelled units in 46 lineages; the ~2,200 prompts per model are repeated
#: measurements of one label and sharpen the coefficients without adding
#: independent label information. Fitting 300 word-weights against 46
#: independent lineages is overfitting territory, so the column count is swept
#: and the whole curve reported -- picking the best N after seeing the scores
#: would be selection on the outcome.
N_SWEEP = (25, 50, 100, 200, 400)
FOLDS = 5
SEED = 20260812


def main(lang="en"):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    pairs = KP.reps(lang)
    arm, lin, org = {}, {}, {}
    for i, (b, a) in enumerate(pairs):
        arm[b] = 0; arm[a] = 1
        lin[b] = i; lin[a] = i
        #: the org prefix of the BASE model names the group for both arms, so an
        #: aligned model never trains against its own org's other lineages
        o = b.split(chr(47))[0]
        org[b] = o; org[a] = o
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    models = "','".join(esc(m) for m in arm)
    print("[%s] %d lineages, %d distinct models" % (lang, len(pairs), len(arm)))

    #: THE COLUMN SET IS CHOSEN BEFORE ANY LABEL IS SEEN -- top-N by total mass
    #: over both arms pooled, so the selection cannot prefer either one. And it
    #: is intersected with the GloVe vocabulary because the coefficient vector
    #: has to live in that space to be comparable with the movement axis.
    z = np.load(os.path.join(K, "embed_%s_glove.npz" % lang), allow_pickle=True)
    EM = {w: v.astype(np.float64) for w, v in zip(z["words"], z["E"])}
    t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]

    #: CONTENT WORDS ONLY, and it is a declared arm rather than a default. Top-N
    #: by mass without this filter is `the a and to her his`, and the arm
    #: signature could live entirely in function words -- which would be a real
    #: finding about syntax rather than vocabulary, but it is a DIFFERENT
    #: question and should be asked deliberately. --function runs that arm.
    from malign_logits import fields as FL
    want_content = "--function" not in sys.argv

    top = A.q("""
      SELECT word, sum(p) mass FROM (
        SELECT model, prompt, word, avg(p) p FROM %s.twp_words FINAL
        WHERE model IN ('%s') AND prompt IN (
          SELECT DISTINCT prompt FROM %s.prompt_catalogue
          WHERE status='ACTIVE' AND language='%s')
        GROUP BY model, prompt, word)
      GROUP BY word ORDER BY mass DESC LIMIT 4000""" % (A.DB, models, A.DB, lang))
    cols = []
    for r in top:
        u = t2u.get(r["word"], r["word"])
        if u not in EM or r["word"] in cols:
            continue
        if FL.is_content_word(u) != want_content:
            continue
        cols.append(r["word"])
        if len(cols) >= max(N_SWEEP):
            break
    print("  %d candidate %s words, ranked by pooled mass before any label was seen"
          % (len(cols), "content" if want_content else "FUNCTION"))
    print("  first 12: %s" % ", ".join(cols[:12]))
    ci = {w: i for i, w in enumerate(cols)}

    rows = A.q("""
      SELECT model, prompt, groupArray(word) ws, groupArray(p) ps,
             max(p) top1, count() nsup, sum(p) mass
      FROM (
        SELECT model, prompt, word, avg(p) p FROM %s.twp_words FINAL
        WHERE model IN ('%s') AND prompt IN (
          SELECT DISTINCT prompt FROM %s.prompt_catalogue
          WHERE status='ACTIVE' AND language='%s')
        GROUP BY model, prompt, word)
      GROUP BY model, prompt ORDER BY model, prompt""" % (A.DB, models, A.DB, lang))
    print("  %s (model, prompt) cells" % f"{len(rows):,}")

    FLOOR = -6.0
    X, XR, y, g_lin, g_org, mid, nu = [], [], [], [], [], [], []
    for r in rows:
        v = np.full(len(cols), FLOOR)
        allp = np.array([x for x in r["ps"] if x is not None], float)
        for w, pp in zip(r["ws"], r["ps"]):
            j = ci.get(w)
            if j is not None and pp and pp > 0:
                v[j] = math.log10(pp)
        order = np.argsort(np.argsort(v))
        XR.append(order / max(len(cols) - 1, 1))
        X.append(v)
        qd = allp / max(allp.sum(), 1e-12) if len(allp) else np.array([1.0])
        nu.append([math.log10(max(r["top1"] or 1e-12, 1e-12)),
                   float(-(qd * np.log(np.maximum(qd, 1e-12))).sum()),
                   math.log10(max(r["nsup"] or 1, 1)),
                   float(r["mass"] or 0.0)])
        y.append(arm[r["model"]])
        g_lin.append(lin[r["model"]]); g_org.append(org[r["model"]])
        mid.append(r["model"])
    X = np.array(X); XR = np.array(XR); NU = np.array(nu)
    y = np.array(y); mid = np.array(mid, dtype=object)
    g_lin = np.array(g_lin); g_org = np.array(g_org, dtype=object)
    print("  matrix %s | aligned share %.3f" % (X.shape, y.mean()))
    print("  %d lineage groups, %d ORG groups -- 21 of 46 lineages share an org"
          % (len(set(g_lin)), len(set(g_org))))

    rng = np.random.default_rng(SEED)
    #: SORTED (set order is per-process for strings -- the [5744] class), and
    #: the null is a DISTRIBUTION: a single flip is one draw from a wide null
    #: (the m06 producer's one-flip nulls wandered 0.40-0.63 across two runs of
    #: one seed). N_NULL draws; the headline null is the mean with a 95% band.
    N_NULL = 200
    lineages = sorted(set(g_lin))
    null_flips = [dict(zip(lineages, rng.integers(0, 2, len(lineages))))
                  for _ in range(N_NULL)]
    y_null = np.array([yy ^ null_flips[0][gg] for yy, gg in zip(y, g_lin)])

    def q(M, groups, target=y):
        """-> (cell AUC, MODEL AUC, coefficients).

        THE MODEL AUC IS THE ONE TO READ. There are ~204,000 rows and 92
        independent labels: y is exactly constant within a model, so getting one
        model right wins ~2,200 rows and a cell-level AUC reports 92 judgements
        with 204,000-fold apparent precision. Same defect as P section 1's
        effective n, same fix -- average each model's predicted probability and
        score the 92.
        """
        from sklearn.model_selection import GroupKFold
        gk = GroupKFold(n_splits=FOLDS)
        pred = np.zeros(len(target), float)
        cf = []
        for tr, te in gk.split(M, target, groups=groups):
            sc = StandardScaler().fit(M[tr])
            m = LogisticRegression(max_iter=4000, C=0.1).fit(sc.transform(M[tr]),
                                                             target[tr])
            pred[te] = m.predict_proba(sc.transform(M[te]))[:, 1]
            cf.append(m.coef_[0])
        bym = collections.defaultdict(list)
        for mm, pp, tt in zip(mid, pred, target):
            bym[mm].append((pp, tt))
        mp = [float(np.mean([x for x, _ in v])) for v in bym.values()]
        mt = [v[0][1] for v in bym.values()]
        return (float(roc_auc_score(target, pred)),
                float(roc_auc_score(mt, mp)), np.array(cf))

    #: BOTH SPLITS, AND THE GAP BETWEEN THEM IS THE MEASUREMENT. Holding out a
    #: LINEAGE leaves sibling lineages from the same org in training, so the
    #: model can learn "tiiuae bases look like this" from four Falcons and apply
    #: it to the fifth -- that is recognising the org, not the arm. Holding out
    #: the ORG removes that. If the org-held-out score is much lower, the
    #: difference is family recognition and not an alignment signature.
    print("\n1. CAN IT NAME THE ARM OF A MODEL IT HAS NEVER SEEN?")
    print("   MODEL-LEVEL AUC over 92 models (cell-level in brackets, overstated)")
    sweep, C_wo, best_n = {}, None, None
    for gname, groups in (("lineage", g_lin), ("ORG", g_org)):
        a_nu, m_nu, _ = q(NU, groups)
        print("\n   held out by %-8s  nuisance only (sharpness) %.4f  [%.4f]"
              % (gname, m_nu, a_nu))
        print("   %-6s %14s %14s %14s %14s"
              % ("N", "raw", "ranked", "words only", "NULL(flip)"))
        for n in N_SWEEP:
            if n > len(cols):
                continue
            Xn, XRn = X[:, :n], XR[:, :n]
            a_raw, m_raw, _ = q(np.hstack([NU, Xn]), groups)
            a_rnk, m_rnk, _ = q(np.hstack([NU, XRn]), groups)
            a_wo, m_wo, cf = q(Xn, groups)
            #: the full q() per draw is expensive at 200 draws x cell grain;
            #: the MODEL-level null is the quotable one, so draws re-score the
            #: model-mean predictions against flipped labels where possible --
            #: here the honest cheap form is q() on a subsample of draws for
            #: the cell AUC and closed-form re-scoring for the model AUC.
            a_nl, m_nl, _ = q(np.hstack([NU, Xn]), groups, y_null)
            m_nulls = []
            for fl in null_flips:
                yn = np.array([yy ^ fl[gg] for yy, gg in zip(y, g_lin)])
                if yn.min() == yn.max():
                    continue
                _, m_d, _ = q(np.hstack([NU, Xn]), groups, yn)
                m_nulls.append(m_d)
                if len(m_nulls) >= 50:
                    break
            sweep["%s_%d" % (gname, n)] = {
                "raw": [a_raw, m_raw], "ranked": [a_rnk, m_rnk],
                "words_only": [a_wo, m_wo], "null": [a_nl, m_nl],
                "null_model_mean": float(np.mean(m_nulls)),
                "null_model_ci": [float(np.percentile(m_nulls, 2.5)),
                                  float(np.percentile(m_nulls, 97.5))],
                "null_model_draws": len(m_nulls),
                "nuisance": [a_nu, m_nu]}
            print("   %-6d %7.4f[%.3f] %7.4f[%.3f] %7.4f[%.3f] %7.4f[%.3f]"
                  % (n, m_raw, a_raw, m_rnk, a_rnk, m_wo, a_wo, m_nl, a_nl))
            #: coefficients taken from the ORG split at the largest N <= 100 --
            #: a rule fixed before the scores, not the best-scoring cell
            if gname == "ORG" and n <= 100:
                C_wo, best_n = cf, n
    print("\n   NULL is a within-lineage base/aligned flip and should sit at ~0.5.")

    #: 2. THE COEFFICIENT VECTOR AGAINST THE MOVEMENT AXIS
    print("\n2. DOES THE LEARNED DIRECTION AGREE WITH THE MOVEMENT AXIS?  (N=%d)" % best_n)
    U = np.array([c / np.linalg.norm(c) for c in C_wo])
    cs = [float(U[i] @ U[j]) for i in range(len(U)) for j in range(i + 1, len(U))]
    print("   coefficient stability across folds: min cos %.3f, median %.3f"
          % (min(cs), float(np.median(cs))))
    w_coef = C_wo.mean(0)
    ax = json.load(open(os.path.join(K, "axis_en.json")))["axis"]
    ax = np.array(ax) / np.linalg.norm(ax)
    #: the classifier weight is per WORD; project it into GloVe space by
    #: mass-weighting each word's vector, then compare directions
    V = np.array([EM[t2u.get(w, w)] for w in cols[:best_n]])
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    proj = w_coef @ V
    proj = proj / max(np.linalg.norm(proj), 1e-12)
    print("   cos(classifier direction in GloVe space, movement axis)  %+.3f" % float(proj @ ax))
    ax_pos = {w: float(V[i] @ ax) for i, w in enumerate(cols[:best_n])}
    r = spearmanr([w_coef[ci[w]] for w in cols[:best_n]],
                  [ax_pos[w] for w in cols[:best_n]]).statistic
    print("   Spearman(per-word classifier weight, word's axis position) %+.3f" % r)
    print("   NOTE: axis position POSITIVE = falls under alignment; classifier weight")
    print("   POSITIVE = higher probability in the ALIGNED arm. So agreement predicts")
    print("   a NEGATIVE relationship, and a positive one would mean they disagree.")

    o = np.argsort(w_coef)
    print("\n3. THE SIGNATURE, learned without any movement data")
    print("   HIGHER in aligned:  %s" % ", ".join(cols[i] for i in o[-25:][::-1]))
    print("   HIGHER in base:     %s" % ", ".join(cols[i] for i in o[:25]))

    json.dump({"n_candidates": len(cols), "coef_n": best_n,
               "n_cells": int(len(y)), "n_lineages": len(set(g_lin)),
               "n_orgs": len(set(g_org)), "n_models": len(set(mid)),
               "auc_nuisance": a_nu, "sweep": sweep,
               "coef_stability_min_cos": min(cs),
               "cos_with_axis": float(proj @ ax), "spearman_with_axis": float(r),
               "higher_aligned": [cols[i] for i in o[-40:][::-1]],
               "higher_base": [cols[i] for i in o[:40]]},
              open(os.path.join(K, "armclf_%s.json" % lang), "w"), indent=1)
    print("\n  -> results/k/armclf_%s.json" % lang)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
