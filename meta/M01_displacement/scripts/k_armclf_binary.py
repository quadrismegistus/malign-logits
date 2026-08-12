"""Arm classification with the entropy confound removed by construction.

    uv run python meta/M01_displacement/scripts/k_armclf_binary.py en

RH's design. The probability version (`k_armclf`) needs a nuisance block because
aligned models are sharper: top-1 mass alone identifies the arm at model-level
AUC 0.889, and the stored-support size at 0.828. Controlling for that after the
fact is second best. These two variants remove it before the fact.

**A. BINARY, RANK-BASED WITHIN THE CELL.** Each (model, prompt, word) is 1 if the
word is among the cell's top-N by probability and 0 otherwise. Every cell then
contributes EXACTLY N ones, so total mass, top-1 mass and support size are
constant across cells by construction and cannot carry any signal. What is left
is purely WHICH words are in the running.

THE RANK MATTERS AND "STORED OR NOT" WOULD NOT DO. `twp_words` keeps a
threshold-determined number of words per cell -- about 125 on average but varying
-- and that count is one of the strongest arm predictors on its own. So a binary
built from "is the word present in the table" would smuggle sharpness back in
through the denominator. Top-N BY RANK is the version that is actually free of it.

**B. COLLAPSED TO PER-MODEL COUNTS.** Sum A over prompts: one row per model, each
value the FRACTION of prompts where that word was in the model's top-N. This
makes the unit honest rather than correcting for it afterwards -- the design
matrix becomes 92 independent observations, which is what the label always had.
Everything in `k_armclf` that needed a model-level AUC computed post hoc is
simply the natural unit here.

The cost is that 92 rows against N columns overfits quickly, so N stays small and
the penalty stays heavy. That suits the question: the coefficients are directly
readable as "aligned models keep this word in their top-N at X% of prompts,
base models at Y%".

HELD OUT BY ORG IN BOTH. 33 groups; 21 of 46 lineages share an org with another,
so holding out a lineage leaves siblings in training and the model can recognise
the org rather than the arm.

THE NULL IS A WITHIN-LINEAGE ARM FLIP, and under the org split it lands at ~0.51
in the probability version, so it is a working null rather than a decorative one.
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
CAND = 200
TOPN = (5, 20, 50)          #: the rank cutoff defining "in the running"
KS = (5, 10, 25, 50, 100, 200)
FOLDS = 5
SEED = 20260812


def main(lang="en"):
    from malign_logits import fields as FL
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    pairs = KP.reps(lang)
    arm, org, lin = {}, {}, {}
    for i, (b, a) in enumerate(pairs):
        arm[b] = 0; arm[a] = 1
        o = b.split("/")[0]; org[b] = o; org[a] = o
        lin[b] = i; lin[a] = i
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    models = "','".join(esc(m) for m in arm)

    z = np.load(os.path.join(K, "embed_%s_glove.npz" % lang), allow_pickle=True)
    EM = set(z["words"].tolist())
    t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]
    top = A.q("""
      SELECT word, sum(p) mass FROM (
        SELECT model, prompt, word, avg(p) p FROM %s.twp_words FINAL
        WHERE model IN ('%s') AND prompt IN (
          SELECT DISTINCT prompt FROM %s.prompt_catalogue
          WHERE status='ACTIVE' AND language='%s')
        GROUP BY model, prompt, word)
      GROUP BY word ORDER BY mass DESC LIMIT 6000""" % (A.DB, models, A.DB, lang))
    cols = []
    for r in top:
        u = t2u.get(r["word"], r["word"])
        if u in EM and r["word"] not in cols and FL.is_content_word(u):
            cols.append(r["word"])
        if len(cols) >= CAND:
            break
    ci = {w: i for i, w in enumerate(cols)}

    rows = A.q("""
      SELECT model, prompt, groupArray(word) ws, groupArray(p) ps
      FROM (
        SELECT model, prompt, word, avg(p) p FROM %s.twp_words FINAL
        WHERE model IN ('%s') AND prompt IN (
          SELECT DISTINCT prompt FROM %s.prompt_catalogue
          WHERE status='ACTIVE' AND language='%s')
        GROUP BY model, prompt, word)
      GROUP BY model, prompt ORDER BY model, prompt""" % (A.DB, models, A.DB, lang))
    print("[%s] %s cells | %d models | %d candidate content words"
          % (lang, f"{len(rows):,}", len(arm), len(cols)))
    RES = {"n_cells": len(rows), "n_models": len(arm), "n_candidates": len(cols)}

    for N in TOPN:
        #: A: binary, top-N BY RANK within the cell
        B, y, g, mid = [], [], [], []
        for r in rows:
            ps = np.array([p if p else 0.0 for p in r["ps"]], float)
            order = np.argsort(-ps)[:N]
            inrun = {r["ws"][i] for i in order}
            v = np.zeros(len(cols), np.float32)
            for w in inrun:
                j = ci.get(w)
                if j is not None:
                    v[j] = 1.0
            B.append(v); y.append(arm[r["model"]])
            g.append(org[r["model"]]); mid.append(r["model"])
        B = np.array(B); y = np.array(y)
        g = np.array(g, dtype=object); mid = np.array(mid, dtype=object)

        #: B: collapse to one row per model -- the fraction of prompts where the
        #: word was in the running. This IS the independent unit.
        bym = collections.defaultdict(list)
        for i, m in enumerate(mid):
            bym[m].append(i)
        mods = sorted(bym)
        C = np.array([B[bym[m]].mean(0) for m in mods])
        cy = np.array([arm[m] for m in mods])
        cg = np.array([org[m] for m in mods], dtype=object)
        cov = float((B.sum(1) / N).mean())
        print("\n=== TOP-%d BY RANK | every cell contributes exactly %d ones ===" % (N, N))
        print("  %.0f%% of each cell's top-%d falls inside the %d candidate columns"
              % (100 * cov, N, len(cols)))

        def ev_cells(M, k):
            gk = GroupKFold(n_splits=FOLDS)
            pred = np.zeros(len(y))
            for tr, te in gk.split(M, y, groups=g):
                sc = StandardScaler().fit(M[tr][:, :k])
                m = LogisticRegression(max_iter=4000, C=0.1).fit(
                    sc.transform(M[tr][:, :k]), y[tr])
                pred[te] = m.predict_proba(sc.transform(M[te][:, :k]))[:, 1]
            d = collections.defaultdict(list)
            for mm, pp, tt in zip(mid, pred, y):
                d[mm].append((pp, tt))
            mp = np.array([np.mean([a for a, _ in v]) for v in d.values()])
            mt = np.array([v[0][1] for v in d.values()])
            acc = max(accuracy_score(mt, (mp > t).astype(int)) for t in np.unique(mp))
            return roc_auc_score(mt, mp), acc

        def ev_models(k):
            #: LEAVE ONE ORG OUT, because 92 rows and 33 groups is small enough
            #: that 5-fold would leave a lot of variance in the split itself
            logo = LeaveOneGroupOut()
            pred = np.zeros(len(cy))
            for tr, te in logo.split(C[:, :k], cy, groups=cg):
                sc = StandardScaler().fit(C[tr][:, :k])
                m = LogisticRegression(max_iter=4000, C=0.1).fit(
                    sc.transform(C[tr][:, :k]), cy[tr])
                pred[te] = m.predict_proba(sc.transform(C[te][:, :k]))[:, 1]
            acc = max(accuracy_score(cy, (pred > t).astype(int)) for t in np.unique(pred))
            return roc_auc_score(cy, pred), acc

        print("  %-5s %19s %21s" % ("k", "A: binary cells", "B: per-model counts"))
        print("  %-5s %9s %9s %10s %10s" % ("", "AUC", "acc", "AUC", "acc"))
        curves = {}
        for k in KS:
            if k > len(cols):
                continue
            a1, c1 = ev_cells(B, k)
            a2, c2 = ev_models(k)
            curves[k] = {"cells": [a1, c1], "models": [a2, c2]}
            print("  %-5d %9.4f %8.1f%% %10.4f %9.1f%%" % (k, a1, 100 * c1, a2, 100 * c2))

        RES.setdefault("topn", {})[N] = {"coverage": cov, "curves": curves}
        if N == 20:
            #: the readable form: how often is each word in the running, by arm
            base = C[cy == 0].mean(0); algn = C[cy == 1].mean(0)
            d = algn - base
            o = np.argsort(d)
            print("\n  IN THE RUNNING MORE OFTEN WHEN ALIGNED (top-20, share of prompts)")
            for i in o[-10:][::-1]:
                print("     %-14s base %5.1f%%  aligned %5.1f%%  %+5.1f"
                      % (cols[i], 100 * base[i], 100 * algn[i], 100 * d[i]))
            print("  IN THE RUNNING MORE OFTEN WHEN BASE")
            for i in o[:10]:
                print("     %-14s base %5.1f%%  aligned %5.1f%%  %+5.1f"
                      % (cols[i], 100 * base[i], 100 * algn[i], 100 * d[i]))

            #: THE COEFFICIENTS, WHICH ARE NOT THE MARGINAL DIFFERENCES ABOVE.
            #: The table above is each word on its own; these are the weights
            #: the model gives when all k compete, so a word can be marginally
            #: large and get no weight because a correlated neighbour carries
            #: it. Fitted per held-out org and averaged, with the fold-to-fold
            #: stability printed -- an unstable coefficient vector is not a
            #: direction and must not be projected anywhere.
            KC = 50
            cf = []
            for tr, te in LeaveOneGroupOut().split(C[:, :KC], cy, groups=cg):
                sc = StandardScaler().fit(C[tr][:, :KC])
                cf.append(LogisticRegression(max_iter=4000, C=0.1).fit(
                    sc.transform(C[tr][:, :KC]), cy[tr]).coef_[0])
            cf = np.array(cf)
            U = cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-12)
            st = [float(U[i] @ U[j]) for i in range(len(U)) for j in range(i + 1, len(U))]
            w = cf.mean(0)
            print("\n  COEFFICIENTS, k=%d, per-model counts, leave-one-org-out" % KC)
            print("  stability across the %d folds: min cos %.3f, median %.3f"
                  % (len(U), min(st), float(np.median(st))))
            oc = np.argsort(w)
            print("    strongest ALIGNED weight: %s"
                  % ", ".join("%s %+.2f" % (cols[i], w[i]) for i in oc[-10:][::-1]))
            print("    strongest BASE weight:    %s"
                  % ", ".join("%s %+.2f" % (cols[i], w[i]) for i in oc[:10]))

            #: AGAINST THE MOVEMENT AXIS. Sign convention: a coefficient is
            #: POSITIVE when the word is in the running more often in the
            #: ALIGNED arm; an axis position is POSITIVE when the word FALLS
            #: under alignment. Agreement is therefore a NEGATIVE correlation.
            from scipy.stats import spearmanr
            zz = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
            EMV = {ww: vv.astype(np.float64) for ww, vv in zip(zz["words"], zz["E"])}
            ax = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"])
            ax = ax / np.linalg.norm(ax)
            V, keep = [], []
            for i, ww in enumerate(cols[:KC]):
                u = t2u.get(ww, ww)
                if u in EMV:
                    v = EMV[u]; V.append(v / max(np.linalg.norm(v), 1e-12)); keep.append(i)
            V = np.array(V)
            pos = V @ ax
            rho = spearmanr(w[keep], pos).statistic
            proj = w[keep] @ V
            proj = proj / max(np.linalg.norm(proj), 1e-12)
            print("\n  AGAINST THE MOVEMENT AXIS (%d of %d words in GloVe)" % (len(keep), KC))
            print("    Spearman(coefficient, word axis position)  %+.3f" % rho)
            print("    cos(coefficient direction, movement axis)  %+.3f" % float(proj @ ax))
            print("    negative = AGREEMENT (falls under alignment <-> lower when aligned)")
            RES["coefficients"] = {
                "k": KC, "stability_min_cos": min(st),
                "spearman_with_axis": float(rho),
                "cos_with_axis": float(proj @ ax),
                "weights": {cols[i]: float(w[i]) for i in range(KC)},
                "marginal_share": {cols[i]: [float(base[i]), float(algn[i])]
                                   for i in range(len(cols))}}
    json.dump(RES, open(os.path.join(K, "armclf_binary_%s.json" % lang), "w"), indent=1)
    print("\n  -> results/k/armclf_binary_%s.json" % lang)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
