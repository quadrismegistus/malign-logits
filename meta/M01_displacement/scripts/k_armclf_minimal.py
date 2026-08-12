"""How few words are needed to tell an aligned model from its base?

    uv run python meta/M01_displacement/scripts/k_armclf_minimal.py en

`k_armclf` shows 100 content words identify the arm at model-level AUC 0.937 held
out by org, against 0.889 for the single top-1 mass feature. RH's question: what
is the MINIMUM, and does "you only need ten words" mean anything?

**IT MEANS SOMETHING ONLY UNDER NESTED SELECTION, AND THE STABILITY MATTERS MORE
THAN THE COUNT.** Choosing the ten best-separating words on the full sample and
then reporting their accuracy is selection on the outcome; with 400 candidates
and 92 labelled units it would produce a large number that means nothing. Here
the words are ranked INSIDE each training fold, by the absolute t-statistic of
the arm difference computed on training models only, and scored on the held-out
org. The test models never influence which words are chosen.

TWO CURVES, AND THEY ANSWER DIFFERENT QUESTIONS:

    mass-ranked     the k commonest content words, chosen label-blind. Honest by
                    construction and not minimal: it asks how far the ordinary
                    vocabulary gets you.
    selected        the k most arm-discriminative words per training fold. This
                    is the minimum, and it is only interpretable with the
                    stability figure beside it.

STABILITY IS REPORTED AS A JACCARD OVER THE FOLDS' CHOSEN SETS. If five folds
pick nearly the same k words, there is a nameable set and "these ten words" is a
claim. If they pick different words each time and all of them work, the claim is
about the COUNT and not the identity -- a weaker and more interesting statement,
and one that would be actively misleading if reported as a word list.

THE UNIT IS THE MODEL, NOT THE CELL. 92 models, 33 org groups; the ~2,220 prompts
per model are repeated measurements of one label. Both AUC and accuracy are
computed after averaging each model's predicted probability. Accuracy uses a
threshold chosen on the same predictions and is therefore optimistic; it is
printed because it is the number people want, with the count of misclassified
models beside it so the n is visible.
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
CAND = 400
KS = (2, 3, 5, 8, 10, 15, 25, 50, 100)
FOLDS = 5


def main(lang="en"):
    from malign_logits import fields as FL
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    pairs = KP.reps(lang)
    arm, org = {}, {}
    for b, a in pairs:
        arm[b] = 0; arm[a] = 1
        o = b.split("/")[0]; org[b] = o; org[a] = o
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
    X, y, g, mid = [], [], [], []
    for r in rows:
        v = np.full(len(cols), -6.0)
        for w, p in zip(r["ws"], r["ps"]):
            j = ci.get(w)
            if j is not None and p and p > 0:
                v[j] = math.log10(p)
        X.append(v); y.append(arm[r["model"]])
        g.append(org[r["model"]]); mid.append(r["model"])
    X = np.array(X); y = np.array(y)
    g = np.array(g, dtype=object); mid = np.array(mid, dtype=object)
    print("[%s] %s cells | %d models | %d org groups | %d candidate words"
          % (lang, f"{len(y):,}", len(set(mid)), len(set(g)), len(cols)))

    gkf = GroupKFold(n_splits=FOLDS)

    def score(pred):
        bym = collections.defaultdict(list)
        for m, p, t in zip(mid, pred, y):
            bym[m].append((p, t))
        mp = np.array([np.mean([a for a, _ in v]) for v in bym.values()])
        mt = np.array([v[0][1] for v in bym.values()])
        acc = max(accuracy_score(mt, (mp > t).astype(int)) for t in np.unique(mp))
        return roc_auc_score(mt, mp), acc, int(round((1 - acc) * len(mt)))

    def evaluate(k, selected):
        """`selected=False` takes the k commonest words; `selected=True` ranks
        words INSIDE each training fold by |t| of the arm difference across
        training MODELS -- so the held-out org never influences the choice."""
        pred = np.zeros(len(y), float)
        chosen = []
        for tr, te in gkf.split(X, y, groups=g):
            if not selected:
                idx = np.arange(k)
            else:
                #: model means first, so a model with more prompts does not
                #: dominate the t-statistic that picks the words
                mm = collections.defaultdict(list)
                for i in tr:
                    mm[mid[i]].append(i)
                A0, A1 = [], []
                for m, ii in mm.items():
                    (A1 if arm[m] else A0).append(X[ii].mean(0))
                A0, A1 = np.array(A0), np.array(A1)
                sd = np.sqrt(A0.var(0) / max(len(A0), 1) + A1.var(0) / max(len(A1), 1))
                t = np.abs(A1.mean(0) - A0.mean(0)) / np.maximum(sd, 1e-9)
                idx = np.argsort(-t)[:k]
            chosen.append(frozenset(cols[i] for i in idx))
            sc = StandardScaler().fit(X[tr][:, idx])
            m = LogisticRegression(max_iter=4000, C=0.1).fit(sc.transform(X[tr][:, idx]),
                                                             y[tr])
            pred[te] = m.predict_proba(sc.transform(X[te][:, idx]))[:, 1]
        js = [len(a & b) / len(a | b) for i, a in enumerate(chosen)
              for b in chosen[i + 1:]]
        return score(pred) + (float(np.mean(js)) if js else 1.0, chosen)

    print("\nHOW FAR DOES THE ORDINARY VOCABULARY GET YOU?  (k commonest, label-blind)")
    print("  %-5s %8s %10s %9s" % ("k", "AUC", "best acc", "wrong/92"))
    out = {"mass": {}, "selected": {}}
    for k in KS:
        a, acc, wrong, _, _ = evaluate(k, False)
        out["mass"][k] = [a, acc, wrong]
        print("  %-5d %8.4f %9.1f%% %9d" % (k, a, 100 * acc, wrong))

    print("\nAND THE MINIMUM, words chosen INSIDE each training fold")
    print("  %-5s %8s %10s %9s %10s" % ("k", "AUC", "best acc", "wrong/92", "stability"))
    sel_sets = {}
    for k in KS:
        a, acc, wrong, jac, chosen = evaluate(k, True)
        out["selected"][k] = [a, acc, wrong, jac]
        sel_sets[k] = chosen
        print("  %-5d %8.4f %9.1f%% %9d %10.2f" % (k, a, 100 * acc, wrong, jac))
    print("  stability = mean Jaccard between the five folds' chosen sets.")
    print("  Near 1.0 means the folds agree and the WORDS are the claim;")
    print("  low means only the COUNT is, and a word list would mislead.")

    for k in (5, 10):
        if k in sel_sets:
            cnt = collections.Counter(w for s in sel_sets[k] for w in s)
            core = [w for w, c in cnt.most_common() if c == FOLDS]
            print("\n  k=%d: chosen in ALL five folds (%d words): %s"
                  % (k, len(core), ", ".join(core) if core else "(none)"))
            print("       chosen in some: %s"
                  % ", ".join(w for w, c in cnt.most_common() if c < FOLDS))

    json.dump({"n_models": len(set(mid)), "n_orgs": len(set(g)),
               "n_candidates": len(cols), "curves": out,
               "core_at_10": sorted(w for w, c in collections.Counter(
                   w for s in sel_sets.get(10, []) for w in s).items() if c == FOLDS)},
              open(os.path.join(K, "armclf_minimal_%s.json" % lang), "w"), indent=1)
    print("\n  -> results/k/armclf_minimal_%s.json" % lang)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
