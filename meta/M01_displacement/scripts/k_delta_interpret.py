"""What IS the delta direction? Extremes, correlates, and the within-word test.

    uv run python meta/M01_displacement/scripts/k_delta_interpret.py en
    -> results/k/delta_interpret_<lang>.json

THE THING THAT PREDICTS BEST HAS NEVER BEEN LOOKED INSIDE. `k_delta_predict`
established that a cell-level delta recovers 68-82% of the word-identity
headroom against GloVe's 18-21%, and every interpretation attempt in P so far --
section 4's poles, section 7's decomposition -- has been aimed at the WEAKER
instruments. This points section 7's questions at the strongest one.

**THE DECISIVE NUMBER HERE IS THE WITHIN-WORD AUC.** P section 2's ceiling says
87% of movement variance is within a word across sites and no word-level feature
can reach any of it. A delta is per (prompt, word), so it can in principle. The
test is narrow and unambiguous: hold the word FIXED and ask whether the delta
projection separates that word's own falling cells from its own rising ones. A
word-level feature scores exactly 0.5 here by construction -- it assigns one
value to every cell of the word and cannot order them at all. So anything above
0.5 is variance no feature in this document has ever touched.

Reported against `log p_base` computed the same way, because p_base DOES vary
within a word and is the honest floor rather than 0.5.

THE DIRECTION IS FIT ON ALL CELLS AND THAT IS DELIBERATE. This script describes
a direction rather than estimating generalisation -- `k_delta_predict` already
did the held-out version and is the number to quote for performance. Fitting on
everything here maximises the fidelity of the thing being described, and no claim
below depends on out-of-sample behaviour. Where a number could be read as
performance it is labelled in-sample.

RATINGS ARE FILTERED ON `_instrument`, NOT ON THE FILENAME. `ratings_en.json`
carries 157 entries rated with the zh instrument and 744 non-ASCII words; 0.6% is
small and the filter costs nothing.
"""
import collections
import hashlib
import json
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
DATA = os.path.join(ROOT, "data/raw")
NPC = 256
SEED = 20260813


def main(lang="en"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    from k_frequency import fpm

    z = np.load(os.path.join(DATA, "delta_verbs_%s.npz" % lang), allow_pickle=True)
    D = z["D"].astype(np.float32)
    key = {(p, w): i for i, (p, w) in enumerate(zip(z["prompt_sha16"], z["word"]))}
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    ep = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                     for b, a in KP.reps(lang))
    rows = A.q("""
      SELECT word, prompt, p_base, cls FROM (
        SELECT m.word word, m.prompt prompt, m.p_base p_base, m.cls cls,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                             ORDER BY m.p_base DESC) rb,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                             ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m
        INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='%s') p ON m.prompt=p.prompt
        WHERE m.rule='canonical' AND m.p_base > 0 AND (%s))
      WHERE (rb<=50 OR ra<=50) AND cls IN ('fall','rise')
      ORDER BY word, prompt""" % (A.DB, A.DB, lang, ep))

    X_i, y, g, pr, lp, ptxt = [], [], [], [], [], {}
    for r in rows:
        h = sha(r["prompt"])
        i = key.get((h, r["word"]))
        if i is None:
            continue
        X_i.append(i); y.append(1 if r["cls"] == "fall" else 0)
        g.append(r["word"]); pr.append(h); lp.append(np.log10(r["p_base"]))
        ptxt[h] = r["prompt"]
    X_i = np.array(X_i); y = np.array(y); g = np.array(g)
    pr = np.array(pr); lp = np.array(lp, np.float32)
    print("[%s] %s cells | %s words | %s prompts | fall rate %.3f"
          % (lang, format(len(y), ","), format(len(set(g)), ","),
             format(len(set(pr)), ","), y.mean()))

    pca = PCA(n_components=NPC, random_state=SEED).fit(D[::7])
    DP = pca.transform(D).astype(np.float32)
    F = DP[X_i]
    sc = StandardScaler().fit(F)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(F), y)
    proj = clf.decision_function(sc.transform(F))
    print("  in-sample AUC of the fitted direction %.4f" % roc_auc_score(y, proj))

    #: ---- THE WITHIN-WORD TEST -------------------------------------------
    byw = collections.defaultdict(list)
    for i, w in enumerate(g):
        byw[w].append(i)
    aucs, aucs_p, ns = [], [], []
    for w, idx in byw.items():
        idx = np.array(idx)
        yy = y[idx]
        if yy.min() == yy.max() or len(idx) < 10:
            continue
        aucs.append(roc_auc_score(yy, proj[idx]))
        aucs_p.append(roc_auc_score(yy, lp[idx]))
        ns.append(len(idx))
    aucs = np.array(aucs); aucs_p = np.array(aucs_p); ns = np.array(ns)
    wmean = lambda a: float((a * ns).sum() / ns.sum())
    print("\n  WITHIN-WORD, %d words with both classes and >=10 cells" % len(aucs))
    print("    delta projection   median %.4f  weighted mean %.4f  %.1f%% of words >0.5"
          % (float(np.median(aucs)), wmean(aucs), 100 * (aucs > .5).mean()))
    print("    log p_base         median %.4f  weighted mean %.4f  %.1f%% >0.5"
          % (float(np.median(aucs_p)), wmean(aucs_p), 100 * (aucs_p > .5).mean()))
    print("    a WORD-LEVEL feature scores exactly 0.5 here by construction")

    #: ---- WHAT THE DIRECTION CORRELATES WITH -----------------------------
    wm = {w: float(proj[np.array(idx)].mean()) for w, idx in byw.items()}
    #: per-word scores written out so the pole-list builder has one producer per
    #: instrument rather than recomputing the fit
    with open(os.path.join(K, "delta_word_scores_%s.tsv" % lang), "w",
              encoding="utf-8") as fh:
        fh.write("word\tproj\tn_cells\n")
        for w in sorted(wm, key=lambda x: -wm[x]):
            fh.write("%s\t%.6f\t%d\n" % (w, wm[w], len(byw[w])))

    rate = json.load(open(os.path.join(K, "ratings_%s.json" % lang)))["ratings"]
    rate = {w: v for w, v in rate.items() if v.get("_instrument") == lang}
    scales = ("register_level", "concreteness", "charge", "transgressiveness",
              "valence", "bodily_harm", "vulgarity")
    ws = sorted(wm)
    print("\n  WORD-MEAN PROJECTION correlates with (Spearman)")
    corr = {}
    for s in scales:
        pairs = [(wm[w], rate[w][s]) for w in ws if w in rate and s in rate[w]]
        if len(pairs) > 100:
            r = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic
            corr[s] = float(r)
            print("    %-18s %+.3f   (n=%d)" % (s, r, len(pairs)))
    meas = "coca_fic" if lang == "en" else "SUBTLEX_CH"
    fq = [(wm[w], np.log10(fpm(w, lang, meas))) for w in ws
          if fpm(w, lang, meas) and fpm(w, lang, meas) > 0]
    if fq:
        r = spearmanr([a for a, _ in fq], [b for _, b in fq]).statistic
        corr["log_fpm"] = float(r)
        print("    %-18s %+.3f   (n=%d)" % ("log frequency", r, len(fq)))
    ln = [(wm[w], len(w)) for w in ws]
    corr["length"] = float(spearmanr([a for a, _ in ln], [b for _, b in ln]).statistic)
    print("    %-18s %+.3f   (n=%d)" % ("length", corr["length"], len(ln)))
    try:
        au = {}
        for l2 in open(os.path.join(K, "word_auc_%s.tsv" % lang), encoding="utf-8"):
            p = l2.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0] != "word":
                au.setdefault(p[0], float(p[2]))
        sh = [w for w in ws if w in au]
        r = spearmanr([wm[w] for w in sh], [au[w] for w in sh]).statistic
        corr["per_word_arm_auc"] = float(r)
        print("    %-18s %+.3f   (n=%d)  <- the ARM instrument, a different outcome"
              % ("per-word arm AUC", r, len(sh)))
    except Exception as e:
        print("    per-word arm AUC unavailable: %s" % str(e)[:60])

    #: ---- EXTREME CELLS, WHICH ARE READABLE ------------------------------
    o = np.argsort(proj)
    def show(idx, lab):
        print("\n  %s" % lab)
        seen = set()
        for i in idx:
            k2 = (g[i], pr[i])
            if k2 in seen:
                continue
            seen.add(k2)
            t = ptxt[pr[i]].replace("\n", " ")[-58:]
            print("    %-16s ...%s" % (g[i], t))
            if len(seen) >= 12:
                break
    show(o[::-1], "MOST FALL-SIDE cells (word, and the site it sat at)")
    show(o, "MOST RISE-SIDE cells")

    out = {"lang": lang, "n_cells": int(len(y)), "n_words_within": int(len(aucs)),
           "within_word": {"delta_median": float(np.median(aucs)),
                           "delta_wmean": wmean(aucs),
                           "delta_frac_above_half": float((aucs > .5).mean()),
                           "pbase_median": float(np.median(aucs_p)),
                           "pbase_wmean": wmean(aucs_p)},
           "in_sample_auc": float(roc_auc_score(y, proj)),
           "correlates": corr}
    p = os.path.join(K, "delta_interpret_%s.json" % lang)
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
