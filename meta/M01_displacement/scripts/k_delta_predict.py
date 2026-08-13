"""Can a CELL-level feature beat the word-identity ceiling?

    uv run python meta/M01_displacement/scripts/k_delta_predict.py en
    -> results/k/delta_predict_<lang>.json

THE BAR IS NOT THE OTHER FEATURES, IT IS THE ORACLE. P section 2 measures the
best any function of the word ALONE can reach: a split-half oracle using the
word's own identity buys +0.121 AUC over base probability, and ICC(1) 0.131 says
87% of the movement variance is within a word across sites. GloVe reaches 18-21%
of that headroom, the rated norms 7%. All of them are word-level, so all of them
are competing for the same 13%.

The delta is a different object -- one vector per (prompt, word), so the same word
differs at every site -- and it is therefore NOT bounded by the oracle. Three
outcomes, declared here before the run:

    delta << +0.121   a better word feature and nothing more; P section 2 stands
    delta ~= +0.121   it reaches the word ceiling by a different route
    delta >> +0.121   part of the "unreachable" 87% is reachable, and section 2
                      needs rewriting from a ceiling into a claim about one class

**THE OBVIOUS WAY TO WIN DISHONESTLY IS TO LEARN THE SITE.** V(prompt + word)
contains the prompt, so a delta could carry which PROMPT this is rather than
which word-at-this-prompt, and sites differ in how much they move things. Site
identity is not a discovery. Two controls, both reported:

  - SITE-ONLY MODEL. Replace the delta with V(prompt), constant within a prompt
    and carrying no word information at all. Whatever it scores is the ceiling on
    what a site-learning model could get, and the delta only earns the difference.
  - PROMPT-DISJOINT FOLDS. Beside the by-word split, a split where no test prompt
    appears in training. A model living on site identity cannot transfer.

HELD OUT BY WORD, five-fold GroupKFold, matching section 1 exactly so the number
is comparable to the ones already in the document. The nuisance is `log p_base`
alone -- the same baseline the +0.121 is measured over.

THE DELTA STORE IS A CACHE, NOT A RECORD (`k_delta_embed`): float16 in data/raw,
rebuildable in nine minutes, sidecar committed.
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

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
DATA = os.path.join(ROOT, "data/raw")
FOLDS = 5
SEED = 20260813
NPC = 256


def main(lang="en"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    z = np.load(os.path.join(DATA, "delta_verbs_%s.npz" % lang), allow_pickle=True)
    D = z["D"].astype(np.float32)
    key = {(p, w): i for i, (p, w) in enumerate(zip(z["prompt_sha16"], z["word"]))}
    print("delta store: %s vectors, dim %d" % (format(D.shape[0], ","), D.shape[1]))

    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    #: THE POPULATION IS k_ceiling's, VERBATIM, because the +0.1207 this is
    #: measured against was computed on it. A first version of this script
    #: dropped the representative-pair restriction and the rb/ra<=50 eligibility
    #: and pulled 8,623,990 rows against the ceiling's population -- a number
    #: computed on one population compared to a ceiling from another is the
    #: comparison this document keeps warning about.
    import k_population as KP
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
    print("  %s movement rows fetched" % format(len(rows), ","))

    X_i, y, g, pr = [], [], [], []
    for r in rows:
        h = sha(r["prompt"])
        i = key.get((h, r["word"]))
        if i is None:
            continue
        X_i.append(i); y.append(1 if r["cls"] == "fall" else 0)
        g.append(r["word"]); pr.append(h)
    X_i = np.array(X_i); y = np.array(y); g = np.array(g); pr = np.array(pr)
    lp = np.array([np.log10(r["p_base"]) for r in rows
                   if key.get((sha(r["prompt"]), r["word"])) is not None], np.float32)
    print("  %s cells matched | %s words | %s prompts | fall rate %.3f"
          % (format(len(y), ","), format(len(set(g)), ","),
             format(len(set(pr)), ","), y.mean()))

    #: PCA the delta once, on the store, so every fold sees the same basis. This
    #: is UNSUPERVISED and uses no labels, so it does not leak; refitting per
    #: fold would only add noise.
    pca = PCA(n_components=NPC, random_state=SEED).fit(D[::7])
    DP = pca.transform(D).astype(np.float32)
    print("  PCA %d -> %d, explained %.3f" % (D.shape[1], NPC,
                                              float(pca.explained_variance_ratio_.sum())))
    #: the site-only control: the prompt's own vector, reconstructed as the mean
    #: delta at that prompt is NOT the prompt vector -- so use a per-prompt
    #: one-hot-free proxy: the prompt centroid of deltas carries site info only.
    cent = {}
    for h in set(pr):
        cent[h] = DP[X_i[pr == h]].mean(0)
    SITE = np.stack([cent[h] for h in pr])

    def run(feats, groups, label):
        gk = GroupKFold(n_splits=FOLDS)
        p = np.zeros(len(y))
        for tr, te in gk.split(feats, y, groups=groups):
            sc = StandardScaler().fit(feats[tr])
            m = LogisticRegression(max_iter=3000, C=0.1)
            m.fit(sc.transform(feats[tr]), y[tr])
            p[te] = m.predict_proba(sc.transform(feats[te]))[:, 1]
        a = roc_auc_score(y, p)
        print("    %-34s AUC %.4f" % (label, a))
        return float(a)

    #: THE ORACLE IS COMPUTED HERE, ON THESE ROWS, rather than read from
    #: k_ceiling. A first version compared against P's +0.1207 and got a nuisance
    #: AUC of 0.5046 where that table reports 0.5818 -- the populations differ
    #: (4,064 words here against 2,760 there) and chasing another script's
    #: internals to make them agree is how a comparison ends up between two
    #: things measured on different data. Split-half by word: half a word's cells
    #: predict the other half, which is the best any function of the word alone
    #: can do ON THIS POPULATION, so nuisance/site/delta/oracle are all internal.
    rng = np.random.default_rng(SEED)
    byw = collections.defaultdict(list)
    for i, w in enumerate(g):
        byw[w].append(i)
    orc = np.full(len(y), np.nan)
    for w in sorted(byw):
        idx = np.array(sorted(byw[w]))
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        h1, h2 = idx[perm[:len(idx) // 2]], idx[perm[len(idx) // 2:]]
        if len(h1) == 0 or len(h2) == 0:
            continue
        orc[h2] = y[h1].mean(); orc[h1] = y[h2].mean()
    ok = ~np.isnan(orc)
    oracle = float(roc_auc_score(y[ok], orc[ok]))
    base_ok = float(roc_auc_score(y[ok], lp[ok]))
    print("\n  ORACLE on THIS population: split-half by word %.4f | p_base %.4f"
          " | headroom %+.4f  (n=%s of %s)"
          % (oracle, base_ok, oracle - base_ok, format(int(ok.sum()), ","),
             format(len(y), ",")))
    print("  (P section 2 reports 0.7025 / 0.5818 / +0.1207 on ITS population)")

    out = {"lang": lang, "n_cells": int(len(y)), "n_words": int(len(set(g))),
           "n_prompts": int(len(set(pr))), "npc": NPC,
           "oracle": oracle, "oracle_pbase": base_ok,
           "oracle_headroom": oracle - base_ok}
    N = lp.reshape(-1, 1)
    for gname, groups in (("by WORD", g), ("by PROMPT", pr)):
        print("\n  held out %s" % gname)
        r = {}
        r["nuisance"] = run(N, groups, "log p_base alone")
        r["site"] = run(np.hstack([N, SITE]), groups, "+ site vector (control)")
        r["delta"] = run(np.hstack([N, DP[X_i]]), groups, "+ DELTA (cell-level)")
        r["gain_delta"] = r["delta"] - r["nuisance"]
        r["gain_site"] = r["site"] - r["nuisance"]
        r["delta_over_site"] = r["delta"] - r["site"]
        print("    gain over nuisance: site %+.4f | delta %+.4f | delta-over-site %+.4f"
              % (r["gain_site"], r["gain_delta"], r["delta_over_site"]))
        print("    oracle headroom ON THIS POPULATION is %+.4f" % (oracle - base_ok))
        out[gname.split()[-1].lower()] = r
    p = os.path.join(K, "delta_predict_%s.json" % lang)
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
