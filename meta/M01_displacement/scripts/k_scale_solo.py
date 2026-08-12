"""Each norm ALONE, held out by word. Which single dimension predicts movement?

    uv run python meta/M01_displacement/scripts/k_scale_solo.py

Every prediction number in K so far is for the seven or eighteen norms TOGETHER,
while every claim about which dimension matters rests on in-sample correlations.
That is the combination that collapsed for charge and transgressiveness once they
were held out by word, so the per-dimension version has to be run before any
scale is called the best explanation.

TWO POPULATIONS, BECAUSE WARRINER COVERS 22% OF THE VERBS. On the full verb set
only the ten coder columns exist; the eight Warriner columns exist on a quarter
of it. Running each norm on whatever population it happens to cover would make
the increments incomparable -- a norm scored on 4,075 words and one scored on
1,042 are not competing on the same task. So the eighteen are compared on the
INTERSECTION, where all of them are defined, and the ten are additionally
reported on the full set. The two tables are not comparable to each other and are
printed separately for that reason.

EACH NORM IS SCORED AGAINST ITS OWN SHUFFLE, not against the nuisance floor
alone, because the floor drifts with the population and a norm's increment over
it is not a clean quantity.

READ THE RANKING WITH THE RESOLUTION IN MIND. Between two runs of `k_register`
the nuisance floor moved 0.0004 purely from a 326-cell change in the covered
population. Differences below roughly 0.001 in this table are not resolvable, and
the sensible reading is which norms clear that band, not which one is top.

RELIABILITIES ARE PRINTED BESIDE THE INCREMENTS. Comparing raw correlations
across scales measured with different reliability systematically favours the
better-measured one -- concreteness at IAA 0.83 against register_level at 0.60.
Out-of-sample increments are less exposed to this than correlations are, since a
noisy predictor is penalised in both the fit and the test, but the reliabilities
are shown so the reader can discount rather than take the order at face value.
"""
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
from k_frequency import fpm

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
SEED = 20260812
#: pilot IAA, results/k/iaa.json. Warriner columns are human norms and carry no
#: comparable figure here, so they are shown as "--" rather than given a number.
REL = {"valence": .90, "vulgarity": .88, "bodily_harm": .88, "charge": .87,
       "transgressiveness": .83, "concreteness": .83, "register_level": .60}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from malign_logits import fields as FL

    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    verbs = {u for u in rate
             if (FL._byu().get(u.strip().lower()) or ("", "x"))[1].startswith("vv")}
    rv = {u: v for u, v in rate.items() if u in verbs}
    T, coder, ext = KP2.feature_table("en", rv)

    #: SITES MUST ELICIT VERBS OR THE COMPARISON IS NOT COMMENSURABLE. Restricting
    #: the WORDS to lexical verbs is not enough: at a site like "She slowly took
    #: off her ___" the verbs in the top-50 are long-shot candidates competing
    #: against nouns, and their movement is pooled with verbs at "He began to
    #: ___" where verbs are the real competitors. Within-site ranking only means
    #: something when the candidates are of a kind. The M01 minimal-pair corpus
    #: is verb-eliciting BY DESIGN, so pair_role in (MARKED, UNMARKED) selects
    #: the sites where the question is well posed, rather than inferring
    #: elicitation from the model's own output -- which would select sites on
    #: the base model's behaviour, a property correlated with the outcome.
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    ep = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                     for b, a in __import__("k_population").reps("en"))
    rows = A.q("""
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
                      WHERE status='ACTIVE' AND language='en'
                        AND pair_role IN ('MARKED','UNMARKED')) p ON m.prompt=p.prompt
          WHERE m.rule='canonical' AND (%s))
        WHERE (rb<=50 OR ra<=50) AND cls IN ('fall','rise'))
      WHERE rw <= %d""" % (A.DB, A.DB, ep, KP2.CAP))
    print("VERB-ELICITING SITES ONLY: M01 minimal pairs, %s mover cells over %d prompts"
          % (f"{len(rows):,}", len({r["prompt"] for r in rows})))
    rng = np.random.default_rng(SEED)
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    res = {}

    for want_ext, label in ((False, "TEN CODER NORMS, all verbs"),
                            (True, "ALL EIGHTEEN, Warriner-covered verbs")):
        cols = coder + (ext if want_ext else [])
        elig = {u for u in T if not want_ext or all(c in T[u] for c in ext)}
        Xn, C, y, g, site, fq = [], [], [], [], [], {}
        for r in rows:
            u = t2u.get(r["word"])
            if u is None or u not in elig or r["p_base"] <= 0:
                continue
            if u not in fq:
                fq[u] = fpm(u, "en", "coca_fic")
            if not fq[u]:
                continue
            Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
            C.append([float(T[u][c]) for c in cols])
            y.append(1 if r["cls"] == "fall" else 0)
            g.append(u); site.append(hash((r["prompt"], r["base"], r["aligned"])))
        Xn = np.array(Xn); C = np.array(C); y = np.array(y)
        g = np.array(g, object); site = np.array(site)
        words = sorted(set(g))

        def run(M):
            p = np.zeros(len(y))
            for tr, te in gkf.split(M, y, groups=g):
                sc = StandardScaler().fit(M[tr])
                p[te] = LogisticRegression(max_iter=4000).fit(
                    sc.transform(M[tr]), y[tr]).predict_proba(sc.transform(M[te]))[:, 1]
            return float(roc_auc_score(y, p)), KP2.per_site_auc(site, y, p)[0]

        print("\n%s | %s cells | %d words | %d norms"
              % (label, f"{len(y):,}", len(words), len(cols)))
        b = run(Xn)
        print("  nuisance floor  pooled %.4f  per-site %.4f\n" % b)
        print("  %-26s %10s %10s   %6s" % ("norm", "pooled", "per-site", "rel"))
        r = {}
        for j, s in enumerate(cols):
            real = run(np.column_stack([Xn, C[:, j]]))
            m = dict(zip(words, rng.permutation([T[w][s] for w in words])))
            shuf = run(np.column_stack([Xn, np.array([m[u] for u in g])]))
            r[s] = (real[0] - shuf[0], real[1] - shuf[1])
            print("  %-26s %+10.4f %+10.4f   %6s"
                  % (s, r[s][0], r[s][1],
                     ("%.2f" % REL[s]) if s in REL else "--"))
        print("\n  ranked by per-site increment over its own shuffle")
        for s, v in sorted(r.items(), key=lambda t: -t[1][1]):
            bar = "#" * max(0, int(round(v[1] * 1000)))
            print("    %-26s %+.4f %s" % (s, v[1], bar))
        print("  (differences below ~0.001 are not resolvable; the floor itself")
        print("   moves that much on a 326-cell change in population)")
        res[label] = {"n_cells": int(len(y)), "n_words": len(words),
                      "floor": list(b), "solo": {k: list(v) for k, v in r.items()}}

    json.dump(res, open(os.path.join(K, "scale_solo_en.json"), "w"), indent=1)
    print("\n  -> results/k/scale_solo_en.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
