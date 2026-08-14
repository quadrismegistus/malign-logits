"""How much of the movement axis is concreteness? Two independent measures.

    uv run python meta/M01_displacement/scripts/k_concreteness.py

Concreteness is the only rated norm that survives out of sample once the sites
are restricted to the verb-eliciting minimal pairs (+0.0055 per-site, first of
ten, `k_scale_solo`). So the question the length decomposition asked about length
now has to be asked about concreteness: is the axis a concreteness direction?

TWO MEASURES, EACH ON ITS OWN OVERLAP, because a single one cannot separate the
construct from the instrument:

    coder concreteness    our LLM rating, all rated verbs, IAA 0.83
    Brysbaert Conc.M      human norms, 39,954 words, whatever intersects

They calibrate at 0.88 in K's own records, so they should give the same answer.
If they do not, the disagreement is about the instrument and neither number means
what it appears to. Each is used on ITS OWN coverage rather than on the
intersection, so the coder measure is not penalised for Brysbaert's gaps.

THE SAME THREE QUESTIONS AS `k_length`, and the third is the one that decides:

    1. VARIANCE   R2 of the axis projection on concreteness, word level.
    2. GEOMETRY   the GloVe direction that best predicts concreteness, and its
                  cosine with the axis. Sharing variance and being the same
                  direction are different claims.
    3. PREDICTION project concreteness out of the axis and re-run held out by
                  word. If the concreteness-free axis still beats its shuffle,
                  the axis is not concreteness. If it collapses, it is, and the
                  register reading of the poles was decoration on a concreteness
                  effect.

RUN ON THE VERB-ELICITING SITES, the M01 minimal pairs, because that is the
population where the per-norm result exists. Running it on all sites would test
the axis against a concreteness effect measured where the effect is not there.

BRYSBAERT IS THE ONE TO BELIEVE IF THEY DIVERGE. It is human, it is the external
standard our coder was validated against, and it cannot inherit any bias from the
instrument that produced the ratings whose behaviour we are trying to explain.
"""
import csv
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
import k_population as KP
from k_frequency import fpm

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
BRYS = ("/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources/"
        "Concreteness_ratings_Brysbaert_et_al_BRM.txt")
SEED = 20260812


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def fetch_pairs():
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    ep = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                     for b, a in KP.reps("en"))
    return A.q("""
      SELECT word, prompt, base, aligned, cls, p_base FROM (
        SELECT *, row_number() OVER (PARTITION BY word
                 ORDER BY cityHash64(word, prompt, base, aligned)) rw FROM (
          SELECT m.word word, m.prompt prompt, m.base base, m.aligned aligned,
                 m.cls cls, m.p_base p_base,
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


def main():
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    EM = {w: unit(v.astype(np.float64)) for w, v in zip(z["words"], z["E"])}
    #: --pairs-axis uses the axis REFITTED on verb-eliciting sites. The default
    #: axis was fitted over every ACTIVE prompt; k_axis --pairs refits it on the
    #: M01 minimal pairs, where concreteness rises to +0.311 and overtakes
    #: register_level at -0.209 as the top-correlating scale. Since the whole
    #: point of this script is to say which named component the axis is made of,
    #: the answer may depend on which axis, and both are run rather than one
    #: being picked.
    af = "axis_en_pairs.json" if "--pairs-axis" in sys.argv else "axis_en.json"
    ax = unit(np.array(json.load(open(os.path.join(K, af)))["axis"]))
    print("AXIS: %s" % af)
    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]

    CC = {}
    with open(BRYS, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                CC[r["Word"].strip().lower()] = float(r["Conc.M"])
            except (KeyError, TypeError, ValueError):
                pass

    #: `--register` runs the identical decomposition on the register pair, so
    #: concreteness and register are judged by the same procedure rather than by
    #: a correlation for one and a full decomposition for the other. Each
    #: construct gets one CODER measure and one INDEPENDENT measure, on its own
    #: coverage: Brysbaert's human norms for concreteness, the SUBTLEX-over-
    #: academic genre ratio for register. The register direction here is fitted
    #: from the SCALE, not from near-synonym difference vectors -- that
    #: construction failed in `k_residual`, correlating -0.008 with the coder
    #: scale because averaging 385 GloVe difference vectors yields a frequency
    #: direction.
    def regidx(w):
        a, b = fpm(w, "en", "SUBTLEX_US"), fpm(w, "en", "coca_acad")
        return math.log10(a / b) if a and b else None

    if "--register" in sys.argv:
        #: THE THIRD COLUMN IS THE ONE REGISTRAR ASKED FOR. The SUBTLEX/acad
        #: index is a ratio of two frequencies and the axis correlates with log
        #: frequency at +0.220, so part of its R2 may be frequency wearing a
        #: genre ratio. Residualising it on log coca_fic -- the SAME frequency
        #: that sits in the nuisance block, so the comparison is against what
        #: the model already controls for -- leaves the part of the genre ratio
        #: that overall frequency does not explain. If the residualised index
        #: keeps its share, register survives; if it collapses toward the coder
        #: scale's 0.047, the largest named component was frequency.
        raw = {w: v for w in EM for v in [regidx(w)] if v is not None}
        fq = {w: math.log10(f) for w in raw for f in [fpm(w, "en", "coca_fic")] if f}
        ws = sorted(set(raw) & set(fq))
        X = np.column_stack([np.ones(len(ws)), [fq[w] for w in ws]])
        yv = np.array([raw[w] for w in ws])
        rr = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
        MEAS = {"coder register_level": {w: float(rate[w]["register_level"])
                                         for w in EM if w in rate},
                "SUBTLEX/acad index": raw,
                "SUBTLEX resid on freq": dict(zip(ws, rr))}
    else:
        MEAS = {"coder concreteness": {w: float(rate[w]["concreteness"])
                                       for w in EM if w in rate},
                "Brysbaert Conc.M": {w: CC[w.strip().lower()]
                                     for w in EM if w.strip().lower() in CC}}
    ka, kb = list(MEAS)[:2]
    both = [w for w in MEAS[ka] if w in MEAS[kb]]
    print("COVERAGE OF THE %d GLOVE VERBS" % len(EM))
    for n, d in MEAS.items():
        print("  %-22s %5d words (%.0f%%)" % (n, len(d), 100 * len(d) / len(EM)))
    print("  the two agree at rho %+.3f over the %d shared"
          % (spearmanr([MEAS[ka][w] for w in both],
                       [MEAS[kb][w] for w in both]).statistic, len(both)))

    def r2(y, X):
        X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - (y - X @ b).var() / y.var()

    print("\n1. VARIANCE AND 2. GEOMETRY, each measure on ITS OWN overlap")
    print("   %-22s %7s %9s %14s %16s"
          % ("measure", "n", "R2 axis", "rho w/ axis", "cos(axis, dir)"))
    FREE = {}
    #: EMITTED, not only printed. @dario at [6164] could verify three of six rows
    #: of P §7's table and not these, because this producer computed them and
    #: wrote nothing: no `json.dump` anywhere, and `results/k/analysis_en.log`
    #: holds none of the values and is untracked. So the numbers reached the doc
    #: by transcription, which is the only mechanism consistent with the
    #: concreteness row carrying `0.09209674697588033` -- the LENGTH row's value,
    #: whose full-precision form has exactly one hit in the repository
    #: (`results/k/length_en_glove.json`). Ruled at [6166]: the values were
    #: unverifiable, not unrecoverable, and this is the discharge.
    TABLE = {}
    for n, d in MEAS.items():
        ws = sorted(d)
        E = np.array([EM[w] for w in ws]); P = E @ ax
        c = np.array([d[w] for w in ws])
        cd = unit(Ridge(alpha=1.0).fit(E, c).coef_)
        FREE[n] = unit(ax - (ax @ cd) * cd)
        r2_axis = r2(P, [c])
        rho = spearmanr(P, c).statistic
        cos_dir = float(ax @ cd)
        cos_free = float(ax @ FREE[n])
        TABLE[n] = {"n": len(ws), "r2_axis": float(r2_axis), "rho_with_axis": float(rho),
                    "cos_axis_dir": cos_dir, "cos_axis_free": cos_free}
        print("   %-22s %7d %9.4f %+14.3f %+16.3f"
              % (n, len(ws), r2_axis, rho, cos_dir))
        print("     cos(axis, concreteness-free axis) %+.3f" % cos_free)

    #: FULL PRECISION IN THE ARTIFACT, four decimals on screen. A rounded value
    #: is a name and the full-precision value is closer to a relation: `0.0921`
    #: matches three unrelated CSVs in `data/` as a coincidental substring, and
    #: `0.09209674697588033` matches exactly one file. Only the second can
    #: establish where a number came from.
    _p = os.path.join(K, "concreteness_en.json")
    json.dump({"measures": TABLE, "axis": af}, open(_p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(_p, ROOT))

    #: 3. PREDICTION on the verb-eliciting sites
    rows = fetch_pairs()
    rng = np.random.default_rng(SEED)
    print("\n3. PREDICTION, verb-eliciting sites, held out by word")
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    for n, d in MEAS.items():
        Xn, cols, y, g, site, fq = [], [], [], [], [], {}
        free = FREE[n]
        for r in rows:
            u = t2u.get(r["word"])
            if u is None or u not in d or r["p_base"] <= 0:
                continue
            if u not in fq:
                fq[u] = fpm(u, "en", "coca_fic")
            if not fq[u]:
                continue
            Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
            cols.append([float(EM[u] @ ax), float(EM[u] @ free), d[u]])
            y.append(1 if r["cls"] == "fall" else 0)
            g.append(u); site.append(hash((r["prompt"], r["base"], r["aligned"])))
        Xn = np.array(Xn); C = np.array(cols); y = np.array(y)
        g = np.array(g, object); site = np.array(site)
        ws = sorted(set(g))
        shf = dict(zip(ws, rng.permutation([float(EM[w] @ free) for w in ws])))
        S = np.array([shf[u] for u in g])

        def run(M):
            p = np.zeros(len(y))
            for tr, te in gkf.split(M, y, groups=g):
                sc = StandardScaler().fit(M[tr])
                p[te] = LogisticRegression(max_iter=4000).fit(
                    sc.transform(M[tr]), y[tr]).predict_proba(sc.transform(M[te]))[:, 1]
            return float(roc_auc_score(y, p)), KP2.per_site_auc(site, y, p)[0]
        print("\n   %s | %s cells | %d words" % (n, f"{len(y):,}", len(ws)))
        b = run(Xn)
        a_ = run(np.column_stack([Xn, C[:, 0]]))
        f_ = run(np.column_stack([Xn, C[:, 1]]))
        c_ = run(np.column_stack([Xn, C[:, 2]]))
        s_ = run(np.column_stack([Xn, S]))
        print("     nuisance                    %.4f / %.4f" % b)
        print("     + axis                      %.4f / %.4f   over floor %+.4f / %+.4f"
              % (a_[0], a_[1], a_[0] - b[0], a_[1] - b[1]))
        print("     + concreteness alone        %.4f / %.4f   over floor %+.4f / %+.4f"
              % (c_[0], c_[1], c_[0] - b[0], c_[1] - b[1]))
        print("     + CONCRETENESS-FREE axis    %.4f / %.4f   over its shuffle %+.4f / %+.4f"
              % (f_[0], f_[1], f_[0] - s_[0], f_[1] - s_[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
