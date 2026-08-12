"""Plan K's primary analysis: does a word's rated charge predict how alignment
moves it, once base probability and corpus frequency are partialled out?

    uv run python meta/M01_displacement/scripts/k_analysis.py en
    uv run python meta/M01_displacement/scripts/k_analysis.py zh

THE PARTIAL IS THE RESULT AND THE RAW IS THE FOIL. `X_metonymy.md` records a
-0.33 nuisance floor: net movement already tracks base probability at -0.33 at
neutral prompts, and "any word-level scale correlating near -0.3 in this
campaign has explained nothing." Charged words are rarer and lower-probability,
so a raw charge-movement correlation is the floor until shown otherwise. Both
are printed side by side; a scale whose partial collapses toward zero has been
explained by probability and frequency rather than by meaning.

THE NULL IS A LABEL PERMUTATION, NOT A RESAMPLE. Words at one site compete for
mass, so their movements are coupled. Shuffling the RATINGS across words while
leaving every movement number exactly as measured preserves that coupling and
breaks only the link under test. Resampling words would destroy the dependence
structure and give a null that is too narrow.

TWO OUTCOMES, BOTH REPORTED, because they answer different questions:
    net    rises - falls over the word's cells (the X_metonymy net, a count)
    delta  mean (p_aligned - p_base) over the word's cells (mass, continuous)
A word can move in many cells by a little or few cells by a lot.

FREQUENCY IS REGISTER-MATCHED. English uses `coca_fic`, not `fpm_COCA`: the
prompts are narrative fiction, the two correlate at only 0.81, and `scream` is
nearly 3x more frequent in fiction than in COCA overall. SUBTLEX-US is run
alongside as the lineage-matched check. Chinese uses SUBTLEX-CH.

WORDS WITH NO FREQUENCY ENTRY ARE DROPPED FROM THE PARTIAL, NOT ZEROED, and the
count is reported. Zeroing would put every uncovered word at the bottom of the
frequency rank, which is the exact direction that manufactures the effect under
test.
"""
import json
import math
import os
import random
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
K = os.path.join(ROOT, "meta/M01_displacement/results/k")
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
SCALES = ("vulgarity", "register_level", "transgressiveness", "charge",
          "valence", "bodily_harm", "concreteness")
DRAWS = 5000
SEED = 20260812


def q(sql):
    r = subprocess.run([CH, "client", "--query", sql + " FORMAT JSONEachRow"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[:400])
    return [json.loads(l) for l in r.stdout.strip().split("\n") if l.strip()]


#: NUMPY, because the first version was pure Python and could not finish. The
#: permutation is O(draws x n) and n is ~17,000 words, so 20,000 draws over 28
#: tests is ~9.5 billion operations. Vectorised it is seconds. Draws reduced to
#: 5,000, which at this n resolves p far below any threshold that matters.
import numpy as np


def ranks(v):
    a = np.asarray(v, dtype=float)
    order = a.argsort(kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1)
    #: average ties, or a scale with few distinct values (ours are 1-7) gets an
    #: arbitrary within-tie ordering and a correlation that depends on it
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    return sums[inv] / cnt[inv]


def pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def resid(y, X):
    """Residualise y on the columns of X by least squares."""
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def perm_p(x, y, obs, draws=DRAWS, seed=SEED):
    """Two-sided p by permuting the RATING labels across words.

    The labels move; every movement number stays exactly where it was measured.
    Words at one site compete for mass and are therefore coupled, and shuffling
    labels preserves that coupling while breaking only the link under test.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    yc = y - y.mean(); yss = np.sqrt((yc * yc).sum())
    xc = x - x.mean(); xss = np.sqrt((xc * xc).sum())
    if yss == 0 or xss == 0:
        return 1.0
    idx = np.argsort(rng.random((draws, len(x))), axis=1)
    r = (xc[idx] @ yc) / (xss * yss)
    return float((np.abs(r) >= abs(obs)).sum() + 1) / (draws + 1)


def main(lang):
    import k_population as KP
    from k_frequency import fpm
    norm = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))
    rate = json.load(open(os.path.join(K, "ratings_%s.json" % lang)))["ratings"]
    t2u = norm["token_to_unit"]

    edges = KP.reps(lang)
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    pairs = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                        for b, a in edges)
    rows = q("""
      SELECT word,
             countIf(cls='rise') - countIf(cls='fall') AS net,
             avg(delta) AS mdelta, avg(p_base) AS pbase, count() AS cells
      FROM (
        SELECT m.word AS word, m.cls AS cls, m.delta AS delta, m.p_base AS p_base,
               row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                                  ORDER BY m.p_base DESC) rb,
               row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                                  ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m
        INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='%s') p ON m.prompt=p.prompt
        WHERE m.rule='canonical' AND (%s))
      WHERE rb<=50 OR ra<=50 GROUP BY word""" % (DB, DB, lang, pairs))
    print("[%s] %d population words with movement" % (lang, len(rows)))

    #: AGGREGATE TOKENS ONTO THE RATING UNIT. A unit's movement is the sum over
    #: every token that normalises onto it -- the rating is a property of the
    #: word, the movement a property of the token.
    agg = {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in rate:
            continue
        a = agg.setdefault(u, {"net": 0, "dsum": 0.0, "pw": 0.0, "cells": 0})
        a["net"] += r["net"]
        a["dsum"] += r["mdelta"] * r["cells"]
        a["pw"] += r["pbase"] * r["cells"]
        a["cells"] += r["cells"]
    for u, a in agg.items():
        a["delta"] = a["dsum"] / a["cells"]
        a["pbase"] = a["pw"] / a["cells"]
    print("[%s] %d rating units joined to movement" % (lang, len(agg)))

    measures = (["coca_fic", "SUBTLEX_US"] if lang == "en" else ["SUBTLEX_CH"])
    for meas in measures:
        units = [u for u in agg if fpm(u, lang, meas) is not None and agg[u]["pbase"] > 0]
        print("\n[%s] FREQUENCY = %s | %d of %d units have an entry (%.0f%% dropped, "
              "not zeroed)" % (lang, meas, len(units), len(agg),
                               100 * (1 - len(units) / len(agg))))
        lp = ranks([math.log10(agg[u]["pbase"]) for u in units])
        lf = ranks([math.log10(fpm(u, lang, meas)) for u in units])
        print("     control collinearity: rank(log p_base) ~ rank(log fpm)  rho %+.3f"
              % pearson(lp, lf))
        for out_name in ("net", "delta"):
            y = ranks([agg[u][out_name] for u in units])
            yr = resid(y, [lp, lf])
            print("\n     OUTCOME = %s" % out_name)
            print("       %-18s %8s %9s %9s   %s"
                  % ("scale", "raw rho", "partial", "perm p", "verdict"))
            base = pearson(ranks([agg[u]["pbase"] for u in units]), y)
            print("       %-18s %+8.3f %9s %9s   the nuisance floor"
                  % ("(base probability)", base, "--", "--"))
            for s in SCALES:
                x = ranks([rate[u][s] for u in units])
                raw = pearson(x, y)
                par = pearson(resid(x, [lp, lf]), yr)
                p = perm_p(resid(x, [lp, lf]), yr, par)
                keep = abs(par) >= 0.05 and p <= 0.05
                print("       %-18s %+8.3f %+9.3f %9.4f   %s"
                      % (s, raw, par, p,
                         "survives" if keep else
                         "collapses to the floor" if abs(par) < 0.05 else "n.s."))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
