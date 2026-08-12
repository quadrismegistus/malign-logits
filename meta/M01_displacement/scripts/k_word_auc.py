"""Per-word arm AUC: one comparable number for every (word, in-context tag).

    uv run python meta/M01_displacement/scripts/k_word_auc.py en
    uv run python meta/M01_displacement/scripts/k_word_auc.py en --source LITERARY
    -> results/k/word_auc_<lang>[_<source>].tsv        every feature, sorted
       and the top 100 each side printed

WHY THIS EXISTS RATHER THAN READING THE BAND COEFFICIENTS. `k_armclf_bands_ctx`
fits 240 separate logistic models, one per band of 50 features, and their
coefficients ARE NOT COMPARABLE ACROSS BANDS. Three reasons, none fixable by
rescaling: each fit allocates a fixed L2 budget across its own fifty features, so
magnitude depends on how much signal that band holds; correlated features inside
a band split their weight, so a redundant band gives each word less than an
independent one does; and standardisation happens within band within fold, so the
units differ. Normalising each band's vector to unit norm makes the entries
"share of THIS band's direction", which is comparable as relative importance
inside a band and still not as effect size across bands.

**So the sweep produces 240 local directions and no global vector.** Concatenating
them would be a category error. A single fit over all 15,832 features is not
available either -- there are 92 labelled models.

WHAT IS COMPARABLE IS A UNIVARIATE STATISTIC COMPUTED IDENTICALLY FOR EVERY
FEATURE. For each (word, tag): the per-model share of that model's top-20 slots,
then the AUC of that one number over the 92 models. Bounded, null at exactly 0.5,
no fitting, no penalty, no feature set to be relative to. It is the vector to
quote across POS, across frequency, and against the movement axis.

ROWS ARE COMPOSITIONS BEFORE THE AUC IS TAKEN. Fixing the top-N at 20 makes each
CELL contribute 20 slots, but the SHARE of those landing inside the tagged
candidate set varies by model -- base 13.656, aligned 13.923 of 20 -- and that
scale factor alone separates the arms at AUC 0.750. Dividing each model's row by
its own total removes it by construction.

THE UNIT IS THE MODEL. 92 of them; the ~2,220 prompts per model are repeated
measurements of one label. A feature must be present in at least MIN_MODELS
models to be scored, or its AUC is a statement about which models happen to have
seen it.

NO HOLDOUT AND NONE NEEDED. This is a descriptive statistic per feature, not a
fitted model, so there is nothing to overfit and nothing to hold out. What it
cannot tell you is whether a word's separation GENERALISES to unseen models --
that is what the held-out classifiers are for, and they agree with this ranking
at Spearman -0.451 against the movement axis.

**0.5 IS THE NULL FOR ONE FEATURE AND NOT FOR THE TABLE'S CENTRE.** Rows sum to 1,
so every model spends the same total across features, and one arm concentrating
its mass on a minority of words pushes the OTHER thousand slightly toward the
other arm. The see-saw moves the whole distribution: on LITERARY prompts the
median feature scores 0.572 and features present in 80+ models score 0.671, while
on the full prompt set the median is 0.504. Same instrument, same normalisation.

So the script reports three references and the table should be read against them:
the observed MEDIAN, a within-lineage arm-flip NULL median, and the AUC of the
concentration scalar itself (row Gini) as a single number. Where the median is far
from 0.5 the ABSOLUTE AUCs carry a global arm difference that is not about any
particular word, and only the RANKING is safe to quote.
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
TOPN = 20
MIN_MODELS = 20
KEEP = ("verb", "noun", "adjective", "adverb")
SHOW = 100


def main(lang="en"):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    SRC = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else None
    out = os.path.join(K, "word_auc_%s%s.tsv" % (lang, "_" + SRC.lower() if SRC else ""))

    tagf = os.path.join(K, "pos_context_%s.tsv" % lang)
    TAG = {}
    for ln in open(tagf, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 4:
            TAG[(p[0], p[1])] = p[3]

    pairs = KP.reps(lang)
    arm, lin = {}, {}
    for i, (b, a) in enumerate(pairs):
        arm[b] = 0; arm[a] = 1
        lin[b] = i; lin[a] = i
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    models = "','".join(esc(m) for m in arm)
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    rows = A.q("""
      SELECT model, prompt, groupArray(word) ws, groupArray(p) ps
      FROM %s.twp_words WHERE model IN ('%s') AND prompt IN (
        SELECT DISTINCT prompt FROM %s.prompt_catalogue
        WHERE status='ACTIVE' AND language='%s'%s)
      GROUP BY model, prompt""" % (A.DB, models, A.DB, lang,
                                   " AND source='%s'" % SRC if SRC else ""))
    mods = sorted(arm)
    mi = {m: i for i, m in enumerate(mods)}
    cnt = collections.defaultdict(lambda: np.zeros(len(mods)))
    nc = np.zeros(len(mods))
    for r in rows:
        ps = np.array([p if p else 0.0 for p in r["ps"]], float)
        i = mi[r["model"]]; nc[i] += 1
        h = sha(r["prompt"])
        for j in np.argsort(-ps)[:TOPN]:
            t = TAG.get((h, r["ws"][j]))
            if t in KEEP:
                cnt[(r["ws"][j], t)][i] += 1
    y = np.array([arm[m] for m in mods])
    print("[%s%s] %s cells | %d models | %s (word,tag) seen"
          % (lang, "/" + SRC if SRC else "", f"{len(rows):,}", len(mods), f"{len(cnt):,}"))

    feats = [f for f, v in cnt.items() if (v > 0).sum() >= MIN_MODELS]
    RAW = np.stack([cnt[f] for f in feats], 1) / np.maximum(nc, 1)[:, None]
    C = RAW / np.maximum(RAW.sum(1, keepdims=True), 1e-12)
    auc = np.array([roc_auc_score(y, C[:, j]) for j in range(len(feats))])
    nmod = (RAW > 0).sum(0)
    br = C[y == 0].mean(0); ar = C[y == 1].mean(0)
    print("  %d features in >=%d models\n" % (len(feats), MIN_MODELS))
    print("  distribution %.3f / %.3f / %.3f / %.3f / %.3f  (5,25,50,75,95 pct)"
          % tuple(np.percentile(auc, [5, 25, 50, 75, 95])))
    print("  |AUC-0.5| > 0.15: %d (%.0f%%)"
          % ((np.abs(auc - .5) > .15).sum(), 100 * (np.abs(auc - .5) > .15).mean()))

    #: three references, because 0.5 is the null for ONE feature and not for the
    #: centre of a table whose rows are constrained to sum to 1
    rng = np.random.default_rng(20260812)
    flip = {i: int(rng.integers(0, 2)) for i in set(lin.values())}
    yn = np.array([arm[m] ^ flip[lin[m]] for m in mods])
    nul = np.array([roc_auc_score(yn, C[:, j]) for j in range(len(feats))])
    S = np.sort(C, 1)
    gini = 1 - 2 * ((np.arange(1, S.shape[1] + 1) - .5) / S.shape[1] * S).sum(1)
    print("\n  REFERENCES  observed median %.3f | arm-flip null median %.3f"
          " | concentration (row Gini) alone AUC %.3f"
          % (np.median(auc), np.median(nul), roc_auc_score(y, gini)))
    if abs(np.median(auc) - .5) > .02:
        print("  ** the table is off-centre: quote the RANKING, not the absolute AUC."
              "\n     a global arm difference in concentration tilts every feature at once.")

    o = np.argsort(auc)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("word\ttag\tauc\tauc_flipnull\tn_models\tbase_share\taligned_share\n")
        for j in np.argsort(-auc):
            fh.write("%s\t%s\t%.4f\t%.4f\t%d\t%.6f\t%.6f\n"
                     % (feats[j][0], feats[j][1], auc[j], nul[j], nmod[j], br[j], ar[j]))

    for lab, idx in (("MOST BASE-SIDE (lowest AUC)", o[:SHOW]),
                     ("MOST ALIGNED-SIDE (highest AUC)", o[::-1][:SHOW])):
        print("\n%s -- top %d" % (lab, SHOW))
        for k in range(0, len(idx), 4):
            print("   " + "  ".join("%-22s %.3f" % ("%s/%s" % feats[j], auc[j])
                                    for j in idx[k:k + 4]))

    print("\nby POS, median |AUC-0.5|")
    for t in KEEP:
        m = [abs(auc[j] - .5) for j, f in enumerate(feats) if f[1] == t]
        if m:
            print("  %-11s %.4f  (%d features)" % (t, float(np.median(m)), len(m)))

    try:
        z = np.load(os.path.join(K, "embed_%s_glove.npz" % lang), allow_pickle=True)
        EM = {w: v for w, v in zip(z["words"], z["E"])}
        t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]
        ax = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"])
        ax = ax / np.linalg.norm(ax)
        ok = [(j, f) for j, f in enumerate(feats) if t2u.get(f[0], f[0]) in EM]
        pos = [float(EM[t2u.get(f[0], f[0])] @ ax
                     / max(np.linalg.norm(EM[t2u.get(f[0], f[0])]), 1e-9)) for _, f in ok]
        r = spearmanr([auc[j] for j, _ in ok], pos).statistic
        print("\nvs the movement axis: Spearman %+.3f over %d features"
              " (negative = agreement)" % (r, len(ok)))
    except Exception as e:
        print("\naxis comparison unavailable: %s" % str(e)[:80])
    print("\n  -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
                  else "en"))
