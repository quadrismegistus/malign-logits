"""Does the DPO preference direction, measured in an actual alignment corpus,
predict which words alignment pushes down?

    uv run python meta/M01_displacement/scripts/k_dpo_corpus.py

RH's question: why would any of this be true, given post-training corpora are not
fancy in their language? The answer this tests is that they do not need to be
fancy, only NARROW. The base model carries the register range of the web and of
published fiction, including dialogue, which is what our prompts elicit;
preference data is written in one register. If that is the mechanism, the words
alignment pushes down should be the words preference data disprefers -- and that
is measurable directly rather than inferred from COCA genre columns.

THE INSTRUMENT IS INTERNALLY CONTROLLED, which is why it beats the corpus-ratio
version. Anthropic HH-RLHF gives `chosen` and `rejected` for the SAME
conversation, so

    dpo_index(w) = log10( count(w | chosen) / count(w | rejected) )

holds prompt, topic, annotator pool and collection period fixed and varies only
what was preferred. The COCA spoken-over-academic index cannot separate "the
register of alignment data" from "the register of edited non-fiction"; this does
not have that problem because both sides are the same kind of text.

CHOSEN AND REJECTED SHARE THEIR PREFIX AND THE SUFFIX IS THE WHOLE SIGNAL. In
HH-RLHF the two strings are identical up to the point where the responses
diverge -- often hundreds of characters of shared conversation. Counting the full
strings would count the shared text on both sides, diluting the contrast toward
zero and producing a null that looks like a finding. Only the diverging suffix is
counted, and the script reports what fraction of each string that is, so the
reader can see the dilution that was avoided.

SMOOTHING AND A FLOOR. Words appearing on one side only would give an infinite
ratio, so counts are smoothed by +1 and a word must reach MIN_COUNT total
occurrences before it gets an index. Both are declared rather than tuned: the
floor is set for stability of the ratio, not chosen after seeing which value
makes the correlation largest.

THE HARMLESS AND HELPFUL SUBSETS ARE SEPARATED where the dataset allows, because
they are different manipulations. Harmlessness preference is the one displacement
theory is about; helpfulness preference is a different construct that happens to
live in the same file.

WHAT WOULD REFUTE THE REGISTER-CAPTURE READING. If the DPO index does not
correlate with the movement axis and does not predict movement, then alignment is
not moving words toward the register its preference data prefers, and the
register story needs a different mechanism -- or the effect is installed at SFT
rather than at DPO, which `Findings U` already says is where the cutting happens.
That last possibility means a null here is NOT a refutation of register capture,
only of its DPO-stage version, and it must be reported that way.
"""
import collections
import json
import math
import os
import re
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
MIN_COUNT = 40          #: total occurrences across both sides before a word scores
SEED = 20260812
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def diverging(chosen, rejected):
    """The suffix of each string after their common prefix.

    Returns ("", "") when one is a prefix of the other, which happens and would
    otherwise contribute a one-sided count with no contrast in it.
    """
    n = min(len(chosen), len(rejected))
    i = 0
    while i < n and chosen[i] == rejected[i]:
        i += 1
    a, b = chosen[i:], rejected[i:]
    return (a, b) if a and b else ("", "")


def counts(split):
    from datasets import load_dataset
    d = load_dataset("Anthropic/hh-rlhf", **({"data_dir": split} if split else {}),
                     split="train")
    ch, rj = collections.Counter(), collections.Counter()
    kept = tot_c = tot_r = shared = 0
    for r in d:
        a, b = diverging(r["chosen"], r["rejected"])
        if not a:
            continue
        kept += 1
        shared += len(r["chosen"]) - len(a)
        tot_c += len(a); tot_r += len(b)
        for w in WORD.findall(a.lower()):
            ch[w] += 1
        for w in WORD.findall(b.lower()):
            rj[w] += 1
    print("  %-18s %6d pairs used | diverging suffix is %.0f%% of the chosen "
          "string on average | %s / %s word tokens"
          % (split or "all", kept,
             100 * tot_c / max(tot_c + shared, 1),
             f"{sum(ch.values()):,}", f"{sum(rj.values()):,}"))
    return ch, rj


def main():
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    EM = {w: v for w, v in zip(z["words"], z["E"])}
    axis = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"], np.float32)

    print("HH-RLHF, diverging suffixes only")
    IDX = {}
    for split in (None, "harmless-base", "helpful-base"):
        try:
            ch, rj = counts(split)
        except Exception as e:
            print("  %-18s unavailable: %s" % (split, str(e)[:70])); continue
        idx = {}
        for w in set(ch) | set(rj):
            if ch[w] + rj[w] < MIN_COUNT:
                continue
            idx[w] = math.log10((ch[w] + 1) / (rj[w] + 1))
        IDX[split or "all"] = idx
        print("    %d words above the %d-occurrence floor" % (len(idx), MIN_COUNT))

    #: 1. AGREEMENT WITH THE AXIS. + on the axis = FALLS under alignment;
    #: + on the dpo index = PREFERRED by the annotators. Register capture
    #: predicts a NEGATIVE correlation: dispreferred words fall.
    print("\n1. DPO INDEX AGAINST THE MOVEMENT AXIS")
    print("   register capture predicts NEGATIVE: dispreferred words fall\n")
    print("   %-16s %8s %13s %14s %16s"
          % ("subset", "n words", "rho w/ axis", "rho w/ coder", "rho w/ SUBTLEX"))
    for name, idx in IDX.items():
        common = [u for u in EM if u.strip().lower() in idx and u in rate]
        if len(common) < 50:
            print("   %-16s too few (%d)" % (name, len(common))); continue
        v = [idx[u.strip().lower()] for u in common]
        ra = spearmanr(v, [float(EM[u] @ axis) for u in common]).statistic
        rc = spearmanr(v, [rate[u]["register_level"] for u in common]).statistic
        sub = [(idx[u.strip().lower()],
                math.log10(fpm(u, "en", "SUBTLEX_US") / fpm(u, "en", "coca_acad")))
               for u in common
               if fpm(u, "en", "SUBTLEX_US") and fpm(u, "en", "coca_acad")]
        rs = spearmanr([a for a, _ in sub], [b for _, b in sub]).statistic if len(sub) > 50 else float("nan")
        print("   %-16s %8d %+13.3f %+14.3f %+16.3f" % (name, len(common), ra, rc, rs))

    key = "harmless-base" if "harmless-base" in IDX else "all"
    idx = IDX[key]
    o = sorted(idx, key=lambda w: idx[w])
    print("\n2. THE POLES OF THE PREFERENCE DIRECTION (%s)" % key)
    print("   most REJECTED:  %s" % ", ".join(o[:30]))
    print("   most CHOSEN:    %s" % ", ".join(o[-30:][::-1]))

    #: 3. DOES IT PREDICT MOVEMENT?
    rows = KP2.fetch("en", False)
    rng = np.random.default_rng(SEED)
    Xn, c, y, g, site, fq = [], [], [], [], [], {}
    for r in rows:
        u = t2u.get(r["word"])
        if u is None or u not in EM or r["p_base"] <= 0:
            continue
        lw = u.strip().lower()
        if lw not in idx:
            continue
        if u not in fq:
            fq[u] = fpm(u, "en", "coca_fic")
        if not fq[u]:
            continue
        Xn.append([math.log10(r["p_base"]), math.log10(fq[u])])
        c.append(idx[lw]); y.append(1 if r["cls"] == "fall" else 0)
        g.append(u); site.append(hash((r["prompt"], r["base"], r["aligned"])))
    Xn = np.array(Xn); C = np.array(c)[:, None]; y = np.array(y)
    g = np.array(g, object); site = np.array(site)
    words = sorted(set(g))
    sh = dict(zip(words, rng.permutation([idx[w.strip().lower()] for w in words])))
    S = np.array([sh[u] for u in g])[:, None]
    print("\n3. DOES THE PREFERENCE DIRECTION PREDICT MOVEMENT? held out by word")
    print("   %s cells | %d words" % (f"{len(y):,}", len(words)))
    gkf = GroupKFold(n_splits=KP2.FOLDS)
    out = {}
    for nm, M in (("nuisance", Xn), ("+ dpo index", np.hstack([Xn, C])),
                  ("+ dpo SHUFFLED", np.hstack([Xn, S]))):
        p = np.zeros(len(y))
        for tr, te in gkf.split(M, y, groups=g):
            sc = StandardScaler().fit(M[tr])
            p[te] = LogisticRegression(max_iter=4000).fit(
                sc.transform(M[tr]), y[tr]).predict_proba(sc.transform(M[te]))[:, 1]
        ps, _ = KP2.per_site_auc(site, y, p)
        out[nm] = [float(roc_auc_score(y, p)), ps]
        print("   %-18s pooled %.4f   per-site %.4f" % (nm, out[nm][0], ps))
    print("   dpo index over its OWN shuffle  %+.4f / %+.4f"
          % (out["+ dpo index"][0] - out["+ dpo SHUFFLED"][0],
             out["+ dpo index"][1] - out["+ dpo SHUFFLED"][1]))
    print("\n   A NULL HERE DOES NOT REFUTE REGISTER CAPTURE, only its DPO-stage")
    print("   version: Findings U puts the cutting at SFT, not DPO.")

    json.dump({"min_count": MIN_COUNT, "subsets": {k: len(v) for k, v in IDX.items()},
               "auc": out}, open(os.path.join(K, "dpo_corpus_en.json"), "w"), indent=1)
    print("\n  -> results/k/dpo_corpus_en.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
