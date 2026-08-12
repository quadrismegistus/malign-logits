"""How well could ANY word-level feature predict movement? The split-half oracle.

    uv run python meta/M01_displacement/scripts/k_ceiling.py en
    uv run python meta/M01_displacement/scripts/k_ceiling.py zh

`k_predict` reports that the rated norms barely predict which way alignment moves
a word. That result is uninterpretable without this one, because it conflates two
completely different findings:

    (a) MEANING does not predict movement, but other word properties might
    (b) NOTHING about the word predicts movement, because movement is a property
        of the word AT A SITE and not of the word

THE ORACLE THAT SETTLES IT. Take a word's own cells and split them in half at
random. Use the fall rate observed in half A to predict individual cells in half
B. This is the best score achievable by any function of the word alone -- it uses
the word's identity, which strictly dominates any finite set of word features,
and it is measured out of sample so it is not a training fit.

If the oracle is near chance, the norms have not failed; the design has a ceiling
and no word-level instrument can beat it. If the oracle is high and the norms are
near chance, then word identity carries real information the norms are missing
and a better instrument is worth building.

THE SPLIT IS BY CELL, NOT BY WORD, AND THAT IS THE POINT. `k_predict` holds out
WORDS, which is right for "does this generalise to a new word" -- but it makes
the oracle uncomputable, since a held-out word has no observed rate. Splitting a
word's own cells asks the different and prior question: is a word's direction of
movement even STABLE across the sites it appears at?

REPORTED ALONGSIDE, because they are the same fact from three directions:
    oracle AUC        out-of-sample, using only the word's own other cells
    ICC               share of outcome variance that is BETWEEN words
    per-site oracle   the same oracle scored within (prompt, base, aligned),
                      which is the comparison `k_predict`'s per-site AUC is
                      measured against
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

SEED = 20260812
MIN_CELLS = 10      #: a word needs this many movers before a split half means anything
MIN_SITE = 4


def fetch(lang, pairs=False):
    """`pairs=True` restricts to the verb-eliciting M01 minimal pairs.

    The headroom this script measures is the denominator for every "% of the
    headroom recovered" figure in P, and those were all computed over every
    ACTIVE prompt -- including noun-eliciting sites where a verb in the top-50 is
    a long shot. If the ceiling differs on the restricted population, the
    headline 21%-vs-7% differs with it, so the denominator has to be available on
    both populations rather than assumed stable.
    """
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    ep = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                     for b, a in KP.reps(lang))
    return A.q("""
      SELECT word, prompt, base, aligned, cls, p_base, p_aligned FROM (
        SELECT m.word word, m.prompt prompt, m.base base, m.aligned aligned,
               m.cls cls, m.p_base p_base, m.p_aligned p_aligned,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                             ORDER BY m.p_base DESC) rb,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt
                             ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m
        INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='%s'%s) p ON m.prompt=p.prompt
        WHERE m.rule='canonical' AND (%s))
      WHERE (rb<=50 OR ra<=50) AND cls IN ('fall','rise')"""
               % (A.DB, A.DB, lang,
                  " AND pair_role IN ('MARKED','UNMARKED')" if pairs else "", ep))


def main(lang):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    rows = fetch(lang, "--pairs" in sys.argv)
    verbs = None
    if "--verbs" in sys.argv:
        #: the same restriction k_predict uses, so the two are comparable
        from malign_logits import fields as FL
        rate_units = json.load(open(os.path.join(
            ROOT, "meta/M01_displacement/results/k/ratings_%s.json" % lang)))["ratings"]
        t2u = json.load(open(os.path.join(
            ROOT, "meta/M01_displacement/results/k/normalisation_%s.json"
            % lang)))["token_to_unit"]
        if lang == "en":
            ok = {u for u in rate_units
                  if (FL._byu().get(u.strip().lower()) or ("", "x"))[1].startswith("vv")}
        else:
            import jieba.posseg as pseg
            ok = {u for u in rate_units
                  for seg in [list(pseg.cut(u.strip()))]
                  if len(seg) == 1 and seg[0].flag.startswith("v")}
        verbs = {tok for tok, u in t2u.items() if u in ok}

    byw = collections.defaultdict(list)
    for r in rows:
        if verbs is not None and r["word"] not in verbs:
            continue
        #: THE SITE KEY IS A STRING, NOT hash(). Python randomises string
        #: hashing per process, so hash((prompt, base, aligned)) differs between
        #: runs; grouping by it is fine WITHIN a run but it cannot be used as a
        #: sort key, and it was one.
        byw[r["word"]].append(("%s\x00%s\x00%s" % (r["prompt"], r["base"], r["aligned"]),
                               1 if r["cls"] == "fall" else 0,
                               math.log10(r["p_base"]) if r["p_base"] > 0 else -9.0))
    words = {w: v for w, v in byw.items() if len(v) >= MIN_CELLS}
    print("\n[%s]%s %s mover cells | %d words | %d words with >=%d cells"
          % (lang, " VERBS ONLY" if verbs is not None else "",
             f"{len(rows):,}", len(byw), len(words), MIN_CELLS))

    #: SPLIT EACH WORD'S OWN CELLS. Half A supplies the rate, half B is scored.
    #: log p_base is carried through so the nuisance model can be scored on
    #: EXACTLY the same cells -- comparing the oracle here against k_predict's
    #: nuisance AUC over there would be two populations and two splits.
    #: SORTED, BECAUSE THE SEED ALONE DOES NOT MAKE THIS DETERMINISTIC. `words`
    #: is built by insertion from the ClickHouse result set, and that query has
    #: no ORDER BY, so row order can differ between runs. The rng then draws its
    #: permutations against a different word order and the split changes. Two
    #: identical invocations returned headroom +0.1186 and +0.1174 before this
    #: line existed -- a 0.0012 spread that is a useful resolution figure and a
    #: bad property for a number other documents cite.
    y, pred, site, pb, kept = [], [], [], [], 0
    for w in sorted(words):
        #: SORTED WITHIN THE WORD TOO. Sorting only the words left the CELLS in
        #: result-set order, so the permutation still landed differently between
        #: runs. Both levels have to be fixed.
        v = sorted(words[w])
        idx = rng.permutation(len(v))
        h = len(v) // 2
        rate = float(np.mean([v[i][1] for i in idx[:h]]))
        kept += 1
        for i in idx[h:]:
            s, lab, lp = v[i]
            y.append(lab); pred.append(rate); site.append(s); pb.append(lp)
    y = np.array(y); pred = np.array(pred)
    site = np.array(site, dtype=object); pb = np.array(pb)
    print("  oracle scored on %s held-out cells from %d words" % (f"{len(y):,}", kept))

    auc = roc_auc_score(y, pred)
    #: THE HEADROOM IS THE NUMBER THAT MATTERS, not the oracle on its own. The
    #: oracle knows the word's identity and therefore also knows everything
    #: p_base knows about it, so its raw AUC includes the nuisance floor.
    auc_pb = roc_auc_score(y, pb)
    print("\n  ORACLE AUC (word identity, out of sample)          %.4f" % auc)
    print("  log p_base alone, SAME cells                        %.4f" % auc_pb)
    print("  HEADROOM above the nuisance any word feature has    %+.4f" % (auc - auc_pb))
    print("  base rate of fall in the scored half                %.4f" % y.mean())

    idx = collections.defaultdict(list)
    for i, s in enumerate(site):
        idx[s].append(i)
    ps = [roc_auc_score(y[ii], pred[ii]) for ii in idx.values()
          if len(ii) >= MIN_SITE and y[ii].min() != y[ii].max()
          and len(set(pred[ii])) > 1]
    print("  ORACLE per-site AUC                                 %.4f  (%d sites)"
          % (float(np.mean(ps)) if ps else float("nan"), len(ps)))

    #: ICC: how much of the fall/rise variance lives BETWEEN words rather than
    #: within a word across its sites. One minus this is the share no word-level
    #: feature can ever reach.
    grand = float(np.mean([t[0] for v in words.values() for t in v]))
    ns = [len(v) for v in words.values()]
    means = [float(np.mean([t[0] for t in v])) for v in words.values()]
    ssb = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    ssw = sum(sum((t[0] - m) ** 2 for t in v)
              for v, m in zip(words.values(), means))
    k = len(words)
    n_tot = sum(ns)
    msb = ssb / (k - 1)
    msw = ssw / (n_tot - k)
    n0 = (n_tot - sum(n * n for n in ns) / n_tot) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    print("\n  ICC(1): share of fall/rise variance BETWEEN words   %.4f" % icc)
    print("  -> %.0f%% of the variance is WITHIN a word, across the sites it"
          % (100 * (1 - icc)))
    print("     appears at, and is unreachable by any word-level feature.")

    out = {"lang": lang, "min_cells": MIN_CELLS, "n_words": len(words),
           "n_scored_cells": int(len(y)), "oracle_auc": float(auc), "p_base_auc": float(auc_pb),
           "headroom": float(auc - auc_pb),
           "oracle_per_site_auc": float(np.mean(ps)) if ps else None,
           "n_sites": len(ps), "icc": float(icc), "fall_rate": float(y.mean())}
    #: THE SUFFIX MUST CARRY EVERY FLAG THAT CHANGES THE POPULATION. An earlier
    #: version encoded --verbs and not --pairs, so a pairs run silently
    #: overwrote the unrestricted file with different numbers under the same
    #: name. It was caught only because this script prints the path it wrote.
    p = os.path.join(ROOT, "meta/M01_displacement/results/k/ceiling_%s%s%s.json"
                     % (lang, "_verbs" if verbs is not None else "",
                        "_pairs" if "--pairs" in sys.argv else ""))
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
