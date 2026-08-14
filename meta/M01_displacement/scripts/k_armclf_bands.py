"""Where in the lexicon does the arm signature live? Frequency x POS bands.

    uv run python meta/M01_displacement/scripts/k_armclf_bands.py en 2>&1 | tee log
    ... --max-rank 3000   how deep to go within each POS (default 2000)
    ... --band 50         words per band

Section 6 of P shows a classifier tells base from aligned at AUC 0.959 from word
probabilities. This asks WHERE: rank each part of speech by corpus mass, walk
down it in bands of 50, and score each band on its own. A band that classifies
carries the signature; a band at 0.5 does not.

RESUMABLE BY DESIGN. One JSON line per (pos, band) appended to
`results/k/armclf_bands_<lang>.jsonl` and flushed immediately; on start the file
is read and finished bands are skipped. A sweep that dies at band 90 of 160
resumes at 90, and re-running after adding a POS costs only the new bands.

BANDS ARE EQUAL-SIZED AND THAT IS THE POINT. Comparing a 300-word band against a
50-word one measures capacity, not signal -- the same trap as comparing the
4,075-word and 1,042-word populations in P section 9. Every band here is the same
width, so the AUCs are comparable to each other and the only thing varying is
which part of the lexicon they cover.

THE UNIT IS THE MODEL. 92 models in 33 org groups, leave-one-org-out. Each value
is the FRACTION of that model's prompts where the word was in the cell's top-N by
rank -- so total mass, top-1 mass and support size are constant by construction
and the sharpness confound cannot enter. Holding out the ORG rather than the
lineage matters: 21 of 46 lineages share an org, so a lineage holdout lets the
model recognise tiiuae instead of the arm.

COVERAGE IS RECORDED PER BAND because it decides whether a null is a finding. A
band of rank-1800 nouns that appears in almost no cell's top-N will score 0.5
whatever alignment does, and that is an absence of measurement rather than an
absence of effect. `mean_hits` is how many of the band's 50 words land in an
average cell's top-N.
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
#: --source LITERARY restricts to the 102 novel-snippet prompts. IT MATTERS FOR
#: THE POS COMPARISON AND NOT FOR MUCH ELSE. Measured over the top-20 slots:
#:
#:                    LITERARY        M01_PAIRS (the designed minimal pairs)
#:     verb          2.98 of 20        11.88 of 20
#:     noun          4.06              0.37
#:     adjective     1.18              0.13
#:     adverb        1.62              1.42
#:
#: The designed pairs are 59% verbs and 2% nouns, and they outnumber the literary
#: cells 14:1, so a POS comparison on the full corpus is reading prompt design as
#: much as lexical structure. In particular an adverb band at rank 500+ scores
#: 0.500 against a 0.500 null because adverbs are never in contention there --
#: an absence of MEASUREMENT, not of effect.
#:
#: THE COST IS POWER: 97 prompts per model against 2,220, so each per-model rate
#: is built from a twenty-third of the observations and a null here cannot
#: distinguish absence from noise.
TOPN = 20            #: the rank cutoff defining "in the running"
POS = {"vv": "verb", "nn": "noun", "jj": "adjective", "rr": "adverb"}


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main(lang="en"):
    from malign_logits import fields as FL
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    MAXR, BAND = arg("--max-rank", 2000), arg("--band", 50)
    SRC = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else None
    out_path = os.path.join(K, "armclf_bands_%s%s.jsonl"
                            % (lang, "_" + SRC.lower() if SRC else ""))
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try:
                d = json.loads(ln)
                done.add((d["pos"], d["rank_lo"]))
            except Exception:
                pass
        print("resuming: %d bands already in %s" % (len(done), os.path.basename(out_path)))

    pairs = KP.reps(lang)
    arm, org = {}, {}
    for b, a in pairs:
        arm[b] = 0; arm[a] = 1
        o = b.split("/")[0]; org[b] = o; org[a] = o
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    models = "','".join(esc(m) for m in arm)
    t2u = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))["token_to_unit"]

    #: RANK EACH POS SEPARATELY BY POOLED MASS, label-blind. No GloVe filter here
    #: -- that requirement belongs to the axis projection, not to classification,
    #: and imposing it costs coverage for nothing. P section 6 pays 61% coverage
    #: for exactly that mistake.
    print("ranking vocabulary by pooled mass within each POS ...", flush=True)
    top = A.q("""
      SELECT word, sum(p) mass FROM %s.twp_words WHERE model IN ('%s') AND prompt IN (
        SELECT DISTINCT prompt FROM %s.prompt_catalogue
        WHERE status='ACTIVE' AND language='%s'%s)
      GROUP BY word ORDER BY mass DESC LIMIT 40000"""
              % (A.DB, models, A.DB, lang,
                 " AND source='%s'" % SRC if SRC else ""))
    byu = FL._byu()
    bypos = collections.defaultdict(list)
    for r in top:
        u = t2u.get(r["word"], r["word"])
        e = byu.get(u.strip().lower())
        if not e:
            continue
        tag = e[1][:2]
        if tag in POS and len(bypos[tag]) < MAXR:
            bypos[tag].append(r["word"])
    for t in POS:
        print("  %-10s %5d words ranked" % (POS[t], len(bypos.get(t, []))))
    allw = [w for t in POS for w in bypos.get(t, [])]
    ci = {w: i for i, w in enumerate(allw)}

    #: ACCUMULATE STRAIGHT INTO PER-MODEL COUNTS. The cell-level matrix would be
    #: 204,438 wide and is never needed: only the per-model fractions are, which is 92 x V floats.
    print("streaming %d candidate words into per-model counts ..." % len(allw), flush=True)
    rows = A.q("""
      SELECT model, prompt, groupArray(word) ws, groupArray(p) ps
      FROM %s.twp_words WHERE model IN ('%s') AND prompt IN (
        SELECT DISTINCT prompt FROM %s.prompt_catalogue
        WHERE status='ACTIVE' AND language='%s'%s)
      GROUP BY model, prompt"""
              % (A.DB, models, A.DB, lang, " AND source='%s'" % SRC if SRC else ""))
    mods = sorted(arm)
    mi = {m: i for i, m in enumerate(mods)}
    C = np.zeros((len(mods), len(allw)), np.float32)
    ncell = np.zeros(len(mods))
    for r in rows:
        ps = np.array([p if p else 0.0 for p in r["ps"]], float)
        i = mi[r["model"]]
        ncell[i] += 1
        for j in np.argsort(-ps)[:TOPN]:
            c = ci.get(r["ws"][j])
            if c is not None:
                C[i, c] += 1
    hits = C.sum(1) / np.maximum(ncell, 1)
    C = C / np.maximum(ncell, 1)[:, None]

    #: COMPOSITIONAL NORMALISATION, AND WITHOUT IT EVERY BAND IS CONTAMINATED.
    #: Fixing the top-N at 20 makes each CELL contribute exactly 20 slots, but
    #: the SHARE of those 20 that lands inside the candidate set still varies by
    #: model -- base 13.656, aligned 13.923 of 20 -- and that one number
    #: separates the arms at AUC 0.750 on its own. A model with a higher global
    #: candidate rate has a higher rate for EVERY word in the set, so each band
    #: inherits the 0.75 whether or not its own words carry anything. It is how
    #: a band of rank-150 nouns with mean_hits 0.08 scored 0.949.
    #:
    #: Dividing each model's row by its own total candidate hits makes the row a
    #: COMPOSITION: of the in-candidate slots this model has, what share goes to
    #: this word. The scale factor is then removed by construction rather than
    #: controlled for, which is what made the binary design work in the first
    #: place. The pre-fix sweep is kept as armclf_bands_en.CONTAMINATED.jsonl.
    RAW = C.copy()          #: pre-normalisation counts, kept for the coverage column
    scale = C.sum(1, keepdims=True)
    print("  per-model candidate share, base %.3f vs aligned %.3f of top-%d --"
          % (float((C.sum(1)[np.array([arm[m] for m in mods]) == 0]).mean()),
             float((C.sum(1)[np.array([arm[m] for m in mods]) == 1]).mean()), TOPN))
    print("  that scale factor alone separates the arms at AUC 0.750, so rows are")
    print("  normalised to compositions before any band is scored.")
    C = C / np.maximum(scale, 1e-12)
    cy = np.array([arm[m] for m in mods])
    cg = np.array([org[m] for m in mods], dtype=object)
    print("  %s cells | %d models | %d org groups | mean %.1f of top-%d inside "
          "the candidate set" % (f"{len(rows):,}", len(mods), len(set(cg)),
                                 hits.mean(), TOPN))

    rng = np.random.default_rng(20260812)
    linof = {m: i for i, (b, a) in enumerate(pairs) for m in (b, a)}
    flip = {i: rng.integers(0, 2) for i in set(linof.values())}
    ynull = np.array([cy[i] ^ flip[linof[m]] for i, m in enumerate(mods)])

    jobs = [(t, lo) for t in POS if bypos.get(t)
            for lo in range(0, len(bypos[t]), BAND)
            if (t, lo) not in done]
    print("%d bands to run (%d already done)\n" % (len(jobs), len(done)), flush=True)

    fh = open(out_path, "a")
    logo = LeaveOneGroupOut()
    for tag, lo in tqdm(jobs, desc="bands", unit="band"):
        words = bypos[tag][lo:lo + BAND]
        idx = [ci[w] for w in words]
        M = C[:, idx]
        #: COVERAGE IS THE RAW COUNT, NOT THE NORMALISED SHARE. An earlier
        #: version read this off the normalised matrix, so the column labelled
        #: "hits" was reporting the band's share of a model's candidate slots
        #: and the coverage filter was selecting on the wrong quantity.
        mean_hits = float(RAW[:, idx].sum(1).mean())
        mean_share = float(M.sum(1).mean())
        rec = {"pos": tag, "pos_name": POS[tag], "rank_lo": lo,
               "rank_hi": lo + len(words), "n_words": len(words),
               "mean_hits": mean_hits, "mean_share": mean_share, "top_n": TOPN}
        #: a band nobody's top-N ever reaches cannot be scored, and reporting
        #: 0.5 for it would read as "no effect here" rather than "no measurement"
        if mean_hits < 1e-6 or M.std() == 0:
            rec.update({"auc": None, "acc": None, "skipped": "no coverage"})
        else:
            pred = np.zeros(len(cy))
            cf = []
            for tr, te in logo.split(M, cy, groups=cg):
                sc = StandardScaler().fit(M[tr])
                m = LogisticRegression(max_iter=4000, C=0.1).fit(sc.transform(M[tr]),
                                                                 cy[tr])
                pred[te] = m.predict_proba(sc.transform(M[te]))[:, 1]
                cf.append(m.coef_[0])
            cf = np.array(cf)
            U = cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-12)
            st = [float(U[i] @ U[j]) for i in range(len(U)) for j in range(i + 1, len(U))]
            #: A PER-BAND NULL, because a global one cannot say what THIS band
            #: should score. Arms are flipped within a random half of lineages,
            #: which leaves every model and every value where it is and destroys
            #: only the direction. A band whose real AUC is not clear of its own
            #: null has not shown anything, however high it looks.
            pn = np.zeros(len(cy))
            for tr, te in logo.split(M, ynull, groups=cg):
                sc = StandardScaler().fit(M[tr])
                pn[te] = LogisticRegression(max_iter=4000, C=0.1).fit(
                    sc.transform(M[tr]), ynull[tr]).predict_proba(sc.transform(M[te]))[:, 1]
            null_auc = float(roc_auc_score(ynull, pn))
            acc = max(accuracy_score(cy, (pred > t).astype(int)) for t in np.unique(pred))
            w = cf.mean(0)
            o = np.argsort(w)
            rec.update({
                "auc": float(roc_auc_score(cy, pred)), "acc": float(acc),
                "null_auc": null_auc,
                "n_wrong": int(round((1 - acc) * len(cy))),
                "stability_min_cos": float(min(st)) if st else 1.0,
                "coef": {words[i]: round(float(w[i]), 4) for i in range(len(words))},
                "top_aligned": [words[i] for i in o[-5:][::-1]],
                "top_base": [words[i] for i in o[:5]]})
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    fh.close()

    #: the summary reads the whole file, so it is the same whether this run did
    #: one band or all of them
    recs = [json.loads(l) for l in open(out_path)]
    print("\n%-10s %-12s %7s %8s %8s %9s %8s"
          % ("POS", "ranks", "hits", "AUC", "acc", "stability", "NULL"))
    for t in POS:
        rr = sorted((r for r in recs if r["pos"] == t), key=lambda r: r["rank_lo"])
        for r in rr:
            if r.get("auc") is None:
                print("%-10s %-12s %7.2f      -- no coverage --" %
                      (POS[t], "%d-%d" % (r["rank_lo"], r["rank_hi"]), r["mean_hits"]))
            else:
                print("%-10s %-12s %7.2f %8.4f %7.1f%% %9.3f %8.4f"
                      % (POS[t], "%d-%d" % (r["rank_lo"], r["rank_hi"]),
                         r["mean_hits"], r["auc"], 100 * r["acc"],
                         r["stability_min_cos"], r.get("null_auc", float("nan"))))
    print("\n  -> %s" % os.path.relpath(out_path, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
                  else "en"))
