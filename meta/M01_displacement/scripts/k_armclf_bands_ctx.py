"""Where the arm signature lives, by frequency x IN-CONTEXT part of speech.

    uv run python meta/M01_displacement/scripts/k_armclf_bands_ctx.py en
    ... --max-rank 5000   depth within each POS (default 5000)
    ... --band 50 --source LITERARY

SUPERSEDES `k_armclf_bands.py`, whose POS came from `fields._byu()` -- the most
frequent reading of a word FORM, out of context. Measured over all 365,892
(prompt, word) pairs, BYU's "noun" label is 41.2% verbs in context and its
"adjective" label only 40.5% adjectives; verb and adverb hold at 97.3% and 94.8%.
So that script's noun and adjective bands were mixed populations. Its "noun" band
0-50 was `fall break kiss punch strike stroke touch change work sign dance tear
love` -- verbs at sites like "She began to ___".

**THE FEATURE IS A (WORD, IN-CONTEXT TAG) PAIR, NOT A WORD.** `kiss` is a verb
after "She began to" and a noun after "He gave her a", so a word has no single
POS and assigning one by majority would rebuild a softer version of the same
defect. `kiss`-as-verb and `kiss`-as-noun are separate columns counted from the
same cells, which makes each band a population of USAGES rather than of forms.
1,494 of 21,871 word types have no tag holding 90% of their occurrences.

Tags come from `results/k/pos_context_en.tsv`, built by `k_pos_context.py`:
spaCy en_core_web_sm, the tag of the LAST token of prompt+word, keyed by sha16 of
the prompt TEXT because prompt ids do not travel.

EVERYTHING ELSE IS AS THE SUPERSEDED SCRIPT, INCLUDING TWO FIXES IT NEEDED:

  - COMPOSITIONAL ROWS. Fixing the top-N at 20 makes each CELL contribute 20
    slots, but the SHARE of those landing inside the candidate set varies by
    model (base 13.656, aligned 13.923 of 20) and separates the arms at AUC
    0.750 on its own. Every band inherits it unless rows are normalised to sum
    to 1, which removes it by construction.
  - A PER-BAND NULL. A global null cannot say what a given band should score.
    Each band runs its own within-lineage arm flip, and coverage is read from
    the PRE-normalisation matrix -- in the superseded script the normalisation
    was inserted above the line computing coverage, silently redefining the one
    column that had caught the confound in the first place.

DEPTH IS BOUNDED BY COVERAGE, NOT BY CHOICE. Words at rank 5000+ appear in about
30 of 204,438 cells, so most models see none and the band has no variance; those
are written with `auc: null` and a reason rather than 0.5, because an absence of
measurement must not read as an absence of effect.

MORE BANDS BUY TESTS, NOT POWER. Every band is 50 features against 92 models
under leave-one-org-out however the vocabulary is sliced, so with several hundred
bands the honest reading is the DISTRIBUTION of real-minus-null gaps rather than
any single row.
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
KEEP = ("verb", "noun", "adjective", "adverb")


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


def main(lang="en"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    MAXR, BAND = arg("--max-rank", 5000), arg("--band", 50)
    SRC = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else None
    out = os.path.join(K, "armclf_bandsctx_%s%s.jsonl"
                       % (lang, "_" + SRC.lower() if SRC else ""))
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            try:
                d = json.loads(ln); done.add((d["pos"], d["rank_lo"]))
            except Exception:
                pass
        print("resuming: %d bands already recorded" % len(done))

    tagf = os.path.join(K, "pos_context_%s.tsv" % lang)
    if not os.path.exists(tagf):
        print("no POS cache at %s -- run k_pos_context.py first" % tagf); return 1
    TAG = {}
    for ln in open(tagf, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 4:
            TAG[(p[0], p[1])] = p[3]
    print("POS cache: %s (prompt, word) pairs" % f"{len(TAG):,}")

    pairs = KP.reps(lang)
    arm, org, lin = {}, {}, {}
    for i, (b, a) in enumerate(pairs):
        arm[b] = 0; arm[a] = 1
        o = b.split("/")[0]; org[b] = o; org[a] = o
        lin[b] = i; lin[a] = i
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    models = "','".join(esc(m) for m in arm)
    sha16 = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    rows = A.q("""
      SELECT model, prompt, groupArray(word) ws, groupArray(p) ps
      FROM %s.twp_words WHERE model IN ('%s') AND prompt IN (
        SELECT DISTINCT prompt FROM %s.prompt_catalogue
        WHERE status='ACTIVE' AND language='%s'%s)
      GROUP BY model, prompt""" % (A.DB, models, A.DB, lang,
                                   " AND source='%s'" % SRC if SRC else ""))
    mods = sorted(arm)
    mi = {m: i for i, m in enumerate(mods)}
    cnt = collections.defaultdict(lambda: np.zeros(len(mods), np.float32))
    ncell = np.zeros(len(mods))
    untagged = 0
    for r in rows:
        ps = np.array([p if p else 0.0 for p in r["ps"]], float)
        i = mi[r["model"]]; ncell[i] += 1
        h = sha16(r["prompt"])
        for j in np.argsort(-ps)[:TOPN]:
            w = r["ws"][j]
            t = TAG.get((h, w))
            if t is None:
                untagged += 1; continue
            if t in KEEP:
                cnt[(w, t)][i] += 1
    print("  %s cells | %d models | %s (word,tag) features | %s top-20 slots untagged"
          % (f"{len(rows):,}", len(mods), f"{len(cnt):,}", f"{untagged:,}"))

    feats = sorted(cnt, key=lambda k: -cnt[k].sum())
    RAW = np.stack([cnt[f] for f in feats], 1) / np.maximum(ncell, 1)[:, None]
    C = RAW / np.maximum(RAW.sum(1, keepdims=True), 1e-12)
    cy = np.array([arm[m] for m in mods])
    cg = np.array([org[m] for m in mods], dtype=object)
    rng = np.random.default_rng(20260812)
    flip = {i: rng.integers(0, 2) for i in set(lin.values())}
    ynull = np.array([cy[i] ^ flip[lin[m]] for i, m in enumerate(mods)])
    fi = {f: i for i, f in enumerate(feats)}
    bypos = collections.defaultdict(list)
    for f in feats:
        if len(bypos[f[1]]) < MAXR:
            bypos[f[1]].append(f)
    for t in KEEP:
        print("  %-11s %5d (word,tag) features ranked" % (t, len(bypos.get(t, []))))

    jobs = [(t, lo) for t in KEEP if bypos.get(t)
            for lo in range(0, len(bypos[t]), BAND) if (t, lo) not in done]
    print("%d bands to run\n" % len(jobs), flush=True)
    fh = open(out, "a")
    logo = LeaveOneGroupOut()
    for tag, lo in tqdm(jobs, desc="bands", unit="band"):
        fs = bypos[tag][lo:lo + BAND]
        idx = [fi[f] for f in fs]
        M = C[:, idx]
        rec = {"pos": tag, "rank_lo": lo, "rank_hi": lo + len(fs), "n_feats": len(fs),
               "mean_hits": float(RAW[:, idx].sum(1).mean()), "top_n": TOPN,
               "source": SRC or "ALL"}
        if rec["mean_hits"] < 1e-9 or M.std() == 0:
            rec.update({"auc": None, "skipped": "no coverage"})
        else:
            def fit(target):
                p = np.zeros(len(target)); cf = []
                for tr, te in logo.split(M, target, groups=cg):
                    sc = StandardScaler().fit(M[tr])
                    m = LogisticRegression(max_iter=4000, C=0.1).fit(
                        sc.transform(M[tr]), target[tr])
                    p[te] = m.predict_proba(sc.transform(M[te]))[:, 1]
                    cf.append(m.coef_[0])
                return p, np.array(cf)
            pred, cf = fit(cy)
            pn, _ = fit(ynull)
            w = cf.mean(0); o = np.argsort(w)
            acc = max(accuracy_score(cy, (pred > t).astype(int)) for t in np.unique(pred))
            rec.update({"auc": float(roc_auc_score(cy, pred)), "acc": float(acc),
                        "null_auc": float(roc_auc_score(ynull, pn)),
                        "coef": {"%s/%s" % f: round(float(w[i]), 4)
                                 for i, f in enumerate(fs)},
                        "top_aligned": ["%s/%s" % fs[i] for i in o[-5:][::-1]],
                        "top_base": ["%s/%s" % fs[i] for i in o[:5]]})
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    fh.close()
    print("\n  -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
                  else "en"))
