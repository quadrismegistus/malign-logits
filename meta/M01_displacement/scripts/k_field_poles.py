"""Do named SEMANTIC FIELDS sit at the extremes of our axes more than chance?

    uv run python meta/M01_displacement/scripts/k_field_poles.py en
    -> results/k/field_poles_<lang>.json

THE NAMING QUESTION ASKED THROUGH TAXONOMIES INSTEAD OF RATING SCALES. P has
tried eighteen rated norms and they do not predict; every one is a single
continuous scale a coder assigns to a word. A semantic field is a different kind
of name -- a discrete class built by lexicographers for reasons that have nothing
to do with this study -- and there are hundreds of them rather than eighteen. If
`bodily action` or `violence` sits reliably at one pole, that IS a name, supplied
by an external taxonomy and not by us reading our own word list.

DISTINCT FROM `v_axis_vs_fields.csv`, which correlates a per-LEXICON quantity
with the axis across five lexicons. This asks a per-FIELD question: does THIS
class of words sit further along the axis than a random set of the same size?

THE NULL IS A RANDOM WORD SET OF MATCHED SIZE, drawn from the same shared
vocabulary. That matters more than it sounds: fields vary from three members to
several hundred, and a small field's mean is far more extreme by chance than a
large one's. Comparing raw means across fields of different sizes would rank the
small ones top every time, which is the shape of error this campaign keeps
paying for. Everything is reported as a z against its own size-matched null.

ALL FOUR INSTRUMENTS, ONE ORIENTATION -- fall/base is HIGH throughout, the arm
AUC negated to match, as in `k_instrument_poles`. A field that is genuinely a
name for the direction should land at the same pole on all four; one that only
does so on a single instrument is measuring that instrument.

MULTIPLICITY IS REAL AND REPORTED. Hundreds of fields tested against several
instruments will produce extremes by chance alone, so the count expected at each
threshold is printed beside the count observed, and a Benjamini-Hochberg q is
given. A field list without that is a fishing report.
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
MINN = 8
NPERM = 4000
SEED = 20260813


def tsv(path, col=2):
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > col and p[0] != "word":
            try:
                d.setdefault(p[0], float(p[col]))
            except ValueError:
                pass
    return d


def axis_scores(axis_file, emb_file):
    if not (os.path.exists(os.path.join(K, axis_file))
            and os.path.exists(os.path.join(K, emb_file))):
        return {}
    ax = np.array(json.load(open(os.path.join(K, axis_file)))["axis"], np.float32)
    ax /= max(np.linalg.norm(ax), 1e-12)
    z = np.load(os.path.join(K, emb_file), allow_pickle=True)
    E = z["E"].astype(np.float32)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return {w: float(v) for w, v in zip(z["words"], E @ ax)}


def main(lang="en"):
    from malign_logits import fields as FL

    inst = {}
    #: zh quotes the NO-POS arm table per P section 5b -- the tagged table
    #: defends its tag with half the vocabulary
    #: optional arm-table variant (e.g. `zhdom` for the domain-matched en
    #: table); output file carries the same suffix so nothing is overwritten
    VAR = (sys.argv[sys.argv.index("--arm-variant") + 1]
           if "--arm-variant" in sys.argv else None)
    au = tsv(os.path.join(K, "word_auc_%s%s.tsv"
                          % ((lang if lang == "en" else lang + "_nopos"),
                             "_" + VAR if VAR else "")))
    if au:
        inst["armAUC"] = {w: -v for w, v in au.items()}
    inst["axisGloVe"] = axis_scores("axis_%s.json" % lang,
                                    "embed_%s_glove.npz" % lang)
    #: --wide (zh): the widened bge store over the full armAUC vocabulary, and
    #: the DELTA IS DROPPED -- it is movement-verbs by construction, so keeping
    #: it in the intersection would cut the vocabulary straight back to the 423
    #: rated verbs the widening exists to escape. Two instruments, declared.
    WIDE = "--wide" in sys.argv
    inst["axisBGE"] = axis_scores("axis_%s_bge.json" % lang,
                                  "embed_%s_bge%s.npz" % (lang, "_wide" if WIDE else ""))
    if not WIDE:
        inst["delta"] = tsv(os.path.join(K, "delta_word_scores_%s.tsv" % lang), col=1)
    inst = {k: v for k, v in inst.items() if v}
    shared = sorted(set.intersection(*[set(v) for v in inst.values()]))
    print("[%s] instruments %s | shared vocabulary %s words"
          % (lang, ", ".join(inst), format(len(shared), ",")))

    #: word -> set of field labels. English: the four lemma-keyed taxonomies via
    #: _lookup's BYU-lemma peel. Chinese: the zh USAS port -- SAME tagset, which
    #: is its reason to exist per fields.py -- looked up directly, because the
    #: peel is English morphology. Primary tag both sides, matching what
    #: _lookup's return shape effectively gave English. zh coverage is thin on
    #: compounds and our units are tokenizer boundaries, so coverage is MEASURED
    #: and printed: absence must not read as a semantic fact.
    #: ALL TAGS, BOTH LANGUAGES, as of the zh run. The first en run used the
    #: primary tag only -- not by decision but by a shape accident: _usas()
    #: values are (primary, tuple-of-all) and the iteration collected the one
    #: bare string and skipped the tuple. In zh the first-entry-wins primary is
    #: demonstrably noisy (the port files "run" under N5, quantities), and
    #: all-tags takes zh from 10 usable fields to 73 on the same vocabulary. A
    #: word now votes in every field the lexicon files it under, which is what
    #: "does this FIELD sit at the pole" means; the en primary-only result is
    #: superseded by the all-tags rerun, not deleted.
    def flat(v):
        if isinstance(v, str):
            yield v
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                yield from flat(x)
    memb = collections.defaultdict(set)
    if lang == "en":
        for src, table in (("usas", FL._usas()), ("gi", FL._gi()),
                           ("wordnet", FL._wordnet()), ("rid", FL._rid())):
            try:
                for w in shared:
                    v = FL._lookup(w, table)
                    for t in flat(v) if v else ():
                        if t:
                            memb["%s:%s" % (src, t)].add(w)
            except Exception as e:
                print("  %s unavailable: %s" % (src, str(e)[:70]))
    else:
        hit = 0
        for w in shared:
            ta = FL.usas_zh(w, all_tags=True)
            if ta:
                hit += 1
                for t in ta:
                    memb["usas:%s" % t].add(w)
        print("  usas_zh coverage: %d/%d shared words (%.1f%%)"
              % (hit, len(shared), 100 * hit / max(len(shared), 1)))
    fields = {f: sorted(ws) for f, ws in memb.items() if len(ws) >= MINN}
    print("  %s fields with >=%d members" % (format(len(fields), ","), MINN))

    rng = np.random.default_rng(SEED)
    out, rowsz = {}, []
    for iname, sc in inst.items():
        vals = np.array([sc[w] for w in shared])
        idx = {w: i for i, w in enumerate(shared)}
        #: one null distribution per SIZE, reused across fields of that size
        sizes = sorted({len(ws) for ws in fields.values()})
        null = {}
        for n in sizes:
            draws = np.array([vals[rng.choice(len(vals), n, replace=False)].mean()
                              for _ in range(NPERM)])
            null[n] = (draws.mean(), draws.std() + 1e-12, draws)
        for f, ws in fields.items():
            m = float(vals[[idx[w] for w in ws]].mean())
            mu, sd, draws = null[len(ws)]
            zsc = (m - mu) / sd
            p = (min((draws >= m).sum(), (draws <= m).sum()) + 1) / (NPERM + 1) * 2
            rowsz.append({"instrument": iname, "field": f, "n": len(ws),
                          "mean": m, "z": float(zsc), "p": float(min(p, 1.0))})
    #: Benjamini-Hochberg over everything tested
    ps = np.array([r["p"] for r in rowsz])
    o = np.argsort(ps)
    q = np.empty_like(ps)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, ps[i] * len(ps) / (len(ps) - rank))
        q[i] = prev
    for r, qq in zip(rowsz, q):
        r["q"] = float(qq)

    n_sig = sum(1 for r in rowsz if r["q"] < 0.05)
    print("\n  %s field x instrument tests | %d at q<0.05 | %.0f expected at p<0.05 by chance"
          % (format(len(rowsz), ","), n_sig, 0.05 * len(rowsz)))

    #: a field only counts as a NAME if it lands the same way on all instruments
    byf = collections.defaultdict(dict)
    for r in rowsz:
        byf[r["field"]][r["instrument"]] = r
    names = list(inst)
    cons = []
    for f, d in byf.items():
        if len(d) < len(names):
            continue
        zs = [d[i]["z"] for i in names]
        if all(x > 0 for x in zs) or all(x < 0 for x in zs):
            cons.append((float(np.mean(zs)), f, d))
    cons.sort()
    print("  %d fields agree in SIGN across all %d instruments" % (len(cons), len(names)))

    def show(rows, lab):
        print("\n  %s" % lab)
        print("    %-42s %5s %7s  %s" % ("field", "n", "mean z", "  ".join(
            "%-9s" % i for i in names)))
        for mz, f, d in rows:
            print("    %-42s %5d %+7.2f  %s"
                  % (f[:42], d[names[0]]["n"], mz,
                     "  ".join("%+9.2f" % d[i]["z"] for i in names)))
    show(cons[::-1][:14], "MOST FALL / BASE-SIDE fields (consistent on all instruments)")
    show(cons[:14], "MOST RISE / ALIGNED-SIDE fields")

    out = {"lang": lang, "instruments": names, "n_fields": len(fields),
           "n_tests": len(rowsz), "n_q05": int(n_sig),
           "n_sign_consistent": len(cons),
           "tests": sorted(rowsz, key=lambda r: r["z"]),
           "consistent": [{"field": f, "mean_z": mz,
                           "n": d[names[0]]["n"],
                           "z": {i: d[i]["z"] for i in names},
                           "q": {i: d[i]["q"] for i in names}} for mz, f, d in cons]}
    p = os.path.join(K, "field_poles_%s%s%s.json"
                     % (lang, "_" + VAR if VAR else "", "_wide" if WIDE else ""))
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
