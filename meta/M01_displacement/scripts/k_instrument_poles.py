"""One comparable top-100 per pole, for every instrument P has.

    uv run python meta/M01_displacement/scripts/k_instrument_poles.py
    -> findings/instrument_poles.md

FOUR INSTRUMENTS, TWO OUTCOMES, AND THEY ARE ONLY COMPARABLE AFTER TWO FIXES.

    per-word arm AUC     ARM identity   high = aligned-side   word_auc_en.tsv
    movement axis/GloVe  MOVEMENT       high = FALLS          axis_en.json
    movement axis/bge    MOVEMENT       high = FALLS          axis_en_bge.json
    delta projection     MOVEMENT       high = FALLS          delta_word_scores_en.tsv

**FIX 1, SIGN.** A faller is a word alignment pushes DOWN, so it is high in base
and low in aligned: fall-side and base-side are the same pole reached from two
outcomes. The arm AUC therefore runs BACKWARDS relative to the other three, which
is why P reports their agreement as NEGATIVE correlations (-0.461 axis, -0.495
delta). Everything here is flipped to one convention -- **pole A is fall/base,
pole B is rise/aligned** -- so the lists can be read down the page.

**FIX 2, POPULATION.** The four cover different vocabularies: GloVe has 6,084
verbs, the delta 4,064 words, the arm AUC 4,106 (word, tag) features collapsed to
words. Top-100 lists drawn from different vocabularies are not comparable -- a
word can be absent from one list because it is unranked or because it was never a
candidate. Every list below is drawn from the SHARED vocabulary, and the
per-instrument full-vocabulary counts are printed so the restriction is visible.

WHAT THE OVERLAP TABLE IS FOR. Correlation over the whole vocabulary answers
"do these agree on average"; the poles answer "do they agree about the extremes",
which is the question a reader of a word list is actually asking. They can come
apart: two instruments correlating at 0.5 can share most of their tails or almost
none of them.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
OUT = os.path.join(ROOT, "meta/M01_displacement/findings/instrument_poles.md")
TOP = 100


def tsv(path, col=2, keycol=0):
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > col and p[keycol] != "word":
            try:
                d.setdefault(p[keycol], float(p[col]))
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


def main():
    from scipy.stats import spearmanr

    #: value convention after this block: HIGH = fall/base pole
    inst = {}
    au = tsv(os.path.join(K, "word_auc_en.tsv"), col=2)
    inst["arm AUC (flipped)"] = {w: -v for w, v in au.items()}
    inst["axis / GloVe"] = axis_scores("axis_en.json", "embed_en_glove.npz")
    inst["axis / bge"] = axis_scores("axis_en_bge.json", "embed_en_bge.npz")
    inst["delta / bge-sub"] = tsv(os.path.join(K, "delta_word_scores_en.tsv"), col=1)
    inst = {k: v for k, v in inst.items() if v}

    print("full-vocabulary coverage")
    for k, v in inst.items():
        print("  %-20s %s words" % (k, format(len(v), ",")))
    shared = sorted(set.intersection(*[set(v) for v in inst.values()]))
    print("  %-20s %s words  <- every list below" % ("SHARED", format(len(shared), ",")))

    names = list(inst)
    print("\npairwise Spearman on the shared vocabulary (all oriented fall/base = high)")
    M = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = spearmanr([inst[a][w] for w in shared],
                          [inst[b][w] for w in shared]).statistic
            M[(a, b)] = float(r)
            print("  %-20s %-20s %+.3f" % (a, b, r))

    poles = {}
    for k in names:
        s = sorted(shared, key=lambda w: -inst[k][w])
        poles[k] = (s[:TOP], s[-TOP:][::-1])

    print("\ntop-%d OVERLAP between instruments" % TOP)
    ov = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa = len(set(poles[a][0]) & set(poles[b][0]))
            rb = len(set(poles[a][1]) & set(poles[b][1]))
            ov["%s|%s" % (a, b)] = {"fall": fa, "rise": rb}
            print("  %-20s %-20s  fall %3d/%d   rise %3d/%d" % (a, b, fa, TOP, rb, TOP))

    L = []
    W = L.append
    W("# Instrument poles, one convention\n")
    W("Every instrument in P that scores a word, oriented the same way and drawn")
    W("from the SHARED vocabulary so the lists are comparable. Produced by")
    W("`k_instrument_poles.py`.\n")
    W("**POLE A is fall / base-side. POLE B is rise / aligned-side.** A faller is")
    W("pushed down by alignment, so it is high in base and low in aligned: the")
    W("movement and arm outcomes reach the same pole from opposite directions, and")
    W("the arm AUC is negated here to match. That is why P reports their agreement")
    W("as negative correlations.\n")
    W("    instrument           full vocab   outcome")
    for k in names:
        o = "ARM identity" if "arm" in k else "MOVEMENT"
        W("    %-20s %8s   %s" % (k, format(len(inst[k]), ","), o))
    W("    %-20s %8s   <- every list below" % ("SHARED", format(len(shared), ",")))
    W("\n## Agreement\n")
    W("Spearman over the shared vocabulary, and overlap of the top-%d poles --" % TOP)
    W("two instruments can correlate and still disagree about their extremes,")
    W("which is what a reader of a word list actually cares about.\n")
    W("    pair                                      rho     fall    rise")
    for (a, b), r in M.items():
        o = ov["%s|%s" % (a, b)]
        W("    %-18s %-18s %+.3f  %3d/%d %3d/%d"
          % (a, b, r, o["fall"], TOP, o["rise"], TOP))
    for k in names:
        for lab, ws in (("POLE A -- fall / base-side", poles[k][0]),
                        ("POLE B -- rise / aligned-side", poles[k][1])):
            W("\n## %s -- %s\n" % (k, lab))
            for i in range(0, len(ws), 6):
                W("    " + "  ".join("%-14s" % w for w in ws[i:i + 6]))
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n  -> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
