"""Do the English and Chinese arm poles point the same way in a shared space?

    uv run python meta/M01_displacement/scripts/k_pole_bridge.py
    -> results/k/pole_bridge.json

THE CLAIM BEING TESTED, AND WHY EYEBALLING IT IS NOT ENOUGH. The zh aligned pole
reads like a translation of the en one -- 关注 focus, 寻求 seek, 准备 prepare,
进行 conduct -- and the base poles both look like deixis and bodily action. That
is a real observation and a terrible piece of evidence: I chose which words to
notice, I know what the English pole is, and near-synonyms are exactly what a
reader primed by the English list will find. This turns it into a number with a
null.

ONE ENCODER, BOTH LANGUAGES. bge-m3 is multilingual, so en and zh vectors live in
one space. P forbids cross-ENCODER claims (GloVe is English-only); this is
cross-LANGUAGE inside a single encoder, which is the comparison bge exists for.
The two npz files are checked to carry the same `model` string before anything is
computed -- if a future run re-embeds one language with something else, this
refuses rather than silently comparing two spaces.

**THE LANGUAGE CENTROID IS REMOVED FIRST AND THE TEST IS VACUOUS WITHOUT IT.**
Multilingual encoders separate languages far more strongly than they separate
anything within a language, so raw cosines would put every English word nearer
every other English word and the answer would be "yes" no matter what the poles
were. Each language's vectors are centred on that language's own mean, which
deletes the language direction and the bulk of the anisotropy with it.

THE NULL IS A PERMUTATION OF THE zh LABELS, not of the words. The statistic is
the DIFFERENCE of two cosines -- how much closer the zh aligned centroid sits to
the en aligned centroid than to the en base centroid -- so a null has to hold the
English side and the Chinese vocabulary fixed and destroy only the zh pole
assignment. Permuting AUC values across zh words does exactly that.

WHAT A POSITIVE RESULT WOULD AND WOULD NOT SHOW. It would show the two arm
signatures occupy corresponding regions of one semantic space, which is a
stronger claim than "the direction replicates" because it does not go through our
movement rule in either language. It would NOT show the poles are translations of
each other, and it cannot: centroid geometry is compatible with two overlapping
but distinct regions. It is also 34 zh models against 92 en, so the zh side is
the noisy one.
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
TOP = 100
NPERM = 2000
SEED = 20260813


def pole(tsv, emb_words, t2u, n=TOP):
    """(base_side, aligned_side) word lists, lowest and highest AUC, restricted
    to words the encoder actually has a vector for BEFORE the top-n is taken --
    otherwise n varies with coverage and the two poles are different sizes."""
    rows = []
    for ln in open(os.path.join(K, tsv), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 3 or p[0] == "word":
            continue
        u = t2u.get(p[0], p[0])
        if u in emb_words:
            rows.append((float(p[2]), u))
    rows.sort()
    seen, base, algn = set(), [], []
    for _, u in rows:
        if u not in seen and len(base) < n:
            seen.add(u); base.append(u)
    for _, u in reversed(rows):
        if u not in seen and len(algn) < n:
            seen.add(u); algn.append(u)
    return base, algn, rows


def main():
    ze = np.load(os.path.join(K, "embed_en_bge.npz"), allow_pickle=True)
    zz = np.load(os.path.join(K, "embed_zh_bge.npz"), allow_pickle=True)
    m_en, m_zh = str(ze["model"]), str(zz["model"])
    if m_en != m_zh:
        raise SystemExit("REFUSING: en embedded with %r, zh with %r -- not one "
                         "space, and a cosine between them means nothing"
                         % (m_en, m_zh))
    print("one encoder, both languages: %s" % m_en)

    #: centre each language on ITS OWN mean. See the module docstring: without
    #: this the test cannot fail.
    EN = {w: v for w, v in zip(ze["words"], ze["E"] - ze["E"].mean(0))}
    ZH = {w: v for w, v in zip(zz["words"], zz["E"] - zz["E"].mean(0))}

    t2u_en = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    t2u_zh = json.load(open(os.path.join(K, "normalisation_zh.json")))["token_to_unit"]
    eb, ea, _ = pole("word_auc_en.tsv", EN, t2u_en)
    zb, za, zrows = pole("word_auc_zh_nopos.tsv", ZH, t2u_zh)
    print("poles: en %d/%d, zh %d/%d  (base/aligned, embedding-covered)"
          % (len(eb), len(ea), len(zb), len(za)))

    unit = lambda M: M / max(np.linalg.norm(M), 1e-12)
    cen = lambda ws, D: unit(np.mean([D[w] for w in ws], 0))
    EA, EB = cen(ea, EN), cen(eb, EN)
    ZA, ZB = cen(za, ZH), cen(zb, ZH)

    def stat(ZA, ZB):
        return (float(ZA @ EA - ZA @ EB), float(ZB @ EB - ZB @ EA))

    d_al, d_ba = stat(ZA, ZB)
    print("\n  cos(zh aligned, en aligned) %+.4f   cos(zh aligned, en base) %+.4f"
          % (ZA @ EA, ZA @ EB))
    print("  cos(zh base,    en base)    %+.4f   cos(zh base,    en aligned) %+.4f"
          % (ZB @ EB, ZB @ EA))
    print("  contrast: aligned %+.4f | base %+.4f  (positive = poles correspond)"
          % (d_al, d_ba))

    #: permute the zh POLE ASSIGNMENT, holding English and the zh vocabulary fixed
    rng = np.random.default_rng(SEED)
    words = [u for _, u in zrows]
    uniq = list(dict.fromkeys(words))
    n = min(TOP, len(uniq) // 2)
    na = nb = 0
    for _ in range(NPERM):
        p = rng.permutation(len(uniq))
        pa = [uniq[i] for i in p[:n]]; pb = [uniq[i] for i in p[n:2 * n]]
        a, b = stat(cen(pa, ZH), cen(pb, ZH))
        na += a >= d_al; nb += b >= d_ba
    p_al = (na + 1) / (NPERM + 1); p_ba = (nb + 1) / (NPERM + 1)
    print("  permutation null, %d draws: aligned p=%.4f | base p=%.4f"
          % (NPERM, p_al, p_ba))

    #: A SECOND NULL, BECAUSE THE FIRST ONE IS TOO EASY TO BEAT. A uniform
    #: permutation builds semantically INCOHERENT zh sets, whose centroids
    #: partly cancel, so the observed contrast could be measuring coherence
    #: rather than correspondence. Measured centroid norms: aligned 0.1237, base
    #: 0.1625, random-100 0.1015 -- less of a gap than feared, and note the BASE
    #: pole is the more coherent of the two while having the WEAKER contrast,
    #: which already argues against the coherence reading.
    #:
    #: This null draws COHERENT sets not selected by arm: a random seed word and
    #: its 100 nearest zh neighbours. It also absorbs the register objection --
    #: some seeds land in institutional-procedural vocabulary, so if that region
    #: simply translates well, these draws will show it.
    ZW = [u for u in dict.fromkeys(words)]
    M = np.stack([ZH[w] for w in ZW])
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    ca = cb = 0
    NC = 500
    best = (-9, None)
    for _ in range(NC):
        i = int(rng.integers(len(ZW)))
        nb_ = np.argsort(-(M @ M[i]))[:TOP]
        S = cen([ZW[j] for j in nb_], ZH)
        a = float(S @ EA - S @ EB)
        ca += a >= d_al; cb += a >= d_ba
        if a > best[0]:
            best = (a, ZW[i])
    p_ca = (ca + 1) / (NC + 1); p_cb = (cb + 1) / (NC + 1)
    print("  COHERENT null (%d seed+100NN sets, not arm-selected):" % NC)
    print("    aligned contrast %+.4f beaten by %.1f%% of them (p=%.4f)"
          % (d_al, 100 * ca / NC, p_ca))
    print("    base    contrast %+.4f beaten by %.1f%% of them (p=%.4f)"
          % (d_ba, 100 * cb / NC, p_cb))
    print("    strongest coherent draw %+.4f, seeded on %r" % best)

    rep = {"encoder": m_en, "top_n": TOP, "n_perm": NPERM,
           "cos": {"zhA_enA": float(ZA @ EA), "zhA_enB": float(ZA @ EB),
                   "zhB_enB": float(ZB @ EB), "zhB_enA": float(ZB @ EA)},
           "contrast": {"aligned": d_al, "base": d_ba},
           "p": {"aligned": p_al, "base": p_ba},
           "coherent_null": {"n": NC, "p_aligned": p_ca, "p_base": p_cb,
                             "strongest_draw": best[0], "strongest_seed": best[1]},
           "poles": {"en_base": eb[:25], "en_aligned": ea[:25],
                     "zh_base": zb[:25], "zh_aligned": za[:25]}}
    out = os.path.join(K, "pole_bridge.json")
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1)
    print("\n  -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
