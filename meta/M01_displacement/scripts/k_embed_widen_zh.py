"""Widen the zh bge store to the full arm-AUC vocabulary, for the field test.

    uv run python meta/M01_displacement/scripts/k_embed_widen_zh.py
    -> results/k/embed_zh_bge_wide.npz    (NEW file; the original is untouched)

WHY. The zh field test's shared vocabulary was 423 words, and the constraint is
OVER-DETERMINED: `embed_zh_bge` holds only coder-rated verbs, and the delta
store is movement-verbs by construction, so armAUC's 1,212 words are cut to the
same rated-verb sublanguage by two routes at once (armAUC∩bge = armAUC∩delta =
423 exactly). Widening the embedding alone buys nothing while the delta stays in
the intersection; the widened field run therefore uses TWO instruments (armAUC +
axis/bge) over ~1,212 words, declared, instead of three over 423.

SAME ENCODER, SAME SPACE, SAME GATE. Words are encoded bare with the same
bge-m3 the axis was fitted in, so projecting them onto `axis_zh_bge.json` is a
same-space operation. The synonym-sanity gate from `k_embed` is recomputed on
this store and the script REFUSES TO WRITE if the gap is not positive -- the
gate is about the encoder-on-bare-words, which passed at 0.319, but a gate that
is inherited rather than re-run is a gate that cannot fire.

BARE-WORD bge IS STILL THE WEAK INSTRUMENT (en synonym gap 0.138 vs GloVe
0.400; zh is better at 0.319). This store exists for the FIELD test, where the
unit is a field mean over >=8 words and the size-matched null absorbs
word-level noise -- not for fine per-word geometry.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_embed as KE

K = os.path.join(ROOT, "meta/M01_displacement/results/k")


def main():
    from sentence_transformers import SentenceTransformer

    words = set()
    for ln in open(os.path.join(K, "word_auc_zh_nopos.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2 and p[0] != "word":
            words.add(p[0])
    old = np.load(os.path.join(K, "embed_zh_bge.npz"), allow_pickle=True)
    model = str(old["model"])
    have = {w: v for w, v in zip(old["words"], old["E"])}
    new_words = sorted(w for w in words if w not in have)
    print("armAUC vocabulary %d | already embedded %d | to encode %d"
          % (len(words), len(words) - len(new_words), len(new_words)))

    m = SentenceTransformer(model)
    E_new = m.encode(new_words, normalize_embeddings=True, batch_size=64,
                     show_progress_bar=False)
    allw = sorted(set(have) | set(new_words))
    lut = {**have, **{w: v for w, v in zip(new_words, E_new)}}
    E = np.stack([lut[w] for w in allw]).astype(np.float32)

    #: the gate, RE-RUN on this store rather than inherited
    pv = {w: lut[w] for w in {x for pr in KE.SYN_ZH + KE.UNREL_ZH for x in pr}
          if w in lut}
    missing = [w for pr in KE.SYN_ZH + KE.UNREL_ZH for w in pr if w not in lut]
    if missing:
        extra = m.encode(sorted(set(missing)), normalize_embeddings=True,
                         show_progress_bar=False)
        pv.update({w: v for w, v in zip(sorted(set(missing)), extra)})
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    s = float(np.median([cos(pv[a], pv[b]) for a, b in KE.SYN_ZH]))
    u = float(np.median([cos(pv[a], pv[b]) for a, b in KE.UNREL_ZH]))
    print("GATE synonym sanity: syn %.4f  unrel %.4f  gap %.4f" % (s, u, s - u))
    if s - u <= 0:
        print("REFUSING TO WRITE: the gate did not pass on this store.")
        return 1

    out = os.path.join(K, "embed_zh_bge_wide.npz")
    np.savez_compressed(out, words=np.array(allw, dtype=object), E=E,
                        syn_median=s, unrel_median=u, syn_gap=s - u,
                        anisotropy=old["anisotropy"], model=model)
    print("-> %s  (%d words; original store untouched)"
          % (os.path.relpath(out, ROOT), len(allw)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
