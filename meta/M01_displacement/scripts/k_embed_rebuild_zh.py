"""Rebuild the zh bge store on CPU, after the mps single-character corruption.

    uv run python meta/M01_displacement/scripts/k_embed_rebuild_zh.py
    -> results/k/embed_zh_bge.npz        (canonical name, SAME population)
    -> results/k/embed_zh_bge_wide.npz   (canonical + the armAUC vocabulary)

THE DEFECT THIS REPAIRS, measured on 2026-08-13. The committed zh store held,
for 576 of its 3,978 rows (89% of the 647 single-character words, 0% of
everything else), vectors that are NOT the model's output for those strings --
median cos 0.51 against a CPU referee, worst 0.16. The build path was a large
multi-batch encode on this machine's mps (torch 2.11), which corrupts some
single-CJK-character embeddings and does so DETERMINISTICALLY, so a re-audit
down the same path reproduces the corruption and reads as clean. It surfaced
only because `k_embed_widen_zh`'s re-run gate refused to reproduce the stored
gate (0.0896 vs 0.319): an inherited gate cannot fire. The English store audits
clean against the same referee (0 of 6,120).

THE REBUILD IS CPU, WHERE alone == in-batch == referee. The population is the
ORIGINAL store's word list, unchanged, so the axis refit that follows estimates
the same quantity on the same words. The corrupted store is moved aside as
`embed_zh_bge.MPSCORRUPT.npz`, not deleted -- staleness in the filename, the
artifact kept as evidence, per the campaign's convention.

THE GATE IS RE-RUN ON THE STORE'S OWN ROWS, not on separately-encoded probes.
That distinction is the whole lesson: the old gate passed because its probes
were encoded in a small (uncorrupted) call while the store rows came from the
corrupted path, so the gate certified an instrument other than the one in use.
Here the probe vectors ARE store rows wherever the store has them.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_embed as KE

K = os.path.join(ROOT, "meta/M01_displacement/results/k")


def gate(lut, enc):
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    need = sorted({w for pr in KE.SYN_ZH + KE.UNREL_ZH for w in pr if w not in lut})
    extra = dict(zip(need, enc(need))) if need else {}
    pv = {**{w: lut[w] for w in lut}, **extra}
    s = float(np.median([cos(pv[a], pv[b]) for a, b in KE.SYN_ZH]))
    u = float(np.median([cos(pv[a], pv[b]) for a, b in KE.UNREL_ZH]))
    return s, u


def main():
    from sentence_transformers import SentenceTransformer

    old_p = os.path.join(K, "embed_zh_bge.npz")
    corrupt_p = os.path.join(K, "embed_zh_bge.MPSCORRUPT.npz")
    if not os.path.exists(corrupt_p):
        os.rename(old_p, corrupt_p)
        print("corrupted store moved aside -> %s" % os.path.basename(corrupt_p))
    old = np.load(corrupt_p, allow_pickle=True)
    words = sorted(str(w) for w in old["words"])
    model = str(old["model"])

    m = SentenceTransformer(model, device="cpu")
    enc = lambda ws: m.encode(list(ws), normalize_embeddings=True, batch_size=32,
                              show_progress_bar=False)
    print("encoding %d rated-verb words on CPU ..." % len(words))
    E = np.asarray(enc(words), np.float32)
    lut = dict(zip(words, E))

    s, u = gate(lut, enc)
    print("GATE on store rows: syn %.4f  unrel %.4f  gap %.4f" % (s, u, s - u))
    if s - u <= 0:
        print("REFUSING TO WRITE"); return 1
    rng = np.random.default_rng(20260813)
    idx = rng.choice(len(words), 400, replace=False)
    A = E[idx]
    ani = float(np.median((A @ A.T)[np.triu_indices(len(idx), 1)]))
    np.savez_compressed(old_p, words=np.array(words, dtype=object), E=E,
                        syn_median=s, unrel_median=u, syn_gap=s - u,
                        anisotropy=ani, model=model)
    print("-> %s  (%d words, CPU)" % (os.path.basename(old_p), len(words)))

    au = set()
    for ln in open(os.path.join(K, "word_auc_zh_nopos.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2 and p[0] != "word":
            au.add(p[0])
    extra_w = sorted(w for w in au if w not in lut)
    print("encoding %d armAUC extras ..." % len(extra_w))
    E2 = np.asarray(enc(extra_w), np.float32)
    allw = sorted(set(words) | set(extra_w))
    lut2 = {**lut, **dict(zip(extra_w, E2))}
    W = np.stack([lut2[w] for w in allw]).astype(np.float32)
    s2, u2 = gate(lut2, enc)
    print("GATE on wide rows: syn %.4f  unrel %.4f  gap %.4f" % (s2, u2, s2 - u2))
    if s2 - u2 <= 0:
        print("REFUSING TO WRITE wide"); return 1
    np.savez_compressed(os.path.join(K, "embed_zh_bge_wide.npz"),
                        words=np.array(allw, dtype=object), E=W,
                        syn_median=s2, unrel_median=u2, syn_gap=s2 - u2,
                        anisotropy=ani, model=model)
    print("-> embed_zh_bge_wide.npz  (%d words, CPU, one convention)" % len(allw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
