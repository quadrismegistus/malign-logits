"""Embed the K verb vocabulary as BARE WORDS, and check the encoder first.

    uv run python meta/M01_displacement/scripts/k_embed.py en bge
    uv run python meta/M01_displacement/scripts/k_embed.py zh bge
    -> results/k/embed_<lang>_<name>.npz   words + matrix, one row per rating unit

WHY THIS EXISTS. `k_ceiling` shows word identity beats the nuisance by 0.12 AUC
while the eighteen rated norms buy +0.003 of it. So there is real word-level
information the rating instrument is not capturing, and the question is whether a
distributional representation captures it. If it does, the finding is that
alignment sorts words along a dimension the affective vocabulary does not name.
If it does not, the word-level signal is not semantic and the honest move is to
stop building word features and build a word-by-site instrument.

THE VECTOR MUST BE A PURE FUNCTION OF THE WORD. No prompt, no context, no
template. The moment context enters, the feature is no longer word-level and its
score is no longer comparable to the 0.12 headroom, which was computed over
exactly the class of features that are constant within a word.

BGE-M3 ON A BARE WORD IS OUT OF DISTRIBUTION AND THE CAMPAIGN'S EXISTING GATE
DOES NOT COVER IT. Docket [459] gate-checked this encoder for word-sensitivity
and synonym sanity on `prompt + " " + word` -- a sentence. Handing it a single
token is a different use, and a gate passed for one use is not evidence about
another. So this script runs its own check before writing anything:

    SYNONYM SANITY   near-synonyms must embed closer than unrelated pairs, on
                     bare words. Reported as the two medians and the gap; a gap
                     at or below zero means the encoder is not representing
                     bare-word meaning and the downstream null would be about
                     the encoder rather than about alignment.
    ANISOTROPY       median pairwise cosine over a random sample. Sentence
                     encoders famously put everything in a narrow cone; if the
                     median is very high the usable variance is small and the
                     PCA sweep downstream matters more than the raw dimension.

Both numbers are written into the npz so a reader never has to take the check on
trust, and the script REFUSES TO WRITE if the synonym gap is not positive.
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
#: `glove-wiki-gigaword-300` is ALREADY ON DISK at ~/gensim-data, pulled on
#: 9 Aug for `M02_frame_exit/scripts/pole_axis_glove_replication.py`. Reusing the
#: campaign's existing vectors rather than a new download keeps the instrument
#: consistent across findings -- [5195] reports the pole-axis test as "flat on
#: two independent embedders" and this is one of the two.
MODELS = {"bge": "BAAI/bge-m3", "glove": "glove-wiki-gigaword-300"}
SEED = 20260812

#: near-synonym pairs and unrelated pairs, verbs, chosen before any embedding was
#: computed. Deliberately ordinary: an encoder that cannot separate these on bare
#: words cannot be trusted with `stabbed` against `whacked`.
SYN_EN = [("kill", "murder"), ("shout", "yell"), ("begin", "start"),
          ("hit", "strike"), ("grab", "seize"), ("look", "gaze"),
          ("hate", "despise"), ("fix", "repair"), ("cry", "weep"),
          ("throw", "hurl"), ("break", "shatter"), ("laugh", "chuckle")]
UNREL_EN = [("kill", "bake"), ("shout", "knit"), ("begin", "swim"),
            ("hit", "read"), ("grab", "sing"), ("look", "digest"),
            ("hate", "plough"), ("fix", "dance"), ("cry", "compute"),
            ("throw", "spell"), ("break", "whisper"), ("laugh", "measure")]
SYN_ZH = [("杀", "杀害"), ("喊", "叫喊"), ("开始", "起始"), ("打", "击打"),
          ("抓", "抓住"), ("看", "注视"), ("恨", "憎恨"), ("修", "修理")]
UNREL_ZH = [("杀", "烤"), ("喊", "编织"), ("开始", "游泳"), ("打", "阅读"),
            ("抓", "唱歌"), ("看", "消化"), ("恨", "耕种"), ("修", "跳舞")]


def is_verb(u, lang):
    from malign_logits import fields as FL
    if lang == "en":
        e = FL._byu().get(u.strip().lower())
        return bool(e) and e[1].startswith("vv")
    import jieba.posseg as pseg
    seg = list(pseg.cut(u.strip()))
    return len(seg) == 1 and seg[0].flag.startswith("v")


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _encoder(name):
    """-> encode(list_of_words) giving (kept_words, unit-norm matrix).

    GloVe HAS NO SUBWORD FALLBACK, so uncovered words are DROPPED and the count
    reported -- never zero-vectored. A zero vector is a point in the space and
    would put every uncovered word at the same location, which is the direction
    that manufactures structure in a study whose headline is a number near zero.
    """
    if name == "glove":
        import gensim.downloader as api
        KV = api.load(MODELS[name])

        def enc(ws):
            keep = [w for w in ws if w.strip().lower() in KV]
            if not keep:
                return [], np.zeros((0, KV.vector_size), np.float32)
            E = np.array([KV[w.strip().lower()] for w in keep], np.float32)
            E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
            return keep, E
        return enc

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELS[name])

    def enc(ws):
        return list(ws), m.encode(list(ws), normalize_embeddings=True,
                                  batch_size=64, show_progress_bar=False)
    return enc


def main(lang, name):
    if name == "glove" and lang != "en":
        print("[%s/glove] glove-wiki-gigaword-300 is English. Use bge for Chinese, "
              "and do not compare a zh/bge number to an en/glove one." % lang)
        return 1
    rate = json.load(open(os.path.join(K, "ratings_%s.json" % lang)))["ratings"]
    words = sorted(u for u in rate if is_verb(u, lang))
    print("[%s/%s] %d lexical verbs to embed as bare words" % (lang, name, len(words)))

    encode = _encoder(name)
    syn, unrel = (SYN_EN, UNREL_EN) if lang == "en" else (SYN_ZH, UNREL_ZH)
    probe = sorted({w for p in syn + unrel for w in p})
    pk, P = encode(probe)
    if len(pk) < len(probe):
        print("  gate probe: %d of %d probe words are in the vocabulary"
              % (len(pk), len(probe)))
    pv = dict(zip(pk, P))
    syn = [p for p in syn if p[0] in pv and p[1] in pv]
    unrel = [p for p in unrel if p[0] in pv and p[1] in pv]
    s = float(np.median([cos(pv[a], pv[b]) for a, b in syn]))
    u = float(np.median([cos(pv[a], pv[b]) for a, b in unrel]))
    print("  GATE synonym sanity on BARE WORDS: synonyms %.4f, unrelated %.4f, "
          "gap %+.4f" % (s, u, s - u))
    if s - u <= 0:
        print("  REFUSING TO WRITE: the encoder does not separate near-synonyms "
              "from unrelated words on bare input, so any downstream null would "
              "be a fact about the encoder.")
        return 1

    words, E = encode(words)
    print("  vocabulary coverage: %d words embedded (uncovered words are dropped, "
          "never zero-vectored)" % len(words))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(E), size=min(400, len(E)), replace=False)
    S = E[idx] @ E[idx].T
    aniso = float(np.median(S[np.triu_indices(len(idx), k=1)]))
    print("  GATE anisotropy: median pairwise cosine over %d words = %.4f"
          % (len(idx), aniso))
    print("  -> %d x %d matrix" % E.shape)

    p = os.path.join(K, "embed_%s_%s.npz" % (lang, name))
    np.savez_compressed(p, words=np.array(words, dtype=object), E=E.astype(np.float32),
                        syn_median=s, unrel_median=u, syn_gap=s - u,
                        anisotropy=aniso, model=MODELS[name])
    print("  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en",
                  sys.argv[2] if len(sys.argv) > 2 else "bge"))
