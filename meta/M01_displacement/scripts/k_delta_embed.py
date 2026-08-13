"""Build the CELL-level delta store: V(prompt + word) - V(prompt), bge-m3.

    uv run python meta/M01_displacement/scripts/k_delta_embed.py en
    -> data/delta_verbs_<lang>.npz     (float16, NOT committed -- see below)

WHY A CELL-LEVEL VECTOR IS A DIFFERENT KIND OF OBJECT. Every feature in P so far
assigns one vector per WORD: the eighteen rated norms, GloVe, bare-word bge, the
movement axis. P section 2 measures what that can ever buy -- a split-half oracle
using the word's own identity reaches +0.121 AUC over base probability, and
**ICC(1) is 0.131, so 87% of the movement variance is WITHIN a word across
sites** and no word-level feature can reach any of it. That is not a limitation
of our particular features; it is a ceiling on the whole class.

`V(prompt + word) - V(prompt)` is not in that class. The same word gets a
different vector at every site, so it is not bounded by the word-identity oracle,
and the question it makes askable for the first time is whether the 87% is
reachable at all.

THE PROBE SAID THE CONSTRUCT IS SOUND BEFORE THIS WAS RUN (`k_delta_probe`).
Bare-word bge separates near-synonyms from unrelated pairs by 0.1382 at
anisotropy 0.5569; the within-prompt delta separates them by 0.6027 at 0.1610,
which beats GloVe's 0.4001 reference using the encoder that was the study's worst
instrument an hour earlier. The synonym cosine barely moves (0.677 -> 0.732) --
what collapses is the UNRELATED cosine, 0.539 -> 0.129. Subtracting the prompt
removes the shared component, and anisotropy is a shared component.

IT ALSO DISSOLVES THE CHINESE UNIT PROBLEM RATHER THAN MITIGATING IT. We never
ask what the word IS, only what appending it DOES, so pkuseg's 71.6% isolation
rate and the 46% of zh types it never isolates stop being relevant. The join rule
is the language's own -- no space in Chinese, since inserting one would both
misdescribe the string the model predicted into and hand the segmenter a boundary.

THE STORE IS KEYED (prompt_sha16, word) AND SERVES EVERY MODEL PAIR. The delta
depends on the site and the candidate, not on which pair is being scored, so
293,172 encodings cover 6,157,263 English verb-movement cells. That ratio is why
this is affordable at all.

**FLOAT16, IN `data/raw/`, WHICH IS ALREADY GITIGNORED.** 293k x 1024 is 1.2 GB at
float32 and 600 MB at float16; the repo has a recorded incident of a 137 MB file
swept into a commit. Results derived from this store are committed; the store is
rebuildable from this script in about twenty minutes and is treated as a cache,
not as a record. The sidecar json carries the encoder id, the population query
and the counts, so a reader can tell what a rebuild would have to reproduce.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_analysis as A

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
#: `data/raw` is ALREADY gitignored (RH), so the store needs no new ignore
#: rule -- adding one would be a second mechanism for a job already done.
DATA = os.path.join(ROOT, "data/raw")
MODEL = "BAAI/bge-m3"
BATCH = 64


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main(lang="en"):
    from sentence_transformers import SentenceTransformer

    emb = "embed_%s_%s.npz" % (lang, "glove" if lang == "en" else "bge")
    z = np.load(os.path.join(K, emb), allow_pickle=True)
    verbs = sorted(set(z["words"].tolist()))
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    vs = "','".join(esc(v) for v in verbs)
    print("[%s] %d verbs from %s" % (lang, len(verbs), emb))

    rows = A.q("""
      SELECT DISTINCT m.prompt AS prompt, m.word AS word
      FROM %s.movement m
      INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                  WHERE status='ACTIVE' AND language='%s') pc ON m.prompt=pc.prompt
      WHERE m.rule='canonical' AND m.cls IN ('fall','rise')
        AND m.word IN ('%s')""" % (A.DB, A.DB, lang, vs))
    pairs = [(r["prompt"], r["word"]) for r in rows]
    prompts = sorted({p for p, _ in pairs})
    print("  %s (prompt, word) pairs over %s prompts"
          % (format(len(pairs), ","), format(len(prompts), ",")))

    m = SentenceTransformer(MODEL)
    print("  %s on %s" % (MODEL, m.device))
    enc = lambda xs: m.encode(list(xs), normalize_embeddings=True, batch_size=BATCH,
                              show_progress_bar=False)
    #: the language's own join rule, matching k_pos_context
    join = (lambda p, w: (p if p.endswith((" ", "\n")) else p + " ") + w) \
        if lang == "en" else (lambda p, w: p + w)

    t0 = time.time()
    PV = {}
    for i in range(0, len(prompts), 512):
        ch = prompts[i:i + 512]
        for p, v in zip(ch, enc(ch)):
            PV[p] = v
    print("  prompts encoded in %.1fs" % (time.time() - t0))

    D = np.zeros((len(pairs), PV[prompts[0]].shape[0]), np.float16)
    keys_p, keys_w = [], []
    t0 = time.time()
    for i in range(0, len(pairs), 512):
        ch = pairs[i:i + 512]
        E = enc([join(p, w) for p, w in ch])
        for j, ((p, w), v) in enumerate(zip(ch, E)):
            d = v - PV[p]
            n = np.linalg.norm(d)
            D[i + j] = (d / (n if n > 1e-12 else 1.0)).astype(np.float16)
            keys_p.append(sha16(p)); keys_w.append(w)
        if i and i % 25600 == 0:
            el = time.time() - t0
            print("    %s/%s  %.0f/sec  eta %.1f min"
                  % (format(i, ","), format(len(pairs), ","), i / el,
                     (len(pairs) - i) / max(i / el, 1) / 60), flush=True)

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "delta_verbs_%s.npz" % lang)
    np.savez_compressed(out, D=D, prompt_sha16=np.array(keys_p),
                        word=np.array(keys_w, dtype=object), model=MODEL)
    side = {"encoder": MODEL, "lang": lang, "n_pairs": len(pairs),
            "n_prompts": len(prompts), "n_words": len(set(keys_w)),
            "dim": int(D.shape[1]), "dtype": "float16",
            "join": "space" if lang == "en" else "none",
            "population": "movement rule=canonical, cls in (fall,rise), verbs from " + emb,
            "seconds": round(time.time() - t0, 1)}
    json.dump(side, open(os.path.join(K, "delta_verbs_%s.json" % lang), "w"), indent=1)
    print("\n  %s vectors in %.1f min -> %s (%.0f MB)"
          % (format(len(pairs), ","), (time.time() - t0) / 60,
             os.path.relpath(out, ROOT), os.path.getsize(out) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
