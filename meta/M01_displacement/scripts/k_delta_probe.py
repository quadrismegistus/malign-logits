"""Is V(prompt + word) - V(prompt) a better word vector than V(word)?

    uv run python meta/M01_displacement/scripts/k_delta_probe.py en
    uv run python meta/M01_displacement/scripts/k_delta_probe.py zh
    -> results/k/delta_probe_<lang>.json

THE PROBLEM THIS IS PROBING. `k_embed` hands bge-m3 a bare word, which is a
one-word DOCUMENT to a passage encoder, not a type. Measured by its own gate:
English near-synonyms sit at 0.6774 and unrelated pairs at 0.5393, a gap of
0.1382, with anisotropy 0.5293 -- everything is close to everything. GloVe on the
same construct gives a gap of 0.4001 at anisotropy 0.0368. So every fine
geometric claim built on bare-word bge is built on the weak instrument, which is
the most likely reason the cross-language pole test behaved badly.

THE PROPOSAL, RH's: subtract the prompt. A difference of two nearby points
cancels their shared component, and anisotropy IS a shared component, so the
difference should be better conditioned than either endpoint. It also buys two
things that matter more than conditioning:

  - it needs no segmentation, so the Chinese unit problem disappears -- we never
    ask what the word IS, only what appending it DOES. 46% of zh word types are
    never isolated by pkuseg and this construct does not care.
  - it is a per-(prompt, word) vector, which is the grain the campaign has never
    had. ICC 0.131 says 87% of movement variance is WITHIN a word across sites,
    so a type vector is the wrong object for most of the signal by construction.

THIS SCRIPT IS THE CHEAP CHECK BEFORE THE EXPENSIVE RUN. Embedding every
(prompt, word) pair in the corpus is ~366k forward passes for English alone. The
gate pairs are ~40 words over a sample of real prompts, which is under a
thousand. If the synonym gap does not improve here, the idea is dead for a few
minutes instead of a few hours -- and if it does, the same numbers say by how
much before anything is committed to.

TWO READINGS, AND THE WITHIN-PROMPT ONE IS PRIMARY. Pooling the difference across
prompts rebuilds a type vector and reintroduces the object we are trying to get
away from; comparing two words' deltas AT THE SAME PROMPT holds context exactly
fixed and is the construct the eventual instrument would use. Both are reported.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_analysis as A
import k_embed as KE

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
NPROMPT = 24
SEED = 20260813


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def main(lang="en"):
    from sentence_transformers import SentenceTransformer

    syn = KE.SYN_EN if lang == "en" else KE.SYN_ZH
    unrel = KE.UNREL_EN if lang == "en" else KE.UNREL_ZH
    words = sorted({w for p in syn + unrel for w in p})
    print("[%s] %d probe words, %d synonym pairs, %d unrelated pairs"
          % (lang, len(words), len(syn), len(unrel)))

    rows = A.q("""SELECT DISTINCT prompt FROM %s.prompt_catalogue
                  WHERE status='ACTIVE' AND language='%s' ORDER BY prompt""" % (A.DB, lang))
    allp = [r["prompt"] for r in rows]
    rng = np.random.default_rng(SEED)
    prompts = [allp[i] for i in rng.choice(len(allp), min(NPROMPT, len(allp)),
                                           replace=False)]
    print("  %d prompts sampled from %d active" % (len(prompts), len(allp)))

    m = SentenceTransformer("BAAI/bge-m3")
    enc = lambda xs: m.encode(list(xs), normalize_embeddings=True, batch_size=32,
                              show_progress_bar=False)

    #: the join rule is the language's, exactly as k_pos_context: no space in zh
    join = (lambda p, w: (p if p.endswith((" ", "\n")) else p + " ") + w) \
        if lang == "en" else (lambda p, w: p + w)

    BARE = {w: v for w, v in zip(words, enc(words))}
    PV = {p: v for p, v in zip(prompts, enc(prompts))}
    texts, keys = [], []
    for p in prompts:
        for w in words:
            texts.append(join(p, w)); keys.append((p, w))
    E = enc(texts)
    D = {}
    for (p, w), v in zip(keys, E):
        d = v - PV[p]
        D[(p, w)] = d / max(np.linalg.norm(d), 1e-12)
    print("  %d prompt+word encodings" % len(texts))

    cos = lambda a, b: float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    out = {"lang": lang, "n_prompts": len(prompts), "n_words": len(words)}

    #: 1. BARE WORDS, recomputed here so the comparison is on the same pairs
    s = med([cos(BARE[a], BARE[b]) for a, b in syn])
    u = med([cos(BARE[a], BARE[b]) for a, b in unrel])
    ani = med([cos(BARE[words[i]], BARE[words[j]])
               for i in range(len(words)) for j in range(i + 1, len(words))])
    out["bare"] = {"syn": s, "unrel": u, "gap": s - u, "anisotropy": ani}
    print("\n  BARE WORD          syn %.4f  unrel %.4f  GAP %.4f  anisotropy %.4f"
          % (s, u, s - u, ani))

    #: 2. WITHIN-PROMPT DELTA -- primary
    ss = [cos(D[(p, a)], D[(p, b)]) for p in prompts for a, b in syn]
    uu = [cos(D[(p, a)], D[(p, b)]) for p in prompts for a, b in unrel]
    aa = [cos(D[(p, words[i])], D[(p, words[j])]) for p in prompts
          for i in range(len(words)) for j in range(i + 1, len(words))]
    s2, u2, a2 = med(ss), med(uu), med(aa)
    out["delta_within"] = {"syn": s2, "unrel": u2, "gap": s2 - u2, "anisotropy": a2,
                           "n_syn_obs": len(ss)}
    print("  DELTA within-prompt syn %.4f  unrel %.4f  GAP %.4f  anisotropy %.4f"
          % (s2, u2, s2 - u2, a2))

    #: 3. DELTA POOLED ACROSS PROMPTS -- rebuilds a type vector, reported for
    #: contrast rather than because it is the object we want
    T = {}
    for w in words:
        v = np.mean([D[(p, w)] for p in prompts], 0)
        T[w] = v / max(np.linalg.norm(v), 1e-12)
    s3 = med([cos(T[a], T[b]) for a, b in syn])
    u3 = med([cos(T[a], T[b]) for a, b in unrel])
    a3 = med([cos(T[words[i]], T[words[j]])
              for i in range(len(words)) for j in range(i + 1, len(words))])
    out["delta_pooled"] = {"syn": s3, "unrel": u3, "gap": s3 - u3, "anisotropy": a3}
    print("  DELTA pooled        syn %.4f  unrel %.4f  GAP %.4f  anisotropy %.4f"
          % (s3, u3, s3 - u3, a3))

    base = out["bare"]["gap"]
    print("\n  VERDICT vs bare-word gap %.4f:  within-prompt %+.4f | pooled %+.4f"
          % (base, out["delta_within"]["gap"] - base, out["delta_pooled"]["gap"] - base))
    print("  (GloVe/en reference gap is 0.4001 at anisotropy 0.0368)")
    p = os.path.join(K, "delta_probe_%s.json" % lang)
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
