"""Cache pilot: what does storing sentence embeddings cost in time and disk?

    uv run python meta/M06_generation/scripts/m06_cache_pilot.py [--n 100]

RH's question before committing to a cached re-encode. Measures, on the
EXPENSIVE case (bge-m3, CPU, Chinese sentences from f11_l2):

    encode + cache write   s/passage and s/sentence
    cache size delta       bytes/sentence, measured on disk not estimated
    cache READ             what a second run actually costs

Then projects to the populations we would actually run. No claim is made
from an estimate where a measurement was available.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

CJK = re.compile(r"[一-鿿]")
BGE = "BAAI/bge-m3"
THREADS = 6


def du(path):
    """Bytes on disk, measured."""
    out = subprocess.run(["du", "-sk", path], capture_output=True, text=True).stdout
    return int(out.split()[0]) * 1024 if out.strip() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    import torch
    import jieba
    import stanza
    from sentence_transformers import SentenceTransformer
    from malign_logits.cache import get_cache
    from malign_logits.embedding import drift_metrics_from_embeddings

    torch.set_num_threads(THREADS)
    cache = get_cache()
    root = cache.root

    q = ("SELECT model, prompt, text FROM malign_logits.gen_sequences "
         "WHERE corpus='f11_l2' LIMIT 4000 FORMAT JSONEachRow")
    rows = []
    for ln in subprocess.run(["clickhouse", "client", "-q", q],
                             capture_output=True, text=True).stdout.strip().split("\n"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if CJK.search(r["prompt"]):
            rows.append(r)
        if len(rows) >= args.n * 3:
            break

    zh_nlp = stanza.Pipeline("zh-hans", processors="tokenize", verbose=False,
                             use_gpu=False)
    work = []
    for r in rows:
        ss = [s.text.strip() for s in zh_nlp(r["text"]).sentences if s.text.strip()]
        nw = len([w for w in jieba.cut(r["text"]) if w.strip()])
        if nw < 75 or len(ss) < 3:
            continue
        ss = list(ss)
        ss[0] = r["prompt"] + " " + ss[0]
        work.append((r["prompt"], r["text"], ss))
        if len(work) >= args.n:
            break
    n_sent = sum(len(ss) for _, _, ss in work)
    print("pilot: %d passages, %d sentences (bge-m3, CPU, Chinese)"
          % (len(work), n_sent))

    emb = SentenceTransformer(BGE, device="cpu")
    print("cache root: %s" % root)
    before = du(root)

    t0 = time.time()
    for prompt, text, ss in work:
        v = emb.encode(ss, show_progress_bar=False)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
        cache.set_sent_embeddings(BGE, prompt, text, v.tolist())
        drift_metrics_from_embeddings(v.tolist())
    t_write = time.time() - t0
    after = du(root)
    delta = after - before

    t1 = time.time()
    hits = 0
    for prompt, text, ss in work:
        got = cache.get_sent_embeddings(BGE, prompt, text)
        if got is not None:
            hits += 1
            drift_metrics_from_embeddings(got)
    t_read = time.time() - t1

    print("\nMEASURED")
    print("  encode + cache write   %7.1f s   = %.3f s/passage, %.4f s/sentence"
          % (t_write, t_write / len(work), t_write / n_sent))
    print("  cache READ + metrics   %7.1f s   = %.3f s/passage  (%d/%d hits)"
          % (t_read, t_read / len(work), hits, len(work)))
    print("  speedup on a second run                 %.0fx" % (t_write / max(t_read, 1e-9)))
    print("  disk delta             %7.1f MB  = %.1f KB/sentence, %.1f KB/passage"
          % (delta / 1e6, delta / n_sent / 1024, delta / len(work) / 1024))

    print("\nPROJECTED (measured rates, not guessed)")
    for name, npass, nsent in (("f11_l2 cross-lingual, paired", 23677, 154000),
                               ("f11_l2 both langs, all", 26981, 173793),
                               ("passage corpus (F15)", 35230, 192122)):
        print("  %-30s %6s passages: %5.1f h encode, %6.1f GB cache, %4.1f min on re-read"
              % (name, format(npass, ","), npass * (t_write / len(work)) / 3600,
                 nsent * (delta / n_sent) / 1e9, npass * (t_read / len(work)) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
