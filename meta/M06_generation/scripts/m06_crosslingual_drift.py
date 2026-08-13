"""Cross-lingual drift: bge-m3 trajectory drift on f11_l2, Chinese and English.

    uv run python meta/M06_generation/scripts/m06_crosslingual_drift.py [--cap 3]
    -> results/crosslingual_drift.json + crosslingual_drift_cells.parquet

Runs plan_crosslingual_drift (committed before this file existed). INSTRUMENT
RUN: it prints coverage, the mps-vs-CPU diagnostic and per-language
distributions, and DELIBERATELY PRINTS NO ARM CONTRAST -- the base/aligned
split is a separate plan with its own declared directions, and computing it
here would spend the confirmatory read before anyone declared it.

CPU is mandatory for both languages (mps-CJK family, three instances). The
diagnostic measures whether that hazard reaches SENTENCE-length Chinese, which
neither prior instance tested; it does not gate this run's output, which is
CPU-computed either way.

Chinese is written first so its parquet lands before the English half runs.
"""
import argparse
import collections
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

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = "clickhouse"
BGE = "BAAI/bge-m3"
SEED = 20260813
MIN_SENTS = 3

MIN_WORDS_EN = 75
THREADS = 6               # leave headroom for the concurrent F3 run

CJK = re.compile(r"[一-鿿]")
MIN_WORDS_ZH = 75         # SAME number as English; jieba supplies the words


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3)
    args = ap.parse_args()

    import torch
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from malign_logits.embedding import drift_metrics_from_embeddings, _split_sentences
    from corpus_metrics import truncate_to_min_sentences

    torch.set_num_threads(THREADS)

    #: fetch, split by PROMPT language, cap per (model, prompt) over sorted keys
    by_cell = collections.defaultdict(list)
    n = 0
    for r in ch_rows("SELECT model, prompt, text FROM malign_logits.gen_sequences "
                     "WHERE corpus='f11_l2'"):
        n += 1
        lang = "zh" if CJK.search(r["prompt"]) else "en"
        by_cell[(lang, r["model"], r["prompt"])].append(r["text"])
    print("f11_l2 rows: %s | cells: %s" % (format(n, ","), format(len(by_cell), ",")))
    for lang in ("zh", "en"):
        ks = [k for k in by_cell if k[0] == lang]
        print("  %s: %d cells, %d models, %d prompts"
              % (lang, len(ks), len({k[1] for k in ks}), len({k[2] for k in ks})))

    rng = np.random.default_rng(SEED)
    work = collections.defaultdict(list)
    for k in sorted(by_cell):
        v = sorted(by_cell[k])
        if len(v) > args.cap:
            v = [v[i] for i in sorted(rng.choice(len(v), args.cap, replace=False))]
        for i, t in enumerate(v):
            work[k[0]].append((k[1], k[2], i, t))
    print("capped sample: zh %s | en %s (cap %d/cell)"
          % (format(len(work["zh"]), ","), format(len(work["en"]), ","), args.cap))

    #: apply each language's own floor -- SAME criterion (75 words, >=3
    #: sentences), each language's own trained segmenter supplying the units
    import jieba
    import stanza
    zh_nlp = stanza.Pipeline("zh-hans", processors="tokenize", verbose=False,
                             use_gpu=False)
    kept = collections.defaultdict(list)
    dropped = collections.Counter()
    sens = collections.Counter()
    t0 = time.time()
    for j, (model, prompt, i, text) in enumerate(work["zh"]):
        ss = [s.text.strip() for s in zh_nlp(text).sentences if s.text.strip()]
        nw = len([w for w in jieba.cut(text) if w.strip()])
        for thr in (50, 75, 100):
            if nw >= thr and len(ss) >= MIN_SENTS:
                sens["zh_words_%d" % thr] += 1
        if nw < MIN_WORDS_ZH or len(ss) < MIN_SENTS:
            dropped["zh"] += 1
            continue
        ss = list(ss)
        ss[0] = prompt + " " + ss[0]
        kept["zh"].append((model, prompt, i, ss, nw))
        if (j + 1) % 4000 == 0:
            print("  zh segmentation %d/%d (%.1f min)"
                  % (j + 1, len(work["zh"]), (time.time() - t0) / 60))
    for model, prompt, i, text in work["en"]:
        tr = truncate_to_min_sentences(text, min_words=MIN_WORDS_EN)
        t2 = tr[0] if isinstance(tr, tuple) else tr
        if t2 is None:
            dropped["en"] += 1
            continue
        ss = _split_sentences(t2)
        if len(ss) < MIN_SENTS:
            dropped["en"] += 1
            continue
        ss = list(ss)
        ss[0] = prompt + " " + ss[0]
        kept["en"].append((model, prompt, i, ss, len(t2)))
    for lang in ("zh", "en"):
        tot = len(work[lang])
        print("%s floors: kept %s of %s (%.1f%%), dropped %s"
              % (lang, format(len(kept[lang]), ","), format(tot, ","),
                 100 * len(kept[lang]) / max(tot, 1), format(dropped[lang], ",")))
    print("zh sensitivity (>=3 sents AND >= N jieba words): %s"
          % {k: v for k, v in sorted(sens.items())})

    emb = SentenceTransformer(BGE, device="cpu")
    print("embedder device: %s (CPU mandatory, mps-CJK family)" % emb.device)

    #: CACHE THE VECTORS, so this is the LAST full encode of this corpus.
    #: Measured: encode+write 0.321 s/passage against a cached read at
    #: 0.004 s/passage -- 88x on any second run ([5858] pilot).
    #: PASS THE NDARRAY, NEVER `.tolist()`. The stash serialises ndarrays
    #: natively (verified: dtype, shape and BITWISE equality all preserved),
    #: while a list turns every float into a ~20-char decimal repr: measured
    #: through this cache at 26.3 KB/vector against 8.8 KB for the ndarray.
    #: One method call was 3x the storage of the whole campaign's embeddings.
    from malign_logits.cache import get_cache
    cache = get_cache()

    #: DIAGNOSTIC, not a gate: does the mps hazard reach sentence-length Chinese?
    diag = None
    if torch.backends.mps.is_available() and kept["zh"]:
        flat = sorted({s for _, _, _, ss, _ in kept["zh"][:400] for s in ss},
                      key=lambda s: (len(s), s))[:60]
        a = emb.encode(flat, show_progress_bar=False)
        emb.to("mps")
        b = emb.encode(flat, show_progress_bar=False)
        emb.to("cpu")
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        cos = (an * bn).sum(1)
        diag = {"n": len(flat), "min_cos": float(cos.min()),
                "median_cos": float(np.median(cos)),
                "n_below_0999": int((cos < 0.999).sum()),
                "shortest_len": int(min(len(s) for s in flat)),
                "longest_len": int(max(len(s) for s in flat))}
        print("DIAGNOSTIC mps vs cpu on %d zh SENTENCES (len %d-%d): min cos %.4f, "
              "median %.4f, %d below 0.999"
              % (diag["n"], diag["shortest_len"], diag["longest_len"],
                 diag["min_cos"], diag["median_cos"], diag["n_below_0999"]))

    out = {"plan": "plans/plan_crosslingual_drift.md", "embedder": BGE,
           "cap": args.cap, "device": "cpu", "mps_diagnostic": diag,
           "zh_sensitivity": dict(sens), "languages": {}}

    for lang in ("zh", "en"):
        rows, t0 = [], time.time()
        n_hit = 0
        for i, (model, prompt, sidx, ss, ln) in enumerate(kept[lang]):
            ctext = "\n".join(ss)
            got = cache.get_sent_embeddings(BGE, prompt, ctext)
            if got is not None:
                v = np.asarray(got, dtype=np.float32)   # accepts either form
                n_hit += 1
            else:
                v = emb.encode(ss, show_progress_bar=False)
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
                cache.set_sent_embeddings(BGE, prompt, ctext, v)
            d = drift_metrics_from_embeddings(v.tolist())
            #: PERSIST THE WHOLE METRIC VECTOR, not two fields of seven.
            #: The encode is the expensive step; everything derived from it is
            #: free AT THIS MOMENT and costs a full re-encode later. Storing
            #: two fields cost exactly that once already.
            #: `mean_pairwise` is the extra scalar the ORDERING test needs:
            #: under a random sentence order the expected successive distance
            #: IS the mean of all pairwise distances, so
            #: mean_drift - mean_pairwise is a pure ordering measure with
            #: composition held fixed by construction -- computable forever
            #: from this column, with no embeddings retained.
            sim = v @ v.T
            n = len(v)
            mean_pairwise = (float((1.0 - sim).sum() / (n * (n - 1)))
                             if n > 1 else float("nan"))
            rows.append({"lang": lang, "model": model, "prompt": prompt,
                         "sample_idx": sidx, "n_sents": len(ss), "length": ln,
                         "total_drift": d["total_drift"],
                         "mean_drift": d["mean_drift"],
                         "max_drift": d["max_drift"],
                         "std_drift": d["std_drift"],
                         "path_length": d["path_length"],
                         "directedness": d["directedness"],
                         "mean_pairwise": mean_pairwise})
            if i == 99:
                r = (time.time() - t0) / 100
                print("  %s throughput %.3f s/passage -> ETA %.1f min"
                      % (lang, r, r * (len(kept[lang]) - 100) / 60))
            if (i + 1) % 2000 == 0:
                print("  %s %d/%d (%.1f min)" % (lang, i + 1, len(kept[lang]),
                                                 (time.time() - t0) / 60))
        print("  %s cache: %s hits of %s (%.1f%%)"
              % (lang, format(n_hit, ","), format(len(kept[lang]), ","),
                 100 * n_hit / max(len(kept[lang]), 1)))
        df = pd.DataFrame(rows)
        pq = os.path.join(OUTD, "crosslingual_drift_%s_cells.parquet" % lang)
        df.to_parquet(pq)
        out["languages"][lang] = {
            "n_passages": len(df), "n_models": int(df.model.nunique()),
            "n_prompts": int(df.prompt.nunique()),
            "total_drift_median": float(df.total_drift.median()),
            "total_drift_iqr": [float(df.total_drift.quantile(.25)),
                                float(df.total_drift.quantile(.75))],
            "mean_drift_median": float(df.mean_drift.median()),
            "n_sents_median": float(df.n_sents.median())}
        print("%s: %s passages -> %s | total_drift median %.4f (IQR %.4f-%.4f)"
              % (lang, format(len(df), ","), os.path.basename(pq),
                 out["languages"][lang]["total_drift_median"],
                 out["languages"][lang]["total_drift_iqr"][0],
                 out["languages"][lang]["total_drift_iqr"][1]))
        json.dump(out, open(os.path.join(OUTD, "crosslingual_drift.json"), "w"),
                  indent=1)

    print("\nNO ARM CONTRAST COMPUTED -- the base/aligned split is a separate "
          "plan with its own declared directions (plan section: what is not "
          "declared).")
    print("  -> %s" % os.path.relpath(os.path.join(OUTD, "crosslingual_drift.json"), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
