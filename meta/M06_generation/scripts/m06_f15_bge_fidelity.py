"""F1: does the drift claim survive a second embedder? bge-m3 vs MiniLM.

    uv run python meta/M06_generation/scripts/m06_f15_bge_fidelity.py [--n 60]
    -> results/f15_bge_fidelity.json + per-passage parquet

Runs plan_f15_on_passages amendment F1 (committed before this file existed).
THE GATING CHECK: drift is one of the two quadrant axes, so P2 AND P3 inherit
the requirement. Surprisal is untouched (the same GPT-2 values from the main
run's cells parquet), so quadrants move only through drift, and the comparison
is embedder-vs-embedder on IDENTICAL passages, identical truncation, identical
sentence split and prefix recipe.

The pass mark is declared in the plan as "large majority, no threshold set by
the author" -- a fidelity check whose bar is chosen after looking is not a
check. Both embedders' numbers are reported whatever they say.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CELLS = os.path.join(OUTD, "f15_on_passages_cells.parquet")
CH = "clickhouse"
SEED = 20260813
BGE = "BAAI/bge-m3"


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def sign_test(ds):
    ds = np.asarray(ds, float)
    up = int((ds > 0).sum()); dn = int((ds < 0).sum())
    lo = min(up, dn)
    p = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
    return {"median": float(np.median(ds)), "mean": float(np.mean(ds)),
            "n": len(ds), "up": up, "dn": dn, "p_sign": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="passages per (pair, role)")
    args = ap.parse_args()

    import pandas as pd
    from malign_logits.embedding import (drift_metrics_from_embeddings,
                                         _split_sentences)
    from corpus_metrics import truncate_to_min_sentences
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(CELLS)
    print("main-run cells: %s passages, %d pairs"
          % (format(len(df), ","), df.pair.nunique()))

    #: declared subsample: n per (pair, role), seed over sorted keys
    rng = np.random.default_rng(SEED)
    keep = []
    for k in sorted(set(zip(df.pair, df.role))):
        sub = df[(df.pair == k[0]) & (df.role == k[1])].sort_values(
            ["prompt_id", "sample_idx"])
        idx = (rng.choice(len(sub), args.n, replace=False)
               if len(sub) > args.n else np.arange(len(sub)))
        keep.append(sub.iloc[sorted(idx)])
    sel = pd.concat(keep)
    want = set(zip(sel.pair, sel.role, sel.prompt_id, sel.sample_idx))
    print("subsample: %s passages (%d per pair-role)" % (format(len(sel), ","), args.n))

    #: re-fetch text for exactly those keys
    texts = {}
    for r in ch_rows("SELECT pair, role, prompt_id, sample_idx, prompt, text "
                     "FROM malign_logits.gen_sequences "
                     "WHERE corpus='passage' AND forced_word=''"):
        k = (r["pair"], r["role"], r["prompt_id"], r["sample_idx"])
        if k in want:
            texts[k] = (r["prompt"], r["text"])
    print("texts recovered: %s of %s" % (format(len(texts), ","), format(len(want), ",")))

    #: same truncation + split + prefix recipe as the main producer
    work = []
    for r in sel.itertuples():
        k = (r.pair, r.role, r.prompt_id, r.sample_idx)
        if k not in texts:
            continue
        prm, text = texts[k]
        tr = truncate_to_min_sentences(text, min_words=75)
        t2 = tr[0] if isinstance(tr, tuple) else tr
        if t2 is None:
            continue
        sents = _split_sentences(t2)
        if len(sents) < 3:
            continue
        ss = list(sents)
        ss[0] = prm + " " + ss[0]
        work.append((k, ss))
    print("reproduced for scoring: %s passages" % format(len(work), ","))

    emb = SentenceTransformer(BGE)
    dev = str(emb.device)

    #: device gate ON THE STORE'S OWN ROWS, refuse-to-write on failure
    if "mps" in dev or "cuda" in dev:
        flat = sorted({s for _, ss in work for s in ss}, key=lambda s: (len(s), s))
        pick = flat[:20] + list(rng.choice(flat[20:], 40, replace=False))
        a = emb.encode(pick, show_progress_bar=False)
        emb.to("cpu")
        b = emb.encode(pick, show_progress_bar=False)
        emb.to(dev)
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        cos = (an * bn).sum(1)
        print("device gate (%s vs cpu): %d rows, min cos %.4f" % (dev, len(pick), cos.min()))
        if cos.min() < 0.999:
            raise SystemExit("REFUSING TO WRITE: device gate failed, min cos %.4f" % cos.min())
    else:
        print("embedder device %s: gate not needed" % dev)

    rows = []
    t0 = time.time()
    for i, (k, ss) in enumerate(work):
        v = emb.encode(ss, show_progress_bar=False)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
        d = drift_metrics_from_embeddings(v.tolist())
        rows.append({"pair": k[0], "role": k[1], "prompt_id": k[2],
                     "sample_idx": k[3], "bge_total_drift": d["total_drift"],
                     "bge_mean_drift": d["mean_drift"]})
        if i == 199:
            r = (time.time() - t0) / 200
            print("throughput: %.3f s/passage -> ETA %.1f min for %d"
                  % (r, r * (len(work) - 200) / 60, len(work)))
        if (i + 1) % 1000 == 0:
            print("  bge %d/%d (%.1f min elapsed)" % (i + 1, len(work),
                                                      (time.time() - t0) / 60))

    bg = pd.DataFrame(rows)
    m = sel.merge(bg, on=["pair", "role", "prompt_id", "sample_idx"], how="inner")
    print("\njoined: %s passages with both embedders" % format(len(m), ","))
    pq = os.path.join(OUTD, "f15_bge_fidelity_cells.parquet")
    m.to_parquet(pq)

    out = {"plan": "plans/plan_f15_on_passages.md#F1", "embedder": BGE,
           "n_per_cell": args.n, "n_passages": len(m),
           "passage_spearman": None}
    from scipy.stats import spearmanr
    rho = spearmanr(m.total_drift, m.bge_total_drift).statistic
    out["passage_spearman"] = float(rho)
    print("per-passage drift agreement (Spearman, MiniLM vs bge): %+.3f" % rho)

    #: P2 under each embedder, paired per pair
    print("\nP2 (aligned - base drift), paired per pair, under each embedder")
    piv = {}
    for col, lab in (("total_drift", "MiniLM"), ("bge_total_drift", "bge-m3")):
        c = m.groupby(["pair", "role"])[col].mean().unstack("role")
        c = c.dropna(subset=["aligned", "base"])
        piv[lab] = c["aligned"] - c["base"]
        r5 = sign_test(piv[lab].values)
        out["P2_" + lab] = r5
        print("  %-8s med %+.4f (mean %+.4f)  %d/%d  p %.3g  (n %d)"
              % (lab, r5["median"], r5["mean"], r5["up"], r5["dn"],
                 r5["p_sign"], r5["n"]))
    j = piv["MiniLM"].to_frame("a").join(piv["bge-m3"].to_frame("b"), how="inner")
    agree = int((np.sign(j.a) == np.sign(j.b)).sum())
    out["P2_sign_agreement"] = {"agree": agree, "of": len(j)}
    print("  SIGN AGREEMENT: %d of %d pairs" % (agree, len(j)))

    #: quadrant flow under bge drift (surprisal unchanged)
    print("\nP3 quadrant flow under each embedder (same surprisal, medians "
          "recomputed within this subsample)")
    med_s = float(m.mean_surprisal.median())
    for col, lab in (("total_drift", "MiniLM"), ("bge_total_drift", "bge-m3")):
        md = float(m[col].median())
        q = np.select([(m[col] >= md) & (m.mean_surprisal >= med_s),
                       (m[col] >= md) & (m.mean_surprisal < med_s),
                       (m[col] < md) & (m.mean_surprisal >= med_s)],
                      ["Q2", "Q1", "Q3"], default="Q4")
        mm = m.assign(q=q)
        sh = (mm.groupby(["pair", "role"]).q.value_counts(normalize=True)
              .unstack(fill_value=0.0))
        res = {}
        for qq in ("Q1", "Q2", "Q3", "Q4"):
            if qq not in sh.columns:
                continue
            c = sh[qq].unstack("role").dropna()
            r5 = sign_test((c["aligned"] - c["base"]).values)
            res[qq] = r5
            print("  %-8s %s  med %+.4f  %d/%d  p %.3g"
                  % (lab, qq, r5["median"], r5["up"], r5["dn"], r5["p_sign"]))
        out["P3_" + lab] = res

    p = os.path.join(OUTD, "f15_bge_fidelity.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
