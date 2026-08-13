"""F15 on passages: surprisal x drift quadrants on the M06 corpus.

    uv run python meta/M06_generation/scripts/m06_f15_on_passages.py --smoke
    uv run python meta/M06_generation/scripts/m06_f15_on_passages.py --cap N
    -> results/f15_on_passages{_smoke,}.json + per-passage parquet

Runs plan_f15_on_passages (committed f2f5c804, BEFORE this file existed).
METRIC CODE IS IMPORTED from malign_logits.embedding -- the committed F15
instrument -- never reimplemented here; this file is fetch, strata, unit and
the device gate. Sequential surprisal on non-cuda mirrors the committed
pipeline's own choice. Truncation min_words=75 per the F15 finding text (the
script default 100 is the recorded discrepancy, see the plan).
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
FLAGS = os.path.join(ROOT, "meta/M06_generation/data/m06_text_flags.parquet")
CH = "clickhouse"
EXCLUDE = "SmolLM2-360M"
SMOKE_BASES = ("LLM360/Amber", "allenai/Olmo-3-1025-7B",
               "meta-llama/Llama-3.1-8B", "google/gemma-2-9b")
MIN_WORDS = 75      # the finding text's rule, not the script default
MIN_SENTS = 3       # the committed pipeline's floor
SEED = 20260813
REF = "gpt2"        # plan: the only reference still independent of the roster
EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"   # plan: full-population primary


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


def gate_embedder(embedder, sents_by_row):
    """k_delta_embed-style second-device gate ON THE STORE'S OWN ROWS."""
    import torch
    dev = str(embedder.device)
    if "mps" not in dev and "cuda" not in dev:
        print("embedder device %s: gate not needed" % dev)
        return
    flat = [s for sents in sents_by_row for s in sents]
    flat = sorted(set(flat), key=lambda s: (len(s), s))
    rng = np.random.default_rng(SEED)
    pick = flat[:20] + list(rng.choice(flat[20:], min(40, max(0, len(flat) - 20)),
                                       replace=False))
    a = embedder.encode(pick, show_progress_bar=False)
    cpu = embedder.to("cpu")
    b = cpu.encode(pick, show_progress_bar=False)
    embedder.to(dev)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    cos = (an * bn).sum(1)
    print("device gate (%s vs cpu): %d rows, min cos %.4f, median %.4f"
          % (dev, len(pick), cos.min(), np.median(cos)))
    if cos.min() < 0.999:
        raise SystemExit("REFUSING TO WRITE: second-device gate failed "
                         "(min cos %.4f < 0.999) -- the mps hazard rule" % cos.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cap", type=int, default=3,
                    help="max samples per (pair, role, prompt)")
    args = ap.parse_args()

    import pandas as pd
    from malign_logits.embedding import (drift_metrics_from_embeddings,
                                         passage_surprisal,
                                         surprisal_metrics_from_tokens,
                                         _get_embedder, _split_sentences)
    from corpus_metrics import truncate_to_min_sentences

    flags = pd.read_parquet(FLAGS).rename(columns={"seq_idx": "sample_idx"})
    flags = flags[~flags.pair.str.contains(EXCLUDE)]
    flags["degenerate"] = ((flags.top_word_share >= 0.20)
                           | (flags.non_ascii_alpha_share >= 0.20))
    flags["english"] = flags.english_nltkwords_share >= 0.60
    fidx = {(r.pair, r.role, r.prompt_id, r.sample_idx): (r.degenerate, r.english)
            for r in flags.itertuples()}

    #: fetch undisturbed, stratum, cap per (pair, role, prompt) over sorted keys
    by_cell = collections.defaultdict(list)
    n_rows = 0
    for r in ch_rows("SELECT pair, role, prompt_id, sample_idx, prompt, text "
                     "FROM malign_logits.gen_sequences "
                     "WHERE corpus='passage' AND forced_word=''"):
        if EXCLUDE in r["pair"]:
            continue
        if args.smoke and not any(r["pair"].startswith(b) for b in SMOKE_BASES):
            continue
        fl = fidx.get((r["pair"], r["role"], r["prompt_id"], r["sample_idx"]))
        if fl is None or fl[0] or not fl[1]:
            continue
        n_rows += 1
        by_cell[(r["pair"], r["role"], r["prompt_id"])].append(
            (r["sample_idx"], r["prompt"], r["text"]))
    rng = np.random.default_rng(SEED)
    rows = []
    for k in sorted(by_cell):
        v = sorted(by_cell[k])
        if len(v) > args.cap:
            idx = rng.choice(len(v), args.cap, replace=False)
            v = [v[i] for i in sorted(idx)]
        rows.extend((k[0], k[1], k[2], s, p, t) for s, p, t in v)
    print("stratum rows %s -> capped sample %s passages (cap %d/cell)"
          % (format(n_rows, ","), format(len(rows), ","), args.cap))

    #: truncation, the F15 rule
    kept, n_short, n_fewsents = [], 0, 0
    for pair, role, pid, sidx, prm, text in rows:
        tr = truncate_to_min_sentences(text, min_words=MIN_WORDS)
        t2 = tr[0] if isinstance(tr, tuple) else tr
        if t2 is None:
            n_short += 1
            continue
        sents = _split_sentences(t2)
        if len(sents) < MIN_SENTS:
            n_fewsents += 1
            continue
        kept.append((pair, role, pid, sidx, prm, t2, sents))
    surv = collections.Counter((p, r) for p, r, *_ in kept)
    tot = collections.Counter((p, r) for p, r, *_ in rows)
    sr = {"%s:%s" % k: round(surv[k] / tot[k], 3) for k in sorted(tot)}
    print("truncation: %d kept | %d under %d words | %d under %d sentences"
          % (len(kept), n_short, MIN_WORDS, n_fewsents, MIN_SENTS))
    per_role = collections.defaultdict(list)
    for k2, v in sr.items():
        per_role[k2.split(":")[1]].append(v)
    print("survival by role: %s" % {k2: round(float(np.mean(v)), 3)
                                    for k2, v in per_role.items()})

    #: drift -- committed encoding recipe (prefix on first sentence, normalise)
    embedder = _get_embedder(EMBEDDER)
    sent_lists = []
    for pair, role, pid, sidx, prm, t2, sents in kept:
        ss = list(sents)
        ss[0] = prm + " " + ss[0]
        sent_lists.append(ss)
    gate_embedder(embedder, sent_lists[:200])
    out_rows = []
    for i, (pair, role, pid, sidx, prm, t2, sents) in enumerate(kept):
        vecs = embedder.encode(sent_lists[i], show_progress_bar=False)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        d = drift_metrics_from_embeddings(vecs.tolist())
        out_rows.append({"pair": pair, "role": role, "prompt_id": pid,
                         "sample_idx": sidx, "n_sents": len(sents),
                         "n_words": len(t2.split()),
                         "total_drift": d["total_drift"],
                         "mean_drift": d["mean_drift"]})
        if (i + 1) % 500 == 0:
            print("  drift %d/%d" % (i + 1, len(kept)))

    #: surprisal -- sequential on non-cuda, the committed pipeline's choice
    for i, (pair, role, pid, sidx, prm, t2, sents) in enumerate(kept):
        ps = passage_surprisal(t2, model_name=REF, prompt_prefix=prm)
        s = surprisal_metrics_from_tokens(ps["token_surprisals"])
        out_rows[i]["mean_surprisal"] = s["mean_surprisal"]
        out_rows[i]["n_tokens"] = s.get("n_tokens", ps.get("n_tokens"))
        if (i + 1) % 500 == 0:
            print("  surprisal %d/%d" % (i + 1, len(kept)))

    df = pd.DataFrame(out_rows)
    suf = "_smoke" if args.smoke else ""
    pq = os.path.join(OUTD, "f15_on_passages%s_cells.parquet" % suf)
    df.to_parquet(pq)
    print("per-passage metrics persisted: %s rows -> %s"
          % (format(len(df), ","), os.path.basename(pq)))

    #: quadrants -- pooled medians over both arms, computed once, reported
    med_d = float(df.total_drift.median())
    med_s = float(df.mean_surprisal.median())
    df["quadrant"] = np.select(
        [(df.total_drift >= med_d) & (df.mean_surprisal >= med_s),
         (df.total_drift >= med_d) & (df.mean_surprisal < med_s),
         (df.total_drift < med_d) & (df.mean_surprisal >= med_s)],
        ["Q2", "Q1", "Q3"], default="Q4")

    cell = df.groupby(["pair", "role"]).agg(
        surp=("mean_surprisal", "mean"), drift=("total_drift", "mean"),
        n=("total_drift", "size")).reset_index()
    qshare = (df.groupby(["pair", "role"]).quadrant
              .value_counts(normalize=True).unstack(fill_value=0.0)
              .reset_index())

    out = {"plan": "plans/plan_f15_on_passages.md", "ref": REF,
           "embedder": EMBEDDER, "min_words": MIN_WORDS,
           "median_drift": med_d, "median_surprisal": med_s,
           "n_passages": len(df),
           "truncation": {"kept": len(kept), "short": n_short,
                          "few_sents": n_fewsents}}
    print("\npaired contrasts, aligned - base per pair "
          "(medians travel, means beside)")
    piv = cell.pivot(index="pair", columns="role")
    for m, lab in (("surp", "P1 mean_surprisal"), ("drift", "P2 total_drift")):
        ds = (piv[(m, "aligned")] - piv[(m, "base")]).dropna()
        r5 = sign_test(ds.values)
        out[lab.split()[0]] = r5
        print("  %-18s med %+.4f (mean %+.4f)  %d/%d  p %.3g  (n %d)"
              % (lab, r5["median"], r5["mean"], r5["up"], r5["dn"],
                 r5["p_sign"], r5["n"]))
    qp = qshare.pivot(index="pair", columns="role") \
        if "role" in qshare.columns else None
    for q in ("Q1", "Q2", "Q3", "Q4"):
        if qp is not None and (q, "aligned") in qp.columns:
            ds = (qp[(q, "aligned")] - qp[(q, "base")]).dropna()
            r5 = sign_test(ds.values)
            out["P3_" + q] = r5
            print("  %-18s med %+.4f (mean %+.4f)  %d/%d  p %.3g"
                  % ("P3 share " + q, r5["median"], r5["mean"], r5["up"],
                     r5["dn"], r5["p_sign"]))

    p = os.path.join(OUTD, "f15_on_passages%s.json" % suf)
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
