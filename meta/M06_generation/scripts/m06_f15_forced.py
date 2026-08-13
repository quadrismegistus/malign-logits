"""F3: do the F15 movements alter after forcing a faller or a riser?

    uv run python meta/M06_generation/scripts/m06_f15_forced.py [--cap 2]
    -> results/f15_forced.json + per-passage parquet

Runs plan_f15_on_passages amendment F3 (committed before this file existed).
RH's question. I5 asked what forcing does to COMPOSITION (axis score) and the
ascent branch to LEVEL (second-order markers); both said "dragged
symmetrically, no alignment-specific response". This asks the third grain:
TRAJECTORY (drift) and PREDICTABILITY (reference surprisal).

Three declared readings, none directional except P6:
  F3a  does the arm contrast SURVIVE forcing (P6: yes -- a disposition is not
       abolished by one injected word); measured on the MATCHED arm, which is
       forced but not transgressive, against the undisturbed gap.
  F3b  does forcing ITSELF move the metrics (Q3, open: a forced faller is
       off-policy for the aligned model, so its continuation may be locally
       surprising -- or the model may recover into generic continuation).
  F3c  the DiD (Q4, open: null like every other DiD in this series?).
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
FLAGS = os.path.join(ROOT, "meta/M06_generation/data/m06_text_flags.parquet")
MAIN = os.path.join(OUTD, "f15_on_passages_cells.parquet")
CH = "clickhouse"
EXCLUDE = "SmolLM2-360M"
FENCED = "deepseek"        # [5776] text-grain fence, committed
ARMS = ("faller", "matched", "riser_matched")
MIN_WORDS = 75
MIN_SENTS = 3
SEED = 20260813
REF = "gpt2"
EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"


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
    ap.add_argument("--cap", type=int, default=2,
                    help="passages per (pair, prompt, role, arm)")
    args = ap.parse_args()

    import pandas as pd
    from malign_logits.embedding import (drift_metrics_from_embeddings,
                                         passage_surprisal,
                                         surprisal_metrics_from_tokens,
                                         _get_embedder, _split_sentences)
    from corpus_metrics import truncate_to_min_sentences

    arms = json.load(open(os.path.join(ROOT, "data/forced_arms_46reps_drmatch.json")))
    armof = {}
    for c in arms["cells"]:
        for col, an in (("faller", "faller"), ("matched", "matched"),
                        ("riser", "riser"), ("riser_matched", "riser_matched"),
                        ("faller-matched", "matched"),
                        ("riser-matched", "riser_matched")):
            w = c.get(col)
            if w:
                armof[(c["pair"], c["prompt"], w)] = an
    print("arm lookup: %s entries" % format(len(armof), ","))

    flags = pd.read_parquet(FLAGS).rename(columns={"seq_idx": "sample_idx"})
    flags["degenerate"] = ((flags.top_word_share >= 0.20)
                           | (flags.non_ascii_alpha_share >= 0.20))
    flags["english"] = flags.english_nltkwords_share >= 0.60
    fidx = {(r.pair, r.role, r.prompt_id, r.sample_idx): (r.degenerate, r.english)
            for r in flags.itertuples()}

    by_cell = collections.defaultdict(list)
    n_rows = n_arm = 0
    for r in ch_rows("SELECT pair, role, prompt_id, sample_idx, prompt, text, "
                     "forced_word FROM malign_logits.gen_sequences "
                     "WHERE corpus='passage' AND forced_word != ''"):
        if EXCLUDE in r["pair"] or FENCED in r["pair"]:
            continue
        fl = fidx.get((r["pair"], r["role"], r["prompt_id"], r["sample_idx"]))
        if fl is None or fl[0] or not fl[1]:
            continue
        n_rows += 1
        prm = r["prompt_id"][len(r["pair"]) + 1:]
        arm = armof.get((r["pair"], prm, r["forced_word"]))
        if arm not in ARMS:
            continue
        n_arm += 1
        by_cell[(r["pair"], prm, r["role"], arm)].append(
            (r["sample_idx"], r["prompt"], r["text"]))
    print("forced rows in stratum %s | in declared arms %s | cells %s"
          % (format(n_rows, ","), format(n_arm, ","), format(len(by_cell), ",")))

    rng = np.random.default_rng(SEED)
    work = []
    for k in sorted(by_cell):
        v = sorted(by_cell[k])
        if len(v) > args.cap:
            idx = sorted(rng.choice(len(v), args.cap, replace=False))
            v = [v[i] for i in idx]
        work.extend((k, s, p, t) for s, p, t in v)
    print("capped sample: %s passages (cap %d/cell)" % (format(len(work), ","), args.cap))

    kept = []
    n_short = 0
    for k, sidx, prm, text in work:
        tr = truncate_to_min_sentences(text, min_words=MIN_WORDS)
        t2 = tr[0] if isinstance(tr, tuple) else tr
        if t2 is None:
            n_short += 1
            continue
        sents = _split_sentences(t2)
        if len(sents) < MIN_SENTS:
            n_short += 1
            continue
        ss = list(sents)
        ss[0] = prm + " " + ss[0]
        kept.append((k, sidx, prm, t2, ss))
    print("after truncation: %s kept, %s dropped" % (format(len(kept), ","),
                                                     format(n_short, ",")))

    emb = _get_embedder(EMBEDDER)
    rows, t0 = [], time.time()
    for i, (k, sidx, prm, t2, ss) in enumerate(kept):
        v = emb.encode(ss, show_progress_bar=False)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
        d = drift_metrics_from_embeddings(v.tolist())
        rows.append({"pair": k[0], "prompt": k[1], "role": k[2], "arm": k[3],
                     "sample_idx": sidx, "total_drift": d["total_drift"],
                     "mean_drift": d["mean_drift"], "n_sents": len(ss)})
        if (i + 1) % 2000 == 0:
            print("  drift %d/%d (%.1f min)" % (i + 1, len(kept), (time.time() - t0) / 60))
    t0 = time.time()
    for i, (k, sidx, prm, t2, ss) in enumerate(kept):
        ps = passage_surprisal(t2, model_name=REF, prompt_prefix=prm)
        s = surprisal_metrics_from_tokens(ps["token_surprisals"])
        rows[i]["mean_surprisal"] = s["mean_surprisal"]
        if (i + 1) % 2000 == 0:
            print("  surprisal %d/%d (%.1f min)" % (i + 1, len(kept), (time.time() - t0) / 60))

    df = pd.DataFrame(rows)
    pq = os.path.join(OUTD, "f15_forced_cells.parquet")
    df.to_parquet(pq)
    print("per-passage metrics persisted: %s rows -> %s"
          % (format(len(df), ","), os.path.basename(pq)))

    out = {"plan": "plans/plan_f15_on_passages.md#F3", "cap": args.cap,
           "n_passages": len(df), "arms": list(ARMS),
           "excluded": [EXCLUDE, FENCED]}

    #: F3a -- does the arm contrast survive forcing? matched arm vs undisturbed
    print("\nF3a: aligned - base gap, per pair, FORCED-MATCHED arm vs UNDISTURBED")
    main = pd.read_parquet(MAIN)
    main = main[~main.pair.str.contains(FENCED)]
    for metric in ("mean_surprisal", "total_drift"):
        res = {}
        mm = (df[df.arm == "matched"].groupby(["pair", "role"])[metric].mean()
              .unstack("role").dropna(subset=["aligned", "base"]))
        r5 = sign_test((mm["aligned"] - mm["base"]).values)
        res["forced_matched"] = r5
        u = (main.groupby(["pair", "role"])[metric].mean()
             .unstack("role").dropna(subset=["aligned", "base"]))
        r6 = sign_test((u["aligned"] - u["base"]).values)
        res["undisturbed"] = r6
        print("  %-15s forced-matched med %+.4f  %d/%d  p %.3g   |   "
              "undisturbed med %+.4f  %d/%d  p %.3g"
              % (metric, r5["median"], r5["up"], r5["dn"], r5["p_sign"],
                 r6["median"], r6["up"], r6["dn"], r6["p_sign"]))
        out["F3a_" + metric] = res

    #: F3b/F3c -- forcing itself, and the DiD
    print("\nF3b: arm - matched, paired per (pair, prompt), within arm")
    for metric in ("mean_surprisal", "total_drift"):
        cell = df.groupby(["pair", "prompt", "role", "arm"])[metric].mean()
        piv = cell.unstack("arm")
        res = {}
        for role in ("aligned", "base"):
            for a1 in ("faller", "riser_matched"):
                if a1 not in piv.columns:
                    continue
                sub = piv.xs(role, level="role").dropna(subset=[a1, "matched"])
                r5 = sign_test((sub[a1] - sub["matched"]).values)
                res["%s:%s" % (role, a1)] = r5
                print("  %-15s %-8s %-14s med %+.4f (mean %+.4f)  %d/%d  p %.3g  (n %d)"
                      % (metric, role, a1 + "-matched", r5["median"], r5["mean"],
                         r5["up"], r5["dn"], r5["p_sign"], r5["n"]))
        a = piv.xs("aligned", level="role").dropna(subset=["faller", "matched"])
        b = piv.xs("base", level="role").dropna(subset=["faller", "matched"])
        ax = (a["faller"] - a["matched"]).rename("a")
        bx = (b["faller"] - b["matched"]).rename("b")
        j = ax.to_frame().join(bx.to_frame(), how="inner")
        r5 = sign_test((j.a - j.b).values)
        res["DiD"] = r5
        print("  %-15s %-8s %-14s med %+.4f (mean %+.4f)  %d/%d  p %.3g  (n %d)"
              % (metric, "F3c", "DiD faller", r5["median"], r5["mean"],
                 r5["up"], r5["dn"], r5["p_sign"], r5["n"]))
        out["F3bc_" + metric] = res

    p = os.path.join(OUTD, "f15_forced.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
