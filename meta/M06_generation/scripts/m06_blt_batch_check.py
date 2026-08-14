"""Does batching BLT on CPU pay, and does it change the answer?

    uv run python meta/M06_generation/scripts/m06_blt_batch_check.py [--n 24]
    -> results/blt_batch_check.json

Two questions, and the second is the one that matters. Batching a causal LM
requires PADDING, and padding only leaves the result untouched if the mask is
right and the model honours it. A wrong mask still returns finite, ordered,
plausible surprisals -- the same shape as the MPS failure, where 2.9035 sat
where 1.9955 belonged and looked like a number.

So this measures speed AND agreement against the one-at-a-time path, with
sequential CPU as the referee, exactly as CPU was the referee for MPS.

Padding side matters for a causal model: left-padding puts pad tokens BEFORE
real content, so every real position's relative offset shifts. This tries the
padding the tokenizer declares and reports which it used rather than assuming.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
BLT = "itazap/blt-1b-hf"
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
TOL = 1e-3          # against the campaign's smallest effect, not float epsilon


def ch_rows(q):
    o = subprocess.run([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                       capture_output=True, text=True).stdout.strip()
    return [json.loads(l) for l in o.split("\n") if l]


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    tk = AutoTokenizer.from_pretrained(BLT, trust_remote_code=True)
    tk.model_max_length = 4096
    model = AutoModelForCausalLM.from_pretrained(
        BLT, trust_remote_code=True, dtype=torch.float32).eval()

    texts = []
    for pred in ("length(text) != lengthUTF8(text)", "length(text) = lengthUTF8(text)"):
        texts += [r["text"] for r in ch_rows(
            "SELECT text FROM malign_logits.gen_sequences WHERE corpus='f11_l2' "
            "AND forced_word='' AND %s AND length(text) BETWEEN 300 AND 1400 "
            "ORDER BY cityHash64(text) LIMIT %d" % (pred, a.n // 2))]
    ids_all = [tk(t, add_special_tokens=False)["input_ids"] for t in texts]
    lens = [len(i) for i in ids_all]
    print("%d passages, %d-%d bytes (pad waste at batch %d would be %.0f%%)"
          % (len(texts), min(lens), max(lens), a.batch,
             100 * (1 - sum(lens) / (len(lens) * max(lens)))))

    def surp_from_logits(lg, ids):
        lp = torch.log_softmax(lg.float(), -1)
        idx = torch.tensor(ids[1:])
        return (-lp[:len(ids) - 1].gather(1, idx[:, None]).squeeze(1)).numpy().astype(np.float64)

    #: SEQUENTIAL, the referee
    t0 = time.time()
    seq = []
    for ids in ids_all:
        with torch.no_grad():
            lg = model(torch.tensor([ids])).logits[0]
        seq.append(surp_from_logits(lg, ids))
    t_seq = time.time() - t0

    #: BATCHED with right padding: pads sit AFTER real content, so every real
    #: position keeps its offset. Left padding would shift them all.
    pad_id = tk.pad_token_id if tk.pad_token_id is not None else 0
    t0 = time.time()
    bat = [None] * len(ids_all)
    order = sorted(range(len(ids_all)), key=lambda i: lens[i])   # length-sort cuts pad waste
    for s in range(0, len(order), a.batch):
        chunk = order[s:s + a.batch]
        mx = max(lens[i] for i in chunk)
        x = torch.full((len(chunk), mx), pad_id, dtype=torch.long)
        am = torch.zeros((len(chunk), mx), dtype=torch.long)
        for r, i in enumerate(chunk):
            x[r, :lens[i]] = torch.tensor(ids_all[i])
            am[r, :lens[i]] = 1
        with torch.no_grad():
            out = model(x, attention_mask=am).logits
        for r, i in enumerate(chunk):
            bat[i] = surp_from_logits(out[r], ids_all[i])
    t_bat = time.time() - t0

    worst, worst_bpb, rows = 0.0, 0.0, []
    for t, s_, b_ in zip(texts, seq, bat):
        d = float(np.abs(s_ - b_).max())
        nb = len(t.encode())
        dbpb = abs(float(s_.sum() - b_.sum()) / np.log(2) / nb)
        worst = max(worst, d)
        worst_bpb = max(worst_bpb, dbpb)
        rows.append({"bytes": nb, "max_abs_token_diff": d, "bits_per_byte_diff": dbpb})
    verdict = "BATCHING AGREES" if worst_bpb < TOL else "BATCHING DIVERGES"
    print("\n%s: worst per-byte |d| %.2e, worst bits/byte |d| %.2e (tol %.0e)"
          % (verdict, worst, worst_bpb, TOL))
    print("speed: sequential %.1f B/s, batched(%d) %.1f B/s -> %.2fx"
          % (sum(lens) / t_seq, a.batch, sum(lens) / t_bat, t_seq / t_bat))

    out = {"ref": BLT, "verdict": verdict, "batch": a.batch, "n": len(texts),
           "worst_token_diff": worst, "worst_bits_per_byte_diff": worst_bpb,
           "tolerance": TOL, "seq_bytes_per_s": sum(lens) / t_seq,
           "batched_bytes_per_s": sum(lens) / t_bat, "speedup": t_seq / t_bat,
           "pad_token_id": pad_id, "padding": "right", "rows": rows}
    os.makedirs(OUTD, exist_ok=True)
    p = os.path.join(OUTD, "blt_batch_check.json")
    json.dump(out, open(p, "w"), indent=1)
    print("-> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
