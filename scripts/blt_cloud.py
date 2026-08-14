#!/usr/bin/env python3
"""BLT byte-level surprisal over the exported passage set, shardable for a fleet.

    python scripts/blt_cloud.py --input blt_passages.jsonl.gz --out /workspace/blt \
        --shard 0 --of 4 [--limit N] [--dtype float32]

WHY THIS EXISTS. `m06_blt_surprisal.py` is lacan's PILOT: it reads the corpus
with `--n` per language and has no input file or shard flag. The exported
`data/raw/blt_passages.jsonl.gz` (483,085 deduped passages, 420 MiB of text) has
no consumer. This is that consumer, and the scoring is lifted VERBATIM from the
pilot rather than reimplemented -- a sibling implementation is not the source
(campaign rule, [5697]).

    ids = tk(text, add_special_tokens=False)["input_ids"]
    lg  = model(torch.tensor([ids])).logits[0]
    lp  = torch.log_softmax(lg.float(), -1)
    sur = -lp[:-1].gather(1, ids[1:][:, None]).squeeze(1)

DTYPE IS A LOGIT DIFFERENCE and the pilot ran float32, so this defaults to
float32 and prints what it used. BLT 1B in fp32 is ~4 GB and fits any 24 GB card
with room for batching; there is no reason to introduce a dtype delta to save
memory we are not short of.

OUTPUT, mirroring the twp pattern: one .jsonl of per-passage summaries plus a
.f32 sidecar holding the per-token surprisal arrays, with each jsonl row
carrying {row, n} into it. The arrays are what `ref_surprisal` stores; the
summary is what the analysis reads. Resume is by readback of the jsonl, so a
killed shard restarts where it stopped.
"""
import argparse, gzip, json, os, sys, time
import numpy as np

BLT = "itazap/blt-1b-hf"


def done_keys(path):
    """(prompt, text) already scored in this shard's own output."""
    got = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                got.add((r.get("prompt"), r.get("text_sha")))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="/workspace/blt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    a = ap.parse_args()

    import torch, hashlib
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}[a.dtype]
    print("device %s | compute dtype %s | shard %d/%d" % (dev, a.dtype, a.shard, a.of),
          flush=True)

    tk = AutoTokenizer.from_pretrained(BLT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BLT, trust_remote_code=True, dtype=dt).eval().to(dev)

    os.makedirs(a.out, exist_ok=True)
    jl = os.path.join(a.out, "blt_shard%02d.jsonl" % a.shard)
    fb = os.path.join(a.out, "blt_shard%02d.f32" % a.shard)
    done = done_keys(jl)
    #: ROW COUNTER FROM THE FILE'S OWN SIZE, never a remembered count -- the
    #: defect twp_cloud.py fixed for its .f16 and the same one applies here.
    row = os.path.getsize(fb) // 4 if os.path.exists(fb) else 0
    print("resuming: %d already scored, %d floats in the sidecar" % (len(done), row),
          flush=True)

    n_seen = n_new = 0
    t0 = time.time()
    with gzip.open(a.input, "rt") as src, open(jl, "a") as out, open(fb, "ab") as sb:
        for i, line in enumerate(src):
            if i % a.of != a.shard:
                continue
            r = json.loads(line)
            n_seen += 1
            if a.limit and n_seen > a.limit:
                break
            text, prompt = r["text"], r["prompt"]
            sha = hashlib.sha256(text.encode()).hexdigest()[:16]
            if (prompt, sha) in done:
                continue
            ids = tk(text, add_special_tokens=False)["input_ids"]
            if len(ids) < 2:
                continue
            with torch.no_grad():
                lg = model(torch.tensor([ids], device=dev)).logits[0]
            lp = torch.log_softmax(lg.float(), -1)
            idx = torch.tensor(ids[1:], device=dev)
            sur = (-lp[:-1].gather(1, idx[:, None]).squeeze(1)).cpu().numpy().astype(np.float32)
            sb.write(sur.tobytes())
            nb = len(text.encode())
            out.write(json.dumps({
                "prompt": prompt, "text_sha": sha, "script": r.get("script"),
                "corpora": r.get("corpora"), "n_bytes": nb, "n_chars": r.get("n_chars"),
                "n_tokens": len(ids), "row": row, "n": int(sur.size),
                "bits_per_byte": float(sur.sum() / np.log(2) / nb),
                "ref": BLT, "dtype": a.dtype}) + "\n")
            row += int(sur.size)
            n_new += 1
            if n_new % 200 == 0:
                out.flush(); sb.flush()
                el = (time.time() - t0) / 60
                print("  %d scored  %.1f min  %.1f/s" % (n_new, el, n_new / max(el * 60, 1)),
                      flush=True)
    print("shard %d done: %d seen, %d newly scored, %.1f min"
          % (a.shard, n_seen, n_new, (time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
