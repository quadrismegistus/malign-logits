"""Is MPS safe for BLT, and is it faster? CPU is the referee.

    uv run python meta/M06_generation/scripts/m06_blt_mps_check.py
    -> results/blt_mps_check.json

THE PRIOR HAZARD IS SPECIFIC AND DOCUMENTED. This campaign found MPS corrupting
`bge-m3` embeddings of SHORT CHINESE strings: deterministic, self-consistent
under its own path, and therefore invisible to any check that does not compare
against CPU. So this does not ask "does MPS run" -- it will -- but "does MPS
agree with CPU, at the lengths and scripts where the known failure lives".

Probes are chosen to include that failure region rather than to flatter the
device: short Chinese first, then long Chinese, then English at both lengths.
A device that agrees on 1,000-byte English and diverges on a 20-byte Chinese
sentence would pass a naive benchmark and fail the corpus.

Speed is measured on the SAME passages in the same process, so the comparison is
not across runs with different thermal or memory conditions.
"""
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


def ch_rows(q):
    o = subprocess.run([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                       capture_output=True, text=True).stdout.strip()
    return [json.loads(l) for l in o.split("\n") if l]


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not torch.backends.mps.is_available():
        print("MPS not available; nothing to compare.")
        return 0
    torch.set_num_threads(6)
    tk = AutoTokenizer.from_pretrained(BLT, trust_remote_code=True)
    tk.model_max_length = 4096
    model = AutoModelForCausalLM.from_pretrained(
        BLT, trust_remote_code=True, dtype=torch.float32).eval()

    probes = [("zh short", "她把兔子攥在手里，它一动不动。"),
              ("zh v.short", "今天天气很好。"),
              ("en short", "She squeezed the rabbit in her grip and it went still.")]
    for lang, pred in (("zh", "length(text) != lengthUTF8(text)"),
                       ("en", "length(text) = lengthUTF8(text)")):
        for r in ch_rows("SELECT text FROM malign_logits.gen_sequences WHERE "
                         "corpus='f11_l2' AND forced_word='' AND %s AND "
                         "length(text) BETWEEN 600 AND 1400 LIMIT 3" % pred):
            probes.append(("%s long" % lang, r["text"]))

    def score(dev):
        model.to(dev)
        out, t0 = [], time.time()
        nb = 0
        for _, t in probes:
            ids = tk(t, add_special_tokens=False)["input_ids"]
            x = torch.tensor([ids]).to(dev)
            with torch.no_grad():
                lg = model(x).logits[0]
            lp = torch.log_softmax(lg.float(), -1)
            idx = torch.tensor(ids[1:]).to(dev)
            s = (-lp[:-1].gather(1, idx[:, None]).squeeze(1)).cpu().numpy().astype(np.float64)
            out.append(s)
            nb += len(ids)
        if dev == "mps":
            torch.mps.synchronize()
        return out, time.time() - t0, nb

    cpu, t_cpu, nb = score("cpu")
    mps, t_mps, _ = score("mps")
    model.to("cpu")

    rows, worst = [], 0.0
    for (lab, t), a, b in zip(probes, cpu, mps):
        d = float(np.abs(a - b).max())
        bpb_a = float(a.sum() / np.log(2) / len(t.encode()))
        bpb_b = float(b.sum() / np.log(2) / len(t.encode()))
        worst = max(worst, abs(bpb_a - bpb_b))
        rows.append({"probe": lab, "bytes": len(t.encode()),
                     "max_abs_token_diff": d, "bits_per_byte_cpu": bpb_a,
                     "bits_per_byte_mps": bpb_b,
                     "bits_per_byte_diff": bpb_a - bpb_b})
        print("  %-10s %5d B   max|d| per byte %.2e   bits/byte cpu %.4f mps %.4f  (d %+.2e)"
              % (lab, len(t.encode()), d, bpb_a, bpb_b, bpb_a - bpb_b))

    #: the tolerance is set against the SMALLEST EFFECT THIS CAMPAIGN MEASURES
    #: (the ordering contrast, 2-4e-3 nats), not against float32 epsilon.
    TOL = 1e-3
    verdict = "MPS AGREES" if worst < TOL else "MPS DIVERGES"
    print("\n%s: worst bits/byte disagreement %.2e (tolerance %.0e)"
          % (verdict, worst, TOL))
    print("speed: cpu %.1f B/s, mps %.1f B/s -> %.2fx"
          % (nb / t_cpu, nb / t_mps, t_cpu / t_mps))

    out = {"ref": BLT, "verdict": verdict, "worst_bits_per_byte_diff": worst,
           "tolerance": TOL, "cpu_bytes_per_s": nb / t_cpu,
           "mps_bytes_per_s": nb / t_mps, "speedup": t_cpu / t_mps,
           "probes": rows}
    os.makedirs(OUTD, exist_ok=True)
    p = os.path.join(OUTD, "blt_mps_check.json")
    json.dump(out, open(p, "w"), indent=1)
    print("-> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
