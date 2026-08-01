"""CONTINUATION-SIDE REGISTER PROBE — do the two markers STEER differently by arm?

    .venv/bin/python scripts/m03_register_continuation.py

WHY, AND WHAT IT IS NOT. `m03_register_probe.py` measured the base model's
PREFERENCE between `should` and `ought to` and found it flat by arm (15 of 36,
p=0.405, 0.035 of a within-arm sd). [1942].2 raised the logical limit, against
its own convenience: **a model can be indifferent between two markers while
producing systematically different continuations after each, differently by
arm** -- and the confound [1882] worried about lives in the CONTINUATION, not
in the marker's own probability.

**THIS DOES NOT REOPEN [1883].2.** That ruling was made on marker preference
and stands on the measurement it was made on. This settles one thing only:
whether the registration carries a scope line saying the marker contrast is
cleared on preference but not on continuation behaviour.

READOUT DECLARED AT [1943].2, BEFORE THE RUN:

    JS GAP FLAT     the markers steer alike in both arms -> THE SCOPE LINE
                    COMES OUT
    JS GAP SHIFTS   they steer differently by arm -> THE SCOPE LINE STAYS,
                    and whether that touches the third-marker call belongs to
                    the pen, not to this file

**THE ASYMMETRY IS DELIBERATE: a flat result REMOVES a caveat and a shifted
result only KEEPS one already written.** A post-hoc measurement whose upside is
a NEW claim is not worth running; one that can only simplify is.

AND THE COST ESTIMATE IT ANSWERS. [1942].2 called this "expensive". It is 144
forward passes -- exactly what the marker probe cost. **A cost estimate is a
claim like any other, and this one nearly retired a live question as
unaffordable.**

METHOD. Both markers leave the model expecting a bare infinitive, so the
compared position is the same part of speech in both branches. Per stem:

    p = P(next token | stem + " should")
    q = P(next token | stem + " ought to")
    JS(p, q)   -- full vocabulary, never truncated to top-k

Paired by scenario x person exactly as the marker probe was, because the
unpaired and paired numbers there had OPPOSITE SIGNS and the pairing is what
made that visible.
"""

import argparse
import math
import os
import statistics
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
BASE = "allenai/Olmo-3-1025-7B"


def js(p, q):
    """Jensen-Shannon divergence in nats, full vocabulary."""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return torch.sum(a[mask] * (torch.log(a[mask]) - torch.log(b[mask]))).item()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def next_dist(model, tok, text, device):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    return torch.softmax(logits, dim=-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from m03_register_probe import stems

    rows = stems()[: args.limit]
    n_inst = sum(1 for r in rows if r[0] == "inst")
    print(f"CONTINUATION PROBE — {args.model}")
    print(f"{len(rows)} stems: {n_inst} institutional, {len(rows)-n_inst} individual\n")
    print("DECLARED AT [1943].2 BEFORE THE RUN: FLAT removes the scope line; "
          "SHIFTED\nkeeps one already written. Neither reopens [1883].2.\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(device).eval()

    by, per_row = {"inst": [], "indiv": []}, []
    for sp, pn, sid, stem in rows:
        p = next_dist(model, tok, stem + " should", device)
        q = next_dist(model, tok, stem + " ought to", device)
        d = js(p, q)
        by[sp].append(d)
        per_row.append((sid, sp, pn, d))

    print(f"{'arm':<8}{'n':<5}{'median JS(should||ought to)':<32}{'mean':<10}{'sd'}")
    for sp in ("indiv", "inst"):
        v = by[sp]
        print(f"{sp:<8}{len(v):<5}{statistics.median(v):<32.4f}"
              f"{statistics.mean(v):<10.4f}{statistics.stdev(v):.4f}")

    pairs = {}
    for sid, sp, pn, d in per_row:
        pairs.setdefault((sid, pn), {})[sp] = d
    both = [v["inst"] - v["indiv"] for v in pairs.values() if len(v) == 2]
    pos = sum(1 for d in both if d > 0)
    n = len(both)
    k = min(pos, n - pos)
    p_sign = min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)
    sd = statistics.stdev(both)
    se = sd / math.sqrt(n)
    print(f"\nPAIRED by scenario x person, n={n}")
    print(f"  {pos} of {n} positive     sign test two-sided p = {p_sign:.3f}")
    print(f"  mean {statistics.mean(both):+.4f}  95% CI "
          f"[{statistics.mean(both)-2.03*se:+.4f}, "
          f"{statistics.mean(both)+2.03*se:+.4f}]")
    print(f"  median {statistics.median(both):+.4f}")
    pooled = statistics.stdev([d for v in by.values() for d in v])
    print(f"  effect {abs(statistics.mean(both))/pooled:.3f} of a within-arm sd "
          f"({pooled:.4f})")

    out = os.path.join(ROOT, "data", "m03_register_continuation.csv")
    with open(out, "w") as f:
        f.write("scenario_id,speaker,person,js_should_vs_ought\n")
        for r in per_row:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nwrote {out} ({len(per_row)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
