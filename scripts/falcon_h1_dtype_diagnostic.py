#!/usr/bin/env python3
"""Why did Falcon-H1-7B produce 5,166 all-NaN logit rows? Ruled [3015].

THE HYPOTHESIS, AND WHY IT IS TESTABLE FOR FREE
-----------------------------------------------
`twp_cloud.py` loads EVERY checkpoint with `torch_dtype=torch.float16`. Both
Falcon-H1 configs declare `torch_dtype: bfloat16` — the models were TRAINED in
bf16, whose exponent range is float32's. **fp16 tops out at 65504 and the SSM
path accumulates over the sequence**, so a hybrid Mamba/attention model run in
fp16 can overflow to inf and thence to NaN where a pure transformer of the same
size does not. That is the standing explanation for why exactly the two
Falcon-H1 checkpoints, and no others, came back empty.

**THE COMPUTE DTYPE AND THE STORAGE DTYPE ARE TWO DECISIONS THAT SHARE A NAME.**
RH's 2026-08-01 ruling — quoted in `twp_cloud.py` — was about the STORE: "the
existing store is MIXED (49 models f16, 87 f32) and a uniform store is worth
more than a marginally more precise one". That ruling is about `_LOGIT["v"] =
lg.half()`, the cast on the way out. It says nothing about how the forward pass
is computed, and a finite bf16 logit casts to f16 losslessly enough for a store
(|logit| is order 10, against f16's 65504). **So loading in bf16 and storing in
f16 keeps RH's ruling exactly and is not a re-litigation of it.**

WHAT THIS SCRIPT DOES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------
One prompt, one checkpoint, three compute dtypes, on whatever device is
available. It reports whether the last-position logit vector is finite, and
where the first non-finite value appears in the layer stack when it is not.

**IT DOES NOT WRITE TO ANY STASH.** A diagnostic that writes is a producer, and
a producer needs a registration. This one prints.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_MODEL = "tiiuae/Falcon-H1-7B-Base"
#: A PROMPT FROM THE ACTUAL BATTERY, not a toy string. "Hello" is one token and
#: exercises almost nothing; the failure is hypothesised to accumulate over a
#: sequence, so a one-token probe could come back clean on a genuinely broken
#: configuration and be read as a fix.
DEFAULT_PROMPT = "The man reached out and touched her"


def probe(mid, prompt, dtype_name, device, want_hidden):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=dt, trust_remote_code=True).to(device).eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=want_hidden)
    lg = out.logits[0, -1, :].float()

    n = int(torch.isnan(lg).sum())
    i = int(torch.isinf(lg).sum())
    finite = bool(torch.isfinite(lg).all())
    rep = {"dtype": dtype_name, "finite": finite, "n_nan": n, "n_inf": i,
           "vocab": int(lg.shape[0])}
    if finite:
        rep["max"] = float(lg.max())
        rep["min"] = float(lg.min())
        top = torch.topk(lg, 5)
        rep["top5"] = [(tok.decode([int(t)]), round(float(v), 2))
                       for v, t in zip(top.values, top.indices)]
        #: THE STORAGE CAST, CHECKED RATHER THAN ASSUMED. The whole proposal is
        #: "compute in bf16, store in f16". If the f16 cast of a finite bf16
        #: vector were itself non-finite the proposal would be empty, so the
        #: claim is measured here instead of argued.
        h = lg.half()
        rep["survives_f16_cast"] = bool(torch.isfinite(h).all())
        rep["max_abs"] = round(float(lg.abs().max()), 2)
    elif want_hidden:
        #: WHERE it breaks, not just THAT it breaks. If layer 0 is already
        #: non-finite the cause is the embedding or the input, not accumulation,
        #: and the bf16 story would be wrong.
        for li, hs in enumerate(out.hidden_states):
            if not bool(torch.isfinite(hs.float()).all()):
                rep["first_bad_hidden_layer"] = li
                rep["n_hidden_layers"] = len(out.hidden_states) - 1
                break
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--dtypes", default="float16,bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--hidden", action="store_true",
                    help="locate the first non-finite layer (more memory)")
    a = ap.parse_args()

    import torch
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model  {a.model}\ndevice {dev}\nprompt {a.prompt!r}\n")

    reps = []
    for dn in a.dtypes.split(","):
        try:
            r = probe(a.model, a.prompt, dn.strip(), dev, a.hidden)
        except Exception as e:
            r = {"dtype": dn.strip(), "ERROR": f"{type(e).__name__}: {str(e)[:160]}"}
        reps.append(r)
        print(f"  {r}", flush=True)

    print()
    ok = {r["dtype"] for r in reps if r.get("finite")}
    bad = {r["dtype"] for r in reps if r.get("finite") is False}
    if bad and ok:
        print(f"  DIAGNOSIS: non-finite under {sorted(bad)}, finite under "
              f"{sorted(ok)} -- ON THIS PROMPT, ON THIS DEVICE.")
        print("  A single prompt establishes that the dtype MATTERS. It does "
              "not establish that the working dtype works on all 2,583.")
    elif bad and not ok:
        print(f"  NON-FINITE UNDER EVERY DTYPE TRIED ({sorted(bad)}). The dtype "
              "hypothesis does not survive; look at the device or the "
              "architecture support, not the precision.")
    elif ok and not bad:
        print(f"  FINITE UNDER EVERY DTYPE TRIED ({sorted(ok)}). **This device "
              "does not reproduce the cloud failure**, so it cannot confirm the "
              "cause -- the failure was on an A100 under CUDA kernels this box "
              "does not run. A non-reproduction is not an exoneration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
