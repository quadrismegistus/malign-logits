#!/usr/bin/env python
"""twp_layers_time.py — time twp-across-layers on one prompt, one model, locally.

    scripts/twp_layers_time.py --model meta-llama/Llama-3.1-8B \
        --prompt "She was so angry she wanted to"

**LOAD IS TIMED SEPARATELY FROM COMPUTE, because they scale differently and are
paid differently.** Load is once per model and is dominated by disk; the
expansion is once per prompt and is dominated by the card. A single wall-clock
number for "how long does a prompt take" hides which of the two a fleet is
actually buying, and the answer decides whether to shard by model or by prompt.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--prompt", default="She was so angry she wanted to")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--stride", type=int, default=1,
                    help="read every Nth layer (1 = all)")
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from malign_logits import twp

    print("model  %s\nprompt %r\ndtype  %s\n" % (a.model, a.prompt, a.dtype))

    t = time.perf_counter()
    tok, _ = twp.load_tokenizer(a.model)
    t_tok = time.perf_counter() - t

    dev = twp.pick_device()
    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(dev).eval()
    t_load = time.perf_counter() - t

    t = time.perf_counter()
    bmask = twp.boundary_mask(tok, model.config.vocab_size)
    t_mask = time.perf_counter() - t

    n_hs = model.config.num_hidden_layers + 1
    layers = list(range(0, n_hs, a.stride))
    if layers[-1] != n_hs - 1:
        layers.append(n_hs - 1)

    #: baseline: plain twp, final layer only -- the thing the corpus already holds
    twp.reset_batch()
    t = time.perf_counter()
    w1, r1, calls = twp.expand(model, tok, a.prompt, dev, bmask)
    t_twp = time.perf_counter() - t

    twp.reset_batch()
    t = time.perf_counter()
    out, st = twp.expand_layers(model, tok, a.prompt, dev, bmask, layers)
    t_all = time.perf_counter() - t

    print("TIMINGS  (device %s)" % dev)
    print("  tokenizer load          %7.2f s" % t_tok)
    print("  MODEL LOAD              %7.2f s   <- once per model" % t_load)
    print("  boundary mask           %7.2f s" % t_mask)
    print("  twp, final layer only   %7.2f s   %d words, %d batched calls"
          % (t_twp, len(w1), calls))
    print("  twp ACROSS %2d LAYERS    %7.2f s   %.1fx twp   <- once per prompt"
          % (len(layers), t_all, t_all / max(t_twp, 1e-9)))
    print("  union prefixes %d over %d depths %s"
          % (st["union_prefixes"], st["passes"], st["union_sizes"]))
    print("  head(hidden[-1]) vs logits, max prob diff: %.2e" % st["head_err"])

    fin = st["layers"][-1]
    same = set(out[fin][0]) == set(w1)
    print("\n  FINAL LAYER REPRODUCES twp: %s (%d vs %d words)"
          % (same, len(out[fin][0]), len(w1)))

    print("\nWHAT EACH LAYER SAYS")
    for l in st["layers"]:
        w, _res = out[l]
        top = sorted(w.items(), key=lambda kv: -kv[1])[:a.top]
        print("  L%-3d %s" % (l, "  ".join("%s %.3f" % (k[0], v) for k, v in top)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
