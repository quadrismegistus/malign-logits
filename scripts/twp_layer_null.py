#!/usr/bin/env python
"""twp_layer_null.py — is a word's excursion through the stack LARGE?

    scripts/twp_layer_null.py --model meta-llama/Llama-3.1-8B \
        --prompt "She was so angry she wanted to" --watch punch kill scream

**THE NULL IS ALREADY IN THE DATA AND I NEARLY PROPOSED BUYING IT SEPARATELY.**
`expand_layers` expands, at every layer, every token above theta AT THAT LAYER --
so the union across layers is every word any layer thought probable, each with a
per-layer trajectory. Asking "does `punch` move a lot between L27 and L32" is
therefore answerable from one run: compute the same excursion for every word in
the union and see where `punch` falls in that distribution. It is a column, not
an experiment.

**THE CENSORING IS INFORMATIVE AND IT BIASES TOWARD SIGNIFICANCE.** Per-layer
pruning is preserved by design, so a word above theta at one layer and below it
at another is CENSORED there, not measured. The words with complete trajectories
are the ones that stayed probable, which makes the null tighter than the truth --
so a test word looks MORE exceptional than it is. Both the complete-case null and
the censored count are reported, because quoting the first without the second
would be the reassuring half.
"""
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


def excursion(p_a, p_b):
    """log10 ratio between two layers. Symmetric, scale-free, sign-carrying."""
    return math.log10(max(p_b, 1e-12) / max(p_a, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--prompt", default="She was so angry she wanted to")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--from-layer", type=int, default=27)
    ap.add_argument("--to-layer", type=int, default=-1)
    ap.add_argument("--watch", nargs="*", default=["punch", "kill", "scream", "hit"])
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from malign_logits import twp

    tok, _ = twp.load_tokenizer(a.model)
    dev = twp.pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), trust_remote_code=True).to(dev).eval()
    bmask = twp.boundary_mask(tok, model.config.vocab_size)
    n_hs = model.config.num_hidden_layers + 1
    layers = list(range(n_hs))
    t0 = time.perf_counter()
    out, st = twp.expand_layers(model, tok, a.prompt, dev, bmask, layers)
    took = time.perf_counter() - t0

    L_a = a.from_layer if a.from_layer >= 0 else n_hs + a.from_layer
    L_b = a.to_layer if a.to_layer >= 0 else n_hs + a.to_layer

    def dist(l):
        d = {}
        for (surf, _t1), p in out[l][0].items():
            d[surf] = d.get(surf, 0.0) + p
        return d
    A, B = dist(L_a), dist(L_b)

    both = sorted(set(A) & set(B))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    ex = {w: excursion(A[w], B[w]) for w in both}

    print("model %s\nprompt %r\nlayers L%d -> L%d   (%.1f s, %d union prefixes)\n"
          % (a.model, a.prompt, L_a, L_b, took, st["union_prefixes"]))
    print("THE NULL: every word in the union, not a chosen vocabulary")
    print("  words at L%-2d              %4d" % (L_a, len(A)))
    print("  words at L%-2d              %4d" % (L_b, len(B)))
    print("  COMPLETE (both layers)   %4d   <- the null is computed on these"
          % len(both))
    print("  censored: fell below θ   %4d" % len(only_a))
    print("  censored: rose above θ   %4d" % len(only_b))
    print("  ** censoring is %.0f%% of L%d's words, and it biases the null TIGHT **"
          % (100.0 * len(only_a) / max(1, len(A)), L_a))

    vals = sorted(ex.values())
    def pct(v):
        return 100.0 * sum(1 for x in vals if x <= v) / len(vals)
    q = lambda f: vals[min(len(vals) - 1, int(f * len(vals)))]
    print("\n  excursion log10(p_L%d / p_L%d) over the %d complete words:"
          % (L_b, L_a, len(vals)))
    print("     min %+.2f   p10 %+.2f   median %+.2f   p90 %+.2f   max %+.2f"
          % (vals[0], q(.10), q(.50), q(.90), vals[-1]))
    print("     |excursion| median %.2f  (a typical word moves %.1fx)"
          % (sorted(abs(v) for v in vals)[len(vals)//2],
             10 ** sorted(abs(v) for v in vals)[len(vals)//2]))

    print("\n  WATCHED WORDS, placed IN that distribution:")
    print("    %-10s %9s %9s %8s %10s" % ("word", "p@L%d"%L_a, "p@L%d"%L_b,
                                          "log10", "percentile"))
    for w in a.watch:
        pa, pb = A.get(w), B.get(w)
        if pa is None and pb is None:
            print("    %-10s %9s %9s      absent at both layers" % (w, "-", "-")); continue
        if pa is None or pb is None:
            print("    %-10s %9s %9s      CENSORED — cannot be placed"
                  % (w, "%.4f"%pa if pa else "<θ", "%.4f"%pb if pb else "<θ")); continue
        e = ex[w]
        print("    %-10s %9.4f %9.4f %+8.2f %9.0f%%" % (w, pa, pb, e, pct(e)))

    print("\n  READING: a watched word is only interesting if its percentile is")
    print("  extreme. Mid-distribution means the stack does this to everything.")
    if a.json:
        json.dump({"model": a.model, "prompt": a.prompt, "L_a": L_a, "L_b": L_b,
                   "excursions": ex, "censored_down": only_a,
                   "censored_up": only_b}, open(a.json, "w"), indent=1)
        print("\n  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
