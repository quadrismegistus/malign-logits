#!/usr/bin/env python
"""verify_twp_layers.py — the multi-layer driver against the single-layer one.

    scripts/verify_twp_layers.py --model allenai/OLMo-2-0425-1B [--ref <sha>]

Three claims, in the order that a failure would matter:

    1. THE REFACTOR IS INERT.  `expand` after the boundary rule was factored
       into `_boundary_for` / `_account` produces bit-identical output to
       `expand` before it.  Checked against a REVISION, materialised from git,
       not against a copy someone remembered to keep.

    2. LAYER -1 IS THE DEFAULT PATH.  `expand_layers(layers=[-1])` reproduces
       `expand()` exactly -- same words, same masses, same four-way residual.
       This is the free validation: the final layer's word distribution IS the
       stored twp cell, so if it moves, the readout is wrong and no interior
       layer is readable either.

    3. THE SHARING IS REAL.  Report union prefixes against solo prefixes for a
       full-stack run.  The cost argument for the whole fleet is that a
       per-layer lens costs one set of forward passes rather than n_layers of
       them; that ratio is the measurement of it, not the assertion.

**WHY A GIT REF AND NOT A SECOND MODULE.** `scripts/verify_twp_extraction.py`
compared `twp_cloud` against `malign_logits.twp` -- which was a real check the
day it was written and became a TAUTOLOGY the moment twp_cloud was converted to
import the package. It now compares the module to itself and passes for free.
A ref-based baseline cannot rot that way: whatever `--ref` names is what the
instrument used to be, and the check keeps meaning as long as git does.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

#: the baseline module is written INTO the package directory on purpose: `DICT`
#: resolves relative to __file__, so a copy anywhere else silently reads a
#: different dictionary -- or none -- and the CJK rule would differ for a reason
#: that has nothing to do with the refactor under test.
BASELINE = os.path.join(ROOT, "malign_logits", "_twp_baseline.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--ref", default="53079ce2",
                    help="revision of malign_logits/twp.py to compare against")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()

    import numpy as np, torch
    from transformers import AutoModelForCausalLM

    src = subprocess.run(["git", "show", "%s:malign_logits/twp.py" % a.ref],
                         cwd=ROOT, capture_output=True, text=True)
    if src.returncode:
        print("cannot read %s: %s" % (a.ref, src.stderr.strip())); return 2
    open(BASELINE, "w").write(src.stdout)

    fails = []
    def chk(name, ok, detail=""):
        print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    try:
        import malign_logits.twp as NEW
        import malign_logits._twp_baseline as OLD

        tok, _ = NEW.load_tokenizer(a.model)
        dev = NEW.pick_device()
        mdl = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=getattr(torch, a.dtype),
            trust_remote_code=True).to(dev).eval()
        bmask = NEW.boundary_mask(tok, mdl.config.vocab_size)

        pop = json.load(open(os.path.join(ROOT, "data",
                                          "f11_delta_population.json")))
        prompts = [p["text"] for p in pop["prompts"]][:a.prompts]

        # ---- 1. the refactor is inert -----------------------------------
        print("\n1. REFACTOR INERT  (%s -> working tree)" % a.ref[:8])
        for p in prompts:
            OLD.reset_batch(); wo, ro, co = OLD.expand(mdl, tok, p, dev, bmask)
            NEW.reset_batch(); wn, rn, cn = NEW.expand(mdl, tok, p, dev, bmask)
            same = (set(wo) == set(wn) and all(wo[k] == wn[k] for k in wo)
                    and ro == rn and co == cn)
            #: KEY ORDER TOO, not just the values -- the residual dict is
            #: json.dump'd into the transport line, so a reordered dict is a
            #: changed artifact even when every number matches.
            order = list(ro.keys()) == list(rn.keys())
            chk("expand %r" % p[:30], same and order,
                "%d words | residual %s | keys %s" %
                (len(wn), "eq" if ro == rn else "DIFF",
                 "eq" if order else "REORDERED"))

        # ---- 2. layer -1 is the default path ----------------------------
        print("\n2. LAYER -1 == DEFAULT PATH")
        for p in prompts:
            NEW.reset_batch(); w1, r1, _ = NEW.expand(mdl, tok, p, dev, bmask)
            NEW.reset_batch()
            out, st = NEW.expand_layers(mdl, tok, p, dev, bmask, [-1])
            w2, r2 = out[st["layers"][0]]
            same = set(w1) == set(w2) and all(w1[k] == w2[k] for k in w1)
            res = all(r1[k] == r2[k] for k in
                      ("tail", "drop", "open", "mojibake", "total"))
            chk("expand_layers([-1]) %r" % p[:22], same and res,
                "%d words | residual %s | head_err %.2e" %
                (len(w2), "eq" if res else "DIFF", st["head_err"]))

        # ---- 3. the sharing, measured -----------------------------------
        print("\n3. FULL STACK: what the sharing is worth")
        n_hs = mdl.config.num_hidden_layers + 1
        layers = list(range(0, n_hs))
        p = prompts[0]
        #: WALL TIME, NOT JUST PREFIX COUNTS. Prefixes price the forward passes;
        #: they do NOT price the readouts, and the readouts are not free -- each
        #: is a (chunk x d) @ (d x V) matmul, so n_layers of them can cost more
        #: than the forward that produced the hidden states. Only a clock sees
        #: that, and only the clock is a budget.
        import time
        NEW.reset_batch(); t0 = time.perf_counter()
        NEW.expand(mdl, tok, p, dev, bmask)
        t_twp = time.perf_counter() - t0
        NEW.reset_batch(); t0 = time.perf_counter()
        out, st = NEW.expand_layers(mdl, tok, p, dev, bmask, layers)
        t_all = time.perf_counter() - t0
        ratio = st["solo_prefixes"] / max(1, st["union_prefixes"])
        final = st["layer_prefixes"][st["n_hidden"] - 1]
        print("    prompt      %r" % p[:56])
        print("    layers      %d (hidden_states is %d long)"
              % (len(layers), st["n_hidden"]))
        print("    union       %d prefixes over %d depths  %s"
              % (st["union_prefixes"], st["passes"], st["union_sizes"]))
        print("    layer-by-   %d prefixes summed over %d layers"
              % (st["solo_prefixes"], len(layers)))
        print("    SHARING     %.2fx  (union against layer-by-layer)" % ratio)
        print("    plain twp   %d prefixes (final layer alone)" % final)
        print("    COST        %.2fx twp by prefixes | %.2fx BY THE CLOCK"
              "  <-- THE BUDGET NUMBER"
              % (st["cost_vs_twp"], t_all / max(1e-9, t_twp)))
        print("    wall        twp %.2fs -> all-layer %.2fs" % (t_twp, t_all))
        #: the per-layer profile, because the cost is not spread evenly and the
        #: expensive layers are the ones whose words are least readable
        prof = st["layer_prefixes"]
        print("    per layer   %s" % "  ".join(
            "%d:%d" % (l, prof[l]) for l in layers))
        chk("union <= solo", st["union_prefixes"] <= st["solo_prefixes"])
        chk("sharing exceeds 1x", ratio > 1.0, "%.2fx" % ratio)

        # ---- what it reads, so the numbers have a referent ---------------
        print("\n    top word by layer (mass, and the censoring):")
        top = sorted(out[layers[-1]][0].items(), key=lambda kv: -kv[1])[:1]
        watch = [k for k, _ in top]
        for l in layers[::max(1, len(layers) // 8)] + [layers[-1]]:
            w, r = out[l]
            best = sorted(w.items(), key=lambda kv: -kv[1])[:3]
            got = w.get(watch[0])
            print("      layer %-3d %-44s   %s %s"
                  % (l, "  ".join("%s %.3f" % (k[0], v) for k, v in best),
                     watch[0][0], "%.4f" % got if got is not None
                     else "< theta"))
    finally:
        if os.path.exists(BASELINE):
            os.remove(BASELINE)

    print("\n%s" % ("VERIFIED: %d checks, no differences" % (
        3 * a.prompts + 2) if not fails else "FAILED: %s" % ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
