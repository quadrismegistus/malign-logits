#!/usr/bin/env python
"""verify_twp_extraction.py — prove malign_logits/twp.py == scripts/twp_cloud.py.

    scripts/verify_twp_extraction.py --model HuggingFaceTB/SmolLM2-360M

**BOTH MODULES ARE IMPORTED AND RUN SIDE BY SIDE ON THE SAME MODEL.** Not a diff
of source text -- a diff of OUTPUT. A source diff proves the bytes match; this
proves the instrument does, which is the claim that matters and the one a caller
depends on.

The extraction is a PURE MOVE, so the standard here is **bit-identical**, not a
tolerance. RH's tolerance ruling covers the lens validating against STORED cells,
where fp32-vs-producer-dtype legitimately differs. Two copies of the same code on
the same weights in the same process have no such excuse: any difference at all
means the move was not pure.

Checks, in order of what they would catch:

    constants   RULE_VERSION, THETA, MAX_DEPTH, RULE_COMMITS, DICT sha
    boundary    boundary_mask identical over the whole vocabulary
    surface     clean_surface agreeing on every candidate string
    expand      the word distribution, the residual, and the batch count
"""
import argparse, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--prompts", type=int, default=4)
    a = ap.parse_args()

    import numpy as np, torch
    import twp_cloud as OLD
    import malign_logits.twp as NEW

    fails = []
    def chk(name, ok, detail=""):
        print("  [%s] %-40s %s" % ("PASS" if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    # ---- constants ------------------------------------------------------
    for k in ("RULE_VERSION", "THETA", "MAX_DEPTH", "RULE_COMMITS"):
        chk("const %s" % k, getattr(OLD, k) == getattr(NEW, k),
            repr(getattr(NEW, k))[:40])
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    chk("DICT resolves to the same bytes", sha(OLD.DICT) == sha(NEW.DICT),
        sha(NEW.DICT))
    chk("DICT path string identical",
        os.path.realpath(OLD.DICT) == os.path.realpath(NEW.DICT))

    # ---- the model, loaded once and shared ------------------------------
    from transformers import AutoModelForCausalLM
    tok_o, _ = OLD.load_tokenizer(a.model)
    tok_n, _ = NEW.load_tokenizer(a.model)
    chk("load_tokenizer agrees on vocab size",
        len(tok_o) == len(tok_n), "%d" % len(tok_n))
    dev = OLD.pick_device()
    mdl = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, trust_remote_code=True).to(dev).eval()

    n = mdl.config.vocab_size
    bo, bn = OLD.boundary_mask(tok_o, n), NEW.boundary_mask(tok_n, n)
    chk("boundary_mask identical over %d tokens" % n,
        np.array_equal(np.asarray(bo), np.asarray(bn)),
        "%d boundary tokens" % int(np.asarray(bn).sum()))

    strs = [tok_n.convert_ids_to_tokens(i) for i in range(0, n, max(1, n // 4000))]
    diff = [s for s in strs if OLD.clean_surface(s) != NEW.clean_surface(s)]
    chk("clean_surface identical on %d samples" % len(strs), not diff,
        "%d differ" % len(diff))

    # ---- expand, the whole instrument ----------------------------------
    import json
    pop = json.load(open(os.path.join(ROOT, "data", "f11_delta_population.json")))
    prompts = [p["text"] for p in pop["prompts"]][:a.prompts]
    for p in prompts:
        OLD.reset_batch(); NEW.reset_batch()
        wo, ro, co = OLD.expand(mdl, tok_o, p, dev, bo)
        wn, rn, cn = NEW.expand(mdl, tok_n, p, dev, bn)
        same_keys = set(wo) == set(wn)
        same_vals = same_keys and all(wo[k] == wn[k] for k in wo)
        same_res = ro == rn
        chk("expand %r" % p[:26],
            same_keys and same_vals and same_res and co == cn,
            "%d words, residual %s, batches %d" % (len(wn), "eq" if same_res else "DIFF", cn))

    print("\n%s" % ("EXTRACTION VERIFIED: identical on every check"
                    if not fails else "FAILED: %s" % ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
