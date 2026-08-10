#!/usr/bin/env python
"""twp_patch_depth.py — AT WHAT DEPTH IS ALIGNMENT'S WORK ALREADY DONE?

    scripts/twp_patch_depth.py --base meta-llama/Llama-3.1-8B \
        --aligned meta-llama/Llama-3.1-8B-Instruct

A CAUSAL measure, because the observational ones are confounded and we have now
hit every confound in turn:

    the lens        head(norm(h_L)) mid-stack is out of distribution
    the head        no pair in the roster froze it; 100% of rows move
    accumulation    the gap at layer L contains everything done at layers < L,
                    so onset-in-curve is not onset-in-mechanism (lacan [5222].1)

Patching answers a counterfactual instead of describing a curve. Run the BASE
model, replace its residual stream at depth L with the ALIGNED model's, continue
through the base's remaining blocks AND THE BASE'S HEAD, and ask how much of the
aligned model's behaviour you recover.

    recovery(L) = median over words of
                  (log p_patched - log p_base) / (log p_aligned - log p_base)

    0 = patching changed nothing; the aligned behaviour is not present at L
    1 = patching reproduced it entirely; alignment's work is DONE by L

**IT IS NOT CONFOUNDED BY ACCUMULATION.** A curve asks where a difference
becomes visible; this asks where the information is SUFFICIENT. If recovery is
already 0.8 at layer 10, alignment's intervention is deep whatever the readout
shows -- and if recovery stays ~0 until layer 30, it is shallow whatever the
weights did.

**THE BASE'S HEAD IS USED THROUGHOUT**, so the readout is held fixed by
construction and the frozen-head problem does not arise.

**AND ||dh_L||/||h_L|| IS MEASURED HERE RATHER THAN CITED.** F05-revised claims
hidden states are "nearly identical between base and aligned through 97% of the
network". That document is graded D and rescoped, its revision contradicted its
own original, and this project should not take it on faith. The number is free
once both forward passes exist.

CAVEAT, stated because it is the assumption the design rests on: patching across
two checkpoints assumes their residual bases are comparable. For a fine-tune
that is the initialisation, not an assumption (lacan [5222].1) -- it would NOT
hold across families.
"""
import argparse, json, math, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="She was so angry she wanted to")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rule", default="canonical", choices=("canonical", "lens"))
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits import movement as MV

    rule = MV.CANONICAL if a.rule == "canonical" else MV.LENS
    cell = Step(Checkpoint(a.base), Checkpoint(a.aligned)).cell(a.prompt)
    m = cell.movement(rule)
    words = list(m.fallers) + list(m.risers)
    tok, _ = twp.load_tokenizer(a.base)
    #: FIRST TOKEN of each word: patching gives one forward pass per layer, and
    #: a full word expansion per layer would be 33x the cost for a quantity the
    #: first token already carries the sign of.
    tids = {}
    for w in words:
        e = tok.encode(" " + w, add_special_tokens=False)
        if e: tids.setdefault(e[0], []).append(w)
    ids = sorted(tids)
    print("step %s | rule %s | fallers %d risers %d | %d distinct first-tokens"
          % (cell.step.label if hasattr(cell, "step") else "?", rule.name,
             len(m.fallers), len(m.risers), len(ids)))

    dev = twp.pick_device()
    pids, _s, _r = twp._prompt_ids(tok, a.prompt, "inherited")
    x = torch.tensor([pids], device=dev)

    #: ---- both arms' hidden states and reference log-probs ---------------
    H, logp = {}, {}
    for tag, mid in (("base", a.base), ("aligned", a.aligned)):
        mdl = AM.from_pretrained(mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
        with torch.no_grad():
            o = mdl(x, output_hidden_states=True)
        H[tag] = [h.detach().clone() for h in o.hidden_states]
        logp[tag] = torch.log_softmax(o.logits[0, -1, :].float(), -1).cpu()
        if tag == "base":
            base_model = mdl            # kept resident: it does the patching
        else:
            del mdl
            try: torch.mps.empty_cache()
            except Exception: pass
        del o

    n_blocks = base_model.config.num_hidden_layers
    print("\n  ||dh_L|| / ||h_L||  -- MEASURED, not cited from F05")
    dh = {}
    for L in range(len(H['base'])):
        hb, ha = H['base'][L].float(), H['aligned'][L].float()
        dh[L] = float((ha - hb).norm() / hb.norm())
    for L in range(0, len(dh), 4):
        print("    L%-3d %.4f" % (L, dh[L]))
    print("    L%-3d %.4f  (final)" % (len(dh)-1, dh[len(dh)-1]))

    #: ---- the patch ------------------------------------------------------
    blocks = base_model.model.layers
    state = {"L": None}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        rep = H['aligned'][state["L"] + 1].to(h.dtype)
        h = rep.clone()
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    print("\n  RECOVERY: base run, residual replaced by ALIGNED at depth L,")
    print("  continued through BASE blocks and the BASE head.")
    print("  0 = nothing recovered   1 = aligned behaviour fully reproduced\n")
    print("  %-6s %10s %10s" % ("patch@L", "recovery", "||dh||/||h||"))
    rows = []
    lb, la = logp['base'], logp['aligned']
    denom = {i: float(la[i] - lb[i]) for i in ids}
    usable = [i for i in ids if abs(denom[i]) > 1e-3]
    for L in range(n_blocks):
        state["L"] = L
        h = blocks[L].register_forward_hook(hook)
        with torch.no_grad():
            o = base_model(x)
        h.remove()
        lp = torch.log_softmax(o.logits[0, -1, :].float(), -1).cpu()
        rec = statistics.median((float(lp[i] - lb[i]) / denom[i]) for i in usable)
        rows.append({"layer": L, "recovery": rec, "dh": dh[L+1]})
        if L % 2 == 0 or L >= n_blocks - 2:
            bar = "#" * max(0, min(40, int(40 * rec)))
            print("  %-6d %10.3f %10.4f  %s" % (L, rec, dh[L+1], bar))
    half = next((r["layer"] for r in rows if r["recovery"] >= 0.5), None)
    nine = next((r["layer"] for r in rows if r["recovery"] >= 0.9), None)
    print("\n  depth at which HALF of alignment's behaviour is recoverable : L%s of %d"
          % (half, n_blocks))
    print("  depth at which 90%% is recoverable                            : L%s of %d"
          % (nine, n_blocks))
    print("  words: %d fallers + %d risers, %d usable first-tokens"
          % (len(m.fallers), len(m.risers), len(usable)))
    if a.json:
        json.dump({"base": a.base, "aligned": a.aligned, "prompt": a.prompt,
                   "rows": rows, "dh": dh, "half": half, "ninety": nine},
                  open(a.json, "w"), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
