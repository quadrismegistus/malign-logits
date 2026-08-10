#!/usr/bin/env python
"""twp_patch_weights.py — HOW MANY ALIGNED BLOCKS DOES THE BEHAVIOUR NEED?

    scripts/twp_patch_weights.py --base meta-llama/Llama-3.1-8B \
        --aligned meta-llama/Llama-3.1-8B-Instruct

The complement to `twp_patch_depth.py`, and the test its result demands.

Representation-patching (swap the residual stream at depth L, keep base weights)
found half of alignment's behaviour recoverable only at L26 of 32. **But it
holds the WEIGHTS at base**, so it asks "is the aligned representation
sufficient GIVEN BASE MACHINERY?" A distributed change whose early parts only
pay off in concert with the later aligned weights scores ZERO on that test while
being genuinely deep. It structurally cannot distinguish inert-early from
early-that-needs-late.

This swaps the weights instead, in both directions, which is what separates them:

    TOP-K      blocks 0..N-K-1 from BASE, the last K from ALIGNED
               "how many aligned blocks from the TOP suffice?"
    BOTTOM-K   the first K from ALIGNED, the rest from BASE
               "do aligned blocks at the BOTTOM do anything at all?"

    recovery = median over words of
               (log p_hybrid - log p_base) / (log p_aligned - log p_base)

READ THE TWO TOGETHER, because either alone misleads:

    top-K reaches 1.0 at small K AND bottom-K stays ~0   -> genuinely shallow
    top-K needs most of the stack                        -> deep
    bottom-K > 0                                         -> early blocks do
                                                            real work, whatever
                                                            the top-K curve says
    both curves rise slowly and neither saturates        -> distributed and
                                                            interacting; no
                                                            single depth is
                                                            "where alignment is"

**THE HEAD AND EMBEDDINGS ARE HELD AT BASE THROUGHOUT.** Only transformer blocks
are swapped, so the readout is fixed by construction and none of the
frozen-head problem arises. The final norm stays base too -- it is not a block.

CAVEAT: a hybrid model is not a model anyone trained. Blocks from two
checkpoints compose only because one is a fine-tune of the other and the
residual basis is shared by initialisation. This is a decomposition of a
difference, not a simulation, and the interaction is not identified.
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
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch, copy
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits import movement as MV

    rule = MV.CANONICAL if a.rule == "canonical" else MV.LENS
    cell = Step(Checkpoint(a.base), Checkpoint(a.aligned)).cell(a.prompt)
    m = cell.movement(rule)
    tok, _ = twp.load_tokenizer(a.base)
    ids = sorted({tok.encode(" " + w, add_special_tokens=False)[0]
                  for w in list(m.fallers) + list(m.risers)
                  if tok.encode(" " + w, add_special_tokens=False)})
    dev = twp.pick_device()
    pids, _s, _r = twp._prompt_ids(tok, a.prompt, "inherited")
    x = torch.tensor([pids], device=dev)

    base = AM.from_pretrained(a.base, dtype=getattr(torch, a.dtype)).to(dev).eval()
    with torch.no_grad():
        lb = torch.log_softmax(base(x).logits[0, -1, :].float(), -1).cpu()
    aligned = AM.from_pretrained(a.aligned, dtype=getattr(torch, a.dtype)).to(dev).eval()
    with torch.no_grad():
        la = torch.log_softmax(aligned(x).logits[0, -1, :].float(), -1).cpu()

    N = base.config.num_hidden_layers
    base_blocks = list(base.model.layers)
    al_blocks = list(aligned.model.layers)
    usable = [i for i in ids if abs(float(la[i] - lb[i])) > 1e-3]
    denom = {i: float(la[i] - lb[i]) for i in usable}
    print("step base->%s | rule %s | fallers %d risers %d | %d usable tokens | %d blocks"
          % (a.aligned.split('/')[-1], rule.name, len(m.fallers), len(m.risers),
             len(usable), N))
    print("  head, embeddings and final norm stay BASE throughout.\n")

    def run(sel):
        """sel[i] True -> use the ALIGNED block at position i."""
        base.model.layers = torch.nn.ModuleList(
            [al_blocks[i] if sel[i] else base_blocks[i] for i in range(N)])
        with torch.no_grad():
            lp = torch.log_softmax(base(x).logits[0, -1, :].float(), -1).cpu()
        return statistics.median(float(lp[i] - lb[i]) / denom[i] for i in usable)

    ks = sorted(set(list(range(0, N + 1, a.step)) + [N]))
    print("  %-5s %12s %12s" % ("K", "top-K", "bottom-K"))
    print("  %-5s %12s %12s" % ("", "last K aligned", "first K aligned"))
    rows = []
    for k in ks:
        top = run([i >= N - k for i in range(N)])
        bot = run([i < k for i in range(N)])
        rows.append({"k": k, "top": top, "bottom": bot})
        bt = "#" * max(0, min(30, int(30 * top)))
        bb = "#" * max(0, min(30, int(30 * bot)))
        print("  %-5d %12.3f %12.3f  %-31s|%s" % (k, top, bot, bt, bb))

    base.model.layers = torch.nn.ModuleList(base_blocks)
    half_top = next((r["k"] for r in rows if r["top"] >= 0.5), None)
    half_bot = next((r["k"] for r in rows if r["bottom"] >= 0.5), None)
    bot_max = max(r["bottom"] for r in rows[:-1]) if len(rows) > 1 else float('nan')
    print("\n  aligned blocks from the TOP needed for half the behaviour : %s of %d"
          % (half_top, N))
    print("  aligned blocks from the BOTTOM needed for half             : %s of %d"
          % (half_bot, N))
    print("  best recovery from BOTTOM blocks alone (excluding all-N)   : %.3f" % bot_max)
    print("\n  READING: bottom-K > 0 means early aligned blocks do real work even")
    print("  though the aligned REPRESENTATION at that depth was not sufficient.")
    print("  Both curves rising slowly = distributed and interacting; no single")
    print("  depth is 'where alignment is'.")
    if a.json:
        json.dump({"base": a.base, "aligned": a.aligned, "prompt": a.prompt,
                   "N": N, "rows": rows}, open(a.json, "w"), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
