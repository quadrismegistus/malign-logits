#!/usr/bin/env python
"""twp_head_swap.py — is the base/aligned gap the STACK or the HEAD?

    scripts/twp_head_swap.py --base LLM360/Amber --aligned LLM360/AmberSafe

## WHY THIS RUN EXISTS

The per-layer contrast `head(norm(h_L))` is out of distribution mid-stack, and
the defence was that the defect is COMMON-MODE: two arms read at the same depth
through the same head, so the lens cancels in the difference (lacan, [5222].2).

**The condition failed.** Measured:

    Llama-3.1-8B   ||dW_U||/||W_U|| = 6.56e-02   final-norm gain 1.96e-03
    Amber          ||dW_U||/||W_U|| = 3.48e-02   final-norm gain 1.13e-02

The fine-tune moved the unembedding by 3-7% of its norm. So a gap between arms
mixes two causes and the mixture is unknown.

## THE DECOMPOSITION, WHICH NEEDS NO NEW FORWARD PASSES

Four readings per layer, from two sets of hidden states and two heads:

    S_b H_b   base stack,    base head      (the base arm as normally read)
    S_a H_a   aligned stack, aligned head   (the aligned arm as normally read)
    S_a H_b   aligned stack read through the BASE head
    S_b H_a   base stack read through the ALIGNED head

    total gap    = log P(S_b H_b) - log P(S_a H_a)     what I have been quoting
    STACK effect = log P(S_b H_b) - log P(S_a H_b)     head held fixed at base
    HEAD effect  = log P(S_b H_b) - log P(S_b H_a)     stack held fixed at base

If STACK ~ total, the contrast survives and the head is a rounding error. If
HEAD carries much of it, the depth reading is substantially an artifact of the
readout -- and lacan's prediction is that it inflates PROMOTION specifically,
which is the side I twice called the larger effect.

**A swapped head is out of distribution for the OTHER arm's stack too**, so
these are not clean counterfactuals -- they are a decomposition of a difference,
not a simulation of a model. The interaction term is not identified. Reported as
what it is.
"""
import argparse, json, math, os, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="She was so angry she wanted to")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rule", default="canonical", choices=("canonical", "lens"))
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits import movement as MV

    rule = MV.CANONICAL if a.rule == "canonical" else MV.LENS
    step = Step(Checkpoint(a.base), Checkpoint(a.aligned))
    cell = step.cell(a.prompt)
    m = cell.movement(rule)
    fal, ris = list(m.fallers), list(m.risers)
    pre = cell.pre.probs
    flt = [w for w in pre if w not in set(fal) | set(ris)
           and pre[w] >= max(rule.min_prob, 1e-9)]
    print("step %s | rule %s | fallers %d risers %d flats %d"
          % (step.label, rule.name, len(fal), len(ris), len(flt)))
    vocab = sorted(set(fal) | set(ris) | set(flt))

    #: collect hidden states AND heads from both arms, then cross them
    H, heads, norms = {}, {}, {}
    tok = None
    for tag, mid in (("base", a.base), ("aligned", a.aligned)):
        t, _ = twp.load_tokenizer(mid)
        dev = twp.pick_device()
        mdl = AM.from_pretrained(mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
        if tok is None:
            tok = t
            bmask = twp.boundary_mask(t, mdl.config.vocab_size)
            pids, _s, _r = twp._prompt_ids(t, a.prompt, "inherited")
            paths = {}
            for w in vocab:
                ids = t.encode(" " + w, add_special_tokens=False)
                if ids and twp.clean_surface(t.decode(ids).strip()) == w:
                    paths[w] = ids
            need = {(): None}
            for ids in paths.values():
                for k in range(len(ids) + 1): need[tuple(ids[:k])] = None
            prefixes = sorted(need, key=lambda x: (len(x), x))
        with torch.no_grad():
            per = {}
            for i in range(0, len(prefixes), 32):
                ch = prefixes[i:i+32]
                ids_t, att = twp._pad(tok, pids, ch, dev)
                o = mdl(ids_t, attention_mask=att, output_hidden_states=True)
                for j, pre_t in enumerate(ch):
                    per[pre_t] = [h[j, -1, :].float().cpu() for h in o.hidden_states]
                del o
        H[tag] = per
        heads[tag] = mdl.get_output_embeddings().weight.detach().float().cpu()
        n = twp._final_norm(mdl)
        norms[tag] = (getattr(n, "weight", None).detach().float().cpu()
                      if getattr(n, "weight", None) is not None else None,
                      float(getattr(n, "variance_epsilon", 1e-5)))
        del mdl
        try: torch.mps.empty_cache()
        except Exception: pass

    n_hs = len(next(iter(H['base'].values())))
    bm = torch.tensor(bmask)

    def probs(stack_tag, head_tag, l):
        """word -> p, reading stack_tag's layer l through head_tag's head."""
        W = heads[head_tag]; gain, eps = norms[head_tag]
        rows = {}
        for pre_t, hs in H[stack_tag].items():
            h = hs[l]
            if l != n_hs - 1 and gain is not None:      # apply that head's own norm
                h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * gain
            rows[pre_t] = torch.softmax(h @ W.T, -1)
        out = {}
        for w, ids in paths.items():
            p = 1.0
            for k, t_ in enumerate(ids):
                p *= float(rows[tuple(ids[:k])][t_])
            p *= float(rows[tuple(ids)][bm].sum())
            out[w] = p
        return out

    print("\n  gap = log10 P(base) - log10 P(other). TOTAL is what I quoted;")
    print("  STACK holds the head at base; HEAD holds the stack at base.\n")
    print("  %-6s | %-22s | %-22s" % ("", "FALLERS", "RISERS"))
    print("  %-6s | %7s %7s %6s | %7s %7s %6s"
          % ("layer", "total", "stack", "head", "total", "stack", "head"))
    rows = []
    for l in list(range(0, n_hs, 4)) + [n_hs - 1]:
        Pbb, Paa = probs('base', 'base', l), probs('aligned', 'aligned', l)
        Pab = probs('aligned', 'base', l)      # aligned stack, base head
        Pba = probs('base', 'aligned', l)      # base stack, aligned head
        def med(ws, A, B):
            v = [math.log10(max(A[w], 1e-30) / max(B[w], 1e-30)) for w in ws if w in A]
            return statistics.median(v) if v else float('nan')
        r = {"layer": l,
             "fal_total": med(fal, Pbb, Paa), "fal_stack": med(fal, Pbb, Pab),
             "fal_head": med(fal, Pbb, Pba),
             "ris_total": med(ris, Pbb, Paa), "ris_stack": med(ris, Pbb, Pab),
             "ris_head": med(ris, Pbb, Pba)}
        rows.append(r)
        print("  %-6d | %+7.2f %+7.2f %+6.2f | %+7.2f %+7.2f %+6.2f"
              % (l, r["fal_total"], r["fal_stack"], r["fal_head"],
                 r["ris_total"], r["ris_stack"], r["ris_head"]))
    f = rows[-1]
    def share(tot, part):
        return float('nan') if abs(tot) < 1e-9 else 100.0 * part / tot
    print("\n  AT THE OUTPUT:")
    print("    fallers  total %+.2f = stack %+.2f (%.0f%%) + head %+.2f (%.0f%%)"
          % (f["fal_total"], f["fal_stack"], share(f["fal_total"], f["fal_stack"]),
             f["fal_head"], share(f["fal_total"], f["fal_head"])))
    print("    risers   total %+.2f = stack %+.2f (%.0f%%) + head %+.2f (%.0f%%)"
          % (f["ris_total"], f["ris_stack"], share(f["ris_total"], f["ris_stack"]),
             f["ris_head"], share(f["ris_total"], f["ris_head"])))
    print("\n  shares do not sum to 100: the interaction is NOT identified, and a")
    print("  swapped head is out of distribution for the other arm's stack too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
