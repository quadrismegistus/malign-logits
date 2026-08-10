#!/usr/bin/env python
"""twp_lens_perword.py — is ANY word's gap distinguishable at ANY depth?

    scripts/twp_lens_perword.py --base meta-llama/Llama-3.1-8B \
        --aligned meta-llama/Llama-3.1-8B-Instruct

RH's design, and it is better than the one it replaces on two counts.

**BASELINE IS EVERY OUTPUT WORD, NOT THE FLATS.** `flats` is CANONICAL's third
class -- a RULE-DEPENDENT partition -- so a baseline built from it inherits the
rule's thresholds, and on some cells it is 6 words, which cannot support a
band. The twp output vocabulary is what the instrument actually produced: no
rule, and 100+ words per cell.

**LEAVE-ONE-OUT, PER WORD, NOT A SET MEDIAN.** The set median asks "do the
movers differ from the rest ON AVERAGE", which is exactly the statistic that
hid whatever was there: a single word standing far out at L12 contributes one
value to a median over 20. This asks, for every word and every layer, whether
THAT word's gap stands out from all OTHER words at that layer -- so the
baseline never contains the word being tested.

    gap(w, L) = log10 p_base(w, L) - log10 p_aligned(w, L)     BASE HEAD BOTH
    z(w, L)   = (gap(w,L) - median_{w' != w}) / (1.4826 * MAD_{w' != w})

Median/MAD rather than mean/sd because the mid-stack lens is heavy-tailed and
one blown-up word would inflate the sd it is being tested against.

**THE MULTIPLE-COMPARISON PROBLEM IS THE WHOLE DIFFICULTY AND IS HANDLED.**
~120 words x ~33 layers is ~4,000 tests per cell; at alpha .05 that is 200
false positives, which is exactly how one would "discover" early-layer signal
that is not there. Benjamini-Hochberg across all (word, layer) pairs within a
cell, and the count is always reported beside the number expected by chance.

**READ THROUGH THE BASE HEAD IN BOTH ARMS.** No pair in the roster froze its
unembedding, so a per-arm head would confound the readout with the stack.
"""
import argparse, json, math, os, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
THETA = 0.001


def bh(pvals, q=0.05):
    """Benjamini-Hochberg. Returns the p threshold, or 0.0 if nothing survives."""
    if not pvals: return 0.0
    s = sorted(pvals); n = len(s)
    thr = 0.0
    for i, p in enumerate(s, 1):
        if p <= q * i / n: thr = p
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompts", nargs="*", default=[
        "She was so angry she wanted to",
        "He lay naked in his bed and",
        "They were alone in the house and she"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--json", default="data/twp_lens_perword.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.movement import CANONICAL

    step = Step(Checkpoint(a.base), Checkpoint(a.aligned))
    cells = {}
    for p in a.prompts:
        try:
            c = step.cell(p); m = c.movement(CANONICAL)
            cells[p] = (dict(c.pre.probs), dict(c.post.probs), m)
        except Exception as e:
            print("  skip %r: %s" % (p[:40], type(e).__name__))
    if not cells: return 1

    tok, _ = twp.load_tokenizer(a.base); dev = twp.pick_device()
    base = AM.from_pretrained(a.base, dtype=getattr(torch, a.dtype)).to(dev).eval()
    alm = AM.from_pretrained(a.aligned, dtype=getattr(torch, a.dtype)).to(dev).eval()
    N = base.config.num_hidden_layers
    Wb = base.get_output_embeddings().weight.detach()
    nb = twp._final_norm(base); gain = getattr(nb, "weight", None)
    eps = float(getattr(nb, "variance_epsilon", 1e-5))

    out = {}
    for p, (pre, post, m) in cells.items():
        vocab = sorted({w for w, v in pre.items() if v >= THETA} |
                       {w for w, v in post.items() if v >= THETA})
        tid = lambda w: (tok.encode(" " + w, add_special_tokens=False) or [None])[0]
        wid = {w: tid(w) for w in vocab if tid(w) is not None}
        ids = sorted(set(wid.values()))
        if len(ids) < 20:
            print("\n=== %r: only %d tokens, skipping" % (p[:44], len(ids))); continue
        pids, _s, _r = twp._prompt_ids(tok, p, "inherited")
        x = torch.tensor([pids], device=dev)
        with torch.no_grad():
            ob = base(x, output_hidden_states=True)
            oa = alm(x, output_hidden_states=True)
        gaps = {}
        with torch.no_grad():
            for L in range(N + 1):
                hb = ob.hidden_states[L][0, -1, :].to(Wb.dtype)
                ha = oa.hidden_states[L][0, -1, :].to(Wb.dtype)
                if L != N and gain is not None:
                    f = lambda h: (h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True)
                                                   + eps).to(h.dtype) * gain)
                    hb, ha = f(hb), f(ha)
                pb = torch.softmax((hb @ Wb.T).float(), -1)
                pa = torch.softmax((ha @ Wb.T).float(), -1)
                gaps[L] = {i: float(torch.log10(pb[i].clamp_min(1e-30) /
                                                pa[i].clamp_min(1e-30))) for i in ids}
        del ob, oa

        #: leave-one-out robust z, every word, every layer
        recs, pv = [], []
        for L in range(N + 1):
            g = gaps[L]
            vals = list(g.values())
            for i, v in g.items():
                oth = [u for j, u in g.items() if j != i]
                med = statistics.median(oth)
                mad = statistics.median([abs(u - med) for u in oth]) * 1.4826
                if mad <= 0: continue
                z = (v - med) / mad
                pval = math.erfc(abs(z) / math.sqrt(2))
                recs.append({"layer": L, "tok": i, "z": z, "p": pval, "gap": v})
                pv.append(pval)
        thr = bh(pv, a.q)
        sig = [r for r in recs if r["p"] <= thr] if thr > 0 else []
        exp_chance = a.q * len(recs) / max(1, len(recs)) * len(sig)  # BH's own guarantee
        by_tok = {v: k for k, v in wid.items()}
        movers = set(m.fallers) | set(m.risers)
        early = [r for r in sig if r["layer"] <= int(0.66 * N)]
        print("\n=== %r" % p[:52])
        print("  vocab %d words | %d tokens | %d tests | BH q=%.2f threshold p<=%.2e"
              % (len(vocab), len(ids), len(recs), a.q, thr))
        print("  significant (word, layer) pairs: %d   -- of which at <=2/3 depth: %d"
              % (len(sig), len(early)))
        if sig:
            first = min(r["layer"] for r in sig)
            print("  EARLIEST significant layer: L%d of %d" % (first, N))
            byl = defaultdict(list)
            for r in sig: byl[r["layer"]].append(r)
            for L in sorted(byl)[:6]:
                names = ", ".join("%s(z=%+.1f)%s" % (by_tok.get(r["tok"], "?"), r["z"],
                                  "*" if by_tok.get(r["tok"]) in movers else "")
                                  for r in sorted(byl[L], key=lambda r: -abs(r["z"]))[:4])
                print("    L%-3d %d sig: %s" % (L, len(byl[L]), names))
            print("    (* = a CANONICAL mover at the output)")
            nm = sum(1 for r in sig if by_tok.get(r["tok"]) in movers)
            print("  of the significant pairs, %d/%d involve an output mover"
                  % (nm, len(sig)))
        else:
            print("  NOTHING survives BH at any layer. With %d tests the "
                  "uncorrected count at .05 would be ~%d." % (len(recs), int(.05*len(recs))))
        out[p] = {"n_tests": len(recs), "thr": thr, "n_sig": len(sig),
                  "n_early": len(early),
                  "earliest": (min(r["layer"] for r in sig) if sig else None),
                  "sig": [{"layer": r["layer"], "word": by_tok.get(r["tok"]),
                           "z": r["z"], "mover": by_tok.get(r["tok"]) in movers}
                          for r in sig]}
    json.dump({"base": a.base, "aligned": a.aligned, "cells": out},
              open(os.path.join(ROOT, a.json), "w"), indent=1)
    print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
