#!/usr/bin/env python
"""twp_demotion_depth.py — WHERE in the stack does alignment demote a word?

    scripts/twp_demotion_depth.py --base meta-llama/Llama-3.1-8B \
        --aligned meta-llama/Llama-3.1-8B-Instruct \
        --prompt "She was so angry she wanted to"

RH's question, and it is a better one than the trajectory-watching it replaces:
**do the words alignment demotes AT THE OUTPUT get demoted early or late inside
the stack?**

## WHY THIS DESIGN AND NOT THE OBVIOUS ONE

**It compares base against aligned AT THE SAME LAYER.** The previous attempt
compared layer 27 against layer 32 within one model, which makes the lens the
dominant term: the unembedding is trained for the final layer, so a mid-stack
readout is noisy in a way that is a property of (layer, architecture). Reading
both arms at the same depth cancels most of that -- the aligned model is a
fine-tune of the base, so layer i corresponds to layer i. **That correspondence
is a fine-tune assumption and does NOT hold across families.**

**The set is defined at the OUTPUT, which is the well-measured end.** Selecting
on a mid-stack quantity would be selecting on noise.

**CENSORING IS THE SIGNAL, NOT MISSING DATA.** twp keeps per-layer theta
pruning, so a word can be above theta in one arm and below it in the other. That
event -- base has it, aligned does not -- IS a demotion, and it is exactly the
case the previous attempt had to throw away. Four states are tracked at every
layer and all four are reported:

    BOTH    measured gap  log10(p_base / p_aligned)
    B_ONLY  aligned below theta -> gap is a LOWER BOUND, log10(p_base/theta)
    A_ONLY  base below theta    -> promotion, upper bound
    NEITHER not live at this depth yet; counted, never silently dropped

## DECLARED BEFORE LOOKING

    DEMOTED   final-layer p_base / p_aligned >= 2.0, with p_base >= theta
    CONTROL   final-layer |log10 ratio| < 0.10 (unmoved), MATCHED to the
              demoted set on final-layer p_base by decile

Matching on p_base is not decoration: lacan measured a **-0.33 nuisance floor**
tonight -- net movement tracks base probability even at neutral prompts -- so an
unmatched control would reproduce that floor and be mistaken for a result.
"""
import argparse, json, math, os, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

THETA = 0.001
DEMOTE_RATIO = 2.0
UNMOVED_LOG = 0.10


def layer_dists(model_id, prompt, dtype, twp, torch, AutoModel):
    tok, _ = twp.load_tokenizer(model_id)
    dev = twp.pick_device()
    m = AutoModel.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                  trust_remote_code=True).to(dev).eval()
    bmask = twp.boundary_mask(tok, m.config.vocab_size)
    layers = list(range(m.config.num_hidden_layers + 1))
    twp.reset_batch()
    out, st = twp.expand_layers(m, tok, prompt, dev, bmask, layers)
    d = {}
    for l in layers:
        agg = defaultdict(float)
        for (surf, _t1), p in out[l][0].items():
            agg[surf] += p
        d[l] = dict(agg)
    del m
    try: torch.mps.empty_cache()
    except Exception: pass
    return d, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="She was so angry she wanted to")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp

    t0 = time.perf_counter()
    B, stb = layer_dists(a.base, a.prompt, a.dtype, twp, torch, AM)
    A, sta = layer_dists(a.aligned, a.prompt, a.dtype, twp, torch, AM)
    took = time.perf_counter() - t0
    nb, na = max(B), max(A)
    if nb != na:
        print("LAYER COUNTS DIFFER (%d vs %d) -- correspondence undefined, "
              "refusing." % (nb, na)); return 2
    FIN = nb

    fb, fa = B[FIN], A[FIN]
    demoted, control_pool = [], []
    for w, pb in fb.items():
        if pb < THETA: continue
        pa = fa.get(w, 0.0)
        r = math.log10(pb / max(pa, 1e-12))
        if pa > 0 and pb / pa >= DEMOTE_RATIO: demoted.append((w, pb, pa, r))
        elif pa > 0 and abs(r) < UNMOVED_LOG: control_pool.append((w, pb, pa, r))

    #: MATCH ON FINAL-LAYER p_base, because demotion correlates with it
    def dec(p): return int(min(9, max(0, math.log10(max(p, 1e-6)) + 6)))
    pool = defaultdict(list)
    for row in control_pool: pool[dec(row[1])].append(row)
    control, unmatched = [], 0
    for w, pb, pa, r in demoted:
        c = pool.get(dec(pb))
        if c: control.append(c.pop())
        else: unmatched += 1

    print("prompt %r\nbase %s\naligned %s\n(%.1f s, %d layers)\n"
          % (a.prompt, a.base, a.aligned, took, FIN + 1))
    print("SETS, declared before looking (demote ratio >=%.1f, unmoved |log10|<%.2f)"
          % (DEMOTE_RATIO, UNMOVED_LOG))
    print("  final-layer words in base   %4d" % len(fb))
    print("  DEMOTED                     %4d" % len(demoted))
    print("  control pool (unmoved)      %4d" % len(control_pool))
    print("  control MATCHED on p_base   %4d   (%d demoted had no match)"
          % (len(control), unmatched))
    if not demoted or not control:
        print("\n  too few words to compare on this prompt."); return 1
    print("\n  top demoted: %s" % ", ".join(
        "%s %.3f->%.3f" % (w, pb, pa) for w, pb, pa, _ in
        sorted(demoted, key=lambda x: -x[3])[:6]))

    def gaps(words, l):
        """(measured gaps, censoring counts) at layer l."""
        g, st = [], defaultdict(int)
        for w, _pb, _pa, _r in words:
            pb, pa = B[l].get(w), A[l].get(w)
            if pb and pa: st['BOTH'] += 1; g.append(math.log10(pb / pa))
            elif pb and not pa: st['B_ONLY'] += 1; g.append(math.log10(pb / THETA))
            elif pa and not pb: st['A_ONLY'] += 1; g.append(-math.log10(pa / THETA))
            else: st['NEITHER'] += 1
        return g, st

    print("\n  GAP = log10(p_base / p_aligned) at each layer. Positive = base "
          "favours it.\n  B_ONLY/A_ONLY are theta crossings, entered as BOUNDS.\n")
    print("  %-5s %26s %26s" % ("", "DEMOTED", "CONTROL (matched)"))
    print("  %-5s %8s %6s %10s %8s %6s %10s"
          % ("layer", "median", "n", "live/cens", "median", "n", "live/cens"))
    rows=[]
    for l in range(FIN + 1):
        gd, sd = gaps(demoted, l); gc, sc = gaps(control, l)
        md = statistics.median(gd) if gd else float('nan')
        mc = statistics.median(gc) if gc else float('nan')
        rows.append({"layer": l, "demoted_median": md, "control_median": mc,
                     "demoted_n": len(gd), "control_n": len(gc),
                     "demoted_states": dict(sd), "control_states": dict(sc)})
        if l % 2 == 0 or l >= FIN - 3:
            print("  %-5d %8s %6d %10s %8s %6d %10s"
                  % (l, ("%+.2f"%md) if gd else "   -", len(gd),
                     "%d/%d" % (len(gd), sd['NEITHER']),
                     ("%+.2f"%mc) if gc else "   -", len(gc),
                     "%d/%d" % (len(gc), sc['NEITHER'])))
    sep = [r for r in rows if r["demoted_n"] and r["control_n"]
           and r["demoted_median"] - r["control_median"] > 0.5]
    print("\n  FIRST LAYER where demoted-median exceeds control-median by >0.5: %s"
          % (sep[0]["layer"] if sep else "never"))
    print("  final-layer separation: %+.2f (demoted) vs %+.2f (control)"
          % (rows[FIN]["demoted_median"], rows[FIN]["control_median"]))
    if a.json:
        json.dump({"prompt": a.prompt, "base": a.base, "aligned": a.aligned,
                   "rows": rows, "demoted": [w for w,_,_,_ in demoted],
                   "control": [w for w,_,_,_ in control]}, open(a.json,'w'), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
