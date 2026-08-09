#!/usr/bin/env python
"""twp_word_depth.py — score a KNOWN word at every layer. No threshold.

    scripts/twp_word_depth.py --base meta-llama/Llama-3.1-8B \
        --aligned meta-llama/Llama-3.1-8B-Instruct \
        --prompt "She was so angry she wanted to"

## WHY THIS EXISTS: theta IS A DISCOVERY DEVICE, NOT A MEASUREMENT ONE

`expand_layers` needs theta because it DISCOVERS a vocabulary -- it has to decide
which tokens are worth expanding. That made "where does alignment demote this
word" unanswerable below layer 18: the target words sat under theta in both arms
and the instrument reported `dead`, not a small number.

Lowering theta is the wrong fix and the cost says so. Measured on this prompt,
summed over 33 layers, the seed set grows **19x at 1e-4, 214x at 1e-5, 973x at
1e-6** -- and it grows worst in the mid-stack, where the readout is `efon` and
`$MESS`. It would buy hours of compute to compare one arm's lens noise against
the other's.

**The right lever is to stop discovering and start scoring.** The vocabulary is
already known from the OUTPUT layer, where the unembedding is the one the model
was trained with. Scoring a known word needs no threshold at all, so every word
has a value at every layer and the censoring disappears.

## THE VOCABULARY IS THE UNION OF BOTH ARMS' OUTPUT WORDS

Not the base's alone. A word alignment PROMOTES -- `scream` is 0.194 in aligned
Llama and under theta in the base -- is invisible to a base-only vocabulary, and
that omission is what made an earlier attempt unable to say anything about the
single word its story rested on.

## THE QUANTITY IS twp's, NOT A NEW ONE

    p_L(word) = [ prod_i p_L(t_i | prompt + t_1..t_{i-1}) ] * P_L(boundary | full)

Same chain rule, same boundary mask (`rule_version 3`, the CJK trie, the
script-transition cut). One canonical token path per word rather than a sum over
paths, so the final-layer value is a LOWER BOUND on twp's stored cell, not a
reproduction of it -- stated because a number that nearly matches a corpus is
exactly the kind that gets quoted as if it did.
"""
import argparse, json, math, os, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
THETA = 0.001


def readouts_for(model, twp, n_hs):
    return {l: (twp.FinalReadout() if l == n_hs - 1 else twp.LayerReadout(model, l))
            for l in range(n_hs)}


def score_words(model, tok, prompt, words, dev, bmask, twp, torch, cjk=None):
    """{word: {layer: p}} -- every word, every layer, no threshold anywhere."""
    pids, _s, _r = twp._prompt_ids(tok, prompt, "inherited")
    n_hs = model.config.num_hidden_layers + 1
    ro = readouts_for(model, twp, n_hs)

    #: one canonical path per word: the leading-space form, which is how a
    #: continuation is actually tokenised
    paths = {}
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if ids and twp.clean_surface(tok.decode(ids).strip()) == w:
            paths[w] = ids
    #: every prefix any word needs, deduplicated -- `kill` and `killed` share one
    need = {(): None}
    for ids in paths.values():
        for k in range(len(ids) + 1):
            need[tuple(ids[:k])] = None
    prefixes = sorted(need, key=lambda t: (len(t), t))

    dists = {}          # prefix -> {layer: full softmax row}
    B = 32
    with torch.no_grad():
        for i in range(0, len(prefixes), B):
            chunk = prefixes[i:i + B]
            ids, att = twp._pad(tok, pids, chunk, dev)
            out = model(ids, attention_mask=att, output_hidden_states=True)
            for l in range(n_hs):
                rows = torch.softmax(ro[l](out), -1).float().cpu()
                for j, pre in enumerate(chunk):
                    dists.setdefault(pre, {})[l] = rows[j]
            del out

    res = {}
    for w, ids in paths.items():
        per = {}
        for l in range(n_hs):
            p = 1.0
            for k, t in enumerate(ids):
                p *= float(dists[tuple(ids[:k])][l][t])
            #: the boundary factor -- the word must END here, same mask as twp
            p *= float(dists[tuple(ids)][l][bmask].sum())
            per[l] = p
        res[w] = per
    return res, len(prefixes)


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
    store = {}
    vocab = set(); finalp = {}
    for tag, mid in (("base", a.base), ("aligned", a.aligned)):
        tok, _ = twp.load_tokenizer(mid)
        dev = twp.pick_device()
        m = AM.from_pretrained(mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
        bmask = twp.boundary_mask(tok, m.config.vocab_size)
        twp.reset_batch()
        out, _st = twp.expand_layers(m, tok, a.prompt, dev, bmask,
                                     [m.config.num_hidden_layers])
        agg = defaultdict(float)
        for (s, _t1), p in out[m.config.num_hidden_layers][0].items():
            agg[s] += p
        finalp[tag] = dict(agg)
        vocab |= {w for w, p in agg.items() if p >= THETA}
        store[tag] = (m, tok, dev, bmask)

    vocab = sorted(vocab)
    print("prompt %r" % a.prompt)
    print("VOCABULARY = union of both arms' output words above theta")
    print("  base-only %d | aligned-only %d | shared %d | UNION %d"
          % (len({w for w,p in finalp['base'].items() if p>=THETA} - set(finalp['aligned'])),
             len({w for w,p in finalp['aligned'].items() if p>=THETA} - set(finalp['base'])),
             len({w for w,p in finalp['base'].items() if p>=THETA} &
                 {w for w,p in finalp['aligned'].items() if p>=THETA}), len(vocab)))

    scored = {}
    for tag in ("base", "aligned"):
        m, tok, dev, bmask = store[tag]
        scored[tag], npre = score_words(m, tok, a.prompt, vocab, dev, bmask,
                                        twp, torch)
        print("  %-8s scored %d/%d words from %d shared prefixes"
              % (tag, len(scored[tag]), len(vocab), npre))
        del m
        try: torch.mps.empty_cache()
        except Exception: pass

    common = sorted(set(scored['base']) & set(scored['aligned']))
    n_hs = max(scored['base'][common[0]]) + 1
    fin = n_hs - 1
    gap = lambda w, l: math.log10(max(scored['base'][w][l], 1e-30) /
                                  max(scored['aligned'][w][l], 1e-30))
    dem = sorted((w for w in common if gap(w, fin) > math.log10(2)),
                 key=lambda w: -gap(w, fin))
    pro = sorted((w for w in common if gap(w, fin) < -math.log10(2)),
                 key=lambda w: gap(w, fin))
    unm = [w for w in common if abs(gap(w, fin)) < 0.10]
    print("\n  at the OUTPUT: %d demoted, %d promoted, %d unmoved (of %d scored)"
          % (len(dem), len(pro), len(unm), len(common)))
    print("  demoted : %s" % ", ".join(dem[:10]))
    print("  promoted: %s" % ", ".join(pro[:10]))

    print("\n  MEDIAN GAP log10(base/aligned) BY LAYER — no censoring, every "
          "word live at every layer")
    print("  %-6s %9s %9s %9s" % ("layer", "demoted", "promoted", "unmoved"))
    rows = []
    for l in range(n_hs):
        md = statistics.median([gap(w, l) for w in dem]) if dem else float('nan')
        mp = statistics.median([gap(w, l) for w in pro]) if pro else float('nan')
        mu = statistics.median([gap(w, l) for w in unm]) if unm else float('nan')
        rows.append({"layer": l, "demoted": md, "promoted": mp, "unmoved": mu})
        if l % 2 == 0 or l >= n_hs - 3:
            print("  %-6d %+9.2f %+9.2f %+9.2f" % (l, md, mp, mu))
    sep = [r for r in rows if r["demoted"] - r["unmoved"] > 0.5]
    print("\n  first layer with demoted-unmoved > 0.5 : %s"
          % (sep[0]["layer"] if sep else "never"))
    print("  took %.1f s" % (time.perf_counter() - t0))
    if a.json:
        json.dump({"prompt": a.prompt, "rows": rows, "demoted": dem,
                   "promoted": pro, "unmoved": unm,
                   "scored": {k: {w: v for w, v in s.items()}
                              for k, s in scored.items()}},
                  open(a.json, "w"), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
