#!/usr/bin/env python
"""twp_word_floor.py — the NUISANCE FLOOR for per-layer base/aligned gaps.

    scripts/twp_word_floor.py --prompt "She was so angry she wanted to"

The pen's (b3) at [5221]: my control was matched only on being UNMOVED at the
output, and nobody has measured what an ARBITRARY output-defined word set does
through the stack under this instrument. So the floor is built two ways:

    RANDOM    output-stable words sampled at random, resampled B times ->
              a band, not a line
    BANDS     words binned by base output probability -> does the gap depend
              on base probability, which is where lacan's -0.33 lives

## WHAT THIS NULL IS, AND WHAT IT IS NOT -- RH's INSTRUCTION, 2026-08-09

**"I don't think it's the definitive null so don't declare anything dead if
null."** Honoured here and in the printed output. This measures ONE thing: what
an arbitrary output-defined word set does through the stack, under this lens, on
this prompt, in this pair. It cannot speak to:

    - whether head(norm(h_L)) mid-stack is a quantity or a lens artifact
      (lacan's head-free projections arbitrate that, not this)
    - whether act-to-vocalisation is displacement or a frequency effect
    - anything cross-prompt or cross-family

**A flat floor makes the demoted/promoted curves READABLE. It does not make them
true.** A non-flat floor does NOT kill them either -- it would mean the contrast
has to be read against a sloping baseline, which is a harder analysis and not a
dead one. Nothing here retires anything on its own.
"""
import argparse, json, math, os, random, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
THETA = 0.001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="She was so angry she wanted to")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from twp_word_depth import score_words

    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    finalp, scored = {}, {}
    vocab = set()
    for tag, mid in (("base", a.base), ("aligned", a.aligned)):
        tok, _ = twp.load_tokenizer(mid)
        dev = twp.pick_device()
        m = AM.from_pretrained(mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
        bmask = twp.boundary_mask(tok, m.config.vocab_size)
        L = m.config.num_hidden_layers
        twp.reset_batch()
        out, _ = twp.expand_layers(m, tok, a.prompt, dev, bmask, [L])
        agg = defaultdict(float)
        for (s, _t1), p in out[L][0].items():
            agg[s] += p
        finalp[tag] = dict(agg)
        vocab |= {w for w, p in agg.items() if p >= THETA}
        if tag == "base":
            store = (m, tok, dev, bmask)
        else:
            store2 = (m, tok, dev, bmask)
    vocab = sorted(vocab)
    for tag, (m, tok, dev, bmask) in (("base", store), ("aligned", store2)):
        scored[tag], _n = score_words(m, tok, a.prompt, vocab, dev, bmask,
                                      twp, torch)
        del m
        try: torch.mps.empty_cache()
        except Exception: pass

    common = sorted(set(scored['base']) & set(scored['aligned']))
    n_hs = max(scored['base'][common[0]]) + 1
    fin = n_hs - 1
    g = lambda w, l: math.log10(max(scored['base'][w][l], 1e-30) /
                                max(scored['aligned'][w][l], 1e-30))

    dem = [w for w in common if g(w, fin) > math.log10(2)]
    pro = [w for w in common if g(w, fin) < -math.log10(2)]
    unm = [w for w in common if abs(g(w, fin)) < 0.10]

    #: THE RANDOM FLOOR: a BAND from resampling, not a single line -- a single
    #: random draw would be one more curve to eyeball against
    k = max(len(dem), 6)
    band = {}
    for l in range(n_hs):
        meds = []
        for _ in range(a.boot):
            meds.append(statistics.median(g(w, l) for w in rng.sample(common, k)))
        meds.sort()
        band[l] = (meds[int(.025 * len(meds))], meds[len(meds)//2],
                   meds[int(.975 * len(meds))])

    #: THE BASE-PROBABILITY BANDS: is the gap a function of p_base at output?
    by_band = defaultdict(list)
    for w in common:
        pb = finalp['base'].get(w, 0.0)
        by_band[min(3, max(0, int(-math.log10(max(pb, 1e-6)) )))].append(w)

    print("prompt %r\n%s vs %s\n" % (a.prompt, a.base, a.aligned))
    print("*** THIS IS A NULL, NOT THE NULL (RH, 2026-08-09). A flat floor makes")
    print("*** the contrast READABLE; it does not make it true. A sloping floor")
    print("*** does not kill it either -- it makes the analysis harder.\n")
    print("  scored %d words | demoted %d | promoted %d | unmoved %d"
          % (len(common), len(dem), len(pro), len(unm)))
    print("  random floor: median of %d-word samples, %d resamples, 95%% band"
          % (k, a.boot))

    print("\n  %-6s %8s %8s %8s | %-22s" % ("layer","demoted","promoted","unmoved",
                                            "RANDOM FLOOR 95% band"))
    rows=[]
    for l in range(n_hs):
        md = statistics.median([g(w,l) for w in dem]) if dem else float('nan')
        mp = statistics.median([g(w,l) for w in pro]) if pro else float('nan')
        mu = statistics.median([g(w,l) for w in unm]) if unm else float('nan')
        lo, mid, hi = band[l]
        out_of = ""
        if dem and (md > hi or md < lo): out_of += "D"
        if pro and (mp > hi or mp < lo): out_of += "P"
        rows.append({"layer": l, "demoted": md, "promoted": mp, "unmoved": mu,
                     "floor_lo": lo, "floor_med": mid, "floor_hi": hi,
                     "outside": out_of})
        if l % 2 == 0 or l >= n_hs - 3:
            print("  %-6d %+8.2f %+8.2f %+8.2f | %+6.2f .. %+6.2f  %s"
                  % (l, md, mp, mu, lo, hi, out_of))
    print("\n  D = demoted median outside the random band; P = promoted outside")
    first_d = next((r["layer"] for r in rows if "D" in r["outside"]), None)
    first_p = next((r["layer"] for r in rows if "P" in r["outside"]), None)
    print("  first layer demoted leaves the floor : %s" % first_d)
    print("  first layer promoted leaves the floor: %s" % first_p)
    n_out = sum(1 for r in rows if r["outside"])
    print("  layers where at least one set is outside: %d of %d" % (n_out, n_hs))

    print("\n  BASE-PROBABILITY BANDS at the output (lacan's -0.33 lives here):")
    print("  %-14s %5s %8s %8s %8s" % ("p_base band","n","L8","L20","L32"))
    for b in sorted(by_band):
        ws = by_band[b]
        if len(ws) < 3: continue
        lab = "1e-%d..1e-%d" % (b, b+1)
        print("  %-14s %5d %+8.2f %+8.2f %+8.2f"
              % (lab, len(ws), statistics.median(g(w,8) for w in ws),
                 statistics.median(g(w,20) for w in ws),
                 statistics.median(g(w,32) for w in ws)))
    print("\n  took %.1f s" % (time.perf_counter() - t0))
    print("  NOTHING IS DECLARED DEAD BY THIS RUN.")
    if a.json:
        json.dump({"prompt": a.prompt, "rows": rows, "demoted": dem,
                   "promoted": pro, "unmoved": unm, "k": k, "boot": a.boot},
                  open(a.json,"w"), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
