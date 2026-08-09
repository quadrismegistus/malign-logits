#!/usr/bin/env python
"""twp_word_residual.py — the per-layer gap with the base-rate trend removed.

    scripts/twp_word_residual.py --prompt "She was so angry she wanted to"

## THE CONFOUND THIS EXISTS FOR, MEASURED NOT ASSUMED

`twp_word_floor.py` showed the per-layer gap depends on a word's OUTPUT base
probability, in the same direction as the demoted effect:

    p_base band      n     L32 median gap
    1e-1..1e-2      17        +0.17
    1e-2..1e-3      82        +0.04
    1e-3..1e-4       6        -0.35

The demoted set is SELECTED by an output ratio and is therefore enriched in
high-p_base words, so part of its +0.49 could be that gradient. This is lacan's
-0.33 nuisance floor arriving in a new quantity, and the standing rule is that a
contrast published without its floor is unscaled.

## TWO CONTROLS, BECAUSE THEY FAIL DIFFERENTLY

**MATCHED** -- for each demoted word draw a non-demoted word from the same
p_base decile, resample for a band. Interpretable, and throws data away: with
106 words some deciles are thin, and a thin decile matches badly and silently.

**RESIDUALISED** -- at each layer regress gap on log10(p_base at OUTPUT) across
ALL words, and compare residuals. Uses everything, and assumes the trend is
linear in log p, which it may not be.

Agreement between the two is the point. Either alone is one specification.

**REGRESS ON OUTPUT p_base, NOT SAME-LAYER p_base.** The confound is that the
SET was selected using an output quantity. Residualising on same-layer
probability would remove the effect itself wherever alignment demotes by
lowering p at that layer -- controlling away the thing under test.

## RH's STANDING INSTRUCTION, 2026-08-09

**"I don't think it's the definitive null so don't declare anything dead if
null."** Carried. This removes ONE confound. A surviving effect is not proven; a
vanishing one is not dead -- it would mean the effect is not separable from base
rate BY THIS CONTROL, on ONE prompt, under a lens whose mid-stack status lacan's
head-free projections have yet to arbitrate.
"""
import argparse, json, math, os, random, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
THETA = 0.001


def ols(xs, ys):
    """slope, intercept. Plain least squares; no library, no surprises."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="She was so angry she wanted to")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from twp_word_depth import score_words

    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    finalp, scored, keep = {}, {}, {}
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
        keep[tag] = (m, tok, dev, bmask)
    vocab = sorted(vocab)
    for tag in ("base", "aligned"):
        m, tok, dev, bmask = keep[tag]
        scored[tag], _ = score_words(m, tok, a.prompt, vocab, dev, bmask, twp, torch)
        del m
        try: torch.mps.empty_cache()
        except Exception: pass

    common = sorted(set(scored['base']) & set(scored['aligned']))
    n_hs = max(scored['base'][common[0]]) + 1
    fin = n_hs - 1
    g = lambda w, l: math.log10(max(scored['base'][w][l], 1e-30) /
                                max(scored['aligned'][w][l], 1e-30))
    lp = {w: math.log10(max(finalp['base'].get(w, 1e-6), 1e-6)) for w in common}

    dem = [w for w in common if g(w, fin) > math.log10(2)]
    pro = [w for w in common if g(w, fin) < -math.log10(2)]
    unm = [w for w in common if abs(g(w, fin)) < 0.10]
    other = [w for w in common if w not in set(dem) | set(pro)]

    print("prompt %r\n%s vs %s" % (a.prompt, a.base, a.aligned))
    print("*** ONE confound removed. Surviving != proven; vanishing != dead. ***\n")
    print("  words %d | demoted %d | promoted %d | unmoved %d | non-demoted pool %d"
          % (len(common), len(dem), len(pro), len(unm), len(other)))

    #: how strong IS the nuisance, per layer
    print("\n  the nuisance itself: OLS slope of gap on log10(p_base at output)")
    print("  %-6s %8s %8s" % ("layer", "slope", "at L"))
    slopes = {}
    for l in range(n_hs):
        b, _c = ols([lp[w] for w in common], [g(w, l) for w in common])
        slopes[l] = b
        if l % 6 == 0 or l == fin:
            print("  %-6d %+8.3f" % (l, b))

    #: MATCHED control by decile of p_base, resampled
    decile = lambda w: int(max(0, min(5, -lp[w])))
    pool = defaultdict(list)
    for w in other: pool[decile(w)].append(w)
    thin = [d for d in {decile(w) for w in dem} if len(pool.get(d, [])) < 3]

    rows = []
    print("\n  %-6s | %-24s | %-24s" % ("", "DEMOTED", "PROMOTED"))
    print("  %-6s | %7s %15s | %7s %15s"
          % ("layer", "resid", "matched 95%", "resid", "matched 95%"))
    for l in range(n_hs):
        b, c = ols([lp[w] for w in common], [g(w, l) for w in common])
        resid = {w: g(w, l) - (b * lp[w] + c) for w in common}
        rd = statistics.median(resid[w] for w in dem) if dem else float('nan')
        rp = statistics.median(resid[w] for w in pro) if pro else float('nan')
        def matched_band(target):
            meds = []
            for _ in range(a.boot):
                pick = []
                for w in target:
                    cand = pool.get(decile(w)) or other
                    pick.append(rng.choice(cand))
                meds.append(statistics.median(resid[x] for x in pick))
            meds.sort()
            return meds[int(.025*len(meds))], meds[int(.975*len(meds))]
        dlo, dhi = matched_band(dem) if dem else (float('nan'),)*2
        plo, phi = matched_band(pro) if pro else (float('nan'),)*2
        rows.append({"layer": l, "resid_dem": rd, "resid_pro": rp,
                     "dem_lo": dlo, "dem_hi": dhi, "pro_lo": plo, "pro_hi": phi,
                     "slope": b,
                     "dem_out": bool(rd > dhi or rd < dlo),
                     "pro_out": bool(rp > phi or rp < plo)})
        if l % 2 == 0 or l >= n_hs - 2:
            print("  %-6d | %+7.2f %+7.2f..%+.2f %s | %+7.2f %+7.2f..%+.2f %s"
                  % (l, rd, dlo, dhi, "*" if rows[-1]["dem_out"] else " ",
                     rp, plo, phi, "*" if rows[-1]["pro_out"] else " "))

    print("\n  * = residual median outside the p_base-MATCHED 95%% band")
    run = 0; best = 0; start = None; bstart = None
    for r in rows:
        if r["dem_out"] and r["pro_out"]:
            run += 1; start = start if start is not None else r["layer"]
            if run > best: best, bstart = run, start
        else:
            run = 0; start = None
    print("  longest run with BOTH outside: %d layers, from L%s"
          % (best, bstart if bstart is not None else "-"))
    print("  final layer: demoted %+.2f (band %+.2f..%+.2f), promoted %+.2f (band %+.2f..%+.2f)"
          % (rows[fin]["resid_dem"], rows[fin]["dem_lo"], rows[fin]["dem_hi"],
             rows[fin]["resid_pro"], rows[fin]["pro_lo"], rows[fin]["pro_hi"]))
    if thin:
        print("  ** THIN DECILES (fewer than 3 controls): %s -- matching there is"
              " nominal and the band is optimistic **" % thin)
    print("\n  took %.1f s\n  NOTHING IS DECLARED DEAD BY THIS RUN." % (time.perf_counter()-t0))
    if a.json:
        json.dump({"prompt": a.prompt, "rows": rows, "demoted": dem,
                   "promoted": pro, "slopes": slopes, "thin_deciles": thin},
                  open(a.json,"w"), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
