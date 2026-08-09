#!/usr/bin/env python
"""twp_word_power.py — is a stratum's silence an ABSENCE or a POWER FAILURE?

    scripts/twp_word_power.py

`twp_word_onset.py` reported that the marginal-magnitude fallers never separate
sustainedly while the largest separate at L17-21, and I called that a
dose-response. **It is not evidence of one on its own.** If a larger endpoint gap
is simply EASIER TO DETECT EARLIER, the endpoint-selection artifact predicts the
same monotone pattern. The two are separated by asking what the instrument COULD
have seen:

    MDE(layer, n) = half-width of the p_base-matched null band for a set of
                    size n -- the smallest median residual that would land
                    outside it

    observed << MDE   -> silence is UNDERPOWERED. Nothing is learned.
    observed ~ 0, MDE small -> silence is a real absence.

**A null quotes its MDE or it is not a null** -- the campaign's standing rule,
applied to my own stratum before anyone else has to apply it for me.

The strata are n=3..13, so the expectation going in is that most of the silence
is power, not absence. Reporting it either way.
"""
import argparse, json, math, os, random, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
THETA = 0.001


def ols(xs, ys):
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    if sxx <= 0: return 0.0, my
    return (sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx,
            my - sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx*mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompts", nargs="*", default=[
        "She was so angry she wanted to",
        "The widow was so bereft she wanted to",
        "The child was so excited she wanted to"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--at", type=int, default=19, help="layer to power-check")
    ap.add_argument("--seed", type=int, default=20260809)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from twp_word_depth import score_words

    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    finalp = {p: {} for p in a.prompts}; scored = {p: {} for p in a.prompts}
    held = {}
    for tag, mid in (("base", a.base), ("aligned", a.aligned)):
        tok, _ = twp.load_tokenizer(mid); dev = twp.pick_device()
        m = AM.from_pretrained(mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
        bmask = twp.boundary_mask(tok, m.config.vocab_size)
        L = m.config.num_hidden_layers
        for p in a.prompts:
            twp.reset_batch()
            out, _ = twp.expand_layers(m, tok, p, dev, bmask, [L])
            agg = defaultdict(float)
            for (s, _t1), q in out[L][0].items(): agg[s] += q
            finalp[p][tag] = dict(agg)
        held[tag] = (m, tok, dev, bmask)
    for tag in ("base", "aligned"):
        m, tok, dev, bmask = held[tag]
        for p in a.prompts:
            vocab = sorted({w for w, q in finalp[p]['base'].items() if q >= THETA} |
                           {w for w, q in finalp[p]['aligned'].items() if q >= THETA})
            scored[p][tag], _ = score_words(m, tok, p, vocab, dev, bmask, twp, torch)
        del m
        try: torch.mps.empty_cache()
        except Exception: pass

    print("*** A NULL QUOTES ITS MDE OR IT IS NOT A NULL ***\n")
    for p in a.prompts:
        S = scored[p]
        common = sorted(set(S['base']) & set(S['aligned']))
        n_hs = max(S['base'][common[0]]) + 1; fin = n_hs - 1
        g = lambda w, l: math.log10(max(S['base'][w][l], 1e-30) /
                                    max(S['aligned'][w][l], 1e-30))
        lp = {w: math.log10(max(finalp[p]['base'].get(w, 1e-6), 1e-6)) for w in common}
        dem = [w for w in common if g(w, fin) > math.log10(2)]
        pool = [w for w in common if abs(g(w, fin)) <= math.log10(2)]
        b, c = ols([lp[w] for w in common], [g(w, a.at) for w in common])
        resid = {w: g(w, a.at) - (b*lp[w] + c) for w in common}
        ws = sorted(dem, key=lambda w: abs(g(w, fin)))
        k = max(1, len(ws)//3)
        strata = [("marginal", ws[:k]), ("middle", ws[k:2*k]), ("largest", ws[2*k:])]
        print("=== %r   layer L%d" % (p, a.at))
        print("  %-9s %3s %9s %9s %9s  %s"
              % ("stratum", "n", "observed", "MDE", "obs/MDE", "verdict"))
        for lab, s in strata:
            if not s: continue
            meds = sorted(statistics.median(resid[x] for x in rng.choices(pool, k=len(s)))
                          for _ in range(a.boot))
            lo, hi = meds[int(.025*len(meds))], meds[int(.975*len(meds))]
            mde = (hi - lo) / 2.0
            obs = statistics.median(resid[w] for w in s)
            ratio = abs(obs) / mde if mde > 0 else float('inf')
            if abs(obs) > mde: v = "DETECTED"
            elif ratio > 0.5: v = "underpowered (obs is %.0f%% of MDE)" % (100*ratio)
            else: v = "consistent with ABSENCE (obs %.0f%% of MDE)" % (100*ratio)
            print("  %-9s %3d %+9.3f %9.3f %9.2f  %s" % (lab, len(s), obs, mde, ratio, v))
        #: what fraction of the demoted set's FINAL effect would be needed
        fin_med = statistics.median(
            g(w, fin) - (lambda bc: bc[0]*lp[w]+bc[1])(
                ols([lp[x] for x in common], [g(x, fin) for x in common])) for w in dem)
        print("  final-layer demoted residual %+.2f -- an L%d effect of %.0f%% of "
              "that would be detectable in the largest stratum"
              % (fin_med, a.at,
                 100 * (lambda s: (sorted(statistics.median(resid[x] for x in
                        rng.choices(pool, k=len(s))) for _ in range(200))[195]) )(strata[-1][1]) / abs(fin_med)))
        print()
    print("  took %.1f s" % (time.perf_counter()-t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
