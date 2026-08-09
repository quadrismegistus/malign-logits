#!/usr/bin/env python
"""twp_word_onset.py — WHEN does alignment demote its fallers and promote its risers?

    scripts/twp_word_onset.py --prompts-file /tmp/prompts.txt

RH's two questions, treated symmetrically:

    for the words alignment DEMOTES at the output, when does it demote them?
    for the words alignment PROMOTES at the output, when does it promote them?

**CONDITIONING ON THE FALLERS IS THE POPULATION, NOT A BIAS.** "When does
alignment demote the words it demotes" cannot be asked without selecting on the
words it demotes. That is a different question from F05's "where does
displacement happen in general", which samples all movers and answers 92-98%
depth. Both can be true: displacement can be overwhelmingly final-layer for the
average word while the words alignment actually acts on diverge earlier.

**THE CONCERN THAT DOES SURVIVE IS ENDPOINT SELECTION, AND IT IS CHECKABLE.**
Selecting on the output gap enriches for words whose gap is large and
persistent, and a persistent gap is likelier to be detectable early -- which
could pull onset earlier with nothing interesting happening. So the set is
STRATIFIED by output-gap magnitude:

    dose-response (big fallers early, marginal ones late) -> informative
    every stratum at the same onset regardless of magnitude -> the endpoint
        worry is dead
    onset tracks magnitude perfectly -> the worry is live and named

Two models are loaded ONCE each and all prompts scored against them, because
loading dominates at this scale.
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
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx
    return b, my - b*mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--aligned", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompts", nargs="*", default=[
        "She was so angry she wanted to",
        "The widow was so bereft she wanted to",
        "The child was so excited she wanted to"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--json", default="/tmp/onset.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from twp_word_depth import score_words

    rng = random.Random(a.seed)
    t0 = time.perf_counter()

    #: pass 1: the output vocabulary of each arm, per prompt
    finalp = {p: {} for p in a.prompts}
    scored = {p: {} for p in a.prompts}
    resid_mass = {}
    for tag, mid in (("base", a.base), ("aligned", a.aligned)):
        tok, _ = twp.load_tokenizer(mid)
        dev = twp.pick_device()
        m = AM.from_pretrained(mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
        bmask = twp.boundary_mask(tok, m.config.vocab_size)
        L = m.config.num_hidden_layers
        for p in a.prompts:
            twp.reset_batch()
            out, _ = twp.expand_layers(m, tok, p, dev, bmask, [L])
            agg = defaultdict(float)
            for (s, _t1), q in out[L][0].items(): agg[s] += q
            finalp[p][tag] = dict(agg)
            r = out[L][1]
            resid_mass.setdefault(p, {})[tag] = float(
                r.get('tail', 0) + r.get('drop', 0) + r.get('open', 0)
                + r.get('mojibake', 0))
        #: second sweep for the union needs BOTH arms' vocab, so scoring waits
        finalp[p].setdefault(tag, {})
        if tag == "base":
            keep = (m, tok, dev, bmask)
        else:
            #: now both vocabularies exist -- score with the aligned model, then
            #: reload nothing: the base is still resident in `keep`
            for p in a.prompts:
                vocab = sorted({w for w, q in finalp[p]['base'].items() if q >= THETA} |
                               {w for w, q in finalp[p]['aligned'].items() if q >= THETA})
                scored[p]['aligned'], _ = score_words(m, tok, p, vocab, dev, bmask, twp, torch)
            del m
            try: torch.mps.empty_cache()
            except Exception: pass
            m2, tok2, dev2, bm2 = keep
            for p in a.prompts:
                vocab = sorted({w for w, q in finalp[p]['base'].items() if q >= THETA} |
                               {w for w, q in finalp[p]['aligned'].items() if q >= THETA})
                scored[p]['base'], _ = score_words(m2, tok2, p, vocab, dev2, bm2, twp, torch)
            del m2
            try: torch.mps.empty_cache()
            except Exception: pass

    report = {}
    for p in a.prompts:
        S = scored[p]
        common = sorted(set(S['base']) & set(S['aligned']))
        n_hs = max(S['base'][common[0]]) + 1; fin = n_hs - 1
        g = lambda w, l: math.log10(max(S['base'][w][l], 1e-30) /
                                    max(S['aligned'][w][l], 1e-30))
        lp = {w: math.log10(max(finalp[p]['base'].get(w, 1e-6), 1e-6)) for w in common}
        #: **THE WORD SETS ARE THE CAMPAIGN'S, NOT MINE.** I had been using a
        #: bare output ratio (>2 falls, <0.5 rises), which is NOT what this
        #: project means by a riser: `CANONICAL` tests risers against the
        #: RENORMALISATION NULL, because every survivor gains a little when a
        #: faller's mass is removed and "the null is what separates
        #: redistribution from bookkeeping". My promoted sets were full of
        #: bookkeeping, which is the likeliest reason the promotion side was
        #: the flaky half of every result tonight.
        #:
        #: THE ASYMMETRY IS PRESERVED AND MUST BE: risers are tested against
        #: the null, FALLERS ARE NOT -- a faller is a bare ratio rule. Nothing
        #: here may describe fallers as "beyond renormalisation".
        #:
        #: Residuals are passed because the null needs total mass and
        #: true_word_probs is truncated at theta; `residual_share` says how
        #: much of the distribution the approximation rests on.
        from malign_logits.movement import movement, CANONICAL
        mv = movement({w: S['base'][w][fin] for w in common},
                      {w: S['aligned'][w][fin] for w in common},
                      rule=CANONICAL,
                      residual_pre=resid_mass[p]['base'],
                      residual_post=resid_mass[p]['aligned'])
        dem = [w for w in mv.fallers if w in set(common)]
        pro = [w for w in mv.risers if w in set(common)]
        pool = [w for w in common if w not in set(dem) | set(pro)]
        print("  [movement CANONICAL] fallers %d risers %d | residual_share %.3f"
              % (len(dem), len(pro), mv.diagnostics.get('residual_share', 0.0)))

        resid = {}
        for l in range(n_hs):
            b, c = ols([lp[w] for w in common], [g(w, l) for w in common])
            resid[l] = {w: g(w, l) - (b*lp[w] + c) for w in common}

        def onset(words, sign):
            """first layer of a 3-layer sustained run outside the matched band."""
            if len(words) < 3: return None
            run = 0
            for l in range(n_hs):
                meds = sorted(statistics.median(resid[l][x] for x in
                              rng.choices(pool, k=len(words)))
                              for _ in range(a.boot))
                lo, hi = meds[int(.025*len(meds))], meds[int(.975*len(meds))]
                v = statistics.median(resid[l][w] for w in words)
                out = (v > hi) if sign > 0 else (v < lo)
                run = run + 1 if out else 0
                if run >= 3: return l - 2
            return None

        def strata(words, sign):
            """terciles by |output gap|, smallest first."""
            ws = sorted(words, key=lambda w: abs(g(w, fin)))
            k = max(1, len(ws)//3)
            return [("marginal", ws[:k]), ("middle", ws[k:2*k]), ("largest", ws[2*k:])]

        row = {"prompt": p, "n": len(common), "n_dem": len(dem), "n_pro": len(pro),
               "layers": n_hs,
               "onset_dem": onset(dem, +1), "onset_pro": onset(pro, -1),
               "final_dem": statistics.median(resid[fin][w] for w in dem) if dem else None,
               "final_pro": statistics.median(resid[fin][w] for w in pro) if pro else None,
               "strata_dem": [(lab, len(ws), onset(ws, +1)) for lab, ws in strata(dem, +1)],
               "strata_pro": [(lab, len(ws), onset(ws, -1)) for lab, ws in strata(pro, -1)]}
        report[p] = row

        print("\n=== %r  (%d layers, %d words)" % (p, n_hs, len(common)))
        print("  %-9s n=%-3d ONSET %-6s final resid %+.2f"
              % ("DEMOTED", len(dem), row["onset_dem"], row["final_dem"] or float('nan')))
        for lab, n, o in row["strata_dem"]:
            print("       %-9s n=%-3d onset %s" % (lab, n, o))
        print("  %-9s n=%-3d ONSET %-6s final resid %+.2f"
              % ("PROMOTED", len(pro), row["onset_pro"], row["final_pro"] or float('nan')))
        for lab, n, o in row["strata_pro"]:
            print("       %-9s n=%-3d onset %s" % (lab, n, o))

    print("\n" + "="*66)
    print("  %-38s %8s %8s" % ("prompt", "dem", "pro"))
    for p, r in report.items():
        print("  %-38s %8s %8s" % (p[:38], r["onset_dem"], r["onset_pro"]))
    print("\n  onset = first layer of a 3-layer sustained run outside the")
    print("  p_base-matched 95%% band. None = never sustained.")
    print("  took %.1f s\n  NOTHING IS DECLARED DEAD BY THIS RUN." % (time.perf_counter()-t0))
    json.dump(report, open(a.json, "w"), indent=1)
    print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
