#!/usr/bin/env python
"""twp_onset_canonical.py — WHEN, for the campaign's OWN risers and fallers.

    scripts/twp_onset_canonical.py --base LLM360/Amber --aligned LLM360/AmberSafe

**SELECTION COMES FROM THE CORPUS, MEASUREMENT FROM LIVE LAYERS.** The word sets
are `Step(Checkpoint(a), Checkpoint(b)).cell(prompt).movement(CANONICAL)` --
read from the twp store, through the object that already guards them:

    mixed rule_version RAISES (a v1 arm against a v3 arm books an INSTRUMENT
        change as alignment movement)
    the partition is SUMMED, not overwritten (rebuilding it from rows drops
        mass on 20% of payloads and up to 99.9% on the smallest)
    `prompt` carries the catalogue row, so stratification is one attribute away

Earlier versions of this analysis recomputed the output distribution live and
applied a bare ratio (>2 falls, <0.5 rises). That was wrong twice over: it
bypassed those guards, and **it is not what this project means by a riser.**
CANONICAL tests risers against the RENORMALISATION NULL -- every survivor gains
when a faller's mass is removed, and the null is what separates redistribution
from bookkeeping. On these prompts it cuts the sets by a third to a half.

**THE ASYMMETRY IS THE RULE'S AND IS PRESERVED.** Risers are tested against the
null; FALLERS ARE NOT -- a faller is a bare ratio rule. Nothing here may
describe fallers as "beyond renormalisation", and the two sets are not
symmetric evidence even when they are printed side by side.

Only the per-layer scoring is computed live, because the store holds no
per-layer data. The final layer of that scoring reproduces twp, which is the
free check that the live path and the stored cell agree.
"""
import argparse, json, math, os, random, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


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
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM as AM
    from malign_logits import twp
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.movement import CANONICAL
    from twp_word_depth import score_words

    rng = random.Random(a.seed)
    t0 = time.perf_counter()

    #: ---- SELECTION, from the store -------------------------------------
    step = Step(Checkpoint(a.base), Checkpoint(a.aligned))
    sets = {}
    print("STEP %r  direction=%s" % (step.label, step.direction))
    for p in a.prompts:
        c = step.cell(p)
        m = c.movement(CANONICAL)
        pre = c.pre.probs
        flats = [w for w in pre
                 if w not in set(m.fallers) | set(m.risers)
                 and pre[w] >= CANONICAL.min_prob]
        sets[p] = {"fallers": list(m.fallers), "risers": list(m.risers),
                   "flats": flats, "pre": dict(pre),
                   "resid": m.diagnostics.get("residual_share", float("nan")),
                   "domain": getattr(getattr(c, "prompt", None), "domain", None)}
        print("  %-40s fallers %3d risers %3d flats %3d | resid %.3f | domain %s"
              % (p[:40], len(m.fallers), len(m.risers), len(flats),
                 sets[p]["resid"], sets[p]["domain"]))

    #: ---- MEASUREMENT, live, per layer ----------------------------------
    scored = {p: {} for p in a.prompts}
    held = {}
    for tag, mid in (("base", a.base), ("aligned", a.aligned)):
        tok, _ = twp.load_tokenizer(mid); dev = twp.pick_device()
        m = AM.from_pretrained(mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
        bmask = twp.boundary_mask(tok, m.config.vocab_size)
        for p in a.prompts:
            v = sorted(set(sets[p]["fallers"]) | set(sets[p]["risers"])
                       | set(sets[p]["flats"]))
            scored[p][tag], _ = score_words(m, tok, p, v, dev, bmask, twp, torch)
        del m
        try: torch.mps.empty_cache()
        except Exception: pass

    out = {}
    for p in a.prompts:
        S = scored[p]
        common = sorted(set(S['base']) & set(S['aligned']))
        if not common: print("\n=== %r: nothing scorable" % p); continue
        n_hs = max(S['base'][common[0]]) + 1
        g = lambda w, l: math.log10(max(S['base'][w][l], 1e-30) /
                                    max(S['aligned'][w][l], 1e-30))
        lp = {w: math.log10(max(sets[p]["pre"].get(w, 1e-6), 1e-6)) for w in common}
        fal = [w for w in sets[p]["fallers"] if w in S['base'] and w in S['aligned']]
        ris = [w for w in sets[p]["risers"] if w in S['base'] and w in S['aligned']]
        flt = [w for w in sets[p]["flats"] if w in S['base'] and w in S['aligned']]
        resid = {}
        for l in range(n_hs):
            b, c = ols([lp[w] for w in common], [g(w, l) for w in common])
            resid[l] = {w: g(w, l) - (b*lp[w] + c) for w in common}

        def onset_and_mde(words, sign):
            """(onset, MDE at that layer or at 2/3 depth). None with its MDE."""
            if len(words) < 3 or len(flt) < 3: return None, float("nan")
            run, first = 0, None
            mde_at = None
            for l in range(n_hs):
                meds = sorted(statistics.median(resid[l][x] for x in
                              rng.choices(flt, k=len(words)))
                              for _ in range(a.boot))
                lo, hi = meds[int(.025*len(meds))], meds[int(.975*len(meds))]
                if l == int(0.66*n_hs): mde_at = (hi-lo)/2.0
                v = statistics.median(resid[l][w] for w in words)
                ok = (v > hi) if sign > 0 else (v < lo)
                run = run+1 if ok else 0
                if run >= 3 and first is None: first = l-2
            return first, mde_at

        of, mf = onset_and_mde(fal, +1)
        orr, mr = onset_and_mde(ris, -1)
        fin = n_hs-1
        out[p] = {"n_fal": len(fal), "n_ris": len(ris), "n_flat": len(flt),
                  "onset_fallers": of, "onset_risers": orr,
                  "mde_fallers": mf, "mde_risers": mr, "layers": n_hs,
                  "final_fal": statistics.median(resid[fin][w] for w in fal) if fal else None,
                  "final_ris": statistics.median(resid[fin][w] for w in ris) if ris else None}
        print("\n=== %r  (%d layers)" % (p, n_hs))
        print("  FALLERS n=%-3d onset %-5s  final resid %+.2f   MDE@2/3 %.3f"
              % (len(fal), of, out[p]["final_fal"] or float('nan'), mf))
        print("  RISERS  n=%-3d onset %-5s  final resid %+.2f   MDE@2/3 %.3f"
              % (len(ris), orr, out[p]["final_ris"] or float('nan'), mr))
        print("  control = FLATS (n=%d), the rule's own third class" % len(flt))

    print("\n" + "="*68)
    print("  %-40s %8s %8s" % ("prompt", "fallers", "risers"))
    for p, r in out.items():
        print("  %-40s %8s %8s" % (p[:40], r["onset_fallers"], r["onset_risers"]))
    print("\n  onset = first layer of a 3-layer run outside the FLATS band.")
    print("  A None carries its MDE above; a null without one is not a null.")
    print("  RISERS are null-tested by CANONICAL, FALLERS ARE NOT -- these two")
    print("  columns are not symmetric evidence.")
    print("  took %.1f s" % (time.perf_counter()-t0))
    if a.json:
        json.dump({"step": step.label, "base": a.base, "aligned": a.aligned,
                   "results": out}, open(a.json, "w"), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
