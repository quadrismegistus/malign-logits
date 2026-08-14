"""PILOT: the logit lens on one contradiction triple, one model pair.

    uv run python lens_pilot_contradiction.py

Plan: `meta/M02_frame_exit/plans/contradiction_lens.md`. **This is an instrument
shakedown and NOT evidence.** Scope set by RH: one triple, one pair, chosen
because it is the LARGEST base-to-aligned superposition collapse in the roster
(Amber ratio 0.725 -> AmberSafe 1.580, against a neutralization null of 1.006).
Selecting the most favourable case is legitimate for checking that a measure
resolves anything at all and illegitimate as evidence of anything. Amber's
aligned arm is also past neutralization toward resolution, where most aligned
arms sit AT the null, so it is atypical twice over.

WHAT IT ANSWERS: does the measure separate the three states at all, does the
depth profile show anything, and does `LayerReadout.verify` fire.
WHAT IT DOES NOT ANSWER: anything about contradiction.

THE MEASURE, per layer L of the BOTH prompt:

    A_mass(L) = mass on A-set,  B_mass(L) = mass on B-set

where A-set is the top-k of the POLE_A prompt's own final-layer distribution
minus its intersection with pole_b's, and vice versa. **Two masses, never
subtracted into one number** -- superposition is both elevated, resolution is
one, neutralization is neither, and a single scalar cannot separate the first
from the third. That is the defect this measure exists to fix.

THE NULL is A/B-sets from a DIFFERENT, content-disjoint, SAME-FRAME group scored
on this group's BOTH prompt. f11_loyal's frame is "and chose to"; f11_captive
(free/captive) and f11_reason (rational/irrational) share it and share no pole
content. Without it, "elevated" has no referent -- the lesson the ratio had to
learn the hard way.

DECLARED BEFORE RUNNING, so none is a choice made on the curves:
    k = 20        the pole-set size
    WORD level    via `twp.expand_layers`, `rule_version 3`, theta=0.001 -- the
                  SAME rule as the ratio, so the two instruments are comparable.
                  The first draft of this pilot used raw token softmax and was
                  not (RH caught it).
    BOS           `expand_layers` resolves it under the registry's own per-model
                  policy; never a hand-rolled encode.

WHY expand_layers AND NOT A HAND-ROLLED LayerReadout LOOP. The first draft
projected EVERY layer including the last with `LayerReadout`. `expand_layers`
uses `FinalReadout` at the last layer instead, because `head(hidden[-1])` is
mathematically the model's logits and numerically is NOT -- ~1e-2 away in logit
space at fp16, propagating into every mass. Reading the model's own logits there
means **the final layer reproduces the stored twp cell by construction**, which
is free validation the hand-rolled version threw away.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

from malign_logits.twp import (expand_layers, boundary_mask, bos_policy_for,  # noqa: E402
                                THETA)

K = 20
PAIR = ("LLM360/Amber", "LLM360/AmberSafe")
TARGET = "f11_loyal"
NULLS = ("f11_captive", "f11_reason")


def groups():
    Q = json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))
    return {g["group"]: g for g in Q["quintuplets"] if g["status"] != "RETIRED"}


def run_model(mid, prompts):
    """{prompt: {layer: {word: mass}}} via twp's own by-layer expansion."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    bmask = boundary_mask(tok, model.config.vocab_size)
    try:
        policy = bos_policy_for(mid)
    except Exception:
        policy = "inherited"
    out, first = {}, True
    for p in prompts:
        nL = model.config.num_hidden_layers + 1     # hidden_states is N+1 long
        res, stats = expand_layers(model, tok, p, "cpu", bmask,
                                   layers=_layers(nL),
                                   theta=THETA, bos_policy=policy, verify=True)
        if first:
            #: head_err is the one-line check: head(hidden[-1]) must BE the
            #: model's logits. It is reported, never assumed.
            print("   head_err=%r  policy=%s  n_hidden=%d  passes=%d  cost_vs_twp=%.1fx"
                  % (stats.get("head_err"), policy, stats["n_hidden"],
                     stats["passes"], stats.get("cost_vs_twp", float("nan"))))
            first = False
        out[p] = {l: (_by_word(v[0]), v[1]) for l, v in res.items()}
    del model
    return out


def _by_word(words):
    """Sum the (word, FIRST TOKEN) partition down to a word.

    **`expand_layers` returns `words[(surface, t1)]`, not `words[surface]`.**
    twp rows partition a word over its possible first tokens and MUST BE SUMMED;
    the ClickHouse table happens to carry one t1 per word because the summation
    already happened before ingest, but the in-memory structure does not.
    Treating the key as a word silently reads one token's share as the whole
    word's mass -- and the first draft of this pilot did exactly that, in a
    script written the same night the rule was being catalogued.
    """
    out = {}
    for k, m in words.items():
        w = k[0] if isinstance(k, tuple) else k
        out[w] = out.get(w, 0.0) + m
    return out


def _layers(nL):
    """The layer subset, DECLARED, and why it is a subset.

    **`expand_layers` walks a frontier that is the UNION of live prefixes across
    the requested layers, so its cost is set by the MOST DIFFUSE layer in the
    set.** Layer 0 is the embeddings: at theta=0.001 its readout is near-uniform
    and clears the threshold on a large number of tokens, and every one of
    MAX_DEPTH passes then carries that frontier for every layer. Asking for all
    33 layers did not return in 7.5 minutes on one prompt.

    So: five layers at declared relative depths, embeddings EXCLUDED. Excluding
    layer 0 costs nothing the lens could read anyway -- the readout basis is the
    unembedding, and the embedding layer is the furthest from it.

    This is a declared choice made BEFORE seeing any curve, and it is the pilot's
    only deviation from the plan's measure.
    """
    last = nL - 1
    want = [0.125, 0.25, 0.5, 0.75, 1.0]
    return sorted({max(1, int(round(d * last))) for d in want})


def sets_from(words_a, words_b, k=K):
    """Top-k WORDS of each pole's own final layer, with the overlap removed."""
    ta = {w for w, _ in sorted(words_a.items(), key=lambda x: -x[1])[:k]}
    tb = {w for w, _ in sorted(words_b.items(), key=lambda x: -x[1])[:k]}
    return sorted(ta - tb), sorted(tb - ta)


def main():
    G = groups()
    g = G[TARGET]
    prompts = [g["pole_a"], g["pole_b"], g["both"]]
    for nm in NULLS:
        prompts += [G[nm]["pole_a"], G[nm]["pole_b"]]
    prompts = list(dict.fromkeys(prompts))

    print("PILOT %s -> %s   group %s   k=%d" % (PAIR[0], PAIR[1], TARGET, K))
    print("   %s" % g["both"])
    #: **CACHE THE PER-LAYER DISTRIBUTIONS, NOT ONLY THE DERIVED MASSES.**
    #: The forward passes are the whole cost (~12 min for this cell) and the
    #: first version persisted only A_mass/B_mass, so the k-sweep that killed
    #: the top-k measure had to fall back to final-layer twp, and validating
    #: the replacement measure would have meant paying the cost again. Both the
    #: word dict AND the residual are cached: `tail` is what distinguishes "no
    #: mass" from "below theta", and a cache that dropped it would reintroduce
    #: the absent-vs-empty ambiguity on every later read.
    cache = os.path.join(CAMP, "results", "lens_pilot_layers.json")
    if os.path.exists(cache):
        raw = json.load(open(cache))
        res = {m: {p: {int(l): (d["words"], d["resid"]) for l, d in pv.items()}
                   for p, pv in mv.items()} for m, mv in raw.items()}
        print("   reusing %s" % os.path.relpath(cache, ROOT))
    else:
        res = {}
        for mid in PAIR:
            print("\n%s" % mid)
            res[mid] = run_model(mid, prompts)
        with open(cache, "w") as fh:
            json.dump({m: {p: {str(l): {"words": w, "resid": r}
                               for l, (w, r) in pv.items()}
                           for p, pv in mv.items()} for m, mv in res.items()}, fh)
        print("   cached %s" % os.path.relpath(cache, ROOT))

    print("\n" + "=" * 78)
    print("A_mass / B_mass ON THE BOTH PROMPT, BY RELATIVE DEPTH")
    print("=" * 78)
    rows = []
    for mid, arm in zip(PAIR, ("base", "aligned")):
        P = res[mid]
        last = max(P[g["both"]])
        A, B = sets_from(P[g["pole_a"]][last][0], P[g["pole_b"]][last][0])
        both = {l: v[0] for l, v in P[g["both"]].items()}
        resid = {l: v[1] for l, v in P[g["both"]].items()}
        nL = last + 1
        nullsets = [sets_from(P[G[nm]["pole_a"]][last][0], P[G[nm]["pole_b"]][last][0])
                    for nm in NULLS]
        mass = lambda d, ws: float(sum(d.get(w, 0.0) for w in ws))
        print("\n  %-8s |A-set|=%d |B-set|=%d   layers=%d" % (arm, len(A), len(B), nL))
        print("    A-set: %s" % ", ".join(A[:8]))
        print("    B-set: %s" % ", ".join(B[:8]))
        #: **`tail` IS THE POINT.** A_mass of 0.0 can mean "the model gave it
        #: no mass" or "it fell below theta=0.001 and is ABSENT from the dict".
        #: Those are different facts and dict.get(w, 0.0) conflates them. tail
        #: is the mass below threshold: near 1.0 means the layer is unreadable
        #: at this theta and every 0.0 beside it is uninterpretable.
        print("    %-7s %9s %9s %11s %11s %8s %7s"
              % ("depth", "A_mass", "B_mass", "null_A", "null_B", "tail", "nwords"))
        #: NO DISPLAY FILTER. The previous one (`abs(d - round(d,1)) > 0.04`)
        #: was a leftover from the all-33-layers draft and silently dropped
        #: depths 0.25 and 0.75 -- two of the five layers computed -- from the
        #: printout AND from the CSV, because rows.append sat inside it.
        for L in sorted(both):
            d = L / (nL - 1)
            na = float(np.mean([mass(both[L], x[0]) for x in nullsets]))
            nb = float(np.mean([mass(both[L], x[1]) for x in nullsets]))
            tl = resid[L].get("tail", float("nan"))
            print("    %-7.2f %9.4f %9.4f %11.4f %11.4f %8.4f %7d"
                  % (d, mass(both[L], A), mass(both[L], B), na, nb, tl, len(both[L])))
            rows.append((arm, L, d, mass(both[L], A), mass(both[L], B), na, nb,
                         tl, len(both[L])))
    import pandas as pd
    out = os.path.join(CAMP, "results", "lens_pilot_contradiction.csv")
    pd.DataFrame(rows, columns=["arm", "layer", "depth", "A_mass", "B_mass",
                                "null_A", "null_B", "tail",
                                "n_words"]).to_csv(out, index=False)
    print("\nwrote %s" % os.path.relpath(out, ROOT))
    print("\nREAD THIS AS: does the pair of masses separate, and where in depth.")
    print("It is one triple on the roster's most extreme pair and says nothing")
    print("about contradiction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
