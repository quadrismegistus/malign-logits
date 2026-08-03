#!/usr/bin/env python3
"""THE 44-EDGE FLIP RATE AND LEAK BOUND. [3750].3(a), load-bearing for N's §4.

The truncation leak: a faller with `Q < 0.5 P` can fall under theta, go
unscored, and read EXACTLY ZERO. Its mass is then filed in the residual, which
`movement()` carries as a NON-FALLER -- on the survivor side of the split the
null's ratio is made of.

    R = 1 - sum_fallers Q      too LARGE by the unscored fallers' post mass
    ratio = R / S              too large
    tail_excess = Q_res - P_res * ratio    MORE NEGATIVE

Arm A predicts negative tail_excess one-sided, so the artifact pushes toward the
prediction. This measures how far, per edge.

Bound is WORST CASE: each unscored faller assumed to hold exactly theta.
"""
import json, os, sys, statistics as st
sys.path.insert(0, "/Users/rj416/github/malign-logits")
sys.path.insert(0, "/Users/rj416/github/malign-logits/scripts")
sys.path.insert(0, "/Users/rj416/github/malign-logits/meta/scripts")
from malign_logits.cache import get_cache
from malign_logits.movement import movement, word_probs, CANONICAL, RESIDUAL_KEY
import m01_concentration as CC

TH = 0.001
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flip_rate_44.json")

_p, models, _h, _d = CC.frozen_population()
edges, _ = CC.operation_edges(models)
cm = get_cache()

def mid(o):
    return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

rows = []
for ei, (fam, pos, step) in enumerate(edges, 1):
    pre, post = mid(step.pre), mid(step.post)
    prompts = sorted({k["prompt"] for k in cm._stash("true_word_probs").keys()
                      if isinstance(k, dict) and k.get("model") == pre})
    n_cell = n_fall = n_zero = n_ris = n_flip = 0
    leaks, rels, shifts = [], [], []
    for pr in prompts:
        try:
            A, B = word_probs(pre, pr), word_probs(post, pr)
        except ValueError:
            continue                       # malformed cell: refused and named, skipped here
        if A is None or B is None:
            continue
        n_cell += 1
        P = {**A.probs, RESIDUAL_KEY: A.residual}
        Q = {**B.probs, RESIDUAL_KEY: B.residual}
        m = movement(P, Q)
        fall = set(m.fallers)
        nz = sum(1 for f in fall if Q.get(f, 0.0) == 0.0)
        n_fall += len(fall); n_zero += nz
        R = 1.0 - sum(Q.get(f, 0.0) for f in fall)
        S = sum(P.get(k, 0.0) for k in set(P) | set(Q) if k not in fall)
        if S <= 0 or R <= 0:
            continue
        dR = nz * TH
        leaks.append(dR); rels.append(dR / R)
        i1, i2 = R / S, (R - dR) / S
        shifts.append(abs(i1 - i2) / i1)
        for k in set(P) | set(Q):
            if k in fall or k == RESIDUAL_KEY:
                continue
            p, q = P.get(k, 0.0), Q.get(k, 0.0)
            if not (max(p, q) > CANONICAL.min_prob and (q - p) > CANONICAL.delta):
                continue
            n_ris += 1
            if (q > p * i1) != (q > p * i2):
                n_flip += 1
    rec = {"edge": ei, "family": fam, "pre": pre, "post": post, "cells": n_cell,
           "fallers": n_fall, "fallers_Q_zero": n_zero,
           "frac_fallers_zero": n_zero / n_fall if n_fall else None,
           "leak_median": st.median(leaks) if leaks else None,
           "leak_max": max(leaks) if leaks else None,
           "rel_leak_median": st.median(rels) if rels else None,
           "ratio_shift_max": max(shifts) if shifts else None,
           "riser_candidates": n_ris, "flips": n_flip,
           "flip_rate": n_flip / n_ris if n_ris else None}
    rows.append(rec)
    print(f"  [{ei:>2}/{len(edges)}] {fam:<26} cells {n_cell:>5}  "
          f"Q=0 {100*(rec['frac_fallers_zero'] or 0):>5.1f}%  "
          f"flip {100*(rec['flip_rate'] or 0):>6.3f}%", flush=True)
    json.dump(rows, open(OUT, "w"), indent=2)

fr = [r["flip_rate"] for r in rows if r["flip_rate"] is not None]
fz = [r["frac_fallers_zero"] for r in rows if r["frac_fallers_zero"] is not None]
print("\n=== 44-EDGE SUMMARY")
print(f"  edges measured        {len(rows)}")
print(f"  fallers reading Q=0   median {100*st.median(fz):.1f}%   min {100*min(fz):.1f}%   max {100*max(fz):.1f}%")
print(f"  FLIP RATE             median {100*st.median(fr):.3f}%   min {100*min(fr):.3f}%   MAX {100*max(fr):.3f}%")
print(f"\nwrote {OUT}")
