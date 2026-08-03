#!/usr/bin/env python3
"""THE LEAK BOUND AT (EDGE x LANGUAGE STRATUM). [3756]/[3759].

[3756] promotes the language split to a DECLARED PER-STRATUM READOUT carrying the
full primary machinery -- including the worst-case bias column. zh words ride
different token trees, and theta-truncation interacts with vocabulary
granularity, so **the leak is free to differ by stratum at identical n**. A
per-edge column cannot serve a per-stratum readout.

Same measurement as `leak_flip_rate_44.py`, grouped by (edge x stratum).

THE LEAK. A faller is `Q < 0.5 P`. Falling far enough puts a word under theta,
where it is unscored and reads EXACTLY ZERO; its mass is filed in the residual,
which `movement()` carries as a NON-FALLER -- on the survivor side of the split
the null's ratio is made of.

    R = 1 - sum_fallers Q     too LARGE by the unscored fallers' post mass
    ratio = R / S             too large
    tail_excess               MORE NEGATIVE -- the direction arm A predicts

Bound is WORST CASE and capped by the post arm's own unresolved bucket:

    dR = min( n_unscored_fallers * theta , Q_residual )

NOTHING HERE TOUCHES `tail_excess`. These are ratio-level bounds and riser
classification counts -- protected computations (1), (2), (3) are not computed
and not seen by this script.

    python scripts/leak_flip_rate_by_stratum.py
"""
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meta", "scripts"))

from malign_logits.cache import get_cache                      # noqa: E402
from malign_logits.movement import (movement, word_probs,      # noqa: E402
                                    CANONICAL, RESIDUAL_KEY)
from malign_logits.prompts import Prompts                      # noqa: E402
import m01_concentration as CC                                 # noqa: E402

THETA = 0.001
OUT = os.path.join(ROOT, "data", "leak_flip_rate_by_stratum.json")
CJK = re.compile(r"[一-鿿]")


def stimulus_texts():
    """Distinct stimuli, sentinels out. Second identities collapse by set()."""
    out = set()
    for p in Prompts().all():
        t = p if isinstance(p, str) else (getattr(p, "text", None) or str(p))
        if not re.match(r"^<<<.*>>>$", t):
            out.add(t)
    return out


def mid(o):
    return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)


def measure(pre, post, prompts):
    """One (edge, stratum) row. Returns the record, or None if no cells."""
    n_cell = n_fall = n_zero = n_ris = n_flip = 0
    leaks, rels, shifts, capped = [], [], [], 0
    for pr in prompts:
        try:
            A, B = word_probs(pre, pr), word_probs(post, pr)
        except ValueError:
            continue                       # refused-and-named cell; not a primary
        if A is None or B is None:
            continue
        n_cell += 1
        P = {**A.probs, RESIDUAL_KEY: A.residual}
        Q = {**B.probs, RESIDUAL_KEY: B.residual}
        m = movement(P, Q)
        fall = set(m.fallers)
        nz = sum(1 for f in fall if Q.get(f, 0.0) == 0.0)
        n_fall += len(fall)
        n_zero += nz
        R = 1.0 - sum(Q.get(f, 0.0) for f in fall)
        S = sum(P.get(k, 0.0) for k in set(P) | set(Q) if k not in fall)
        if S <= 0 or R <= 0:
            continue
        naive = nz * THETA
        dR = min(naive, B.residual)        # the residual cap, [3755].3
        if dR < naive:
            capped += 1
        leaks.append(dR)
        rels.append(dR / R)
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
    if not n_cell:
        return None
    return {"cells": n_cell, "fallers": n_fall, "fallers_Q_zero": n_zero,
            "frac_fallers_zero": n_zero / n_fall if n_fall else None,
            "cells_where_cap_binds": capped,
            "leak_median": st.median(leaks) if leaks else None,
            "leak_max": max(leaks) if leaks else None,
            "rel_leak_median": st.median(rels) if rels else None,
            "ratio_shift_max": max(shifts) if shifts else None,
            "riser_candidates": n_ris, "flips": n_flip,
            "flip_rate": n_flip / n_ris if n_ris else None}


def main():
    stim = stimulus_texts()
    zh = {t for t in stim if CJK.search(t)}
    print(f"stimuli {len(stim)}   zh {len(zh)}   en {len(stim) - len(zh)}")

    cm = get_cache()
    have = {}
    for k in cm._stash("true_word_probs").keys():
        if isinstance(k, dict):
            have.setdefault(k["model"], set()).add(k["prompt"])

    _p, models, _h, _d = CC.frozen_population()
    edges, _ = CC.operation_edges(models)

    rows = []
    for ei, (fam, _pos, step) in enumerate(edges, 1):
        pre, post = mid(step.pre), mid(step.post)
        both = have.get(pre, set()) & have.get(post, set()) & stim
        for strat, sel in (("zh", both & zh), ("en", both - zh)):
            rec = measure(pre, post, sorted(sel))
            if rec is None:
                print(f"  [{ei:>2}] {strat}  {fam:<24} NO CELLS")
                continue
            rec = {"edge": ei, "stratum": strat, "family": fam,
                   "pre": pre, "post": post, **rec}
            rows.append(rec)
            print(f"  [{ei:>2}/{len(edges)}] {strat}  {fam:<24} "
                  f"cells {rec['cells']:>5}  "
                  f"Q=0 {100 * (rec['frac_fallers_zero'] or 0):>5.1f}%  "
                  f"flip {100 * (rec['flip_rate'] or 0):>6.3f}%", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)

    print("\n=== BY STRATUM")
    for strat in ("zh", "en"):
        fr = [r["flip_rate"] for r in rows
              if r["stratum"] == strat and r["flip_rate"] is not None]
        fz = [r["frac_fallers_zero"] for r in rows
              if r["stratum"] == strat and r["frac_fallers_zero"] is not None]
        if not fr:
            continue
        print(f"  {strat}: rows {len(fr):>3}   "
              f"fallers-at-zero median {100 * st.median(fz):>5.1f}% "
              f"max {100 * max(fz):>5.1f}%   "
              f"flip median {100 * st.median(fr):.3f}% MAX {100 * max(fr):.3f}%")
    print(f"\nrows {len(rows)}   wrote {OUT}")


if __name__ == "__main__":
    main()
