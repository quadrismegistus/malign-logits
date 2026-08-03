#!/usr/bin/env python3
"""(a) DID `cell_roles` READ CONTAMINATED RISER MEMBERSHIP? YES — AND THIS BOUNDS IT.

Ordered at [3791].1(a). `scripts/m01_norms.py::cell_roles` emits
`(word, |delta|, role)` for every faller and riser, reading `m.fallers` and
**`m.risers`** off `movement()`. Before [3777]'s fix, `_movement` put the
residual bucket in the faller set whenever the tail lost more than half its mass,
which inflated the renormalisation null and **rejected risers that the corrected
null admits**.

WHAT IS AND IS NOT AFFECTED, established by before/after capture on one edge:

    m.fallers   UNCHANGED   (0 of 300 cells moved; the bucket was stripped from
                             the reported list either way, and no WORD's faller
                             status depends on the bucket)
    |delta|     UNCHANGED   (Q - P; no ratio in it, so the WEIGHTS are clean)
    m.risers    MOVED       (22 of 33 affected cells, +0 to +5 risers each)

**So the contamination is riser MEMBERSHIP and nothing else** — and it is
one-directional: the inflated null admitted FEWER risers, so every consumer of
`cell_roles` UNDER-sampled risers in an affected cell.

This measures the affected fraction and the riser delta over N's full declared
population, per edge, so the bound is a distribution and not one pooled number.

    python scripts/audit_riser_contamination.py
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

from malign_logits.cache import get_cache                       # noqa: E402
from malign_logits.movement import (movement, word_probs,       # noqa: E402
                                    CANONICAL, RESIDUAL_KEY)
from malign_logits.prompts import Prompts                       # noqa: E402
import m01_concentration as CC                                  # noqa: E402

OUT = os.path.join(ROOT, "data", "audit_riser_contamination.json")


def stimulus_texts():
    out = set()
    for p in Prompts().all():
        t = p if isinstance(p, str) else (getattr(p, "text", None) or str(p))
        if not re.match(r"^<<<.*>>>$", t):
            out.add(t)
    return out


def mid(o):
    return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)


def contaminated_risers(P, Q, fall):
    """The riser set the OLD code produced: same rule, null from the ratio
    computed with the residual bucket IN the faller set."""
    keys = set(P) | set(Q)
    old_fall = set(fall) | {RESIDUAL_KEY}
    R = 1.0 - sum(Q.get(k, 0.0) for k in old_fall)
    S = sum(P.get(k, 0.0) for k in keys if k not in old_fall)
    if S <= 0:
        return None
    ratio = R / S
    return {k for k in keys if k not in old_fall and k != RESIDUAL_KEY
            and max(P.get(k, 0.0), Q.get(k, 0.0)) > CANONICAL.min_prob
            and (Q.get(k, 0.0) - P.get(k, 0.0)) > CANONICAL.delta
            and Q.get(k, 0.0) > P.get(k, 0.0) * ratio}


def main():
    stim = stimulus_texts()
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
        sel = sorted(have.get(pre, set()) & have.get(post, set()) & stim)
        n = aff = moved = lost = 0
        for pr in sel:
            try:
                A, B = word_probs(pre, pr), word_probs(post, pr)
            except ValueError:
                continue
            if A is None or B is None:
                continue
            n += 1
            if not (A.residual >= CANONICAL.min_prob
                    and B.residual < CANONICAL.fall_ratio * A.residual):
                continue
            aff += 1
            P = {**A.probs, RESIDUAL_KEY: A.residual}
            Q = {**B.probs, RESIDUAL_KEY: B.residual}
            m = movement(P, Q)
            old = contaminated_risers(P, Q, m.fallers)
            if old is None:
                continue
            new = set(m.risers)
            if old != new:
                moved += 1
                lost += len(new - old)          # risers the OLD code missed
        rows.append({"edge": ei, "family": fam, "pre": pre, "post": post,
                     "cells": n, "affected": aff, "riser_set_moved": moved,
                     "risers_missed_by_old_code": lost,
                     "affected_rate": aff / n if n else None})
        print(f"  [{ei:>2}/{len(edges)}] {fam:<24} cells {n:>5}  "
              f"affected {aff:>5} ({100 * (aff / n if n else 0):>5.1f}%)  "
              f"riser-set moved {moved:>5}  risers missed {lost:>6}", flush=True)
        json.dump(rows, open(OUT, "w"), indent=2)

    tc = sum(r["cells"] for r in rows)
    ta = sum(r["affected"] for r in rows)
    tm = sum(r["riser_set_moved"] for r in rows)
    tl = sum(r["risers_missed_by_old_code"] for r in rows)
    ar = [r["affected_rate"] for r in rows if r["affected_rate"] is not None]
    print(f"\n=== BOUND ON (a), declared population")
    print(f"  cells                              {tc:>8,}")
    print(f"  AFFECTED (residual met faller rule){ta:>8,}   {100 * ta / tc:.2f}% pooled")
    print(f"    per-edge rate: median {100 * st.median(ar):.2f}%  "
          f"min {100 * min(ar):.2f}%  max {100 * max(ar):.2f}%")
    print(f"  cells whose RISER SET moved        {tm:>8,}   {100 * tm / tc:.2f}% of all")
    print(f"  risers the OLD code MISSED         {tl:>8,}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
