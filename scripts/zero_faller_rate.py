#!/usr/bin/env python3
"""ZERO-FALLER RATE ON N's DECLARED POPULATION. [3784], a KNOWN ANSWER.

N's §3 carries **14.55% (16,583 of 113,959)**, measured over 44 x 2,590 — the
SUPERSET, before the stimulus rule existed (second identities deduplicated,
sentinel excluded). The declared population is **44 x 2,578 = 113,432**, and the
registration says the rate *"is NOT this number and the producer reports it from
the analysed population."*

**This script does not supply the registration's figure. It supplies a KNOWN
ANSWER the producer is checked against** — a second derivation, by a different
seat, in a different script, computed before the producer exists. If the two
disagree, one of them is wrong and the disagreement is the finding.

A zero-faller cell is one where `movement()` returns no fallers: no mass
departed, so the mass-migration claim has nothing to land. §6.5 excludes them
from arm A and counts them in the run record.

**Reports the rate per edge and per stratum, never one pooled number** — the
same rule §4.1's companion column follows, for the same reason.

    python scripts/zero_faller_rate.py
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
from malign_logits.movement import movement, word_probs, RESIDUAL_KEY  # noqa: E402
from malign_logits.prompts import Prompts                       # noqa: E402
import m01_concentration as CC                                  # noqa: E402

OUT = os.path.join(ROOT, "data", "zero_faller_rate.json")
CJK = re.compile(r"[一-鿿]")


def stimulus_texts():
    out = set()
    for p in Prompts().all():
        t = p if isinstance(p, str) else (getattr(p, "text", None) or str(p))
        if not re.match(r"^<<<.*>>>$", t):
            out.add(t)
    return out


def mid(o):
    return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)


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

    rows, refused, missing = [], 0, 0
    for ei, (fam, _pos, step) in enumerate(edges, 1):
        pre, post = mid(step.pre), mid(step.post)
        both = have.get(pre, set()) & have.get(post, set()) & stim
        for strat, sel in (("zh", both & zh), ("en", both - zh)):
            n = z = 0
            for pr in sorted(sel):
                try:
                    A, B = word_probs(pre, pr), word_probs(post, pr)
                except ValueError:
                    refused += 1
                    continue
                if A is None or B is None:
                    missing += 1
                    continue
                n += 1
                m = movement({**A.probs, RESIDUAL_KEY: A.residual},
                             {**B.probs, RESIDUAL_KEY: B.residual})
                if not m.fallers:
                    z += 1
            rows.append({"edge": ei, "stratum": strat, "family": fam,
                         "pre": pre, "post": post, "cells": n,
                         "zero_faller": z, "rate": z / n if n else None})
            print(f"  [{ei:>2}/{len(edges)}] {strat}  {fam:<24} "
                  f"cells {n:>5}  zero-faller {z:>5}  "
                  f"{100 * (z / n if n else 0):>6.2f}%", flush=True)
            json.dump(rows, open(OUT, "w"), indent=2)

    tot_c = sum(r["cells"] for r in rows)
    tot_z = sum(r["zero_faller"] for r in rows)
    print(f"\n=== KNOWN ANSWER, declared population")
    print(f"  cells analysed        {tot_c:,}   (declared 113,432)")
    print(f"  refused (malformed)   {refused}      missing {missing}")
    print(f"  ZERO-FALLER CELLS     {tot_z:,}")
    print(f"  POOLED RATE           {100 * tot_z / tot_c:.2f}%"
          f"   (superset figure was 14.55% of 113,959)")
    for strat in ("zh", "en"):
        R = [r for r in rows if r["stratum"] == strat and r["rate"] is not None]
        c = sum(r["cells"] for r in R)
        z = sum(r["zero_faller"] for r in R)
        print(f"  {strat}: pooled {100 * z / c:.2f}%   per-edge median "
              f"{100 * st.median([r['rate'] for r in R]):.2f}%   "
              f"min {100 * min(r['rate'] for r in R):.2f}%   "
              f"max {100 * max(r['rate'] for r in R):.2f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
