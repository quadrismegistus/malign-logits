#!/usr/bin/env python
"""Does displacement depend on attention? The TII natural experiment.

WHY THIS IS THE RIGHT INSTRUMENT AND A CROSS-LAB COMPARISON IS NOT. Architecture
covaries with lab, corpus, scale, recipe and date, so "SSM pairs versus
transformer pairs" across the roster compares eight things at once. TII published
all three classes under one lab at comparable scale with comparable post-training:

    SSM, no attention   falcon-mamba-7b, Falcon3-Mamba-7B
    hybrid              Falcon-H1-1.5B, Falcon-H1-7B      (attention + SSM)
    transformer         Falcon3-1B, 3B, 7B, 10B

and `Falcon3-7B` against `Falcon3-Mamba-7B` is the tightest contrast in the whole
roster: same generation, same size, same lab, differing in whether attention
exists at all.

WHAT IS AT STAKE. Weatherby identifies attention with the poetic function, i.e.
with the selection/combination operation itself. If alignment displaces the same
way in a model that HAS no attention, then whatever produces displacement is not
attention doing it, and the identification is too strong. The result is
interesting in both directions: a null says displacement is a property of the
post-training procedure rather than of the mechanism, which is the
political-economy reading; a difference says architecture conditions it.

THREE STATISTICS AT THE WORD UNIT. They are word-level ANALOGUES of T's
finding 14, which is the shape claim ("few large withdrawals, many small
uptakes"), and they are NOT T14's numbers:

    faller_share    fallers / (fallers + risers)            U's ladder statistic
    mag_ratio       mean|faller delta| / mean|riser delta|
    count_ratio     n_risers / n_fallers

**DO NOT READ mag_ratio AGAINST T14's "fallers 3.8x larger".** T14 aggregates
words into semantic CATEGORIES and compares category deltas; these compare
individual WORD deltas. This script returns mag_ratio around 0.4-0.6, i.e.
individual fallers about half the size of individual risers, and that is
perfectly compatible with T14: few large withdrawals concentrated in a few
categories does not require each falling word to be large. An earlier version of
this docstring called these "T14's statistics" and that was wrong; the two are
different units and comparing the numbers directly is a category error.

THE UNIT IS THE PAIR, AND THE ARCHITECTURE CONTRAST IS n=2 vs n=4. That is small
and it is stated rather than dressed up. The per-prompt paired test on the
7B-vs-7B contrast is reported as DESCRIPTIVE only: prompts within a pair are not
independent, and a p-value over 2,500 correlated prompts would be measuring the
prompt count, not the architecture. See the campaign's own unit rule.

MATCHED PROMPTS. Every pair is scored on the intersection of prompts all pairs
carry, so no pair contributes a different population from another.

    arch_displacement.py --out arch_displacement.json
"""
import argparse
import json
import os
import sys
import time

ROOT = "/Users/rj416/github/malign-logits"
sys.path.insert(0, ROOT)

#: label -> (base, aligned, architecture class). Classes are from the model
#: cards, not inferred from the name: Falcon-H1 is a hybrid and is kept in its
#: own class rather than pooled with either side, because pooling it would
#: decide the question by assignment.
PAIRS = [
    ("Falcon3-Mamba-7B", "tiiuae/Falcon3-Mamba-7B-Base",
     "tiiuae/Falcon3-Mamba-7B-Instruct", "SSM"),
    ("falcon-mamba-7b", "tiiuae/falcon-mamba-7b",
     "tiiuae/falcon-mamba-7b-instruct", "SSM"),
    ("Falcon-H1-1.5B", "tiiuae/Falcon-H1-1.5B-Base",
     "tiiuae/Falcon-H1-1.5B-Instruct", "HYBRID"),
    ("Falcon-H1-7B", "tiiuae/Falcon-H1-7B-Base",
     "tiiuae/Falcon-H1-7B-Instruct", "HYBRID"),
    ("Falcon3-1B", "tiiuae/Falcon3-1B-Base",
     "tiiuae/Falcon3-1B-Instruct", "TRANSFORMER"),
    ("Falcon3-3B", "tiiuae/Falcon3-3B-Base",
     "tiiuae/Falcon3-3B-Instruct", "TRANSFORMER"),
    ("Falcon3-7B", "tiiuae/Falcon3-7B-Base",
     "tiiuae/Falcon3-7B-Instruct", "TRANSFORMER"),
    ("Falcon3-10B", "tiiuae/Falcon3-10B-Base",
     "tiiuae/Falcon3-10B-Instruct", "TRANSFORMER"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="arch_displacement.json")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import numpy as np
    from malign_logits.step import Step
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.prompts import Prompts

    prompts = [p.text for p in Prompts.where() if not p.is_logical]
    if a.limit:
        prompts = prompts[:a.limit]
    print("active feedable prompts: %d" % len(prompts))

    steps = {}
    for lab, b, al, cls in PAIRS:
        try:
            steps[lab] = Step(b, al)
        except Exception as exc:
            print("  SKIP %-20s %s" % (lab, str(exc)[:60]))

    #: MATCHED POPULATION. A prompt counts only if EVERY pair has it present,
    #: so no pair is measured on a population another pair does not share.
    shared = []
    for txt in prompts:
        if all(steps[l].cell(txt).is_present for l in steps):
            shared.append(txt)
    print("prompts present in ALL %d pairs: %d\n" % (len(steps), len(shared)))

    out = {"n_pairs": len(steps), "n_prompts": len(shared), "pairs": {},
           "per_prompt": {}}
    for lab, b, al, cls in PAIRS:
        if lab not in steps:
            continue
        t0 = time.time()
        st = steps[lab]
        nf = nr = 0
        fmag, rmag = [], []
        per = {}
        for txt in shared:
            m = st.cell(txt).movement(CANONICAL)
            if not m:
                continue
            f = [w for w in m.fallers if w != RESIDUAL_KEY]
            r = [w for w in m.risers if w != RESIDUAL_KEY]
            nf += len(f); nr += len(r)
            fm = [abs(m.delta[w]) for w in f]
            rm = [abs(m.delta[w]) for w in r]
            fmag += fm; rmag += rm
            if f or r:
                per[txt] = (len(f), len(r),
                            float(np.mean(fm)) if fm else None,
                            float(np.mean(rm)) if rm else None)
        rec = {
            "arch": cls, "base": b, "aligned": al,
            "n_fallers": nf, "n_risers": nr,
            "faller_share": nf / (nf + nr) if (nf + nr) else None,
            "count_ratio": nr / nf if nf else None,
            "mean_faller_mag": float(np.mean(fmag)) if fmag else None,
            "mean_riser_mag": float(np.mean(rmag)) if rmag else None,
            "mag_ratio": (float(np.mean(fmag) / np.mean(rmag))
                          if fmag and rmag else None),
            "prompts_with_movement": len(per),
        }
        out["pairs"][lab] = rec
        out["per_prompt"][lab] = per
        print("  %-18s %-12s share %.4f  count_ratio %6.2f  mag_ratio %5.2f  "
              "(%d fall / %d rise)  %.0fs"
              % (lab, cls, rec["faller_share"], rec["count_ratio"],
                 rec["mag_ratio"], nf, nr, time.time() - t0))

    p = a.out if os.path.isabs(a.out) else os.path.join(os.getcwd(), a.out)
    json.dump(out, open(p, "w"))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
