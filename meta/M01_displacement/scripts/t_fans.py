"""Sibling fans: hold the base AND the pre-checkpoint constant, vary one thing.

    uv run python t_fans.py
    uv run python t_fans.py --limit 25

RH's design, and it is a better control than the cross-family ladder in
`t_ladder.py`. There, every family differs in base, pretraining corpus, scale
and recipe at once, so a difference between families names nothing. Here the
pre-checkpoint is SHARED and exactly one variable moves.

TWO FANS.

  METHOD FAN -- `archangel_sft_pythia2-8b` -> {dpo, kto, ppo, slic}. One base
  (pythia-2.8b), one SFT, four preference methods diverging from it. Isolates
  the METHOD. The campaign has never compared preference methods against each
  other with the SFT held fixed.

  DATA FAN -- `Llama-3.1-8B` -> five Tulu SFT variants: full, no-math,
  no-persona, no-safety, no-wildchat. One base, one recipe, five training
  corpora. Isolates the DATA, and **`no-safety-data` is a direct ablation of
  the mechanism** -- if displacement is a safety-training artefact, dropping
  safety data should weaken it. Nothing in this campaign has tested that; every
  previous result is a correlate.

A REGISTRY-LABELLING ARTEFACT HID MOST OF THE METHOD FAN, and it is worth
recording because it silently shrank an earlier analysis. The shared SFT
checkpoint is filed under `family=archangel-dpo`, while the kto, ppo and slic
checkpoints sit in `archangel-kto`, `archangel-ppo` and `archangel-slic`. A
per-family ladder search therefore finds no base and no SFT for three of the
four methods and skips them: `t_ladder.py` saw one Archangel family where there
are four. **The families here are declared, not derived**, precisely because
derivation from the `family` field is what lost them.

WHAT IS MEASURED, per arm, exactly as in the ladder: word-level JS over the
union support (residual kept), CANONICAL fallers and risers, and the Jaccard
BETWEEN ARMS -- do two preference methods, or two data ablations, move the SAME
words? That last one is the question a fan can answer and a ladder cannot.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FANS = {
    "method": {
        "pre": "ContextualAI/archangel_sft_pythia2-8b",
        "base": "EleutherAI/pythia-2.8b",
        "arms": {"dpo": "ContextualAI/archangel_sft-dpo_pythia2-8b",
                 "kto": "ContextualAI/archangel_sft-kto_pythia2-8b",
                 "ppo": "ContextualAI/archangel_sft-ppo_pythia2-8b",
                 "slic": "ContextualAI/archangel_sft-slic_pythia2-8b"},
    },
    "data": {
        "pre": "meta-llama/Llama-3.1-8B",
        "base": None,
        "arms": {"full": "allenai/Llama-3.1-Tulu-3-8B-SFT",
                 "no-math": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data",
                 "no-persona": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data",
                 "no-safety": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data",
                 "no-wildchat": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data"},
    },
}


def js(p, q):
    """JS in bits over the union support. Residual kept: it is real untruncated
    mass, and dropping it renormalises each arm over its own support."""
    keys = set(p) | set(q)
    a = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    a, b = a / a.sum(), b / b.sum()
    m = 0.5 * (a + b)
    kl = lambda x: float(np.sum(x[(x > 0) & (m > 0)] * np.log2(x[(x > 0) & (m > 0)] / m[(x > 0) & (m > 0)])))
    return 0.5 * kl(a) + 0.5 * kl(b)


def measure(pre, post, texts):
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.step import Step
    st = Step(Checkpoint(pre), Checkpoint(post))
    rows = []
    for t in texts:
        c = st.cell(t)
        if not c.is_present:
            continue
        m = c.movement(CANONICAL)
        rows.append(dict(prompt=t, js=js(c.pre.probs, c.post.probs),
                         fallers=frozenset(w for w in (m.fallers if m else []) if w != RESIDUAL_KEY),
                         risers=frozenset(w for w in (m.risers if m else []) if w != RESIDUAL_KEY)))
    D = pd.DataFrame(rows)
    if len(D):
        D["n_fall"] = D["fallers"].map(len)
        D["n_rise"] = D["risers"].map(len)
    return D


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from malign_logits.prompts import Prompts
    texts = sorted({p.text for p in Prompts.all(status="ACTIVE")
                    if all(ord(c) < 128 for c in p.text) and not getattr(p, "is_logical", False)})
    if a.limit:
        texts = texts[:a.limit]
    print("prompts: %d\n" % len(texts))

    rows, jrows = [], []
    for fan, spec in FANS.items():
        print("=" * 74)
        print("%s FAN   pre = %s" % (fan.upper(), spec["pre"]))
        print("=" * 74)
        if spec["base"]:
            D = measure(spec["base"], spec["pre"], texts)
            if len(D):
                print("  %-14s %6d cells  JS %.4f  fall %5.2f  rise %5.2f   (the shared rung)"
                      % ("base>sft", len(D), D["js"].mean(), D["n_fall"].mean(), D["n_rise"].mean()))
        got = {}
        print("  %-14s %6s %9s %7s %7s %10s" % ("arm", "cells", "JS", "fall", "rise", "fallshare"))
        for name, ck in spec["arms"].items():
            D = measure(spec["pre"], ck, texts)
            if not len(D):
                print("  %-14s   no cells" % name)
                continue
            got[name] = D
            fs = D["n_fall"].sum() / max(D["n_fall"].sum() + D["n_rise"].sum(), 1)
            rows.append(dict(fan=fan, arm=name, checkpoint=ck, cells=len(D),
                             js=float(D["js"].mean()), fall=float(D["n_fall"].mean()),
                             rise=float(D["n_rise"].mean()), faller_share=float(fs)))
            print("  %-14s %6d %9.4f %7.2f %7.2f %9.1f%%"
                  % (name, len(D), D["js"].mean(), D["n_fall"].mean(), D["n_rise"].mean(), 100 * fs), flush=True)
        #: THE QUESTION A FAN CAN ANSWER AND A LADDER CANNOT: do two arms
        #: diverging from ONE checkpoint move the same words?
        if len(got) > 1:
            print("\n  Jaccard BETWEEN ARMS (same pre-checkpoint, one variable changed):")
            print("  %-26s %11s %11s" % ("pair", "fallers", "risers"))
            for x, y in itertools.combinations(sorted(got), 2):
                M = got[x].merge(got[y], on="prompt", suffixes=("_a", "_b"))
                assert len(M) <= min(len(got[x]), len(got[y])), "duplicate prompt keys"
                if not len(M):
                    continue
                fj = float(np.nanmean([jac(p, q) for p, q in zip(M["fallers_a"], M["fallers_b"])]))
                rj = float(np.nanmean([jac(p, q) for p, q in zip(M["risers_a"], M["risers_b"])]))
                jrows.append(dict(fan=fan, a=x, b=y, faller_jaccard=fj, riser_jaccard=rj, n=len(M)))
                print("  %-26s %11.4f %11.4f" % ("%s vs %s" % (x, y), fj, rj))
        print()

    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "t_fans.csv"), index=False)
    pd.DataFrame(jrows).to_csv(os.path.join(OUT, "t_fans_jaccard.csv"), index=False)

    d = S[S["fan"] == "data"]
    if {"full", "no-safety"} <= set(d["arm"]):
        f = d[d["arm"] == "full"].iloc[0]
        n = d[d["arm"] == "no-safety"].iloc[0]
        print("THE ABLATION THAT MATTERS: full against no-safety-data")
        print("  JS            %.4f  vs  %.4f   (%.0f%% of full)" % (f["js"], n["js"], 100 * n["js"] / f["js"]))
        print("  fallers/site  %.2f  vs  %.2f" % (f["fall"], n["fall"]))
        print("  faller share  %.1f%% vs  %.1f%%" % (100 * f["faller_share"], 100 * n["faller_share"]))
        print("  -> if displacement were a safety-data artefact, no-safety should be markedly lower.")
    print("\nwrote t_fans.csv, t_fans_jaccard.csv")


if __name__ == "__main__":
    main()
