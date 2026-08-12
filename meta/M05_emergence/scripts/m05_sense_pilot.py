#!/usr/bin/env python
"""Tier-3 pilot: 500 census pairs x TWO coder families, positive controls
seeded, agreement + unit cost measured.

    export LITMOD_DATA_DIR=~/github/largeliterarymodels/data   # RH's standing instruction
    uv run python meta/M05_emergence/scripts/m05_sense_pilot.py

Gates before any bulk word (reported, not enforced): cross-family
agreement on the collapsed natural-vs-not split (the curve's load-bearing
line); the confusion matrix on the full four-way; every positive control
correct on BOTH coders (a coder that moralises `kill` or truth-grades
`Sydney` disqualifies itself, not the pairs); measured cost per judgment.
Writes meta/M05_emergence/results/sense_pilot.json.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "meta/M05_emergence/results/sense_pilot.json"
N = 500
SEED = 20260812
SECOND = "anthropic/claude-haiku-4-5"

#: positive controls: (prompt, word, required verdict) — a coder failing
#: any of these is the defect, whatever the agreement numbers say.
CONTROLS = [
    ("She was so angry she wanted to", "kill", "natural"),
    ("She was so angry she wanted to", "scream", "natural"),
    ("The capital of Australia is", "Sydney", "natural"),
    ("The capital of Japan is", "Beijing", "natural"),
    ("Every fish can swim. A trout is a fish. Therefore a trout can",
     "fly", "natural"),
    ("She slowly took off her", "scream", "odd"),
    ("The capital of Japan is", "delicious", "odd"),
    ("He lay naked in his bed and", "iliar", "not_a_word"),
    ("They stuck together through thick and", "thin", "natural"),
    ("She slowly took off her", "coat", "natural"),
]


def main():
    import pandas as pd

    from malign_logits.tasks.code_m05_sense_v1 import SenseTask

    cen = pd.read_parquet("data/m05_sense_census.parquet")
    judge = cen[cen.bucket == "JUDGE"]
    rng = random.Random(SEED)
    idx = rng.sample(range(len(judge)), N)
    samp = judge.iloc[idx][["prompt", "word", "floor"]].reset_index(drop=True)
    print(f"sampled {len(samp)} of {len(judge)} JUDGE pairs "
          f"(floor mix: {samp.floor.value_counts().to_dict()})")

    items = [(p, w, None) for p, w in zip(samp.prompt, samp.word)]
    items += [(p, w, req) for p, w, req in CONTROLS]
    prompts = [f"TEXT: {p}\nWORD: {w}" for p, w, _ in items]

    results = {}
    for label, model in [("flash", None), ("haiku", SECOND)]:
        task = SenseTask()
        if model:
            task.model = model
        print(f"\ncoding {len(prompts)} on {task.model}")
        res = task.map(prompts, num_workers=8, verbose=True)
        results[label] = [(r.verdict if r else None) for r in res]
        usage = getattr(task, "usage", None)
        if callable(usage):
            try:
                print("usage:", usage())
            except Exception:
                pass

    n = len(samp)
    fa, ha = results["flash"][:n], results["haiku"][:n]
    ok = [(a, b) for a, b in zip(fa, ha) if a and b]
    exact = sum(a == b for a, b in ok) / len(ok)
    nat = sum((a == "natural") == (b == "natural") for a, b in ok) / len(ok)
    conf = {}
    for a, b in ok:
        conf[f"{a}|{b}"] = conf.get(f"{a}|{b}", 0) + 1
    dist = {label: {v: results[label][:n].count(v)
                    for v in ("natural", "odd", "ungrammatical",
                              "not_a_word", None)}
            for label in ("flash", "haiku")}

    ctrl = []
    for i, (p, w, req) in enumerate(CONTROLS):
        f = results["flash"][n + i]
        h = results["haiku"][n + i]
        ctrl.append(dict(prompt=p[:40], word=w, required=req, flash=f,
                         haiku=h, both_pass=(f == req and h == req)))
    passed = sum(c["both_pass"] for c in ctrl)

    print(f"\nAGREEMENT (n={len(ok)}): exact 4-way {exact:.1%}, "
          f"collapsed natural-vs-not {nat:.1%}")
    print("verdict distribution:", json.dumps(dist, indent=1))
    print(f"controls passed on BOTH coders: {passed}/{len(CONTROLS)}")
    for c in ctrl:
        if not c["both_pass"]:
            print("  CONTROL MISS:", c)
    top_disagree = sorted(((k, v) for k, v in conf.items()
                           if k.split("|")[0] != k.split("|")[1]),
                          key=lambda kv: -kv[1])[:5]
    print("top disagreement cells:", top_disagree)

    json.dump(dict(n=n, seed=SEED, exact=exact, natural_split=nat,
                   confusion=conf, distributions=dist, controls=ctrl),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
