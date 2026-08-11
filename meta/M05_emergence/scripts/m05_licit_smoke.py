#!/usr/bin/env python
"""Smoke the licit-set coder (code_m05_licit_v1) on 8 adversarial prompts,
then run the three checks RH's doubt names, on real store data:

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_licit_smoke.py

  1. SATURATION: how many of the 13 classes come back licit per prompt?
     If ~all, the instrument has no dynamic range and dies here.
  2. WITNESS/TAGGER AGREEMENT: append each witness to its prompt, tag with
     the same spaCy pipeline as the artifact -- does the claimed class
     match the tagged pos_class?
  3. DISCRIMINATION ON REAL DISTRIBUTIONS: join the licit sets to
     data/m05_syntax_tags.parquet and score licit / illicit / format mass
     shares at early, mid and final rungs of BOTH ladders for these
     prompts. The instrument earns the 584-call run only if the licit
     share MOVES across training.

Smoke only: 8 prompts, nothing frozen, nothing quotable.
"""
import os
import sys

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PROMPTS = [
    "She slowly took off her",
    "The capital of Japan is",
    "They stuck together through thick and",
    "All ravens are black. Coco is a raven. Coco must be",
    "Every fish can swim. A trout is a fish. Therefore a trout can",
    "Adam Smith described the invisible hand of the",
    "After losing the game he slammed the table and said",
    "She put the key in the box and left the room. The key was still in the",
]
FORMAT_BAND = {"PUNCT", "X", "SYM"}

#: DECLARED convention equivalences for scoring and witness-checking: the
#: coder and PTB place these boundaries differently, and both placements
#: are defensible, so the join meets on the merged class rather than
#: manufacturing disagreement. PART/ADP: infinitival and particle "to"
#: (PTB TO vs IN). NUM/NOUN: bare numerals in nominal slots ("one") tag NN.
#: NOT merged: DET/ADJ -- "her the" and "her red" must stay distinguishable.
EQUIV = [{"ADP", "PART"}, {"NUM", "NOUN"}, {"AUX", "VERB"}]


def eq(a, b):
    if a == b:
        return True
    return any(a in g and b in g for g in EQUIV)


def expand(classes):
    out = set(classes)
    for g in EQUIV:
        if out & g:
            out |= g
    return out
RUNGS = {
    "olmo": [("allenai/Olmo-3-1025-7B@stage1-step1000-tokens5B", "step1000"),
             ("allenai/Olmo-3-1025-7B@stage1-step16000-tokens68B", "step16k"),
             ("allenai/Olmo-3-1025-7B", "base-main")],
    "pythia": [("EleutherAI/pythia-6.9b@step128", "step128"),
               ("EleutherAI/pythia-6.9b@step2000", "step2000"),
               ("EleutherAI/pythia-6.9b@step143000", "step143k")],
}


def main():
    import json

    import pandas as pd
    import spacy

    from malign_logits.movement import word_probs
    from malign_logits.tasks.code_m05_licit_v1 import LicitSetTask

    #: resolve the two OLMo revisions from the population (labels above are
    #: a guess at the revision string; the population file is the truth)
    pop = json.load(open("data/m05_checkpoint_population.json"))["checkpoints"]
    olmo = []
    for step, label in [(1000, "step1000"), (16000, "step16k")]:
        c = next(c for c in pop if c.get("stage") == "stage1"
                 and c.get("step") == step)
        olmo.append((f"{c['model_id']}@{c['revision']}", label))
    olmo.append(("allenai/Olmo-3-1025-7B", "base-main"))
    RUNGS["olmo"] = olmo

    task = LicitSetTask()
    print(f"coding {len(PROMPTS)} prompts on {task.model} at temp "
          f"{task.temperature}\n")
    results = task.map([f"TEXT:\n{p}" for p in PROMPTS], num_workers=4,
                       verbose=False)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    sys.path.insert(0, os.path.join(ROOT, "meta/M05_emergence/scripts"))
    from m05_syntax_tags import pos_class as pc

    tags = pd.read_parquet("data/m05_syntax_tags.parquet")
    tagmap = {(r.prompt, r.word): r.pos_class for r in tags.itertuples()}

    agree = tries = 0
    licit_sets = {}
    for p, res in zip(PROMPTS, results):
        if res is None:
            print(f"CODER FAILED on: {p!r}")
            continue
        lic = {w.pos for w in res.licit}
        marg = {w.pos for w in res.marginal}
        licit_sets[p] = (lic, marg)
        print(f'PROMPT: "{p}"')
        print(f"  frame: {res.frame}")
        print(f"  licit ({len(lic)}/13): "
              + ", ".join(f"{w.pos}:{w.example}" for w in res.licit))
        print(f"  marginal ({len(marg)}): "
              + (", ".join(f"{w.pos}:{w.example}" for w in res.marginal)
                 or "-"))
        # ---- check 2: witness/tagger agreement -------------------------
        bad = []
        for w in list(res.licit) + list(res.marginal):
            doc = nlp(f"{p} {w.example}")
            start = len(p) + 1
            toks = [t for t in doc if t.idx >= start] or [doc[-1]]
            got = pc(toks[0].tag_, toks[0].pos_)
            tries += 1
            if eq(got, w.pos):
                agree += 1
            else:
                bad.append(f"{w.pos}:{w.example} tagged {got}")
        if bad:
            print(f"  WITNESS DISAGREEMENTS: {'; '.join(bad)}")
        print()
    print(f"witness/tagger agreement: {agree}/{tries} "
          f"({agree / max(tries, 1):.0%})\n")

    # ---- check 3: discrimination on real rungs ---------------------------
    print("=" * 72)
    print("licit / illicit / format MASS SHARE of resolved mass "
          "(strict = licit only)")
    for ladder, rungs in RUNGS.items():
        print(f"\n{ladder.upper()}")
        for model, label in rungs:
            tot = {"licit": 0.0, "illicit": 0.0, "format": 0.0}
            n_cells = 0
            for p in PROMPTS:
                if p not in licit_sets:
                    continue
                wp = word_probs(model, p)
                if wp is None or wp.n_rows == 0:
                    continue
                n_cells += 1
                lic = expand(licit_sets[p][0])
                for w, prob in wp.probs.items():
                    cls = tagmap.get((p, w))
                    if cls is None:
                        doc = nlp(f"{p} {w}")
                        start = len(p) + 1
                        toks = [t for t in doc if t.idx >= start] or [doc[-1]]
                        cls = pc(toks[0].tag_, toks[0].pos_)
                    if cls in FORMAT_BAND:
                        tot["format"] += prob
                    elif cls in lic:
                        tot["licit"] += prob
                    else:
                        tot["illicit"] += prob
            s = sum(tot.values()) or 1.0
            print(f"  {label:10} cells {n_cells}/8   "
                  f"licit {tot['licit'] / s:.1%}   "
                  f"illicit {tot['illicit'] / s:.1%}   "
                  f"format {tot['format'] / s:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
