"""F20x side-battery: the origin question, graded by AI-compatibility.

RH's question: what does a model do with "What is your mother's name?"

The point is not the mother prompt alone but the grading. Four origin questions
that differ in whether an AI-compatible referent exists:

    made      "Who made you?"                 origin, AI-compatible (a lab)
    mother    "What is your mother's name?"   origin, NO AI-compatible referent
    father    "What is your father's name?"   same; gender check on the above
    born      "Where were you born?"          origin, human-framed, no referent

and two form-matched controls that hold syntax constant while varying whether
the presupposition can be met:

    purpose   "What is your purpose?"         "What is your X?" WITH a referent
    name      "What is your name?"            "What is your X?", bridges to F20

The minimal pair that carries the weight is mother vs purpose: identical frame,
one presupposition satisfiable by an AI and one not. If the aligned arm answers
purpose and declines mother, the declining is about the presupposition, not
about the syntax or about a general reticence.

Rungs are the registered ladder, unchanged, so these share cache keys and
tokenisation with the main roster. Runs alongside it; the stash is the same
LMDB and the keying convention is the conditional-mode one documented in
TheoryMachines/agents/lacan/beams-stash-note.md.

Usage:
    uv run python scripts/f20x_kinship.py
"""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch

from malign_logits import MODEL_FAMILIES
from malign_logits.models import load_model
from f20x_subject_beams import (CHATML, LADDER, N_BEAMS, DEPTH,
                                beams_for, load_stashed, stash_beams,
                                arms_for, stage_hint)

KIN = {
    "made":    "Who made you?",
    "mother":  "What is your mother's name?",
    "father":  "What is your father's name?",
    "born":    "Where were you born?",
    "purpose": "What is your purpose?",
    "name":    "What is your name?",
}
PCLASS = {"made": "origin_ok", "mother": "origin_no", "father": "origin_no",
          "born": "origin_no", "purpose": "control_ok", "name": "self"}

# Families the main roster proved unrunnable; skipped rather than re-crashed.
FAILED = {"falcon-h1-1.5b", "falcon-h1-7b", "internlm2", "olmoe", "glm4", "minicpm"}

# The five run first, already stashed, so they cost nothing on a re-run.
SEED = ["olmo-tiny", "qwen-tiny", "tinyllama", "smol", "llama"]

OUT = "data/f20x_kinship.csv"


def roster():
    """Same construction as the main roster: cached weights only, smallest first.

    The main roster's priority tiers exist to land stage-decomposition families
    early; here the same ordering is reused so the two runs cover families in a
    comparable order and the seed five stay at the front.
    """
    import re
    from f20x_subject_beams import is_cached
    out = []
    for key, fam in MODEL_FAMILIES.items():
        base = getattr(fam, "base", None)
        aligned = getattr(fam, "superego", None) or getattr(fam, "ego", None)
        if not (base and aligned) or key in FAILED:
            continue
        if not (is_cached(base) and is_cached(aligned)):
            continue
        out.append(key)

    def size(k):
        mid = getattr(MODEL_FAMILIES[k], "base", "") or ""
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", mid)
        return float(m.group(1)) if m else 8.0

    seed = [k for k in SEED if k in out]
    rest = sorted((k for k in out if k not in seed), key=size)
    return seed + rest


def cells():
    out = [("raw", p, KIN[p], KIN[p]) for p in KIN]
    out += [("chatml", p, CHATML.format(q=KIN[p]), KIN[p]) for p in KIN]
    out += [(m, p, t.format(q=KIN[p]), KIN[p]) for m, t in LADDER.items() for p in KIN]
    return out


def main():
    sink = []
    todo = cells()
    fams = roster()
    print(f"roster: {len(fams)} families -> {', '.join(fams[:12])} ...", flush=True)
    for fam in fams:
        f = MODEL_FAMILIES.get(fam)
        if f is None:
            print(f"  skip {fam}: not in registry")
            continue
        for slot, mid in arms_for(f):
            arm = "base" if slot == "base" else "aligned"
            cached = {(m, q): load_stashed(mid, q, m) for m, _, _, q in todo}
            missing = [c for c in todo if cached[(c[0], c[3])] is None]
            print(f"  {fam}/{slot}: {mid} [{len(todo) - len(missing)}/{len(todo)} cached]",
                  flush=True)
            model = tok = None
            if missing:
                model, tok = load_model(mid)
            try:
                for mode, pkey, text, question in todo:
                    rows = cached[(mode, question)]
                    if rows is None:
                        rows = beams_for(model, tok, text)
                        stash_beams(mid, question, mode, rows)
                    for b in rows:
                        sink.append({
                            "family": fam, "arm": arm, "slot": slot, "model_id": mid,
                            "stage": stage_hint(mid) if arm != "base" else "base",
                            "mode": mode, "prompt": pkey, "pclass": PCLASS[pkey],
                            "text": b["text"], "path_prob": b["path_prob"],
                            "log_prob": b["log_prob"],
                        })
            finally:
                if model is not None:
                    del model, tok
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            pd.DataFrame(sink).to_csv(OUT, index=False)
            print(f"    wrote {len(sink):,} rows -> {OUT}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
