"""Passage-level entropy: the control that a single-position regressor cannot be.

WHY. The rung-2-vs-3 control asks whether referential stabilisation survives what
alignment's general tightening predicts. It was first run with NEXT-TOKEN entropy at
one position, then with retained beam mass over a 10-token horizon. Both are
point-ish measures against a PASSAGE-coded outcome, and the composite's mediation
went 0.5% -> 7.1% between one position and ten, which is the wrong direction to
extrapolate from. This measures the thing itself: mean per-token entropy over each
model's OWN completions, teacher-forced, at the length the coding actually used.

It also fixes an axis the beam measure could not: retained beam mass is concentration
under a MODE-SEEKING search, while the coded completions are SAMPLED. This scores the
sampled text directly.

METHOD. One forward pass per completion. Entropy is taken over completion positions
only, never the prompt. Models are loaded one at a time and freed; results append
after every model so an interrupted run keeps everything already computed -- this
project has lost finished work to interruption before.

    uv run .venv/bin/python scripts/f20x_passage_entropy.py [--n 120] [--limit N]
"""
import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

OUT = os.path.join(PATH_DATA, "f20x_passage_entropy.csv")
ALIGNED = ("ego", "superego", "reinforced_superego")


def roster():
    c = pd.read_parquet(os.path.join(PATH_DATA, "f20x_codings.parquet"),
                        columns=["family", "arm", "model_id", "base_model_id",
                                 "question", "text"])
    c = c[c.family != "olmo-think"].copy()
    term = {}
    for f, g in c[c.arm.isin(ALIGNED)].groupby("family"):
        for a in reversed(ALIGNED):
            if a in set(g.arm):
                term[f] = a
                break
    is_term = np.array([term.get(r.family) == r.arm for r in c.itertuples()])
    return c[(c.arm == "base").to_numpy() | is_term]


@torch.no_grad()
def model_entropy(model_id, rows, n, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # trust_remote_code is needed on BOTH calls. Omitting it here cost four
    # models (both m-a-p lineages) on the first run: the model call had it, the
    # tokenizer call did not, so loading stopped at an interactive prompt that a
    # background process auto-declines. The error names the repo, not the call.
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device, trust_remote_code=True)
    model.eval()
    sub = rows.sample(min(n, len(rows)), random_state=20260728)
    vals = []
    for r in sub.itertuples():
        prompt = f"Q: {r.question}\nA:"
        pids = tok.encode(prompt, add_special_tokens=True)
        cids = tok.encode(str(r.text), add_special_tokens=False)
        if not cids:
            continue
        ids = torch.tensor([pids + cids], device=model.device)
        logits = model(ids).logits[0].float()
        # positions predicting the completion tokens
        lg = logits[len(pids) - 1: len(pids) - 1 + len(cids)]
        lp = torch.log_softmax(lg, dim=-1)
        ent = -(lp.exp() * lp).sum(-1)
        vals.append(float(ent.mean()))
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return float(np.mean(vals)) if vals else np.nan, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="completions sampled per model")
    ap.add_argument("--limit", type=int, default=0, help="stop after N models (smoke)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    keep = roster()
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT).model_id)
        print(f"resuming: {len(done)} models already computed")
    todo = [m for m in sorted(keep.model_id.unique()) if m not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} models to score, n={a.n} completions each\n", flush=True)

    for i, mid in enumerate(todo, 1):
        rows = keep[keep.model_id == mid]
        try:
            e, k = model_entropy(mid, rows, a.n, a.device)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] FAIL {mid}: {type(exc).__name__}: {exc}", flush=True)
            e, k = np.nan, 0
        rec = pd.DataFrame([dict(model_id=mid, mean_token_entropy=e, n_scored=k,
                                 arm=rows.arm.iloc[0], family=rows.family.iloc[0],
                                 base_model_id=rows.base_model_id.iloc[0])])
        rec.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"[{i}/{len(todo)}] {mid}  entropy={e:.4f}  n={k}", flush=True)


if __name__ == "__main__":
    main()
