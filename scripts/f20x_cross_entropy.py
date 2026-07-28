"""Cross-scored passage entropy: is the regressor a property of the MODEL, or of
the model and its own output jointly?

`f20x_passage_entropy.py` scores each model on its OWN completions. That mixes two
things: (a) the distribution changed under alignment, and (b) the TEXT changed under
alignment. An aligned model no more certain in general, but producing text it happens
to be certain about, would read as tightening. It also means x and y are computed
over the same passages.

This scores each model on its PARTNER ARM'S completions, so both arms of a lineage
are measured over a common text set and the difference is a model property alone.
Combined with the own-text numbers it gives the full 2x2 per lineage:

    ent(base, base_text)      ent(base, aligned_text)
    ent(aligned, base_text)   ent(aligned, aligned_text)

Own-text is the diagonal; this run fills the off-diagonal.

    uv run .venv/bin/python scripts/f20x_cross_entropy.py [--n 120]
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

OUT = os.path.join(PATH_DATA, "f20x_cross_entropy.csv")
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
def score(model_id, texts, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device, trust_remote_code=True)
    model.eval()
    vals = []
    for q, t in texts:
        pids = tok.encode(f"Q: {q}\nA:", add_special_tokens=True)
        cids = tok.encode(str(t), add_special_tokens=False)
        if not cids:
            continue
        logits = model(torch.tensor([pids + cids], device=model.device)).logits[0].float()
        lp = torch.log_softmax(logits[len(pids) - 1: len(pids) - 1 + len(cids)], dim=-1)
        vals.append(float((-(lp.exp() * lp).sum(-1)).mean()))
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    keep = roster()
    # partner arm within each lineage: base <-> its terminal aligned arm
    partner = {}
    for bm, g in keep.groupby("base_model_id"):
        als = sorted(set(g[g.arm != "base"].model_id))
        if not als:
            continue
        for al in als:
            partner[bm] = al
            partner[al] = bm

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT).model_id)
        print(f"resuming: {len(done)} already scored")
    todo = [m for m in sorted(partner) if m not in done]
    print(f"{len(todo)} models to cross-score, n={a.n}\n", flush=True)

    for i, mid in enumerate(todo, 1):
        rows = keep[keep.model_id == partner[mid]]
        sub = rows.sample(min(a.n, len(rows)), random_state=20260728)
        texts = list(zip(sub.question, sub.text))
        try:
            e, k = score(mid, texts, a.device)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] FAIL {mid}: {type(exc).__name__}: {exc}", flush=True)
            e, k = np.nan, 0
        pd.DataFrame([dict(model_id=mid, text_from=partner[mid],
                           cross_token_entropy=e, n_scored=k)]).to_csv(
            OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"[{i}/{len(todo)}] {mid} on {partner[mid]}'s text: {e:.4f} (n={k})", flush=True)


if __name__ == "__main__":
    main()
