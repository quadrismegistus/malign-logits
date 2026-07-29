"""Add the last-token hidden state to multi-token entries already stored.

    uv run .venv/bin/python scripts/f13_backfill_last.py

`last` was added to the pass at docket [607]/[609] while the ladder was mid-run,
and the ladder held the code as loaded, so nothing in that run carries it. This
backfills WITHOUT re-running the ladder.

It touches 4,128 of 115,768 entries. For a SINGLE-token word `mean`, `first` and
`last` are the same vector by identity -- one token, nothing to pool -- so only
n_tok > 1 needs anything, which is why a five-minute backfill replaces a two-hour
restart. That arithmetic is the whole reason no restart happened.
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA
from malign_logits.cache import open_stash
from transformers import AutoModelForCausalLM, AutoTokenizer

store = open_stash(os.path.join(PATH_DATA, "raw", "cache", "preop_embeddings"))
todo = {}
for k in store.keys():
    if not isinstance(k, dict):
        continue
    v = store[k]
    if int(v["n_tok"]) > 1 and "last" not in v:
        todo.setdefault(k["model"], []).append(k)
print(f"backfill: {sum(len(v) for v in todo.values()):,} entries over {len(todo)} models",
      flush=True)

for model_id, keys in sorted(todo.items(), key=lambda kv: -len(kv[1])):
    print(f"\n=== {model_id} ({len(keys):,}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True).to("mps").eval()
    t0 = time.time()
    for i, k in enumerate(keys, 1):
        rec = store[k]
        plen = len(tok.encode(k["prompt"]))
        ids = tok.encode(k["prompt"] + " " + rec["word"], return_tensors="pt").to("mps")
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        span = torch.stack([h[0, plen:, :] for h in hs])
        rec["last"] = span[:, -1, :].to(torch.float16).cpu().numpy()
        store[k] = rec
        if i % 500 == 0:
            print(f"  {i:,}/{len(keys):,}  {i/(time.time()-t0):.1f} it/s", flush=True)
    del model
    torch.mps.empty_cache()
    print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)
print("\nBACKFILL COMPLETE", flush=True)
