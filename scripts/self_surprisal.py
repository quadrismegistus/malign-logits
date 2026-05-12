#!/usr/bin/env python
"""Compute self-surprisal: feed each passage through the model that generated it.

For each passage in corpus_metrics.parquet, loads the generating model and
computes per-token -log P(token_i | context). This is the true information
rate — how surprised the model is by its own output.

Processes one model at a time to manage memory. Caches per-token values
to a dedicated HashStash (stash_self_surprisal). Saves passage-level
means to data/self_surprisal.csv incrementally.

Usage:
    python scripts/self_surprisal.py                    # run all
    python scripts/self_surprisal.py --family olmo      # one family
    python scripts/self_surprisal.py --resume            # skip done models
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits import MODEL_FAMILIES, PATH_DATA_RAW


STASH_PATH = os.path.join(PATH_DATA_RAW, "stash_self_surprisal")
OUTPUT_CSV = "data/self_surprisal.csv"

LAYER_TO_ATTR = {
    "base": "base",
    "ego": "ego",
    "superego": "superego",
    "instruct": "reinforced_superego",
}


def get_stash():
    from hashstash import HashStash
    return HashStash(
        root_dir=STASH_PATH,
        engine="pairtree", compress="lz4", b64=True,
    )


def compute_self_surprisal(text, prompt, model, tokenizer, device):
    """Compute per-token self-surprisal for a passage."""
    full_text = prompt + text if prompt else text
    ids = tokenizer.encode(full_text, return_tensors="pt",
                           truncation=True, max_length=1024).to(device)

    if prompt:
        prefix_ids = tokenizer.encode(prompt, return_tensors="pt")
        start_idx = prefix_ids.shape[1]
    else:
        start_idx = 1

    with torch.no_grad():
        outputs = model(ids)
        logits = outputs.logits[0].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = ids[0]

    surprisals = []
    tokens = []
    for i in range(start_idx, len(token_ids)):
        lp = float(log_probs[i - 1, token_ids[i]])
        surprisals.append(-lp)
        tokens.append(tokenizer.decode([token_ids[i]]))

    if not surprisals:
        return [], 0.0

    tok_surps = list(zip(tokens, [round(s, 4) for s in surprisals]))
    mean_surp = round(float(np.mean(surprisals)), 4)
    return tok_surps, mean_surp


def process_model(family_key, layer_name, model_id, passages_df, stash):
    """Process all passages for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16)
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    print(f"  Loaded in {time.time()-t0:.1f}s on {device}")

    results = []
    cached = 0
    computed = 0

    for _, row in tqdm(passages_df.iterrows(), total=len(passages_df),
                       desc=f"{family_key}/{layer_name}"):
        text = str(row["psg"]).rstrip()
        prompt = str(row.get("prompt", "")).strip()
        label = row.get("label", "")

        cache_key = ("self_surprisal_v1", model_id, prompt, text)
        if cache_key in stash:
            tok_surps = stash[cache_key]
            mean_surp = round(float(np.mean([v for _, v in tok_surps])), 4) if tok_surps else 0.0
            cached += 1
        else:
            tok_surps, mean_surp = compute_self_surprisal(
                text, prompt, model, tokenizer, device)
            stash[cache_key] = tok_surps
            computed += 1

        results.append({
            "family": family_key,
            "model": layer_name,
            "label": label,
            "prompt": prompt,
            "self_surprisal": mean_surp,
            "n_tokens": len(tok_surps),
        })

    print(f"  {cached} cached, {computed} computed")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default=None,
                        help="Single family to process (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models already in output CSV")
    args = parser.parse_args()

    # Load passages
    parquet_path = "data/corpus_metrics.parquet"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv("data/corpus_metrics.csv")

    # Filter to AI passages only
    human_fams = {'dreams', 'waking', 'c20_fiction', 'abstracts'}
    df = df[~df.family.isin(human_fams)].copy()
    print(f"AI passages: {len(df)}")

    stash = get_stash()

    # Load existing results for resume
    existing = set()
    if args.resume and os.path.exists(OUTPUT_CSV):
        done = pd.read_csv(OUTPUT_CSV)
        existing = set(zip(done.family, done.model))
        print(f"Resuming: {len(existing)} (family, model) pairs already done")

    families = [args.family] if args.family else list(MODEL_FAMILIES.keys())
    all_results = []

    # Load existing CSV to append to
    if os.path.exists(OUTPUT_CSV):
        all_results.append(pd.read_csv(OUTPUT_CSV))

    for fam_key in families:
        if fam_key not in MODEL_FAMILIES:
            print(f"Unknown family: {fam_key}")
            continue

        fam = MODEL_FAMILIES[fam_key]
        fam_df = df[df.family == fam_key]
        if fam_df.empty:
            print(f"\n{fam_key}: no passages, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {fam_key} ({fam.name})")
        print(f"{'='*60}")

        for layer_name, attr in LAYER_TO_ATTR.items():
            model_id = getattr(fam, attr, None)
            if model_id is None:
                continue

            if (fam_key, layer_name) in existing:
                print(f"  {layer_name}: already done, skipping")
                continue

            passages = fam_df[fam_df.model == layer_name]
            if passages.empty:
                continue

            result_df = process_model(fam_key, layer_name, model_id,
                                       passages, stash)
            all_results.append(result_df)

            # Save incrementally
            combined = pd.concat(all_results, ignore_index=True)
            # Deduplicate in case of restart
            combined = combined.drop_duplicates(
                subset=['family', 'model', 'label', 'prompt'], keep='last')
            combined.to_csv(OUTPUT_CSV, index=False)
            print(f"  Saved {OUTPUT_CSV} ({len(combined)} rows)")

    print(f"\nDone. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
