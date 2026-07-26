"""F38 Jakobson-space coordinates: surprisal + drift for f38_sample.csv.

Self-surprisal under Llama-3.1-8B base, topic drift via sentence embeddings.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/f38_jakobson.py
"""
import os
import sys
import time

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def compute_surprisal(model, tokenizer, opening, continuation):
    """Mean token surprisal of continuation conditioned on opening."""
    full_text = opening + " " + continuation
    input_ids = tokenizer.encode(full_text, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    opening_ids = tokenizer.encode(opening, add_special_tokens=True)
    opening_len = len(opening_ids)

    if input_ids.shape[1] <= opening_len:
        return None, None, 0

    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (seq_len, vocab)

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_surprisals = []
    for t in range(opening_len, input_ids.shape[1]):
        tid = input_ids[0, t].item()
        lp = -log_probs[t - 1, tid].item()  # surprisal = -log p
        token_surprisals.append(lp)

    if not token_surprisals:
        return None, None, 0

    return float(np.mean(token_surprisals)), float(np.std(token_surprisals)), len(token_surprisals)


def compute_drift(embedder, continuation):
    """Sentence-level embedding drift over the continuation."""
    # Split into sentences (simple)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', continuation.strip())
    sentences = [s for s in sentences if len(s.split()) >= 3]

    if len(sentences) < 2:
        return {'n_sentences': len(sentences), 'mean_drift': None, 'max_drift': None,
                'path_length': None, 'directedness': None}

    embeddings = embedder.encode(sentences, normalize_embeddings=True)

    # Pairwise consecutive cosine distances
    drifts = []
    for i in range(len(embeddings) - 1):
        cos_dist = 1 - float(np.dot(embeddings[i], embeddings[i + 1]))
        drifts.append(cos_dist)

    path_length = sum(drifts)
    direct_dist = 1 - float(np.dot(embeddings[0], embeddings[-1]))
    directedness = direct_dist / path_length if path_length > 0 else 1.0

    return {
        'n_sentences': len(sentences),
        'mean_drift': float(np.mean(drifts)),
        'max_drift': float(np.max(drifts)),
        'path_length': path_length,
        'directedness': directedness,
    }


def main():
    from malign_logits.models import load_model

    df = pd.read_csv(os.path.join(DATA_DIR, 'f38_sample.csv'))
    print(f"Loaded {len(df)} passages", flush=True)

    out_path = os.path.join(DATA_DIR, 'f38_jakobson.csv')

    # Check existing
    done = set()
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        done = set(existing.code)
        print(f"Resuming: {len(done)} done", flush=True)

    # Phase 1: surprisal under Llama-3.1-8B base
    print("Loading Llama-3.1-8B base for surprisal...", flush=True)
    model, tok = load_model('meta-llama/Llama-3.1-8B')
    t0 = time.time()

    surprisal_data = {}
    for i, row in df.iterrows():
        if row['code'] in done:
            continue
        result = compute_surprisal(model, tok, row['opening'], row['continuation'])
        surprisal_data[row['code']] = {
            'mean_surprisal': result[0],
            'std_surprisal': result[1],
            'n_tokens': result[2],
        }
        if (i + 1) % 200 == 0:
            print(f"  Surprisal: {i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Surprisal done ({time.time()-t0:.0f}s)", flush=True)
    del model, tok
    import gc; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Phase 2: drift via sentence embeddings
    print("Loading sentence embedder for drift...", flush=True)
    embedder = SentenceTransformer('BAAI/bge-m3')
    t0 = time.time()

    rows = []
    for i, row in df.iterrows():
        code = row['code']
        if code in done:
            continue

        surp = surprisal_data.get(code, {'mean_surprisal': None, 'std_surprisal': None, 'n_tokens': 0})
        drift = compute_drift(embedder, row['continuation'])

        rows.append({
            'code': code,
            **surp,
            **drift,
        })

        if (i + 1) % 200 == 0:
            print(f"  Drift: {i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Drift done ({time.time()-t0:.0f}s)", flush=True)

    # Save
    result_df = pd.DataFrame(rows)
    if done and os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        result_df = pd.concat([existing, result_df], ignore_index=True)

    result_df.to_csv(out_path, index=False)
    print(f"Saved {len(result_df)} rows to {out_path}", flush=True)


if __name__ == '__main__':
    main()
