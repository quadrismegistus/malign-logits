"""Generate contradiction completions and produce blinded classification file.

Generates 3 completions per (family, layer, prompt) for all 11 contradiction
pairs × 3 prompt types (A, B, AB) × 7 families × {base, aligned}.

Outputs:
  data/f11_classify_blinded.csv  — blinded: site_id, prompt, text
  data/f11_classify_metadata.csv — unblinded: site_id, family, layer, pair, prompt_type

Usage:
    uv run python scripts/f11_gen_classify.py [--family FAMILY] [--n N]
"""
import argparse
import gc
import hashlib
import os
import sys
import random

import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits import MODEL_FAMILIES, PATH_DATA
from malign_logits.cache import get_cache
from malign_logits.models import load_model
from malign_logits.psyche import Psyche


FAMILIES = ['olmo', 'amber', 'llama', 'qwen', 'tulu', 'zephyr', 'olmo-tiny']

N_SAMPLES = 3
MAX_NEW_TOKENS = 80
TEMPERATURE = 1.0
TOP_K = 50

NEUTRAL_PROMPTS = [
    "She walked through the garden and decided to",
    "He opened the book and began to",
    "The teacher stood at the front and started to",
    "She picked up the phone and chose to",
    "He looked out the window and wanted to",
]


def generate_one(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS,
                 temp=TEMPERATURE, top_k=TOP_K):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs['input_ids']
    mask = inputs.get('attention_mask', torch.ones_like(input_ids))
    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=mask,
            max_new_tokens=max_tokens, temperature=temp,
            do_sample=True, top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def run_family(fkey, n_samples, cm):
    fam = MODEL_FAMILIES[fkey]
    pairs = list(Psyche.DEFAULT_CONTRADICTIONS)

    layer_specs = [('base', fam.base)]
    if fam.superego:
        layer_specs.append(('aligned', fam.superego))
    elif fam.ego:
        layer_specs.append(('aligned', fam.ego))

    rows = []

    for layer_name, model_id in layer_specs:
        existing = {}
        for pair in pairs:
            for ptype, pkey in [('A', 'prompt_a'), ('B', 'prompt_b'), ('AB', 'prompt_ab')]:
                prompt = pair[pkey]
                n_cached = cm.count_generations(model_id, prompt, temp=TEMPERATURE)
                if n_cached >= n_samples:
                    for idx in range(n_samples):
                        text = cm.get_generation(model_id, prompt, temp=TEMPERATURE, idx=idx)
                        rows.append({
                            'family': fkey, 'layer': layer_name,
                            'model_id': model_id,
                            'pair': pair['name'], 'prompt_type': ptype,
                            'prompt': prompt, 'idx': idx, 'text': text,
                        })
                    existing[(pair['name'], ptype)] = n_cached

        need_gen = []
        for pair in pairs:
            for ptype, pkey in [('A', 'prompt_a'), ('B', 'prompt_b'), ('AB', 'prompt_ab')]:
                if (pair['name'], ptype) not in existing:
                    need_gen.append((pair, ptype, pkey))

        for prompt in NEUTRAL_PROMPTS:
            n_cached = cm.count_generations(model_id, prompt, temp=TEMPERATURE)
            if n_cached >= n_samples:
                for idx in range(n_samples):
                    text = cm.get_generation(model_id, prompt, temp=TEMPERATURE, idx=idx)
                    rows.append({
                        'family': fkey, 'layer': layer_name,
                        'model_id': model_id,
                        'pair': 'neutral', 'prompt_type': 'N',
                        'prompt': prompt, 'idx': idx, 'text': text,
                    })
            else:
                need_gen.append(({'name': 'neutral'}, 'N', prompt))

        if not need_gen:
            print(f"  {fkey}/{layer_name}: all {len(rows)} cached")
            continue

        print(f"  {fkey}/{layer_name}: generating {len(need_gen)} prompt-groups × {n_samples} samples...")
        model, tokenizer = load_model(model_id)

        for item in need_gen:
            if item[1] == 'N':
                pair_dict, ptype, prompt = {'name': 'neutral'}, 'N', item[2]
            else:
                pair_dict, ptype, pkey = item
                prompt = pair_dict[pkey]

            n_cached = cm.count_generations(model_id, prompt, temp=TEMPERATURE)
            for idx in range(n_cached, n_cached + n_samples):
                text = generate_one(model, tokenizer, prompt)
                cm.set_generation(model_id, prompt, text, temp=TEMPERATURE, idx=idx)
                rows.append({
                    'family': fkey, 'layer': layer_name,
                    'model_id': model_id,
                    'pair': pair_dict['name'], 'prompt_type': ptype,
                    'prompt': prompt, 'idx': idx - n_cached, 'text': text,
                })

        del model, tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return rows


def make_blinded(rows_df):
    """Produce blinded classification file with stable site_ids."""
    blinded = []
    metadata = []

    for _, r in rows_df.iterrows():
        raw = f"{r.family}_{r.layer}_{r.pair}_{r.prompt_type}_{r.prompt}_{r.idx}"
        site_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
        blinded.append({
            'site_id': site_id,
            'prompt': r.prompt,
            'text': r.text[:300],
        })
        metadata.append({
            'site_id': site_id,
            'family': r.family,
            'layer': r.layer,
            'model_id': r.model_id,
            'pair': r.pair,
            'prompt_type': r.prompt_type,
            'idx': r.idx,
        })

    bdf = pd.DataFrame(blinded)
    mdf = pd.DataFrame(metadata)

    bdf = bdf.sample(frac=1, random_state=42).reset_index(drop=True)

    return bdf, mdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, default=None,
                        help='Run a single family (default: all)')
    parser.add_argument('--n', type=int, default=N_SAMPLES,
                        help=f'Samples per config (default: {N_SAMPLES})')
    parser.add_argument('--export-only', action='store_true',
                        help='Skip generation, just export from cache')
    args = parser.parse_args()

    families = [args.family] if args.family else FAMILIES
    cm = get_cache()

    all_rows = []
    for fkey in families:
        print(f"\n{'='*60}")
        print(f"  {fkey}")
        print(f"{'='*60}")
        rows = run_family(fkey, args.n, cm)
        all_rows.extend(rows)
        print(f"  → {len(rows)} rows")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} rows")
    print(f"  Families: {sorted(df.family.unique())}")
    print(f"  Layers: {sorted(df.layer.unique())}")
    print(f"  Pairs: {sorted(df.pair.unique())}")

    bdf, mdf = make_blinded(df)

    bpath = os.path.join(PATH_DATA, "f11_classify_blinded.csv")
    mpath = os.path.join(PATH_DATA, "f11_classify_metadata.csv")
    bdf.to_csv(bpath, index=False)
    mdf.to_csv(mpath, index=False)
    print(f"\nSaved {len(bdf)} blinded rows to {bpath}")
    print(f"Saved {len(mdf)} metadata rows to {mpath}")

    n_contra = len(df[df.pair != 'neutral'])
    n_neutral = len(df[df.pair == 'neutral'])
    print(f"\nContradiction: {n_contra}, Neutral: {n_neutral}")


if __name__ == '__main__':
    main()
