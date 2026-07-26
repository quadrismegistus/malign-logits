"""F37: Score stash exports (generations + beams) with PKU judges.

Usage:
    python f37_score_stash.py generations  # 215k generation stash
    python f37_score_stash.py beams        # 11.5k beams stash
    python f37_score_stash.py all          # both
"""
import argparse
import gc
import json
import os
import time

import torch
import pandas as pd

DATA_DIR = os.environ.get('F37_DATA', '/root/data')
CACHE_DIR = os.path.join(DATA_DIR, 'raw', 'cache', 'reward_scores')
os.makedirs(CACHE_DIR, exist_ok=True)

JUDGES = {
    'pku-reward': 'PKU-Alignment/beaver-7b-v1.0-reward',
    'pku-cost': 'PKU-Alignment/beaver-7b-v1.0-cost',
}

PKU_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {user} ASSISTANT:{assistant}"
FRAME = "Continue this text:"

RUN_META = None

def get_meta():
    global RUN_META
    if RUN_META is None:
        import transformers
        RUN_META = {
            'transformers': transformers.__version__,
            'torch': torch.__version__,
            'dtype': 'float16' if torch.cuda.is_available() else 'float32',
            'platform': 'cuda' if torch.cuda.is_available() else 'cpu',
            'frame': FRAME,
        }
    return RUN_META


def load_judge(judge_key):
    from safe_rlhf.models import AutoModelForScore
    from transformers import AutoTokenizer
    model_id = JUDGES[judge_key]
    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = AutoModelForScore.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
    else:
        model = AutoModelForScore.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return model, tok


def score(model, tok, text):
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=512, add_special_tokens=True)
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.end_scores.item()


def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.jsonl")


def load_cache(name):
    p = cache_path(name)
    c = {}
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                e = json.loads(line)
                c[e['key']] = e['score']
    return c


def save_one(name, key, sc):
    with open(cache_path(name), 'a') as f:
        f.write(json.dumps({'key': key, 'score': sc, 'meta': get_meta()}) + '\n')


def score_jsonl(input_file, cache_name, output_csv):
    """Score a JSONL file of stash exports."""
    rows = []
    with open(input_file) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} items from {input_file}")

    for jk in JUDGES:
        model, tok = load_judge(jk)
        cache = load_cache(f'{cache_name}_{jk}')
        t0 = time.time()
        scored = 0
        results = []

        for i, row in enumerate(rows):
            text = row.get('text', '')
            if not text or text == 'None':
                results.append(None)
                continue

            prompt = row.get('prompt', '')
            model_id = row.get('model', '')
            idx = row.get('idx', i)
            ck = f"{jk}|{model_id}|{prompt[:50]}|{idx}"

            if ck in cache:
                results.append(cache[ck])
            else:
                user = f"{FRAME} {prompt}" if prompt else FRAME
                formatted = PKU_TEMPLATE.format(user=user, assistant=text)
                sc = score(model, tok, formatted)
                cache[ck] = sc
                save_one(f'{cache_name}_{jk}', ck, sc)
                results.append(sc)
                scored += 1
                if scored % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = scored / elapsed
                    eta = (len(rows) - i) / rate / 60 if rate > 0 else 0
                    print(f"  {jk}: {scored} scored ({rate:.1f}/s, ETA {eta:.0f}m)")

        elapsed = time.time() - t0
        print(f"  {jk}: {scored} scored in {elapsed:.0f}s")

        # Add scores to rows
        for i, sc in enumerate(results):
            rows[i][f'{jk}_score'] = sc

        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, output_csv), index=False)
    print(f"Saved {len(df)} rows to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=['generations', 'beams', 'all'])
    args = parser.parse_args()

    if args.target in ('generations', 'all'):
        score_jsonl(
            os.path.join(DATA_DIR, 'f37_stash_generations.jsonl'),
            'stash_gen', 'f37_stash_generations_scored.csv')
    if args.target in ('beams', 'all'):
        score_jsonl(
            os.path.join(DATA_DIR, 'f37_stash_beams.jsonl'),
            'stash_beam', 'f37_stash_beams_scored.csv')
