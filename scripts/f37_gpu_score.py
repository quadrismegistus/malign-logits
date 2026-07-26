"""F37 GPU scoring: load prompt items from JSON, score with both judges.

Designed for the vast.ai GPU instance. Loads each judge once, scores all items.

Usage:
    python f37_gpu_score.py prompts          # score prompt items
    python f37_gpu_score.py gens CSV [CSV..] # score generation CSVs
"""
import argparse
import gc
import json
import os
import sys
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


def cmd_prompts(args):
    items = json.load(open(args.items))
    print(f"Scoring {len(items)} prompt items")
    t0 = time.time()

    for jk in JUDGES:
        model, tok = load_judge(jk)
        cache = load_cache(f'prompts_{jk}')
        rows = []
        scored = 0

        for item in items:
            ck = f"{jk}|{item['item_id']}"
            if ck in cache:
                sc = cache[ck]
            else:
                text = PKU_TEMPLATE.format(user=FRAME, assistant=item['text'])
                sc = score(model, tok, text)
                cache[ck] = sc
                save_one(f'prompts_{jk}', ck, sc)
                scored += 1
                if scored % 100 == 0:
                    print(f"  {jk}: {scored} scored ({time.time()-t0:.0f}s)")

            rows.append({'judge': jk, 'score': sc, **item})

        df = pd.DataFrame(rows)
        out = os.path.join(DATA_DIR, f'f37_prompts_{jk}.csv')
        df.to_csv(out, index=False)
        print(f"  {jk}: {scored} scored, {len(items)-scored} cached -> {out} ({time.time()-t0:.0f}s)")

        del model, tok; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def cmd_gens(args):
    gen_specs = [
        ('disposition_full.csv', 'raw_generation', 'prompt'),
        ('disposition_continue.csv', 'raw_generation', 'prompt'),
        ('disposition_all_stages.csv', 'raw_generation', 'prompt'),
        ('f11_classify_blinded.csv', 'text', 'prompt'),
        ('f21_rerun.csv', 'raw_generation', 'prompt'),
        ('salary_systematic.csv', 'raw_text', 'prompt'),
    ]

    for jk in JUDGES:
        model, tok = load_judge(jk)
        cache = load_cache(f'gens_{jk}')
        t0 = time.time()

        for csv_name, text_col, prompt_col in gen_specs:
            csv_path = os.path.join(DATA_DIR, csv_name)
            if not os.path.exists(csv_path):
                print(f"  SKIP {csv_name}")
                continue

            df = pd.read_csv(csv_path)
            if text_col not in df.columns:
                print(f"  SKIP {csv_name}: no {text_col}")
                continue

            scores = []
            scored = 0
            for idx, row in df.iterrows():
                ck = f"{jk}|{csv_name}|{idx}"
                if ck in cache:
                    scores.append(cache[ck])
                else:
                    gen_text = str(row[text_col])
                    prompt = str(row.get(prompt_col, ''))
                    text = PKU_TEMPLATE.format(user=f"{FRAME} {prompt}", assistant=gen_text)
                    sc = score(model, tok, text)
                    cache[ck] = sc
                    save_one(f'gens_{jk}', ck, sc)
                    scores.append(sc)
                    scored += 1
                    if scored % 1000 == 0:
                        rate = scored / (time.time() - t0)
                        print(f"    {csv_name}: {scored} ({rate:.1f}/s)")

            df[f'{jk}_score'] = scores
            out = os.path.join(DATA_DIR, f'f37_gen_{csv_name}')
            df.to_csv(out, index=False)
            print(f"  {csv_name}: {scored} scored -> {out}")

        del model, tok; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    p = sub.add_parser('prompts')
    p.add_argument('--items', default='f37_prompt_items.json')
    g = sub.add_parser('gens')
    args = parser.parse_args()
    if args.cmd == 'prompts':
        cmd_prompts(args)
    elif args.cmd == 'gens':
        cmd_gens(args)
