"""F37: Reward-model scoring of existing batteries and generations.

Phase 1: PKU-Alignment beaver-7b-v1.0-reward and beaver-7b-v1.0-cost.

Usage:
    uv run python scripts/f37_reward_score.py sanity   # Frame sanity check on Set D
    uv run python scripts/f37_reward_score.py prompts  # Score all 609 prompt items
    uv run python scripts/f37_reward_score.py gens     # Score stored generation CSVs
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, LlamaModel, LlamaPreTrainedModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA

CACHE_DIR = os.path.join(PATH_DATA, "raw", "cache", "reward_scores")
os.makedirs(CACHE_DIR, exist_ok=True)


class LlamaForScore(LlamaPreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        self.model = LlamaModel(config)
        self.score_dim = getattr(config, 'score_dim', 1)
        self.score_bias = getattr(config, 'score_bias', True)
        self.score_head = nn.Linear(config.hidden_size, self.score_dim, bias=self.score_bias)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        scores = self.score_head(last_hidden)
        if attention_mask is not None:
            end_index = attention_mask.long().sum(dim=-1) - 1
        else:
            end_index = torch.full((input_ids.shape[0],), input_ids.shape[1] - 1,
                                   device=input_ids.device)
        end_scores = scores[torch.arange(scores.shape[0], device=scores.device), end_index]
        return end_scores.squeeze(-1)


JUDGES = {
    'pku-reward': 'PKU-Alignment/beaver-7b-v1.0-reward',
    'pku-cost': 'PKU-Alignment/beaver-7b-v1.0-cost',
}

PKU_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {user} ASSISTANT:{assistant}"

FRAMES = {
    'continue': 'Continue this text:',
    'bare': 'Write a sentence.',
    'passage': 'Here is a passage:',
}


def format_for_judge(frame_id, text):
    """Format text in the PKU conversation template.

    The frame goes in the USER slot (instruction).
    The battery string goes in the ASSISTANT slot (model response being scored).
    """
    user = FRAMES[frame_id]
    return PKU_TEMPLATE.format(user=user, assistant=text)


def _cache_path(judge_key):
    return os.path.join(CACHE_DIR, f"{judge_key}.jsonl")


def _load_cache(judge_key):
    path = _cache_path(judge_key)
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry['cache_key']] = entry['score']
    return cache


def _append_cache(judge_key, cache_key, score, meta=None):
    path = _cache_path(judge_key)
    entry = {'cache_key': cache_key, 'score': score}
    if meta:
        entry.update(meta)
    with open(path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def _make_cache_key(judge_key, item_id, frame_id):
    return f"{judge_key}|{item_id}|{frame_id}"


def load_judge(judge_key):
    model_id = JUDGES[judge_key]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = LlamaForScore.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to('mps')
    model.eval()
    return model, tokenizer


def score_text(model, tokenizer, text, max_length=512):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        score = model(**inputs)
    return score.item()


def score_batch(model, tokenizer, items, judge_key, frame_id, cache, batch_label=""):
    """Score a list of (item_id, text) tuples. Returns list of (item_id, score)."""
    results = []
    n_cached = 0
    n_scored = 0

    for item_id, text in items:
        ck = _make_cache_key(judge_key, item_id, frame_id)
        if ck in cache:
            results.append((item_id, cache[ck]))
            n_cached += 1
            continue

        s = score_text(model, tokenizer, text)
        cache[ck] = s
        _append_cache(judge_key, ck, s)
        results.append((item_id, s))
        n_scored += 1

        if n_scored % 50 == 0:
            print(f"  {batch_label} scored {n_scored} (cached {n_cached})...")

    print(f"  {batch_label} done: {n_scored} scored, {n_cached} cached")
    return results


def load_set_d():
    """Load Set D v1 prompts (complete sentences for reward scoring)."""
    set_d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f36_violence_set_d.py')
    globs = {}
    exec(open(set_d_path).read(), globs)
    return globs['SET_D']


def cmd_sanity(args):
    """Frame sanity check: score Set D under 3 frames, both judges."""
    set_d = load_set_d()
    print(f"Set D: {len(set_d)} items")

    rows = []
    for judge_key in JUDGES:
        model, tokenizer = load_judge(judge_key)
        cache = _load_cache(judge_key)

        for frame_id in FRAMES:
            items = []
            for p in set_d:
                text = format_for_judge(frame_id, p['prompt'])
                item_id = f"setd_{p['id']}"
                items.append((item_id, text))

            results = score_batch(model, tokenizer, items, judge_key, frame_id, cache,
                                  f"{judge_key}/{frame_id}")

            for (item_id, sc), p in zip(results, set_d):
                rows.append({
                    'judge': judge_key,
                    'frame': frame_id,
                    'item_id': item_id,
                    'score': sc,
                    **{k: v for k, v in p.items() if k != 'prompt'},
                })

        del model, tokenizer
        gc.collect()
        torch.mps.empty_cache()

    df = pd.DataFrame(rows)
    out = os.path.join(PATH_DATA, "f37_frame_sanity.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

    # Print ordering stability table
    print("\n=== FRAME ORDERING STABILITY ===")
    for judge_key in JUDGES:
        jdf = df[df.judge == judge_key]
        print(f"\n{judge_key}:")
        # Compare frame orderings using Spearman rank correlation
        from scipy.stats import spearmanr
        frames = list(FRAMES.keys())
        for i, f1 in enumerate(frames):
            for f2 in frames[i+1:]:
                s1 = jdf[jdf.frame == f1].set_index('item_id')['score']
                s2 = jdf[jdf.frame == f2].set_index('item_id')['score']
                common = s1.index.intersection(s2.index)
                r, p = spearmanr(s1[common], s2[common])
                print(f"  {f1} vs {f2}: rho={r:.4f}, p={p:.2e}")

    return df


def cmd_prompts(args):
    """Score all 609 prompt items."""
    raise NotImplementedError("Run sanity first, then implement after frame is chosen")


def cmd_gens(args):
    """Score stored generation CSVs."""
    raise NotImplementedError("Run sanity first, then implement after frame is chosen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['sanity', 'prompts', 'gens'])
    args = parser.parse_args()

    if args.command == 'sanity':
        cmd_sanity(args)
    elif args.command == 'prompts':
        cmd_prompts(args)
    elif args.command == 'gens':
        cmd_gens(args)


if __name__ == '__main__':
    main()
