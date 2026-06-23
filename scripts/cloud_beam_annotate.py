#!/usr/bin/env python3
"""Cloud beam search annotation script.

Run on any GPU machine:
    pip install -e .
    python scripts/cloud_beam_annotate.py [--families all] [--prompts all] [--n 100]

Results cached in data/raw/cache/ — rsync back when done.
"""

import argparse
import time
from tqdm import tqdm

from malign_logits.beam import annotate_beams
from malign_logits.experiments import DEFAULT_PROMPTS, INSTITUTIONAL_PROMPTS
from malign_logits.registry import Registry

ALL_FAMILIES = [
    'allenai/OLMo-2-0425-1B',
    'allenai/Olmo-3-1025-7B',
    'meta-llama/Llama-3.1-8B',
    'LLM360/Amber',
    'Qwen/Qwen2.5-7B',
    'EleutherAI/pythia-6.9b',
    'mistralai/Mistral-7B-v0.1',
    'Qwen/Qwen3-8B-Base',
    'deepseek-ai/deepseek-llm-7b-base',
    'tiiuae/falcon-7b',
    '01-ai/Yi-9B',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--prompts', type=str, default='all',
                        choices=['battery', 'institutional', 'all'])
    parser.add_argument('--n', type=int, default=100,
                        help='Number of beams (default: 100)')
    parser.add_argument('--max-tokens', type=int, default=10,
                        help='Tokens per storyline (default: 10)')
    args = parser.parse_args()

    families = ALL_FAMILIES if args.families == 'all' else args.families.split(',')
    if args.prompts == 'battery':
        prompts = DEFAULT_PROMPTS
    elif args.prompts == 'institutional':
        prompts = INSTITUTIONAL_PROMPTS
    else:
        prompts = {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}

    reg = Registry()
    print(f"=== Beam annotate: {len(families)} families × {len(prompts)} prompts × {args.n} beams ===")
    print()

    t0 = time.time()
    for base_id in tqdm(families, desc='Families', unit='fam'):
        short = base_id.split('/')[-1]
        n_ann = len(reg.variants_of(reg.base_of(base_id) or base_id))
        tqdm.write(f'\n--- {short} ({n_ann} annotators) ---')

        ft = time.time()
        for pname, ptext in tqdm(prompts.items(), desc=short, unit='prompt', leave=False):
            try:
                stories = annotate_beams(base_id, ptext, n=args.n,
                                         max_tokens=args.max_tokens)
            except Exception as e:
                tqdm.write(f'  {pname}: FAILED {str(e)[:60]}')

        dur = time.time() - ft
        tqdm.write(f'  {short}: {len(prompts)} prompts, {dur:.0f}s')

    total = time.time() - t0
    print(f'\n=== DONE: {len(families)} families in {total:.0f}s ({total/3600:.1f}h) ===')


if __name__ == '__main__':
    main()
