#!/usr/bin/env python3
"""Cloud batch annotation script.

Run on any GPU machine (vast.ai, CSD3, etc.):
    pip install -e .
    python scripts/cloud_batch_annotate.py [--families FAM1,FAM2,...] [--prompts battery|institutional|all]

Results cached in data/raw/cache/ — rsync back when done.
"""

import argparse
import time
from tqdm import tqdm

from malign_logits.probe import Probe
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
    parser.add_argument('--families', type=str, default='all',
                        help='Comma-separated family IDs or "all"')
    parser.add_argument('--prompts', type=str, default='all',
                        choices=['battery', 'institutional', 'all'],
                        help='Which prompt set to run')
    args = parser.parse_args()

    if args.families == 'all':
        families = ALL_FAMILIES
    else:
        families = args.families.split(',')

    if args.prompts == 'battery':
        prompts = DEFAULT_PROMPTS
    elif args.prompts == 'institutional':
        prompts = INSTITUTIONAL_PROMPTS
    else:
        prompts = {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}

    reg = Registry()
    print(f"=== Batch annotate: {len(families)} families × {len(prompts)} prompts ===")
    print(f"Families: {[f.split('/')[-1] for f in families]}")
    print()

    t0 = time.time()
    for base_id in tqdm(families, desc='Families', unit='fam'):
        short = base_id.split('/')[-1]
        n_ann = len(reg.variants_of(reg.base_of(base_id) or base_id))
        tqdm.write(f'\n--- {short} ({n_ann} annotators) ---')

        ft = time.time()
        try:
            p = Probe(base_id)
            results = p.batch_annotate(prompts)
            n_nodes = sum(len(v) for v in results.values())
            dur = time.time() - ft
            tqdm.write(f'  {short}: {len(results)} prompts, {n_nodes} nodes, {dur:.0f}s')
        except Exception as e:
            import traceback
            tqdm.write(f'  {short}: FAILED {str(e)[:80]}')
            traceback.print_exc()

    total = time.time() - t0
    print(f'\n=== DONE: {len(families)} families in {total:.0f}s ({total/3600:.1f}h) ===')


if __name__ == '__main__':
    main()
