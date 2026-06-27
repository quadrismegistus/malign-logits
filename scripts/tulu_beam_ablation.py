#!/usr/bin/env python3
"""Tulu ablation: beam search + teacher-forcing for 5 Tulu variants.

Runs batch_beam_annotate logic but restricted to:
  - Llama-3.1-8B (base, already has beams — reused for teacher-forcing)
  - Tulu-SFT-no-safety-data
  - Tulu-SFT (with safety)
  - Tulu-DPO
  - Tulu-3.1 (final RLVR)

Each model's beams get teacher-forced through all others.

Usage:
    python scripts/tulu_beam_ablation.py
    python scripts/tulu_beam_ablation.py --n 100 --max-tokens 10
"""

import argparse
import gc
import time

import torch

from malign_logits.beam import batch_beam_annotate
from malign_logits.experiments import DEFAULT_PROMPTS, INSTITUTIONAL_PROMPTS


TULU_VARIANTS = [
    'allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data',
    'allenai/Llama-3.1-Tulu-3-8B-SFT',
    'allenai/Llama-3.1-Tulu-3-8B-DPO',
    'allenai/Llama-3.1-Tulu-3.1-8B',
]

BASE = 'meta-llama/Llama-3.1-8B'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    all_prompts = {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}
    print(f"Tulu ablation: {len(TULU_VARIANTS)} variants + base, "
          f"{len(all_prompts)} prompts, n={args.n}")

    # Monkey-patch Registry.variants_of to return only Tulu variants
    from malign_logits.registry import Registry
    reg = Registry()
    _orig_variants = reg.variants_of

    def _tulu_only_variants(base_id):
        if base_id == BASE:
            return TULU_VARIANTS
        return _orig_variants(base_id)

    reg.variants_of = _tulu_only_variants

    t0 = time.time()
    results = batch_beam_annotate(BASE, all_prompts,
                                  n=args.n, max_tokens=args.max_tokens)

    dur = time.time() - t0
    print(f"\nDone: {len(results)} prompt entries in {dur:.0f}s ({dur/3600:.1f}h)")


if __name__ == "__main__":
    main()
