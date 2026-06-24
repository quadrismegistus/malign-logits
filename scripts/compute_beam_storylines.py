#!/usr/bin/env python3
"""Re-run beam storylines with entropy for all families.

Runs beam_storylines (n=100, depth=10) + batch_beam_annotate for
all families, replacing cached entries without entropy. Uses full
(non-truncated) source names.

Usage:
    python scripts/compute_beam_storylines.py [--family FAMILY]
"""

import argparse
import gc
import time

import torch
from tqdm import tqdm

from malign_logits.beam import batch_beam_annotate
from malign_logits.experiments import DEFAULT_PROMPTS, INSTITUTIONAL_PROMPTS
from malign_logits.registry import Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default=None,
                        help="Only process base models matching this string")
    parser.add_argument("--n", type=int, default=100, help="Number of beams")
    parser.add_argument("--max-tokens", type=int, default=10, help="Tokens per storyline")
    args = parser.parse_args()

    reg = Registry()
    all_prompts = {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}
    bases = reg.all_bases()

    if args.family:
        bases = [b for b in bases if args.family in b]

    print(f"=== Beam storylines: {len(bases)} families × {len(all_prompts)} prompts × {args.n} beams ===")
    print()

    import os, shutil
    t0 = time.time()

    for base_id in sorted(bases):
        short = base_id.split("/")[-1]
        nick = reg.nickname(base_id)
        print(f"\n--- {nick} ({short}) ---")

        ft = time.time()
        try:
            results = batch_beam_annotate(base_id, all_prompts,
                                           n=args.n, max_tokens=args.max_tokens)
            dur = time.time() - ft
            print(f"  {nick}: {len(results)} entries, {dur:.0f}s")
        except Exception as e:
            import traceback
            print(f"  {nick}: FAILED {str(e)[:80]}")
            traceback.print_exc()

        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    total = time.time() - t0
    print(f"\n=== DONE: {len(bases)} families in {total:.0f}s ({total/3600:.1f}h) ===")


if __name__ == "__main__":
    main()
