#!/usr/bin/env python3
"""Re-run beam storylines with entropy for all families.

Runs batch_beam_annotate for all families × all prompts × n beams,
with per-position entropy and non-truncated source names.

Usage:
    python scripts/compute_beam_storylines.py [--family FAMILY] [--mode raw]
    python scripts/compute_beam_storylines.py --mode chat
    python scripts/compute_beam_storylines.py --mode all   # raw + chat + continue
"""

import argparse
import gc
import time

import torch
from tqdm import tqdm

from malign_logits.beam import batch_beam_annotate
from malign_logits.experiments import DEFAULT_PROMPTS, INSTITUTIONAL_PROMPTS
from malign_logits.registry import Registry


def run_mode(mode, bases, all_prompts, args):
    """Run storylines for one mode across all families."""
    print(f"\n{'='*60}")
    print(f"Mode: {mode} — {len(bases)} families × {len(all_prompts)} prompts × {args.n} beams")
    print(f"{'='*60}")

    t0 = time.time()
    for base_id in sorted(bases):
        reg = Registry()
        nick = reg.nickname(base_id)
        print(f"\n--- {nick} ---")

        ft = time.time()
        try:
            results = batch_beam_annotate(base_id, all_prompts,
                                           n=args.n, max_tokens=args.max_tokens,
                                           mode=mode)
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
    print(f"\nMode {mode}: {len(bases)} families in {total:.0f}s ({total/3600:.1f}h)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default=None,
                        help="Only process base models matching this string")
    parser.add_argument("--n", type=int, default=100, help="Number of beams")
    parser.add_argument("--max-tokens", type=int, default=10, help="Tokens per storyline")
    parser.add_argument("--mode", type=str, default="raw",
                        choices=["raw", "chat", "continue", "think", "all"],
                        help="Prompt mode (default: raw)")
    args = parser.parse_args()

    reg = Registry()
    all_prompts = {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}
    bases = reg.all_bases()

    if args.family:
        bases = [b for b in bases if args.family in b]

    if args.mode == "all":
        modes = ["raw", "chat", "continue"]
    else:
        modes = [args.mode]

    for mode in modes:
        run_mode(mode, bases, all_prompts, args)

    print(f"\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
