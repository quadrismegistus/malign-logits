#!/usr/bin/env python3
"""Compute beam_word_probs for all model×prompt pairs that have cached logits.

Loads each model once, runs beam_word_probs (n=1000, depth=3) on all
prompts that don't already have cached beam_words. ~5s per prompt on 1B MPS.

Usage:
    python scripts/compute_beam_words.py [--family FAMILY] [--n 1000] [--depth 3]
    python scripts/compute_beam_words.py --mode chat      # chat template
    python scripts/compute_beam_words.py --mode continue   # "Continue this text:"
    python scripts/compute_beam_words.py --mode all        # run raw + chat + continue
"""

import argparse
import gc
import time
from collections import defaultdict

import torch
from tqdm import tqdm


def run_mode(mode, models, model_prompts, args):
    """Run beam_word_probs for one mode across all models."""
    from malign_logits.cache import get_cache
    from malign_logits.core import beam_word_probs

    cm = get_cache()

    total_need = 0
    for mid in models:
        need = sum(1 for p in model_prompts[mid]
                   if not cm.has_beam_words(mid, p, args.n, args.depth, mode))
        total_need += need

    print(f"\n{'='*60}")
    print(f"Mode: {mode} — {len(models)} models, {total_need} prompts to compute")
    print(f"{'='*60}")

    if total_need == 0:
        print("All done!")
        return 0

    grand_total = 0
    for mid in models:
        short = mid.split("/")[-1]
        prompts = model_prompts[mid]
        need = [p for p in prompts
                if not cm.has_beam_words(mid, p, args.n, args.depth, mode)]

        if not need:
            print(f"  {short}: all {len(prompts)} done")
            continue

        print(f"\n  {short}: {len(need)}/{len(prompts)} prompts...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.float16, device_map="mps"
            )
            model.eval()
        except Exception as e:
            print(f"    SKIP (can't load): {e}")
            continue

        t0 = time.time()
        done = 0
        for p in tqdm(need, desc=f"    {short}", unit="prompt"):
            try:
                words = beam_word_probs(model, tokenizer, p,
                                        n_beams=args.n, depth=args.depth,
                                        mode=mode)
                cm.set_beam_words(mid, p, words, args.n, args.depth, mode)
                done += 1
            except Exception as e:
                tqdm.write(f"    FAIL '{p[:40]}': {e}")

        dur = time.time() - t0
        grand_total += done
        print(f"    {done} prompts in {dur:.0f}s ({dur/max(done,1):.1f}s/prompt)")

        del model
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return grand_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default=None,
                        help="Only process models matching this string")
    parser.add_argument("--n", type=int, default=1000, help="Number of beams")
    parser.add_argument("--depth", type=int, default=3, help="Token depth")
    parser.add_argument("--mode", type=str, default="raw",
                        choices=["raw", "chat", "continue", "think", "all"],
                        help="Prompt mode (default: raw)")
    args = parser.parse_args()

    from malign_logits.cache import get_cache
    from malign_logits.registry import Registry

    cm = get_cache()
    reg = Registry()
    logits_stash = cm._stash("logits")

    # Group by model: find all (model, prompt) pairs with logits
    model_prompts = defaultdict(set)
    for k in logits_stash.keys():
        if isinstance(k, dict):
            model_prompts[k["model"]].add(k["prompt"])

    # Sort: small models first
    SMALL = ['SmolLM2', 'SmolLM3', 'Qwen2.5-0.5B', 'OLMo-2-0425-1B']
    def sort_key(m):
        for i, s in enumerate(SMALL):
            if s in m:
                return i
        return 100

    models = sorted(model_prompts.keys(), key=sort_key)
    if args.family:
        models = [m for m in models if args.family in m]

    if args.mode == "all":
        modes = ["raw", "chat", "continue"]
    else:
        modes = [args.mode]

    total = 0
    for mode in modes:
        total += run_mode(mode, models, model_prompts, args)

    print(f"\nDone! {total} beam_words entries computed across {len(modes)} mode(s).")


if __name__ == "__main__":
    main()
