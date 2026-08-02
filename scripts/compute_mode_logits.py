#!/usr/bin/env python3
"""Cache logits for non-raw modes (continue, chat, think).

For each model with cached raw logits, apply the mode template and run
a single forward pass to get the logit vector at the last position.
Fast: ~0.5s per prompt on 7B MPS (no beam search).

Usage:
    python scripts/compute_mode_logits.py --mode continue
    python scripts/compute_mode_logits.py --mode continue --family Llama
"""

import argparse
import gc
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["chat", "continue", "think"])
    parser.add_argument("--family", type=str, default=None)
    args = parser.parse_args()

    from malign_logits.cache import get_cache
    from malign_logits.core import _apply_mode

    cm = get_cache()
    #: WAS `if "mode" not in k` -- a test for the ABSENCE of the field, which
    #: is how the archived stash spelled "raw". Under the declared schema every
    #: key carries `mode`, so that filter now matches NOTHING and this script
    #: would have found no work and exited cleanly. A filter keyed to a missing
    #: field fails silently the moment the field stops being missing.
    model_prompts = defaultdict(set)
    for k in cm.iter_keys("logits", mode="raw"):
        model_prompts[k["model"]].add(k["prompt"])

    SMALL = ['SmolLM2', 'SmolLM3', 'Qwen2.5-0.5B', 'OLMo-2-0425-1B']
    def sort_key(m):
        for i, s in enumerate(SMALL):
            if s in m:
                return i
        return 100

    models = sorted(model_prompts.keys(), key=sort_key)
    if args.family:
        models = [m for m in models if args.family in m]

    total_need = 0
    for mid in models:
        need = sum(1 for p in model_prompts[mid]
                   if not cm.has_logits(mid, p, args.mode))
        total_need += need

    print(f"Mode: {args.mode} — {len(models)} models, {total_need} prompts to compute")

    if total_need == 0:
        print("All done!")
        return

    grand_total = 0
    for mid in models:
        short = mid.split("/")[-1]
        prompts = [p for p in model_prompts[mid]
                   if not cm.has_logits(mid, p, args.mode)]

        if not prompts:
            print(f"  {short}: all done")
            continue

        print(f"\n  {short}: {len(prompts)} prompts...")
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

        device = next(model.parameters()).device
        t0 = time.time()
        done = 0

        for p in tqdm(prompts, desc=f"    {short}", unit="prompt"):
            try:
                formatted = _apply_mode(p, tokenizer, args.mode)
                input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(input_ids).logits[0, -1, :].float().cpu().numpy()
                cm.set_logits(mid, p, logits, args.mode)
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

    print(f"\nDone! {grand_total} logits cached for mode={args.mode}")


if __name__ == "__main__":
    main()
