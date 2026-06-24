#!/usr/bin/env python3
"""Compute hybrid word_probs from cached beam_words + logits.

For each (model, prompt) with cached beam_words:
  - Single-token words → exact logit softmax probability
  - Multi-token words → beam path probability
  - Renormalize and store in word_probs/ stash

Fast — no model loading, just tokenizer + cached data.

Usage:
    python scripts/compute_word_probs.py [--family FAMILY]
"""

import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default=None)
    args = parser.parse_args()

    from malign_logits.cache import get_cache
    from malign_logits.core import hybrid_word_probs

    cm = get_cache()
    bw_stash = cm._stash("beam_words")

    # Collect all beam_words entries
    entries = []
    for k in bw_stash.keys():
        if not isinstance(k, dict):
            continue
        model = k.get("model", "")
        prompt = k.get("prompt", "")
        if args.family and args.family not in model:
            continue
        if not cm.has_word_probs(model, prompt):
            entries.append((model, prompt))

    print(f"Entries to compute: {len(entries)}")
    if not entries:
        print("All done!")
        return

    # Group by model (load tokenizer once per model)
    by_model = defaultdict(list)
    for model, prompt in entries:
        by_model[model].append(prompt)

    computed = 0
    skipped = 0

    for model in sorted(by_model):
        short = model.split("/")[-1]
        prompts = by_model[model]

        # Load tokenizer (cheap, no model weights)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as e:
            print(f"  {short}: SKIP tokenizer ({e})")
            skipped += len(prompts)
            continue

        done = 0
        for prompt in tqdm(prompts, desc=f"  {short}", unit="prompt", leave=False):
            bw = cm.get_beam_words(model, prompt)
            logits = cm.get_logits(model, prompt)

            if not bw:
                skipped += 1
                continue

            if logits is not None:
                hw = hybrid_word_probs(bw, np.array(logits, dtype=np.float32), tokenizer)
            else:
                # No logits — use beam probs as-is
                total = sum(bw.values())
                hw = {w: p / total for w, p in bw.items()} if total > 0 else bw

            cm.set_word_probs(model, prompt, hw)
            done += 1

        computed += done
        print(f"  {short}: {done}/{len(prompts)} computed")

    print(f"\nDone: {computed} computed, {skipped} skipped")


if __name__ == "__main__":
    main()
