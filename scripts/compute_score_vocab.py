#!/usr/bin/env python3
"""Compute score_vocab for all model×prompt pairs that have cached logits.

Three phases:
1. discover_top_words on EVERY model (not just base) — 200 forward passes each
2. Union word lists across all models in a family for each prompt
3. score_words_from_logits on all models against the union (from cached logits, cheap)

Usage:
    python scripts/compute_score_vocab.py [--family FAMILY] [--phase 1|2|3|all]
"""

import argparse
import gc
import time
from collections import defaultdict

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default=None,
                        help="Only process models matching this string")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which phase to run (1=discover, 2=union, 3=score)")
    args = parser.parse_args()

    from malign_logits.cache import get_cache
    from malign_logits.core import discover_top_words, score_words_from_logits
    from malign_logits.registry import Registry

    cm = get_cache()
    reg = Registry()
    logits_stash = cm._stash("logits")

    # Group logit entries by base family
    family_models = defaultdict(set)  # base_id -> {model_ids}
    model_prompts = defaultdict(set)  # model_id -> {prompts}

    for k in logits_stash.keys():
        if not isinstance(k, dict):
            continue
        model, prompt = k["model"], k["prompt"]
        base = reg.base_of(model) or model
        family_models[base].add(model)
        model_prompts[model].add(prompt)

    if args.family:
        bases = [b for b in family_models if args.family in b]
    else:
        bases = sorted(family_models.keys())

    print(f"Families: {len(bases)}")
    for b in bases:
        models = family_models[b]
        prompts = set()
        for m in models:
            prompts |= model_prompts[m]
        print(f"  {b.split('/')[-1]:30s} {len(models)} models × {len(prompts)} prompts")

    # Phase 1: discover_top_words on EVERY model that has logits
    if args.phase in ("1", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 1: Word discovery (200 passes per model×prompt)")
        print(f"{'='*60}")

        for base_id in bases:
            models = sorted(family_models[base_id])
            for mid in models:
                short = mid.split("/")[-1]
                prompts = model_prompts[mid]

                need = [p for p in prompts if not cm.has_top_words(mid, p)]
                if not need:
                    print(f"  {short}: all {len(prompts)} done")
                    continue

                print(f"\n  {short}: discovering {len(need)}/{len(prompts)} prompts...")
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
                for p in tqdm(need, desc=f"    {short}", unit="prompt"):
                    try:
                        words = discover_top_words(model, tokenizer, p, top_k_first=200)
                        cm.set_top_words(mid, p, words, k=200)
                    except Exception as e:
                        tqdm.write(f"    FAIL '{p[:40]}': {e}")

                dur = time.time() - t0
                print(f"    {len(need)} prompts in {dur:.0f}s ({dur/len(need):.1f}s/prompt)")

                del model
                gc.collect()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

    # Phase 2: Union word lists across family for each prompt
    if args.phase in ("2", "3", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2+3: Union word lists and score all models")
        print(f"{'='*60}")

        total_scored = 0
        total_skipped = 0

        for base_id in bases:
            models = sorted(family_models[base_id])
            short_base = base_id.split("/")[-1]

            # Collect all prompts across the family
            all_prompts = set()
            for mid in models:
                all_prompts |= model_prompts[mid]

            family_scored = 0
            for p in sorted(all_prompts):
                # Union word lists from all models that have discovery for this prompt
                union_words = set()
                for mid in models:
                    tw = cm.get_top_words(mid, p)
                    if tw:
                        union_words |= set(tw.keys())

                if not union_words:
                    continue

                candidate_words = sorted(union_words)

                # Score every model against the union
                for mid in models:
                    if cm.has_score_vocab(mid, p):
                        total_skipped += 1
                        continue
                    logits = cm.get_logits(mid, p)
                    if logits is None:
                        continue
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(mid)
                        scores = score_words_from_logits(
                            torch.tensor(logits), tokenizer, candidate_words
                        )
                        cm.set_score_vocab(mid, p, scores)
                        family_scored += 1
                        total_scored += 1
                    except Exception:
                        pass

            print(f"  {short_base}: scored {family_scored} (skipped {total_skipped})")

        print(f"\nTotal: scored {total_scored}, skipped {total_skipped}")

    print("\nDone!")


if __name__ == "__main__":
    main()
