#!/usr/bin/env python
"""Generate n completions per prompt per model layer using vLLM.

Batches all n generations at once for massive speedup on GPU.
Writes to the same stash format as `malign generate-battery` so
downstream code (corpus_metrics, cross_generation_mmd, etc.) works unchanged.

Usage:
    # Generate for remaining families
    python scripts/vllm_generate.py --families qwen,tulu,zephyr,pythia --n 100

    # Single family
    python scripts/vllm_generate.py --families olmo --n 100

    # Dry run (show what would be generated)
    python scripts/vllm_generate.py --families qwen --n 100 --dry-run

Requires: pip install vllm
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits import MODEL_FAMILIES
from malign_logits.experiments import TIER1_PROMPTS
from malign_logits.embedding import _gen_stash_path


LAYER_NAMES = {
    "base": "base",
    "ego": "ego",
    "superego": "superego",
    "reinforced_superego": "instruct",
}


def get_stash():
    from hashstash import HashStash
    return HashStash(
        root_dir=_gen_stash_path(),
        append_mode=True, engine="pairtree", compress="lz4", b64=True,
    )


def count_existing(stash, prompt, temperature, model_ids):
    key = {
        "prompt": prompt,
        "temperature": temperature,
        "models": tuple(model_ids),
    }
    got = stash.get_all(key)
    return len(got) if got else 0


def generate_family(family_key, n, temperature=1.0, max_tokens=100,
                    prompts_set="tier1", dry_run=False):
    family = MODEL_FAMILIES[family_key]
    stash = get_stash()

    # Collect all model checkpoints for this family
    checkpoints = []
    model_ids = []
    for attr, layer_name in LAYER_NAMES.items():
        model_id = getattr(family, attr, None)
        if model_id is not None:
            checkpoints.append((attr, layer_name, model_id))
            model_ids.append(model_id)

    all_prompts = list(TIER1_PROMPTS.items())
    print(f"\n{'=' * 60}")
    print(f"  {family_key} ({family.name}, {len(checkpoints)} layers)")
    print(f"  {len(all_prompts)} prompts × {n} generations")
    print(f"{'=' * 60}")

    # Check existing per prompt, find max deficit
    per_prompt_existing = {}
    for label, prompt_text in all_prompts:
        existing = count_existing(stash, prompt_text, temperature, model_ids)
        per_prompt_existing[prompt_text] = existing
        if existing < n:
            print(f"  {label}: {existing}/{n}")

    prompts_to_run = [(l, p) for l, p in all_prompts
                      if per_prompt_existing[p] < n]
    if not prompts_to_run:
        print(f"  All prompts have {n}+ generations, skipping")
        return

    print(f"  {len(prompts_to_run)}/{len(all_prompts)} prompts need more generations")

    if dry_run:
        print("  DRY RUN — would generate here")
        return

    from vllm import LLM, SamplingParams

    if not hasattr(generate_family, '_partial'):
        generate_family._partial = {}

    for attr, layer_name, model_id in checkpoints:
        print(f"\n  Loading {layer_name} ({model_id})...")
        t0 = time.time()

        hf_token = os.environ.get("HF_TOKEN") or None
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.85,
            download_dir=os.environ.get("HF_HOME"),
            token=hf_token,
        )

        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")

        # Generate per-prompt with the right deficit count
        # Group prompts by needed count for efficient batching
        from collections import defaultdict
        by_needed = defaultdict(list)
        for label, prompt_text in prompts_to_run:
            needed = n - per_prompt_existing[prompt_text]
            by_needed[needed].append((label, prompt_text))

        total_gens = 0
        t0 = time.time()
        for needed, prompt_group in by_needed.items():
            sampling = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                n=needed,
            )
            prompt_texts = [p for _, p in prompt_group]
            outputs = llm.generate(prompt_texts, sampling)

            for (label, prompt_text), output in zip(prompt_group, outputs):
                existing = per_prompt_existing[prompt_text]
                for gen_idx, completion in enumerate(output.outputs):
                    gen_key = (prompt_text, existing + gen_idx)
                    if gen_key not in generate_family._partial:
                        generate_family._partial[gen_key] = {
                            "prompt": prompt_text}
                    generate_family._partial[gen_key][layer_name] = \
                        completion.text
                total_gens += needed

        gen_time = time.time() - t0
        print(f"  Generated {total_gens} completions in {gen_time:.1f}s "
              f"({gen_time/total_gens:.3f}s/gen)" if total_gens else "")

        # Free GPU memory before loading next model
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Write all complete entries to stash
    print(f"\n  Writing {len(generate_family._partial)} entries to stash...")
    stash_key = {
        "prompt": None,
        "temperature": temperature,
        "models": tuple(model_ids),
    }
    written = 0
    for (prompt_text, gen_idx), entry in generate_family._partial.items():
        stash_key_copy = dict(stash_key)
        stash_key_copy["prompt"] = prompt_text
        stash[stash_key_copy] = entry
        written += 1
    print(f"  Wrote {written} entries")

    generate_family._partial = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", required=True,
                        help="Comma-separated family keys (e.g. qwen,tulu,zephyr,pythia)")
    parser.add_argument("--n", type=int, default=100,
                        help="Generations per prompt per layer (default: 100)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without running")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",")]
    for fam in families:
        if fam not in MODEL_FAMILIES:
            print(f"Unknown family: {fam}")
            print(f"Available: {', '.join(MODEL_FAMILIES.keys())}")
            sys.exit(1)

    for fam in families:
        generate_family(fam, n=args.n, temperature=args.temperature,
                        max_tokens=args.max_tokens, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
