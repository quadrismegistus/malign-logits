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

    prompts = list(TIER1_PROMPTS.items())
    print(f"\n{'=' * 60}")
    print(f"  {family_key} ({family.name}, {len(checkpoints)} layers)")
    print(f"  {len(prompts)} prompts × {n} generations")
    print(f"{'=' * 60}")

    # Check existing
    sample_prompt = prompts[0][1]
    existing = count_existing(stash, sample_prompt, temperature, model_ids)
    if existing >= n:
        print(f"  Already have {existing} generations, skipping")
        return

    needed = n - existing
    print(f"  Existing: {existing}, need {needed} more")

    if dry_run:
        print("  DRY RUN — would generate here")
        return

    from vllm import LLM, SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=needed,
    )

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

        # Generate all prompts at once — vLLM batches internally
        prompt_texts = [p for _, p in prompts]
        t0 = time.time()
        outputs = llm.generate(prompt_texts, sampling)
        gen_time = time.time() - t0
        total_gens = len(prompts) * needed
        print(f"  Generated {total_gens} completions in {gen_time:.1f}s "
              f"({gen_time/total_gens:.3f}s/gen)")

        # Store results — one stash entry per generation, matching existing format
        # Each entry is a dict with "prompt" + layer completions
        # But since we generate one model at a time, we need to accumulate
        # Actually, existing code stores all layers per entry:
        #   stash[key] = {"prompt": prompt, "base": text, "ego": text, ...}
        # We generate one layer at a time, so we need to build partial entries
        # and combine them.
        #
        # Simpler: store each generation as we did before, but we need all
        # layers in one dict. So we accumulate per-prompt, per-generation-index.
        if not hasattr(generate_family, '_partial'):
            generate_family._partial = {}

        for prompt_idx, output in enumerate(outputs):
            label, prompt_text = prompts[prompt_idx]
            for gen_idx, completion in enumerate(output.outputs):
                gen_key = (prompt_text, existing + gen_idx)
                if gen_key not in generate_family._partial:
                    generate_family._partial[gen_key] = {"prompt": prompt_text}
                generate_family._partial[gen_key][layer_name] = completion.text

        # Free GPU memory before loading next model
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Now write all complete entries to stash
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
