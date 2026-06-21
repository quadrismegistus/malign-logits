#!/usr/bin/env python
"""Generate completions via DeepSeek API and store in the malign-logits cache.

DeepSeek's API is OpenAI-compatible and returns top-20 logprobs per token.
Uses chat completions endpoint (completions endpoint doesn't support
deepseek-chat). Logprobs stored in the gen_logprobs cache stash.

Models:
    deepseek-chat       — DeepSeek-V3 (aligned)

Usage:
    # Pilot: tier1 prompts, 5 gens each
    python scripts/deepseek_generate.py --prompts tier1 --n 5

    # Full run: all 47 prompts, 100 gens
    python scripts/deepseek_generate.py --prompts all --n 100

    # BOS-only generation (unconditional)
    python scripts/deepseek_generate.py --bos --n 100

    # Migrate existing flat-file logprobs into cache
    python scripts/deepseek_generate.py --migrate-logprobs

Requires: pip install openai
          export DEEPSEEK_API_KEY=sk-...
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits.experiments import DEFAULT_PROMPTS, TIER1_PROMPTS

API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-chat"

MODEL_PROMPTED = "deepseek/deepseek-chat"
MODEL_RAW = "deepseek/deepseek-chat-raw"

SYSTEM_PROMPT = ("Continue the following text. "
                 "Write only the continuation, no commentary or explanation.")

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0


def get_client():
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: set DEEPSEEK_API_KEY environment variable")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=API_BASE)


def get_cache():
    from malign_logits.cache import get_cache
    return get_cache()


def _extract_logprobs(choice):
    """Extract logprobs list from a chat completion choice."""
    if not choice.logprobs or not choice.logprobs.content:
        return None
    return [
        {
            "token": t.token,
            "logprob": t.logprob,
            "top_logprobs": [
                {"token": tp.token, "logprob": tp.logprob}
                for tp in t.top_logprobs
            ] if t.top_logprobs else [],
        }
        for t in choice.logprobs.content
    ]


def generate_chat(client, prompt, n=1, max_tokens=100, temperature=1.0,
                  logprobs=True, system_prompt=None):
    """Call DeepSeek chat completions endpoint with retry logic."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=20 if logprobs else None,
            )
            return response
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Rate limited, waiting {delay:.0f}s...")
                time.sleep(delay)
            else:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Error: {err[:100]}, retry in {delay:.0f}s...")
                time.sleep(delay)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def run(prompts_set="tier1", n=100, temperature=1.0, max_tokens=100,
        bos=False, dry_run=False, save_lp=True, batch_size=1, raw=False):
    client = get_client()
    cache = get_cache()

    model_id = MODEL_RAW if raw else MODEL_PROMPTED
    system_prompt = None if raw else SYSTEM_PROMPT

    if bos:
        prompt_dict = {"bos": ""}
    elif prompts_set == "all":
        prompt_dict = DEFAULT_PROMPTS
    else:
        prompt_dict = TIER1_PROMPTS

    mode = "raw (no system prompt)" if raw else f"prompted"
    print(f"\nDeepSeek generation: {model_id}")
    print(f"  {len(prompt_dict)} prompts × {n} generations")
    print(f"  temperature={temperature}, max_tokens={max_tokens}")
    print(f"  mode: {mode}")

    total_generated = 0
    total_skipped = 0

    for label, prompt_text in prompt_dict.items():
        existing = cache.count_generations(model_id, prompt_text, temp=temperature)
        needed = n - existing
        if needed <= 0:
            total_skipped += 1
            continue

        print(f"\n  {label}: {existing}/{n} exist, generating {needed}...")

        if dry_run:
            continue

        generated = 0
        while generated < needed:
            batch = min(batch_size, needed - generated)
            try:
                response = generate_chat(
                    client, prompt_text, n=batch,
                    max_tokens=max_tokens, temperature=temperature,
                    logprobs=save_lp, system_prompt=system_prompt,
                )
                for i, choice in enumerate(response.choices):
                    text = choice.message.content
                    idx = existing + generated + i
                    cache.set_generation(model_id, prompt_text, text,
                                         temp=temperature, idx=idx)
                    if save_lp:
                        lp_data = _extract_logprobs(choice)
                        if lp_data:
                            cache.set_gen_logprobs(model_id, prompt_text,
                                                   lp_data, temp=temperature,
                                                   idx=idx)

                generated += batch
                total_generated += batch
                print(f"    {generated}/{needed}", end="\r")
            except Exception as e:
                print(f"    Error at batch {generated}: {e}")
                break

        print(f"    {label}: done ({generated} new)")

    if total_skipped:
        print(f"\n  Skipped {total_skipped} prompts (already complete)")
    print(f"\n  Total new generations: {total_generated}")
    print("Done.")


def migrate_logprobs():
    """Migrate flat-file logprobs from data/raw/deepseek_logprobs/ into cache."""
    from pathlib import Path
    lp_dir = Path("data/raw/deepseek_logprobs")
    if not lp_dir.exists():
        print("No flat-file logprobs to migrate.")
        return

    cache = get_cache()
    files = sorted(lp_dir.glob("*.json"))
    print(f"Migrating {len(files)} logprob files into gen_logprobs cache...")

    migrated = 0
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        model = d["model"]
        prompt = d["prompt"]
        idx = d["idx"]
        lp_data = d["logprobs"]
        if not cache.has_gen_logprobs(model, prompt, temp=1.0, idx=idx):
            cache.set_gen_logprobs(model, prompt, lp_data, temp=1.0, idx=idx)
            migrated += 1

    print(f"  Migrated {migrated}/{len(files)} into cache.")
    if migrated == len(files):
        print(f"  All migrated. You can delete {lp_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompts", default="tier1", choices=["tier1", "all"],
                        help="Prompt set: tier1 (18) or all (47)")
    parser.add_argument("--n", type=int, default=100,
                        help="Generations per prompt (default: 100)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--bos", action="store_true",
                        help="BOS-only unconditional generation")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Generations per API call (default: 1)")
    parser.add_argument("--raw", action="store_true",
                        help="No system prompt (stored as deepseek-chat-raw)")
    parser.add_argument("--no-logprobs", action="store_true",
                        help="Don't save per-token logprobs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated")
    parser.add_argument("--migrate-logprobs", action="store_true",
                        help="Migrate flat-file logprobs into cache")
    args = parser.parse_args()

    if args.migrate_logprobs:
        migrate_logprobs()
        return

    run(
        prompts_set=args.prompts,
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        bos=args.bos,
        dry_run=args.dry_run,
        save_lp=not args.no_logprobs,
        batch_size=args.batch_size,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()
