"""Step-level checkpoint analysis.

Traces repression emergence across SFT training steps by extracting logits
from a sequence of intermediate-step checkpoints (default: OLMo Think-SFT,
10 evenly-spaced steps from 1000 to 43000) and comparing each against the
fixed base model. Three phases:

1. **Download** intermediate checkpoints (snapshot_download per revision).
2. **Extract** last-position logits per (step, prompt), cached to HashStash.
3. **Compute** distribution-level metrics + per-word probability tracks
   from the cached logits (no models loaded).
"""
import gc
import os

import pandas as pd
import torch

from . import PATH_STASH
from .analysis import (
    distribution_entropy, js_divergence, kl_divergence, top_k_overlap,
)
from .embedding import extract_prompt_words
from .experiments import (
    DEFAULT_PROMPTS, DEFAULT_STEPS, STEP_REPO, TIER1_PROMPTS, TRACKED_WORDS,
)
from .models import _load_tokenizer, get_base_logits, load_model


def run_step_analysis(steps=None, prompts_set="tier1", category=None,
                      cache_dir=None, download_only=False, extract_only=False,
                      output_prefix=None):
    """Trace per-prompt distributional shift across SFT training steps.

    Args:
        steps: List of step numbers (default: ``DEFAULT_STEPS``).
        prompts_set: ``"tier1"`` (18 prompts) or ``"all"`` (47 prompts).
        category: Optional prefix filter (e.g. ``"sexual_explicit"``).
        cache_dir: HuggingFace cache directory for the step checkpoints
            (e.g. ``/Volumes/diderot/huggingface``).
        download_only: Phase 1 only (download checkpoints, exit).
        extract_only: Skip phase 1 (assume checkpoints already cached).
        output_prefix: CSV destinations are ``{prefix}_metrics.csv`` and
            ``{prefix}_words.csv`` (default ``data/step_analysis``).

    Returns:
        Tuple ``(metrics_df, words_df)``.
    """
    prompts = TIER1_PROMPTS if prompts_set == "tier1" else DEFAULT_PROMPTS
    if category:
        prompts = {k: v for k, v in prompts.items() if k.startswith(category)}
        if not prompts:
            raise ValueError(f"No prompts matching category '{category}'")

    steps = list(steps) if steps else list(DEFAULT_STEPS)
    repo = STEP_REPO
    output_prefix = output_prefix or "data/step_analysis"

    # ── Phase 1: download ──
    if not extract_only:
        from huggingface_hub import snapshot_download
        print(f"Downloading {len(steps)} checkpoints to {cache_dir or 'default cache'}...")
        for step in steps:
            rev = f"step{step}"
            print(f"\n  Downloading {repo}@{rev}...")
            snapshot_download(repo, revision=rev, cache_dir=cache_dir)
        print("\nAll downloads complete.")
        if download_only:
            return None, None

    # ── Phase 2: extract logits per (step, prompt), cache to stash ──
    from .psyche import _psyche_stash_compat
    stash = _psyche_stash_compat()

    base_name = "allenai/Olmo-3-1025-7B"
    print(f"\nChecking base model logits...")
    base_key_check = ("logits", base_name, list(prompts.values())[0])
    if base_key_check not in stash:
        print("  Base logits not cached — loading base model...")
        base_model, base_tok = load_model(base_name)
        for label, prompt in prompts.items():
            cache_key = ("logits", base_name, prompt)
            if cache_key not in stash:
                logits = get_base_logits(base_model, base_tok, prompt)
                stash[cache_key] = logits.cpu().numpy()
        del base_model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print("  Base logits ready.")

    base_logits_cache = {
        prompt: torch.tensor(stash[("logits", base_name, prompt)])
        for prompt in prompts.values()
    }

    tokenizer = _load_tokenizer(base_name)

    # Per-prompt word lists from generation data (with static tracked-words fallback)
    gen_parquet = "data/gen_battery_raw.parquet"
    if os.path.exists(gen_parquet):
        print("  Loading prompt-specific words from generation data...")
        prompt_word_lists = extract_prompt_words(gen_parquet)
    else:
        prompt_word_lists = {}

    all_words_set = set()
    for label in prompts:
        words = prompt_word_lists.get(label, [])
        for cat, cat_words in TRACKED_WORDS.items():
            words.extend(cat_words)
        prompt_word_lists[label] = list(dict.fromkeys(words))
        all_words_set.update(prompt_word_lists[label])

    word_token_ids = {}
    for word in all_words_set:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            word_token_ids[word] = ids[0]

    print(f"  Tracking {len(word_token_ids)} unique words across {len(prompts)} prompts")

    for step in steps:
        rev = f"step{step}"
        model_id = f"{repo}@{rev}"
        all_cached = all(
            ("logits", model_id, "step", prompt) in stash for prompt in prompts.values()
        )
        if all_cached:
            print(f"\n  step{step}: all logits cached, skipping.")
            continue

        print(f"\n{'=' * 60}\n  Extracting: {rev}\n{'=' * 60}")
        model, _ = load_model(repo, revision=rev, cache_dir=cache_dir)
        for label, prompt in prompts.items():
            cache_key = ("logits", model_id, "step", prompt)
            if cache_key in stash:
                continue
            logits = get_base_logits(model, tokenizer, prompt)
            stash[cache_key] = logits.cpu().numpy()
            print(f"    {label}")
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Phase 3: compute metrics from cached logits (pure math) ──
    print(f"\nComputing metrics...")

    metrics_rows, word_rows = [], []
    for step in steps:
        rev = f"step{step}"
        model_id = f"{repo}@{rev}"
        for label, prompt in prompts.items():
            cache_key = ("logits", model_id, "step", prompt)
            step_logits = torch.tensor(stash[cache_key])
            base_logits = base_logits_cache[prompt]

            ent_base = distribution_entropy(base_logits)
            ent_step = distribution_entropy(step_logits)
            metrics_rows.append({
                "step": step,
                "label": label,
                "prompt": prompt[:60],
                "entropy_base": round(float(ent_base), 6),
                "entropy_step": round(float(ent_step), 6),
                "entropy_drop": round(float(ent_base - ent_step), 6),
                "js_base_step": round(float(js_divergence(base_logits, step_logits)), 6),
                "kl_base_step": round(float(kl_divergence(base_logits, step_logits)), 6),
                "top50_overlap": round(float(top_k_overlap(base_logits, step_logits)), 4),
            })

            step_probs = torch.softmax(step_logits.float(), dim=0)
            base_probs = torch.softmax(base_logits.float(), dim=0)
            for word in prompt_word_lists.get(label, []):
                if word not in word_token_ids:
                    continue
                tid = word_token_ids[word]
                sp = float(step_probs[tid])
                bp = float(base_probs[tid])
                word_cat = "empirical"
                for cat, cat_words in TRACKED_WORDS.items():
                    if word in cat_words:
                        word_cat = cat
                        break
                word_rows.append({
                    "step": step,
                    "label": label,
                    "prompt": prompt[:60],
                    "word": word,
                    "word_category": word_cat,
                    "probability": round(sp, 8),
                    "base_probability": round(bp, 8),
                    "delta": round(sp - bp, 8),
                })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = f"{output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path} ({len(metrics_df)} rows)")

    words_df = pd.DataFrame(word_rows)
    words_path = f"{output_prefix}_words.csv"
    words_df.to_csv(words_path, index=False)
    print(f"Word tracking saved to {words_path} ({len(words_df)} rows)")

    return metrics_df, words_df
