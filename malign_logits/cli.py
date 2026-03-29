"""
CLI entrypoint for malign-logits.

Usage:
    malign download-models                    # Download default family models
    malign download-models --family llama
    malign serve                              # Start model server (default family)
    malign serve --family llama          # Start with Llama 3
    malign ui                                 # Launch Gradio web UI
    malign info                               # Show all families
    malign info --family llama           # Show specific family
"""

import argparse
import sys


def _get_family(args):
    """Get ModelFamily from args, defaulting to DEFAULT_FAMILY."""
    from . import MODEL_FAMILIES, DEFAULT_FAMILY
    key = getattr(args, "family", None) or DEFAULT_FAMILY
    if key not in MODEL_FAMILIES:
        print(f"Unknown family: {key}")
        print(f"Available: {', '.join(MODEL_FAMILIES.keys())}")
        sys.exit(1)
    return key, MODEL_FAMILIES[key]


def cmd_download_models(args):
    """Download model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    _, fam = _get_family(args)

    targets = {}
    if args.model:
        mapping = {"base": fam.base, "sft": fam.ego, "dpo": fam.superego, "instruct": fam.reinforced_superego}
        model_id = mapping.get(args.model)
        if model_id is None:
            print(f"Family {fam.name} has no {args.model} checkpoint.")
            sys.exit(1)
        targets = {args.model: model_id}
    elif args.all:
        for name, model_id in [("base", fam.base), ("sft", fam.ego), ("dpo", fam.superego), ("instruct", fam.reinforced_superego)]:
            if model_id is not None:
                targets[name] = model_id
    else:
        # Default: download all non-RLVR checkpoints
        for name, model_id in [("base", fam.base), ("sft", fam.ego), ("dpo", fam.superego)]:
            if model_id is not None:
                targets[name] = model_id

    for name, model_id in targets.items():
        print(f"\n{'='*60}")
        print(f"Downloading {name}: {model_id}")
        print(f"{'='*60}")
        snapshot_download(model_id)
        print(f"  Done: {model_id}")

    print(f"\nAll downloads complete.")


def cmd_ui(args):
    """Launch Gradio web UI."""
    from .app import launch
    launch(server_name=args.host, server_port=args.port, share=args.share)


def cmd_serve(args):
    """Start model server."""
    from .server import serve
    key, _ = _get_family(args)
    serve(port=args.port, family=key)


def cmd_info(args):
    """Print model families and configuration."""
    from . import MODEL_FAMILIES, DEFAULT_FAMILY

    if args.family:
        key, fam = _get_family(args)
        _print_family(key, fam)
    else:
        print("malign-logits model families:\n")
        for key, fam in MODEL_FAMILIES.items():
            default = " (default)" if key == DEFAULT_FAMILY else ""
            print(f"  {key}{default}")
            _print_family(key, fam, indent=4)
            print()


def _print_family(key, fam, indent=2):
    """Print a single model family."""
    pad = " " * indent
    roles = {
        "base": "Id / primary statistical field",
        "ego": "Ego / socialised subject",
        "superego": "Superego / Name-of-the-Father",
        "reinforced_superego": "Ego-ideal / reinforced superego",
    }
    print(f"{pad}{fam.name} ({fam.n_layers} layers):")
    for attr in ["base", "ego", "superego", "reinforced_superego"]:
        model_id = getattr(fam, attr)
        if model_id is not None:
            print(f"{pad}  {attr:<22s}  {roles[attr]:<34s}  {model_id}")


def cmd_battery(args):
    """Run prompt battery across one or all model families."""
    import gc
    import torch
    from . import MODEL_FAMILIES
    from .psyche import Psyche

    families = [args.family] if args.family else list(MODEL_FAMILIES.keys())
    all_metrics = []

    for key in families:
        fam = MODEL_FAMILIES[key]
        print(f"\n{'=' * 60}")
        print(f"  {key}: {fam.name} ({fam.n_layers} layers)")
        print(f"{'=' * 60}")

        psyche = Psyche.from_family(key, load=True)
        metrics = psyche.battery_metrics()
        metrics["family"] = key
        all_metrics.append(metrics)

        # Free memory before loading next family
        del psyche
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    import pandas as pd
    combined = pd.concat(all_metrics, ignore_index=True)
    cols = ["family", "label", "prompt"] + [
        c for c in combined.columns if c not in ("family", "label", "prompt")
    ]
    combined = combined[cols]

    out = args.output or "data/battery_results.csv"
    combined.to_csv(out, index=False)
    print(f"\nResults saved to {out}")
    print(f"\n{combined.to_string()}")


def cmd_logit_lens(args):
    """Run logit lens analysis across model layers."""
    import gc
    import torch
    import pandas as pd
    from . import MODEL_FAMILIES
    from .models import load_model, logit_lens_words
    from .embedding import extract_prompt_words

    prompt = args.prompt
    key = args.family or "olmo"
    fam = MODEL_FAMILIES[key]

    # Determine words to track
    if args.words:
        words = [w.strip() for w in args.words.split(",")]
    else:
        import os
        gen_parquet = "data/gen_battery_raw.parquet"
        if os.path.exists(gen_parquet):
            prompt_words = extract_prompt_words(gen_parquet)
            # Find a matching label
            from .experiments import TIER1_PROMPTS, DEFAULT_PROMPTS
            all_prompts = {**DEFAULT_PROMPTS, **TIER1_PROMPTS}
            words = None
            for label, p in all_prompts.items():
                if p == prompt and label in prompt_words:
                    words = prompt_words[label][:10]
                    break
        if not words:
            words = ["kill", "fuck", "kiss", "said", "the", "scream", "massage"]
        print(f"Tracking words: {words}")

    # Load each model layer and run logit lens
    layers_to_load = [("base", fam.base)]
    if fam.ego:
        layers_to_load.append(("ego", fam.ego))
    if fam.superego:
        layers_to_load.append(("superego", fam.superego))

    all_dfs = []
    for layer_name, model_id in layers_to_load:
        label = {"base": "BASE", "ego": "SFT", "superego": "DPO"}.get(layer_name, layer_name)
        print(f"\n  {label}: {model_id}")
        model, tokenizer = load_model(model_id)

        df = logit_lens_words(model, tokenizer, prompt, words=words, top_k=args.top_k)
        df["model"] = layer_name
        df["model_id"] = model_id
        all_dfs.append(df)

        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    result = pd.concat(all_dfs, ignore_index=True)

    # Build descriptive filename from prompt and words
    import re
    prompt_slug = re.sub(r'[^a-z0-9]+', '_', prompt.lower().strip())[:50].strip('_')
    words_slug = '_'.join(words[:5])

    if args.output:
        out = args.output
    else:
        basename = f"logit_lens.{key}.{prompt_slug}.{words_slug}"
        out = f"data/{basename}.csv"

    result.to_csv(out, index=False)
    print(f"\nSaved to {out} ({len(result)} rows)")

    # Generate figure
    from .viz import plot_logit_lens
    fig = plot_logit_lens(result, prompt=prompt)
    basename_fig = f"logit_lens.{key}.{prompt_slug}.{words_slug}"
    fig_path = f"figures/{basename_fig}.png"
    fig.write_image(fig_path, scale=2)
    print(f"Figure saved to {fig_path}")


def cmd_step_analysis(args):
    """Trace repression emergence across SFT training steps."""
    import gc
    import torch
    import pandas as pd
    from .experiments import (
        TIER1_PROMPTS, DEFAULT_PROMPTS, TRACKED_WORDS,
        DEFAULT_STEPS, STEP_REPO,
    )
    from .analysis import distribution_entropy, js_divergence, kl_divergence, top_k_overlap
    from .embedding import extract_prompt_words

    prompts = TIER1_PROMPTS if args.prompts == "tier1" else DEFAULT_PROMPTS
    if args.category:
        prompts = {k: v for k, v in prompts.items() if k.startswith(args.category)}
        if not prompts:
            print(f"No prompts matching category '{args.category}'")
            sys.exit(1)

    steps = [int(s) for s in args.steps.split(",")] if args.steps else DEFAULT_STEPS
    cache_dir = args.cache_dir
    repo = STEP_REPO

    # Phase 1: Download
    if not args.extract_only:
        from huggingface_hub import snapshot_download
        print(f"Downloading {len(steps)} checkpoints to {cache_dir or 'default cache'}...")
        for step in steps:
            rev = f"step{step}"
            print(f"\n  Downloading {repo}@{rev}...")
            snapshot_download(repo, revision=rev, cache_dir=cache_dir)
        print("\nAll downloads complete.")
        if args.download_only:
            return

    # Phase 2: Extract logits
    from .models import load_model, get_base_logits
    from .psyche import ModelLayer
    from . import PATH_STASH
    from hashstash import HashStash

    stash = HashStash(root_dir=PATH_STASH)

    # Ensure base model logits are cached (shared with OLMo family)
    base_name = "allenai/Olmo-3-1025-7B"
    base_logits_cache = {}
    print(f"\nChecking base model logits...")
    base_key_check = ("logits", base_name, "base", list(prompts.values())[0])
    if base_key_check not in stash:
        print("  Base logits not cached — loading base model...")
        base_model, base_tok = load_model(base_name)
        for label, prompt in prompts.items():
            cache_key = ("logits", base_name, "base", prompt)
            if cache_key not in stash:
                logits = get_base_logits(base_model, base_tok, prompt)
                stash[cache_key] = logits.numpy()
        del base_model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print("  Base logits ready.")

    # Load base logits for all prompts
    for label, prompt in prompts.items():
        cache_key = ("logits", base_name, "base", prompt)
        base_logits_cache[prompt] = torch.tensor(stash[cache_key])

    # Load tokenizer once (shared across all checkpoints)
    from .models import _load_tokenizer
    tokenizer = _load_tokenizer(base_name)

    # Build per-prompt word lists from generation data + static fallback
    import os
    gen_parquet = "data/gen_battery_raw.parquet"
    if os.path.exists(gen_parquet):
        print("  Loading prompt-specific words from generation data...")
        prompt_word_lists = extract_prompt_words(gen_parquet)
    else:
        prompt_word_lists = {}

    # Also include static tracked words as fallback
    all_words_set = set()
    for label in prompts:
        words = prompt_word_lists.get(label, [])
        # Add static tracked words too
        for cat, cat_words in TRACKED_WORDS.items():
            words.extend(cat_words)
        prompt_word_lists[label] = list(dict.fromkeys(words))  # dedupe, preserve order
        all_words_set.update(prompt_word_lists[label])

    # Encode all unique words to token IDs (leading space for continuation)
    word_token_ids = {}
    for word in all_words_set:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            word_token_ids[word] = ids[0]

    print(f"  Tracking {len(word_token_ids)} unique words across {len(prompts)} prompts")

    # Extract logits per step checkpoint
    for step in steps:
        rev = f"step{step}"
        model_id = f"{repo}@{rev}"

        # Check if all prompts are already cached
        all_cached = all(
            ("logits", model_id, "step", prompt) in stash
            for prompt in prompts.values()
        )
        if all_cached:
            print(f"\n  step{step}: all logits cached, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Extracting: {rev}")
        print(f"{'=' * 60}")

        model, _ = load_model(repo, revision=rev, cache_dir=cache_dir)

        for label, prompt in prompts.items():
            cache_key = ("logits", model_id, "step", prompt)
            if cache_key in stash:
                continue
            logits = get_base_logits(model, tokenizer, prompt)
            stash[cache_key] = logits.numpy()
            print(f"    {label}")

        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Phase 3: Compute metrics (pure math, no models)
    print(f"\nComputing metrics...")

    metrics_rows = []
    word_rows = []

    for step in steps:
        rev = f"step{step}"
        model_id = f"{repo}@{rev}"

        for label, prompt in prompts.items():
            cache_key = ("logits", model_id, "step", prompt)
            step_logits = torch.tensor(stash[cache_key])
            base_logits = base_logits_cache[prompt]

            # Distribution-level metrics
            entropy_base = distribution_entropy(base_logits)
            entropy_step = distribution_entropy(step_logits)
            js = js_divergence(base_logits, step_logits)
            kl = kl_divergence(base_logits, step_logits)
            overlap = top_k_overlap(base_logits, step_logits)

            metrics_rows.append({
                "step": step,
                "label": label,
                "prompt": prompt[:60],
                "entropy_base": round(float(entropy_base), 6),
                "entropy_step": round(float(entropy_step), 6),
                "entropy_drop": round(float(entropy_base - entropy_step), 6),
                "js_base_step": round(float(js), 6),
                "kl_base_step": round(float(kl), 6),
                "top50_overlap": round(float(overlap), 4),
            })

            # Per-word probabilities (prompt-specific word list)
            step_probs = torch.softmax(step_logits.float(), dim=0)
            base_probs = torch.softmax(base_logits.float(), dim=0)

            for word in prompt_word_lists.get(label, []):
                if word not in word_token_ids:
                    continue
                tid = word_token_ids[word]
                sp = float(step_probs[tid])
                bp = float(base_probs[tid])
                # Categorize: check if it's in static tracked categories
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

    # Save
    out_prefix = args.output or "data/step_analysis"

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = f"{out_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path} ({len(metrics_df)} rows)")

    words_df = pd.DataFrame(word_rows)
    words_path = f"{out_prefix}_words.csv"
    words_df.to_csv(words_path, index=False)
    print(f"Word tracking saved to {words_path} ({len(words_df)} rows)")


def cmd_generate_battery(args):
    """Generate text across families, embed, compute metrics."""
    import gc
    import torch
    import pandas as pd
    from . import MODEL_FAMILIES
    from .psyche import Psyche
    from .experiments import TIER1_PROMPTS, DEFAULT_PROMPTS
    from .embedding import (
        generate_many, embed_generations, compute_generation_metrics,
        compute_concept_metrics,
    )

    prompts = TIER1_PROMPTS if args.prompts == "tier1" else DEFAULT_PROMPTS
    if args.category:
        prompts = {k: v for k, v in prompts.items() if k.startswith(args.category)}
        if not prompts:
            print(f"No prompts matching category '{args.category}'")
            sys.exit(1)
    families = [args.family] if args.family else list(MODEL_FAMILIES.keys())
    n = args.n

    # Phase 1: generate (models loaded, one family at a time)
    from .embedding import _gen_stash_path, _check_cached_count
    all_psg = []
    for key in families:
        fam = MODEL_FAMILIES[key]

        # Check how many prompts already have enough cached generations
        model_ids = [fam.base]
        if fam.ego:
            model_ids.append(fam.ego)
        if fam.superego:
            model_ids.append(fam.superego)
        needed_prompts = {}
        for label, prompt in prompts.items():
            cached = _check_cached_count(prompt, temperature=1.0,
                                         model_ids=model_ids)
            if cached < n:
                needed_prompts[label] = prompt

        print(f"\n{'=' * 60}")
        print(f"  {key} ({fam.name}, {fam.n_layers} layers)")
        cached_count = len(prompts) - len(needed_prompts)
        if cached_count:
            print(f"  {cached_count}/{len(prompts)} prompts fully cached, "
                  f"{len(needed_prompts)} need generation")
        else:
            print(f"  {len(prompts)} prompts x {n} generations")
        print(f"{'=' * 60}")

        if needed_prompts:
            psyche = Psyche.from_family(key, load=True)

            for label, prompt in needed_prompts.items():
                print(f"\n  {label}: {prompt[:50]}...")
                generate_many(psyche, prompt, n=n,
                              max_new_tokens=args.tokens)

            del psyche
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print("  All cached, skipping model load.")

        # Collect all results (cached + newly generated)
        psyche_cache = Psyche.from_family(key, load=False)
        for label, prompt in prompts.items():
            df = generate_many(psyche_cache, prompt, n=n,
                               max_new_tokens=args.tokens)
            df["family"] = key
            df["label"] = label
            all_psg.append(df)

    psg_df = pd.concat(all_psg, ignore_index=True)
    print(f"\nTotal generations: {len(psg_df)}")

    # Phase 2: embed (SentenceTransformer, cheap)
    print("\nEmbedding all generations...")
    embeds_df = embed_generations(psg_df)

    # Phase 3: compute metrics per (family, prompt)
    print("Computing metrics...")
    metrics_rows = []
    for (fam, label), idx in psg_df.groupby(["family", "label"]).groups.items():
        sub_psg = psg_df.loc[idx].reset_index(drop=True)
        sub_emb = embeds_df.loc[idx].reset_index(drop=True)

        m = compute_generation_metrics(sub_emb, sub_psg)
        m.update(compute_concept_metrics(sub_emb, sub_psg))
        m["family"] = fam
        m["label"] = label
        m["prompt"] = sub_psg["prompt"].iloc[0][:60]
        m["n_generations"] = len(sub_psg)
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    id_cols = ["family", "label", "prompt", "n_generations"]
    other_cols = [c for c in metrics_df.columns if c not in id_cols]
    metrics_df = metrics_df[id_cols + sorted(other_cols)]

    # Save outputs
    out_prefix = args.output or "data/gen_battery"
    metrics_path = f"{out_prefix}_metrics.csv"
    raw_path = f"{out_prefix}_raw.parquet"

    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Save raw generations + embeddings
    raw_df = pd.concat([psg_df, embeds_df], axis=1)
    try:
        raw_df.to_parquet(raw_path, index=False)
        print(f"Raw data saved to {raw_path}")
    except ImportError:
        raw_csv = raw_path.replace(".parquet", ".csv")
        raw_df.to_csv(raw_csv, index=False)
        print(f"Raw data saved to {raw_csv} (install pyarrow for parquet)")

    print(f"\n{metrics_df.to_string()}")


def _add_family_arg(parser):
    """Add --family argument to a subparser."""
    from . import MODEL_FAMILIES
    parser.add_argument(
        "--family",
        choices=list(MODEL_FAMILIES.keys()),
        default=None,
        help="Model family (default: olmo)",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="malign",
        description="malign-logits: psychoanalytic analysis of LLM probability distributions",
    )
    subparsers = parser.add_subparsers(dest="command")

    # download-models
    dl = subparsers.add_parser(
        "download-models",
        help="Download model weights from HuggingFace",
    )
    dl.add_argument(
        "--model",
        choices=["base", "sft", "dpo", "instruct"],
        help="Download a specific model only",
    )
    dl.add_argument(
        "--all",
        action="store_true",
        help="Download all checkpoints including RLVR",
    )
    _add_family_arg(dl)
    dl.set_defaults(func=cmd_download_models)

    # ui
    ui = subparsers.add_parser("ui", help="Launch Gradio web UI")
    ui.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    ui.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    ui.add_argument("--share", action="store_true", help="Create public Gradio link")
    ui.set_defaults(func=cmd_ui)

    # serve
    sv = subparsers.add_parser("serve", help="Start model server (keeps models loaded)")
    sv.add_argument("--port", type=int, default=8421, help="Port (default 8421)")
    _add_family_arg(sv)
    sv.set_defaults(func=cmd_serve)

    # battery
    bat = subparsers.add_parser("battery", help="Run prompt battery across families")
    _add_family_arg(bat)
    bat.add_argument("--output", "-o", help="Output CSV path (default: data/battery_results.csv)")
    bat.set_defaults(func=cmd_battery)

    # generate-battery
    gb = subparsers.add_parser("generate-battery",
                               help="Generate text across families, embed, compute metrics")
    _add_family_arg(gb)
    gb.add_argument("--prompts", choices=["tier1", "all"], default="tier1",
                    help="Prompt set (default: tier1 = 18 high-variance prompts)")
    gb.add_argument("--category", "-c",
                    help="Filter to prompts starting with this prefix (e.g. sexual_explicit, violence)")
    gb.add_argument("--n", type=int, default=30,
                    help="Generations per prompt per model (default: 30)")
    gb.add_argument("--tokens", type=int, default=100,
                    help="Max new tokens per generation (default: 100)")
    gb.add_argument("--output", "-o",
                    help="Output prefix (default: data/gen_battery)")
    gb.set_defaults(func=cmd_generate_battery)

    # logit-lens
    ll = subparsers.add_parser("logit-lens",
                               help="Run logit lens analysis across network layers")
    ll.add_argument("prompt", help="The prompt to analyze")
    _add_family_arg(ll)
    ll.add_argument("--words", "-w", help="Comma-separated words to always include (default: auto from generations)")
    ll.add_argument("--top-k", type=int, default=5, help="Top-k predictions per layer (default: 5)")
    ll.add_argument("--output", "-o", help="Output CSV path (default: data/logit_lens.csv)")
    ll.set_defaults(func=cmd_logit_lens)

    # step-analysis
    sa = subparsers.add_parser("step-analysis",
                               help="Trace repression across SFT training steps")
    sa.add_argument("--steps", help="Comma-separated step numbers (default: 10 evenly spaced)")
    sa.add_argument("--cache-dir", help="HuggingFace cache dir for checkpoints (e.g. /Volumes/diderot/huggingface)")
    sa.add_argument("--prompts", choices=["tier1", "all"], default="tier1",
                    help="Prompt set (default: tier1)")
    sa.add_argument("--category", "-c", help="Filter to prompts matching this prefix")
    sa.add_argument("--download-only", action="store_true", help="Only download checkpoints")
    sa.add_argument("--extract-only", action="store_true", help="Only extract logits (skip download)")
    sa.add_argument("--output", "-o", help="Output prefix (default: data/step_analysis)")
    sa.set_defaults(func=cmd_step_analysis)

    # info
    info = subparsers.add_parser("info", help="Print model families and configuration")
    _add_family_arg(info)
    info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
