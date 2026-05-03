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
import os
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
    """Open UI in browser (requires `malign serve` running)."""
    import webbrowser
    url = f"http://127.0.0.1:{args.port}"
    print(f"Opening {url}")
    print("Make sure `malign serve` is running.")
    webbrowser.open(url)


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


def cmd_cloud(args):
    """Dispatch cloud subcommands."""
    from .cloud import main as cloud_main
    cloud_main(args)


def cmd_produce_all(args):
    """Run all data production tasks, grouped by family to minimize model reloading."""
    import gc
    import time
    import torch
    import pandas as pd
    from . import MODEL_FAMILIES, TULU_ABLATIONS
    from .psyche import Psyche
    from .experiments import DEFAULT_PROMPTS, TIER1_PROMPTS

    families = args.families.split(",") if args.families else list(MODEL_FAMILIES.keys())
    skip = set(args.skip.split(",")) if args.skip else set()
    gen_n = args.gen_n
    results = {}
    t0 = time.time()

    def _free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Phase 1: per-family tasks (models loaded once per family) ──

    all_battery = []

    for key in families:
        fam = MODEL_FAMILIES[key]
        print(f"\n{'=' * 60}")
        print(f"  {key}: {fam.name} ({fam.n_layers} layers)")
        print(f"{'=' * 60}")

        psyche = Psyche.from_family(key, load=True)

        # Battery
        if "battery" not in skip:
            print(f"\n  ── Battery ({key}) ──")
            try:
                metrics = psyche.battery_metrics()
                metrics["family"] = key
                all_battery.append(metrics)
                metrics.to_csv(f"data/battery_{key}.csv", index=False)
                print(f"  Saved data/battery_{key}.csv ({len(metrics)} prompts)")
                results[f"battery-{key}"] = "done"
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"battery-{key}"] = f"error: {e}"

        # Generation
        if "generate" not in skip:
            print(f"\n  ── Generation ({key}) ──")
            try:
                from .embedding import generate_many
                prompts = TIER1_PROMPTS
                for label, prompt in prompts.items():
                    print(f"    {label}: {prompt[:40]}...")
                    generate_many(psyche, prompt, n=gen_n, max_new_tokens=100)
                results[f"generate-{key}"] = "done"
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"generate-{key}"] = f"error: {e}"

        # Taxonomy
        if "taxonomy" not in skip:
            print(f"\n  ── Taxonomy ({key}) ──")
            try:
                from .taxonomy import run_taxonomy
                run_taxonomy(
                    family_key=key, all_prompts=True,
                    output_path=f"data/taxonomy_{key}.csv",
                    psyche=psyche,
                )
                results[f"taxonomy-{key}"] = "done"
            except ImportError:
                print(f"  Skipping taxonomy (spacy/wordfreq not installed)")
                results[f"taxonomy-{key}"] = "skipped (missing deps)"
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"taxonomy-{key}"] = f"error: {e}"

        del psyche
        _free()

    # Save combined battery
    if all_battery and "battery" not in skip:
        combined = pd.concat(all_battery, ignore_index=True)
        cols = ["family", "label", "prompt"] + [
            c for c in combined.columns if c not in ("family", "label", "prompt")
        ]
        combined = combined[cols]
        combined.to_csv("data/battery_results.csv", index=False)
        print(f"\nCombined battery: data/battery_results.csv ({len(combined)} rows)")

    # ── Phase 2: logit lens (uses Psyche — cached to stash) ──

    if "logit-lens" not in skip:
        import re as _re
        lens_prompts = list(TIER1_PROMPTS.items())[:6]

        for key in families:
            fam = MODEL_FAMILIES[key]
            from .psyche import Psyche as _Psyche
            psyche = _Psyche.from_family(key, load=True)
            for label, prompt in lens_prompts:
                print(f"\n  ── Logit lens: {key} / {label} ──")
                try:
                    analysis = psyche.analyze(prompt)
                    data = analysis.logit_lens_df
                    print(f"  {len(data['rows'])} data points, {len(data['word_sources'])} tracked")
                    results[f"logit-lens-{key}-{label}"] = "done"
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[f"logit-lens-{key}-{label}"] = f"error: {e}"
            del psyche
            _free()

    # ── Phase 3: ablation (loads base once, swaps SFT variants) ──

    if "ablation" not in skip and "tulu" in families:
        print(f"\n{'=' * 60}")
        print(f"  SFT Ablation comparison")
        print(f"{'=' * 60}")
        try:
            from .ablation import run_ablation
            run_ablation()
            results["ablation"] = "done"
        except Exception as e:
            print(f"  ERROR: {e}")
            results["ablation"] = f"error: {e}"

    # ── Phase 4: embed + compute generation metrics ──

    if "generate" not in skip:
        print(f"\n{'=' * 60}")
        print(f"  Embedding + generation metrics")
        print(f"{'=' * 60}")
        try:
            from .embedding import embed_generations, compute_generation_metrics, compute_concept_metrics
            embed_generations()
            gen_metrics = compute_generation_metrics()
            concept_metrics = compute_concept_metrics()
            if gen_metrics is not None:
                gen_metrics.to_csv("data/gen_battery_metrics.csv", index=False)
                print(f"  Saved data/gen_battery_metrics.csv")
            results["embed-metrics"] = "done"
        except Exception as e:
            print(f"  ERROR: {e}")
            results["embed-metrics"] = f"error: {e}"

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ALL TASKS COMPLETE ({elapsed / 3600:.1f}h)")
    print(f"{'=' * 60}")
    for k, v in sorted(results.items()):
        status = "✓" if v == "done" else "✗"
        print(f"  {status} {k:30s} {v}")


def cmd_ablation(args):
    """Run SFT ablation comparison: same base, different SFT data mixtures."""
    from .ablation import run_ablation
    run_ablation(
        ablation_keys=args.ablations or None,
        output_path=args.output,
    )


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
    import re
    import pandas as pd
    from .psyche import Psyche

    prompt = args.prompt
    key = args.family or "olmo"

    psyche = Psyche.from_family(key, load=True)
    analysis = psyche.analyze(prompt, top_k_first=200)

    print(f"Running logit lens for {key}: \"{prompt}\"")
    data = analysis.logit_lens_df
    rows = data["rows"]
    word_sources = data["word_sources"]
    print(f"  {len(rows)} data points across {psyche.n_layers} model layers")
    print(f"  {len(word_sources)} tracked words")

    result = pd.DataFrame(rows)

    tracked = [w for w in word_sources if "declining" in word_sources[w]]
    tracked += [w for w in word_sources if "rising" in word_sources[w] and w not in tracked]

    prompt_slug = re.sub(r'[^a-z0-9]+', '_', prompt.lower().strip())[:50].strip('_')
    words_slug = '_'.join(tracked[:5])

    if args.output:
        out = args.output
    else:
        basename = f"logit_lens.{key}.{prompt_slug}.{words_slug}"
        out = f"data/{basename}.csv"

    result.to_csv(out, index=False)
    print(f"Saved to {out}")

    from .viz import plot_logit_lens
    fig_path = f"figures/logit_lens.{key}.{prompt_slug}.{words_slug}.png"
    plot_logit_lens(result, prompt=prompt, family=key, top_k=args.top_k,
                    min_layers=args.min_layers, save_path=fig_path)
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
                stash[cache_key] = logits.cpu().numpy()
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
            stash[cache_key] = logits.cpu().numpy()
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


def cmd_taxonomy(args):
    """Classify displacement pairs into taxonomy types."""
    from .taxonomy import run_taxonomy
    run_taxonomy(
        family_key=args.family or "olmo",
        all_prompts=args.all_prompts,
        output_path=args.output,
        measure_syntagmatic=not args.no_syntagmatic,
    )


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
    ui = subparsers.add_parser("ui", help="Open UI in browser (requires malign serve)")
    ui.add_argument("--port", type=int, default=8421, help="Server port (default 8421)")
    ui.set_defaults(func=cmd_ui)

    # serve
    sv = subparsers.add_parser("serve", help="Start model server (keeps models loaded)")
    sv.add_argument("--port", type=int, default=8421, help="Port (default 8421)")
    _add_family_arg(sv)
    sv.set_defaults(func=cmd_serve)

    # ablation
    abl = subparsers.add_parser("ablation",
                                help="Compare SFT data ablations (base vs SFT variants)")
    abl.add_argument("ablations", nargs="*", default=None,
                     help="Ablation keys (default: all). Options: standard, no-safety, no-persona, no-math, no-wildchat")
    abl.add_argument("--output", "-o", help="Output CSV (default: data/ablation_results.csv)")
    abl.set_defaults(func=cmd_ablation)

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
    ll.add_argument("--min-layers", type=int, default=8, help="Min layers a top-k word must appear in to be plotted (default: 8)")
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

    # taxonomy
    tx = subparsers.add_parser("taxonomy",
                               help="Classify displacement pairs into taxonomy types")
    _add_family_arg(tx)
    tx.add_argument("--all-prompts", action="store_true",
                    help="Use all 47 prompts (default: Tier-1 subset)")
    tx.add_argument("--output", "-o",
                    help="Output CSV path (default: data/displacement_taxonomy.csv)")
    tx.add_argument("--no-syntagmatic", action="store_true",
                    help="Skip syntagmatic_js measurement (faster; drops the continuous syntagmatic-disruption column)")
    tx.set_defaults(func=cmd_taxonomy)

    # produce-all
    pa = subparsers.add_parser("produce-all", help="Run all data production tasks")
    pa.add_argument("--families", help="Comma-separated families (default: all)")
    pa.add_argument("--skip", default="", help="Comma-separated tasks to skip: battery,ablation,generate,logit-lens,taxonomy")
    pa.add_argument("--gen-n", type=int, default=30, help="Generations per prompt (default: 30)")
    pa.set_defaults(func=cmd_produce_all)

    # cloud
    cloud = subparsers.add_parser("cloud", help="Vast.ai GPU instance management")
    cloud.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    cloud_sub = cloud.add_subparsers(dest="cloud_command")

    cloud_sub.add_parser("launch", help="Find and rent cheapest A100 80GB")
    cloud_sub.add_parser("setup", help="Install malign-logits on instance")

    cr = cloud_sub.add_parser("run", help="Start produce-all in tmux")
    cr.add_argument("--families", help="Comma-separated families")
    cr.add_argument("--skip", default="", help="Tasks to skip")

    cloud_sub.add_parser("status", help="Check progress and cost")
    cloud_sub.add_parser("download", help="Download stash + data back")
    cloud_sub.add_parser("stop", help="Destroy instance")
    cloud_sub.add_parser("attach", help="Attach to tmux session")

    cl = cloud_sub.add_parser("log", help="Tail the batch log")
    cl.add_argument("--lines", "-n", type=int, default=30)

    cs = cloud_sub.add_parser("ssh", help="Open SSH session")
    cs.add_argument("ssh_command", nargs="*")

    cloud.set_defaults(func=cmd_cloud)

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
