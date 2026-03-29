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
