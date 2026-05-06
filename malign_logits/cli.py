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
    """Open UI in browser, or start Vite dev server with --dev."""
    if args.dev:
        import subprocess
        ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
        print(f"Starting Vite dev server (hot reload)...")
        print(f"Make sure `malign serve` is running for the API.")
        subprocess.run(["npm", "run", "dev"], cwd=ui_dir)
    else:
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
    """Run all data-production tasks across families."""
    from .produce import produce_all
    produce_all(
        families=args.families.split(",") if args.families else None,
        skip=args.skip.split(",") if args.skip else None,
        gen_n=args.gen_n,
        force=args.force,
    )


def cmd_ablation(args):
    """Run SFT ablation comparison: same base, different SFT data mixtures."""
    from .ablation import run_ablation
    run_ablation(
        ablation_keys=args.ablations or None,
        output_path=args.output,
    )


def cmd_battery(args):
    """Run prompt battery across one or all model families."""
    from .battery import run_battery
    run_battery(
        families=[args.family] if args.family else None,
        output_path=args.output,
    )


def cmd_logit_lens(args):
    """Run logit lens analysis across model layers."""
    from .logit_lens import run_logit_lens
    run_logit_lens(
        prompt=args.prompt,
        family=args.family or "olmo",
        top_k=args.top_k,
        min_layers=args.min_layers,
        output_path=args.output,
    )


def cmd_step_analysis(args):
    """Trace repression emergence across SFT training steps."""
    from .step_analysis import run_step_analysis
    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    try:
        run_step_analysis(
            steps=steps,
            prompts_set=args.prompts,
            category=args.category,
            cache_dir=args.cache_dir,
            download_only=args.download_only,
            extract_only=args.extract_only,
            output_prefix=args.output,
        )
    except ValueError as e:
        print(str(e))
        sys.exit(1)


def cmd_generate_battery(args):
    """Generate text across families, embed, compute metrics."""
    from .embedding import run_generate_battery
    try:
        run_generate_battery(
            families=[args.family] if args.family else None,
            prompts_set=args.prompts,
            category=args.category,
            n=args.n,
            max_new_tokens=args.tokens,
            output_prefix=args.output,
        )
    except ValueError as e:
        print(str(e))
        sys.exit(1)


def cmd_topic_drift(args):
    """Compute drift + surprisal + metonymy for cached generations."""
    from .embedding import run_topic_drift
    run_topic_drift(
        raw_path=getattr(args, 'input', None),
        output_path=args.output or "data/passage_metrics.csv",
    )


def cmd_surprisal(args):
    """Alias for passage-metrics (backward compat)."""
    cmd_topic_drift(args)


def cmd_taxonomy(args):
    """Classify displacement pairs into taxonomy types."""
    if getattr(args, 'analyze', False):
        from .taxonomy import analyze_taxonomy
        analyze_taxonomy(
            data_dir=getattr(args, 'data_dir', 'data') or 'data',
            output_path=args.output or 'data/taxonomy_summary.csv',
        )
    elif getattr(args, 'baseline', False):
        from .taxonomy import add_aligned_baseline
        add_aligned_baseline(
            family_key=args.family or "olmo",
            input_path=args.output,
            output_path=args.output,
        )
    else:
        from .taxonomy import run_taxonomy
        run_taxonomy(
            family_key=args.family or "olmo",
            all_prompts=args.all_prompts,
            output_path=args.output,
            measure_syntagmatic=not args.no_syntagmatic,
        )


def cmd_precompute(args):
    """Precompute top words, logits, and formation data for all prompts."""
    from .psyche import Psyche
    from .experiments import DEFAULT_PROMPTS, TIER1_PROMPTS
    key, _ = _get_family(args)

    prompts = TIER1_PROMPTS if args.prompts == "tier1" else DEFAULT_PROMPTS
    psyche = Psyche.from_family(key, load=True)
    n = len(prompts)
    print(f"Precomputing {n} prompts for {key} ({psyche.n_layers} layers)...")

    for i, (label, prompt) in enumerate(prompts.items()):
        print(f"\n  [{i+1}/{n}] {label}: {prompt[:50]}...")
        a = psyche.analyze(prompt)
        _ = a.base_words
        if a.ego_words is not None:
            _ = a.ego_words
        if a.superego_words is not None:
            _ = a.superego_words
        if a.instruct_words is not None:
            _ = a.instruct_words
        _ = a.formation_df
        print(f"    {len(a.formation_df)} words scored across {psyche.n_layers} layers")

    print(f"\nDone. All prompts cached to stash.")


def cmd_trajectory(args):
    """Measure trajectory geometry and run fold-vs-wall intervention."""
    from .psyche import Psyche
    key, _ = _get_family(args)

    psyche = Psyche.from_family(key, load=True)
    print(f"Loaded family={key}, n_layers={psyche.n_layers}")

    n_hidden = psyche.primary_process.model.config.num_hidden_layers
    layer = round(n_hidden * 0.8125)
    intervention_layers = [round(n_hidden * f) for f in (0.25, 0.5, 0.75, 0.875)]
    print(f"N_LAYERS={n_hidden}  LAYER={layer}  INTERVENTION_LAYERS={intervention_layers}")

    from .trajectory import run_trajectory_geometry, run_intervention

    run_trajectory_geometry(psyche, key, layer, out_dir="data",
                            n_passages=args.n_passages)

    if not args.skip_intervention:
        if psyche.superego is None:
            print("\nSkipping intervention: need at least base + superego (2 layers)")
        else:
            run_intervention(psyche, key, intervention_layers, out_dir="data",
                             n_epochs=args.n_epochs, lr=args.lr)

    print("\nDone.")


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
    ui.add_argument("--dev", action="store_true", help="Start Vite dev server with hot reload")
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

    # topic-drift
    td = subparsers.add_parser("topic-drift",
                               help="Compute within-generation topic drift (no models needed)")
    td.add_argument("--input", "-i",
                    help="Raw generation parquet (default: data/gen_battery_raw.parquet)")
    td.add_argument("--output", "-o",
                    help="Output CSV path (default: data/topic_drift.csv)")
    td.set_defaults(func=cmd_topic_drift)

    # surprisal
    su = subparsers.add_parser("surprisal",
                               help="Compute GPT-2 surprisal for cached generations")
    su.add_argument("--input", "-i",
                    help="Raw generation parquet (default: data/gen_battery_raw.parquet)")
    su.add_argument("--output", "-o",
                    help="Output CSV path (default: data/surprisal.csv)")
    su.add_argument("--model", default="gpt2",
                    help="Reference model for surprisal (default: gpt2)")
    su.set_defaults(func=cmd_surprisal)

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
    tx.add_argument("--baseline", action="store_true",
                    help="Add aligned-model syntagmatic_js baseline to existing taxonomy CSV")
    tx.add_argument("--analyze", action="store_true",
                    help="Cross-family analysis of all taxonomy CSVs (no models needed)")
    tx.add_argument("--data-dir", default="data",
                    help="Directory containing taxonomy CSVs (for --analyze)")
    tx.set_defaults(func=cmd_taxonomy)

    # precompute
    pc = subparsers.add_parser("precompute",
                               help="Precompute top words + formation data for all prompts")
    _add_family_arg(pc)
    pc.add_argument("--prompts", choices=["tier1", "all"], default="all",
                    help="Prompt set (default: all 47)")
    pc.set_defaults(func=cmd_precompute)

    # trajectory
    tj = subparsers.add_parser("trajectory",
                               help="Trajectory geometry + fold-vs-wall intervention")
    _add_family_arg(tj)
    tj.add_argument("--skip-intervention", action="store_true",
                    help="Run geometry only (skip v2/v2.5/v2.6 intervention)")
    tj.add_argument("--n-epochs", type=int, default=30,
                    help="Training epochs for v2.6 steering vector (default: 30)")
    tj.add_argument("--lr", type=float, default=0.05,
                    help="Learning rate for v2.6 steering vector (default: 0.05)")
    tj.add_argument("--n-passages", type=int, default=None,
                    help="Max passages per prompt from stash (default: all)")
    tj.set_defaults(func=cmd_trajectory)

    # produce-all
    pa = subparsers.add_parser("produce-all", help="Run all data production tasks")
    pa.add_argument("--families", help="Comma-separated families (default: all)")
    pa.add_argument("--skip", default="", help="Comma-separated tasks to skip: battery,ablation,generate,logit-lens,taxonomy,trajectory")
    pa.add_argument("--gen-n", type=int, default=30, help="Generations per prompt (default: 30)")
    pa.add_argument("--force", action="store_true", help="Recompute even if output CSVs exist")
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
