"""
CLI entrypoint for malign-logits.

Usage:
    malign download-models          # Download all 3 default models (base, SFT, DPO)
    malign download-models --all    # Download all 4 models (including RLVR)
    malign download-models --model base
    malign download-models --model sft
    malign download-models --model dpo
    malign download-models --model instruct
"""

import argparse
import sys


def _model_ids():
    from . import BASE_MODEL_NAME, SFT_MODEL_NAME, DPO_MODEL_NAME, INSTRUCT_MODEL_NAME
    return {
        "base": BASE_MODEL_NAME,
        "sft": SFT_MODEL_NAME,
        "dpo": DPO_MODEL_NAME,
        "instruct": INSTRUCT_MODEL_NAME,
    }


def cmd_download_models(args):
    """Download model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    ids = _model_ids()

    if args.model:
        targets = {args.model: ids[args.model]}
    elif args.all:
        targets = ids
    else:
        targets = {k: v for k, v in ids.items() if k != "instruct"}

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
    launch(server_port=args.port, share=args.share)


def cmd_info(args):
    """Print model IDs and project info."""
    ids = _model_ids()
    print("malign-logits model configuration:\n")
    print(f"  {'Layer':<12s}  {'Role':<28s}  {'Model ID'}")
    print(f"  {'─'*12}  {'─'*28}  {'─'*40}")
    roles = {
        "base": "Id / primary statistical field",
        "sft": "Ego / socialised subject",
        "dpo": "Superego / Name-of-the-Father",
        "instruct": "Reinforced superego / ego-ideal",
    }
    for name, model_id in ids.items():
        print(f"  {name:<12s}  {roles[name]:<28s}  {model_id}")


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
        help="Download all 4 models (default downloads 3: base, SFT, DPO)",
    )
    dl.set_defaults(func=cmd_download_models)

    # ui
    ui = subparsers.add_parser("ui", help="Launch Gradio web UI")
    ui.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    ui.add_argument("--share", action="store_true", help="Create public Gradio link")
    ui.set_defaults(func=cmd_ui)

    # info
    info = subparsers.add_parser("info", help="Print model configuration")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
