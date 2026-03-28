"""
Gradio web UI for malign-logits.

Launch with:
    malign ui
    # or
    python -m malign_logits.app
"""

import io
import sys
import threading
import traceback
from contextlib import redirect_stdout

import gradio as gr
import pandas as pd

# Lazy globals — populated by launch()
_psyche = None
_cache = {}  # prompt -> PromptAnalysis
_computing = {}  # prompt -> threading.Event (set when done)


def _get_psyche():
    global _psyche
    if _psyche is None:
        from .psyche import Psyche
        _psyche = Psyche.from_pretrained()
    return _psyche


def _analyze_sync(prompt):
    """Run analysis (expensive). Called in background thread."""
    psyche = _get_psyche()
    analysis = psyche.analyze(prompt)
    # Force computation of core properties
    _ = analysis.base_words
    _ = analysis.ego_words
    _ = analysis.superego_words
    _ = analysis.repression
    _ = analysis.formation_df
    return analysis


def _ensure_analysis(prompt):
    """Get or compute analysis, blocking until ready."""
    if prompt in _cache:
        return _cache[prompt]

    if prompt in _computing:
        _computing[prompt].wait()
        return _cache.get(prompt)

    event = threading.Event()
    _computing[prompt] = event
    try:
        analysis = _analyze_sync(prompt)
        _cache[prompt] = analysis
    finally:
        event.set()
        _computing.pop(prompt, None)

    return analysis


def _submit_background(prompt):
    """Start background compute, return immediately."""
    if prompt in _cache or prompt in _computing:
        return
    thread = threading.Thread(target=_ensure_analysis, args=(prompt,), daemon=True)
    thread.start()


def _capture_print(fn, *args, **kwargs):
    """Capture stdout from a function that prints."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


# ── Gradio callbacks ──────────────────────────────────────────────

def on_analyze(prompt):
    """Main analysis callback. Returns all outputs."""
    if not prompt or not prompt.strip():
        yield (
            "Enter a prompt above.",
            None, None, None, None, "", "",
        )
        return

    prompt = prompt.strip()

    # Show loading state
    yield (
        f"Computing analysis for: **{prompt}**\n\nThis runs ~600 forward passes across 3 models. First run takes a few minutes; cached results are instant.",
        None, None, None, None, "", "",
    )

    try:
        analysis = _ensure_analysis(prompt)

        # Formation report (captured from stdout)
        report_text = _capture_print(analysis.formation_report)

        # Formation DataFrame
        formation_df = analysis.formation_df.copy()

        # Repression DataFrame
        rep_df = analysis.repression.copy()
        rep_df = rep_df[["word", "base", "ego", "superego", "delta", "repressed", "amplified"]]

        # Trajectory plot
        from .viz import plot_formation_trajectories
        traj_fig = plot_formation_trajectories(analysis, min_prob=0.003, top_n=80)

        # Status
        status = f"Analysis complete for: **{prompt}**"

        yield (
            status,
            formation_df,
            rep_df,
            traj_fig,
            None,  # displacement plot placeholder
            report_text,
            "",  # displacement pairs placeholder
        )

    except Exception as e:
        yield (
            f"Error: {e}\n\n```\n{traceback.format_exc()}\n```",
            None, None, None, None, "", "",
        )


def on_displacement(prompt):
    """Compute displacement map (heavier, separate button)."""
    if not prompt or not prompt.strip():
        return None, ""

    prompt = prompt.strip()
    if prompt not in _cache:
        return None, "Run analysis first."

    try:
        analysis = _cache[prompt]
        dm = analysis.displacement_map()

        # Format pairs
        lines = []
        lines.append("## Sublimation pairs (base → ego)")
        lines.append("")
        for src, tgt, sim, layer in (dm.get("sublimation", {}).get("pairs", []))[:25]:
            lines.append(f"  {src:15s} → {tgt:15s}  sim={sim:.4f}  layer={layer}")

        lines.append("")
        lines.append("## Repression pairs (ego → superego)")
        lines.append("")
        for src, tgt, sim, layer in (dm.get("repression", {}).get("pairs", []))[:25]:
            lines.append(f"  {src:15s} → {tgt:15s}  sim={sim:.4f}  layer={layer}")

        pairs_text = "\n".join(lines)

        # Displacement plot
        from .viz import plot_displacement
        disp_fig = plot_displacement(dm, prompt)

        return disp_fig, pairs_text

    except Exception as e:
        return None, f"Error: {e}\n\n```\n{traceback.format_exc()}\n```"


def on_queue_prompts(prompts_text):
    """Queue multiple prompts for background computation."""
    if not prompts_text or not prompts_text.strip():
        return "No prompts to queue."

    prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]
    queued = []
    cached = []
    for p in prompts:
        if p in _cache:
            cached.append(p)
        else:
            _submit_background(p)
            queued.append(p)

    parts = []
    if queued:
        parts.append(f"Queued {len(queued)} prompts for background computation.")
    if cached:
        parts.append(f"{len(cached)} prompts already cached.")
    return " ".join(parts)


def on_check_cache():
    """Report what's cached."""
    if not _cache:
        return "Cache empty. Run an analysis or queue prompts."

    computing = list(_computing.keys())
    lines = [f"**{len(_cache)} prompts cached:**"]
    for p in _cache:
        lines.append(f"- {p}")
    if computing:
        lines.append(f"\n**{len(computing)} computing:**")
        for p in computing:
            lines.append(f"- {p}")
    return "\n".join(lines)


def on_select_cached(prompt):
    """Select a cached prompt from dropdown."""
    return prompt


def get_cached_prompts():
    """Return list of cached prompts for dropdown."""
    return list(_cache.keys())


# ── Gradio UI ─────────────────────────────────────────────────────

def build_app():
    """Build and return the Gradio app."""
    from . import DEFAULT_PROMPTS

    with gr.Blocks(
        title="malign-logits",
        theme=gr.themes.Base(),
    ) as app:
        gr.Markdown(
            "# malign-logits\n"
            "Psychoanalytic analysis of LLM probability distributions. "
            "Traces displacement, condensation, sublimation, and repression "
            "across the alignment pipeline (base → SFT → DPO)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="She was so angry she wanted to",
                    lines=2,
                )
                with gr.Row():
                    analyze_btn = gr.Button("Analyze", variant="primary")
                    displacement_btn = gr.Button("Displacement map")

            with gr.Column(scale=1):
                status_md = gr.Markdown("Enter a prompt and click Analyze.")
                cache_md = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("Formation report"):
                report_text = gr.Code(
                    label="Formation report (base → SFT → DPO)",
                    language=None,
                    lines=30,
                )

            with gr.Tab("Trajectory plot"):
                trajectory_plot = gr.Plot(label="Formation trajectories")

            with gr.Tab("Repression table"):
                repression_df = gr.Dataframe(
                    label="Ego → Superego (repressed and amplified words)",
                    wrap=True,
                )

            with gr.Tab("Formation table"):
                formation_df = gr.Dataframe(
                    label="All layers scored over same vocabulary",
                    wrap=True,
                )

            with gr.Tab("Displacement map"):
                displacement_plot = gr.Plot(label="Displacement map")
                displacement_pairs = gr.Code(
                    label="Displacement pairs (contextual embedding similarity)",
                    language=None,
                    lines=30,
                )

            with gr.Tab("Batch / queue"):
                gr.Markdown(
                    "Queue multiple prompts for background computation. "
                    "One prompt per line. Results are cached for instant access."
                )
                queue_text = gr.Textbox(
                    label="Prompts (one per line)",
                    lines=10,
                    value="\n".join(DEFAULT_PROMPTS.values()),
                )
                with gr.Row():
                    queue_btn = gr.Button("Queue all")
                    check_btn = gr.Button("Check cache")
                queue_status = gr.Markdown("")

        # Wire up callbacks
        analyze_btn.click(
            fn=on_analyze,
            inputs=[prompt_input],
            outputs=[
                status_md,
                formation_df,
                repression_df,
                trajectory_plot,
                displacement_plot,
                report_text,
                displacement_pairs,
            ],
        )

        displacement_btn.click(
            fn=on_displacement,
            inputs=[prompt_input],
            outputs=[displacement_plot, displacement_pairs],
        )

        queue_btn.click(
            fn=on_queue_prompts,
            inputs=[queue_text],
            outputs=[queue_status],
        )

        check_btn.click(
            fn=on_check_cache,
            inputs=[],
            outputs=[queue_status],
        )

    return app


def launch(**kwargs):
    """Build and launch the Gradio app."""
    app = build_app()
    app.launch(**kwargs)


if __name__ == "__main__":
    launch()
