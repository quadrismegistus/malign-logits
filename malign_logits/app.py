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
import time
import traceback
from contextlib import redirect_stdout

import gradio as gr
import pandas as pd

# Lazy globals — populated by launch()
_psyche = None
_models_loaded = False
_model_status = "Models not loaded"
_cache = {}  # prompt -> PromptAnalysis
_computing = {}  # prompt -> threading.Event (set when done)


def _get_psyche():
    global _psyche, _models_loaded, _model_status
    if _psyche is None:
        _model_status = "Loading models... (base)"
        from .psyche import Psyche
        _psyche = Psyche.from_pretrained()
        _models_loaded = True
        _model_status = "Models loaded."
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

def on_analyze(prompt, sort_by, top_n, min_prob, min_delta):
    """Main analysis callback. Returns all outputs."""
    empty = ("Enter a prompt above.", None, None, None, None, "", "", gr.update())
    if not prompt or not prompt.strip():
        yield empty
        return

    prompt = prompt.strip()
    min_delta_val = min_delta if min_delta > 0 else None

    loading_msg = (
        f"Computing analysis for: **{prompt}**\n\n"
        f"{'**Loading models first...** (~90s) ' if not _models_loaded else ''}"
        f"This runs ~600 forward passes across 3 models. "
        f"First run takes a few minutes; cached results are instant."
    )
    yield (loading_msg, None, None, None, None, "", "", gr.update())

    try:
        analysis = _ensure_analysis(prompt)

        # Formation report (captured from stdout)
        report_text = _capture_print(analysis.formation_report)

        # Formation DataFrame
        formation_df = analysis.formation_df.copy()

        # Repression DataFrame
        rep_df = analysis.repression.copy()
        rep_df = rep_df[["word", "base", "ego", "superego", "delta", "repressed", "amplified"]]

        # Trajectory plot with user controls
        from .viz import plot_formation_trajectories
        traj_fig = plot_formation_trajectories(
            analysis,
            min_prob=min_prob,
            min_delta=min_delta_val,
            sort_by=sort_by,
            top_n=int(top_n),
        )

        status = f"Analysis complete for: **{prompt}**"
        dropdown_update = gr.update(choices=_all_known_prompts())

        yield (
            status,
            formation_df,
            rep_df,
            traj_fig,
            None,
            report_text,
            "",
            dropdown_update,
        )

    except Exception as e:
        yield (
            f"Error: {e}\n\n```\n{traceback.format_exc()}\n```",
            None, None, None, None, "", "", gr.update(),
        )


def on_replot(prompt, sort_by, top_n, min_prob, min_delta):
    """Replot trajectory with new settings (no recompute)."""
    if not prompt or not prompt.strip() or prompt.strip() not in _cache:
        return None

    prompt = prompt.strip()
    analysis = _cache[prompt]
    min_delta_val = min_delta if min_delta > 0 else None

    from .viz import plot_formation_trajectories
    return plot_formation_trajectories(
        analysis,
        min_prob=min_prob,
        min_delta=min_delta_val,
        sort_by=sort_by,
        top_n=int(top_n),
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


def _stash_prompts():
    """Extract unique prompts from the HashStash disk cache."""
    prompts = set()
    psyche = _psyche
    if psyche is None or psyche._stash is None:
        return prompts
    try:
        for key in psyche._stash.keys():
            if not isinstance(key, tuple):
                continue
            # Cache keys are tuples like ("top_words", model_id, name, prompt, top_k)
            # or ("analysis", key, fingerprint, prompt, top_k)
            if len(key) >= 4 and key[0] == "top_words":
                prompts.add(key[3])  # prompt is 4th element
            elif len(key) >= 5 and key[0] == "analysis":
                prompts.add(key[3])  # prompt is 4th element
    except Exception:
        pass
    return prompts


def _all_known_prompts():
    """All prompts from in-memory cache + disk stash + defaults."""
    from . import DEFAULT_PROMPTS
    prompts = set(_cache.keys())
    prompts |= _stash_prompts()
    prompts |= set(DEFAULT_PROMPTS.values())
    return sorted(p for p in prompts if isinstance(p, str))


def on_check_cache():
    """Report what's cached and update dropdown."""
    parts = [f"**Status:** {_model_status}"]
    prompts = _all_known_prompts()

    memory_cached = list(_cache.keys())
    disk_cached = _stash_prompts() - set(memory_cached)
    computing = list(_computing.keys())

    if memory_cached:
        parts.append(f"\n**{len(memory_cached)} in memory:**")
        for p in memory_cached:
            parts.append(f"- {p}")
    if disk_cached:
        parts.append(f"\n**{len(disk_cached)} on disk (cached from previous sessions):**")
        for p in sorted(disk_cached):
            parts.append(f"- {p}")
    if computing:
        parts.append(f"\n**{len(computing)} computing:**")
        for p in computing:
            parts.append(f"- {p}")
    if not memory_cached and not disk_cached and not computing:
        parts.append("\nNo cached prompts yet.")

    status_text = "\n".join(parts)
    return status_text, gr.update(choices=prompts)


def on_select_prompt(prompt):
    """When user selects a prompt from dropdown, populate the input."""
    return prompt


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
                prompt_dropdown = gr.Dropdown(
                    choices=list(DEFAULT_PROMPTS.values()),
                    label="Cached / default prompts",
                    interactive=True,
                    allow_custom_value=True,
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="She was so angry she wanted to",
                    lines=2,
                )
                with gr.Row():
                    analyze_btn = gr.Button("Analyze", variant="primary")
                    displacement_btn = gr.Button("Displacement map")

            with gr.Column(scale=1):
                status_md = gr.Markdown(
                    "Models load on first analysis (~90s)."
                    if not _models_loaded
                    else "Models loaded. Enter a prompt."
                )

        with gr.Tabs():
            with gr.Tab("Formation report"):
                report_text = gr.Code(
                    label="Formation report (base → SFT → DPO)",
                    language=None,
                    lines=30,
                )

            with gr.Tab("Trajectory plot"):
                with gr.Row():
                    sort_by = gr.Radio(
                        choices=["delta", "mass"],
                        value="delta",
                        label="Sort by",
                        info="delta = biggest movers; mass = highest total probability",
                    )
                    top_n = gr.Slider(
                        minimum=10, maximum=200, value=60, step=10,
                        label="Top N words",
                    )
                    min_prob = gr.Slider(
                        minimum=0.0, maximum=0.05, value=0.001, step=0.001,
                        label="Min probability",
                    )
                    min_delta = gr.Slider(
                        minimum=0.0, maximum=0.05, value=0.003, step=0.001,
                        label="Min delta (0 = off)",
                        info="Include words with large movement even if low probability",
                    )
                replot_btn = gr.Button("Replot")
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
        plot_inputs = [prompt_input, sort_by, top_n, min_prob, min_delta]

        analyze_btn.click(
            fn=on_analyze,
            inputs=plot_inputs,
            outputs=[
                status_md,
                formation_df,
                repression_df,
                trajectory_plot,
                displacement_plot,
                report_text,
                displacement_pairs,
                prompt_dropdown,
            ],
        )

        replot_btn.click(
            fn=on_replot,
            inputs=plot_inputs,
            outputs=[trajectory_plot],
        )

        displacement_btn.click(
            fn=on_displacement,
            inputs=[prompt_input],
            outputs=[displacement_plot, displacement_pairs],
        )

        prompt_dropdown.change(
            fn=on_select_prompt,
            inputs=[prompt_dropdown],
            outputs=[prompt_input],
        )

        queue_btn.click(
            fn=on_queue_prompts,
            inputs=[queue_text],
            outputs=[queue_status],
        )

        check_btn.click(
            fn=on_check_cache,
            inputs=[],
            outputs=[queue_status, prompt_dropdown],
        )

    return app


def launch(**kwargs):
    """Build and launch the Gradio app."""
    app = build_app()
    app.launch(**kwargs)


if __name__ == "__main__":
    launch()
