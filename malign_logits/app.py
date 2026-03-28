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
_formation_cache = {}  # prompt -> formation DataFrame
_dm_cache = {}  # prompt -> displacement_map result (3-layer)
_dm_full_cache = {}  # prompt -> displacement_map result (all layers)
_computing = {}  # prompt -> threading.Event (set when done)


_SERVER_URL = "http://127.0.0.1:8421"


def _server_available():
    """Check if model server is running."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{_SERVER_URL}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _get_psyche():
    global _psyche, _models_loaded, _model_status
    if _psyche is None:
        from .psyche import Psyche
        if _server_available():
            _psyche = Psyche.from_server(server_url=_SERVER_URL)
            _models_loaded = True
            _model_status = "Connected to model server."
            print(f"Connected to model server at {_SERVER_URL}")
        else:
            _psyche = Psyche.from_cache()
            _model_status = "Cache loaded (no server detected, models not in memory)."
            print("No model server detected. Using cache-only mode.")
    return _psyche


def _ensure_models():
    """Load models if not already loaded. Called on cache miss."""
    global _models_loaded, _model_status
    psyche = _get_psyche()
    if not _models_loaded:
        # Check server again in case it started after app launch
        if _server_available():
            from .psyche import Psyche, RemoteModelLayer
            psyche.primary_process = RemoteModelLayer(
                _SERVER_URL, "base", psyche._model_names["base"], name="base",
            )
            psyche.ego = RemoteModelLayer(
                _SERVER_URL, "ego", psyche._model_names["ego"], name="ego",
            )
            psyche.superego = RemoteModelLayer(
                _SERVER_URL, "superego", psyche._model_names["superego"], name="superego",
            )
            psyche._propagate_stash()
            _models_loaded = True
            _model_status = "Connected to model server."
        else:
            _model_status = "Loading models locally..."
            psyche.load_models()
            _models_loaded = True
            _model_status = "Models loaded locally."


def _poll_progress():
    """Poll server for progress status."""
    try:
        import urllib.request
        import json as _json
        with urllib.request.urlopen(f"{_SERVER_URL}/progress", timeout=2) as resp:
            return _json.loads(resp.read())
    except Exception:
        return None


def _server_analyze(prompt):
    """Ask server to run full analysis (all layers). Blocks until done."""
    import urllib.request
    import json as _json
    data = _json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{_SERVER_URL}/analyze",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return _json.loads(resp.read())


def _analyze_sync(prompt):
    """Run analysis (expensive). Called in background thread."""
    psyche = _get_psyche()
    analysis = psyche.analyze(prompt)
    try:
        _ = analysis.base_words
        _ = analysis.ego_words
        _ = analysis.superego_words
        _ = analysis.repression
        _ = analysis.formation_df
    except RuntimeError:
        _ensure_models()
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


def _format_pairs(dm):
    """Format displacement pairs as text."""
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
    return "\n".join(lines)


def _get_sublimation_sources(dm):
    """Get source words from displacement map for dropdown."""
    sub_pairs = dm.get("sublimation", {}).get("pairs", [])
    if not sub_pairs:
        return []
    sub_df = pd.DataFrame(sub_pairs, columns=["source", "target", "sim", "layer"])
    return sub_df.groupby("source")["sim"].max().nlargest(20).index.tolist()


# ── Gradio callbacks ──────────────────────────────────────────────

def on_analyze(prompt, sort_by, top_n, min_prob, min_delta):
    """Main analysis callback. Blocks until complete (no generator)."""
    if not prompt or not prompt.strip():
        return ("Enter a prompt above.", None, None, None, None, "", "")

    prompt = prompt.strip()
    min_delta_val = min_delta if min_delta > 0 else None

    try:
        server_result = None

        if _server_available():
            _ensure_models()
            server_result = _server_analyze(prompt)

        analysis = _ensure_analysis(prompt)
        _cache[prompt] = analysis

        if server_result and "report" in server_result:
            report_text = server_result["report"]
            formation_df = pd.DataFrame(server_result["formation_df"])
            for col in ["base", "ego", "superego", "ego - base", "superego - ego"]:
                if col in formation_df.columns:
                    formation_df[col] = pd.to_numeric(formation_df[col], errors="coerce").fillna(0)
            rep_df = pd.DataFrame(server_result["repression_df"])
            for col in ["base", "ego", "superego", "delta"]:
                if col in rep_df.columns:
                    rep_df[col] = pd.to_numeric(rep_df[col], errors="coerce").fillna(0)
            if "word" in rep_df.columns:
                keep = [c for c in ["word", "base", "ego", "superego", "delta", "repressed", "amplified"] if c in rep_df.columns]
                rep_df = rep_df[keep]
        else:
            report_text = _capture_print(analysis.formation_report)
            formation_df = analysis.formation_df.copy()
            rep_df = analysis.repression.copy()
            rep_df = rep_df[["word", "base", "ego", "superego", "delta", "repressed", "amplified"]]

        _formation_cache[prompt] = formation_df

        from .viz import plot_formation_trajectories
        traj_fig = plot_formation_trajectories(
            formation_df,
            prompt=prompt,
            min_prob=min_prob,
            min_delta=min_delta_val,
            sort_by=sort_by,
            top_n=int(top_n),
        )

        return (
            f"Analysis complete for: **{prompt}**",
            formation_df, rep_df, traj_fig,
            None, report_text, "",
        )

    except Exception as e:
        return (
            f"Error: {e}\n\n```\n{traceback.format_exc()}\n```",
            None, None, None, None, "", "",
        )


def on_replot(prompt, sort_by, top_n, min_prob, min_delta):
    """Replot trajectory with new settings (no recompute)."""
    print(f"[replot] prompt={prompt!r}, sort_by={sort_by}, top_n={top_n}, "
          f"min_prob={min_prob}, min_delta={min_delta}")
    if not prompt or not prompt.strip():
        print("[replot] no prompt")
        return gr.update()
    prompt = prompt.strip()
    if prompt not in _formation_cache:
        print(f"[replot] prompt not in cache. keys={list(_formation_cache.keys())}")
        return gr.update()
    try:
        min_delta_val = min_delta if min_delta > 0 else None
        df = _formation_cache[prompt]
        print(f"[replot] df shape={df.shape}, columns={list(df.columns)}")
        from .viz import plot_formation_trajectories
        fig = plot_formation_trajectories(
            df,
            prompt=prompt,
            min_prob=min_prob,
            min_delta=min_delta_val,
            sort_by=sort_by,
            top_n=int(top_n),
        )
        print(f"[replot] success, fig type={type(fig)}")
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return gr.update()


def _request_server_displacement(prompt, layers=None):
    """Ask server to compute displacement map and return serialized result."""
    import urllib.request
    import json as _json
    body = {"prompt": prompt}
    if layers is not None:
        body["layers"] = layers
    data = _json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_SERVER_URL}/displacement_map",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = _json.loads(resp.read())

    # Reconstruct DataFrame
    result["df"] = pd.DataFrame(result["df"])
    # Pairs come as lists of lists, convert to tuples
    for axis in ("sublimation", "repression"):
        if axis in result:
            result[axis]["pairs"] = [tuple(p) for p in result[axis].get("pairs", [])]
    return result


def on_displacement(prompt):
    """Compute 3-layer displacement map."""
    if not prompt or not prompt.strip():
        return None, ""
    prompt = prompt.strip()
    if prompt not in _cache:
        return None, "Run analysis first."

    try:
        if _server_available():
            dm = _request_server_displacement(prompt)
        else:
            analysis = _cache[prompt]
            _ensure_models()
            dm = analysis.displacement_map()
        _dm_cache[prompt] = dm

        from .viz import plot_displacement
        return plot_displacement(dm, prompt), _format_pairs(dm)

    except Exception as e:
        return None, f"Error: {e}\n\n```\n{traceback.format_exc()}\n```"


def on_layer_displacement(prompt, source_word):
    """Compute all-layer displacement map and plot layer displacement.

    This is heavy: embeddings at all 32 hidden layers for all significant words.
    """
    if not prompt or not prompt.strip():
        return "Enter a prompt and run analysis first.", None, gr.update()
    prompt = prompt.strip()
    if prompt not in _cache:
        return "Run analysis first.", None, gr.update()

    try:
        if prompt not in _dm_full_cache:
            if _server_available():
                dm = _request_server_displacement(prompt, layers=list(range(1, 33)))
            else:
                analysis = _cache[prompt]
                _ensure_models()
                dm = analysis.displacement_map(layers=list(range(1, 33)))
            _dm_full_cache[prompt] = dm
        else:
            dm = _dm_full_cache[prompt]

        sources = _get_sublimation_sources(dm)
        source_update = gr.update(choices=sources, value=source_word or (sources[0] if sources else None))

        from .viz import plot_layer_displacement
        src = source_word if source_word and source_word in sources else (sources[0] if sources else None)
        if src is None:
            return "No sublimation pairs found.", None, source_update

        figs = plot_layer_displacement(dm, prompt, source_word=src)
        fig = figs[0] if figs else None

        return f"Layer displacement for **{src}** → targets", fig, source_update

    except Exception as e:
        return f"Error: {e}\n\n```\n{traceback.format_exc()}\n```", None, gr.update()


def on_replot_layers(prompt, source_word):
    """Replot layer displacement with different source word (no recompute)."""
    if not prompt or not prompt.strip():
        return "Select a prompt.", None
    prompt = prompt.strip()
    if prompt not in _dm_full_cache:
        return "Run layer displacement first.", None
    if not source_word:
        return "Select a source word.", None

    try:
        dm = _dm_full_cache[prompt]
        from .viz import plot_layer_displacement
        figs = plot_layer_displacement(dm, prompt, source_word=source_word)
        fig = figs[0] if figs else None
        return f"Layer displacement for **{source_word}** → targets", fig
    except Exception as e:
        return f"Error: {e}", None


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
            if len(key) >= 4 and key[0] == "top_words":
                prompts.add(key[3])
            elif len(key) >= 5 and key[0] == "analysis":
                prompts.add(key[3])
    except Exception:
        pass
    return prompts


def _server_prompts():
    """Get analyzed prompts from the server."""
    try:
        import urllib.request
        import json as _json
        with urllib.request.urlopen(f"{_SERVER_URL}/prompts", timeout=5) as resp:
            data = _json.loads(resp.read())
            return set(data.get("prompts", []))
    except Exception:
        return set()


def _all_known_prompts():
    """All prompts from in-memory cache + server + disk stash + defaults."""
    from . import DEFAULT_PROMPTS
    prompts = set(_cache.keys())
    if _server_available():
        prompts |= _server_prompts()
    else:
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

    return "\n".join(parts), gr.update(choices=prompts)


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
                with gr.Row():
                    prompt_dropdown = gr.Dropdown(
                        choices=list(DEFAULT_PROMPTS.values()),
                        label="Cached / default prompts",
                        interactive=True,
                        allow_custom_value=True,
                        scale=5,
                    )
                    refresh_btn = gr.Button("↻", scale=1, min_width=40)
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

            with gr.Tab("Layer displacement"):
                gr.Markdown(
                    "Trace how displacement similarity evolves across all 32 "
                    "hidden layers of the SFT model. Shows where in the network "
                    "the model recognises that a sublimated word is semantically "
                    "related to its displacement target.\n\n"
                    "**Heavy compute:** embeds words at every hidden layer. "
                    "First run takes several minutes; results are cached."
                )
                with gr.Row():
                    layer_btn = gr.Button("Compute layer displacement", variant="primary")
                    layer_source = gr.Dropdown(
                        choices=[],
                        label="Source word",
                        info="Sublimated word to trace through layers (auto-populated after compute)",
                        interactive=True,
                    )
                    layer_replot_btn = gr.Button("Replot source")
                layer_status = gr.Markdown("")
                layer_plot = gr.Plot(label="Displacement through hidden layers")

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

        # ── Wire up callbacks ─────────────────────────────────────

        plot_inputs = [prompt_input, sort_by, top_n, min_prob, min_delta]

        def _refresh_dropdown():
            known = _all_known_prompts()
            for p in _cache:
                if p not in known:
                    known.append(p)
            return gr.update(choices=sorted(known))

        analyze_btn.click(
            fn=on_analyze,
            inputs=plot_inputs,
            outputs=[
                status_md, formation_df, repression_df, trajectory_plot,
                displacement_plot, report_text, displacement_pairs,
            ],
        )

        # Refresh dropdown on page load and after analyze
        app.load(fn=_refresh_dropdown, outputs=[prompt_dropdown])

        replot_inputs = [prompt_input, sort_by, top_n, min_prob, min_delta]

        replot_btn.click(
            fn=on_replot,
            inputs=replot_inputs,
            outputs=[trajectory_plot],
            concurrency_limit=None,
        )

        displacement_btn.click(
            fn=on_displacement,
            inputs=[prompt_input],
            outputs=[displacement_plot, displacement_pairs],
        )

        layer_btn.click(
            fn=on_layer_displacement,
            inputs=[prompt_input, layer_source],
            outputs=[layer_status, layer_plot, layer_source],
        )

        layer_replot_btn.click(
            fn=on_replot_layers,
            inputs=[prompt_input, layer_source],
            outputs=[layer_status, layer_plot],
            concurrency_limit=None,
        )

        refresh_btn.click(
            fn=_refresh_dropdown,
            inputs=[],
            outputs=[prompt_dropdown],
            concurrency_limit=None,
        )

        prompt_dropdown.change(
            fn=on_select_prompt,
            inputs=[prompt_dropdown],
            outputs=[prompt_input],
            concurrency_limit=None,
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
