"""
Model server — keeps models loaded in a separate process.

Start:
    malign serve
    # or
    python -m malign_logits.server

Then connect from Psyche:
    psyche = Psyche.from_server()

Or from the Gradio app:
    malign ui
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading

# Models loaded once at startup
_psyche = None
_progress = {"stage": "idle", "detail": "", "step": 0, "total": 0}
_progress_lock = threading.Lock()


def _set_progress(stage, detail="", step=0, total=0):
    global _progress
    with _progress_lock:
        _progress = {"stage": stage, "detail": detail, "step": step, "total": total}


def _get_psyche():
    global _psyche
    if _psyche is None:
        from .psyche import Psyche
        _set_progress("loading_models", "Loading models...")
        print("Loading models...")
        t0 = time.time()
        _psyche = Psyche.from_pretrained()
        print(f"Models loaded in {time.time() - t0:.1f}s")
        _set_progress("idle")
    return _psyche


class ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        try:
            result = self._dispatch(body)
            self._respond(200, result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._respond(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "models_loaded": _psyche is not None})
        elif self.path == "/info":
            try:
                result = self._dispatch({})
                self._respond(200, result)
            except Exception as e:
                self._respond(500, {"error": str(e)})
        elif self.path == "/progress":
            with _progress_lock:
                self._respond(200, dict(_progress))
        else:
            self._respond(404, {"error": "not found"})

    def _dispatch(self, body):
        path = self.path
        psyche = _get_psyche()

        if path == "/top_words":
            layer_name = body["layer"]
            prompt = body["prompt"]
            top_k = body.get("top_k", 200)
            layer = self._get_layer(psyche, layer_name)
            return {"words": layer.top_words(prompt, top_k_first=top_k)}

        elif path == "/score_vocabulary":
            layer_name = body["layer"]
            prompt = body["prompt"]
            words = body["words"]
            layer = self._get_layer(psyche, layer_name)
            return {"words": layer.score_vocabulary(prompt, words)}

        elif path == "/analyze":
            prompt = body["prompt"]
            top_k = body.get("top_k", 200)
            analysis = psyche.analyze(prompt, top_k_first=top_k)

            layers_to_run = [
                ("base", "Base (primary process)", lambda: analysis.base_words),
                ("ego", "Ego (SFT)", lambda: analysis.ego_words),
                ("superego", "Superego (DPO)", lambda: analysis.superego_words),
            ]
            if psyche.reinforced_superego is not None:
                layers_to_run.append(
                    ("instruct", "Instruct (RLVR)", lambda: analysis.instruct_words),
                )

            results = {}
            for i, (name, desc, fn) in enumerate(layers_to_run):
                _set_progress("analyzing", f"{desc} ({i+1}/{len(layers_to_run)})",
                              step=i, total=len(layers_to_run))
                results[name] = fn()

            # Cache logits for each layer (1 forward pass each, enables fast scoring)
            _set_progress("analyzing", "Caching logits...",
                          step=len(layers_to_run), total=len(layers_to_run) + 3)
            for name, _, _ in layers_to_run:
                layer = self._get_layer(psyche, name)
                _ = layer.logits(prompt)

            # Score focused vocabulary (fast — uses cached logits)
            _set_progress("analyzing", "Scoring focused vocabulary...",
                          step=len(layers_to_run) + 1, total=len(layers_to_run) + 3)
            _ = analysis.focused_base_words
            _ = analysis.focused_ego_words
            _ = analysis.focused_superego_words

            # Build report and DataFrames
            _set_progress("analyzing", "Building report...",
                          step=len(layers_to_run) + 2, total=len(layers_to_run) + 3)

            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                analysis.formation_report()
            report_text = buf.getvalue()

            formation_df = analysis.formation_df
            rep_df = analysis.repression

            _set_progress("idle")
            return {
                "status": "complete",
                "layers": list(results.keys()),
                "report": report_text,
                "formation_df": formation_df.to_dict(orient="records"),
                "repression_df": rep_df.to_dict(orient="records"),
            }

        elif path == "/logits":
            layer_name = body["layer"]
            prompt = body["prompt"]
            layer = self._get_layer(psyche, layer_name)
            import torch
            logits = layer.logits(prompt)
            return {"logits": logits.tolist()}

        elif path == "/displacement_map":
            prompt = body["prompt"]
            layers = body.get("layers", None)
            n_layers = len(layers) if layers else 3

            _set_progress("displacement", f"Computing displacement map ({n_layers} layers)...")

            analysis = psyche.analyze(prompt)
            # Force word distributions
            _ = analysis.base_words
            _ = analysis.ego_words
            _ = analysis.superego_words
            _ = analysis.formation_df

            _set_progress("displacement", f"Computing embeddings across {n_layers} layers...")
            dm = analysis.displacement_map(layers=layers)

            _set_progress("idle")
            result = {
                "sublimation": {
                    "source": dm.get("sublimation", {}).get("source", []),
                    "target": dm.get("sublimation", {}).get("target", []),
                    "pairs": dm.get("sublimation", {}).get("pairs", []),
                },
                "repression": {
                    "source": dm.get("repression", {}).get("source", []),
                    "target": dm.get("repression", {}).get("target", []),
                    "pairs": dm.get("repression", {}).get("pairs", []),
                },
                "df": dm["df"].to_dict(orient="records"),
            }
            return result

        elif path == "/info":
            return {
                "base": psyche.primary_process.model_id,
                "ego": psyche.ego.model_id,
                "superego": psyche.superego.model_id,
                "instruct": psyche.reinforced_superego.model_id if psyche.reinforced_superego else None,
            }

        else:
            raise ValueError(f"Unknown endpoint: {path}")

    def _get_layer(self, psyche, layer_name):
        layer = {"base": psyche.primary_process, "ego": psyche.ego,
                 "superego": psyche.superego}.get(layer_name)
        if layer is None and psyche.reinforced_superego and layer_name == "instruct":
            layer = psyche.reinforced_superego
        if layer is None:
            raise ValueError(f"Unknown layer: {layer_name}")
        return layer

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def serve(port=8421):
    """Start the model server."""
    thread = threading.Thread(target=_get_psyche, daemon=True)
    thread.start()

    server = ThreadingHTTPServer(("127.0.0.1", port), ModelHandler)
    print(f"Model server running on http://127.0.0.1:{port}")
    print(f"Models loading in background...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    serve()
