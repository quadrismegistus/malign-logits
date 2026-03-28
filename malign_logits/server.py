"""
Model server — keeps models loaded in a separate process.

Start:
    malign serve
    # or
    python -m malign_logits.server

Then connect from Psyche:
    psyche = Psyche.from_server()

Or from the Gradio app:
    malign ui --server
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Models loaded once at startup
_psyche = None


def _get_psyche():
    global _psyche
    if _psyche is None:
        from .psyche import Psyche
        print("Loading models...")
        t0 = time.time()
        _psyche = Psyche.from_pretrained()
        print(f"Models loaded in {time.time() - t0:.1f}s")
    return _psyche


class ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        try:
            result = self._dispatch(body)
            self._respond(200, result)
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "models_loaded": _psyche is not None})
        else:
            self._respond(404, {"error": "not found"})

    def _dispatch(self, body):
        path = self.path
        psyche = _get_psyche()

        if path == "/top_words":
            layer_name = body["layer"]
            prompt = body["prompt"]
            top_k = body.get("top_k", 200)
            layer = {"base": psyche.primary_process, "ego": psyche.ego,
                     "superego": psyche.superego}.get(layer_name)
            if layer is None and psyche.reinforced_superego and layer_name == "instruct":
                layer = psyche.reinforced_superego
            if layer is None:
                raise ValueError(f"Unknown layer: {layer_name}")
            return {"words": layer.top_words(prompt, top_k_first=top_k)}

        elif path == "/score_vocabulary":
            layer_name = body["layer"]
            prompt = body["prompt"]
            words = body["words"]
            layer = {"base": psyche.primary_process, "ego": psyche.ego,
                     "superego": psyche.superego}.get(layer_name)
            if layer is None and psyche.reinforced_superego and layer_name == "instruct":
                layer = psyche.reinforced_superego
            if layer is None:
                raise ValueError(f"Unknown layer: {layer_name}")
            return {"words": layer.score_vocabulary(prompt, words)}

        elif path == "/logits":
            layer_name = body["layer"]
            prompt = body["prompt"]
            layer = {"base": psyche.primary_process, "ego": psyche.ego,
                     "superego": psyche.superego}.get(layer_name)
            if layer is None:
                raise ValueError(f"Unknown layer: {layer_name}")
            import torch
            logits = layer.logits(prompt)
            return {"logits": logits.tolist()}

        elif path == "/embedding":
            prompt = body["prompt"]
            word = body["word"]
            hidden_layer = body["layer_idx"]
            model = psyche.ego.model
            tokenizer = psyche.ego.tokenizer
            device = psyche.ego.device
            import torch
            text = prompt + " " + word
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            prompt_len = len(tokenizer.encode(prompt))
            with torch.no_grad():
                outputs = model(ids, output_hidden_states=True)
                hidden = outputs.hidden_states[hidden_layer]
                word_hidden = hidden[0, prompt_len:, :].mean(dim=0).cpu()
            emb = torch.nn.functional.normalize(
                word_hidden.float().unsqueeze(0), dim=-1,
            ).squeeze()
            return {"embedding": emb.tolist()}

        elif path == "/displacement_map":
            prompt = body["prompt"]
            layers = body.get("layers", None)
            analysis = psyche.analyze(prompt)
            # Force word distributions (writes to stash)
            _ = analysis.base_words
            _ = analysis.ego_words
            _ = analysis.superego_words
            _ = analysis.formation_df
            # Run displacement_map (writes embeddings to stash)
            dm = analysis.displacement_map(layers=layers)
            # Serialize the result — pairs are simple lists, df as records
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

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Quieter logging
        pass


def serve(port=8421):
    """Start the model server."""
    # Load models in background while server starts
    thread = threading.Thread(target=_get_psyche, daemon=True)
    thread.start()

    server = HTTPServer(("127.0.0.1", port), ModelHandler)
    print(f"Model server running on http://127.0.0.1:{port}")
    print(f"Models loading in background...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    serve()
