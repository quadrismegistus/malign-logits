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
import math
import mimetypes
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
import threading

_UI_DIR = Path(__file__).parent / "ui_dist"

# Models loaded once at startup
_psyche = None
_psyche_lock = threading.Lock()
_progress = {"stage": "idle", "detail": "", "step": 0, "total": 0}
_progress_lock = threading.Lock()
_family = None  # set by serve()


def _sanitize(obj):
    """Replace NaN/Inf with None recursively so JSON is valid."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _set_progress(stage, detail="", step=0, total=0):
    global _progress
    with _progress_lock:
        _progress = {"stage": stage, "detail": detail, "step": step, "total": total}


def _get_psyche():
    global _psyche
    with _psyche_lock:
        if _psyche is None:
            from .psyche import Psyche
            _set_progress("loading_models", "Loading models...")
            if _family:
                from . import MODEL_FAMILIES
                fam = MODEL_FAMILIES[_family]
                print(f"Loading {fam.name} ({fam.n_layers} layers)...")
            else:
                print("Loading models...")
            t0 = time.time()
            if _family:
                _psyche = Psyche.from_family(_family, load=True)
            else:
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
        elif self.path == "/prompts":
            try:
                result = self._dispatch({})
                self._respond(200, result)
            except Exception as e:
                self._respond(200, {"prompts": []})
        else:
            self._serve_static()

    def _serve_static(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        file_path = _UI_DIR / path.lstrip("/")
        if not file_path.is_file():
            file_path = _UI_DIR / "index.html"
        if not file_path.is_file():
            self._respond(404, {"error": "not found"})
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

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
                ("base", "BASE"),
            ]
            if psyche.ego is not None:
                layers_to_run.append(("sft", "SFT"))
            if psyche.superego is not None:
                layers_to_run.append(("dpo", "DPO"))
            if psyche.reinforced_superego is not None:
                layers_to_run.append(("rlvr", "RLVR"))

            n_layers = len(layers_to_run)
            total_steps = top_k * n_layers

            results = {}
            for i, (name, desc) in enumerate(layers_to_run):
                base_step = i * top_k
                def _progress(step, total, _desc=desc, _base=base_step, _tk=top_k):
                    _set_progress("analyzing",
                                  f"{_desc} ({step}/{_tk} passes)",
                                  step=_base + step, total=total_steps)
                _set_progress("analyzing",
                              f"{desc} (0/{top_k} passes)",
                              step=base_step, total=total_steps)
                layer = self._get_layer(psyche, name)
                results[name] = layer.top_words(
                    prompt, top_k_first=top_k, progress_callback=_progress)

            _set_progress("analyzing", "Caching logits",
                          step=total_steps, total=total_steps)
            for name, _ in layers_to_run:
                layer = self._get_layer(psyche, name)
                _ = layer.logits(prompt)

            _set_progress("analyzing", "Scoring vocabulary",
                          step=total_steps, total=total_steps)
            _ = analysis.focused_base_words
            if psyche.ego is not None:
                _ = analysis.focused_ego_words
            if psyche.superego is not None:
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

            formation_df = analysis.formation_df.copy()
            # Fill NaN in numeric columns (JSON can't serialize NaN)
            num_cols = formation_df.select_dtypes(include="number").columns
            formation_df[num_cols] = formation_df[num_cols].fillna(0)
            formation_df["trajectory"] = formation_df["trajectory"].fillna("flat")

            rep_df = analysis.repression.copy()
            num_cols = rep_df.select_dtypes(include="number").columns
            rep_df[num_cols] = rep_df[num_cols].fillna(0)

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

        elif path == "/perplexity":
            layer_name = body["layer"]
            prompt = body["prompt"]
            layer = self._get_layer(psyche, layer_name)
            return {"perplexity": layer.perplexity(prompt)}

        elif path == "/displacement_map":
            prompt = body["prompt"]
            layers = body.get("layers", None)
            n_layers = len(layers) if layers else 3

            _set_progress("displacement", f"Computing displacement map ({n_layers} layers)...")

            analysis = psyche.analyze(prompt)
            _ = analysis.base_words
            if psyche.ego is not None:
                _ = analysis.ego_words
            if psyche.superego is not None:
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

        elif path == "/prompts":
            # Return all prompts that have been analyzed (in stash)
            prompts = set()
            stash = psyche._stash
            if stash is not None:
                try:
                    for key in stash.keys():
                        if isinstance(key, tuple) and len(key) >= 3 and key[0] == "top_words":
                            p = key[2]
                            if isinstance(p, str):
                                prompts.add(p)
                except Exception:
                    pass
            return {"prompts": sorted(prompts)}

        elif path == "/logit_lens":
            prompt = body["prompt"]
            _set_progress("logit_lens", "Computing logit lens...")
            analysis = psyche.analyze(prompt)
            def _ll_progress(detail, step, total):
                _set_progress("logit_lens", detail, step=step, total=total)
            data = analysis.compute_logit_lens(progress_callback=_ll_progress)
            _set_progress("idle")
            return data

        elif path == "/generate":
            prompt = body["prompt"]
            n = body.get("n", 5)
            max_tokens = body.get("max_tokens", 100)
            temperature = body.get("temperature", 1.0)

            _set_progress("generating", "Generating...", step=0, total=n)

            from .embedding import generate_many_with_progress

            MODEL_LABELS = {"base": "base", "ego": "sft",
                            "superego": "dpo", "instruct": "rlvr"}

            def _gen_progress(done, total):
                _set_progress("generating", f"Generation {done}/{total}",
                              step=done, total=total)

            psg_df = generate_many_with_progress(
                psyche, prompt, n=n, max_new_tokens=max_tokens,
                temperature=temperature, progress_callback=_gen_progress,
            )

            generations = []
            gen_counter = {}
            for _, row in psg_df.iterrows():
                model = row["model"]
                gen_counter.setdefault(model, 0)
                generations.append({
                    "model": MODEL_LABELS.get(model, model),
                    "text": row["psg"],
                    "gen_id": gen_counter[model],
                })
                gen_counter[model] += 1

            _set_progress("generating", "Embedding...",
                          step=n, total=n)

            import pandas as pd
            import numpy as np
            from .embedding import (
                _get_embedder, concept_seeds, concept_vector, score_concept,
            )

            embedder = _get_embedder()
            texts = [g["text"] for g in generations]
            vecs = embedder.encode(texts)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, len(vecs)))
            coords = pca.fit_transform(vecs)

            seeds = concept_seeds()
            concept_scores = {}
            for name, seed_pair in seeds.items():
                axis, midpoint = concept_vector(
                    embedder, seed_pair["positive"], seed_pair["negative"],
                )
                scores = score_concept(vecs, axis, midpoint)
                concept_scores[name] = [round(float(s), 4) for s in scores]

            for i, g in enumerate(generations):
                g["pca_x"] = round(float(coords[i, 0]), 4)
                g["pca_y"] = round(float(coords[i, 1] if coords.shape[1] > 1 else 0), 4)
                for name in concept_scores:
                    g[name] = concept_scores[name][i]

            _set_progress("idle")
            return {
                "generations": generations,
                "concept_axes": list(seeds.keys()),
                "pca_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            }

        elif path == "/contradiction":
            pairs = body.get("pairs", None)
            _set_progress("contradiction", "Analyzing contradictions...")
            def _ct_progress(detail, step, total):
                _set_progress("contradiction", detail, step=step, total=total)
            results = psyche.contradiction_analysis(
                pairs=pairs, progress_callback=_ct_progress)
            _set_progress("idle")
            return {"results": results}

        elif path == "/passage-metrics":
            text = body.get("text", "")
            if not text.strip():
                raise ValueError("No text provided")
            import pandas as pd
            from .embedding import compute_passage_metrics
            psg_df = pd.DataFrame([{
                "model": "custom", "psg": text.strip(),
                "family": "custom", "label": "custom",
            }])
            result = compute_passage_metrics(psg_df, min_sentences=2)
            if result.empty:
                raise ValueError("Text too short or degenerate")
            row = result.iloc[0].to_dict()
            row.pop("psg", None)
            return row

        elif path == "/passage-tokens":
            psg = body.get("psg", "").rstrip()
            prompt_prefix = body.get("prompt", "").strip()
            if not psg:
                raise ValueError("No psg provided")
            from hashstash import HashStash
            from . import PATH_STASH
            stash = HashStash(
                root_dir=PATH_STASH + "_gen_metrics",
                engine="pairtree", compress="lz4", b64=True,
            )
            ts_key = ("token_surprisals_v3", "gpt2", prompt_prefix, psg)
            tok_surps = stash.get(ts_key) if ts_key in stash else None
            if tok_surps is None:
                from .embedding import passage_surprisal
                s = passage_surprisal(psg, prompt_prefix=prompt_prefix)
                tok_surps = s["token_surprisals"]
            return {"tokens": tok_surps}

        elif path == "/passage-metrics-csv":
            import os
            base = os.path.dirname(os.path.dirname(__file__))
            for name in ["corpus_metrics.parquet", "corpus_metrics.csv"]:
                fpath = os.path.join(base, "data", name)
                if os.path.exists(fpath):
                    break
            else:
                return {"rows": []}
            import pandas as pd
            if fpath.endswith(".parquet"):
                df = pd.read_parquet(fpath)
            else:
                df = pd.read_csv(fpath)
            # Drop heavy/unused columns (psg loaded via /passage-tokens on click)
            drop = ['psg'] + [c for c in df.columns if c.startswith(('max_', 'std_',
                              'n_tokens', 'token_path', 'path_length'))]
            df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
            return {"rows": _sanitize(df.to_dict(orient="records"))}

        elif path == "/info":
            info = {
                "base": psyche.primary_process.model_id,
                "n_layers": psyche.n_layers,
            }
            if psyche.ego is not None:
                info["sft"] = psyche.ego.model_id
            if psyche.superego is not None:
                info["dpo"] = psyche.superego.model_id
            if psyche.reinforced_superego is not None:
                info["rlvr"] = psyche.reinforced_superego.model_id
            return info

        else:
            raise ValueError(f"Unknown endpoint: {path}")

    def _get_layer(self, psyche, layer_name):
        mapping = {
            "base": psyche.primary_process,
            "sft": psyche.ego,
            "dpo": psyche.superego,
            "rlvr": psyche.reinforced_superego,
        }
        layer = mapping.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer not available: {layer_name}")
        return layer

    def do_OPTIONS(self):
        self._cors_headers()
        self.send_response(204)
        self.end_headers()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(_sanitize(data)).encode())

    def log_message(self, format, *args):
        pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def serve(port=8421, family=None, host="0.0.0.0"):
    """Start the model server."""
    global _family
    _family = family
    thread = threading.Thread(target=_get_psyche, daemon=True)
    thread.start()

    server = ThreadingHTTPServer((host, port), ModelHandler)
    print(f"Model server running on http://{host}:{port}")
    print(f"Models loading in background...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    serve()
