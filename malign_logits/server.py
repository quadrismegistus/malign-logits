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
            self._respond(200, {
                "status": "ok",
                "models_loaded": _psyche is not None,
                "data_only": _psyche is None,
            })
        elif self.path == "/info":
            if _psyche is not None:
                try:
                    result = self._dispatch({})
                    self._respond(200, result)
                except Exception as e:
                    self._respond(500, {"error": str(e)})
            else:
                self._respond(200, {
                    "base": _family or "data-only",
                    "n_layers": 0,
                    "data_only": True,
                })
        elif self.path == "/progress":
            with _progress_lock:
                self._respond(200, dict(_progress))
        elif self.path == "/prompts":
            try:
                result = self._dispatch({})
                self._respond(200, result)
            except Exception as e:
                self._respond(200, {"prompts": []})
        elif self.path.startswith("/api/"):
            self._handle_probe_api()
        else:
            self._serve_static()

    def _handle_probe_api(self):
        """Probe API: on-demand computation from cached data."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        endpoint = parsed.path

        try:
            if endpoint == "/api/tree":
                from .probe import Probe
                from .metrics import tree_metrics
                p = Probe(params["model"])
                result = tree_metrics(p, params.get("prompt", "She was so angry she wanted to"),
                                      n_gens=int(params.get("n", 100)),
                                      max_tokens=int(params.get("T", 10)))
                self._respond(200, result)

            elif endpoint == "/api/tree_compare":
                from .probe import Probe
                from .metrics import tree_compare
                result = tree_compare(
                    Probe(params["base"]), Probe(params["aligned"]),
                    params.get("prompt", "She was so angry she wanted to"),
                    n_gens=int(params.get("n", 100)),
                    max_tokens=int(params.get("T", 10)))
                self._respond(200, result)

            elif endpoint == "/api/tree_sankey":
                from .probe import Probe, _resolve_prompt
                from scripts.export_tree_sankey import build_sankey_tree
                result = build_sankey_tree(
                    params["model"], params.get("prompt", "anger"),
                    n_gens=int(params.get("n", 100)),
                    max_tokens=int(params.get("T", 10)))
                self._respond(200, result)

            elif endpoint == "/api/distribution":
                from .probe import Probe, _resolve_prompt
                from scipy.special import softmax
                p = Probe(params["model"])
                prompt = _resolve_prompt(params.get("prompt", "anger"))
                logits = p.logits(prompt, gen=0,
                                  pos=int(params.get("pos", 0)),
                                  max_tokens=int(params.get("T", 10)))
                probs = softmax(logits)
                tok = p.tokenizer
                k = int(params.get("k", 30))
                top_idx = probs.argsort()[-k:][::-1]
                result = [{"token": tok.decode([int(i)]).strip(),
                           "prob": float(probs[i]),
                           "token_id": int(i)}
                          for i in top_idx]
                self._respond(200, {"tokens": result, "model": params["model"]})

            elif endpoint == "/api/trajectory":
                from .probe import Probe, _resolve_prompt
                from .metrics import axis_trajectory, violence_procedural_axes
                import numpy as np
                p = Probe(params["model"])
                prompt = _resolve_prompt(params.get("prompt", "anger"))
                embed = p.embedding_matrix()
                v_ax, p_ax = violence_procedural_axes(embed, p.tokenizer)
                axis = params.get("axis", "violence")
                ax = v_ax if axis == "violence" else p_ax
                df = axis_trajectory(p, prompt, embed, ax, axis_name=axis)
                self._respond(200, df.to_dict(orient="records"))

            elif endpoint == "/api/census":
                from .probe import Probe
                prompt = params.get("prompt", "anger")
                df = Probe.census(prompt, max_tokens=int(params.get("T", 10)))
                self._respond(200, df.to_dict(orient="records"))

            elif endpoint == "/api/census_compare":
                from .probe import Probe
                prompt = params.get("prompt", "anger")
                df = Probe.census_compare(prompt, max_tokens=int(params.get("T", 10)))
                self._respond(200, df.to_dict(orient="records"))

            elif endpoint == "/api/families":
                from .probe import Probe
                df = Probe.families()
                self._respond(200, df.to_dict(orient="records"))

            elif endpoint == "/api/beam/index":
                from .cache import get_cache
                from .registry import Registry
                from collections import defaultdict
                cm = get_cache()
                reg = Registry()
                models = set()
                prompts = set()
                sources_by_model = defaultdict(set)
                for k in cm.iter_beam_keys():
                    m = k.get("model", "")
                    models.add(m)
                    prompts.add(k.get("prompt", ""))
                    sources_by_model[m].add(k.get("source", ""))
                sources_map = {m: sorted(s) for m, s in sources_by_model.items()}
                nicks = {m: reg.nickname(m) for m in models}
                # Build source nickname lookup (truncated name → nickname)
                from .registry import NICKNAMES
                _src_nick = {}
                for full_id, nick in NICKNAMES.items():
                    short = full_id.split("/")[-1]
                    full_trunc = short.replace("-", "_")
                    legacy_trunc = full_trunc[:20]
                    _src_nick[full_trunc] = nick
                    _src_nick[short] = nick
                    if legacy_trunc not in _src_nick:
                        _src_nick[legacy_trunc] = nick
                    elif _src_nick[legacy_trunc] != nick:
                        _src_nick[legacy_trunc] = legacy_trunc
                self._respond(200, {"models": sorted(models), "prompts": sorted(prompts), "sources": sources_map, "nicknames": nicks, "source_nicknames": _src_nick})

            elif endpoint == "/api/beam/storylines":
                from .cache import get_cache
                cm = get_cache()
                model = params.get("model", "")
                prompt = params.get("prompt", "")
                source = params.get("source", "")
                top_n = int(params.get("n", 20))
                data = None
                for k in cm.iter_beam_keys():
                    if k.get("model") != model or k.get("prompt") != prompt:
                        continue
                    t = k.get("type", "")
                    if t == "beam_cross_v1":
                        if source and k.get("source", "") != source:
                            continue
                        data = cm.get_beams(k)
                        break
                    elif t == "beam_annotated_v1" and not source:
                        data = cm.get_beams(k)
                        break
                if data is None:
                    self._respond(200, {"storylines": [], "error": "not found"})
                    return
                storylines = []
                for i, s in enumerate(data[:top_n]):
                    entry = {
                        "rank": i,
                        "text": s.get("text", ""),
                        "tokens": s.get("token_texts", []),
                        "path_prob": s.get("path_prob", 0),
                        "log_prob": s.get("log_prob", 0),
                    }
                    if "annotations" in s:
                        annots = {}
                        for ann_name, ann in s["annotations"].items():
                            annots[ann_name] = {
                                "token_resist": ann.get("token_resist", []),
                                "total_resist": ann.get("total_resist", 0),
                                "mean_resist": ann.get("mean_resist", 0),
                            }
                        entry["annotations"] = annots
                    if "base_token_probs" in s:
                        entry["base_token_probs"] = s["base_token_probs"]
                    storylines.append(entry)
                self._respond(200, {"storylines": storylines, "model": model, "prompt": prompt})

            elif endpoint == "/api/beam/sankey":
                from .cache import get_cache
                from collections import Counter
                cm = get_cache()
                model = params.get("model", "")
                prompt = params.get("prompt", "")
                depth = int(params.get("depth", 1))
                mode = params.get("mode", "beam")
                source_data = {}

                if mode in ("logit", "word"):
                    from .registry import Registry
                    from transformers import AutoTokenizer
                    import numpy as np
                    reg = Registry()
                    base_id = model
                    logits_stash = cm._stash("logits")
                    family_models = [base_id] + list(reg.variants_of(base_id))
                    for mid in family_models:
                        short = mid.split("/")[-1].replace("-", "_")
                        # Prefer word_probs (hybrid: exact logit + beam)
                        wp_data = cm.get_word_probs(mid, prompt)
                        if wp_data and isinstance(wp_data, dict):
                            top = sorted(wp_data.items(), key=lambda x: -x[1])[:15]
                            source_data[short] = {w: round(p * 100, 1) for w, p in top}
                            continue
                        # Fall back to beam_words
                        bw_data = cm.get_beam_words(mid, prompt)
                        if bw_data and isinstance(bw_data, dict):
                            top = sorted(bw_data.items(), key=lambda x: -x[1])[:15]
                            source_data[short] = {w: round(p * 100, 1) for w, p in top}
                            continue
                        # Fall back to score_vocab
                        sv_data = cm.get_score_vocab(mid, prompt)
                        if sv_data and isinstance(sv_data, dict):
                            top = sorted(sv_data.items(), key=lambda x: -x[1])[:15]
                            source_data[short] = {w: round(p * 100, 1) for w, p in top}
                            continue
                        # Fall back to raw logits → decode top tokens into words
                        logit_key = {"model": mid, "prompt": prompt}
                        logits = logits_stash.get(logit_key)
                        if logits is None:
                            continue
                        from scipy.special import softmax
                        probs = softmax(np.array(logits, dtype=np.float32))
                        top_idx = probs.argsort()[-50:][::-1]
                        try:
                            from transformers import AutoTokenizer
                            tok = AutoTokenizer.from_pretrained(mid)
                        except Exception:
                            continue
                        word_probs = {}
                        for idx in top_idx:
                            raw = tok.decode([int(idx)])
                            w = raw.strip().strip(".,;:!?\"'()[]{}—-–")
                            if not w or not w[0].isalpha():
                                continue
                            is_complete = raw.startswith(" ") or raw.startswith("\n") or len(tok.encode(" " + w, add_special_tokens=False)) == 1
                            if is_complete:
                                word_probs[w] = word_probs.get(w, 0) + float(probs[idx])
                            else:
                                word_probs[w + "…"] = word_probs.get(w + "…", 0) + float(probs[idx])
                        merged = {}
                        for w, p in word_probs.items():
                            if w.endswith("…"):
                                base_w = w[:-1]
                                full = None
                                for cand in word_probs:
                                    if not cand.endswith("…") and cand.startswith(base_w):
                                        full = cand
                                        break
                                if full:
                                    merged[full] = merged.get(full, 0) + p
                                else:
                                    merged[w] = merged.get(w, 0) + p
                            else:
                                merged[w] = merged.get(w, 0) + p
                        top = sorted(merged.items(), key=lambda x: -x[1])[:15]
                        source_data[short] = {w: round(p * 100, 1) for w, p in top}
                else:
                    for k in cm.iter_beam_keys():
                        if k.get("type") != "beam_cross_v1":
                            continue
                        if k.get("model") != model or k.get("prompt") != prompt:
                            continue
                        src = k.get("source", "")
                        data = cm.get_beams(k)
                        if not data:
                            continue
                        counts = Counter()
                        for s in data:
                            toks = s.get("token_texts", [])
                            key = " ".join(toks[:depth]) if len(toks) >= depth else ""
                            if key:
                                counts[key] += 1
                        source_data[src] = dict(counts.most_common(15))
                # In beam mode, add raw logit probabilities as annotations
                logit_probs = {}
                if mode == "beam":
                    try:
                        logits_stash = cm._stash("logits")
                        from scipy.special import softmax
                        import numpy as np
                        from .registry import Registry
                        reg = Registry()
                        family_models = [model] + list(reg.variants_of(model))
                        for mid in family_models:
                            lkey = {"model": mid, "prompt": prompt}
                            logits = logits_stash.get(lkey)
                            if logits is None:
                                continue
                            probs = softmax(np.array(logits, dtype=np.float32))
                            from transformers import AutoTokenizer
                            tok = AutoTokenizer.from_pretrained(mid)
                            top_idx = probs.argsort()[-15:][::-1]
                            short = mid.split("/")[-1].replace("-", "_")
                            logit_probs[short] = {tok.decode([int(i)]).strip(): round(float(probs[i]), 4) for i in top_idx}
                    except Exception:
                        pass
                # Build nickname map for source labels
                from .registry import NICKNAMES
                source_nicks = {}
                _nick_lookup = {}
                _ambiguous = set()
                for full_id, nick in NICKNAMES.items():
                    short = full_id.split("/")[-1]
                    full_trunc = short.replace("-", "_")
                    legacy_trunc = full_trunc[:20]
                    _nick_lookup[full_trunc] = nick
                    _nick_lookup[short] = nick
                    if legacy_trunc in _nick_lookup and _nick_lookup[legacy_trunc] != nick:
                        _ambiguous.add(legacy_trunc)
                    else:
                        _nick_lookup[legacy_trunc] = nick
                for a in _ambiguous:
                    _nick_lookup.pop(a, None)
                for src in source_data:
                    source_nicks[src] = _nick_lookup.get(src, src)
                self._respond(200, {"sources": source_data, "model": model, "prompt": prompt, "depth": depth, "logit_probs": logit_probs, "nicknames": source_nicks})

            elif endpoint == "/api/data/csv":
                import os, pandas as pd
                name = params.get("name", "")
                if not name or ".." in name or "/" in name:
                    self._respond(400, {"error": "invalid name"})
                    return
                base_dir = os.path.dirname(os.path.dirname(__file__))
                for ext in [".csv", ".parquet"]:
                    fpath = os.path.join(base_dir, "data", name + ext)
                    if os.path.exists(fpath):
                        break
                else:
                    self._respond(404, {"error": f"not found: {name}"})
                    return
                if fpath.endswith(".parquet"):
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath)
                limit = int(params.get("limit", 5000))
                self._respond(200, {"rows": _sanitize(df.head(limit).to_dict(orient="records")),
                                     "total": len(df), "columns": list(df.columns)})

            else:
                self._respond(404, {"error": f"Unknown endpoint: {endpoint}"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._respond(500, {"error": str(e)})

    def _serve_static(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        file_path = (_UI_DIR / path.lstrip("/")).resolve()
        if not file_path.is_relative_to(_UI_DIR.resolve()):
            self._respond(404, {"error": "not found"})
            return
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
            # Look up text from generation cache if psg is empty
            if not psg:
                model_id = body.get("model_id", "")
                prompt = body.get("gen_prompt", "")
                idx = body.get("idx", 0)
                if model_id:
                    from .cache import get_cache as _gc
                    _cache = _gc()
                    for temp in [1.0, 0.0]:
                        text = _cache.get_generation(model_id, prompt, temp=temp, idx=idx)
                        if text:
                            psg = text
                            prompt_prefix = prompt
                            break
            if not psg:
                raise ValueError("No psg provided")
            from .cache import get_cache
            cache = get_cache()
            # Try Pythia first, then GPT-2
            tok_surps = cache.get_ref_surprisal("EleutherAI/pythia-1b-deduped", prompt_prefix, psg)
            if tok_surps is None:
                tok_surps = cache.get_ref_surprisal("gpt2", prompt_prefix, psg)
            if tok_surps is None:
                from .embedding import passage_surprisal
                s = passage_surprisal(psg, prompt_prefix=prompt_prefix,
                                      model_name="EleutherAI/pythia-1b-deduped")
                tok_surps = s["token_surprisals"]
            # Per-sentence drift + tokens grouped by sentence
            sentences = []
            sent_vecs = cache.get_sent_embeddings("BAAI/bge-m3", prompt_prefix, psg)
            if sent_vecs and len(sent_vecs) >= 2 and tok_surps:
                import numpy as np
                from .embedding import _split_sentences
                from transformers import AutoTokenizer
                sents = _split_sentences(psg)
                if prompt_prefix and sents:
                    sents[0] = prompt_prefix + " " + sents[0]
                vecs = np.array(sent_vecs)
                centroid = vecs.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

                # Group tokens into sentences by reconstructing from token text
                # Strategy: concatenate token strings, find sentence boundaries
                # in the concatenated text, assign tokens accordingly
                tok_texts = [t for t, _ in tok_surps]
                tok_positions = []  # (start_char, end_char) in concatenated text
                pos = 0
                for t in tok_texts:
                    tok_positions.append((pos, pos + len(t)))
                    pos += len(t)
                full_tok_text = "".join(tok_texts)

                # Find each sentence's span in the token text
                # Strip prompt from first sentence since tokens don't include it
                search_from = 0
                tok_idx = 0
                for si, s in enumerate(sents[:len(vecs)]):
                    cos_dist = 1.0 - float(np.dot(vecs[si], centroid))
                    match_text = s
                    if si == 0 and prompt_prefix:
                        match_text = s[len(prompt_prefix):].lstrip()

                    # Find this sentence's end in the token text
                    end_pos = full_tok_text.find(match_text.rstrip(), search_from)
                    if end_pos >= 0:
                        end_pos += len(match_text.rstrip())
                    else:
                        # Fallback: advance by sentence length
                        end_pos = search_from + len(match_text)

                    # Collect tokens whose start falls within this sentence
                    sent_tokens = []
                    while tok_idx < len(tok_surps):
                        t_start, t_end = tok_positions[tok_idx]
                        if t_start >= end_pos:
                            break
                        sent_tokens.append(list(tok_surps[tok_idx]))
                        tok_idx += 1

                    search_from = end_pos
                    sentences.append({
                        "drift": round(cos_dist, 4),
                        "tokens": sent_tokens,
                    })

                # Remaining tokens to last sentence
                while tok_idx < len(tok_surps):
                    if sentences:
                        sentences[-1]["tokens"].append(list(tok_surps[tok_idx]))
                    tok_idx += 1

            return {"tokens": tok_surps, "sentences": sentences}

        elif path == "/passage-metrics-csv":
            import os
            base = os.path.dirname(os.path.dirname(__file__))
            for name in ["jakobson.parquet", "corpus_metrics.parquet", "corpus_metrics.csv"]:
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
            # Compatibility aliases for jakobson.parquet → UI expectations
            if "ref_surprisal" in df.columns and "surprisal_pythia_1b_deduped" not in df.columns:
                df["surprisal_pythia_1b_deduped"] = df["ref_surprisal"]
            if "total_drift" in df.columns and "drift_bge_m3" not in df.columns:
                df["drift_bge_m3"] = df["total_drift"]
            if "layer" in df.columns and "model" not in df.columns:
                df["model"] = df["layer"]
            if "prompt_label" in df.columns and "label" not in df.columns:
                df["label"] = df["prompt_label"]
            if "genre" in df.columns and "genre_type" not in df.columns:
                df["genre_type"] = df["genre"]
            if "psg" not in df.columns:
                df["psg"] = ""
            # Drop unused columns but keep psg (needed for passage viewer)
            drop = [c for c in df.columns if c.startswith(('max_', 'std_',
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
        self.send_response(204)  # status line MUST precede send_header calls
        self._cors_headers()
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


def serve(port=8421, family=None, host="127.0.0.1", data_only=False):
    """Start the model server."""
    global _family
    _family = family

    if not data_only:
        thread = threading.Thread(target=_get_psyche, daemon=True)
        thread.start()
        print(f"Models loading in background...")
    else:
        print(f"Data-only mode — no models loaded. API endpoints serve cached data.")

    server = ThreadingHTTPServer((host, port), ModelHandler)
    print(f"Server running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    serve()
