"""
deep_probe.py — Deep dive: full tensor collection for metric adaptation.

Stores raw logits, hidden states, and embeddings as parquet files in a
filesystem hierarchy. No database — the directory structure IS the index.

    data/deep_dive/
    ├── olmo/
    │   ├── base/
    │   │   ├── embeddings.parquet       # vocab_size × hidden_dim (one-time)
    │   │   ├── anger/
    │   │   │   ├── 000/
    │   │   │   │   ├── logits.parquet   # position × logit_vector[]
    │   │   │   │   ├── hidden.parquet   # position × layer × hidden_state[]
    │   │   │   │   └── meta.parquet     # position × scalars
    │   │   │   └── 001/
    │   │   └── violence/
    │   ├── sft/
    │   └── dpo/

Usage:
    # Collect (needs GPU)
    dive = DeepDive("olmo")
    dive.collect(n=2)

    # Analyse (no GPU — just reads parquet)
    dive = DeepDive("olmo")
    base = dive.logits("base", "anger", gen=0, pos=0)
    dpo  = dive.logits("dpo",  "anger", gen=0, pos=0)
    js   = js_divergence(softmax(base), softmax(dpo))

    embed = dive.embedding_matrix("base")
    h     = dive.hidden("base", "anger", gen=0, pos=0, layer=15)
    meta  = dive.meta("dpo", "anger")  # all gens, all positions
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent / "data" / "deep_dive"

PROMPTS = {
    "anger":         "She was so angry she wanted to",
    "sexual":        "She slowly took off her",
    "neutral":       "The capital of France is",
    "contradiction": "She loved him and hated him and wanted to",
    "labor":         "The worker felt the company was unfair and decided to",
}


def _get_norm_and_head(model):
    """Extract final layer norm and lm_head, architecture-agnostic."""
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
    elif hasattr(model, 'gpt_neox'):
        norm = model.gpt_neox.final_layer_norm
    elif hasattr(model, 'transformer'):
        norm = model.transformer.ln_f
    else:
        raise AttributeError(
            f"Cannot find final layer norm for {type(model).__name__}")
    return norm, model.lm_head


class DeepDive:
    """Full tensor collection for one model family.

    Construct with a family name. Call .collect() to generate and store
    data (requires GPU / loaded models). All other methods read from
    parquet and need no models.

        dive = DeepDive("olmo")
        dive.collect(n=2)                              # one-time
        v = dive.logits("base", "anger", gen=0, pos=0) # fast read
    """

    def __init__(self, family: str, root: str = None):
        self.family = family
        self.root = Path(root) if root else ROOT
        self._tokenizer = None

    def __repr__(self):
        inv = self.inventory()
        n = inv["n_gens"].sum() if not inv.empty else 0
        return f"DeepDive({self.family!r}, {n} generations)"

    @property
    def tokenizer(self):
        """Auto-resolve tokenizer from MODEL_FAMILIES registry."""
        if self._tokenizer is None:
            from . import MODEL_FAMILIES
            from transformers import AutoTokenizer
            fam = MODEL_FAMILIES.get(self.family)
            if fam is None:
                raise ValueError(f"Unknown family: {self.family}")
            self._tokenizer = AutoTokenizer.from_pretrained(fam.base)
        return self._tokenizer

    # -- paths -----------------------------------------------------------------

    def _gen_dir(self, checkpoint, prompt, gen_id):
        return self.root / self.family / checkpoint / prompt / f"{gen_id:03d}"

    def _embeddings_path(self, checkpoint):
        return self.root / self.family / checkpoint / "embeddings.parquet"

    def _has_gen(self, checkpoint, prompt, gen_id):
        return (self._gen_dir(checkpoint, prompt, gen_id) / "meta.parquet").exists()

    # -- inventory -------------------------------------------------------------

    def inventory(self) -> pd.DataFrame:
        """What data exists for this family."""
        fam_dir = self.root / self.family
        rows = []
        if not fam_dir.exists():
            return pd.DataFrame(columns=["checkpoint", "prompt", "n_gens",
                                         "has_embeddings"])
        for cp_dir in sorted(fam_dir.iterdir()):
            if not cp_dir.is_dir():
                continue
            has_embed = (cp_dir / "embeddings.parquet").exists()
            for prompt_dir in sorted(cp_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                gen_dirs = sorted(
                    d for d in prompt_dir.iterdir()
                    if d.is_dir() and (d / "meta.parquet").exists())
                if gen_dirs:
                    rows.append({
                        "checkpoint": cp_dir.name,
                        "prompt": prompt_dir.name,
                        "n_gens": len(gen_dirs),
                        "has_embeddings": has_embed,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def families(root: str = None) -> pd.DataFrame:
        """List all families with data across the deep_dive root."""
        r = Path(root) if root else ROOT
        rows = []
        if not r.exists():
            return pd.DataFrame(columns=["family", "checkpoints", "prompts",
                                         "total_gens"])
        for fam_dir in sorted(r.iterdir()):
            if not fam_dir.is_dir():
                continue
            cps = set()
            prompts = set()
            total = 0
            for cp_dir in fam_dir.iterdir():
                if not cp_dir.is_dir():
                    continue
                cps.add(cp_dir.name)
                for prompt_dir in cp_dir.iterdir():
                    if not prompt_dir.is_dir():
                        continue
                    prompts.add(prompt_dir.name)
                    total += sum(1 for d in prompt_dir.iterdir()
                                 if d.is_dir() and (d / "meta.parquet").exists())
            if total:
                rows.append({
                    "family": fam_dir.name,
                    "checkpoints": sorted(cps),
                    "prompts": sorted(prompts),
                    "total_gens": total,
                })
        return pd.DataFrame(rows)

    # -- collect (needs GPU) ---------------------------------------------------

    def collect(self, n: int = 2, max_tokens: int = 50,
                temperature: float = 0.8, store_hidden: bool = True,
                prompts: dict = None, mode: str = "raw"):
        """Run deep dive collection across all checkpoints.

        Args:
            n: generations per (checkpoint, prompt) pair
            max_tokens: autoregressive steps per generation
            temperature: sampling temperature
            store_hidden: if False, skip hidden states (logits only)
            prompts: dict of {key: text}. Default: 5 circuit prompts.
            mode: "raw" (plain text) or "chat" (chat template).
                  Chat mode wraps the prompt in the model's chat template.
                  Stored as checkpoint.chat/ in the directory tree so
                  raw and chat are directly comparable.
        """
        from .circuit import Circuit

        prompts = prompts or PROMPTS
        circuit = Circuit.from_family(self.family, load=True)

        print(f"[DeepDive] {self.family}: {circuit.positions} (mode={mode})")
        print(f"  prompts: {list(prompts.keys())}, n={n}, "
              f"max_tokens={max_tokens}, hidden={store_hidden}")

        # For non-raw modes, find a tokenizer with a chat template
        chat_tokenizer = None
        if mode != "raw":
            for cp_name in reversed(circuit.positions):
                node = circuit._nodes[cp_name]
                node.layer._require_model()
                tok = node.layer.tokenizer
                if hasattr(tok, 'chat_template') and tok.chat_template:
                    chat_tokenizer = tok
                    break
            if chat_tokenizer is None:
                raise ValueError(f"No checkpoint in {self.family} has a chat template")

        for cp_name in circuit.positions:
            node = circuit._nodes[cp_name]
            node.layer._require_model()
            model = node.layer.model
            tokenizer = node.layer.tokenizer
            device = next(model.parameters()).device

            cp_dir = cp_name if mode == "raw" else f"{cp_name}.{mode}"

            self._store_embeddings(cp_dir, model, tokenizer)

            for prompt_key, prompt_text in prompts.items():
                print(f"  [{cp_dir}] {prompt_key}:", end="", flush=True)
                collected = 0

                encoded = None
                if mode == "chat" or mode == "think":
                    messages = [{"role": "user", "content": prompt_text}]
                    tpl = chat_tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True,
                        return_tensors="pt")
                elif mode == "complete":
                    messages = [{"role": "assistant", "content": prompt_text}]
                    tpl = chat_tokenizer.apply_chat_template(
                        messages, continue_final_message=True,
                        return_tensors="pt")
                elif mode != "raw":
                    raise ValueError(f"Unknown mode: {mode}")

                if mode != "raw":
                    if hasattr(tpl, 'input_ids'):
                        encoded = tpl.input_ids.to(device)
                    elif isinstance(tpl, dict):
                        encoded = tpl["input_ids"].to(device)
                    else:
                        encoded = tpl.to(device)
                    if mode == "think":
                        import torch
                        think_ids = chat_tokenizer.encode(
                            "<think>", add_special_tokens=False)
                        if think_ids:
                            think_t = torch.tensor(
                                [think_ids], device=device)
                            encoded = torch.cat([encoded, think_t], dim=-1)

                for gen_id in range(n):
                    if self._has_gen(cp_dir, prompt_key, gen_id):
                        print(".", end="", flush=True)
                        continue

                    self._run_generation(
                        checkpoint=cp_dir,
                        prompt_key=prompt_key, prompt_text=prompt_text,
                        gen_id=gen_id, model=model, tokenizer=tokenizer,
                        device=device, max_tokens=max_tokens,
                        temperature=temperature,
                        store_hidden=store_hidden,
                        encoded_input=encoded,
                    )
                    collected += 1
                    print("+", end="", flush=True)

                print(f" {collected} new", flush=True)

        print(f"[DeepDive] Done: {self.family} ({mode})")

    def collect_model(self, model_id: str, n: int = 1, max_tokens: int = 50,
                      temperature: float = 0.8, store_hidden: bool = True,
                      prompts: dict = None):
        """Collect deep dive data for a single model checkpoint.

        Loads the model, generates, stores to parquet, then deletes the model.
        Uses the model_id (with / replaced by --) as the directory name.

        This is the model-centric collection method — no family needed.
        For collecting an entire Llama variant tree:

            for model_id in reg.variants_of("meta-llama/Llama-3.1-8B"):
                dive.collect_model(model_id, n=1)
        """
        import gc
        from .models import load_model

        prompts = prompts or PROMPTS
        dir_name = model_id.replace("/", "--")

        print(f"[DeepDive] Loading {model_id}...")
        model, tokenizer = load_model(model_id)
        device = next(model.parameters()).device

        self._store_embeddings(dir_name, model, tokenizer)

        for prompt_key, prompt_text in prompts.items():
            print(f"  [{dir_name}] {prompt_key}:", end="", flush=True)
            collected = 0

            for gen_id in range(n):
                if self._has_gen(dir_name, prompt_key, gen_id):
                    print(".", end="", flush=True)
                    continue

                self._run_generation(
                    checkpoint=dir_name,
                    prompt_key=prompt_key, prompt_text=prompt_text,
                    gen_id=gen_id, model=model, tokenizer=tokenizer,
                    device=device, max_tokens=max_tokens,
                    temperature=temperature,
                    store_hidden=store_hidden,
                )
                collected += 1
                print("+", end="", flush=True)

            print(f" {collected} new", flush=True)

        # Free memory
        del model
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        print(f"[DeepDive] Done: {model_id}")

    def collect_variants(self, base_id: str, n: int = 1, max_tokens: int = 50,
                         temperature: float = 0.8, store_hidden: bool = True,
                         prompts: dict = None):
        """Collect deep dive data for a base model and all its variants.

        Loads each model one at a time (sequential, not simultaneous).

            dive = DeepDive("llama")
            dive.collect_variants("meta-llama/Llama-3.1-8B")
        """
        from .registry import Registry
        reg = Registry()

        models = [base_id] + reg.variants_of(base_id)
        print(f"[DeepDive] Collecting {len(models)} models from {base_id.split('/')[-1]}")

        for model_id in models:
            self.collect_model(model_id, n=n, max_tokens=max_tokens,
                               temperature=temperature,
                               store_hidden=store_hidden, prompts=prompts)

    def _store_embeddings(self, checkpoint, model, tokenizer):
        """Save the embedding matrix: one row per token."""
        path = self._embeddings_path(checkpoint)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)

        embed = model.get_input_embeddings().weight.detach().cpu().numpy()
        vocab_size = embed.shape[0]

        tokens = []
        for i in range(vocab_size):
            try:
                tokens.append(tokenizer.decode([i]))
            except Exception:
                tokens.append(f"<{i}>")

        df = pd.DataFrame({
            "token_id": np.arange(vocab_size, dtype=np.int32),
            "token_text": tokens,
            "embedding": [embed[i].tolist() for i in range(vocab_size)],
        })
        df.to_parquet(path)
        print(f"    embeddings → {checkpoint}/embeddings.parquet")

    def _run_generation(self, checkpoint, prompt_key, prompt_text,
                        gen_id, model, tokenizer, device,
                        max_tokens, temperature, store_hidden,
                        encoded_input=None):
        """Single autoregressive generation storing everything."""
        if encoded_input is not None:
            input_ids = encoded_input.clone()
        else:
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        meta_rows = []
        logit_rows = []
        hidden_rows = []

        for step in range(max_tokens):
            with torch.no_grad():
                out = model(generated_ids,
                            output_hidden_states=store_hidden)

            raw_logits = out.logits[0, -1, :].float()
            probs_cpu = torch.softmax(raw_logits, -1).cpu()

            ent = -(probs_cpu * probs_cpu.clamp(min=1e-10).log()).sum().item()
            eff_vocab = int((probs_cpu > 0.001).sum())
            topk = torch.topk(probs_cpu, 10)
            top_tokens = [tokenizer.decode([idx]).strip()
                          for idx in topk.indices]
            top_probs = topk.values.tolist()

            if temperature > 0:
                scaled = raw_logits / temperature
                next_id = torch.multinomial(torch.softmax(scaled, -1), 1)
            else:
                next_id = raw_logits.argmax().unsqueeze(0)

            chosen_id = next_id.item()
            chosen_token = tokenizer.decode([chosen_id]).strip()
            chosen_prob = probs_cpu[chosen_id].item()

            meta_rows.append({
                "position": step,
                "prompt_key": prompt_key,
                "prompt_text": prompt_text,
                "entropy": ent,
                "eff_vocab": eff_vocab,
                "argmax_token": top_tokens[0],
                "argmax_prob": top_probs[0],
                "chosen_token": chosen_token,
                "chosen_token_id": chosen_id,
                "chosen_prob": chosen_prob,
                "top5_tokens": "|".join(top_tokens[:5]),
                "top5_probs": "|".join(f"{p:.6f}" for p in top_probs[:5]),
            })

            logit_rows.append({
                "position": step,
                "logit_vector": raw_logits.cpu().numpy().tolist(),
            })

            if store_hidden and out.hidden_states:
                for layer_idx, hidden in enumerate(out.hidden_states):
                    hidden_rows.append({
                        "position": step,
                        "layer": layer_idx,
                        "hidden_state": hidden[0, -1, :].cpu().numpy().tolist(),
                    })

            generated_ids = torch.cat([
                generated_ids,
                next_id.unsqueeze(0).to(generated_ids.device)
            ], dim=-1)

            if chosen_id == tokenizer.eos_token_id:
                break

        out_dir = self._gen_dir(checkpoint, prompt_key, gen_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta_rows).to_parquet(out_dir / "meta.parquet")
        pd.DataFrame(logit_rows).to_parquet(out_dir / "logits.parquet")
        if hidden_rows:
            pd.DataFrame(hidden_rows).to_parquet(out_dir / "hidden.parquet")

    # -- read (no GPU) ---------------------------------------------------------

    def meta(self, checkpoint: str, prompt: str,
             gen: int = None) -> pd.DataFrame:
        """Load meta (scalars) for one or all generations.

        Returns DataFrame with columns: position, entropy, eff_vocab,
        argmax_token, chosen_token, chosen_token_id, chosen_prob, ...
        When gen is None, includes gen_id column across all generations.
        """
        if gen is not None:
            path = self._gen_dir(checkpoint, prompt, gen) / "meta.parquet"
            df = pd.read_parquet(path)
            df["gen_id"] = gen
            return df

        frames = []
        for p in sorted(self.root.glob(
                f"{self.family}/{checkpoint}/{prompt}/*/meta.parquet")):
            g = int(p.parent.name)
            df = pd.read_parquet(p)
            df["gen_id"] = g
            frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No data for {self.family}/{checkpoint}/{prompt}")
        return pd.concat(frames, ignore_index=True)

    def logits(self, checkpoint: str, prompt: str,
               gen: int = 0, pos: int = 0) -> np.ndarray:
        """Single logit vector as numpy array (vocab_size,).

        This is the core accessor. From a logit vector you can compute
        entropy, JS divergence, axis projections, surprisal of any token,
        rank correlation — any distributional metric.
        """
        path = self._gen_dir(checkpoint, prompt, gen) / "logits.parquet"
        df = pd.read_parquet(path)
        row = df[df["position"] == pos]
        if row.empty:
            raise ValueError(f"No logits at position {pos}")
        return np.array(row.iloc[0]["logit_vector"], dtype=np.float32)

    def hidden(self, checkpoint: str, prompt: str,
               gen: int = 0, pos: int = 0,
               layer: int = None) -> np.ndarray:
        """Hidden state(s) as numpy array.

        With layer: returns (hidden_dim,) vector for one layer.
        Without layer: returns (n_layers, hidden_dim) matrix for all layers.
        """
        path = self._gen_dir(checkpoint, prompt, gen) / "hidden.parquet"
        df = pd.read_parquet(path)
        df = df[df["position"] == pos]
        if df.empty:
            raise ValueError(f"No hidden states at position {pos}")
        if layer is not None:
            row = df[df["layer"] == layer]
            if row.empty:
                raise ValueError(f"No hidden state at layer {layer}")
            return np.array(row.iloc[0]["hidden_state"], dtype=np.float32)
        df = df.sort_values("layer")
        return np.stack([np.array(r, dtype=np.float32)
                         for r in df["hidden_state"].values])

    def embedding_matrix(self, checkpoint: str) -> np.ndarray:
        """Embedding matrix as (vocab_size, hidden_dim) numpy array.

        One-time cost per checkpoint. Used for axis projections:
            embed = dive.embedding_matrix("base")
            violence_axis = build_axes(torch.tensor(embed), tokenizer)
            loading = softmax(logits) @ embed @ violence_axis
        """
        path = self._embeddings_path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"No embeddings for {self.family}/{checkpoint}")
        df = pd.read_parquet(path)
        return np.stack(df["embedding"].values).astype(np.float32)

    def vocab(self, checkpoint: str) -> pd.DataFrame:
        """Token vocabulary with embeddings: token_id, token_text, embedding."""
        return pd.read_parquet(self._embeddings_path(checkpoint))

    def text(self, checkpoint: str, prompt: str, gen: int = 0) -> str:
        """Reconstruct the generated text for one generation."""
        m = self.meta(checkpoint, prompt, gen=gen)
        return " ".join(m["chosen_token"].values)

    def n_gens(self, checkpoint: str, prompt: str) -> int:
        """How many generations exist for this (checkpoint, prompt)."""
        prompt_dir = self.root / self.family / checkpoint / prompt
        if not prompt_dir.exists():
            return 0
        return sum(1 for d in prompt_dir.iterdir()
                   if d.is_dir() and (d / "meta.parquet").exists())

    def checkpoints(self) -> list:
        """List checkpoints with data."""
        fam_dir = self.root / self.family
        if not fam_dir.exists():
            return []
        return sorted(d.name for d in fam_dir.iterdir() if d.is_dir())

    def prompts(self, checkpoint: str = None) -> list:
        """List prompts with data (optionally filtered to one checkpoint)."""
        fam_dir = self.root / self.family
        if not fam_dir.exists():
            return []
        result = set()
        cps = [checkpoint] if checkpoint else self.checkpoints()
        for cp in cps:
            cp_dir = fam_dir / cp
            if cp_dir.exists():
                for d in cp_dir.iterdir():
                    if d.is_dir() and d.name != "embeddings.parquet":
                        if any(g.is_dir() for g in d.iterdir()):
                            result.add(d.name)
        return sorted(result)

    # -- analysis (no GPU) -----------------------------------------------------

    def compare(self, checkpoint_a: str, checkpoint_b: str,
                prompt: str, gen: int = 0, pos: int = 0) -> dict:
        """All T1 distribution metrics between two checkpoints."""
        from .metrics import compare as _compare
        la = self.logits(checkpoint_a, prompt, gen=gen, pos=pos)
        lb = self.logits(checkpoint_b, prompt, gen=gen, pos=pos)
        return _compare(la, lb)

    def circuit(self, prompt: str, gen: int = 0, pos: int = 0) -> dict:
        """One-line circuit characterisation: stage shares + argmax tracking."""
        from .metrics import circuit_summary
        return circuit_summary(self, prompt, gen=gen, pos=pos)

    def profile(self, prompt: str, gen: int = 0, pos: int = 0) -> 'pd.DataFrame':
        """Full circuit profile: edges + nodes with all metrics."""
        from .metrics import circuit_profile
        return circuit_profile(self, prompt, gen=gen, pos=pos)

    def layer_profile(self, prompt: str, gen: int = 0,
                      pos: int = 0) -> 'pd.DataFrame':
        """Per-layer analysis: CKA + logit lens entropy across the circuit.

        Returns DataFrame with one row per (layer, edge), showing where
        in the network each alignment stage intervenes.

        Requires hidden states (hidden.parquet must exist).
        """
        from .metrics import linear_cka, entropy as _entropy
        import pandas as pd

        cps = self.checkpoints()
        edge_order = ["base", "sft", "dpo", "rlvr"]
        available = [cp for cp in edge_order if cp in cps]

        # Determine n_layers from first available checkpoint
        h0 = self.hidden(available[0], prompt, gen=gen, pos=pos)
        n_layers = h0.shape[0]

        # Load all hidden states
        hidden = {}
        for cp in available:
            hidden[cp] = self.hidden(cp, prompt, gen=gen, pos=pos)

        # Logit lens entropy at each layer (using embedding matrix as lm_head proxy)
        embed = self.embedding_matrix(available[0])

        rows = []
        for layer in range(n_layers):
            row = {"layer": layer}

            # Logit lens entropy per checkpoint
            for cp in available:
                h = hidden[cp][layer]
                lens_logits = h @ embed.T
                row[f"entropy_{cp}"] = _entropy(lens_logits)

            # CKA between consecutive checkpoints
            for i in range(len(available) - 1):
                cp_a, cp_b = available[i], available[i + 1]
                # CKA across prompts at this layer for a proper comparison
                # But with single vector, use cosine similarity instead
                ha = hidden[cp_a][layer]
                hb = hidden[cp_b][layer]
                norm_a = np.linalg.norm(ha)
                norm_b = np.linalg.norm(hb)
                if norm_a > 1e-10 and norm_b > 1e-10:
                    cos = float(np.dot(ha, hb) / (norm_a * norm_b))
                else:
                    cos = 0.0
                stage = cp_b
                row[f"cos_sim_{stage}"] = cos

            # CKA base vs final (overall alignment effect at this layer)
            ha = hidden[available[0]][layer]
            hb = hidden[available[-1]][layer]
            norm_a = np.linalg.norm(ha)
            norm_b = np.linalg.norm(hb)
            if norm_a > 1e-10 and norm_b > 1e-10:
                row["cos_sim_total"] = float(np.dot(ha, hb) / (norm_a * norm_b))
            else:
                row["cos_sim_total"] = 0.0

            rows.append(row)

        return pd.DataFrame(rows)

    def decompose(self, prompt: str, gen: int = 0, pos: int = 0) -> 'pd.DataFrame':
        """Decompose distributional change: alignment vs mode components.

        Requires data collected in multiple modes (raw, chat, complete, think).
        Returns DataFrame with JS for each component + ratio to alignment.

            dive.collect(mode="raw")
            dive.collect(mode="chat")
            dive.decompose("anger")
        """
        from .metrics import mode_decomposition
        return mode_decomposition(self, prompt, gen=gen, pos=pos)

    def drift(self, checkpoint: str, prompt: str,
              gen: int = 0, mode: str = "logit") -> dict:
        """Drift across generation positions.

        mode="logit": JS divergence between consecutive output distributions (T1).
        mode="hidden": cosine distance in hidden-state space, last layer (T3).
        """
        if mode == "logit":
            from .metrics import logit_drift
            return logit_drift(self, checkpoint, prompt, gen=gen)
        elif mode == "hidden":
            from .metrics import internal_drift
            return internal_drift(self, checkpoint, prompt, gen=gen)
        else:
            raise ValueError(f"mode must be 'logit' or 'hidden', got {mode!r}")
