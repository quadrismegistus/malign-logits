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
                prompts: dict = None):
        """Run deep dive collection across all checkpoints.

        Args:
            n: generations per (checkpoint, prompt) pair
            max_tokens: autoregressive steps per generation
            temperature: sampling temperature
            store_hidden: if False, skip hidden states (logits only)
            prompts: dict of {key: text}. Default: 5 circuit prompts.
        """
        from .circuit import Circuit

        prompts = prompts or PROMPTS
        circuit = Circuit.from_family(self.family, load=True)

        print(f"[DeepDive] {self.family}: {circuit.positions}")
        print(f"  prompts: {list(prompts.keys())}, n={n}, "
              f"max_tokens={max_tokens}, hidden={store_hidden}")

        for cp_name in circuit.positions:
            node = circuit._nodes[cp_name]
            node.layer._require_model()
            model = node.layer.model
            tokenizer = node.layer.tokenizer
            device = next(model.parameters()).device

            self._store_embeddings(cp_name, model, tokenizer)

            for prompt_key, prompt_text in prompts.items():
                print(f"  [{cp_name}] {prompt_key}:", end="", flush=True)
                collected = 0

                for gen_id in range(n):
                    if self._has_gen(cp_name, prompt_key, gen_id):
                        print(".", end="", flush=True)
                        continue

                    self._run_generation(
                        checkpoint=cp_name,
                        prompt_key=prompt_key, prompt_text=prompt_text,
                        gen_id=gen_id, model=model, tokenizer=tokenizer,
                        device=device, max_tokens=max_tokens,
                        temperature=temperature,
                        store_hidden=store_hidden,
                    )
                    collected += 1
                    print("+", end="", flush=True)

                print(f" {collected} new", flush=True)

        print(f"[DeepDive] Done: {self.family}")

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
                        max_tokens, temperature, store_hidden):
        """Single autoregressive generation storing everything."""
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
