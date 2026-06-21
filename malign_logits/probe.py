"""
probe.py — Model-centric data collection and analysis.

The primary interface for the project. Each model is identified by its
HuggingFace model ID. Data is stored as parquet files keyed by model ID.
Relations between models (sft_of, dpo_of, etc.) are in the Registry.

    data/deep_dive/
    ├── allenai--Olmo-3-1025-7B/
    │   ├── embeddings.parquet
    │   ├── anger/000/{logits,hidden,meta}.parquet
    │   └── anger/001/...
    ├── allenai--Olmo-3-7B-Instruct-SFT/
    │   └── ...

Usage:
    from malign_logits.probe import Probe

    # Single model
    p = Probe("allenai/Olmo-3-1025-7B")
    p.collect(n=2)
    p.logits("anger", gen=0, pos=0)

    # Compare two models
    Probe.compare("allenai/Olmo-3-1025-7B",
                  "allenai/Olmo-3-7B-Instruct-DPO", "anger")

    # All variants of a base
    Probe.collect_tree("meta-llama/Llama-3.1-8B", n=1)
    Probe.compare_tree("meta-llama/Llama-3.1-8B", "anger")
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


def _to_dir(model_id: str) -> str:
    """HuggingFace ID to filesystem-safe directory name."""
    return model_id.replace("/", "--")


def _from_dir(dir_name: str) -> str:
    """Filesystem directory name back to HuggingFace ID."""
    return dir_name.replace("--", "/", 1)


class Probe:
    """One model's stored data: logits, hidden states, embeddings.

    Construct with a HuggingFace model ID. Data lives in
    data/deep_dive/{model_id with -- for /}/.

        p = Probe("allenai/Olmo-3-1025-7B")
        p.logits("anger", gen=0, pos=0)   # numpy (vocab_size,)
        p.hidden("anger", gen=0, pos=0)   # numpy (n_layers, hidden_dim)
        p.meta("anger")                   # DataFrame
    """

    def __init__(self, model_id: str, root: str = None):
        self.model_id = model_id
        self.root = Path(root) if root else ROOT
        self._tokenizer = None

    def __repr__(self):
        n = self._count_gens()
        return f"Probe({self.model_id!r}, {n} gens)"

    @property
    def _dir(self):
        return self.root / _to_dir(self.model_id)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    # -- paths -----------------------------------------------------------------

    def _gen_dir(self, prompt, gen_id):
        return self._dir / prompt / f"{gen_id:03d}"

    def _embeddings_path(self):
        return self._dir / "embeddings.parquet"

    def _has_gen(self, prompt, gen_id):
        return (self._gen_dir(prompt, gen_id) / "meta.parquet").exists()

    def _count_gens(self):
        if not self._dir.exists():
            return 0
        total = 0
        for prompt_dir in self._dir.iterdir():
            if prompt_dir.is_dir() and prompt_dir.name != "embeddings.parquet":
                total += sum(1 for d in prompt_dir.iterdir()
                             if d.is_dir() and (d / "meta.parquet").exists())
        return total

    # -- collect (needs GPU) ---------------------------------------------------

    def collect(self, n: int = 2, max_tokens: int = 50,
                temperature: float = 0.8, store_hidden: bool = True,
                prompts: dict = None):
        """Load this model, generate, store to parquet, free memory."""
        import gc
        from .models import load_model

        prompts = prompts or PROMPTS

        print(f"[Probe] Loading {self.model_id}...")
        model, tokenizer = load_model(self.model_id)
        self._tokenizer = tokenizer
        device = next(model.parameters()).device

        self._store_embeddings(model, tokenizer)

        for prompt_key, prompt_text in prompts.items():
            print(f"  {prompt_key}:", end="", flush=True)
            collected = 0

            for gen_id in range(n):
                if self._has_gen(prompt_key, gen_id):
                    print(".", end="", flush=True)
                    continue

                self._run_generation(
                    prompt_key=prompt_key, prompt_text=prompt_text,
                    gen_id=gen_id, model=model, tokenizer=tokenizer,
                    device=device, max_tokens=max_tokens,
                    temperature=temperature, store_hidden=store_hidden,
                )
                collected += 1
                print("+", end="", flush=True)

            print(f" {collected} new", flush=True)

        del model
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        print(f"[Probe] Done: {self.model_id}")

    def _store_embeddings(self, model, tokenizer):
        path = self._embeddings_path()
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

        pd.DataFrame({
            "token_id": np.arange(vocab_size, dtype=np.int32),
            "token_text": tokens,
            "embedding": [embed[i].tolist() for i in range(vocab_size)],
        }).to_parquet(path)
        print(f"  embeddings saved")

    def _run_generation(self, prompt_key, prompt_text, gen_id,
                        model, tokenizer, device,
                        max_tokens, temperature, store_hidden):
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        meta_rows, logit_rows, hidden_rows = [], [], []

        for step in range(max_tokens):
            with torch.no_grad():
                out = model(generated_ids,
                            output_hidden_states=store_hidden)

            raw_logits = out.logits[0, -1, :].float()
            probs_cpu = torch.softmax(raw_logits, -1).cpu()

            ent = -(probs_cpu * probs_cpu.clamp(min=1e-10).log()).sum().item()
            eff_vocab = int((probs_cpu > 0.001).sum())
            topk = torch.topk(probs_cpu, 10)
            top_tokens = [tokenizer.decode([idx]).strip() for idx in topk.indices]
            top_probs = topk.values.tolist()

            if temperature > 0:
                next_id = torch.multinomial(
                    torch.softmax(raw_logits / temperature, -1), 1)
            else:
                next_id = raw_logits.argmax().unsqueeze(0)

            chosen_id = next_id.item()

            meta_rows.append({
                "position": step,
                "prompt_key": prompt_key,
                "prompt_text": prompt_text,
                "entropy": ent,
                "eff_vocab": eff_vocab,
                "argmax_token": top_tokens[0],
                "argmax_prob": top_probs[0],
                "chosen_token": tokenizer.decode([chosen_id]).strip(),
                "chosen_token_id": chosen_id,
                "chosen_prob": probs_cpu[chosen_id].item(),
                "top5_tokens": "|".join(top_tokens[:5]),
                "top5_probs": "|".join(f"{p:.6f}" for p in top_probs[:5]),
            })

            logit_rows.append({
                "position": step,
                "logit_vector": raw_logits.cpu().numpy().tolist(),
            })

            if store_hidden and out.hidden_states:
                for layer_idx, h in enumerate(out.hidden_states):
                    hidden_rows.append({
                        "position": step,
                        "layer": layer_idx,
                        "hidden_state": h[0, -1, :].cpu().numpy().tolist(),
                    })

            generated_ids = torch.cat([
                generated_ids,
                next_id.unsqueeze(0).to(generated_ids.device)
            ], dim=-1)

            if chosen_id == tokenizer.eos_token_id:
                break

        out_dir = self._gen_dir(prompt_key, gen_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta_rows).to_parquet(out_dir / "meta.parquet")
        pd.DataFrame(logit_rows).to_parquet(out_dir / "logits.parquet")
        if hidden_rows:
            pd.DataFrame(hidden_rows).to_parquet(out_dir / "hidden.parquet")

    # -- read (no GPU) ---------------------------------------------------------

    def logits(self, prompt: str, gen: int = 0, pos: int = 0) -> np.ndarray:
        """Logit vector as numpy (vocab_size,)."""
        path = self._gen_dir(prompt, gen) / "logits.parquet"
        df = pd.read_parquet(path)
        row = df[df["position"] == pos]
        if row.empty:
            raise ValueError(f"No logits at position {pos}")
        return np.array(row.iloc[0]["logit_vector"], dtype=np.float32)

    def hidden(self, prompt: str, gen: int = 0, pos: int = 0,
               layer: int = None) -> np.ndarray:
        """Hidden states. With layer: (hidden_dim,). Without: (n_layers, hidden_dim)."""
        path = self._gen_dir(prompt, gen) / "hidden.parquet"
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

    def meta(self, prompt: str, gen: int = None) -> pd.DataFrame:
        """Meta (scalars + text). gen=None returns all generations."""
        if gen is not None:
            df = pd.read_parquet(self._gen_dir(prompt, gen) / "meta.parquet")
            df["gen_id"] = gen
            return df
        frames = []
        prompt_dir = self._dir / prompt
        if not prompt_dir.exists():
            raise FileNotFoundError(f"No data for {self.model_id}/{prompt}")
        for gdir in sorted(prompt_dir.iterdir()):
            if gdir.is_dir() and (gdir / "meta.parquet").exists():
                df = pd.read_parquet(gdir / "meta.parquet")
                df["gen_id"] = int(gdir.name)
                frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No data for {self.model_id}/{prompt}")
        return pd.concat(frames, ignore_index=True)

    def embedding_matrix(self) -> np.ndarray:
        """(vocab_size, hidden_dim) numpy array."""
        path = self._embeddings_path()
        if not path.exists():
            raise FileNotFoundError(f"No embeddings for {self.model_id}")
        df = pd.read_parquet(path)
        return np.stack(df["embedding"].values).astype(np.float32)

    def vocab(self) -> pd.DataFrame:
        """Token table: token_id, token_text, embedding."""
        return pd.read_parquet(self._embeddings_path())

    def text(self, prompt: str, gen: int = 0) -> str:
        """Reconstructed generated text."""
        return " ".join(self.meta(prompt, gen=gen)["chosen_token"].values)

    def prompts(self) -> list:
        """Prompts with data."""
        if not self._dir.exists():
            return []
        return sorted(d.name for d in self._dir.iterdir()
                       if d.is_dir() and any(g.is_dir() for g in d.iterdir()))

    def n_gens(self, prompt: str) -> int:
        prompt_dir = self._dir / prompt
        if not prompt_dir.exists():
            return 0
        return sum(1 for d in prompt_dir.iterdir()
                   if d.is_dir() and (d / "meta.parquet").exists())

    # -- family resolution -----------------------------------------------------

    FAMILIES = {
        "olmo3-7b":      "allenai/Olmo-3-1025-7B",
        "olmo2-1b":      "allenai/OLMo-2-0425-1B",
        "llama3.1-8b":   "meta-llama/Llama-3.1-8B",
        "qwen2.5-7b":    "Qwen/Qwen2.5-7B",
        "qwen2.5-0.5b":  "Qwen/Qwen2.5-0.5B",
        "qwen3-8b":      "Qwen/Qwen3-8B-Base",
        "amber-7b":      "LLM360/Amber",
        "mistral-7b":    "mistralai/Mistral-7B-v0.1",
        "pythia-6.9b":   "EleutherAI/pythia-6.9b",
        "deepseek-7b":   "deepseek-ai/deepseek-llm-7b-base",
        "smollm2-360m":  "HuggingFaceTB/SmolLM2-360M",
        "smollm3-3b":    "HuggingFaceTB/SmolLM3-3B-Base",
    }

    @classmethod
    def resolve(cls, name: str) -> str:
        """Resolve a friendly family name to a HuggingFace base model ID.

        Accepts either a family name ("olmo") or a full model ID
        ("allenai/Olmo-3-1025-7B"). Returns the model ID unchanged
        if it contains a slash.
        """
        if "/" in name:
            return name
        if name in cls.FAMILIES:
            return cls.FAMILIES[name]
        raise ValueError(
            f"Unknown family: {name}. "
            f"Available: {', '.join(sorted(cls.FAMILIES.keys()))}")

    @classmethod
    def families(cls) -> pd.DataFrame:
        """List all known families with their base model IDs and variant counts."""
        from .registry import Registry
        reg = Registry()
        rows = []
        for name, base_id in sorted(cls.FAMILIES.items()):
            info = reg.info(base_id)
            variants = reg.variants_of(base_id)
            rows.append({
                "family": name,
                "base_model": base_id,
                "variants": len(variants),
                "org": info.org if info else "",
                "country": info.country if info else "",
                "org_type": info.org_type if info else "",
                "scale": info.scale if info else "",
            })
        return pd.DataFrame(rows)

    # -- class methods: multi-model operations ---------------------------------

    @staticmethod
    def compare(model_a: str, model_b: str, prompt: str,
                gen: int = 0, pos: int = 0) -> dict:
        """All T1 metrics between two models."""
        from .metrics import compare as _compare
        pa = Probe(model_a)
        pb = Probe(model_b)
        return _compare(pa.logits(prompt, gen, pos),
                        pb.logits(prompt, gen, pos))

    @classmethod
    def collect_tree(cls, name: str, n: int = 1, max_tokens: int = 50,
                     temperature: float = 0.8, store_hidden: bool = True,
                     prompts: dict = None):
        """Collect data for a base model and all its variants.

        Accepts family name ("olmo") or HuggingFace ID.
        """
        from .registry import Registry
        base_id = cls.resolve(name)
        reg = Registry()
        models = [base_id] + reg.variants_of(base_id)
        print(f"[Probe] Collecting {len(models)} models from {base_id}")
        for model_id in models:
            p = Probe(model_id)
            p.collect(n=n, max_tokens=max_tokens, temperature=temperature,
                      store_hidden=store_hidden, prompts=prompts)

    @classmethod
    def compare_tree(cls, name: str, prompt: str,
                     gen: int = 0, pos: int = 0) -> pd.DataFrame:
        """Compare all variants of a base model on one prompt.

        Accepts family name ("olmo") or HuggingFace ID.
        """
        from .registry import Registry
        from .metrics import js_divergence, base_token_surprisal, entropy

        base_id = cls.resolve(name)
        reg = Registry()
        base_probe = Probe(base_id)
        base_logits = base_probe.logits(prompt, gen, pos)
        base_argmax = int(np.argmax(base_logits))
        tok = base_probe.tokenizer
        base_token = tok.decode([base_argmax]).strip()

        rows = []
        for model_id in reg.variants_of(base_id):
            try:
                p = Probe(model_id)
                logits = p.logits(prompt, gen, pos)
            except (FileNotFoundError, ValueError):
                continue

            parent, relation = reg.parent_of(model_id)
            rows.append({
                "model": model_id,
                "relation": relation or "",
                "stage": reg.stage_of(model_id),
                "js_from_base": js_divergence(base_logits, logits),
                f"surprisal_{base_token}": base_token_surprisal(base_logits, logits),
                "entropy": entropy(logits),
                "argmax": tok.decode([int(np.argmax(logits))]).strip(),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def inventory(root: str = None) -> pd.DataFrame:
        """All models with stored data."""
        r = Path(root) if root else ROOT
        if not r.exists():
            return pd.DataFrame(columns=["model_id", "prompts", "total_gens"])
        rows = []
        for model_dir in sorted(r.iterdir()):
            if not model_dir.is_dir():
                continue
            model_id = _from_dir(model_dir.name)
            prompts = []
            total = 0
            for prompt_dir in model_dir.iterdir():
                if not prompt_dir.is_dir():
                    continue
                n = sum(1 for d in prompt_dir.iterdir()
                        if d.is_dir() and (d / "meta.parquet").exists())
                if n:
                    prompts.append(prompt_dir.name)
                    total += n
            if total:
                rows.append({
                    "model_id": model_id,
                    "prompts": prompts,
                    "total_gens": total,
                })
        return pd.DataFrame(rows)
