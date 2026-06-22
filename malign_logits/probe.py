"""
probe.py — Model-centric data collection and analysis.

Each model is identified by its HuggingFace model ID. Data is stored
in HashStash (lmdb) via CacheManager, keyed by (model, prompt, gen, pos).

    from malign_logits.probe import Probe

    # Single model
    p = Probe("allenai/Olmo-3-1025-7B")
    p.collect(n=2)
    p.logits("anger", gen=0, pos=0)

    # Compare two models
    Probe.compare("allenai/Olmo-3-1025-7B",
                  "allenai/Olmo-3-7B-Instruct-DPO", "anger")

    # All variants of a base
    Probe.collect_tree("llama3.1-8b", n=1)
    Probe.compare_tree("llama3.1-8b", "anger")
"""

import numpy as np
import pandas as pd
import torch

PROMPTS = {
    "anger":         "She was so angry she wanted to",
    "sexual":        "She slowly took off her",
    "neutral":       "The capital of France is",
    "contradiction": "She loved him and hated him and wanted to",
    "labor":         "The worker felt the company was unfair and decided to",
}


def _get_cache():
    from .cache import get_cache
    return get_cache()


class Probe:
    """One model's stored data: logits, hidden states, embeddings.

        p = Probe("allenai/Olmo-3-1025-7B")
        p.logits("anger", gen=0, pos=0)   # numpy (vocab_size,)
        p.hidden("anger", gen=0, pos=0)   # numpy (n_layers, hidden_dim)
        p.meta("anger")                   # list of dicts (one per position)
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._tokenizer = None

    def __repr__(self):
        prompts = self.prompts()
        n = sum(self.n_gens(p) for p in prompts) if prompts else 0
        return f"Probe({self.model_id!r}, {n} gens)"

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    # -- collect (needs GPU) ---------------------------------------------------

    def collect(self, n: int = 2, max_tokens: int = 50,
                temperature: float = 0.8,
                prompts: dict = None, mode: str = "raw"):
        """Load this model, generate, store to HashStash, free memory.

        Always stores logits, hidden states, and meta at every position.
        Hidden states can't be recreated without the exact generation
        trajectory, so they are never optional.

        mode: "raw" (plain text), "chat" (chat template), "complete"
              (assistant-only), "think" (chat + <think>).
              Stored under model_id::mode in the cache.
        """
        import gc
        from .models import load_model

        prompts = prompts or PROMPTS
        cache = _get_cache()
        store_id = self.model_id if mode == "raw" else f"{self.model_id}::{mode}"

        print(f"[Probe] Loading {self.model_id} (mode={mode})...")
        model, tokenizer = load_model(self.model_id)
        self._tokenizer = tokenizer
        device = next(model.parameters()).device

        # Find chat template if needed
        encoded_inputs = {}
        if mode != "raw":
            if not (hasattr(tokenizer, 'chat_template') and tokenizer.chat_template):
                raise ValueError(f"{self.model_id} has no chat template")
            for prompt_key, prompt_text in prompts.items():
                if mode in ("chat", "think"):
                    msgs = [{"role": "user", "content": prompt_text}]
                    tpl = tokenizer.apply_chat_template(
                        msgs, add_generation_prompt=True, return_tensors="pt")
                elif mode == "complete":
                    msgs = [{"role": "assistant", "content": prompt_text}]
                    tpl = tokenizer.apply_chat_template(
                        msgs, continue_final_message=True, return_tensors="pt")
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                enc = tpl.input_ids if hasattr(tpl, 'input_ids') else (
                    tpl["input_ids"] if isinstance(tpl, dict) else tpl)
                enc = enc.to(device)
                if mode == "think":
                    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
                    if think_ids:
                        enc = torch.cat([enc, torch.tensor([think_ids], device=device)], dim=-1)
                encoded_inputs[prompt_key] = enc

        for prompt_key, prompt_text in prompts.items():
            print(f"  {prompt_key}:", end="", flush=True)
            collected = 0

            for gen_id in range(n):
                if cache.has_probe(store_id, prompt_key,
                                   gen=gen_id, pos=0, max_tokens=max_tokens):
                    print(".", end="", flush=True)
                    continue

                self._run_generation(
                    prompt_key=prompt_key, prompt_text=prompt_text,
                    gen_id=gen_id, model=model, tokenizer=tokenizer,
                    device=device, max_tokens=max_tokens,
                    temperature=temperature,
                    cache=cache, store_id=store_id,
                    encoded_input=encoded_inputs.get(prompt_key),
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

        print(f"[Probe] Done: {store_id}")

    def _run_generation(self, prompt_key, prompt_text, gen_id,
                        model, tokenizer, device,
                        max_tokens, temperature, cache,
                        store_id=None, encoded_input=None):
        store_id = store_id or self.model_id
        if encoded_input is not None:
            input_ids = encoded_input.clone()
        else:
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        meta_rows = []

        for step in range(max_tokens):
            with torch.no_grad():
                out = model(generated_ids,
                            output_hidden_states=True)

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

            # Store logits at this position
            cache.set_probe_logits(
                store_id, prompt_key,
                raw_logits.cpu().numpy(), gen=gen_id, pos=step,
                max_tokens=max_tokens)

            # Store hidden states at this position
            if out.hidden_states:
                hidden_np = np.stack([
                    h[0, -1, :].cpu().numpy() for h in out.hidden_states
                ])  # (n_layers, hidden_dim)
                cache.set_probe_hidden(
                    store_id, prompt_key,
                    hidden_np, gen=gen_id, pos=step,
                    max_tokens=max_tokens)

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

            generated_ids = torch.cat([
                generated_ids,
                next_id.unsqueeze(0).to(generated_ids.device)
            ], dim=-1)

            if chosen_id == tokenizer.eos_token_id:
                break

        # Store meta for this generation (all positions at once)
        cache.set_probe_meta(store_id, prompt_key, meta_rows, gen=gen_id,
                            max_tokens=max_tokens)

    def teacher_force(self, token_source: str, prompt: str,
                      gen: int = 0, max_tokens: int = None):
        """Run this model on another model's generated tokens.

        Feeds the token sequence from token_source's generation into
        this model, collecting logits and hidden states at each position.
        Both models process IDENTICAL input, so any hidden state difference
        is purely from weight changes — no path dependency confound.

        Results stored under prompt key "{prompt}::tf_{source_short}".

            # Base generated tokens
            base = Probe("allenai/Olmo-3-1025-7B")
            # Feed base tokens through aligned model
            aligned = Probe("allenai/Olmo-3-7B-Instruct-DPO")
            aligned.teacher_force("allenai/Olmo-3-1025-7B", "anger")
            # Now compare hidden states — clean, no path dependency
            h_base = base.hidden("anger", pos=10)
            h_aligned = aligned.hidden("anger::tf_Olmo-3-1025-7B", pos=10)
        """
        import gc
        from .models import load_model

        source = Probe(token_source)
        source_meta = source.meta(prompt, gen=gen, max_tokens=max_tokens)
        token_ids = source_meta["chosen_token_id"].tolist()
        prompt_text = source_meta.iloc[0]["prompt_text"]
        T = len(token_ids)

        source_short = token_source.split("/")[-1]
        tf_prompt = f"{prompt}::tf_{source_short}"
        cache = _get_cache()

        if cache.has_probe(self.model_id, tf_prompt, gen=gen, pos=0, max_tokens=T):
            print(f"[Probe] Already teacher-forced: {self.model_id} on {token_source}/{prompt}")
            return

        print(f"[Probe] Teacher-forcing {self.model_id} with tokens from {token_source}/{prompt} ({T} tokens)...")
        model, tokenizer = load_model(self.model_id)
        self._tokenizer = tokenizer
        device = next(model.parameters()).device

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        meta_rows = []

        for step in range(T):
            with torch.no_grad():
                out = model(generated_ids, output_hidden_states=True)

            raw_logits = out.logits[0, -1, :].float()
            probs_cpu = torch.softmax(raw_logits, -1).cpu()

            ent = -(probs_cpu * probs_cpu.clamp(min=1e-10).log()).sum().item()
            eff_vocab = int((probs_cpu > 0.001).sum())
            topk = torch.topk(probs_cpu, 10)
            top_tokens = [tokenizer.decode([idx]).strip() for idx in topk.indices]
            top_probs = topk.values.tolist()

            forced_id = token_ids[step]

            cache.set_probe_logits(
                self.model_id, tf_prompt,
                raw_logits.cpu().numpy(), gen=gen, pos=step, max_tokens=T)

            if out.hidden_states:
                hidden_np = np.stack([
                    h[0, -1, :].cpu().numpy() for h in out.hidden_states
                ])
                cache.set_probe_hidden(
                    self.model_id, tf_prompt,
                    hidden_np, gen=gen, pos=step, max_tokens=T)

            meta_rows.append({
                "position": step,
                "prompt_key": tf_prompt,
                "prompt_text": prompt_text,
                "entropy": ent,
                "eff_vocab": eff_vocab,
                "argmax_token": top_tokens[0],
                "argmax_prob": top_probs[0],
                "chosen_token": tokenizer.decode([forced_id]).strip(),
                "chosen_token_id": forced_id,
                "chosen_prob": probs_cpu[forced_id].item() if forced_id < len(probs_cpu) else 0.0,
                "top5_tokens": "|".join(top_tokens[:5]),
                "top5_probs": "|".join(f"{p:.6f}" for p in top_probs[:5]),
            })

            generated_ids = torch.cat([
                generated_ids,
                torch.tensor([[forced_id]], device=device)
            ], dim=-1)

        cache.set_probe_meta(self.model_id, tf_prompt, meta_rows, gen=gen,
                            max_tokens=T)

        del model
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        print(f"[Probe] Done: {self.model_id} teacher-forced on {token_source}/{prompt}")

    # -- read (no GPU) ---------------------------------------------------------

    def _resolve_T(self, prompt, gen, max_tokens):
        """Find max_tokens for a given (prompt, gen) if not specified.

        Checks known lengths in descending order. Falls back to None
        for legacy data stored without T in the key.
        """
        if max_tokens is not None:
            return max_tokens
        cache = _get_cache()
        for T in [100, 50, 20, 10, 5]:
            if cache.has_probe(self.model_id, prompt, gen=gen, pos=0, max_tokens=T):
                return T
        # Legacy: data stored without T in key
        old_key = {"model": self.model_id, "prompt": prompt, "gen": gen, "pos": 0}
        if old_key in cache._stash("probe_logits"):
            return None
        return 20

    def _probe_get(self, stash_name, prompt, gen, pos, max_tokens):
        """Get from probe stash with legacy fallback."""
        cache = _get_cache()
        T = self._resolve_T(prompt, gen, max_tokens)
        if T is not None:
            return cache._stash(stash_name).get(
                {"model": self.model_id, "prompt": prompt,
                 "gen": gen, "pos": pos, "T": T})
        # Legacy key (no T)
        key = {"model": self.model_id, "prompt": prompt, "gen": gen, "pos": pos}
        s = cache._stash(stash_name)
        return s[key] if key in s else None

    def _meta_get(self, prompt, gen, max_tokens):
        """Get meta with legacy fallback."""
        cache = _get_cache()
        T = self._resolve_T(prompt, gen, max_tokens)
        if T is not None:
            return cache.get_probe_meta(self.model_id, prompt, gen, max_tokens=T)
        # Legacy key (no T)
        key = {"model": self.model_id, "prompt": prompt, "gen": gen}
        s = cache._stash("probe_meta")
        return s[key] if key in s else None

    def logits(self, prompt: str, gen: int = 0, pos: int = 0,
               max_tokens: int = None) -> np.ndarray:
        """Logit vector as numpy (vocab_size,)."""
        v = self._probe_get("probe_logits", prompt, gen, pos, max_tokens)
        if v is None:
            raise FileNotFoundError(
                f"No logits for {self.model_id}/{prompt} gen={gen} pos={pos}")
        return np.asarray(v, dtype=np.float32)

    def hidden(self, prompt: str, gen: int = 0, pos: int = 0,
               layer: int = None, max_tokens: int = None) -> np.ndarray:
        """Hidden states. With layer: (hidden_dim,). Without: (n_layers, hidden_dim)."""
        v = self._probe_get("probe_hidden", prompt, gen, pos, max_tokens)
        if v is None:
            raise FileNotFoundError(
                f"No hidden for {self.model_id}/{prompt} gen={gen} pos={pos}")
        h = np.asarray(v, dtype=np.float32)
        if layer is not None:
            return h[layer]
        return h

    def meta(self, prompt: str, gen: int = None,
             max_tokens: int = None) -> pd.DataFrame:
        """Meta (scalars + text). gen=None returns all generations."""
        if gen is not None:
            rows = self._meta_get(prompt, gen, max_tokens)
            if rows is None:
                raise FileNotFoundError(
                    f"No meta for {self.model_id}/{prompt} gen={gen}")
            df = pd.DataFrame(rows)
            df["gen_id"] = gen
            return df

        frames = []
        for g in range(self.n_gens(prompt, max_tokens=max_tokens)):
            rows = self._meta_get(prompt, g, max_tokens)
            if rows:
                df = pd.DataFrame(rows)
                df["gen_id"] = g
                frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No meta for {self.model_id}/{prompt}")
        return pd.concat(frames, ignore_index=True)

    def embedding_matrix(self) -> np.ndarray:
        """(vocab_size, hidden_dim) numpy array.

        Loaded from cache if available, otherwise extracted from the model
        weights on the fly (requires downloading the model but no inference).
        """
        cache = _get_cache()
        v = cache.get_probe_embeddings(self.model_id)
        if v is not None:
            return np.asarray(v, dtype=np.float32)

        from transformers import AutoModelForCausalLM
        print(f"[Probe] Loading {self.model_id} for embeddings...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16)
        embed = model.get_input_embeddings().weight.detach().cpu().numpy()
        cache.set_probe_embeddings(self.model_id, embed)
        del model
        return embed.astype(np.float32)

    def text(self, prompt: str, gen: int = 0) -> str:
        """Reconstructed generated text."""
        return " ".join(self.meta(prompt, gen=gen)["chosen_token"].values)

    def n_gens(self, prompt: str, max_tokens: int = None) -> int:
        T = self._resolve_T(prompt, 0, max_tokens)
        if T is not None:
            return _get_cache().count_probe_gens(self.model_id, prompt,
                                                  max_tokens=T)
        # Legacy: count without T
        cache = _get_cache()
        s = cache._stash("probe_logits")
        g = 0
        while {"model": self.model_id, "prompt": prompt, "gen": g, "pos": 0} in s:
            g += 1
        return g

    def prompts(self, max_tokens: int = None) -> list:
        """Prompts with data."""
        result = []
        for p in PROMPTS:
            if self._resolve_T(p, 0, max_tokens) is not None:
                result.append(p)
            elif max_tokens is not None:
                pass  # explicitly requested T, not found
            else:
                # Check legacy
                cache = _get_cache()
                key = {"model": self.model_id, "prompt": p, "gen": 0, "pos": 0}
                if key in cache._stash("probe_logits"):
                    result.append(p)
        return result

    def trajectory(self, prompt: str, axis: str = "violence",
                   gen: int = 0) -> 'pd.DataFrame':
        """Track semantic axis loading at every position through a generation.

        Shows where in semantic space (violence-land, procedural-land, etc.)
        the generation lives over time. Both output-level and hidden-level.

            base.trajectory("anger", "violence")
            # → DataFrame: position, output_violence, hidden_violence, chosen_token
        """
        from .metrics import axis_trajectory, violence_procedural_axes
        embed = self.embedding_matrix()
        tok = self.tokenizer
        v_axis, p_axis = violence_procedural_axes(embed, tok)
        ax = v_axis if axis == "violence" else p_axis
        return axis_trajectory(self, prompt, embed, ax, axis_name=axis, gen=gen)

    # -- cross-model analysis (hidden states) ------------------------------------

    def hidden_divergence(self, other_model: str, prompt: str,
                          gen: int = 0) -> dict:
        """How hidden distance between this model and another evolves over positions.

        Returns dict with per-position distances, growth rate, mean.
        """
        from .metrics import hidden_divergence_trajectory
        return hidden_divergence_trajectory(
            self, Probe(other_model), prompt, gen=gen)

    def hidden_by_prompt(self, other_model: str,
                         gen: int = 0, pos: int = 0) -> dict:
        """Is hidden distance content-dependent? Per-prompt distances + variance."""
        from .metrics import hidden_distance_by_prompt
        return hidden_distance_by_prompt(
            self, Probe(other_model), gen=gen, pos=pos)

    def formation(self, prompt: str = "anger",
                  gen: int = 0, pos: int = 0, k: int = 30) -> 'pd.DataFrame':
        """Track top-k token probabilities across base→SFT→DPO pipeline."""
        from .metrics import formation_trajectory
        return formation_trajectory(self, prompt=prompt, gen=gen, pos=pos, k=k)

    # -- family resolution -----------------------------------------------------

    FAMILIES = {
        # Allen AI (US, nonprofit)
        "olmo3-7b":      "allenai/Olmo-3-1025-7B",
        "olmo2-1b":      "allenai/OLMo-2-0425-1B",
        # Meta (US, corporate)
        "llama3.1-8b":   "meta-llama/Llama-3.1-8B",
        # Alibaba (CN, corporate)
        "qwen2.5-7b":    "Qwen/Qwen2.5-7B",
        "qwen2.5-0.5b":  "Qwen/Qwen2.5-0.5B",
        "qwen3-8b":      "Qwen/Qwen3-8B-Base",
        # Google (US, corporate)
        "gemma-7b":      "google/gemma-7b",
        # Mistral (FR, corporate)
        "mistral-7b":    "mistralai/Mistral-7B-v0.1",
        # TII (AE, state)
        "falcon-7b":     "tiiuae/falcon-7b",
        # Baichuan (CN, corporate)
        "baichuan2-7b":  "baichuan-inc/Baichuan2-7B-Base",
        # Shanghai AI Lab (CN, academic)
        "internlm2.5-7b": "internlm/internlm2_5-7b",
        # 01.AI (CN, corporate)
        "yi-9b":         "01-ai/Yi-9B",
        # EleutherAI (US, nonprofit)
        "pythia-6.9b":   "EleutherAI/pythia-6.9b",
        # DeepSeek (CN, corporate)
        "deepseek-7b":   "deepseek-ai/deepseek-llm-7b-base",
        # LLM360 (US, academic)
        "amber-7b":      "LLM360/Amber",
        # HuggingFace (US, corporate)
        "smollm2-360m":  "HuggingFaceTB/SmolLM2-360M",
        "smollm3-3b":    "HuggingFaceTB/SmolLM3-3B-Base",
    }

    @classmethod
    def resolve(cls, name: str) -> str:
        """Resolve family name ("olmo3-7b") or pass through HuggingFace ID."""
        if "/" in name:
            return name
        if name in cls.FAMILIES:
            return cls.FAMILIES[name]
        raise ValueError(
            f"Unknown family: {name}. "
            f"Available: {', '.join(sorted(cls.FAMILIES.keys()))}")

    @classmethod
    def families(cls) -> pd.DataFrame:
        """All known families with metadata."""
        from .registry import Registry
        reg = Registry()
        rows = []
        for name, base_id in sorted(cls.FAMILIES.items()):
            info = reg.info(base_id)
            rows.append({
                "family": name,
                "base_model": base_id,
                "variants": len(reg.variants_of(base_id)),
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
                     temperature: float = 0.8,
                     prompts: dict = None):
        """Collect data for a base model and all its variants."""
        from .registry import Registry
        base_id = cls.resolve(name)
        reg = Registry()
        models = [base_id] + reg.variants_of(base_id)
        print(f"[Probe] Collecting {len(models)} models from {base_id}")
        for model_id in models:
            p = Probe(model_id)
            p.collect(n=n, max_tokens=max_tokens, temperature=temperature,
                      prompts=prompts)

    @classmethod
    def compare_tree(cls, name: str, prompt: str,
                     gen: int = 0, pos: int = 0) -> pd.DataFrame:
        """Compare all variants of a base model on one prompt."""
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
                f"resistance_{base_token}": base_token_surprisal(
                    base_logits, logits),
                "entropy": entropy(logits),
                "argmax": tok.decode([int(np.argmax(logits))]).strip(),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def inventory() -> pd.DataFrame:
        """All models with stored probe data."""
        cache = _get_cache()
        s = cache._stash("probe_logits")
        models = set()
        for key in s:
            if isinstance(key, dict) and "model" in key:
                models.add(key["model"])
        rows = []
        for model_id in sorted(models):
            prompts = [p for p in PROMPTS
                       if cache.has_probe(model_id, p, gen=0, pos=0)]
            n = sum(cache.count_probe_gens(model_id, p) for p in prompts)
            if prompts:
                rows.append({
                    "model_id": model_id,
                    "prompts": prompts,
                    "total_gens": n,
                })
        return pd.DataFrame(rows)
