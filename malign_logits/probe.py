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

# Reverse lookup: text → name (all sources)
_PROMPT_NAMES = {v: k for k, v in PROMPTS.items()}
try:
    from .experiments import DEFAULT_PROMPTS as _DP, INSTITUTIONAL_PROMPTS as _IP
    _PROMPT_NAMES.update({v: k for k, v in _DP.items()})
    _PROMPT_NAMES.update({v: k for k, v in _IP.items()})
except ImportError:
    pass


def _resolve_prompt(prompt: str) -> str:
    """Resolve a prompt name to text, or pass through text directly.

    Accepts either:
        "anger"                              → "She was so angry she wanted to"
        "death_1"                            → "The doctor told her she had six months to"
        "She was so angry she wanted to"     → "She was so angry she wanted to"
        "Any arbitrary prompt"               → "Any arbitrary prompt"
    """
    if prompt in PROMPTS:
        return PROMPTS[prompt]
    if prompt in _ALL_PROMPTS:
        return _ALL_PROMPTS[prompt]
    return prompt


def _get_all_prompts():
    """Lazily load all prompt dictionaries."""
    try:
        from .experiments import DEFAULT_PROMPTS, INSTITUTIONAL_PROMPTS
        return {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}
    except ImportError:
        return {}


_ALL_PROMPTS = _get_all_prompts()


def _prompt_name(prompt_text: str) -> str:
    """Get friendly name for a prompt text, or return text itself."""
    return _PROMPT_NAMES.get(prompt_text, prompt_text)


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
                temperature: float = 1.0,
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
            display = _prompt_name(prompt_text) if prompt_text in _PROMPT_NAMES else prompt_key
            print(f"  {display}:", end="", flush=True)
            collected = 0

            for gen_id in range(n):
                if cache.has_probe(store_id, prompt_text,
                                   gen=gen_id, pos=0, max_tokens=max_tokens):
                    print(".", end="", flush=True)
                    continue

                self._run_generation(
                    prompt_text=prompt_text,
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

    def _run_generation(self, prompt_text, gen_id,
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
                store_id, prompt_text,
                raw_logits.cpu().numpy(), gen=gen_id, pos=step,
                max_tokens=max_tokens)

            # Store hidden states at this position
            if out.hidden_states:
                hidden_np = np.stack([
                    h[0, -1, :].cpu().numpy() for h in out.hidden_states
                ])  # (n_layers, hidden_dim)
                cache.set_probe_hidden(
                    store_id, prompt_text,
                    hidden_np, gen=gen_id, pos=step,
                    max_tokens=max_tokens)

            meta_rows.append({
                "position": step,
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
        cache.set_probe_meta(store_id, prompt_text, meta_rows, gen=gen_id,
                            max_tokens=max_tokens)

    def replay(self, prompt: str, gen: int = 0, max_tokens: int = 100):
        """Replay a stored generation through the model in one forward pass.

        Reads the generated text from probe_meta, tokenizes prompt + text,
        runs a single forward pass with output_hidden_states=True, and
        caches logits + hidden at every position.

        This is NOT autoregressive — one forward pass gives all positions.
        ~3s for a 7B model on 100 tokens.
        """
        import gc
        from .models import load_model

        prompt_text = _resolve_prompt(prompt)
        cache = _get_cache()

        # Check if already replayed
        if cache.has_probe(self.model_id, prompt_text, gen=gen, pos=0,
                           max_tokens=max_tokens):
            return

        # Get stored text
        meta = self._meta_get(prompt_text, gen, max_tokens)
        if meta is None:
            raise FileNotFoundError(
                f"No stored text for {self.model_id}/{prompt_text} gen={gen}")

        # Extract the generated text
        if isinstance(meta, list) and len(meta) > 0:
            text = meta[0].get("generated_text") or meta[0].get("chosen_text", "")
        else:
            raise ValueError(f"Meta has no text for {self.model_id}/{prompt_text}")

        print(f"[Probe] Replaying {self.model_id} / {_prompt_name(prompt_text)} gen={gen}...",
              end="", flush=True)
        model, tokenizer = load_model(self.model_id)
        self._tokenizer = tokenizer
        device = next(model.parameters()).device

        # Tokenize full sequence: prompt + generated text
        full_ids = tokenizer.encode(prompt_text + text, return_tensors="pt").to(device)
        prompt_len = len(tokenizer.encode(prompt_text))

        # Single forward pass
        with torch.no_grad():
            out = model(full_ids, output_hidden_states=True)

        # Cache logits + hidden at each generation position
        n_gen = min(full_ids.shape[1] - prompt_len, max_tokens)
        meta_rows = []

        for step in range(n_gen):
            seq_pos = prompt_len + step
            raw_logits = out.logits[0, seq_pos, :].float()
            probs_cpu = torch.softmax(raw_logits, -1).cpu()

            ent = -(probs_cpu * probs_cpu.clamp(min=1e-10).log()).sum().item()
            eff_vocab = int((probs_cpu > 0.001).sum())
            topk = torch.topk(probs_cpu, 10)
            top_tokens = [tokenizer.decode([idx]).strip() for idx in topk.indices]
            top_probs = topk.values.tolist()

            # The actual next token (from the stored generation)
            if seq_pos + 1 < full_ids.shape[1]:
                chosen_id = full_ids[0, seq_pos + 1].item()
            else:
                chosen_id = tokenizer.eos_token_id or 0

            cache.set_probe_logits(
                self.model_id, prompt_text,
                raw_logits.cpu().numpy(), gen=gen, pos=step,
                max_tokens=max_tokens)

            if out.hidden_states:
                hidden_np = np.stack([
                    h[0, seq_pos, :].cpu().numpy() for h in out.hidden_states
                ])
                cache.set_probe_hidden(
                    self.model_id, prompt_text,
                    hidden_np, gen=gen, pos=step,
                    max_tokens=max_tokens)

            meta_rows.append({
                "position": step,
                "prompt_text": prompt_text,
                "generated_text": text,
                "entropy": ent,
                "eff_vocab": eff_vocab,
                "argmax_token": top_tokens[0],
                "argmax_prob": top_probs[0],
                "chosen_token": tokenizer.decode([chosen_id]).strip(),
                "chosen_token_id": chosen_id,
                "chosen_prob": probs_cpu[chosen_id].item() if chosen_id < len(probs_cpu) else 0.0,
                "top5_tokens": "|".join(top_tokens[:5]),
                "top5_probs": "|".join(f"{p:.6f}" for p in top_probs[:5]),
            })

        # Overwrite meta with full per-position data
        cache.set_probe_meta(self.model_id, prompt_text, meta_rows,
                            gen=gen, max_tokens=max_tokens)

        del model
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        print(f" {n_gen} positions cached")

    @classmethod
    def replay_all(cls, prompts: dict = None, gen: int = 0,
                   max_tokens: int = 100):
        """Replay gen=0 for all registered models on canonical prompts.

        Single forward pass per model×prompt. Loads each model once,
        replays all prompts, frees memory.
        """
        import gc
        from .registry import Registry
        from .models import load_model

        prompts = prompts or PROMPTS
        reg = Registry()
        cache = _get_cache()

        models = sorted(reg.models())
        print(f"[Probe] Replaying {len(models)} models × {len(prompts)} prompts")

        for model_id in models:
            # Check if any prompts need replaying
            needs = []
            for name, text in prompts.items():
                if not cache.has_probe(model_id, text, gen=gen, pos=0,
                                       max_tokens=max_tokens):
                    # Check if we have stored text to replay
                    meta_key = {"model": model_id, "prompt": text,
                                "gen": gen, "T": max_tokens}
                    if meta_key in cache._stash("probe_meta"):
                        needs.append((name, text))

            if not needs:
                continue

            try:
                print(f"  {model_id}: ", end="", flush=True)
                model, tokenizer = load_model(model_id)
                device = next(model.parameters()).device

                for name, prompt_text in needs:
                    meta = cache._stash("probe_meta")[
                        {"model": model_id, "prompt": prompt_text,
                         "gen": gen, "T": max_tokens}]
                    text = meta[0].get("generated_text", "")
                    if not text:
                        continue

                    full_ids = tokenizer.encode(
                        prompt_text + text, return_tensors="pt").to(device)
                    prompt_len = len(tokenizer.encode(prompt_text))

                    with torch.no_grad():
                        out = model(full_ids, output_hidden_states=True)

                    n_gen = min(full_ids.shape[1] - prompt_len, max_tokens)
                    meta_rows = []

                    for step in range(n_gen):
                        seq_pos = prompt_len + step
                        raw_logits = out.logits[0, seq_pos, :].float()
                        probs_cpu = torch.softmax(raw_logits, -1).cpu()

                        ent = -(probs_cpu * probs_cpu.clamp(min=1e-10).log()).sum().item()
                        topk = torch.topk(probs_cpu, 10)
                        top_tokens = [tokenizer.decode([idx]).strip()
                                      for idx in topk.indices]
                        top_probs = topk.values.tolist()

                        chosen_id = (full_ids[0, seq_pos + 1].item()
                                     if seq_pos + 1 < full_ids.shape[1]
                                     else tokenizer.eos_token_id or 0)

                        cache.set_probe_logits(
                            model_id, prompt_text,
                            raw_logits.cpu().numpy(), gen=gen, pos=step,
                            max_tokens=max_tokens)

                        if out.hidden_states:
                            hidden_np = np.stack([
                                h[0, seq_pos, :].cpu().numpy()
                                for h in out.hidden_states])
                            cache.set_probe_hidden(
                                model_id, prompt_text,
                                hidden_np, gen=gen, pos=step,
                                max_tokens=max_tokens)

                        meta_rows.append({
                            "position": step, "prompt_text": prompt_text,
                            "generated_text": text, "entropy": ent,
                            "eff_vocab": int((probs_cpu > 0.001).sum()),
                            "argmax_token": top_tokens[0],
                            "argmax_prob": top_probs[0],
                            "chosen_token": tokenizer.decode([chosen_id]).strip(),
                            "chosen_token_id": chosen_id,
                            "chosen_prob": probs_cpu[chosen_id].item()
                            if chosen_id < len(probs_cpu) else 0.0,
                            "top5_tokens": "|".join(top_tokens[:5]),
                            "top5_probs": "|".join(f"{p:.6f}" for p in top_probs[:5]),
                        })

                    cache.set_probe_meta(model_id, prompt_text, meta_rows,
                                        gen=gen, max_tokens=max_tokens)
                    print(f"{name}+ ", end="", flush=True)

                del model
                gc.collect()
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
                print()

            except Exception as e:
                print(f"FAILED: {e}")

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

        prompt_text = _resolve_prompt(prompt)
        source = Probe(token_source)
        source_meta = source.meta(prompt_text, gen=gen, max_tokens=max_tokens)
        token_ids = source_meta["chosen_token_id"].tolist()
        T = len(token_ids)

        source_short = token_source.split("/")[-1]
        tf_prompt = f"{prompt_text}::tf_{source_short}"
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
        """Find max_tokens for a given (prompt, gen)."""
        prompt = _resolve_prompt(prompt)
        if max_tokens is not None:
            return max_tokens
        cache = _get_cache()
        for T in [100, 50, 20, 10, 5]:
            if cache.has_probe(self.model_id, prompt, gen=gen, pos=0, max_tokens=T):
                return T
        return None

    def _probe_get(self, stash_name, prompt, gen, pos, max_tokens):
        """Get from probe stash."""
        prompt = _resolve_prompt(prompt)
        cache = _get_cache()
        T = self._resolve_T(prompt, gen, max_tokens)
        if T is None:
            return None
        key = {"model": self.model_id, "prompt": prompt,
               "gen": gen, "pos": pos, "T": T}
        s = cache._stash(stash_name)
        return s[key] if key in s else None

    def _meta_get(self, prompt, gen, max_tokens):
        """Get meta from probe stash."""
        prompt = _resolve_prompt(prompt)
        cache = _get_cache()
        T = self._resolve_T(prompt, gen, max_tokens)
        if T is None:
            return None
        key = {"model": self.model_id, "prompt": prompt, "gen": gen, "T": T}
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
        prompt = _resolve_prompt(prompt)
        T = self._resolve_T(prompt, 0, max_tokens)
        if T is None:
            return 0
        return _get_cache().count_probe_gens(self.model_id, prompt,
                                              max_tokens=T)

    def prompts(self, max_tokens: int = None) -> list:
        """Prompts with data. Returns friendly names where available."""
        result = []
        for name, text in PROMPTS.items():
            if self._resolve_T(text, 0, max_tokens) is not None:
                result.append(name)
        return result

    def tree(self, prompt: str, n_gens: int = 100,
             max_tokens: int = 10) -> dict:
        """Tree-level metrics: branch distribution, entropy, counts."""
        from .metrics import tree_metrics
        return tree_metrics(self, _resolve_prompt(prompt), n_gens, max_tokens)

    def tree_compare(self, other_model: str, prompt: str,
                     n_gens: int = 100, max_tokens: int = 10) -> dict:
        """Compare tree structures with another model."""
        from .metrics import tree_compare
        return tree_compare(self, Probe(other_model),
                           _resolve_prompt(prompt), n_gens, max_tokens)

    def branches(self, prompt: str, max_tokens: int = 100,
                 depth: int = 1) -> dict:
        """Tree view: group generations by first token(s).

        Returns dict mapping first token(s) → list of gen indices.
        depth=1: group by first token. depth=2: first two tokens. etc.
        """
        prompt_text = _resolve_prompt(prompt)
        cache = _get_cache()
        gen_stash = cache._stash('generations')

        branches = {}
        idx = 0
        while True:
            text = cache.get_generation(self.model_id, prompt_text, temp=1.0, idx=idx)
            if text is None:
                break
            words = text.strip().split()[:depth]
            key = " ".join(words) if words else "?"
            if key not in branches:
                branches[key] = []
            branches[key].append(idx)
            idx += 1

        return dict(sorted(branches.items(), key=lambda x: -len(x[1])))

    def branch_logits(self, prompt: str, branch_token: str,
                      pos: int = 1) -> np.ndarray:
        """Logits at position `pos` for a generation that starts with `branch_token`.

        Finds the first generation starting with that token and replays
        through the model. At position 1+, the logits are conditioned on
        the branch — different first tokens yield different logits.

        For cross-model comparison, use the same branch_token on both models
        (via teacher_force or replay) for a clean path-matched comparison.
        """
        prompt_text = _resolve_prompt(prompt)
        cache = _get_cache()

        # Find a gen that starts with this token
        idx = 0
        while True:
            text = cache.get_generation(self.model_id, prompt_text, temp=1.0, idx=idx)
            if text is None:
                raise FileNotFoundError(
                    f"No generation starting with '{branch_token}' for "
                    f"{self.model_id}/{prompt_text}")
            first_word = text.strip().split()[0] if text.strip() else ""
            if first_word == branch_token:
                return self.logits(prompt_text, gen=idx, pos=pos,
                                   max_tokens=100)
            idx += 1

    def branch_text(self, prompt: str, branch_token: str,
                    n: int = 5) -> list:
        """Get up to n generation texts that start with a given token."""
        prompt_text = _resolve_prompt(prompt)
        cache = _get_cache()
        results = []
        idx = 0
        while len(results) < n:
            text = cache.get_generation(self.model_id, prompt_text, temp=1.0, idx=idx)
            if text is None:
                break
            first_word = text.strip().split()[0] if text.strip() else ""
            if first_word == branch_token:
                results.append(text)
            idx += 1
        return results

    def distance(self, other_model: str, prompt: str,
                 gen_a: int = 0, gen_b: int = 0,
                 n_positions: int = 50) -> dict:
        """All distance measures between this model's generation and another's.

        Returns token Jaccard, bag-of-logits JS, mean position JS,
        hidden centroid distance.
        """
        from .metrics import generation_distance
        return generation_distance(self, Probe(other_model), prompt,
                                   gen_a=gen_a, gen_b=gen_b,
                                   n_positions=n_positions)

    def text_metrics(self, prompt: str, gen: int = 0,
                     embedder: str = "BAAI/bge-m3") -> dict:
        """Text-level metrics: drift, embedding. Uses bge-m3."""
        from .metrics import generation_text_metrics
        return generation_text_metrics(self, prompt, gen=gen, embedder=embedder)

    def text_distance(self, other_model: str, prompt: str,
                      gen_a: int = 0, gen_b: int = 0,
                      embedder: str = "BAAI/bge-m3") -> dict:
        """Text-level distance via sentence embeddings (bge-m3)."""
        from .metrics import cross_generation_text_distance
        return cross_generation_text_distance(
            self, Probe(other_model), prompt,
            gen_a=gen_a, gen_b=gen_b, embedder=embedder)

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

    # -- cross-family aggregation -----------------------------------------------

    @staticmethod
    def census(prompt: str = "anger", n_gens: int = 100,
               max_tokens: int = 10) -> 'pd.DataFrame':
        """Full census: tree + logit metrics for all models with data.

        Returns one row per model with tree stats, position-0 logits,
        and metadata (org, country, relation).
        """
        import pandas as pd
        from .metrics import tree_metrics, entropy as _entropy
        from .registry import Registry

        reg = Registry()
        cache = _get_cache()
        prompt_text = _resolve_prompt(prompt)

        rows = []
        s = cache._stash('probe_meta')
        models = set()
        for key in s:
            if (isinstance(key, dict) and key.get('T') == max_tokens
                and key.get('prompt') == prompt_text and key.get('gen') == 0):
                models.add(key['model'])

        for model_id in sorted(models):
            p = Probe(model_id)
            info = reg.info(model_id)
            base_id = reg.base_of(model_id)
            _, rel = reg.parent_of(model_id)

            t = tree_metrics(p, prompt_text, n_gens, max_tokens)
            if not t or t['n_gens'] < 5:
                continue

            row = {
                'model': model_id,
                'model_short': model_id.split('/')[-1],
                'family': base_id.split('/')[-1] if base_id else '',
                'relation': rel or 'base',
                'org': info.org if info else '',
                'country': info.country if info else '',
                'org_type': info.org_type if info else '',
                'n_gens': t['n_gens'],
                'n_branches': t['n_branches'],
                'branch_entropy': t['branch_entropy'],
                'top_branch': t['top_branch'],
                'top_branch_pct': t['top_branch_pct'],
            }

            try:
                logits = p.logits(prompt_text, gen=0, pos=0, max_tokens=max_tokens)
                row['entropy_pos0'] = _entropy(logits)
            except FileNotFoundError:
                pass

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def census_compare(prompt: str = "anger", n_gens: int = 100,
                       max_tokens: int = 10) -> 'pd.DataFrame':
        """Cross-family comparison: base→aligned tree + logit metrics.

        Returns one row per base→variant pair.
        """
        import pandas as pd
        from .metrics import tree_compare, js_divergence, base_token_surprisal
        from .registry import Registry

        reg = Registry()
        cache = _get_cache()
        prompt_text = _resolve_prompt(prompt)

        models = set()
        s = cache._stash('probe_meta')
        for key in s:
            if (isinstance(key, dict) and key.get('T') == max_tokens
                and key.get('prompt') == prompt_text and key.get('gen') == 0):
                models.add(key['model'])

        rows = []
        for base_id in reg.all_bases():
            if base_id not in models:
                continue
            base_p = Probe(base_id)
            try:
                base_logits = base_p.logits(prompt_text, gen=0, pos=0,
                                            max_tokens=max_tokens)
            except FileNotFoundError:
                continue

            for v in reg.variants_of(base_id):
                if v not in models:
                    continue
                v_p = Probe(v)
                _, rel = reg.parent_of(v)

                try:
                    v_logits = v_p.logits(prompt_text, gen=0, pos=0,
                                          max_tokens=max_tokens)
                    js = js_divergence(base_logits, v_logits)
                    resist = base_token_surprisal(base_logits, v_logits)
                except FileNotFoundError:
                    js = resist = np.nan

                tc = tree_compare(base_p, v_p, prompt_text, n_gens, max_tokens)

                rows.append({
                    'base': base_id,
                    'variant': v,
                    'variant_short': v.split('/')[-1],
                    'relation': rel,
                    'js': js,
                    'resistance': resist,
                    'tree_js': tc.get('tree_js', np.nan) if tc else np.nan,
                    'branches_base': tc.get('n_branches_a', 0) if tc else 0,
                    'branches_aligned': tc.get('n_branches_b', 0) if tc else 0,
                    'H_base': tc.get('branch_entropy_a', 0) if tc else 0,
                    'H_aligned': tc.get('branch_entropy_b', 0) if tc else 0,
                    'n_repressed': len(tc.get('repressed', {})) if tc else 0,
                    'n_amplified': len(tc.get('amplified', {})) if tc else 0,
                })

        return pd.DataFrame(rows)

    def explore_tree(self, prompt: str, coverage: float = 0.5,
                     max_depth: int = 5,
                     path_threshold: float = 0.01,
                     cumul_floor: float = 0.001, max_nodes: int = 5000) -> list:
        """Deterministic tree exploration using path probability.

        Includes any branch where the product of probabilities along
        the path from root exceeds path_threshold (default 0.5%).
        This follows the likely storylines to natural depth.

        Returns list of node dicts with: depth, token, token_id, prob,
        path_prob, entropy, parent, n_children, hidden.
        """
        import gc
        from .models import load_model

        prompt_text = _resolve_prompt(prompt)
        cache = _get_cache()
        tree_key = {"model": self.model_id, "prompt": prompt_text,
                    "path_threshold": path_threshold,
                    "max_depth": max_depth,
                    "type": "explore_tree_v2"}
        cached = cache.get_derived(tree_key)
        if cached is not None:
            return cached

        model, tokenizer = load_model(self.model_id)
        self._tokenizer = tokenizer
        device = next(model.parameters()).device
        nodes = self._explore_tree_with_model(
            prompt_text, model, tokenizer, device,
            path_threshold=path_threshold, max_depth=max_depth,
            max_nodes=max_nodes, cache=cache)

        del model
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        return nodes

    def _explore_tree_with_model(self, prompt_text, model, tokenizer, device,
                                  path_threshold=0.005, max_depth=5,
                                  max_nodes=5000, cache=None):
        """Core tree exploration with a pre-loaded model. No load/unload."""
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        queue = [(0, prompt_ids, -1, "ROOT", 1.0, 1.0, -1)]
        nodes = []

        from scipy.special import softmax as _sfm
        print(f"[Probe] Exploring tree: {self.model_id} / {prompt_text[:40]}", end="", flush=True)

        kv_store = {}

        while queue and len(nodes) < max_nodes:
            depth, ids, parent_idx, token_str, local_prob, path_prob, token_id = queue.pop(0)
            if depth > max_depth:
                continue

            with torch.no_grad():
                if depth == 0:
                    out = model(ids, use_cache=True)
                elif parent_idx in kv_store:
                    new_id = torch.tensor([[token_id]], device=device)
                    out = model(new_id, past_key_values=kv_store[parent_idx],
                               use_cache=True)
                else:
                    out = model(ids, use_cache=True)

            logits = out.logits[0, -1, :].float().cpu().numpy()
            probs = _sfm(logits)

            ent = -float(np.sum(probs * np.log(probs + 1e-10)))
            node_idx = len(nodes)
            node = {
                "depth": depth, "token": token_str, "token_id": token_id,
                "prob": local_prob, "path_prob": path_prob, "entropy": ent,
                "parent": parent_idx, "n_children": 0,
            }
            node["_logits"] = logits
            # Axis loadings and hidden states computed on demand from cached logits,
            # not during exploration (axis_loading is 977ms/node on CPU)
            nodes.append(node)
            if parent_idx >= 0:
                nodes[parent_idx]["n_children"] += 1

            if depth < max_depth:
                sorted_idx = np.argsort(probs)[::-1]
                children_added = False
                for tid in sorted_idx:
                    p_val = float(probs[tid])
                    child_pp = path_prob * p_val if depth > 0 else p_val
                    if child_pp < path_threshold:
                        break
                    word = tokenizer.decode([int(tid)]).strip()
                    new_ids = torch.cat([ids, torch.tensor([[int(tid)]], device=device)], dim=-1)
                    queue.append((depth + 1, new_ids, node_idx, word, p_val, child_pp, int(tid)))
                    children_added = True

                if children_added:
                    kv_store[node_idx] = out.past_key_values

            # Free KV caches for nodes whose children are all processed
            if parent_idx in kv_store:
                parent_children_remaining = any(
                    d > depth and pi == parent_idx
                    for d, _, pi, *_ in queue
                )
                if not parent_children_remaining:
                    del kv_store[parent_idx]

            if len(nodes) % 500 == 0:
                print(".", end="", flush=True)

        kv_store.clear()

        print(f" {len(nodes)} nodes")
        if cache:
            tree_key = {"model": self.model_id, "prompt": prompt_text,
                        "path_threshold": path_threshold, "max_depth": max_depth,
                        "type": "explore_tree_v2"}
            # HashStash stores numpy arrays directly (11x faster than .tolist())
            cache.set_derived(tree_key, nodes)
        return nodes

    def annotate_tree(self, prompt: str, annotators: list = None,
                      coverage: float = 0.5, max_depth: int = 5):
        """Teacher-force family checkpoints through this model's tree.

        Explores this model's tree, then replays each path through
        every annotator model. Adds per-node columns:
            {model_short}_entropy, {model_short}_prob_of_base_token,
            {model_short}_resistance, {model_short}_argmax,
            {model_short}_hidden

        If annotators is None, uses all variants of this model's base.

            base = Probe("allenai/OLMo-2-0425-1B")
            nodes = base.annotate_tree("anger")
            # Each node has base + SFT + DPO + RLVR columns
        """
        import gc
        from .models import load_model
        from .registry import Registry
        from scipy.special import softmax

        prompt_text = _resolve_prompt(prompt)
        reg = Registry()

        # Get or explore the base tree
        nodes = self.explore_tree(prompt_text, coverage=coverage,
                                  max_depth=max_depth)

        # Determine annotator models
        if annotators is None:
            base_id = reg.base_of(self.model_id) or self.model_id
            annotators = reg.variants_of(base_id)
            if self.model_id != base_id:
                annotators = [base_id] + annotators
            annotators = [m for m in annotators if m != self.model_id]

        if not annotators:
            return nodes

        # Skip already-annotated models
        existing = set()
        if nodes:
            for key in nodes[0].keys():
                if key.endswith("_js"):
                    existing.add(key[:-3])
        annotators = [m for m in annotators
                      if m.split("/")[-1].replace("-", "_")[:20] not in existing]
        if not annotators:
            return nodes

        # Build all unique paths (root to each node)
        def get_path(node_idx):
            path = []
            idx = node_idx
            while idx >= 0:
                path.append(idx)
                idx = nodes[idx]["parent"]
            return list(reversed(path))

        # Get tokenizer for token ID lookup
        tok = self.tokenizer
        prompt_ids = tok.encode(prompt_text)

        for model_id in annotators:
            short = model_id.split("/")[-1].replace("-", "_")[:20]
            print(f"  Annotating with {model_id.split('/')[-1]}...", end="", flush=True)

            try:
                model, model_tok = load_model(model_id)
                device = next(model.parameters()).device

                # For each node, build its token path and forward pass
                annotated = 0
                # Group nodes by depth for efficiency
                for node_idx, node in enumerate(nodes):
                    if node["depth"] == 0:
                        # Root: just the prompt
                        ids = tok.encode(prompt_text, return_tensors="pt").to(device)
                    else:
                        # Build token sequence: prompt + path tokens
                        path_indices = get_path(node_idx)
                        path_tokens = [nodes[i]["token"] for i in path_indices
                                       if nodes[i]["depth"] > 0]
                        text = prompt_text + " " + " ".join(path_tokens)
                        ids = tok.encode(text, return_tensors="pt").to(device)

                    with torch.no_grad():
                        out = model(ids, output_hidden_states=True)

                    logits = out.logits[0, -1, :].float().cpu().numpy()
                    probs = softmax(logits)

                    ent = -float(np.sum(probs * np.log(probs + 1e-10)))
                    argmax_id = int(np.argmax(probs))
                    argmax_word = tok.decode([argmax_id]).strip()

                    # Resistance: for each child of this node, how much does
                    # the aligned model resist that child's token?
                    # Stored on the CHILD node as self_resist (resistance to
                    # THIS token, not to this token's children).
                    children = [n for n in nodes if n["parent"] == node_idx]
                    child_prob = 0.0
                    resistance = 0.0
                    delta_resistance = 0.0
                    for child in children:
                        # Use stored token_id (correct for same-tokenizer families).
                        # Fall back to text lookup for cross-tokenizer annotators.
                        child_tid = child.get("token_id", -1)
                        if child_tid < 0 or child_tid >= len(probs):
                            child_text = child.get("token", "")
                            child_tids = model_tok.encode(" " + child_text, add_special_tokens=False)
                            child_tid = child_tids[0] if child_tids else -1
                        if child_tid >= 0 and child_tid < len(probs):
                            cp = float(probs[child_tid])
                            bp = child["prob"]
                            r = -float(np.log2(max(cp, 1e-10)))
                            br = -float(np.log2(max(bp, 1e-10)))
                            dr = r - br
                            child[f"{short}_self_resist"] = dr
                            child[f"{short}_self_prob"] = cp
                    # Keep top-child metrics for backward compat
                    if children:
                        top_child = max(children, key=lambda c: c["prob"])
                        child_tid = top_child.get("token_id", -1)
                        if child_tid >= 0 and child_tid < len(probs):
                            child_prob = float(probs[child_tid])
                            base_child_prob = top_child["prob"]
                            resistance = -float(np.log2(max(child_prob, 1e-10)))
                            base_resistance = -float(np.log2(max(base_child_prob, 1e-10)))
                            delta_resistance = resistance - base_resistance

                    # JS divergence + top movers: full distributional comparison
                    base_logits = node.get("_logits")
                    node_js = 0.0
                    top_gained = ""
                    top_lost = ""
                    abs_resistance = 0.0

                    if base_logits is not None:
                        from .metrics import js_divergence as _js, _align_vocab
                        bl, al = _align_vocab(np.array(base_logits), logits)
                        node_js = _js(bl, al)

                        # Top movers: which tokens gained/lost most?
                        from scipy.special import softmax as _sfm2
                        p_base = _sfm2(bl)
                        p_ann = _sfm2(al)
                        delta = p_ann - p_base
                        top5_gained = np.argsort(delta)[-3:][::-1]
                        top5_lost = np.argsort(delta)[:3]
                        top_gained = "|".join(
                            f"{tok.decode([int(i)]).strip()}({delta[i]:+.3f})"
                            for i in top5_gained if abs(delta[i]) > 0.005)
                        top_lost = "|".join(
                            f"{tok.decode([int(i)]).strip()}({delta[i]:+.3f})"
                            for i in top5_lost if abs(delta[i]) > 0.005)

                        # Absolute resistance: bits for base argmax under annotator
                        base_argmax = int(np.argmax(p_base))
                        abs_resistance = -float(np.log2(max(p_ann[base_argmax], 1e-10)))

                        # Reverse resistance: how alien is aligned's choice to base?
                        ann_argmax = int(np.argmax(p_ann))
                        base_prob_of_ann_choice = float(p_base[ann_argmax])
                        ann_prob_of_ann_choice = float(p_ann[ann_argmax])
                        reverse_resist = (
                            -float(np.log2(max(base_prob_of_ann_choice, 1e-10)))
                            - (-float(np.log2(max(ann_prob_of_ann_choice, 1e-10))))
                        )
                        node[f"{short}_reverse_resist"] = reverse_resist
                        node[f"{short}_ann_choice"] = tok.decode([ann_argmax]).strip()
                        node[f"{short}_ann_choice_base_prob"] = base_prob_of_ann_choice

                    node[f"{short}_entropy"] = ent
                    node[f"{short}_argmax"] = argmax_word
                    node[f"{short}_js"] = node_js
                    node[f"{short}_entropy_delta"] = ent - node["entropy"]
                    node[f"{short}_abs_resistance"] = abs_resistance
                    node[f"{short}_resistance"] = resistance
                    node[f"{short}_delta_resist"] = delta_resistance
                    node[f"{short}_prob_child"] = child_prob
                    node[f"{short}_top_gained"] = top_gained
                    node[f"{short}_top_lost"] = top_lost

                    if out.hidden_states:
                        h_ann = out.hidden_states[-1][0, -1, :].cpu().numpy()
                        node[f"{short}_hidden"] = h_ann

                        # Hidden distance: how differently does this model
                        # represent this point vs the base tree's model?
                        if "hidden" in node:
                            h_base = np.array(node["hidden"])
                            nb = np.linalg.norm(h_base)
                            na = np.linalg.norm(h_ann)
                            if nb > 1e-10 and na > 1e-10:
                                cos_dist = 1.0 - float(np.dot(h_base, h_ann) / (nb * na))
                            else:
                                cos_dist = 1.0
                            node[f"{short}_hidden_dist"] = cos_dist

                    annotated += 1

                del model
                gc.collect()
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass

                print(f" {annotated} nodes")

            except Exception as e:
                print(f" FAILED: {str(e)[:60]}")

        # Update cache with annotated tree (same key as explore_tree)
        cache = _get_cache()
        tree_key = {"model": self.model_id, "prompt": prompt_text,
                    "path_threshold": 0.01, "max_depth": max_depth,
                    "type": "explore_tree_v2"}
        cache.set_derived(tree_key, nodes)

        return nodes

    def batch_annotate(self, prompts: dict = None, max_depth: int = 5):
        """Annotate multiple prompts efficiently — one model load per annotator.

        Instead of loading each annotator N times (once per prompt),
        loads each annotator once and processes all prompts.

            base = Probe("allenai/OLMo-2-0425-1B")
            results = base.batch_annotate()  # all prompts, all annotators
            # results = {prompt_name: [nodes], ...}
        """
        import gc
        from .models import load_model
        from .registry import Registry
        from scipy.special import softmax

        reg = Registry()
        if prompts is None:
            prompts = PROMPTS

        # Step 1: explore all trees — load base model once for all uncached
        trees = {}
        uncached = {}
        cache = _get_cache()
        for pname, ptext in prompts.items():
            tree_key = {"model": self.model_id, "prompt": ptext,
                        "path_threshold": 0.01, "max_depth": max_depth,
                        "type": "explore_tree_v2"}
            cached = cache.get_derived(tree_key)
            if cached is not None:
                trees[pname] = cached
            else:
                uncached[pname] = ptext
        from tqdm import tqdm
        short_name = self.model_id.split("/")[-1]

        if uncached:
            model_base, tok_base = load_model(self.model_id)
            self._tokenizer = tok_base
            device = next(model_base.parameters()).device
            for pname, ptext in tqdm(uncached.items(),
                                     desc=f"{short_name} explore",
                                     unit="prompt"):
                trees[pname] = self._explore_tree_with_model(
                    ptext, model_base, tok_base, device,
                    max_depth=max_depth, cache=cache)
            del model_base
            gc.collect()
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

        # Determine annotators
        base_id = reg.base_of(self.model_id) or self.model_id
        annotators = reg.variants_of(base_id)
        if self.model_id != base_id:
            annotators = [base_id] + annotators
        annotators = [m for m in annotators if m != self.model_id]

        if not annotators:
            return trees

        # Check which annotators are already done (on first prompt)
        first_nodes = list(trees.values())[0]
        existing = set()
        if first_nodes:
            for key in first_nodes[0].keys():
                if key.endswith("_js"):
                    existing.add(key[:-3])
        annotators = [m for m in annotators
                      if m.split("/")[-1].replace("-", "_")[:20] not in existing]
        if not annotators:
            return trees

        tok = self.tokenizer

        # Step 2: for each annotator, load ONCE, annotate ALL prompts
        for ann_idx, model_id in enumerate(annotators):
            short = model_id.split("/")[-1].replace("-", "_")[:20]
            ann_label = model_id.split("/")[-1]

            try:
                model, model_tok = load_model(model_id)
                device = next(model.parameters()).device

                total_nodes = 0
                pbar = tqdm(prompts.items(),
                            desc=f"{short_name} ← {ann_label} [{ann_idx+1}/{len(annotators)}]",
                            unit="prompt")
                for pname, ptext in pbar:
                    nodes = trees[pname]

                    # Group nodes by parent for KV cache reuse
                    children_of = {}
                    for ni, n in enumerate(nodes):
                        children_of.setdefault(n["parent"], []).append(ni)

                    # Process tree depth-first with KV caching
                    # kv_cache[node_idx] = past_key_values after processing that node
                    kv_cache = {}
                    node_results = {}  # node_idx → (logits, probs, ent, out)

                    def process_node(node_idx):
                        node = nodes[node_idx]
                        parent = node["parent"]

                        if node["depth"] == 0:
                            ids = tok.encode(ptext, return_tensors="pt").to(device)
                            with torch.no_grad():
                                out = model(ids, output_hidden_states=True,
                                           use_cache=True)
                            kv_cache[node_idx] = out.past_key_values
                        else:
                            tid = node.get("token_id", -1)
                            if tid < 0:
                                tids = model_tok.encode(" " + node["token"],
                                                        add_special_tokens=False)
                                tid = tids[0] if tids else 0
                            new_id = torch.tensor([[tid]], device=device)

                            if parent in kv_cache:
                                with torch.no_grad():
                                    out = model(new_id,
                                               past_key_values=kv_cache[parent],
                                               output_hidden_states=True,
                                               use_cache=True)
                                kv_cache[node_idx] = out.past_key_values
                            else:
                                # Fallback: encode full path
                                def get_path(ni):
                                    path = []
                                    idx = ni
                                    while idx >= 0:
                                        path.append(idx)
                                        idx = nodes[idx]["parent"]
                                    return list(reversed(path))
                                path_indices = get_path(node_idx)
                                path_tokens = [nodes[i]["token"] for i in path_indices
                                               if nodes[i]["depth"] > 0]
                                text = ptext + " " + " ".join(path_tokens)
                                ids = tok.encode(text, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    out = model(ids, output_hidden_states=True,
                                               use_cache=True)
                                kv_cache[node_idx] = out.past_key_values

                        logits = out.logits[0, -1, :].float().cpu().numpy()
                        probs = softmax(logits)
                        node_results[node_idx] = (logits, probs, out)

                        # Process children (depth-first to reuse cache)
                        for child_idx in children_of.get(node_idx, []):
                            process_node(child_idx)

                        # Free cache if no more children need it
                        if node_idx in kv_cache and node_idx not in [
                            nodes[ci]["parent"] for ci in range(node_idx + 1, len(nodes))
                            if ci not in node_results
                        ]:
                            del kv_cache[node_idx]

                    process_node(0)

                    # Now extract metrics from results
                    for node_idx, node in enumerate(nodes):
                        if node_idx not in node_results:
                            continue
                        logits, probs, out = node_results[node_idx]
                        ent = -float(np.sum(probs * np.log(probs + 1e-10)))
                        ent = -float(np.sum(probs * np.log(probs + 1e-10)))
                        argmax_id = int(np.argmax(probs))
                        argmax_word = tok.decode([argmax_id]).strip()

                        # Self resist for all children
                        children = [n for n in nodes if n["parent"] == node_idx]
                        for child in children:
                            child_tid = child.get("token_id", -1)
                            if child_tid < 0 or child_tid >= len(probs):
                                child_text = child.get("token", "")
                                child_tids = model_tok.encode(" " + child_text,
                                                              add_special_tokens=False)
                                child_tid = child_tids[0] if child_tids else -1
                            if child_tid >= 0 and child_tid < len(probs):
                                cp = float(probs[child_tid])
                                bp = child["prob"]
                                r = -float(np.log2(max(cp, 1e-10)))
                                br = -float(np.log2(max(bp, 1e-10)))
                                child[f"{short}_self_resist"] = r - br
                                child[f"{short}_self_prob"] = cp

                        # Top-child resistance (backward compat)
                        child_prob = resistance = delta_resistance = 0.0
                        if children:
                            top_child = max(children, key=lambda c: c["prob"])
                            child_tid = top_child.get("token_id", -1)
                            if child_tid >= 0 and child_tid < len(probs):
                                child_prob = float(probs[child_tid])
                                base_child_prob = top_child["prob"]
                                resistance = -float(np.log2(max(child_prob, 1e-10)))
                                base_resistance = -float(np.log2(max(base_child_prob, 1e-10)))
                                delta_resistance = resistance - base_resistance

                        # JS + top movers + reverse resist
                        base_logits = node.get("_logits")
                        node_js = 0.0
                        top_gained = top_lost = ""
                        abs_resistance = 0.0

                        if base_logits is not None:
                            from .metrics import js_divergence as _js, _align_vocab
                            bl, al = _align_vocab(np.array(base_logits), logits)
                            node_js = _js(bl, al)
                            from scipy.special import softmax as _sfm2
                            p_base = _sfm2(bl)
                            p_ann = _sfm2(al)
                            delta = p_ann - p_base
                            top5_gained = np.argsort(delta)[-3:][::-1]
                            top5_lost = np.argsort(delta)[:3]
                            top_gained = "|".join(
                                f"{tok.decode([int(i)]).strip()}({delta[i]:+.3f})"
                                for i in top5_gained if abs(delta[i]) > 0.005)
                            top_lost = "|".join(
                                f"{tok.decode([int(i)]).strip()}({delta[i]:+.3f})"
                                for i in top5_lost if abs(delta[i]) > 0.005)
                            base_argmax = int(np.argmax(p_base))
                            abs_resistance = -float(np.log2(max(p_ann[base_argmax], 1e-10)))
                            ann_argmax = int(np.argmax(p_ann))
                            base_prob_of_ann = float(p_base[ann_argmax])
                            ann_prob_of_ann = float(p_ann[ann_argmax])
                            node[f"{short}_reverse_resist"] = (
                                -float(np.log2(max(base_prob_of_ann, 1e-10)))
                                - (-float(np.log2(max(ann_prob_of_ann, 1e-10))))
                            )
                            node[f"{short}_ann_choice"] = tok.decode([ann_argmax]).strip()
                            node[f"{short}_ann_choice_base_prob"] = base_prob_of_ann

                        node[f"{short}_entropy"] = ent
                        node[f"{short}_argmax"] = argmax_word
                        node[f"{short}_js"] = node_js
                        node[f"{short}_entropy_delta"] = ent - node["entropy"]
                        node[f"{short}_abs_resistance"] = abs_resistance
                        node[f"{short}_resistance"] = resistance
                        node[f"{short}_delta_resist"] = delta_resistance
                        node[f"{short}_prob_child"] = child_prob
                        node[f"{short}_top_gained"] = top_gained
                        node[f"{short}_top_lost"] = top_lost

                        if out.hidden_states:
                            h_ann = out.hidden_states[-1][0, -1, :].cpu().numpy()
                            node[f"{short}_hidden"] = h_ann
                            if "hidden" in node:
                                h_base = np.array(node["hidden"])
                                nb = np.linalg.norm(h_base)
                                na = np.linalg.norm(h_ann)
                                if nb > 1e-10 and na > 1e-10:
                                    cos_dist = 1.0 - float(np.dot(h_base, h_ann) / (nb * na))
                                else:
                                    cos_dist = 1.0
                                node[f"{short}_hidden_dist"] = cos_dist

                        total_nodes += 1

                del model
                gc.collect()
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass

            except Exception as e:
                print(f"\n  {ann_label} FAILED: {str(e)[:60]}")

        # Cache all annotated trees
        cache = _get_cache()
        for pname, ptext in prompts.items():
            tree_key = {"model": self.model_id, "prompt": ptext,
                        "path_threshold": 0.01, "max_depth": max_depth,
                        "type": "explore_tree_v2"}
            cache.set_derived(tree_key, trees[pname])

        return trees

    def tree_to_vecdb(self, prompt: str, **kwargs):
        """Explore tree and store in lancedb with hidden states + graph edges."""
        from .vecdb import VecDB
        from .registry import Registry

        nodes = self.explore_tree(prompt, **kwargs)
        reg = Registry()
        info = reg.info(self.model_id)
        _, rel = reg.parent_of(self.model_id)
        base_id = reg.base_of(self.model_id)

        records = []
        for i, n in enumerate(nodes):
            if "hidden" not in n:
                continue
            parent_token = nodes[n["parent"]]["token"] if n["parent"] >= 0 else ""
            path_parts = []
            idx = i
            while idx >= 0:
                path_parts.append(nodes[idx]["token"])
                idx = nodes[idx]["parent"]
            path = " → ".join(reversed(path_parts))

            records.append({
                "node_id": i,
                "parent_id": n["parent"],
                "model": self.model_id,
                "model_short": self.model_id.split("/")[-1],
                "family": base_id.split("/")[-1] if base_id else "",
                "relation": rel or "base",
                "org": info.org if info else "",
                "prompt": prompt,
                "depth": n["depth"],
                "token": n["token"],
                "parent_token": parent_token,
                "path": path,
                "prob": n["prob"],
                "cumul_prob": n["cumul_prob"],
                "entropy": n["entropy"],
                "n_children": n["n_children"],
                "vector": n["hidden"],
            })

        if records:
            db = VecDB()
            hdim = len(records[0]["vector"])
            table_name = f"trees_{hdim}"
            if table_name in db.db.table_names():
                db.db.open_table(table_name).add(records)
            else:
                db.db.create_table(table_name, records)
            print(f"  Stored {len(records)} tree nodes in vecdb ({table_name})")

        return nodes

    def across_prompts(self, max_tokens: int = 10) -> 'pd.DataFrame':
        """This model's tree stats across all prompts."""
        import pandas as pd
        from .metrics import tree_metrics
        rows = []
        for name, text in PROMPTS.items():
            t = tree_metrics(self, text, n_gens=100, max_tokens=max_tokens)
            if not t or t['n_gens'] < 5:
                continue
            rows.append({
                'prompt': name, 'n_branches': t['n_branches'],
                'branch_entropy': t['branch_entropy'],
                'top_branch': t['top_branch'],
                'top_branch_pct': t['top_branch_pct'],
            })
        return pd.DataFrame(rows)

    # -- figure generation -----------------------------------------------------

    @staticmethod
    def figure(kind: str, prompt: str = "anger", family: str = None,
               model: str = None, save: str = None, **kwargs):
        """Generate a publication figure from probe data.

        Kinds:
            'census'       — bar chart of all models' branch entropy
            'tree'         — Sankey tree for one model
            'compare'      — base vs aligned tree side by side
            'distribution' — top-k token probabilities overlay
            'trajectory'   — violence/procedural loading across positions
            'branches'     — branch survival (base→aligned)

        Returns matplotlib Figure. Saves to path if `save` given.
        """
        import matplotlib.pyplot as plt
        from .metrics import tree_metrics, tree_compare

        prompt_text = _resolve_prompt(prompt)

        if kind == 'census':
            df = Probe.census(prompt, max_tokens=kwargs.get('T', 10))
            df = df.sort_values('branch_entropy')
            fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
            colors = {'base': '#4e79a7', 'sft_of': '#f28e2b', 'dpo_of': '#e15759',
                      'rlvr_of': '#59a14f', 'aligned_of': '#76b7b2'}
            for _, row in df.iterrows():
                c = colors.get(row.get('relation', ''), '#999')
                ax.barh(row['model_short'], row['branch_entropy'], color=c, alpha=0.8)
            ax.set_xlabel('Branch entropy (bits)')
            ax.set_title(f'Tree breadth: {prompt}')
            ax.invert_yaxis()

        elif kind == 'tree':
            model_id = model or (Probe.resolve(family) if family else None)
            if not model_id:
                raise ValueError("Specify model= or family=")
            p = Probe(model_id)
            t = p.tree(prompt, max_tokens=kwargs.get('T', 10))
            branches = list(t['branches'].items())[:20]
            fig, ax = plt.subplots(figsize=(8, 5))
            tokens = [b[0] for b in branches]
            pcts = [b[1] * 100 for b in branches]
            ax.barh(tokens[::-1], pcts[::-1], color='#4e79a7')
            ax.set_xlabel('% of generations')
            ax.set_title(f'{model_id.split("/")[-1]} — {prompt}')

        elif kind == 'compare':
            if not family:
                raise ValueError("Specify family=")
            base_id = Probe.resolve(family)
            from .registry import Registry
            reg = Registry()
            variants = reg.variants_of(base_id)
            final = variants[-1] if variants else base_id

            t_base = tree_metrics(Probe(base_id), prompt_text, max_tokens=kwargs.get('T', 10))
            t_aligned = tree_metrics(Probe(final), prompt_text, max_tokens=kwargs.get('T', 10))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
            for ax, t, title in [(ax1, t_base, 'Base'), (ax2, t_aligned, 'Aligned')]:
                branches = list(t['branches'].items())[:15]
                tokens = [b[0] for b in branches]
                pcts = [b[1] * 100 for b in branches]
                ax.barh(tokens[::-1], pcts[::-1], color='#4e79a7' if ax == ax1 else '#e15759')
                ax.set_xlabel('%')
                ax.set_title(f'{title} ({t["n_branches"]} branches, H={t["branch_entropy"]:.1f})')

            fig.suptitle(f'{family} — {prompt}', fontsize=14)

        elif kind == 'distribution':
            models = kwargs.get('models', [])
            if not models and family:
                base_id = Probe.resolve(family)
                from .registry import Registry
                reg = Registry()
                models = [base_id] + reg.variants_of(base_id)
            from scipy.special import softmax
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ['#4e79a7', '#e15759', '#59a14f', '#f28e2b']
            all_tokens = set()
            data = {}
            for i, mid in enumerate(models[:4]):
                try:
                    p = Probe(mid)
                    logits = p.logits(prompt_text, gen=0, pos=0,
                                      max_tokens=kwargs.get('T', 10))
                    probs = softmax(logits)
                    tok = p.tokenizer
                    top = probs.argsort()[-30:][::-1]
                    for t in top:
                        all_tokens.add((tok.decode([int(t)]).strip(), int(t)))
                    data[mid] = probs
                except FileNotFoundError:
                    pass

            tokens_sorted = sorted(all_tokens, key=lambda x: -data[models[0]][x[1]]
                                   if models[0] in data else 0)[:25]
            x = np.arange(len(tokens_sorted))
            w = 0.8 / len(data)
            for i, (mid, probs) in enumerate(data.items()):
                vals = [probs[tid] for _, tid in tokens_sorted]
                ax.bar(x + i * w, vals, w, label=mid.split('/')[-1],
                       color=colors[i % len(colors)], alpha=0.8)
            ax.set_xticks(x + w * len(data) / 2)
            ax.set_xticklabels([t[0] for t in tokens_sorted], rotation=45, ha='right')
            ax.set_ylabel('Probability')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.set_title(f'Distribution overlay — {prompt}')

        else:
            raise ValueError(f"Unknown figure kind: {kind}")

        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved {save}")
        return fig

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
