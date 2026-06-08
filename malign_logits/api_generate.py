"""Frontier model generation via API.

Supports OpenAI, Anthropic, Google, and DeepSeek (OpenAI-compatible).
Stores generations and logprobs (where available) in the malign-logits cache.

Each model gets two conditions:
    {provider}/{model}       — with system prompt (continuation mode)
    {provider}/{model}-raw   — no system prompt (native chat behaviour)
"""

import os
import time

SYSTEM_PROMPT = ("Continue the following text. "
                 "Write only the continuation, no commentary or explanation.")

MAX_RETRIES = 8
RETRY_BASE_DELAY = 2.0


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class Provider:
    name: str
    env_key: str

    def generate(self, model, prompt, max_tokens=100, temperature=1.0,
                 system_prompt=None, logprobs=True):
        """Return (text, logprobs_data_or_None)."""
        raise NotImplementedError

    def check_key(self):
        key = os.environ.get(self.env_key)
        if not key:
            raise RuntimeError(f"Set {self.env_key} environment variable")
        return key


class OpenAIProvider(Provider):
    name = "openai"
    env_key = "OPENAI_API_KEY"

    def __init__(self, base_url=None, env_key=None):
        if env_key:
            self.env_key = env_key
        self._base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.check_key()}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, model, prompt, max_tokens=100, temperature=1.0,
                 system_prompt=None, logprobs=True):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 20

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content

        lp_data = None
        if logprobs and choice.logprobs and choice.logprobs.content:
            lp_data = [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "top_logprobs": [
                        {"token": tp.token, "logprob": tp.logprob}
                        for tp in t.top_logprobs
                    ] if t.top_logprobs else [],
                }
                for t in choice.logprobs.content
            ]
        return text, lp_data


class AnthropicProvider(Provider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.check_key())
        return self._client

    def generate(self, model, prompt, max_tokens=100, temperature=1.0,
                 system_prompt=None, logprobs=True):
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        text = response.content[0].text
        return text, None  # Anthropic doesn't expose logprobs


class GoogleProvider(Provider):
    name = "google"
    env_key = "GEMINI_API_KEY"

    def check_key(self):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        return key

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.check_key())
        return self._client

    def generate(self, model, prompt, max_tokens=100, temperature=1.0,
                 system_prompt=None, logprobs=True):
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text, None


# ---------------------------------------------------------------------------
# Model registry — maps cache model_id prefix to (provider, api_model_name)
# ---------------------------------------------------------------------------

MODELS = {
    # DeepSeek
    "deepseek/deepseek-chat": ("deepseek", "deepseek-chat"),
    # OpenAI
    "openai/gpt-4o": ("openai", "gpt-4o"),
    "openai/gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "openai/gpt-4.1-mini": ("openai", "gpt-4.1-mini"),
    "openai/gpt-4.1-nano": ("openai", "gpt-4.1-nano"),
    # Anthropic
    "anthropic/claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "anthropic/claude-haiku-4-5": ("anthropic", "claude-haiku-4-5-20251001"),
    # Google
    "google/gemini-2.5-flash": ("google", "gemini-2.5-flash"),
}

PROVIDERS = {
    "openai": OpenAIProvider(),
    "deepseek": OpenAIProvider(
        base_url="https://api.deepseek.com",
        env_key="DEEPSEEK_API_KEY",
    ),
    "anthropic": AnthropicProvider(),
    "google": GoogleProvider(),
}


def list_models():
    """Return list of available model cache IDs."""
    return sorted(MODELS.keys())


def _call_with_retry(provider, api_model, prompt, max_tokens, temperature,
                     system_prompt, logprobs):
    for attempt in range(MAX_RETRIES):
        try:
            return provider.generate(
                api_model, prompt, max_tokens=max_tokens,
                temperature=temperature, system_prompt=system_prompt,
                logprobs=logprobs,
            )
        except Exception as e:
            err = str(e)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if "rate" in err.lower() or "429" in err or "overloaded" in err.lower():
                    print(f"    Rate limited, waiting {delay:.0f}s...")
                else:
                    print(f"    Error: {err[:120]}, retry in {delay:.0f}s...")
                time.sleep(delay)
            else:
                raise


def run(model_id, prompts_dict, n=100, temperature=1.0, max_tokens=100,
        raw=False, dry_run=False, save_lp=True):
    from malign_logits.cache import get_cache
    cache = get_cache()

    if model_id not in MODELS:
        print(f"Unknown model: {model_id}")
        print(f"Available: {', '.join(list_models())}")
        return

    provider_key, api_model = MODELS[model_id]
    provider = PROVIDERS[provider_key]

    cache_id = f"{model_id}-raw" if raw else model_id
    system_prompt = None if raw else SYSTEM_PROMPT

    mode = "raw (no system prompt)" if raw else "prompted"
    print(f"\nAPI generation: {cache_id}")
    print(f"  provider: {provider_key}, api_model: {api_model}")
    print(f"  {len(prompts_dict)} prompts × {n} generations")
    print(f"  temperature={temperature}, max_tokens={max_tokens}")
    print(f"  mode: {mode}")
    has_lp = save_lp and provider_key not in ('anthropic', 'google')
    print(f"  logprobs: {'yes' if has_lp else 'no'}")

    # Build work list: (label, prompt_text, start_idx) for each needed generation
    work = []
    total_skipped = 0
    for label, prompt_text in prompts_dict.items():
        existing = cache.count_generations(cache_id, prompt_text, temp=temperature)
        needed = n - existing
        if needed <= 0:
            total_skipped += 1
            continue
        for i in range(needed):
            work.append((label, prompt_text, existing + i))

    if total_skipped:
        print(f"  Skipping {total_skipped} prompts (already complete)")
    if not work:
        print("  Nothing to generate.")
        return
    print(f"  {len(work)} generations to run")

    if dry_run:
        for label, prompt_text, _ in work[:5]:
            print(f"    {label}: {prompt_text[:40]}...")
        if len(work) > 5:
            print(f"    ... and {len(work) - 5} more")
        return

    from tqdm import tqdm
    pbar = tqdm(work, desc=cache_id, unit="gen")
    cur_label = None
    for label, prompt_text, idx in pbar:
        if label != cur_label:
            cur_label = label
            pbar.set_postfix_str(label)
        try:
            text, lp_data = _call_with_retry(
                provider, api_model, prompt_text,
                max_tokens=max_tokens, temperature=temperature,
                system_prompt=system_prompt, logprobs=save_lp,
            )
            cache.set_generation(cache_id, prompt_text, text,
                                 temp=temperature, idx=idx)
            if save_lp and lp_data:
                cache.set_gen_logprobs(cache_id, prompt_text,
                                       lp_data, temp=temperature, idx=idx)
        except Exception as e:
            tqdm.write(f"  Error on {label} idx={idx}: {e}")

    print(f"\n  Done. {len(work)} generations attempted.")
    if total_skipped:
        print(f"  ({total_skipped} prompts were already complete)")
