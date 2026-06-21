"""Unified cache manager for all stash types.

Each data type gets its own HashStash with dict keys:

    cache/
    ├── logits/            {'model', 'prompt'}
    ├── reasoning_logits/  {'model', 'prompt'} → {'thinking', 'post_logits', 'raw_logits'}
    ├── generations/       {'model', 'prompt', 'temp', 'idx'}
    ├── mega_generations/  {'model', 'prompt', 'temp', 'idx'} → [position dicts]
    ├── gen_logprobs/      {'model', 'prompt', 'temp', 'idx'}
    ├── gen_annotations/   {'tagger', 'model', 'prompt', 'temp', 'idx'}
    ├── sent_embeddings/   {'embedder', 'prompt', 'text'}
    ├── ref_surprisal/     {'ref', 'prompt', 'text'}
    ├── self_surprisal/    {'model', 'prompt', 'text'}
    └── word_embeddings/   {'model', 'prompt', 'word', 'k'}

Text values in keys are hashed (SHA256[:16]) to avoid matching issues.
"""

import os

from . import PATH_DATA_RAW

CACHE_ROOT = os.path.join(PATH_DATA_RAW, "cache")


def normalize_text(text: str) -> str:
    """Canonical text normalization for cache keys.

    Always rstrip to avoid trailing whitespace mismatches.
    HashStash hashes the key internally, so storing full text
    doesn't affect path length — it just keeps keys readable.
    """
    return text.rstrip()


class CacheManager:
    def __init__(self, root=None):
        self.root = root or CACHE_ROOT
        self._stashes = {}

    def _stash(self, name):
        if name not in self._stashes:
            from hashstash import HashStash
            self._stashes[name] = HashStash(
                root_dir=os.path.join(self.root, name),
                engine="lmdb", compress="lz4", b64=True,
                map_size=50 * 1024**3,  # 50GB limit
            )
        return self._stashes[name]

    # ── logits ──────────────────────────────────────────────────

    def get_logits(self, model, prompt):
        key = {"model": model, "prompt": prompt}
        s = self._stash("logits")
        return s[key] if key in s else None

    def set_logits(self, model, prompt, logits):
        self._stash("logits")[{"model": model, "prompt": prompt}] = logits

    def has_logits(self, model, prompt):
        return {"model": model, "prompt": prompt} in self._stash("logits")

    # ── generations ─────────────────────────────────────────────

    def get_generation(self, model, prompt, temp=1.0, idx=0):
        key = {"model": model, "prompt": prompt, "temp": temp, "idx": idx}
        s = self._stash("generations")
        return s[key] if key in s else None

    def set_generation(self, model, prompt, text, temp=1.0, idx=0):
        self._stash("generations")[{
            "model": model, "prompt": prompt, "temp": temp, "idx": idx
        }] = text

    def count_generations(self, model, prompt, temp=1.0):
        """Count how many generations exist for this (model, prompt, temp).

        Uses binary search for O(log n) instead of O(n).
        """
        s = self._stash("generations")
        if {"model": model, "prompt": prompt, "temp": temp, "idx": 0} not in s:
            return 0
        # Binary search for upper bound
        lo, hi = 0, 1
        while {"model": model, "prompt": prompt, "temp": temp, "idx": hi} in s:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if {"model": model, "prompt": prompt, "temp": temp, "idx": mid} in s:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def iter_generations(self, model, prompt, temp=1.0):
        """Yield (idx, text) for all generations matching (model, prompt, temp)."""
        n = self.count_generations(model, prompt, temp)
        for idx in range(n):
            text = self.get_generation(model, prompt, temp=temp, idx=idx)
            if text is not None:
                yield idx, text

    # ── generation logprobs (API models) ─────────────────────────

    def get_gen_logprobs(self, model, prompt, temp=1.0, idx=0):
        key = {"model": model, "prompt": prompt, "temp": temp, "idx": idx}
        s = self._stash("gen_logprobs")
        return s[key] if key in s else None

    def set_gen_logprobs(self, model, prompt, logprobs, temp=1.0, idx=0):
        self._stash("gen_logprobs")[{
            "model": model, "prompt": prompt, "temp": temp, "idx": idx
        }] = logprobs

    def has_gen_logprobs(self, model, prompt, temp=1.0, idx=0):
        return {"model": model, "prompt": prompt, "temp": temp,
                "idx": idx} in self._stash("gen_logprobs")

    # ── generation annotations (LLM tagger scores) ──────────────

    def get_gen_annotation(self, tagger, model, prompt, temp=1.0, idx=0):
        key = {"tagger": tagger, "model": model, "prompt": prompt,
               "temp": temp, "idx": idx}
        s = self._stash("gen_annotations")
        return s[key] if key in s else None

    def set_gen_annotation(self, tagger, model, prompt, annotation,
                           temp=1.0, idx=0):
        self._stash("gen_annotations")[{
            "tagger": tagger, "model": model, "prompt": prompt,
            "temp": temp, "idx": idx,
        }] = annotation

    def has_gen_annotation(self, tagger, model, prompt, temp=1.0, idx=0):
        return {"tagger": tagger, "model": model, "prompt": prompt,
                "temp": temp, "idx": idx} in self._stash("gen_annotations")

    # ── sentence embeddings ─────────────────────────────────────

    def get_sent_embeddings(self, embedder, prompt, text):
        key = {"embedder": embedder, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("sent_embeddings")
        return s[key] if key in s else None

    def set_sent_embeddings(self, embedder, prompt, text, vectors):
        self._stash("sent_embeddings")[{
            "embedder": embedder, "prompt": prompt, "text": normalize_text(text)
        }] = vectors

    def has_sent_embeddings(self, embedder, prompt, text):
        return {"embedder": embedder, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("sent_embeddings")

    # ── reference surprisal ─────────────────────────────────────

    def get_ref_surprisal(self, ref_model, prompt, text):
        key = {"ref": ref_model, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("ref_surprisal")
        return s[key] if key in s else None

    def set_ref_surprisal(self, ref_model, prompt, text, tok_surps):
        self._stash("ref_surprisal")[{
            "ref": ref_model, "prompt": prompt, "text": normalize_text(text)
        }] = tok_surps

    def has_ref_surprisal(self, ref_model, prompt, text):
        return {"ref": ref_model, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("ref_surprisal")

    # ── token metrics (drift from hidden states) ─────────────────

    def get_token_metrics(self, ref_model, prompt, text):
        key = {"ref": ref_model, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("token_metrics")
        return s[key] if key in s else None

    def set_token_metrics(self, ref_model, prompt, text, metrics):
        self._stash("token_metrics")[{
            "ref": ref_model, "prompt": prompt, "text": normalize_text(text)
        }] = metrics

    def has_token_metrics(self, ref_model, prompt, text):
        return {"ref": ref_model, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("token_metrics")

    # ── self-surprisal ──────────────────────────────────────────

    def get_self_surprisal(self, model, prompt, text):
        key = {"model": model, "prompt": prompt, "text": normalize_text(text)}
        s = self._stash("self_surprisal")
        return s[key] if key in s else None

    def set_self_surprisal(self, model, prompt, text, tok_surps):
        self._stash("self_surprisal")[{
            "model": model, "prompt": prompt, "text": normalize_text(text)
        }] = tok_surps

    def has_self_surprisal(self, model, prompt, text):
        return {"model": model, "prompt": prompt,
                "text": normalize_text(text)} in self._stash("self_surprisal")

    # ── word embeddings ─────────────────────────────────────────

    def get_word_embedding(self, model, prompt, word, k):
        key = {"model": model, "prompt": prompt, "word": word, "k": k}
        s = self._stash("word_embeddings")
        return s[key] if key in s else None

    def set_word_embedding(self, model, prompt, word, k, embedding):
        self._stash("word_embeddings")[{
            "model": model, "prompt": prompt, "word": word, "k": k
        }] = embedding

    def has_word_embedding(self, model, prompt, word, k):
        return {"model": model, "prompt": prompt, "word": word,
                "k": k} in self._stash("word_embeddings")

    # ── reasoning logits (post-thinking distributions) ──────────

    def get_reasoning(self, model, prompt):
        """Get cached reasoning result: thinking text + post-thinking logits.

        Returns dict with keys: 'thinking', 'post_logits', 'raw_logits'
        or None if not cached.
        """
        key = {"model": model, "prompt": prompt}
        s = self._stash("reasoning_logits")
        return s[key] if key in s else None

    def set_reasoning(self, model, prompt, thinking, post_logits, raw_logits):
        """Cache reasoning result: thinking text + post-thinking logits."""
        self._stash("reasoning_logits")[{"model": model, "prompt": prompt}] = {
            "thinking": thinking,
            "post_logits": post_logits,
            "raw_logits": raw_logits,
        }

    def has_reasoning(self, model, prompt):
        return {"model": model, "prompt": prompt} in self._stash("reasoning_logits")

    # ── mega-generations (F25 position-level trajectories) ──────

    def _mega_key(self, model, prompt, temp=1.0, idx=0, mode="raw"):
        key = {"model": model, "prompt": prompt, "temp": temp, "idx": idx}
        if mode != "raw":
            key["mode"] = mode
        return key

    def get_mega_generation(self, model, prompt, temp=1.0, idx=0, mode="raw"):
        """Get cached position-level trajectory for a single generation."""
        s = self._stash("mega_generations")
        key = self._mega_key(model, prompt, temp, idx, mode)
        return s[key] if key in s else None

    def set_mega_generation(self, model, prompt, positions, temp=1.0, idx=0, mode="raw"):
        """Cache position-level trajectory (list of dicts with step/entropy/top5)."""
        self._stash("mega_generations")[
            self._mega_key(model, prompt, temp, idx, mode)
        ] = positions

    def has_mega_generation(self, model, prompt, temp=1.0, idx=0, mode="raw"):
        return self._mega_key(model, prompt, temp, idx, mode) in self._stash("mega_generations")

    def count_mega_generations(self, model, prompt, temp=1.0, mode="raw"):
        """Count cached mega-generations (binary search on idx)."""
        s = self._stash("mega_generations")
        if self._mega_key(model, prompt, temp, 0, mode) not in s:
            return 0
        lo, hi = 0, 1
        while self._mega_key(model, prompt, temp, hi, mode) in s:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self._mega_key(model, prompt, temp, mid, mode) in s:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ── probe: per-position logits/hidden, per-gen meta, per-model embeddings ──

    def _probe_pos_key(self, model, prompt, gen, pos):
        return {"model": model, "prompt": prompt, "gen": gen, "pos": pos}

    def get_probe_logits(self, model, prompt, gen=0, pos=0):
        s = self._stash("probe_logits")
        key = self._probe_pos_key(model, prompt, gen, pos)
        return s[key] if key in s else None

    def set_probe_logits(self, model, prompt, logits, gen=0, pos=0):
        self._stash("probe_logits")[
            self._probe_pos_key(model, prompt, gen, pos)] = logits

    def get_probe_hidden(self, model, prompt, gen=0, pos=0):
        s = self._stash("probe_hidden")
        key = self._probe_pos_key(model, prompt, gen, pos)
        return s[key] if key in s else None

    def set_probe_hidden(self, model, prompt, hidden, gen=0, pos=0):
        self._stash("probe_hidden")[
            self._probe_pos_key(model, prompt, gen, pos)] = hidden

    def get_probe_meta(self, model, prompt, gen=0):
        s = self._stash("probe_meta")
        key = {"model": model, "prompt": prompt, "gen": gen}
        return s[key] if key in s else None

    def set_probe_meta(self, model, prompt, meta, gen=0):
        self._stash("probe_meta")[
            {"model": model, "prompt": prompt, "gen": gen}] = meta

    def get_probe_embeddings(self, model):
        """Load embedding matrix from numpy file (too large for lmdb)."""
        import numpy as np
        path = os.path.join(self.root, "probe_embeddings",
                            model.replace("/", "--") + ".npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def set_probe_embeddings(self, model, embeddings):
        """Save embedding matrix as numpy file."""
        import numpy as np
        d = os.path.join(self.root, "probe_embeddings")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, model.replace("/", "--") + ".npy")
        np.save(path, embeddings)

    def has_probe(self, model, prompt, gen=0, pos=0):
        return self._probe_pos_key(model, prompt, gen, pos) in self._stash("probe_logits")

    def count_probe_gens(self, model, prompt):
        if not self.has_probe(model, prompt, gen=0, pos=0):
            return 0
        lo, hi = 0, 1
        while self.has_probe(model, prompt, gen=hi, pos=0):
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self.has_probe(model, prompt, gen=mid, pos=0):
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ── psyche derived (discover_top_words, score_vocab, etc.) ──

    def get_derived(self, key):
        s = self._stash("psyche_derived")
        return s[key] if key in s else None

    def set_derived(self, key, value):
        self._stash("psyche_derived")[key] = value

    def has_derived(self, key):
        return key in self._stash("psyche_derived")


# Module-level singleton
_cache = None

def get_cache(root=None) -> CacheManager:
    global _cache
    if _cache is None or (root and _cache.root != root):
        _cache = CacheManager(root=root)
    return _cache
