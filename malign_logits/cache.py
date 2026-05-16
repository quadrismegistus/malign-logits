"""Unified cache manager for all stash types.

Each data type gets its own HashStash with dict keys:

    cache/
    ├── logits/           {'model', 'prompt'}
    ├── generations/      {'model', 'prompt', 'temp', 'idx'}
    ├── sent_embeddings/  {'embedder', 'prompt', 'text'}
    ├── ref_surprisal/    {'ref', 'prompt', 'text'}
    ├── self_surprisal/   {'model', 'prompt', 'text'}
    └── word_embeddings/  {'model', 'prompt', 'word', 'k'}

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


# Module-level singleton
_cache = None

def get_cache(root=None) -> CacheManager:
    global _cache
    if _cache is None or (root and _cache.root != root):
        _cache = CacheManager(root=root)
    return _cache
