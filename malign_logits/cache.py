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

import hashlib
import os

from . import PATH_DATA_RAW

CACHE_ROOT = os.path.join(PATH_DATA_RAW, "cache")


def text_hash(text: str) -> str:
    """Canonical hash for passage text in cache keys."""
    return hashlib.sha256(text.rstrip().encode()).hexdigest()[:16]


class CacheManager:
    def __init__(self, root=None):
        self.root = root or CACHE_ROOT
        self._stashes = {}

    def _stash(self, name):
        if name not in self._stashes:
            from hashstash import HashStash
            self._stashes[name] = HashStash(
                root_dir=os.path.join(self.root, name),
                engine="pairtree", compress="lz4", b64=True,
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
        """Count how many generations exist for this (model, prompt, temp)."""
        s = self._stash("generations")
        count = 0
        while True:
            key = {"model": model, "prompt": prompt, "temp": temp, "idx": count}
            if key not in s:
                break
            count += 1
        return count

    # ── sentence embeddings ─────────────────────────────────────

    def get_sent_embeddings(self, embedder, prompt, text):
        key = {"embedder": embedder, "prompt": prompt, "text": text_hash(text)}
        s = self._stash("sent_embeddings")
        return s[key] if key in s else None

    def set_sent_embeddings(self, embedder, prompt, text, vectors):
        self._stash("sent_embeddings")[{
            "embedder": embedder, "prompt": prompt, "text": text_hash(text)
        }] = vectors

    def has_sent_embeddings(self, embedder, prompt, text):
        return {"embedder": embedder, "prompt": prompt,
                "text": text_hash(text)} in self._stash("sent_embeddings")

    # ── reference surprisal ─────────────────────────────────────

    def get_ref_surprisal(self, ref_model, prompt, text):
        key = {"ref": ref_model, "prompt": prompt, "text": text_hash(text)}
        s = self._stash("ref_surprisal")
        return s[key] if key in s else None

    def set_ref_surprisal(self, ref_model, prompt, text, tok_surps):
        self._stash("ref_surprisal")[{
            "ref": ref_model, "prompt": prompt, "text": text_hash(text)
        }] = tok_surps

    def has_ref_surprisal(self, ref_model, prompt, text):
        return {"ref": ref_model, "prompt": prompt,
                "text": text_hash(text)} in self._stash("ref_surprisal")

    # ── self-surprisal ──────────────────────────────────────────

    def get_self_surprisal(self, model, prompt, text):
        key = {"model": model, "prompt": prompt, "text": text_hash(text)}
        s = self._stash("self_surprisal")
        return s[key] if key in s else None

    def set_self_surprisal(self, model, prompt, text, tok_surps):
        self._stash("self_surprisal")[{
            "model": model, "prompt": prompt, "text": text_hash(text)
        }] = tok_surps

    def has_self_surprisal(self, model, prompt, text):
        return {"model": model, "prompt": prompt,
                "text": text_hash(text)} in self._stash("self_surprisal")

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
