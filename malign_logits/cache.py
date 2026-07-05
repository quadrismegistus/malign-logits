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
    ├── word_embeddings/   {'model', 'prompt', 'word', 'k'}
    ├── top_words_v2/      {'type', 'model', 'prompt', 'k'} — discover_top_words results
    ├── score_vocab_v2/    {'type', 'model', 'prompt', 'words'} — word-level probabilities
    ├── beams/             {'type', 'model', 'prompt', ...} — beam storylines + annotations
    ├── trees/             {'type', 'model', 'prompt', ...} — tree exploration results
    ├── logit_lens/        {'model', 'prompt', 'k'} — per-layer top-k projections
    ├── logit_lens_raw/    {'model', 'prompt'} — per-layer raw projection data
    └── perplexity/        {'model', 'prompt'}

    (psyche_derived, the legacy junk drawer, was retired 2026-07-05: every entry
    was shadowed by the typed stashes above.)

Text values in keys are hashed (SHA256[:16]) to avoid matching issues.
"""

import os

from . import PATH_DATA_RAW

CACHE_ROOT = os.path.join(PATH_DATA_RAW, "cache")

# hashstash encodes serializer/compress/b64 into the on-disk path
# (e.g. lmdb.hashstash.lz4+b64/data.db), so any open that relies on
# library defaults silently resolves to a different, empty store when
# those defaults change (b64 flips to False in hashstash 1.0). Every
# stash open in this project must pin the full format explicitly.
STASH_OPTIONS = dict(
    engine="lmdb",
    serializer="hashstash",
    compress="lz4",
    b64=True,
    map_size=200 * 1024**3,  # 200GB limit
)


def open_stash(root_dir, **overrides):
    """Open a HashStash with every format option pinned (see STASH_OPTIONS)."""
    from hashstash import HashStash
    return HashStash(root_dir=root_dir, **{**STASH_OPTIONS, **overrides})


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
            self._stashes[name] = open_stash(os.path.join(self.root, name))
        return self._stashes[name]

    # ── logits ──────────────────────────────────────────────────

    def get_logits(self, model, prompt, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        s = self._stash("logits")
        return s[key] if key in s else None

    def set_logits(self, model, prompt, logits, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        self._stash("logits")[key] = logits

    def has_logits(self, model, prompt, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        return key in self._stash("logits")

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

    def _probe_pos_key(self, model, prompt, gen, pos, max_tokens):
        return {"model": model, "prompt": prompt, "gen": gen,
                "pos": pos, "T": max_tokens}

    def get_probe_logits(self, model, prompt, gen=0, pos=0, max_tokens=20):
        s = self._stash("probe_logits")
        key = self._probe_pos_key(model, prompt, gen, pos, max_tokens)
        return s[key] if key in s else None

    def set_probe_logits(self, model, prompt, logits, gen=0, pos=0, max_tokens=20):
        self._stash("probe_logits")[
            self._probe_pos_key(model, prompt, gen, pos, max_tokens)] = logits

    def get_probe_hidden(self, model, prompt, gen=0, pos=0, max_tokens=20):
        s = self._stash("probe_hidden")
        key = self._probe_pos_key(model, prompt, gen, pos, max_tokens)
        return s[key] if key in s else None

    def set_probe_hidden(self, model, prompt, hidden, gen=0, pos=0, max_tokens=20):
        self._stash("probe_hidden")[
            self._probe_pos_key(model, prompt, gen, pos, max_tokens)] = hidden

    def get_probe_meta(self, model, prompt, gen=0, max_tokens=20):
        s = self._stash("probe_meta")
        key = {"model": model, "prompt": prompt, "gen": gen, "T": max_tokens}
        return s[key] if key in s else None

    def set_probe_meta(self, model, prompt, meta, gen=0, max_tokens=20):
        self._stash("probe_meta")[
            {"model": model, "prompt": prompt, "gen": gen, "T": max_tokens}] = meta

    def has_probe(self, model, prompt, gen=0, pos=0, max_tokens=20):
        return self._probe_pos_key(model, prompt, gen, pos, max_tokens) in self._stash("probe_logits")

    def count_probe_gens(self, model, prompt, max_tokens=20):
        if not self.has_probe(model, prompt, gen=0, pos=0, max_tokens=max_tokens):
            return 0
        lo, hi = 0, 1
        while self.has_probe(model, prompt, gen=hi, pos=0, max_tokens=max_tokens):
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self.has_probe(model, prompt, gen=mid, pos=0, max_tokens=max_tokens):
                lo = mid + 1
            else:
                hi = mid
        return lo

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

    # ── top words (discover_top_words results) ──────────────────

    def get_top_words(self, model, prompt, k=200):
        key = {"type": "top_words", "model": model, "prompt": prompt, "k": k}
        s = self._stash("top_words_v2")
        return s[key] if key in s else None

    def set_top_words(self, model, prompt, words, k=200):
        self._stash("top_words_v2")[{
            "type": "top_words", "model": model, "prompt": prompt, "k": k
        }] = words

    def has_top_words(self, model, prompt, k=200):
        return {"type": "top_words", "model": model, "prompt": prompt, "k": k} in self._stash("top_words_v2")

    # ── score vocab (word-level probabilities) ─────────────────

    def get_score_vocab(self, model, prompt, words=None):
        key = {"model": model, "prompt": prompt}
        s = self._stash("score_vocab_v2")
        if key in s:
            return s[key]
        # Fall back to old format with words in key
        if words is not None:
            old_key = {"type": "score_vocab", "model": model, "prompt": prompt, "words": tuple(words)}
            if old_key in s:
                return s[old_key]
        else:
            for k in s.keys():
                if isinstance(k, dict) and k.get("model") == model and k.get("prompt") == prompt:
                    return s[k]
        return None

    def set_score_vocab(self, model, prompt, scores, words=None):
        key = {"model": model, "prompt": prompt}
        self._stash("score_vocab_v2")[key] = scores

    def has_score_vocab(self, model, prompt, words=None):
        return self.get_score_vocab(model, prompt, words) is not None

    # ── word probs (hybrid: exact logit + beam for multi-token) ──

    def get_word_probs(self, model, prompt, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        s = self._stash("word_probs")
        return s[key] if key in s else None

    def set_word_probs(self, model, prompt, probs, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        self._stash("word_probs")[key] = probs

    def has_word_probs(self, model, prompt, mode="raw"):
        key = {"model": model, "prompt": prompt}
        if mode != "raw":
            key["mode"] = mode
        return key in self._stash("word_probs")

    # ── beam word probs (word-level via beam search) ─────────────

    def get_beam_words(self, model, prompt, n=1000, depth=3, mode="raw"):
        key = {"type": "beam_words", "model": model, "prompt": prompt, "n": n, "depth": depth}
        if mode != "raw":
            key["mode"] = mode
        s = self._stash("beam_words")
        return s[key] if key in s else None

    def set_beam_words(self, model, prompt, words, n=1000, depth=3, mode="raw"):
        key = {"type": "beam_words", "model": model, "prompt": prompt, "n": n, "depth": depth}
        if mode != "raw":
            key["mode"] = mode
        self._stash("beam_words")[key] = words

    def has_beam_words(self, model, prompt, n=1000, depth=3, mode="raw"):
        key = {"type": "beam_words", "model": model, "prompt": prompt, "n": n, "depth": depth}
        if mode != "raw":
            key["mode"] = mode
        return key in self._stash("beam_words")

    # ── beams (beam search storylines + cross-model annotations) ──

    def get_beams(self, key):
        s = self._stash("beams")
        return s[key] if key in s else None

    def set_beams(self, key, value):
        self._stash("beams")[key] = value

    def has_beams(self, key):
        return key in self._stash("beams")

    def iter_beam_keys(self):
        for k in self._stash("beams").keys():
            if isinstance(k, dict):
                yield k

    # ── trees (explore_tree results) ───────────────────────────

    def get_tree(self, key):
        s = self._stash("trees")
        return s[key] if key in s else None

    def set_tree(self, key, value):
        self._stash("trees")[key] = value

    def has_tree(self, key):
        return key in self._stash("trees")

    # ── logit lens ──────────────────────────────────────────────

    def get_logit_lens(self, model, prompt, k):
        s = self._stash("logit_lens")
        key = {"model": model, "prompt": prompt, "k": k}
        return s[key] if key in s else None

    def set_logit_lens(self, model, prompt, k, value):
        self._stash("logit_lens")[{"model": model, "prompt": prompt, "k": k}] = value

    def get_logit_lens_raw(self, model, prompt):
        s = self._stash("logit_lens_raw")
        key = {"model": model, "prompt": prompt}
        return s[key] if key in s else None

    def set_logit_lens_raw(self, model, prompt, value):
        self._stash("logit_lens_raw")[{"model": model, "prompt": prompt}] = value

    # ── perplexity ──────────────────────────────────────────────

    def get_perplexity(self, model, prompt):
        s = self._stash("perplexity")
        key = {"model": model, "prompt": prompt}
        return s[key] if key in s else None

    def set_perplexity(self, model, prompt, value):
        self._stash("perplexity")[{"model": model, "prompt": prompt}] = value

    # ── derived (typed routing for psyche.py cache keys) ────────

    def _derived_route(self, key):
        """Map a legacy-style derived key to (getter, setter, checker) thunks."""
        t = key.get("type", "") if isinstance(key, dict) else ""
        m, p = key.get("model"), key.get("prompt")
        if t == "top_words":
            k = key.get("k", 200)
            return (lambda: self.get_top_words(m, p, k),
                    lambda v: self.set_top_words(m, p, v, k),
                    lambda: self.has_top_words(m, p, k))
        if t == "beam_words":
            n, d = key.get("n", 1000), key.get("depth", 3)
            return (lambda: self.get_beam_words(m, p, n, d),
                    lambda v: self.set_beam_words(m, p, v, n, d),
                    lambda: self.has_beam_words(m, p, n, d))
        if t == "score_vocab":
            return (lambda: self.get_score_vocab(m, p),
                    lambda v: self.set_score_vocab(m, p, v),
                    lambda: self.has_score_vocab(m, p))
        if t in ("beam_annotated_v1", "beam_cross_v1"):
            return (lambda: self.get_beams(key),
                    lambda v: self.set_beams(key, v),
                    lambda: self.has_beams(key))
        if t == "explore_tree_v3":
            return (lambda: self.get_tree(key),
                    lambda v: self.set_tree(key, v),
                    lambda: self.has_tree(key))
        if t == "logit_lens":
            k = key.get("k")
            return (lambda: self.get_logit_lens(m, p, k),
                    lambda v: self.set_logit_lens(m, p, k, v),
                    lambda: self.get_logit_lens(m, p, k) is not None)
        if t == "logit_lens_raw":
            return (lambda: self.get_logit_lens_raw(m, p),
                    lambda v: self.set_logit_lens_raw(m, p, v),
                    lambda: self.get_logit_lens_raw(m, p) is not None)
        if t == "perplexity":
            return (lambda: self.get_perplexity(m, p),
                    lambda v: self.set_perplexity(m, p, v),
                    lambda: self.get_perplexity(m, p) is not None)
        raise ValueError(f"Unknown derived cache key type {t!r}: {key!r} — "
                         f"add a typed stash in CacheManager._derived_route")

    def get_derived(self, key):
        getter, _, _ = self._derived_route(key)
        return getter()

    def set_derived(self, key, value):
        _, setter, _ = self._derived_route(key)
        setter(value)

    def has_derived(self, key):
        _, _, checker = self._derived_route(key)
        return checker()


# Module-level singleton
_cache = None

def get_cache(root=None) -> CacheManager:
    global _cache
    if _cache is None or (root and _cache.root != root):
        _cache = CacheManager(root=root)
    return _cache
