"""Tests for the cache system.

Run with: pytest tests/ -v
Or just: python tests/test_cache.py
"""

import tempfile
import os
import numpy as np
import pytest

from malign_logits.cache import CacheManager, normalize_text


# ── normalize_text ──────────────────────────────────────────────

def test_normalize_text_strips_trailing():
    assert normalize_text("hello  ") == "hello"
    assert normalize_text("hello\n") == "hello"

def test_normalize_text_preserves_leading():
    assert normalize_text(" hello") == " hello"

def test_normalize_text_consistent():
    assert normalize_text("hello ") == normalize_text("hello  ")
    assert normalize_text("hello\n") == normalize_text("hello")


# ── CacheManager roundtrips ─────────────────────────────────────

@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield CacheManager(root=tmpdir)


def test_logits_roundtrip(cache):
    arr = np.random.randn(100).astype(np.float32)
    cache.set_logits("model-a", "prompt-1", arr)
    assert cache.has_logits("model-a", "prompt-1")
    result = cache.get_logits("model-a", "prompt-1")
    np.testing.assert_array_almost_equal(result, arr)


def test_logits_miss(cache):
    assert not cache.has_logits("model-a", "nonexistent")
    assert cache.get_logits("model-a", "nonexistent") is None


def test_generation_roundtrip(cache):
    cache.set_generation("model-a", "prompt-1", "hello world", temp=1.0, idx=0)
    cache.set_generation("model-a", "prompt-1", "goodbye world", temp=1.0, idx=1)
    assert cache.get_generation("model-a", "prompt-1", 1.0, 0) == "hello world"
    assert cache.get_generation("model-a", "prompt-1", 1.0, 1) == "goodbye world"
    assert cache.get_generation("model-a", "prompt-1", 1.0, 2) is None


def test_generation_count(cache):
    for i in range(5):
        cache.set_generation("model-a", "prompt-1", f"gen-{i}", temp=1.0, idx=i)
    assert cache.count_generations("model-a", "prompt-1", 1.0) == 5
    assert cache.count_generations("model-a", "prompt-1", 0.5) == 0


def test_sent_embeddings_roundtrip(cache):
    vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    cache.set_sent_embeddings("emb-1", "prompt", "some text here", vecs)
    assert cache.has_sent_embeddings("emb-1", "prompt", "some text here")
    result = cache.get_sent_embeddings("emb-1", "prompt", "some text here")
    assert result == vecs


def test_sent_embeddings_text_normalization(cache):
    vecs = [[1.0, 2.0]]
    cache.set_sent_embeddings("emb-1", "p", "text  ", vecs)
    # Trailing space difference should still match
    assert cache.has_sent_embeddings("emb-1", "p", "text")
    assert cache.has_sent_embeddings("emb-1", "p", "text  ")
    assert cache.has_sent_embeddings("emb-1", "p", "text\n")


def test_ref_surprisal_roundtrip(cache):
    tok_surps = [("hello", 2.5), ("world", 1.3)]
    cache.set_ref_surprisal("gpt2", "prompt", "some text", tok_surps)
    assert cache.has_ref_surprisal("gpt2", "prompt", "some text")
    result = cache.get_ref_surprisal("gpt2", "prompt", "some text")
    assert result == tok_surps


def test_self_surprisal_roundtrip(cache):
    tok_surps = [("the", 1.0), ("cat", 3.5)]
    cache.set_self_surprisal("model-a", "prompt", "the cat sat", tok_surps)
    assert cache.has_self_surprisal("model-a", "prompt", "the cat sat")
    result = cache.get_self_surprisal("model-a", "prompt", "the cat sat")
    assert result == tok_surps


def test_word_embedding_roundtrip(cache):
    emb = np.random.randn(768).astype(np.float32)
    cache.set_word_embedding("model-a", "prompt", "kill", 5, emb)
    assert cache.has_word_embedding("model-a", "prompt", "kill", 5)
    result = cache.get_word_embedding("model-a", "prompt", "kill", 5)
    np.testing.assert_array_almost_equal(result, emb)


def test_separate_stashes(cache):
    """Different data types don't interfere."""
    cache.set_logits("model-a", "prompt", np.array([1.0, 2.0]))
    cache.set_generation("model-a", "prompt", "hello", idx=0)
    assert cache.get_logits("model-a", "prompt") is not None
    assert cache.get_generation("model-a", "prompt", idx=0) == "hello"
    assert cache.get_generation("model-a", "prompt", idx=1) is None


# ── Integration: real data ──────────────────────────────────────

def test_real_cache_has_data():
    """Check that the migrated lmdb cache has data."""
    cache = CacheManager()  # default path
    logits_stash = cache._stash("logits")
    count = sum(1 for _ in zip(logits_stash.keys(), range(5)))
    assert count > 0, "No logits in cache — migration may not have run"


def test_real_generations():
    """Check load_generations_from_stash works with new cache."""
    from malign_logits.embedding import load_generations_from_stash
    df = load_generations_from_stash()
    assert len(df) > 0, "No generations loaded"
    assert "family" in df.columns
    assert "psg" in df.columns


# ── Compat wrapper ──────────────────────────────────────────────

def test_stash_compat_contains():
    """StashCompat translates old tuple keys."""
    from malign_logits.embedding import _StashCompat
    compat = _StashCompat()
    # We can't test specific keys without knowing what's cached,
    # but we can verify the interface works
    assert isinstance(("sent_embeddings_v3", "x", "y", "z") in compat, bool)


def test_psyche_cache():
    """Psyche uses CacheManager directly."""
    from malign_logits.psyche import Psyche
    p = Psyche.from_family('olmo', load=False)
    assert p._cache is not None
    assert hasattr(p._cache, 'get_logits')


# ── Run directly ────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
