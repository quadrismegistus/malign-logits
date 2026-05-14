"""Tests for core functionality.

Unit tests use mocked models. Integration tests use SmolLM2 360M
and are skipped in CI (marked with @pytest.mark.slow).

Run fast tests:  pytest tests/ -v -m "not slow"
Run all tests:   pytest tests/ -v
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


# ── Model family registry ──────────────────────────────────────

def test_model_families_exist():
    from malign_logits import MODEL_FAMILIES
    assert len(MODEL_FAMILIES) >= 10
    for key, fam in MODEL_FAMILIES.items():
        assert fam.base is not None
        assert fam.name


def test_model_family_has_superego():
    from malign_logits import MODEL_FAMILIES
    for key, fam in MODEL_FAMILIES.items():
        assert fam.superego is not None or fam.reinforced_superego is not None


def test_prompts_exist():
    from malign_logits.experiments import DEFAULT_PROMPTS, TIER1_PROMPTS
    assert len(DEFAULT_PROMPTS) == 47
    assert len(TIER1_PROMPTS) == 18
    for k, v in DEFAULT_PROMPTS.items():
        assert isinstance(v, str)
        assert len(v) > 5


# ── Analysis functions (mocked model) ──────────────────────────

def _mock_model_and_tokenizer(vocab_size=1000, hidden_dim=64):
    """Create a minimal mock model that returns plausible logits."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.num_hidden_layers = 4
    model.config.hidden_size = hidden_dim

    param = torch.zeros(1)
    model.parameters = MagicMock(return_value=iter([param]))
    model.device = param.device

    # Model forward returns logits
    logits = torch.randn(1, 10, vocab_size)
    output = MagicMock()
    output.logits = logits
    output.hidden_states = [torch.randn(1, 10, hidden_dim) for _ in range(5)]
    model.__call__ = MagicMock(return_value=output)
    model.return_value = output

    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=torch.randint(0, vocab_size, (1, 10)))
    tokenizer.decode = MagicMock(side_effect=lambda ids: "word")
    tokenizer.vocab_size = vocab_size
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    return model, tokenizer


def test_js_divergence():
    from malign_logits.analysis import js_divergence
    a = torch.randn(100)
    b = torch.randn(100)
    js = js_divergence(a, b)
    assert 0 <= js <= np.log(2) + 0.01  # JS bounded by ln(2)
    assert js_divergence(a, a) < 1e-6  # identical = 0


def test_distribution_entropy():
    from malign_logits.analysis import distribution_entropy
    uniform = torch.zeros(100)  # uniform logits
    peaked = torch.zeros(100)
    peaked[0] = 100.0  # very peaked
    h_uniform = distribution_entropy(uniform)
    h_peaked = distribution_entropy(peaked)
    assert h_uniform > h_peaked


# ── Passage metrics (no model needed) ──────────────────────────

def test_drift_metrics():
    from malign_logits.embedding import drift_metrics_from_embeddings
    vecs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    d = drift_metrics_from_embeddings(vecs)
    assert "total_drift" in d
    assert "directedness" in d
    assert d["total_drift"] > 0


def test_surprisal_metrics():
    from malign_logits.embedding import surprisal_metrics_from_tokens
    tok_surps = [("hello", 2.5), ("world", 1.3), ("foo", 3.1)]
    s = surprisal_metrics_from_tokens(tok_surps)
    assert abs(s["mean_surprisal"] - np.mean([2.5, 1.3, 3.1])) < 0.01
    assert s["n_tokens"] == 3


def test_degenerate_detection():
    from malign_logits.embedding import _is_degenerate
    assert _is_degenerate("aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa")
    assert not _is_degenerate("The quick brown fox jumps over the lazy dog")


def test_sentence_splitting():
    from malign_logits.embedding import _split_sentences
    sents = _split_sentences("Hello world. How are you? Fine thanks.")
    assert len(sents) == 3


# ── Cache integration ──────────────────────────────────────────

def test_text_normalization_in_cache():
    """Verify cache normalizes text consistently."""
    from malign_logits.cache import normalize_text
    assert normalize_text("hello ") == normalize_text("hello  ")
    assert normalize_text("hello\n") == normalize_text("hello")
    assert normalize_text(" hello") != normalize_text("hello")


# ── SmolLM2 integration (slow, skip in CI) ─────────────────────

@pytest.mark.slow
def test_smol_psyche_from_family():
    """Load SmolLM2 and run basic analysis."""
    from malign_logits.psyche import Psyche
    psyche = Psyche.from_family("smol", load=True)
    assert psyche.n_layers == 2
    assert psyche.primary_process.model is not None

    logits = psyche.primary_process.logits("The capital of France is")
    assert logits.shape[0] > 1000  # full vocab

    # Clean up
    del psyche
    import gc
    gc.collect()


@pytest.mark.slow
def test_smol_generation():
    """Generate text from SmolLM2."""
    from malign_logits.psyche import Psyche
    psyche = Psyche.from_family("smol", load=True)
    result = psyche.generate("Once upon a time", max_new_tokens=20)
    assert "base" in result
    assert len(result["base"]) > 0
    del psyche
    import gc
    gc.collect()
