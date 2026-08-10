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


def test_model_family_has_superego_or_declares_why_not():
    """Every family reaches a preference-tuned rung, or SAYS WHY IT DOES NOT.

    **THE OLD ASSERTION WAS "every family has a superego", AND IT PASSED FOR THE
    WRONG REASON.** It held across all 49 families only because `tulu-no-safety`
    pointed at the STANDARD family's `Tulu-3-8B-DPO` and `Tulu-3.1-8B` -- a
    pipeline nobody trained. The no-safety SFT does not lead to that DPO. The
    green test was recording a fabricated lineage, and trimming the family to
    its two real rungs is what made it fail.

    Two-rung families are legitimate. What is not legitimate is an absence
    nobody declared, since a missing rung and an unfilled field are identical in
    the data. **So the declaration is checked against the registry, not merely
    read**: `none-published` must be true, and `scoped-to:` must name a family
    where the arm really does exist. A reason that cannot be falsified would be
    the old test with extra steps.
    """
    import json
    import os
    from malign_logits import MODEL_FAMILIES, PATH_DATA

    reg = os.path.join(PATH_DATA, "model_registry.json")
    if not os.path.exists(reg):
        pytest.skip("model_registry.json unavailable")
    rel = json.load(open(reg))["relations"]
    dpo_parents = {r["parent"] for r in rel if r["relation"] == "dpo_of"}

    for key, fam in MODEL_FAMILIES.items():
        if fam.superego is not None or fam.reinforced_superego is not None:
            continue
        why = fam.no_superego
        assert why, (
            "%s has no superego and no `no_superego` reason. An undeclared "
            "absence is indistinguishable from an unfilled field." % key)
        if why == "none-published":
            assert fam.ego not in dpo_parents, (
                "%s declares `none-published`, but the registry holds a dpo_of "
                "child for %s. The arm exists: wire it up, or declare "
                "`scoped-to:`." % (key, fam.ego))
        elif why.startswith("scoped-to:"):
            other = why.split(":", 1)[1]
            assert other in MODEL_FAMILIES, (
                "%s is scoped to %r, which is not a family." % (key, other))
            assert fam.ego in dpo_parents, (
                "%s says its superego is carried by %s, but the registry has "
                "no dpo_of child for %s at all -- so it is not scoped "
                "elsewhere, it is missing." % (key, other, fam.ego))
        else:
            raise AssertionError(
                "%s: unknown no_superego reason %r. Use 'none-published' or "
                "'scoped-to:<family>'; free text cannot be checked."
                % (key, why))


def test_prompts_exist():
    from malign_logits.experiments import DEFAULT_PROMPTS, TIER1_PROMPTS
    assert len(DEFAULT_PROMPTS) >= 47
    assert len(TIER1_PROMPTS) >= 18
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
