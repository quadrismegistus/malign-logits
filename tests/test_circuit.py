"""Tests for Circuit class — classifier, modes, signatures.

Run with: pytest tests/test_circuit.py -v
"""

import numpy as np
import pandas as pd
import pytest

from malign_logits.circuit import Circuit, Mode


# ── Helper to build synthetic trajectory DataFrames ──────────────

def _make_trajectory(n_steps=10, top1="kill", entropy_start=4.0, entropy_slope=-0.05,
                     top5_words="kill|hit|punch|slap|hurt"):
    """Build a synthetic trajectory DataFrame for testing."""
    rows = []
    for step in range(n_steps):
        h = entropy_start + entropy_slope * step
        rows.append({
            "step": step,
            "entropy": max(h, 0.01),
            "top1": top1 if step == 0 else "the",
            "top1_prob": 0.15 if step == 0 else 0.10,
            "top5_words": top5_words if step == 0 else "the|a|to|and|of",
        })
    return pd.DataFrame(rows)


# ── classify_trajectory ─────────────────────────────────────────

class TestClassifyTrajectory:

    def test_foreclosure_blank_no_transgressive(self):
        df = _make_trajectory(top1="______", top5_words="______|____|do|____|__")
        result = Circuit.classify_trajectory(df)
        assert result["signature"] == "foreclosure"
        assert result["step0_is_blank"] is True
        assert result["has_transgressive"] is False

    def test_return_of_repressed_blank_with_transgressive(self):
        df = _make_trajectory(top1="______", top5_words="______|kill|do|____|__")
        result = Circuit.classify_trajectory(df)
        assert result["signature"] == "return_of_repressed"
        assert result["step0_is_blank"] is True
        assert result["has_transgressive"] is True

    def test_repression_argmax_changed(self):
        df = _make_trajectory(top1="scream", top5_words="scream|cry|shout|yell|sob")
        result = Circuit.classify_trajectory(df, base_top1="kill")
        assert result["signature"] == "repression"
        assert result["argmax_preserved"] is False

    def test_transparent_argmax_preserved(self):
        df = _make_trajectory(top1="kill", top5_words="kill|hit|punch|slap|hurt",
                              entropy_slope=-0.01)
        result = Circuit.classify_trajectory(df, base_top1="kill")
        assert result["signature"] == "transparent"
        assert result["argmax_preserved"] is True

    def test_reaction_formation_steep_slope(self):
        df = _make_trajectory(top1="punch", top5_words="punch|hit|strike|slap|hurt",
                              entropy_start=5.0, entropy_slope=-0.25)
        result = Circuit.classify_trajectory(df)
        assert result["signature"] == "reaction_formation"
        assert result["entropy_slope"] < -0.15

    def test_de_foreclosure_base_was_blank(self):
        df = _make_trajectory(top1="seek", top5_words="seek|file|take|go|join")
        result = Circuit.classify_trajectory(df, base_top1="______")
        assert result["signature"] == "de_foreclosure"

    def test_de_foreclosure_blank_sentinel(self):
        df = _make_trajectory(top1="seek", top5_words="seek|file|take|go|join")
        result = Circuit.classify_trajectory(df, base_top1=Circuit.BLANK_SENTINEL)
        assert result["signature"] == "de_foreclosure"

    def test_nan_top1_is_blank(self):
        df = _make_trajectory(top1="nan", top5_words="|take|seek|file|go")
        result = Circuit.classify_trajectory(df)
        assert result["step0_is_blank"] is True

    def test_empty_top1_is_blank(self):
        df = _make_trajectory(top1="", top5_words="|a|b|c|d")
        result = Circuit.classify_trajectory(df)
        assert result["step0_is_blank"] is True

    def test_question_mark_is_blank(self):
        df = _make_trajectory(top1="?", top5_words="?|a|b|c|d")
        result = Circuit.classify_trajectory(df)
        assert result["step0_is_blank"] is True

    def test_too_few_steps_returns_unknown(self):
        df = _make_trajectory(n_steps=3)
        result = Circuit.classify_trajectory(df)
        assert result["signature"] == "unknown"

    def test_no_base_top1_unclassified(self):
        df = _make_trajectory(top1="seek", top5_words="seek|file|take|go|join",
                              entropy_slope=-0.01)
        result = Circuit.classify_trajectory(df)
        assert result["signature"] == "unclassified"

    def test_underscore_variants_are_blank(self):
        for blank in ["_", "▁", "___", "▁▁▁"]:
            df = _make_trajectory(top1=blank, top5_words=f"{blank}|a|b|c|d")
            result = Circuit.classify_trajectory(df)
            assert result["step0_is_blank"] is True, f"{blank!r} should be blank"


# ── signature_summary ────────────────────────────────────────────

class TestSignatureSummary:

    def test_summary_aggregation(self):
        rows = [
            {"layer": "dpo", "prompt_key": "anger", "gen_idx": i,
             "signature": "foreclosure" if i < 7 else "return_of_repressed"}
            for i in range(10)
        ]
        classified_df = pd.DataFrame(rows)
        c = Circuit.__new__(Circuit)
        summary = c.signature_summary(classified_df)

        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["dominant"] == "foreclosure"
        assert row["foreclosure"] == pytest.approx(0.7)
        assert row["return_of_repressed"] == pytest.approx(0.3)
        assert row["n"] == 10

    def test_summary_multiple_prompts(self):
        rows = []
        for pk in ["anger", "love"]:
            for i in range(5):
                rows.append({"layer": "dpo", "prompt_key": pk, "gen_idx": i,
                             "signature": "repression"})
        classified_df = pd.DataFrame(rows)
        c = Circuit.__new__(Circuit)
        summary = c.signature_summary(classified_df)

        assert len(summary) == 2
        assert set(summary["prompt_key"]) == {"anger", "love"}
        assert all(summary["dominant"] == "repression")


# ── Mode enum ────────────────────────────────────────────────────

class TestMode:

    def test_mode_values(self):
        assert Mode.RAW.value == "raw"
        assert Mode.CHAT.value == "chat"
        assert Mode.COMPLETE.value == "complete"
        assert Mode.THINK.value == "think"

    def test_mode_members(self):
        assert len(Mode) == 4


# ── TRANSGRESSIVE_TOKENS ─────────────────────────────────────────

class TestTransgressiveTokens:

    def test_contains_key_tokens(self):
        for tok in ["kill", "fuck", "murder", "naked", "blood"]:
            assert tok in Circuit.TRANSGRESSIVE_TOKENS

    def test_is_frozenset(self):
        assert isinstance(Circuit.TRANSGRESSIVE_TOKENS, frozenset)


# ── BLANK_SENTINEL ───────────────────────────────────────────────

class TestBlankSentinel:

    def test_sentinel_exists(self):
        assert hasattr(Circuit, "BLANK_SENTINEL")
        assert Circuit.BLANK_SENTINEL == "__BLANK__"

    def test_sentinel_not_in_transgressive(self):
        assert Circuit.BLANK_SENTINEL not in Circuit.TRANSGRESSIVE_TOKENS
