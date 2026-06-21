"""Integration test: Circuit end-to-end with SmolLM2-360M.

Uses the smol family (SmolLM2-360M, ~720MB) as a real 2-layer test.
Exercises: from_family → compare → formation → Mode switching.

Run with: pytest tests/test_circuit_integration.py -v
Requires ~1GB RAM and ~30s on MPS.
"""

import pytest
import pandas as pd

from malign_logits.circuit import Circuit, Mode


PROMPT = "She was so angry she wanted to"


@pytest.fixture(scope="module")
def circuit():
    """Load SmolLM2-360M circuit once for all tests."""
    c = Circuit.from_family("smol", load=True)
    return c


class TestCircuitIntegration:

    def test_from_family_loads_nodes(self, circuit):
        assert "base" in circuit.positions
        assert "dpo" in circuit.positions
        assert len(circuit.positions) == 2

    def test_main_path(self, circuit):
        path = circuit.main_path
        assert len(path) == 2
        assert path[0].position == "base"
        assert path[1].position == "dpo"

    def test_compare_produces_edge(self, circuit):
        edge = circuit.compare("base", "dpo", PROMPT)
        assert 0 <= edge.js_divergence <= 0.693
        assert isinstance(edge.source_entropy, float)
        assert isinstance(edge.target_entropy, float)
        assert edge.source_eff_vocab > 0
        assert edge.target_eff_vocab > 0

    def test_compare_self_is_zero(self, circuit):
        edge = circuit.compare("base", "base", PROMPT)
        assert edge.js_divergence == pytest.approx(0.0, abs=1e-6)

    def test_formation_produces_dataframe(self, circuit):
        df = circuit.formation(PROMPT)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "word" in df.columns
        assert "base" in df.columns
        assert "change" in df.columns

    def test_mode_raw_default(self, circuit):
        assert circuit.mode == Mode.RAW

    def test_mode_chat_produces_different_logits(self, circuit):
        node = circuit.nodes["dpo"]
        raw_logits = node.logits(PROMPT, mode=Mode.RAW)
        chat_logits = node.logits(PROMPT, mode=Mode.CHAT)
        assert raw_logits.shape == chat_logits.shape
        # Chat template changes the distribution
        from malign_logits.analysis import js_divergence
        js = js_divergence(raw_logits, chat_logits)
        assert js > 0.01

    def test_node_entropy(self, circuit):
        for pos in circuit.positions:
            node = circuit.nodes[pos]
            h = node.entropy(PROMPT)
            assert 0 < h < 15

    def test_node_top_tokens(self, circuit):
        node = circuit.nodes["base"]
        top = node.top_tokens(PROMPT, k=5)
        assert len(top) == 5
        for token, prob in top:
            assert isinstance(token, str)
            assert 0 < prob <= 1.0

    def test_classify_trajectory_on_synthetic(self, circuit):
        """Verify classify_trajectory works after real model comparison."""
        rows = []
        for step in range(10):
            rows.append({
                "step": step,
                "entropy": 4.0 - 0.1 * step,
                "top1": "kill" if step == 0 else "the",
                "top1_prob": 0.15,
                "top5_words": "kill|hit|punch|slap|hurt" if step == 0 else "the|a|to|and|of",
            })
        df = pd.DataFrame(rows)
        result = Circuit.classify_trajectory(df, base_top1="kill")
        assert result["signature"] == "transparent"
