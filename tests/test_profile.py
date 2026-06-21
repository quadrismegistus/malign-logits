"""Tests for CircuitProfile data structure."""
import tempfile
import pytest
from malign_logits.profile import (
    CircuitProfile, FamilyMetadata, NodeProfile, EdgeProfile, ModeProfile
)


@pytest.fixture
def sample_profile():
    meta = FamilyMetadata(
        family="test", scale="7B", base_corpus="Dolma",
        alignment_data="Tulu 3", alignment_method="SFT+DPO",
        n_layers=3, layer_names=["base", "sft", "dpo"],
        has_chat_template=True, country="US", org="Allen AI",
        open_weight=True, open_data=True,
    )
    profile = CircuitProfile(metadata=meta)
    profile.nodes.append(NodeProfile(
        checkpoint="base", prompt="anger", entropy=5.19,
        effective_vocab=179, top_k=[("kill", 0.098), ("hit", 0.062)],
        violence_loading=0.147, procedural_loading=0.062,
        argmax_token="kill", argmax_prob=0.098,
    ))
    profile.nodes.append(NodeProfile(
        checkpoint="dpo", prompt="anger", entropy=2.96,
        effective_vocab=19, top_k=[("______", 0.31), ("break", 0.08)],
        violence_loading=-0.019, procedural_loading=0.027,
        argmax_token="______", argmax_prob=0.31,
    ))
    profile.edges.append(EdgeProfile(
        from_checkpoint="base", to_checkpoint="dpo", prompt="anger",
        js_divergence=0.41, argmax_change="kill → ______",
        top_gainers=[("______", 0.21), ("break", 0.02)],
        top_losers=[("kill", -0.09), ("hit", -0.05)],
        signature="foreclosure", sft_share=0.92, delta_entropy=-2.23,
    ))
    profile.modes.append(ModeProfile(
        mode="raw", checkpoint="dpo", prompt="anger",
        entropy=2.96, top1_token="______", top1_prob=0.31,
    ))
    profile.modes.append(ModeProfile(
        mode="chat", checkpoint="dpo", prompt="anger",
        entropy=4.42, top1_token="...", top1_prob=0.12,
    ))
    return profile


def test_node_lookup(sample_profile):
    n = sample_profile.node("base", "anger")
    assert n is not None
    assert n.argmax_token == "kill"
    assert n.violence_loading == pytest.approx(0.147)


def test_node_miss(sample_profile):
    assert sample_profile.node("base", "worker") is None


def test_edge_lookup(sample_profile):
    e = sample_profile.edge("base", "dpo", "anger")
    assert e is not None
    assert e.signature == "foreclosure"
    assert e.sft_share == pytest.approx(0.92)


def test_csv_roundtrip(sample_profile):
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_profile.to_csv(tmpdir)
        loaded = CircuitProfile.from_csv(tmpdir, "test")

        assert loaded.metadata.family == "test"
        assert loaded.metadata.scale == "7B"
        assert len(loaded.nodes) == 2
        assert len(loaded.edges) == 1
        assert len(loaded.modes) == 2

        n = loaded.node("base", "anger")
        assert n.argmax_token == "kill"
        assert n.violence_loading == pytest.approx(0.147, abs=1e-4)

        e = loaded.edge("base", "dpo", "anger")
        assert e.signature == "foreclosure"
        assert e.top_gainers[0][0] == "______"


def test_worker_summary(sample_profile):
    # Add worker nodes
    sample_profile.nodes.append(NodeProfile(
        checkpoint="base", prompt="worker", entropy=3.87,
        effective_vocab=48, top_k=[("quit", 0.11)],
        violence_loading=-0.034, procedural_loading=0.184,
        argmax_token="quit", argmax_prob=0.11,
    ))
    sample_profile.nodes.append(NodeProfile(
        checkpoint="dpo", prompt="worker", entropy=3.50,
        effective_vocab=33, top_k=[("seek", 0.13)],
        violence_loading=-0.037, procedural_loading=0.363,
        argmax_token="seek", argmax_prob=0.13,
    ))
    sample_profile.edges.append(EdgeProfile(
        from_checkpoint="base", to_checkpoint="dpo", prompt="worker",
        js_divergence=0.19, argmax_change="quit → seek",
        top_gainers=[("seek", 0.12)], top_losers=[("quit", -0.10)],
        signature="de_foreclosure", sft_share=None, delta_entropy=-0.37,
    ))
    ws = sample_profile.worker_summary()
    assert ws["base_argmax"] == "quit"
    assert ws["aligned_argmax"] == "seek"
    assert ws["mechanism"] == "de_foreclosure"
    assert ws["procedural_loading"] == pytest.approx(0.363)
