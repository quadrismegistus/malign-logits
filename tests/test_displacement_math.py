"""Unit tests for the displacement / repression math.

These functions back the paper's confirmed findings (SFT/DPO division of
labour, displacement mass, id-scores) yet had zero coverage. They are pure
computation — no model downloads — so they run in CI. A deterministic fake
tokenizer lets us assert exact outputs rather than "doesn't crash".

Run: pytest tests/test_displacement_math.py -v
"""

import math

import numpy as np
import pytest
import torch


# ── Deterministic fake tokenizer ───────────────────────────────

class FakeTokenizer:
    """Fixed word→token-id map so word-scoring is exactly assertable.

    ' hug' → [40, 41] where id 40 decodes to '' (a leading byte-pair marker
    with no visible text) to exercise the skip-leading-blank branch.
    """
    _ENC = {" kill": [10], " scream": [20], " the": [30], " hug": [40, 41]}
    _DEC = {10: "kill", 20: "scream", 30: "the", 40: "", 41: "hug"}

    def encode(self, text, add_special_tokens=False):
        return list(self._ENC.get(text, []))

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._DEC.get(int(i), "") for i in ids)


# ── score_words_from_logits ─────────────────────────────────────

def _logits(size, highs=None):
    v = torch.full((size,), -10.0)
    for idx, val in (highs or {}).items():
        v[idx] = val
    return v


def test_score_words_normalized_and_ordered():
    from malign_logits.core import score_words_from_logits
    tok = FakeTokenizer()
    logits = _logits(50, {10: 8.0, 20: 4.0, 30: 0.0})  # kill > scream > the
    scores = score_words_from_logits(logits, tok, ["kill", "scream", "the"])

    assert set(scores) == {"kill", "scream", "the"}
    assert abs(sum(scores.values()) - 1.0) < 1e-6           # normalized
    assert list(scores) == ["kill", "scream", "the"]         # sorted descending
    assert scores["kill"] > scores["scream"] > scores["the"]


def test_score_words_multitoken_skips_blank_first_token():
    from malign_logits.core import score_words_from_logits
    tok = FakeTokenizer()
    # ' hug' → [40, 41]; id 40 decodes '' so score must come from id 41.
    logits = _logits(50, {41: 9.0, 10: 1.0})
    scores = score_words_from_logits(logits, tok, ["hug", "kill"])
    assert scores["hug"] > scores["kill"]                    # read id 41, not 40


def test_score_words_empty_returns_empty():
    from malign_logits.core import score_words_from_logits
    tok = FakeTokenizer()
    assert score_words_from_logits(_logits(50), tok, []) == {}
    assert score_words_from_logits(_logits(50), tok, ["unknown_word"]) == {}


# ── compute_repression ──────────────────────────────────────────

def test_compute_repression_flags_and_deltas():
    from malign_logits.analysis import compute_repression
    ego = {"kill": 0.40, "calm": 0.10}
    superego = {"kill": 0.05, "calm": 0.30}
    base = {"kill": 0.50, "calm": 0.08}
    df = compute_repression(ego, superego, base_words=base, threshold=0.01)

    row = df.set_index("word")
    # delta = ego - superego, exactly
    assert abs(row.loc["kill", "delta"] - 0.35) < 1e-9
    assert abs(row.loc["calm", "delta"] + 0.20) < 1e-9
    # kill is suppressed by the second distribution; calm is amplified
    assert bool(row.loc["kill", "repressed"]) and not bool(row.loc["kill", "amplified"])
    assert bool(row.loc["calm", "amplified"]) and not bool(row.loc["calm", "repressed"])
    # default column names + base column threaded through
    assert row.loc["kill", "sft"] == 0.40 and row.loc["kill", "dpo"] == 0.05
    assert row.loc["kill", "base"] == 0.50
    # ratio sign: ego>superego positive, else negative
    assert row.loc["kill", "ratio"] > 0 and row.loc["calm", "ratio"] < 0


def test_compute_repression_custom_columns():
    from malign_logits.analysis import compute_repression
    df = compute_repression({"x": 0.3}, {"x": 0.1}, col_a="base", col_b="ego")
    assert "base" in df.columns and "ego" in df.columns


# ── compute_id ──────────────────────────────────────────────────

def test_compute_id_excludes_unrepressed():
    from malign_logits.analysis import compute_id
    base = {"kill": 0.5, "hug": 0.1}
    ego = {"kill": 0.4, "hug": 0.2}
    superego = {"kill": 0.05, "hug": 0.2}   # hug: ego==superego → repression 0
    id_scores, analysis = compute_id(base, ego, superego)
    assert "kill" in id_scores
    assert "hug" not in id_scores           # repression <= 0 excluded
    assert analysis["kill"]["repression"] == pytest.approx(0.35)


def test_compute_id_drive_weighting_orders_scores():
    from malign_logits.analysis import compute_id
    # a and b have identical repression (0.3); a has far more base drive.
    base = {"a": 0.9, "b": 0.001}
    ego = {"a": 0.5, "b": 0.5}
    superego = {"a": 0.2, "b": 0.2}
    id_scores, _ = compute_id(base, ego, superego)
    assert id_scores["a"] > id_scores["b"]          # drive weight breaks the tie
    assert list(id_scores)[0] == "a"                # sorted descending


# ── top_movers ──────────────────────────────────────────────────

def test_top_movers_direction():
    from malign_logits.analysis import top_movers
    tok = FakeTokenizer()
    a = _logits(50, {10: 10.0})   # 'kill' high in A
    b = _logits(50, {20: 10.0})   # 'scream' high in B
    out = top_movers(a, b, tok, k=5)
    repressed_words = [t[0] for t in out["repressed"]]
    amplified_words = [t[0] for t in out["amplified"]]
    assert "kill" in repressed_words        # A wants it, B doesn't
    assert "scream" in amplified_words      # B wants it, A doesn't
    # delta sign convention: repressed positive (prob_a > prob_b)
    assert out["repressed"][0][3] > 0
    assert out["amplified"][0][3] < 0


# ── distribution_metrics ────────────────────────────────────────

def test_distribution_metrics_three_layer():
    from malign_logits.analysis import distribution_metrics, distribution_entropy
    torch.manual_seed(0)
    base = torch.randn(200)
    ego = torch.randn(200)
    superego = torch.randn(200)
    m = distribution_metrics(base, ego, superego)

    for key in ("js_base_ego", "js_ego_superego", "entropy_drop_sft", "entropy_drop_dpo"):
        assert key in m
    # entropy_drop_sft is exactly H(base) - H(ego)
    assert m["entropy_drop_sft"] == pytest.approx(
        distribution_entropy(base) - distribution_entropy(ego), abs=1e-6)


def test_distribution_metrics_two_layer():
    from malign_logits.analysis import distribution_metrics
    base = torch.randn(200)
    superego = torch.randn(200)
    m = distribution_metrics(base, None, superego)
    assert "kl_base_superego" in m
    assert "entropy_drop_alignment" in m
    assert "js_base_ego" not in m           # no ego layer


def test_distribution_metrics_identical_is_zero():
    from malign_logits.analysis import distribution_metrics
    x = torch.randn(200)
    m = distribution_metrics(x, x, x)
    assert m["js_base_superego"] < 1e-6
    assert abs(m["entropy_drop_sft"]) < 1e-6


# ── divergence invariants ───────────────────────────────────────

def test_kl_nonnegative_and_zero_on_self():
    from malign_logits.analysis import kl_divergence
    a, b = torch.randn(150), torch.randn(150)
    assert kl_divergence(a, b) >= -1e-9
    assert kl_divergence(a, a) < 1e-6


def test_js_symmetric():
    from malign_logits.analysis import js_divergence
    a, b = torch.randn(150), torch.randn(150)
    assert js_divergence(a, b) == pytest.approx(js_divergence(b, a), abs=1e-9)


def test_top_k_overlap_bounds():
    from malign_logits.analysis import top_k_overlap
    a, b = torch.randn(150), torch.randn(150)
    assert top_k_overlap(a, a, k=50) == 1.0
    assert 0.0 <= top_k_overlap(a, b, k=50) <= 1.0


def test_align_logits_truncates_to_min_vocab():
    from malign_logits.analysis import _align_logits
    a = torch.randn(300)
    b = torch.randn(200)
    aa, bb = _align_logits(a, b)
    assert aa.shape[-1] == bb.shape[-1] == 200


# ── cross-implementation consistency (torch analysis vs numpy metrics) ──

def test_torch_numpy_js_agree():
    """The two JS implementations must agree — equal AND unequal length,
    now that both align by truncate-to-min."""
    from malign_logits.analysis import js_divergence as js_t
    from malign_logits.metrics import js_divergence as js_n
    rng = np.random.default_rng(7)
    for na, nb in [(2000, 2000), (2000, 1500)]:
        a, b = rng.normal(size=na), rng.normal(size=nb)
        assert js_t(torch.tensor(a), torch.tensor(b)) == pytest.approx(js_n(a, b), abs=1e-6)


def test_js_from_probs_matches_logit_wrapper():
    """js_divergence(logits) == js_from_probs(softmax(logits))."""
    from malign_logits.analysis import js_divergence, js_from_probs
    a, b = torch.randn(400), torch.randn(400)
    p = torch.softmax(a, dim=-1)
    q = torch.softmax(b, dim=-1)
    assert js_divergence(a, b) == pytest.approx(js_from_probs(p, q), abs=1e-9)
