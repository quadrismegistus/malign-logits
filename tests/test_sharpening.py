"""Guarantees of the sharpening baseline. It is a GATE, so its failure modes matter more
than its successes: a check that silently reports a flat step as sharpening, or that
certifies a reduction it cannot establish, is worse than no check.
"""
import math

import pytest

from malign_logits.sharpening import _entropy, reduces_to_sharpening, sharpening


class _WP:
    def __init__(self, probs, residual):
        self.probs = probs; self.residual = residual


class _Step:
    label = "sft->dpo"; family = "fam"
    def __init__(self, pre, post):
        class _C:
            def __init__(s, i): s.id = i
        self.pre, self.post = _C("PRE"), _C("POST")


def _install(monkeypatch, mapping, prompts=("a", "b", "c")):
    import malign_logits.movement as M
    import malign_logits.prompts as P
    monkeypatch.setattr(M, "word_probs",
                        lambda mid, t, *a, **k: mapping.get((mid, t)))
    class _P:
        def __init__(s, t): s.text = t
    monkeypatch.setattr(P.Prompts, "where",
                        staticmethod(lambda **kw: [_P(t) for t in prompts]))


def test_entropy_is_in_bits():
    """Four equiprobable outcomes is exactly 2 bits. A natural-log slip would give 1.386
    and every threshold in this module would be wrong by a constant factor."""
    assert _entropy({"a": .25, "b": .25, "c": .25, "d": .25}) == pytest.approx(2.0)
    assert _entropy({"a": 1.0}) == pytest.approx(0.0)


def test_a_concentrating_step_reports_negative_entropy_delta(monkeypatch):
    flat = _WP({"a": .25, "b": .25, "c": .25, "d": .25}, 0.10)
    peak = _WP({"a": .70, "b": .10, "c": .10, "d": .10}, 0.04)
    _install(monkeypatch, {("PRE", t): flat for t in "abc"} |
                          {("POST", t): peak for t in "abc"})
    s = sharpening(_Step(None, None))
    assert s["entropy_delta"] < 0, "concentration must be NEGATIVE entropy delta"
    assert s["top1_delta"] > 0
    assert s["residual_delta"] < 0
    assert not s["is_flat"]


def test_an_unchanged_step_is_flagged_FLAT(monkeypatch):
    """Archangel's role. A step that does not concentrate cannot manufacture divergence,
    and it is the control the whole gate rests on -- if this misfires the gate is blind."""
    d = _WP({"a": .3, "b": .3, "c": .4}, 0.12)
    _install(monkeypatch, {("PRE", t): d for t in "abc"} |
                          {("POST", t): d for t in "abc"})
    assert sharpening(_Step(None, None))["is_flat"] is True


def test_population_is_cells_measured_not_prompts_offered(monkeypatch):
    """A cell absent from either arm is dropped, and `n` must say so. A rate without its
    population is not a number, and this one is quoted as a roster-wide base rate."""
    d = _WP({"a": .5, "b": .5}, 0.1)
    _install(monkeypatch, {("PRE", "a"): d, ("POST", "a"): d,
                           ("PRE", "b"): d})          # 'b' missing post, 'c' missing both
    s = sharpening(_Step(None, None))
    assert s["n"] == 1, "only cells present in BOTH arms count"


def test_returns_none_rather_than_zeros_when_nothing_is_measurable(monkeypatch):
    _install(monkeypatch, {})
    assert sharpening(_Step(None, None)) is None


def test_reduces_to_sharpening_reports_and_does_not_conclude(monkeypatch):
    """The function must NOT return a verdict. With a roster of six a Spearman carries
    almost no power, and a boolean here would launder a weak correlation into a finding.
    It returns the table, the flat families, and their effects; the caller reads it.
    """
    d = _WP({"a": .5, "b": .5}, 0.1)
    peak = _WP({"a": .9, "b": .1}, 0.02)
    _install(monkeypatch, {("PRE", t): d for t in "abc"} |
                          {("POST", t): peak for t in "abc"})
    out = reduces_to_sharpening({"fam": 0.5}, {"fam": _Step(None, None)})
    assert set(out) == {"rows", "spearman", "n_families", "flat_families", "flat_effects"}
    assert not any(isinstance(v, bool) for k, v in out.items()
                   if k not in ("rows",)), "no verdict field"
    assert out["rows"][0]["entropy_delta"] < 0


def test_flat_family_effects_are_surfaced_by_name(monkeypatch):
    """The decisive column. If a family whose distributions DO NOT concentrate still shows
    the full effect, the effect is not reducible -- so the flat families and their effect
    sizes must both be reachable without recomputing anything."""
    d = _WP({"a": .5, "b": .5}, 0.1)
    _install(monkeypatch, {("PRE", t): d for t in "abc"} |
                          {("POST", t): d for t in "abc"})
    out = reduces_to_sharpening({"fam": 0.42}, {"fam": _Step(None, None)})
    assert out["flat_families"] == ["fam"]
    assert out["flat_effects"] == {"fam": 0.42}
