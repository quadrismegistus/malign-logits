"""The decomposition's guarantees. Each of these was watched to fail before it was kept.

The point of the class is EXACTNESS: if the parts do not sum to the whole, an analysis
that attributes divergence to fallers rather than to the tail is attributing it to
nothing. Two of these tests exist because the first implementation got them wrong --
`arrived` was documented as a share of `departed`, and the excess sum was computed over
P's keys alone, which broke the zero-sum identity by 0.006 on the first live cell.
"""
import math

import pytest

from malign_logits.movement import CANONICAL, DRAW, RESIDUAL_KEY, decompose, js_terms

PRE = {"kill": 0.30, "scream": 0.05, "hurt": 0.10, "leave": 0.08, "the": 0.02}
POST = {"scream": 0.28, "hurt": 0.09, "leave": 0.16, "the": 0.02, "cry": 0.03}
RP, RQ = 0.45, 0.42


def _js(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    d = 0.0
    for k in keys:
        a, b = p.get(k, 0.0) / sp, q.get(k, 0.0) / sq
        m = 0.5 * (a + b)
        if m <= 0:
            continue
        if a > 0:
            d += 0.5 * a * math.log2(a / m)
        if b > 0:
            d += 0.5 * b * math.log2(b / m)
    return d


def test_terms_sum_to_the_divergence():
    """The whole premise. A per-word attribution is only meaningful if it is complete."""
    p = {**PRE, RESIDUAL_KEY: RP}
    q = {**POST, RESIDUAL_KEY: RQ}
    assert sum(js_terms(p, q).values()) == pytest.approx(_js(p, q), abs=1e-12)


def test_every_term_is_non_negative():
    """Each word's contribution is a divergence in its own right, so none may be negative
    -- a negative term would let a stratum's total be cancelled by another's."""
    for v in js_terms({**PRE, RESIDUAL_KEY: RP}, {**POST, RESIDUAL_KEY: RQ}).values():
        assert v >= -1e-15


def test_roles_partition_the_total():
    d = decompose(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)
    parts = d["js_fallers"] + d["js_risers"] + d["js_tail"] + d["js_other"]
    assert parts == pytest.approx(d["js_total"], abs=1e-12)


def test_each_role_term_is_pinned_to_its_own_words():
    """`js_other` is a REMAINDER, so the sum-to-total test above is vacuous for any single
    term: zeroing `js_tail` just relocates its mass into `js_other` and the partition
    still balances. A mutant that did exactly that passed all nine of the other tests.
    Each named term is therefore checked against the words it claims to cover.
    """
    from malign_logits.movement import movement
    p = {**PRE, RESIDUAL_KEY: RP}
    q = {**POST, RESIDUAL_KEY: RQ}
    terms = js_terms(p, q)
    m = movement(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)
    d = decompose(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)

    assert terms[RESIDUAL_KEY] > 0, "fixture must actually move the tail, or this is null"
    assert d["js_tail"] == pytest.approx(terms[RESIDUAL_KEY], abs=1e-12)
    assert m.fallers and m.risers, "fixture must produce both roles"
    assert d["js_fallers"] == pytest.approx(sum(terms[w] for w in m.fallers), abs=1e-12)
    assert d["js_risers"] == pytest.approx(sum(terms[w] for w in m.risers), abs=1e-12)
    named = set(m.fallers) | set(m.risers) | {RESIDUAL_KEY}
    assert d["js_other"] == pytest.approx(
        sum(v for k, v in terms.items() if k not in named), abs=1e-12)


def test_js_total_equals_the_plain_figure():
    """The decomposition must not quietly compute a DIFFERENT divergence from `cell.js()`
    -- if it did, `tail_share` would be a share of something nobody reports."""
    d = decompose(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)
    assert d["js_total"] == pytest.approx(
        _js({**PRE, RESIDUAL_KEY: RP}, {**POST, RESIDUAL_KEY: RQ}), abs=1e-12)


def test_excess_is_zero_sum_over_survivors():
    """THE identity that makes `selectivity` not-a-share. sum_non-fallers null == R and
    sum_non-fallers Q == R, so the excesses cancel. If this ever fails, the docstring's
    warning that selectivity has no upper bound has become a lie in the other direction.
    """
    from malign_logits.movement import movement
    m = movement(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)
    fall = set(m.fallers)
    P = {**PRE, RESIDUAL_KEY: RP}
    Q = {**POST, RESIDUAL_KEY: RQ}
    R = 1.0 - sum(Q.get(w, 0.0) for w in fall)
    S = sum(P.get(k, 0.0) for k in set(P) | set(Q) if k not in fall)
    exc = sum(Q.get(k, 0.0) - P.get(k, 0.0) * (R / S)
              for k in set(P) | set(Q) if k not in fall)
    assert exc == pytest.approx(0.0, abs=1e-9)


def test_post_only_words_are_counted_in_the_excess():
    """`cry` exists only in POST. Iterating PRE's keys skips it, and skipping it is what
    broke the zero-sum identity by 0.006 on the first cell measured."""
    assert "cry" in POST and "cry" not in PRE
    d = decompose(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)
    # `captured` is arrived / all positive excess; missing post-only positive excess
    # would shrink the denominator and push the ratio above 1.
    assert d["captured"] is None or d["captured"] <= 1.0 + 1e-9


def test_captured_is_a_share_and_selectivity_is_not():
    d = decompose(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)
    assert 0.0 <= d["captured"] <= 1.0
    assert 0.0 <= d["concentration"] <= 1.0
    assert 0.0 <= d["tail_share"] <= 1.0
    assert d["selectivity"] > 0  # bounded below by nothing useful, and NOT by 1


def test_the_rule_is_honoured():
    """DRAW does not test risers against the null, so it flags more of them -- the
    decomposition must reflect the rule it was given, not a hardcoded one."""
    c = decompose(PRE, POST, CANONICAL, residual_pre=RP, residual_post=RQ)
    w = decompose(PRE, POST, DRAW, residual_pre=RP, residual_post=RQ)
    assert w["n_risers"] >= c["n_risers"]
    assert c["js_total"] == pytest.approx(w["js_total"], abs=1e-12)  # total is rule-free


def test_identical_distributions_have_no_divergence_and_no_parts():
    d = decompose(PRE, dict(PRE), CANONICAL, residual_pre=RP, residual_post=RP)
    assert d["js_total"] == pytest.approx(0.0, abs=1e-12)
    assert d["n_fallers"] == 0
    assert d["tail_share"] is None or d["tail_share"] == pytest.approx(0.0, abs=1e-9)
