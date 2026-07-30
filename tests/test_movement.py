"""Tests for malign_logits.movement.

Every one of these was WATCHED TO FAIL before it was kept -- the day's standing rule:
a checker is not a checker until it has been seen to fail. Where a test could pass
vacuously the fixture is built so that it cannot.
"""
import math

import pytest

from malign_logits.movement import (CANONICAL, DRAW, RESIDUAL_KEY, Rule, movement,
                                    movement_from_logits)


def test_the_null_is_what_separates_redistribution_from_bookkeeping():
    """THE POINT OF THE WHOLE MODULE, in one fixture.

    `bystander` gains mass ONLY because the faller's mass was removed and everything
    renormalised. `real` gains more than that. A rule without the null calls both risers;
    the canonical rule calls only `real`.
    """
    pre = {"faller": 0.40, "real": 0.10, "bystander": 0.30, "quiet": 0.20}
    # faller 0.40 -> 0.10, so inflation = 0.90/0.60 = 1.5 and bystander's null is 0.45.
    # It rises to 0.42: UP in absolute terms, but SHORT of what renormalisation alone
    # would have handed it. Deliberately not 0.45 -- a word sitting exactly on the null
    # is decided by floating point, which is a real property of the rule and is pinned
    # by test_a_word_exactly_on_the_null_is_a_knife_edge below rather than hidden here.
    post = {"faller": 0.10, "real": 0.33, "bystander": 0.42, "quiet": 0.15}
    m = movement(pre, post, CANONICAL)

    assert m.fallers == ["faller"]
    assert "real" in m.risers
    assert "bystander" not in m.risers, (
        "bystander rose 0.30->0.45, exactly its renormalisation share, and must NOT be a "
        f"riser. null={m.null.get('bystander'):.4f} inflation={m.inflation:.4f}")

    # And the same input under DRAW, which has no null, calls it a riser -- which is the
    # difference the two rules exist to make visible.
    assert "bystander" in movement(pre, post, DRAW).risers


def test_draw_and_canonical_disagree_and_that_is_the_reason_both_ship():
    pre = {"f": 0.40, "a": 0.20, "b": 0.20, "c": 0.20}
    post = {"f": 0.05, "a": 0.30, "b": 0.30, "c": 0.35}
    c, d = movement(pre, post, CANONICAL), movement(pre, post, DRAW)
    assert set(d.risers) - set(c.risers), (
        "DRAW must admit at least one riser CANONICAL rejects on this fixture; if not, "
        "the fixture no longer distinguishes the rules and the test is vacuous")


def test_fallers_are_not_tested_against_the_null_and_the_docs_say_so():
    """The declared asymmetry. A word can halve purely because mass left elsewhere, so a
    faller is a bare ratio rule -- and nothing downstream may call fallers
    'beyond renormalisation'."""
    pre = {"x": 0.50, "y": 0.50}
    post = {"x": 0.20, "y": 0.80}
    m = movement(pre, post, CANONICAL)
    assert "x" in m.fallers
    assert "x" not in m.null, "fallers are excluded from the null by construction"
    assert "beyond renormalisation" not in (m.diagnostics.get("rule") or "")


def test_residual_participates_instead_of_being_dropped_or_renormalised_away():
    """On true_word_probs a quarter of the distribution is in the residual. Dropping it
    inflates every survivor's null; renormalising deletes mass that left the scored set.
    Neither may happen silently."""
    pre = {"f": 0.30, "a": 0.20, RESIDUAL_KEY: 0.50}
    post = {"f": 0.05, "a": 0.25, RESIDUAL_KEY: 0.70}

    with_res = movement(pre, post, CANONICAL)
    without = movement({k: v for k, v in pre.items() if k != RESIDUAL_KEY},
                       {k: v for k, v in post.items() if k != RESIDUAL_KEY}, CANONICAL)

    assert with_res.inflation != pytest.approx(without.inflation), (
        "dropping the residual must change the null; if it does not, the residual is "
        "not participating and the approximation is silently wrong")
    assert with_res.diagnostics["residual_share"] == pytest.approx(0.70)
    assert with_res.diagnostics["exact_null"] is False
    assert RESIDUAL_KEY not in with_res.risers + with_res.fallers


def test_residual_can_never_be_a_faller():
    """An undifferentiated bucket has no word to fall. Letting it be one would make tail
    movement masquerade as a lexical event."""
    pre = {"a": 0.10, RESIDUAL_KEY: 0.90}
    post = {"a": 0.85, RESIDUAL_KEY: 0.15}
    m = movement(pre, post, CANONICAL)
    assert RESIDUAL_KEY not in m.fallers


def test_logits_path_reports_an_exact_null_and_word_probs_path_does_not():
    lg_pre = [math.log(x) for x in (0.4, 0.3, 0.2, 0.1)]
    lg_post = [math.log(x) for x in (0.1, 0.5, 0.3, 0.1)]
    lm = movement_from_logits(lg_pre, lg_post, CANONICAL, labels=list("fabc"))
    assert lm.diagnostics["exact_null"] is True

    wm = movement({"f": .4, "a": .3, "b": .2, "c": .1},
                  {"f": .1, "a": .5, "b": .3, "c": .1}, CANONICAL)
    assert wm.diagnostics["exact_null"] is False
    # Same numbers by two routes: the rule must not depend on which door you came in.
    assert sorted(lm.fallers) == sorted(wm.fallers)
    assert sorted(lm.risers) == sorted(wm.risers)


def test_top_riser_ranks_by_excess_not_delta_when_the_null_was_computed():
    """Ranking risers by delta re-introduces exactly what the null removes: a word with a
    large pre-probability gets a large renormalisation gift and wins on delta while
    adding nothing beyond bookkeeping."""
    # excess = delta - P*(inflation-1), so a large pre-probability is charged for the
    # renormalisation gift it receives. Here inflation = 0.98/0.80 = 1.225: `big` gains
    # more absolutely (+0.13 vs +0.05) and less beyond the null (+0.018 vs +0.039).
    pre = {"f": 0.20, "big": 0.50, "small": 0.05, "other": 0.25}
    post = {"f": 0.02, "big": 0.63, "small": 0.10, "other": 0.25}
    m = movement(pre, post, CANONICAL)
    assert {"big", "small"} <= set(m.risers), (
        f"fixture must make BOTH risers or it cannot test the ranking; got {m.risers}")
    assert m.delta["big"] > m.delta["small"]
    assert m.excess["small"] > m.excess["big"], "fixture no longer separates the two"
    assert m.top_riser() == "small"


def test_a_word_exactly_on_the_null_is_a_knife_edge_and_this_is_known():
    """PINNED, NOT FIXED. `f13_movement_table.py` tests `Q > null` with a bare comparison
    and this module ports that exactly. A word landing precisely on its renormalisation
    expectation is therefore decided by floating-point noise -- 0.45 against a computed
    0.4499999999999999 counts as a riser.

    A tolerance would be defensible and is NOT applied, because the package and the
    script would then disagree, which is the divergence this module exists to end. The
    behaviour is recorded here so it is known rather than latent; changing it is a
    decision about the RULE, to be made once and in both places.
    """
    pre = {"faller": 0.40, "real": 0.10, "onnull": 0.30, "quiet": 0.20}
    post = {"faller": 0.10, "real": 0.30, "onnull": 0.45, "quiet": 0.15}
    m = movement(pre, post, CANONICAL)
    assert m.null["onnull"] == pytest.approx(0.45)
    assert "onnull" in m.risers, (
        "documents the knife edge: exactly-on-null currently counts as a riser via float "
        "representation. If this ever flips, the rule changed and both implementations "
        "must move together")


def test_rule_is_a_declared_object_so_thresholds_are_cited_not_retyped():
    assert CANONICAL.null_test is True and DRAW.null_test is False
    assert CANONICAL.min_prob == 0.003 and CANONICAL.delta == 0.003
    assert DRAW.floor == 0.005 and DRAW.theta == 0.001
    custom = Rule(name="strict", min_prob=0.01, fall_ratio=0.25, delta=0.01,
                  null_test=True)
    m = movement({"f": 0.5, "a": 0.5}, {"f": 0.05, "a": 0.95}, custom)
    assert m.rule.name == "strict"


# ---------------------------------------------------------------------------
# Cache accessors. These need the cache; they skip cleanly without it.
# ---------------------------------------------------------------------------

def _cache_or_skip():
    try:
        from malign_logits.cache import get_cache
        return get_cache()
    except Exception:
        pytest.skip("cache unavailable")


def test_word_probs_sums_token_paths_instead_of_overwriting_them():
    """THE DEFECT THIS ACCESSOR EXISTS TO PREVENT.

    The payload is one row per (word, FIRST TOKEN) and those rows are a PARTITION: summed
    over every row, plus the residual, they come to 1.0. `{r["word"]: r["p"] for r in
    rows}` keeps the last path and drops the rest -- silently, and on 20% of payloads.
    """
    from malign_logits.movement import word_probs
    cm = _cache_or_skip()
    model, prompt = ("ContextualAI/archangel_sft_pythia2-8b",
                     "他把身体贴在她身上，低声说")
    payload = cm.get_true_word_probs(model, prompt)
    if payload is None:
        pytest.skip("reference cell not cached")
    rows = payload["rows"]
    naive = {r["word"]: r["p"] for r in rows}
    w = word_probs(model, prompt, cache=cm)

    assert w.collapsed > 0, "reference cell no longer has duplicate surfaces; pick another"
    assert w.total == pytest.approx(1.0, abs=1e-6), (
        "rows plus residual must partition the distribution; if not, summing is the "
        "wrong fold and this accessor's premise is broken")
    assert sum(naive.values()) < sum(w.probs.values()), (
        "the naive comprehension must lose mass here, or the test is vacuous")


def test_movers_refuses_a_mixed_rule_version():
    """A v1 arm against a v3 arm books an INSTRUMENT CHANGE as alignment movement."""
    from malign_logits import movement as mv
    calls = {}

    def fake(model, prompt, theta=0.001, mode="raw", cache=None):
        return mv.WordProbs(probs={"a": 0.5}, residual=0.5,
                            rule_version=calls.setdefault(model, 1 if "pre" in model else 3))

    orig = mv.word_probs
    mv.word_probs = fake
    try:
        with pytest.raises(ValueError, match="rule_version mismatch"):
            mv.movers("pre/model", "post/model", "p")
        assert mv.movers("pre/model", "post/model", "p",
                         allow_mixed_rule_version=True) is not None
    finally:
        mv.word_probs = orig
