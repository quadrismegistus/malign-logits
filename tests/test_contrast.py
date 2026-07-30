"""Guarantees of the comparison layer. Reference values are hand-computed, not
regenerated from the implementation -- a test that asserts what the code already does
confirms nothing, and this file exists because four scripts had each drifted their own
copy of these statistics.
"""
import math

import pytest

from malign_logits.contrast import (METRICS, Contrast, by_field, by_role, rank_sum,
                                    sign_test, _measure)


# --- the tests themselves, against values worked out by hand ----------------

def test_sign_test_matches_the_exact_binomial():
    """3 of 4 positive. Two-sided exact p = 2 * (C(4,0)+C(4,1)) / 2^4 = 10/16."""
    k, n, p = sign_test([1.0, 1.0, 1.0, -1.0])
    assert (k, n) == (3, 4)
    assert p == pytest.approx(0.625)


def test_sign_test_drops_exact_zeros_from_its_population():
    """The population is NON-ZERO differences. On a late-chain step most cells have no
    fallers in either arm, so most differences are exactly zero; counting them would
    inflate n tenfold and shrink p accordingly."""
    k, n, _ = sign_test([1.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    assert n == 3, "zeros must not enter the population"
    assert k == 2


def test_sign_test_with_no_usable_differences_returns_no_p():
    assert sign_test([0.0, 0.0])[2] is None
    assert sign_test([])[2] is None


def test_rank_sum_u_on_complete_separation():
    """a = [1,2,3] entirely below b = [4,5,6]. Rank sum of a is 6, so U = 6 - 3*4/2 = 0."""
    U, z, p = rank_sum([1, 2, 3], [4, 5, 6])
    assert U == pytest.approx(0.0)
    assert z < 0 and p is not None


def test_rank_sum_is_symmetric_in_magnitude():
    U1, z1, p1 = rank_sum([1, 2, 3], [4, 5, 6])
    U2, z2, p2 = rank_sum([4, 5, 6], [1, 2, 3])
    assert z1 == pytest.approx(-z2)
    assert p1 == pytest.approx(p2)


def test_rank_sum_corrects_for_ties():
    """All-tied samples carry no information, and the tie correction must drive the
    variance to zero rather than reporting a confident null. Without the correction the
    variance stays positive and p comes back as a real number on data that has none."""
    U, z, p = rank_sum([1, 1, 1], [1, 1, 1])
    assert p is None, "an all-tied comparison must not yield a p-value"


def test_identical_samples_are_not_significant():
    _, _, p = rank_sum([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert p > 0.9


# --- paired vs stratified must stay distinct --------------------------------

def _paired(a, b):
    return Contrast("A", a, "B", b, paired=True, metric="js", frame="t")


def _strat(a, b):
    return Contrast("A", a, "B", b, paired=False, metric="js", frame="t")


def test_paired_runs_the_sign_test_and_stratified_runs_rank_sum():
    assert _paired([3, 3, 3], [1, 1, 1]).test()["test"] == "sign"
    assert _strat([3, 3, 3], [1, 1, 1]).test()["test"] == "rank_sum"


def test_within_unit_differences_are_refused_on_a_stratified_frame():
    """The two samples are independent there, so element i of one has no partner in the
    other. Zipping them anyway would invent pairings out of list order."""
    with pytest.raises(ValueError, match="paired"):
        _strat([1, 2, 3], [4, 5, 6]).diffs


def test_stratified_reports_both_sample_sizes():
    """`institutional vs neutral` is 55 against 135. A single n hides the asymmetry that
    decides whether the test is trustworthy.

    `.n` is asserted too, and not only `n_display`: a mutant returning `len(self.a)` for
    both branches -- silently reporting 55 as the population of a 190-observation
    comparison -- passed every other test in this file.
    """
    c = _strat([1] * 55, [2] * 135)
    assert c.n_display == "55/135"
    assert c.n == 190, "a stratified population is BOTH samples, not the first"
    assert c.summary()["n"] == 190
    assert c.summary()["n_a"] == 55 and c.summary()["n_b"] == 135
    assert _paired([1] * 12, [2] * 12).n == 12


def test_stratified_has_no_median_difference():
    """None, not nan and not zero -- a missing quantity must not read as a computed one."""
    assert _strat([1, 2], [3, 4]).summary().get("median_diff") is None
    assert _paired([1, 2], [3, 4]).summary()["median_diff"] is not None


def test_paired_n_is_pairs_not_prompts():
    assert _paired([1, 2, 3], [4, 5, 6]).n == 3


def test_direction_names_the_larger_side():
    assert "A > B" in _paired([5, 5, 5], [1, 1, 1]).test()["direction"]
    assert "B > A" in _paired([1, 1, 1], [5, 5, 5]).test()["direction"]


def test_empty_sample_reports_nothing_rather_than_crashing():
    r = _paired([], []).test()
    assert r["p"] is None and r["n"] == 0 and "note" in r


# --- the metric contract ----------------------------------------------------

def test_unknown_metric_raises_at_the_call_site():
    """A typo must fail loudly. Returning a column of Nones would look like missing data
    and be indistinguishable from a genuinely absent measurement."""
    class _S:
        label = "x"
    with pytest.raises(ValueError, match="unknown metric"):
        by_field(_S(), "a", "b", metric="jsd")
    with pytest.raises(ValueError, match="unknown metric"):
        by_role(_S(), "a", "b", metric="departure")


def test_every_advertised_metric_is_reachable():
    """METRICS is the contract. `js` and `l1` come off the Cell; the rest must be keys
    that `decompose` actually returns, or the name is a promise the layer cannot keep."""
    from malign_logits.movement import CANONICAL, decompose
    d = decompose({"a": 0.3, "b": 0.1}, {"a": 0.05, "b": 0.3},
                  CANONICAL, residual_pre=0.6, residual_post=0.65)
    for m in METRICS:
        if m in ("js", "l1"):
            continue
        assert m in d, f"{m} is advertised in METRICS but decompose does not return it"


def test_measure_reports_none_on_a_mixed_rule_version_rather_than_raising():
    """One bad arm must not kill a whole frame, but it must also never enter a number.
    None here, counted as a drop by the caller."""
    class _Cell:
        is_present = True
        def js(self):
            raise ValueError("rule_version mismatch")
    assert _measure(_Cell(), "js", None) is None


def test_measure_reports_none_for_an_absent_cell():
    class _Cell:
        is_present = False
    assert _measure(_Cell(), "js", None) is None


# --- restricting the population --------------------------------------------

def test_where_restricts_the_population_and_changes_the_answer():
    """A frame without `where` POOLS DESIGNS. `MARKED`/`UNMARKED` is a role used by three
    findings; F36 owns 34 clean pairs and F13 owns 17, and they answer differently
    (F13 p=0.049, F36 p=0.86, pooled p=0.25 -- the pool hides both). This test asserts
    the mechanism rather than those numbers: restricting must yield a strict subset.
    """
    from malign_logits.contrast import _population
    allp = list(_population("en", None))
    f36 = list(_population("en", {"finding": "F36"}))
    assert 0 < len(f36) < len(allp)
    assert {p.id for p in f36} < {p.id for p in allp}
    assert all(p.row.get("finding") == "F36" for p in f36)


def test_where_composes_with_language():
    from malign_logits.contrast import _population
    en = list(_population("en", {"finding": "F36"}))
    zh = list(_population("zh", {"finding": "F36"}))
    assert en and zh
    assert all(p.row.get("language") == "en" for p in en)
    assert all(p.row.get("language") == "zh" for p in zh)
    assert not ({p.id for p in en} & {p.id for p in zh})
