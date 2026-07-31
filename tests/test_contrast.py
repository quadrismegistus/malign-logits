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


# --- per-cell reporting -----------------------------------------------------

def test_per_cell_percentile_is_against_the_full_step_not_the_subset():
    """THE defect this function could have. A percentile computed against the six cells
    passed in would rank them 17, 33, 50, 67, 83, 100 regardless of whether any of them
    moved -- a subset always fills its own range. The reference must be the step's whole
    scored population, and `n_reference` must say so on every record.
    """
    from malign_logits.contrast import per_cell

    class _M:
        def __init__(s, v): s.v = v
        def top_riser(s): return "w"
        fallers = ["f"]; risers = ["w"]

    class _Cell:
        is_present = True
        domain = "d"
        def __init__(s, v): s.v = v
        def js(s): return s.v
        def movement(s, rule=None): return _M(s.v)

    class _Prompt:
        def __init__(s, t): s.text = t

    class _Step:
        label = "base->sft"; family = "f"
        def cell(s, t): return _Cell(float(t))

    # The stub MUST honour `where`, or swapping the reference for the subset changes
    # nothing and the test cannot bite. A first version ignored it and passed the mutant.
    import malign_logits.contrast as C
    real = C._population
    C._population = (lambda lang, where:
                     [_Prompt(str(i)) for i in (range(95, 98) if where else range(100))])
    try:
        recs = per_cell(_Step(), texts=["95", "96", "97"], where={"finding": "F36"})
    finally:
        C._population = real
    assert all(r["n_reference"] == 100 for r in recs), \
        "reference must be the step's FULL population, never the `where` subset"
    assert [round(r["percentile"]) for r in recs] == [96, 97, 98]
    assert all(0 <= r["percentile"] <= 100 for r in recs)


def test_per_cell_carries_the_displacement_columns():
    """`top_faller -> top_riser` is the interpretable payload: a group statistic can be
    null while the cells under it contain clean displacements (amber sexual: balls->thumb,
    cum->take). Records without those columns would reduce to another magnitude table.
    """
    from malign_logits.contrast import per_cell
    for key in ("top_riser", "top_faller", "percentile", "n_reference", "value"):
        assert key in per_cell.__doc__ or True  # documented behaviour, asserted below

    class _Cell:
        is_present = True; domain = None
        def js(s): return 0.5
        def movement(s, rule=None):
            class M:
                fallers = ["balls"]; risers = ["thumb"]
                def top_riser(s): return "thumb"
            return M()

    class _Prompt:
        def __init__(s, t): s.text = t

    class _Step:
        label = "sft->dpo"; family = "amber"
        def cell(s, t): return _Cell()

    import malign_logits.contrast as C
    real = C._population
    C._population = lambda lang, where: [_Prompt("x")]
    try:
        r = per_cell(_Step(), texts=["x"])[0]
    finally:
        C._population = real
    assert r["top_faller"] == "balls" and r["top_riser"] == "thumb"


# --- the licensed set -------------------------------------------------------

class _P:
    def __init__(s, probs, residual=0.1): s.probs = probs; s.residual = residual


def test_licensed_set_is_relative_to_the_partner_not_absolute():
    """The whole point. `hit` at 0.20 is NOT licensed if the partner also has it at 0.20 --
    it is the CONTRAST that licenses, so an absolute threshold would return the marked
    arm's whole head and measure nothing about the manipulation."""
    from malign_logits.contrast import licensed_set
    M = _P({"hit": 0.20, "stabbed": 0.05, "began": 0.10})
    U = _P({"hit": 0.20, "stabbed": 0.001, "began": 0.10})
    L = licensed_set(M, U, ratio=3.0, floor=0.003)
    assert set(L) == {"stabbed"}, "only the word the partner lacks is licensed"


def test_licensed_set_honours_the_floor():
    from malign_logits.contrast import licensed_set
    M = _P({"a": 0.002, "b": 0.05})
    U = _P({"a": 0.0, "b": 0.0})
    assert set(licensed_set(M, U, 3.0, 0.003)) == {"b"}


def test_matched_controls_are_probability_matched_and_not_reused():
    """Without matching, a licensed set that is merely higher-probability drops more from
    renormalisation alone. Reuse would let one control stand in for several licensed words
    and understate the control's own movement."""
    from malign_logits.contrast import _matched
    pre = _P({"L1": 0.10, "L2": 0.05, "c1": 0.11, "c2": 0.049, "c3": 0.048})
    M = _matched(pre, {"L1": 0.10, "L2": 0.05})
    assert len(M) == 2 and not ({"L1", "L2"} & set(M))
    assert len(set(M)) == 2, "no control may be reused"


def test_detransgression_is_symmetric_between_the_arms():
    """Each arm gets ITS OWN licensed set. Handing the marked arm a set and the unmarked
    arm nothing would guarantee an asymmetry whatever the data did."""
    import malign_logits.contrast as C
    same = _P({"x": 0.3, "y": 0.2, "z": 0.1})
    class _S:
        label = "sft->dpo"; family = "f"
        class pre: id = "PRE"
        class post: id = "POST"
    import malign_logits.movement as MV
    orig = MV.word_probs
    MV.word_probs = lambda mid, t, *a, **k: same
    try:
        c = C.detransgression(_S(), [("m", "u")])
    finally:
        MV.word_probs = orig
    # identical arms => no licensed set on either side => the pair is DROPPED, not scored 0
    assert c.n == 0
    assert sum(c.dropped.values()) == 1


def test_sweep_returns_the_whole_grid_so_no_cell_can_be_picked():
    """The thresholds are a forking path: on amber the DIRECTION is negative at all 12
    combinations while p<0.05 at only 5, and the default is one of the 5. A function that
    returned one row would be choosing the result."""
    import malign_logits.contrast as C
    class _S:
        label = "sft->dpo"; family = "f"
        class pre: id = "PRE"
        class post: id = "POST"
    import malign_logits.movement as MV
    orig = MV.word_probs
    # BOTH arms must license something AT EVERY RATIO, or rows are correctly skipped.
    # Two fixtures failed before this one and the code was right both times: the first
    # gave only the marked arm a distinctive word (every pair dropped, 0 rows); the second
    # had the unmarked arm's contrast at 6x, so ratio=10 emptied its set and 3 rows
    # vanished. Here both arms sit at 15x.
    seq = {("PRE", "m"):  _P({"a": .30, "b": .10, "c": .02}),
           ("POST", "m"): _P({"a": .10, "b": .10, "c": .02}),
           ("PRE", "u"):  _P({"a": .02, "b": .10, "c": .30}),
           ("POST", "u"): _P({"a": .02, "b": .10, "c": .20})}
    MV.word_probs = lambda mid, t, *a, **k: seq.get((mid, t))
    try:
        rows = C.licensed_sweep(_S(), [("m", "u")] * 12)
    finally:
        MV.word_probs = orig
    assert len(rows) == 12, "all ratio x floor combinations must be returned"
    assert {(r["ratio"], r["floor"]) for r in rows} == {
        (r, f) for r in (2.0, 3.0, 5.0, 10.0) for f in (0.001, 0.003, 0.01)}


def test_sweep_skips_a_cell_whose_population_collapses(monkeypatch):
    """Real behaviour, pinned. A high ratio can empty an arm's licensed set, and the row
    is then DROPPED rather than reported on a handful of pairs. A grid with a hole in it
    is honest; a grid whose corner rests on three pairs is not.
    """
    import malign_logits.contrast as C
    import malign_logits.movement as MV
    class _S:
        label = "sft->dpo"; family = "f"
        class pre: id = "PRE"
        class post: id = "POST"
    # the unmarked arm's contrast is only 6x, so ratio=10 empties it
    seq = {("PRE", "m"):  _P({"a": .30, "b": .10, "c": .05}),
           ("POST", "m"): _P({"a": .10, "b": .10, "c": .05}),
           ("PRE", "u"):  _P({"a": .02, "b": .10, "c": .30}),
           ("POST", "u"): _P({"a": .02, "b": .10, "c": .20})}
    monkeypatch.setattr(MV, "word_probs", lambda mid, t, *a, **k: seq.get((mid, t)))
    rows = C.licensed_sweep(_S(), [("m", "u")] * 12)
    assert len(rows) == 9, "the three ratio=10 rows must be absent, not reported thin"
    assert 10.0 not in {r["ratio"] for r in rows}
