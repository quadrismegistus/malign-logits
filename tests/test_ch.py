"""Tests for malign_logits.ch.

Every one of these was WATCHED TO FAIL before it was kept, per the standing rule.
Two of them exist because the module shipped with the bug they now catch: the
FORMAT-substring test would have caught `formatReadableSize` suppressing the
format clause within the hour, and it did not exist to.

Tests that need the store are skipped when it is absent rather than failing, so
this file is runnable on a machine with no ClickHouse. The ones that matter most
-- the predicate tests -- need nothing.
"""
import pytest

from malign_logits import ch


def _store():
    try:
        ch.scalar("SELECT 1")
        return True
    except Exception:
        return False


needs_store = pytest.mark.skipif(not _store(), reason="no ClickHouse on this machine")


# --------------------------------------------------------------------------
# the predicate tests. no store needed, and these are the ones that bit.
# --------------------------------------------------------------------------

def test_a_format_clause_is_a_trailing_keyword_not_a_substring():
    """THE BUG THIS MODULE SHIPPED WITH, on its first real query.

    `"FORMAT" not in sql.upper()` matches `formatReadableSize(...)`, so an
    ordinary `system.tables` query never had `FORMAT JSONEachRow` appended, got
    TSV back, and failed the JSON parse on line 1. Watched to fail against the
    old predicate before this was kept.
    """
    sql = "SELECT formatReadableSize(total_bytes) AS size FROM system.tables"
    assert "FORMAT" in sql.upper()          # the OLD predicate's view: "has one"
    assert not ch._HAS_FORMAT.search(sql)   # the truth: it has no FORMAT clause
    assert ch._with_format(sql, "JSONEachRow").endswith("FORMAT JSONEachRow")


def test_an_explicit_format_clause_is_respected_and_not_doubled():
    for sql in ("SELECT 1 FORMAT TSV",
                "SELECT 1 FORMAT TSV;",
                "SELECT 1 format tsv",
                "SELECT 1 FORMAT JSONEachRow  "):
        assert ch._HAS_FORMAT.search(sql), sql
        # .upper(): the fixture list includes `format tsv`, and an assertion
        # about case-insensitivity must not itself be case-sensitive.
        assert ch._with_format(sql, "JSONEachRow").upper().count("FORMAT") == 1, sql


def test_a_trailing_semicolon_does_not_produce_a_syntax_error():
    assert ch._with_format("SELECT 1;", "JSONEachRow") == "SELECT 1 FORMAT JSONEachRow"


def test_approx_is_not_equality_because_float32_round_trips_wrong():
    """0.001 stored as Float32 comes back as 0.0010000000474974513.

    `theta = 0.001` matched ZERO of 173 million rows across three tables and the
    failure was an empty result, not an error.
    """
    import numpy as np
    assert float(np.float32(0.001)) != 0.001          # the whole reason
    pred = ch.approx("theta", 0.001)
    assert "=" not in pred.replace("<", "")           # no equality anywhere
    assert pred == "abs(theta - 0.001) < 1e-09"


# --------------------------------------------------------------------------
# transport. these need the store.
# --------------------------------------------------------------------------

@needs_store
def test_types_survive_rather_than_arriving_as_strings():
    r = ch.query("SELECT 1 AS i, 1.5 AS f, 'x' AS s")[0]
    assert r == {"i": 1, "f": 1.5, "s": "x"}
    assert isinstance(r["i"], int) and isinstance(r["f"], float)


@needs_store
def test_a_newline_in_a_value_survives_the_round_trip():
    """THE ARGUMENT FOR THE WHOLE MODULE, as a fixture.

    The same value through TSVRaw yields two lines for one row -- which is how a
    964,679-row table read as 1,621,740 lines and manufactured a divergence
    between two stores that did not exist.
    """
    v = "a\nb"
    assert ch.query("SELECT 'a\\nb' AS v")[0]["v"] == v
    assert len(ch.query("SELECT 'a\\nb' AS v")) == 1
    assert len(ch.raw("SELECT 'a\\nb' AS v FORMAT TSVRaw").splitlines()) == 2


@needs_store
def test_a_tab_in_a_value_survives_too():
    assert ch.query("SELECT 'a\\tb' AS v")[0]["v"] == "a\tb"


@needs_store
def test_scalar_refuses_more_than_one_row_rather_than_taking_the_first():
    """A caller asking for a scalar and silently getting the first of several is
    the shape of a wrong number that never looks wrong."""
    assert ch.scalar("SELECT 7 AS n") == 7
    assert ch.scalar("SELECT 1 WHERE 0", default="none") == "none"
    with pytest.raises(ch.ClickHouseError):
        ch.scalar("SELECT arrayJoin([1, 2]) AS n")
    with pytest.raises(ch.ClickHouseError):
        ch.scalar("SELECT 1 AS a, 2 AS b")


@needs_store
def test_an_error_carries_the_sql_that_failed():
    """RE-AIMED TWICE, AND THE SECOND TIME BY A MUTATION CHECK.

    v1 asserted the table name and the `--- SQL ---` header appear in the
    message. Both do even with the SQL dropped: stderr names the table, the
    header is unconditional. Dropping the SQL left all 13 tests green.

    v2 put a marker in a SQL comment on the theory that only the echoed
    statement could carry it. **ClickHouse echoes the whole query in its own
    error text, comment included**, so that survived the mutation too.

    So a SERVER error cannot test this at all -- stderr already contains what
    the sql field would add. The field earns its keep on errors THIS MODULE
    raises, where there is no server text and the statement is the only
    context a reader gets. Those are the cases tested here.
    """
    marker = "zz_marker_only_in_the_sql_zz"
    sql = "SELECT number AS %s FROM system.numbers LIMIT 100000" % marker
    with pytest.raises(ch.ClickHouseError) as e:
        ch.query(sql, limit_bytes=100)          # OUR error, not the server's
    assert marker in str(e.value)

    sql2 = "SELECT arrayJoin([1, 2]) AS %s" % marker
    with pytest.raises(ch.ClickHouseError) as e:
        ch.scalar(sql2)                          # ours again
    assert marker in str(e.value)


@needs_store
def test_db_is_substituted():
    assert ch.exists("twp_words")
    assert not ch.exists("definitely_not_a_table")


@needs_store
def test_insert_of_nothing_inserts_nothing_and_says_so():
    """An empty INSERT statement is accepted by ClickHouse and reads in a log as
    a successful insert of nothing."""
    assert ch.insert("twp_words", []) == 0


@needs_store
def test_the_result_size_guard_fires():
    with pytest.raises(ch.ClickHouseError) as e:
        ch.query("SELECT number FROM system.numbers LIMIT 100000", limit_bytes=100)
    assert "limit_bytes" in str(e.value)


@needs_store
def test_parquet_and_query_agree():
    sql = "SELECT number AS n FROM system.numbers LIMIT 5"
    assert list(ch.parquet(sql)["n"]) == [r["n"] for r in ch.query(sql)]
