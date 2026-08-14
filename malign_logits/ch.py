#!/usr/bin/env python
"""One way to ask ClickHouse a question.

    from malign_logits import ch

    ch.query("SELECT model, count() AS n FROM {db}.twp_words GROUP BY model")
    ch.scalar("SELECT count() FROM {db}.twp_words")
    ch.df("SELECT * FROM {db}.gen_scores LIMIT 100")
    ch.execute("CREATE TABLE ...")
    ch.insert("vf_manifest_tmp", rows)

WHY THIS EXISTS, IN EVIDENCE RATHER THAN IN PREFERENCE. On 2026-08-14 a survey
found 85 files touching ClickHouse and 69 of them shelling out themselves, plus
TWO private `_q` helpers inside this library that had diverged from each other.
The duplication is the least of it. Every hand-rolled reader repeats the same
three defects, and all three are defects this campaign has already paid for:

  1. **SILENT ROW DROPS.** `ch_read._q`'s consumers do `if len(f) == 4:` and
     `gens._q` does `if len(f) != len(head): continue`. A row that does not
     split into the expected shape is discarded with no count, no file and no
     cause. That is the receiptless disposition booked at [6127] -- and the
     accounting identity that would have caught it cannot, because a dropped
     row enters neither side of it.

  2. **TSV ESCAPING BY HAND.** `ch_read._unesc` exists because omitting it made
     a reconciler report 88 of 250 cells as disagreeing (`didn\\'t` against
     `didn't`) on a table holding zero backslashes. `gens._q` takes an
     `unescape_cols` argument, so the guard is OPT-IN and any string column
     nobody named keeps its escaping. And at [6065] a `TSVRaw` export -- which
     does not escape newlines at all, on prompts that contain them -- returned
     1,621,740 lines against a true 964,679, manufacturing a divergence between
     two stores that did not exist.

  3. **THE FORMAT DECIDES THE POPULATION.** Which is (2) stated generally: a
     serialisation format is a population definition, and a reader that splits
     on a delimiter is asserting that the delimiter does not occur in the data.

**JSONEachRow DISSOLVES ALL THREE.** There is no delimiter to collide with, no
escaping convention to reverse, and a row that will not parse raises instead of
vanishing. It is also already what 61 of the 74 external call sites use, so this
is the majority convention rather than a new one.

**WHAT THIS MODULE DOES NOT DO.** It does not open a connection, pool anything,
or add a dependency: it shells out to the same binary every caller already
shells out to. The value is not the transport, it is that the transport is in
one place with the three lessons above already applied.

**`{db}` IN A QUERY IS SUBSTITUTED** with the configured database, so callers
stop writing `%s.twp_words % DB` and stop getting it wrong when DB is not the
default.
"""
import json
import os
import subprocess

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")

#: A query that returns more than this many bytes is almost certainly a mistake
#: of the "SELECT * FROM a 350M-row table" kind. Raise rather than fill memory;
#: pass `limit_bytes=None` to mean it.
DEFAULT_LIMIT_BYTES = 2_000_000_000


class ClickHouseError(RuntimeError):
    """A failed query, carrying the SQL that failed.

    The SQL is included because the most common cause of a ClickHouse error in
    this repo is a malformed interpolation, and an error message without the
    statement sends the reader to the wrong line.
    """

    def __init__(self, stderr, sql):
        self.sql = sql
        super().__init__("%s\n--- SQL ---\n%s" % (stderr.strip()[:400],
                                                  sql.strip()[:800]))


def _run(sql, stdin=None, limit_bytes=DEFAULT_LIMIT_BYTES):
    sql = sql.replace("{db}", DB)
    r = subprocess.run([CH, "client", "--query", sql],
                       input=stdin, capture_output=True, text=True)
    if r.returncode:
        raise ClickHouseError(r.stderr, sql)
    if limit_bytes is not None and len(r.stdout) > limit_bytes:
        raise ClickHouseError(
            "result exceeded limit_bytes=%d (got %d). Narrow the query or "
            "pass limit_bytes=None deliberately." % (limit_bytes, len(r.stdout)),
            sql)
    return r.stdout


def raw(sql, **kw):
    """Stdout, unparsed. The escape hatch for a FORMAT this module does not wrap.

    Prefer `query`. If you reach for this with `FORMAT TSV` and then split on
    tabs, you have re-created defect (2) in the module docstring.
    """
    return _run(sql, **kw)


def query(sql, **kw):
    """Rows as dicts, via JSONEachRow. Types survive; escaping is not your problem.

    A line that will not parse RAISES, naming the line number and the line. It
    is not skipped, because a reader that skips is a disposition with no
    receipt and this module exists partly to stop writing those.
    """
    if "FORMAT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + " FORMAT JSONEachRow"
    out = _run(sql, **kw)
    rows = []
    for i, line in enumerate(out.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ClickHouseError(
                "unparseable JSONEachRow at line %d: %s\nline: %.200r"
                % (i, e, line), sql)
    return rows


def scalar(sql, default=None, **kw):
    """The single value of a one-row, one-column query, or `default` if no rows.

    Raises if the query returns more than one row or more than one column,
    because a caller asking for a scalar and silently getting the first of
    several is the shape of a wrong number that never looks wrong.
    """
    rows = query(sql, **kw)
    if not rows:
        return default
    if len(rows) > 1:
        raise ClickHouseError("scalar() got %d rows" % len(rows), sql)
    vals = list(rows[0].values())
    if len(vals) != 1:
        raise ClickHouseError("scalar() got %d columns: %s"
                              % (len(vals), list(rows[0])), sql)
    return vals[0]


def _run_bytes(sql, limit_bytes=DEFAULT_LIMIT_BYTES):
    sql = sql.replace("{db}", DB)
    r = subprocess.run([CH, "client", "--query", sql], capture_output=True)
    if r.returncode:
        raise ClickHouseError(r.stderr.decode("utf-8", "replace"), sql)
    if limit_bytes is not None and len(r.stdout) > limit_bytes:
        raise ClickHouseError("result exceeded limit_bytes=%d (got %d)"
                              % (limit_bytes, len(r.stdout)), sql)
    return r.stdout


def parquet(sql, **kw):
    """A DataFrame via `FORMAT Parquet`. Binary transport, for bulk reads.

    Prefer this over `df` when the result is large: Parquet carries types and
    compresses, where JSONEachRow spends a line of text per row. Added
    2026-08-14 because `verse_capacity` needed it and the first version of
    this module could not express it -- `_run` decodes as text, which
    corrupts a binary payload silently rather than failing.
    """
    import io as _io
    import pandas as pd
    if "FORMAT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + " FORMAT Parquet"
    return pd.read_parquet(_io.BytesIO(_run_bytes(sql, **kw)))


def df(sql, **kw):
    """A pandas DataFrame. Empty result gives an empty frame, not an exception."""
    import pandas as pd
    return pd.DataFrame(query(sql, **kw))


def execute(sql, **kw):
    """A statement expected to return nothing: DDL, INSERT ... SELECT, DROP.

    Returns stdout, which is normally empty; a non-empty result is handed back
    rather than discarded, so a caller who used `execute` for a SELECT sees it.
    """
    return _run(sql, **kw)


def insert(table, rows, **kw):
    """Insert dicts into `table` as JSONEachRow. Returns the count inserted.

    Empty input inserts nothing and returns 0 rather than sending a statement
    with no body, which ClickHouse accepts and which reads in a log as a
    successful insert of nothing.
    """
    rows = list(rows)
    if not rows:
        return 0
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    tbl = table if "." in table else "{db}." + table
    _run("INSERT INTO %s FORMAT JSONEachRow" % tbl, stdin=body, **kw)
    return len(rows)


def exists(table):
    """Whether a table exists, by name, with or without a database prefix."""
    tbl = table if "." in table else "%s.%s" % (DB, table)
    d, t = tbl.split(".", 1)
    return bool(scalar("SELECT count() FROM system.tables "
                       "WHERE database='%s' AND name='%s'" % (d, t)))


def columns(table):
    """{name: type} for a table, in position order."""
    tbl = table if "." in table else "%s.%s" % (DB, table)
    d, t = tbl.split(".", 1)
    return {r["name"]: r["type"] for r in query(
        "SELECT name, type FROM system.columns WHERE database='%s' "
        "AND table='%s' ORDER BY position" % (d, t))}
