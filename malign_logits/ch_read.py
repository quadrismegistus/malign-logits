"""ClickHouse read path for twp, with the bulk prefetch that makes it worth it.

    MALIGN_TWP_SOURCE=clickhouse  python your_script.py

WHY A PREFETCH AND NOT A LOOKUP. Measured on this machine, 2026-08-10:

    ClickHouse, one query per cell     192.2  ms/cell
    hashstash via Step/Cell             26.3  ms/cell
    ClickHouse, ONE bulk query           0.097 ms/cell   (2,647 cells in 256 ms)

Point-querying ClickHouse is 7x SLOWER than the stash, because every query pays
parse, plan, index seek and a client round-trip to return ~120 rows. Asking for
a whole model at once is 271x FASTER than the stash. The store is not the
variable -- the ACCESS SHAPE is, and `Step`/`Cell` is written cell-at-a-time, so
it can only ever ask in the shape ClickHouse is worst at. This module fixes that
by loading a model's cells on first touch and serving the rest from memory.

RAW ROWS ARE RETURNED, NOT FOLDED PROBABILITIES. `movement.word_probs` owns the
partition-summing and the malformed-row refusals, and it must own them for both
stores or the rule exists twice. So this returns the same `{"rows": [...],
"residual": {...}}` shape the stash yields and lets `word_probs` do its job.

SOURCE PRECEDENCE IS DECLARED, NOT INCIDENTAL. A cell scored in two payload
directories has two rows: identical theta, rule_version, dict_sha and
bos_policy, both conserving, DIFFERENT values (146 words vs 144; `believe` at
0.014100950 vs 0.014550155). Two observations of one configuration, so neither
supersedes the other and both stay in the table. This picks the most recent run
and says so, rather than leaving it to merge order -- which is how the stash and
ClickHouse came to hold opposite resolutions of the same cell.
"""
import os
import subprocess
from collections import defaultdict

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
#: latest run first
SOURCE_PRECEDENCE = ("f11_twp_delta", "f11_twp", "cloud_run_20260801")

_CACHE = {}          #: (model, theta, mode) -> {prompt: payload}
_MISS = set()        #: models known absent, so a miss is not re-queried per cell


def _unesc(x):
    """Reverse ClickHouse TSV escaping on EVERY string read back.

    Omitting this made a reconciler report 88 of 250 cells as disagreeing --
    `didn\\'t` against `didn't` -- when the table holds zero rows containing a
    backslash. Applied here so the same defect cannot reach the library.
    """
    return (x.replace("\\'", "'").replace("\\t", "\t")
             .replace("\\n", "\n").replace("\\\\", "\\"))


def _q(sql):
    r = subprocess.run([CH, "client", "--query", sql],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("clickhouse: %s" % r.stderr.strip()[:200])
    return r.stdout


def prefetch(model, theta=0.001, mode="raw"):
    """Load every cell for one model in ONE query. Idempotent."""
    key = (model, theta, mode)
    if key in _CACHE:
        return _CACHE[key]
    esc = model.replace("\\", "\\\\").replace("'", "\\'")
    #: argMin over the precedence index picks ONE source per (prompt, word)
    #: without a subquery per cell -- the whole point of the bulk shape.
    order = " ".join("WHEN source='%s' THEN %d" % (s, i)
                     for i, s in enumerate(SOURCE_PRECEDENCE))
    rows = _q(
        "SELECT prompt, word, argMin(p, CASE %s ELSE 99 END) AS p, "
        "       argMin(t1, CASE %s ELSE 99 END) AS t1 "
        #: **NEVER COMPARE A Float32 COLUMN TO A LITERAL WITH `=`.** theta is
        #: stored as Float32, so 0.001 round-trips as 0.0010000000474974513 and
        #: `theta = 0.001` matched ZERO of 300,010 rows while
        #: `abs(theta - 0.001) < 1e-9` matched every one. The failure is an
        #: empty result, not an error: the prefetch returned 0 prompts for a
        #: model plainly in the table, and a caller would read that as "not
        #: scored" rather than as a broken predicate.
        "FROM %s.twp_words WHERE model='%s' AND abs(theta - %r) < 1e-9 "
        "GROUP BY prompt, word FORMAT TSV" % (order, order, DB, esc, theta))
    by = defaultdict(list)
    for line in rows.splitlines():
        f = line.split("\t")
        if len(f) == 4:
            by[_unesc(f[0])].append({"word": _unesc(f[1]), "p": float(f[2]),
                                     "t1": int(f[3])})
    res = _q("SELECT prompt, argMin(tail,i), argMin(drop_,i), argMin(open_,i), "
             "argMin(mojibake,i), argMin(total,i), argMin(rule_version,i) FROM "
             "(SELECT prompt, tail, drop_, open_, mojibake, total, rule_version, "
             " CASE %s ELSE 99 END AS i FROM %s.twp_residual "
             " WHERE model='%s') GROUP BY prompt FORMAT TSV"
             % (order, DB, esc))
    resid = {}
    for line in res.splitlines():
        f = line.split("\t")
        if len(f) == 7:
            resid[_unesc(f[0])] = {"tail": float(f[1]), "drop": float(f[2]),
                                   "open": float(f[3]), "mojibake": float(f[4]),
                                   "total": float(f[5]), "rule_version": int(f[6])}
    out = {}
    for p, ws in by.items():
        r = resid.get(p) or {}
        out[p] = {"rows": ws, "residual": r,
                  "rule_version": r.get("rule_version", 3), "theta": theta}
    _CACHE[key] = out
    if not out:
        _MISS.add(key)
    return out


def ch_twp_payload(model, prompt, theta=0.001, mode="raw"):
    """One cell, in the shape `word_probs` expects. Prefetches the model once."""
    key = (model, theta, mode)
    if key in _MISS:
        return None
    return prefetch(model, theta, mode).get(prompt)


def clear():
    _CACHE.clear()
    _MISS.clear()
