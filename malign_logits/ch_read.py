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


# ---------------------------------------------------------------------------
# LOGITS. A SEPARATE NAME, DELIBERATELY -- NOT A FLAG ON cm.get_logits.
#
# `cm.get_logits` returns the FULL 131,072-dim vector from the .f16 memmap.
# ClickHouse holds log-probabilities TRUNCATED AT p >= 1e-6, a median 3,237
# entries. Those are different mathematical objects, and a flag that swapped
# them under one name would hand callers a different support while every value
# stayed finite and plausibly ranged -- the exact failure `_logit_array`
# documents, one level up. So they get different names and a caller chooses.
#
# WHEN TO USE WHICH
#   cm.get_logits      softmax, entropy, any full-vocabulary operation, or
#                      anything that must sum to 1. The payload is the store.
#   ch_logit_probs     "what did this model put on token T", especially ACROSS
#                      models and prompts. 271x faster in bulk, and the thing
#                      the .f16 memmap is worst at.
#
# THE TRUNCATION TRAVELS WITH EVERY RESULT. `kept` and `mass_kept` come back on
# each call, and `missing_mass` is 1 - mass_kept. A token absent from the result
# is BELOW THRESHOLD, not absent from the model -- that distinction is what the
# twp floor cost us on `___` this morning, and it is not repeated here.

def logit_coverage(model=None):
    """Which cells have logits, with how much mass each kept."""
    w = " WHERE model='%s'" % model.replace("'", "\\'") if model else ""
    out = _q("SELECT model, count() AS cells, round(avg(kept)) AS mean_tokens_kept, "
             "round(avg(mass_kept), 6) AS mean_mass_kept, any(dim) AS vocab "
             "FROM %s.logit_residual%s GROUP BY model ORDER BY model FORMAT TSVWithNames"
             % (DB, w))
    lines = out.splitlines()
    if not lines:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.count("\t") == len(head) - 1]


def ch_logit_probs(model, prompt, as_prob=True):
    """One cell's truncated distribution.

    -> {"probs": {token_id: value}, "kept": n, "mass_kept": m,
        "missing_mass": 1-m, "dim": vocab_size}

    `as_prob=True` exponentiates back to probabilities; False keeps logprobs,
    which is how they are stored and the better space for arithmetic.
    """
    import math
    e = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    rows = _q("SELECT token_id, logprob FROM %s.logit_probs "
              "WHERE model='%s' AND prompt='%s' FORMAT TSV"
              % (DB, e(model), e(prompt)))
    probs = {}
    for l in rows.splitlines():
        if "\t" in l:
            t, v = l.split("\t", 1)
            lp = float(v)
            probs[int(t)] = math.exp(lp) if as_prob else lp
    if not probs:
        return None
    meta = _q("SELECT kept, mass_kept, dim FROM %s.logit_residual "
              "WHERE model='%s' AND prompt='%s' LIMIT 1 FORMAT TSV"
              % (DB, e(model), e(prompt))).strip().split("\t")
    kept, mass, dim = (int(meta[0]), float(meta[1]), int(meta[2])) if len(meta) == 3 \
        else (len(probs), float("nan"), 0)
    return {"probs": probs, "kept": kept, "mass_kept": mass,
            "missing_mass": 1.0 - mass, "dim": dim}


def token_probs(token_ids, models=None, prompts=None, as_prob=True):
    """P(token) across MANY models and prompts in one query.

    This is the query the .f16 store cannot answer without reading 56 GB, and
    the reason the ClickHouse layer exists. Returns one row per
    (model, prompt, token_id) THAT CLEARS THRESHOLD.

    **AN ABSENT ROW MEANS BELOW 1e-6, NOT ZERO.** Join against
    `logit_residual` if you need to distinguish "the model gave it no mass"
    from "the cell was never scored" -- they are different facts and this
    function deliberately does not merge them.
    """
    e = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    w = ["token_id IN (%s)" % ",".join(str(int(t)) for t in token_ids)]
    if models:
        w.append("model IN (%s)" % ",".join("'%s'" % e(m) for m in models))
    if prompts:
        w.append("prompt IN (%s)" % ",".join("'%s'" % e(p) for p in prompts))
    val = "exp(logprob)" if as_prob else "logprob"
    out = _q("SELECT model, prompt, token_id, round(%s, 10) AS value "
             "FROM %s.logit_probs WHERE %s ORDER BY model, prompt, token_id "
             "FORMAT TSVWithNames" % (val, DB, " AND ".join(w)))
    lines = out.splitlines()
    if not lines:
        return []
    head = lines[0].split("\t")
    rows = []
    for l in lines[1:]:
        f = l.split("\t")
        if len(f) == len(head):
            d = dict(zip(head, f))
            d["prompt"] = _unesc(d["prompt"])
            d["token_id"] = int(d["token_id"])
            d["value"] = float(d["value"])
            rows.append(d)
    return rows
