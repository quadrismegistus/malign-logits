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

from . import ch

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
#: latest run first
#: **THE M05 SHARDS RANK LAST, AND THAT IS A CONSISTENCY RULE RATHER THAN A
#: JUDGEMENT ABOUT QUALITY.** 181 cells on the three `main` checkpoints were
#: re-scored by the M05 fleet because they needed hidden sidecars, and their word
#: probabilities differ from the resident ones -- same word COUNTS, different
#: values, two observations on different hardware. `twp_ingest` is first-write-
#: wins, so the store kept `cloud_run_20260801`. The index must therefore choose
#: the same file, on `index_logit_shards`'s own doctrine: **"if the index and twp
#: ever chose differently, one cell's word probabilities and its logit vector
#: would come from different runs, which is worse than either being wrong on its
#: own."**
#:
#: Ranking last touches ONLY cells that collide with a higher-precedence source
#: -- 122 of 54,492. M05's own new cells exist nowhere else, and a rank is never
#: consulted for a cell with one candidate.
SOURCE_PRECEDENCE = ("f11_twp_delta", "f11_twp", "cloud_run_20260801",
                     "m05_0", "m05_1", "m05_2", "m05_3", "m05_4")

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


_MODEL_SOURCE = None


def _model_source():
    """model -> the source to prefer, derived from the per-cell resolution map.

    The map is per CELL because the right choice is a property of the pair, not
    of the source. It collapses to a per-model preference here only because it
    happens to be UNIFORM within each of the three models that need it -- and
    that is asserted rather than assumed: a model whose cells resolve to more
    than one source outside the declared precedence is skipped and reported,
    never averaged into one answer.

    An absent map is not an error; it restores the previous behaviour, which is
    the arbitrary tie-break. `scripts/resolve_twp_sources.py --write` rebuilds it.
    """
    global _MODEL_SOURCE
    if _MODEL_SOURCE is None:
        import json as _json
        import os as _os
        from collections import defaultdict as _dd
        p = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "twp_source_resolution.json")
        per = _dd(set)
        if _os.path.exists(p):
            with open(p) as fh:
                for k, v in _json.load(fh)["cells"].items():
                    s = v.get("source")
                    #: only where the map DISAGREES with the global list; the
                    #: rest is already handled and does not need an override.
                    if s and s not in SOURCE_PRECEDENCE:
                        per[k.split("\x1f", 1)[0]].add(s)
        _MODEL_SOURCE = {m: next(iter(s)) for m, s in per.items() if len(s) == 1}
        _AMBIGUOUS.update({m: sorted(s) for m, s in per.items() if len(s) > 1})
    return _MODEL_SOURCE


#: models whose cells resolve to several non-precedence sources. Left to the
#: global rule and NAMED, because collapsing them to one would be the arbitrary
#: choice this whole mechanism exists to remove.
_AMBIGUOUS = {}


def prefetch(model, theta=0.001, mode="raw"):
    """Load every cell for one model in ONE query. Idempotent."""
    key = (model, theta, mode)
    if key in _CACHE:
        return _CACHE[key]
    esc = model.replace("\\", "\\\\").replace("'", "\\'")
    #: argMin over the precedence index picks ONE source per (prompt, word)
    #: without a subquery per cell -- the whole point of the bulk shape.
    #:
    #: **EVERY UNLISTED SOURCE RANKED 99 AND argMin BROKE THE TIE ARBITRARILY**,
    #: which is not a rule. It is how ClickHouse and the stash came to serve
    #: different observations of one cell -- llm-jp-3-7.2b on '死了' at
    #: 0.01634436 here against 0.01620482 there, each exactly one source's
    #: value ([5311]). SOURCE_PRECEDENCE names three historical directories and
    #: no census or wave box, so for those the tie-break WAS the policy.
    #:
    #: The per-model preference comes from `data/twp_source_resolution.json`,
    #: decided per cell against evidence -- chiefly: prefer the source that also
    #: holds the OTHER ARM of the pair, so that a device difference cannot land
    #: inside the contrast (malign, [5312]). Prepended, so it outranks the
    #: global list. Three models need it and they account for 5,391 cells.
    prefer = _model_source().get(model)
    ranked = ([prefer] if prefer else []) + [s for s in SOURCE_PRECEDENCE
                                             if s != prefer]
    order = " ".join("WHEN source='%s' THEN %d" % (s, i)
                     for i, s in enumerate(ranked))
    rows = ch.query(
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
        #: ORDER BY IS NOT COSMETIC. Without it ClickHouse returns grouped rows
        #: in arbitrary order, and two consecutive prefetches of the same model
        #: differed on 4,051 of 4,413 cells -- word order inside every cell,
        #: never the values. Harmless to `word_probs`, which folds rows into a
        #: dict keyed by surface, and fatal to the cheapest check there is:
        #: an artifact built from this read cannot be compared by hash to one
        #: built a minute earlier. Same lesson as m05_field_flow the same day,
        #: and the campaign's own: a fixed seed does not make a run
        #: reproducible when the data order is not fixed.
        "GROUP BY prompt, word ORDER BY prompt, word" % (order, order, DB, esc, theta))
    #: WAS a TSV split with `if len(f) == 4:` -- which discarded any row that
    #: did not split into four fields, with no count and no cause. A prompt
    #: containing a tab or a newline splits into more, and the reader lost it
    #: silently ([6127]'s receiptless disposition, in the campaign's most-used
    #: read path). `ch.query` uses JSONEachRow: no delimiter to collide with,
    #: no escaping to reverse, and an unparseable row raises.
    by = defaultdict(list)
    for r in rows:
        by[r["prompt"]].append({"word": r["word"], "p": float(r["p"]),
                                "t1": int(r["t1"])})
    res = ch.query("SELECT prompt, argMin(tail,i), argMin(drop_,i), argMin(open_,i), "
             "argMin(mojibake,i), argMin(total,i), argMin(rule_version,i) FROM "
             "(SELECT prompt, tail, drop_, open_, mojibake, total, rule_version, "
             " CASE %s ELSE 99 END AS i FROM %s.twp_residual "
             " WHERE model='%s') GROUP BY prompt ORDER BY prompt"
             % (order, DB, esc))
    resid = {}
    for r in res:
        v = list(r.values())
        resid[r["prompt"]] = {"tail": float(v[1]), "drop": float(v[2]),
                              "open": float(v[3]), "mojibake": float(v[4]),
                              "total": float(v[5]), "rule_version": int(v[6])}
    #: ITERATE THE UNION, NOT THE WORD ROWS. `by` is keyed by prompts that have
    #: at least one word above theta, so a cell that was SCORED and resolved
    #: NOTHING never entered `out` and reached `word_probs` as None -- i.e. as
    #: ABSENT. Present-and-empty and never-scored are different facts and this
    #: loop made them one.
    #:
    #: It is not a corner case and it is not only step0. 1,226 cells across 23
    #: models are residual-only: Lucie 368 + 207, Olmo-3-1025-7B@stage1-step0
    #: 257, salamandra 162 + 104, Pharia 52 + 50, jais 5. Those models have
    #: been quietly dropping cells from every analysis that iterated them.
    #:
    #: The step0 case is the one that names the cost: at that checkpoint the
    #: distribution is flat enough that no token clears theta=0.001 (uniform
    #: over 100,278 tokens is p~1e-5), so 257 of 584 cells resolve no word.
    #: Read as absent, an acquisition curve begins at "no data" where the truth
    #: is "no resolvable word" -- different claims about the same rung.
    #:
    #: Downstream is safe and was checked rather than assumed: `movement` on
    #: empty->empty gives 0 risers, 0 fallers, residual_share 1.0; empty->words
    #: makes the word a riser, which is what a rise from below theta is; and
    #: `Cell.js` is exactly 0.0 between two all-tail cells.
    out = {}
    for p in set(by) | set(resid):
        r = resid.get(p) or {}
        out[p] = {"rows": by.get(p, []), "residual": r,
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


#: **THE `logit_probs` TABLE IS SUPERSEDED (RH, 2026-08-14).** Logits live in
#: the `.f16` sidecar files; the table was last written 2026-08-10 and nothing
#: has been added since. The three functions below still work and still read
#: it, and what they return is whatever was there on 08-10 -- for 123 models,
#: not for anything scored after. Do not build a new analysis on them without
#: knowing that. `cache.get_logits` is the OTHER store and is the live one.
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
