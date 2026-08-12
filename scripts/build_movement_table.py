"""Materialise CANONICAL movement into ClickHouse, so every consumer filters
instead of recomputing.

    uv run python scripts/build_movement_table.py --create --run
    uv run python scripts/build_movement_table.py --run --rule lens
    uv run python scripts/build_movement_table.py --verify 200

WHY A TABLE AND NOT A QUERY
===========================

**The rule is NOT reimplemented in SQL, deliberately.** CANONICAL is
`min_prob=0.003, fall_ratio=0.5, delta=0.003, null_test=True`, and the null term
is a per-cell renormalisation aggregate. All of that is expressible with a
window function, and that is exactly the problem: there would then be two
implementations of one rule, and this campaign has already measured what that
costs -- two seats, one prose rule, 11% apart, neither miscoded.

So `malign_logits.movement.movement()` computes and this script only writes what
it returned. SQL filters the result and never decides it. If the rule changes,
one file changes and the table is rebuilt; nothing can drift because nothing
else knows the thresholds.

**THE FOLD IS `word_probs`, NEVER A DICT COMPREHENSION.** The payload is one row
per (word, first token) and those rows are a PARTITION. 20% of payloads carry a
duplicated surface, and on a Chinese payload the naive comprehension loses 2.7%
of the distribution. A movement table built over a broken fold would be wrong in
a way that falls hardest on exactly the cross-language work it exists to serve.

WHAT A ROW IS
=============

One (base, aligned, prompt, word) with both arms' probabilities and the class
CANONICAL gave it. `still` rows are written too -- a non-mover is data, and the
population for a word-level regression is defined by presence, not by movement.
Writing only movers would rebuild the selection defect the table exists to avoid.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta/M01_displacement/scripts"))

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
TABLE = "movement"
CELLS = "movement_cells"

#: TWO CELLS, NOT A PROMPT. `<<<LOGICAL:BOS>>>` is a real measurement -- the
#: runner dispatches on prompt_id and resolves it to `ids = [bos_id]` per model,
#: so it is that model's UNCONDITIONAL distribution, and base-vs-aligned on it
#: is a legitimate contrast (152 models carry it).
#:
#: **IT HAS BEEN WRONGLY RETIRED BEFORE.** A non-stimulus pass matched on SHAPE
#: -- an empty prompt -- and swept the sentinel out with the four literal
#: special tokens it replaces; `scripts/restore_logical_bos.py` put it back on
#: 2026-07-30, and its docstring records "three instances of a documented hazard
#: reintroduced by people who could state it". The first version of this file
#: made it a fourth, excluding the prompt across all 152 models to avoid two bad
#: rows.
#:
#: What is actually defective is TWO CELLS. `isNaN(p)` over the whole 68M-row
#: table returns exactly 2 rows, both at this prompt, one each on Qwen3-8B-Base
#: and Qwen3-8B. A twp payload's rows are a PARTITION summing to 1, so one NaN
#: makes that CELL's probabilities untrustworthy -- and says nothing about the
#: other 150 models at the same prompt.
#:
#: Declared as cells, by name, and asserted absent. `word_probs` refusing them
#: is correct and must not be wrapped in `try`: its docstring says a caller who
#: does that "converts it into a silent hole".
EXCLUDED_CELLS = frozenset({
    ("Qwen/Qwen3-8B-Base", "<<<LOGICAL:BOS>>>"),
    ("Qwen/Qwen3-8B", "<<<LOGICAL:BOS>>>"),
})

DDL = """
CREATE TABLE IF NOT EXISTS {db}.{tbl} (
    base       LowCardinality(String),
    aligned    LowCardinality(String),
    prompt     String,
    word       String,
    p_base     Float32,
    p_aligned  Float32,
    delta      Float32,
    cls        Enum8('still' = 0, 'rise' = 1, 'fall' = 2),
    relation   LowCardinality(String),
    family     LowCardinality(String),
    lineage    LowCardinality(String),
    rule       LowCardinality(String),
    theta      Float32,
    built      DateTime DEFAULT now()
) ENGINE = MergeTree
ORDER BY (rule, relation, base, aligned, prompt, word)
"""

#: THE CELL-LEVEL COMPANION. The word table cannot answer a JS question: it
#: stores per-word probabilities above theta and NOT the residual mass below it,
#: so a JS computed by summing its rows is silently truncated. Everything here
#: comes from ONE `decompose()` call per cell -- the same library function
#: `magnitude.py` uses -- so the cell quantities and the word classes can never
#: disagree about which cell they describe.
DDL_CELLS = """
CREATE TABLE IF NOT EXISTS {db}.{tbl} (
    base       LowCardinality(String),
    aligned    LowCardinality(String),
    prompt     String,
    relation   LowCardinality(String),
    family     LowCardinality(String),
    lineage    LowCardinality(String),
    resid_base     Float32,
    resid_aligned  Float32,
    js_total   Float32,
    js_fall    Float32,
    js_rise    Float32,
    js_tail    Float32,
    departed   Float32,
    arrived    Float32,
    concentration  Float32,
    n_fall     UInt16,
    n_rise     UInt16,
    n_still    UInt16,
    rule       LowCardinality(String),
    theta      Float32,
    built      DateTime DEFAULT now()
) ENGINE = MergeTree
ORDER BY (rule, relation, base, aligned, prompt)
"""


def q(sql, stdin=None):
    r = subprocess.run([CH, "client", "--query", sql], capture_output=True,
                       text=True, input=stdin)
    if r.returncode:
        raise RuntimeError("clickhouse: %s" % r.stderr.strip()[:400])
    return r.stdout


def esc(s):
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("'", "\\'"))


def roster():
    """EVERY declared checkpoint relation, from the registry. 203 edges.

    `data/model_registry.json` -> `relations` is the canonical list: it is the
    committed, provenanced artifact and `build_model_registry.py` owns it.

    **NOT `x_bodypart_classes.roster()`**, which returns only 58 base>aligned
    pairs and cannot express base->sft, sft->dpo, dpo->kto or an ablation arm.
    The direction convention is checked, not assumed: across every training
    relation the parent is the EARLIER checkpoint (sft_of is base->ego, dpo_of
    is base->superego 33 times and ego->superego 19), so pre=parent, post=child.

    **CH's `model_edges` had drifted from this list and was refreshed on
    2026-08-12 via `ch_ingest.py --registry`; the two now agree exactly at 203.**

    The drift is worth recording because reading it wrongly cost a pass here.
    Six edges existed in CH and not in the registry -- Falcon3-1B/3B/7B ->
    Falcon3-10B -- and they look exactly like edges the registry had lost. They
    are not. `build_model_registry.py:305` declares falcon3-10b in
    `NOT_A_SCALE_RUNG`: it was depth up-scaled from 7B with continual
    pretraining on 2 Teratokens, so it is its own pretraining rather than a rung
    of 7B's, while 1B and 3B are pruned and distilled at 80-100 GT and ARE
    rungs. RH's decision, 2026-08-10, with the token budget as the criterion.
    CH simply held the pre-decision version.

    The builder anticipated this exact misreading -- "considered and excluded
    must not look like never noticed" -- and the misreading happened anyway,
    from diffing two lists without opening the file that produces one of them.
    **A difference between a source and its mirror is a question about the
    mirror, not evidence about the source.**
    """
    R = json.load(open(os.path.join(ROOT, "data/model_registry.json")))
    fam = {m["model_id"]: (m.get("family") or "") for m in R["models"]}
    lm = json.load(open(os.path.join(ROOT, "data/lineage_map_models.json")))
    m2b = lm.get("model_to_base", {})
    out = []
    for r in R["relations"]:
        p, c = r["parent"], r["child"]
        out.append((p, c, r["relation"], fam.get(p, ""),
                    m2b.get(c) or m2b.get(p) or p))
    return out


def cells_for(model):
    """Prompts this model has, via ch_read's bulk prefetch (one query)."""
    from malign_logits.ch_read import prefetch
    try:
        return set(prefetch(model))
    except Exception:
        return set()


def build(rule_name, limit_pairs=None, only_prompts=None):
    from malign_logits.movement import (movement, word_probs, decompose,
                                        CANONICAL, LENS, DRAW)
    RULES = {"canonical": CANONICAL, "lens": LENS, "draw": DRAW}
    rule = RULES[rule_name]
    pairs = roster()[:limit_pairs] if limit_pairs else roster()
    #: SORTED BY PARENT so a parent is prefetched once for all its children, and
    #: the cache is dropped when the parent changes. `ch_read.prefetch` holds a
    #: whole model's cells in memory -- roughly 420k rows -- and 203 edges touch
    #: ~200 models, so an unbounded cache is tens of millions of rows of Python
    #: dict. This is the difference between a 40-minute build and an OOM.
    pairs = sorted(pairs, key=lambda e: (e[0], e[1]))
    import malign_logits.ch_read as CHR
    print("rule=%s  edges=%d" % (rule_name, len(pairs)))
    print("  relations: %s" % dict(collections.Counter(e[2] for e in pairs)))
    total = 0
    t0 = time.time()
    prev_parent = None
    for i, (b, a, relation, family, lineage) in enumerate(pairs, 1):
        if b != prev_parent:
            CHR.clear()
            prev_parent = b
        pb_set, pa_set = cells_for(b), cells_for(a)
        shared = pb_set & pa_set
        if only_prompts is not None:
            shared &= only_prompts
        shared -= {p for m, p in EXCLUDED_CELLS if m in (b, a)}
        assert not any((m, p) in EXCLUDED_CELLS
                       for m in (b, a) for p in shared), \
            "a declared-bad cell reached the admitted set"
        if not shared:
            print("  %3d/%d %-14s %-34s no shared cells"
                  % (i, len(pairs), relation, b.split("/")[-1][:34]))
            continue
        rows, crows = [], []
        for prompt in shared:
            wb = word_probs(b, prompt, theta=rule.theta)
            wa = word_probs(a, prompt, theta=rule.theta)
            if wb is None or wa is None:
                continue
            pre, post = wb.probs, wa.probs
            mv = movement(pre, post, rule,
                          residual_pre=wb.residual, residual_post=wa.residual)
            fall, rise = set(mv.fallers), set(mv.risers)
            try:
                dc = decompose(pre, post, rule, residual_pre=wb.residual,
                               residual_post=wa.residual)
            except Exception:
                dc = {}
            g = lambda k: float(dc.get(k) or 0.0)
            crows.append("%s\t%s\t%s\t%s\t%s\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t"
                         "%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\t%d\t%s\t%.6g"
                         % (esc(b), esc(a), esc(prompt), esc(relation), esc(family),
                            esc(lineage), wb.residual, wa.residual,
                            g("js_total"), g("js_fall"), g("js_rise"), g("js_tail"),
                            g("departed"), g("arrived"), g("concentration"),
                            len(fall), len(rise),
                            len(set(pre) | set(post)) - len(fall) - len(rise),
                            rule_name, rule.theta))
            for w in set(pre) | set(post):
                p0, p1 = pre.get(w, 0.0), post.get(w, 0.0)
                cls = "fall" if w in fall else "rise" if w in rise else "still"
                rows.append("%s\t%s\t%s\t%s\t%.9g\t%.9g\t%.9g\t%s\t%s\t%s\t%s\t%s\t%.6g"
                            % (esc(b), esc(a), esc(prompt), esc(w), p0, p1,
                               p1 - p0, cls, esc(relation), esc(family),
                               esc(lineage), rule_name, rule.theta))
        if rows:
            q("INSERT INTO %s.%s (base,aligned,prompt,word,p_base,p_aligned,"
              "delta,cls,relation,family,lineage,rule,theta) FORMAT TabSeparated"
              % (DB, TABLE), stdin="\n".join(rows) + "\n")
            total += len(rows)
        if crows:
            q("INSERT INTO %s.%s (base,aligned,prompt,relation,family,lineage,"
              "resid_base,resid_aligned,js_total,js_fall,js_rise,js_tail,"
              "departed,arrived,concentration,n_fall,n_rise,n_still,rule,theta) "
              "FORMAT TabSeparated" % (DB, CELLS), stdin="\n".join(crows) + "\n")
        #: drop the CHILD only; the parent stays for its remaining children.
        CHR._CACHE.pop((a, rule.theta, "raw"), None)
        print("  %3d/%d %-14s %-34s %5d cells  %9d rows  (%.0fs)"
              % (i, len(pairs), relation, b.split("/")[-1][:34], len(shared),
                 total, time.time() - t0), flush=True)
    print("\nwrote %d rows in %.0fs" % (total, time.time() - t0))
    return total


def verify(n, rule_name):
    """Recompute a random sample from the library and diff against the table.

    A materialised rule is only trustworthy if something proves it still agrees
    with its source. This is `ch_reconcile.py`'s job for twp, done here for
    movement -- and it must be run after every rebuild, not once.
    """
    from malign_logits.movement import movement, word_probs, CANONICAL, LENS, DRAW
    rule = {"canonical": CANONICAL, "lens": LENS, "draw": DRAW}[rule_name]
    #: JSONEachRow, NEVER TabSeparated. `ch_read._unesc` exists because a
    #: reconciler that read TSV without it reported 88 of 250 cells as
    #: disagreeing -- `didn\'t` against `didn't` -- on a table holding zero
    #: backslashes. The first version of this verifier reproduced that exact
    #: failure and reported apostrophe words as movement disagreements.
    out = q("SELECT DISTINCT base, aligned, prompt FROM %s.%s WHERE rule='%s' "
            "ORDER BY rand() LIMIT %d FORMAT JSONEachRow" % (DB, TABLE, rule_name, n))
    cells = [(r["base"], r["aligned"], r["prompt"])
             for r in (json.loads(l) for l in out.strip().split("\n") if l.strip())]
    agree = disagree = 0
    bad = []
    for b, a, p in cells:
        wb, wa = word_probs(b, p, theta=rule.theta), word_probs(a, p, theta=rule.theta)
        if wb is None or wa is None:
            continue
        mv = movement(wb.probs, wa.probs, rule,
                      residual_pre=wb.residual, residual_post=wa.residual)
        live = {}
        for w in set(wb.probs) | set(wa.probs):
            live[w] = ("fall" if w in set(mv.fallers)
                       else "rise" if w in set(mv.risers) else "still")
        got = q("SELECT word, cls FROM %s.%s WHERE rule='%s' AND base='%s' AND "
                "aligned='%s' AND prompt='%s' FORMAT JSONEachRow"
                % (DB, TABLE, rule_name, esc(b), esc(a), esc(p)))
        tbl = {r["word"]: r["cls"]
               for r in (json.loads(l) for l in got.strip().split("\n") if l.strip())}
        if tbl == live:
            agree += 1
        else:
            disagree += 1
            diff = {w for w in set(tbl) | set(live) if tbl.get(w) != live.get(w)}
            bad.append((b, a, p[:40], sorted(diff)[:6], len(diff)))
    print("VERIFY rule=%s  cells sampled %d   agree %d   DISAGREE %d"
          % (rule_name, agree + disagree, agree, disagree))
    for r in bad[:5]:
        print("   %s -> %s  %r  %d words differ: %s" % (r[0], r[1], r[2], r[4], r[3]))
    return disagree == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--rule", default="canonical", choices=("canonical", "lens", "draw"))
    ap.add_argument("--pairs", type=int, default=None, help="limit edges, for a smoke run")
    ap.add_argument("--zh-only", action="store_true", help="only Chinese prompts")
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--drop", action="store_true")
    a = ap.parse_args()
    if a.drop:
        q("DROP TABLE IF EXISTS %s.%s" % (DB, TABLE))
        q("DROP TABLE IF EXISTS %s.%s" % (DB, CELLS))
        print("dropped both tables")
    if a.create:
        q(DDL.format(db=DB, tbl=TABLE))
        q(DDL_CELLS.format(db=DB, tbl=CELLS))
        print("created %s.%s and %s.%s" % (DB, TABLE, DB, CELLS))
    only = None
    if a.zh_only:
        out = q("SELECT DISTINCT prompt FROM %s.twp_words WHERE match(prompt,'[一-龥]')" % DB)
        only = {l.replace("\\'", "'") for l in out.strip().split("\n") if l}
        print("restricting to %d Chinese prompts" % len(only))
    if a.run:
        q("ALTER TABLE %s.%s DELETE WHERE rule='%s'" % (DB, TABLE, a.rule))
        q("ALTER TABLE %s.%s DELETE WHERE rule='%s'" % (DB, CELLS, a.rule))
        build(a.rule, a.pairs, only)
    if a.verify:
        ok = verify(a.verify, a.rule)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
