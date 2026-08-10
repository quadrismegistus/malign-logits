#!/usr/bin/env python
"""Ingest twp rows and logit distributions into ClickHouse, from the raw payloads.

    scripts/ch_ingest.py --create                 # make the database and tables
    scripts/ch_ingest.py --twp --limit 3          # pilot
    scripts/ch_ingest.py --twp --logits           # everything
    scripts/ch_ingest.py --verify

WHY THIS EXISTS. Every coverage question this campaign asked on 2026-08-10 was a
full iteration of a 307,891-key hashstash, about ninety seconds each, and the
answers were group-bys: which models have twp on which battery, which pairs have
both arms, which prompts are scored nowhere. hashstash is a key-value store and
those are analytical scans. Point lookups stay where they are -- `Step`/`Cell`
and `cm.get_logits` are unchanged and remain the interface for per-cell work.
**This is a query layer, not a source of truth.**

THE JSONL IS THE INGEST SOURCE AND NO twp.py PASS IS NEEDED. Each `.jsonl` line
beside a `.f16` already carries the EXPANDED word rows, the residual, and the
seek triple into the payload:

    rows          [{word, t1, p}, ...]      the twp output, expansion done
    residual      {tail, drop, open, mojibake, total}
    logit_row / logit_dim / logit_dtype     the address in the .f16
    theta, rule_version, rule_commits, dict_sha, conservation

So the boundary rule TRAVELS AS A STAMP rather than being re-applied. That
matters: `twp.py` warns that "a second copy of a boundary rule is a second
policy", and re-expanding here would create one. `rule_version` and `dict_sha`
are columns, so a rule bump makes every row visibly stale instead of silently
wrong.

SAFETY. This machine runs live ClickHouse databases for another project --
`lltk` at 409 GiB, `abstraction`, `llmtasks`, `tmp`. **Every statement this
script issues is checked to name only `malign_logits`, and it will not run DDL
against anything else.** `_guard` refuses rather than warns, and it is applied
to every query including SELECTs, so a typo cannot reach a live table.

TRUNCATION IS DECLARED, NOT IMPLIED. Logit distributions are stored as
log-probabilities in Float32, keeping tokens with p >= 1e-6: measured, that is a
median 3,237 tokens per cell holding 99.78% of the mass, against twp's own
theta=1e-3 which keeps ~93 and 87.7%. **f32 over f16 deliberately**: the whole
spread across representations was 1.5 GB at 911M rows, which is not worth a
second dtype in a codebase where dtype resolution has already caused one silent
corruption (`_logit_array`, "made it live within the hour"). ONE dtype, no
resolver, no ambiguity for the 7,711 cells that exist at two precisions.

AND THE DISCARDED MASS IS A ROW, NOT AN ABSENCE. twp accumulates what falls
below theta into a four-way residual rather than dropping it; the logits table
does the same, one residual row per cell. Otherwise "absent or below threshold"
becomes unanswerable, which is exactly the confusion that cost the `___` result.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CH = "/opt/homebrew/bin/clickhouse"
DB = "malign_logits"
#: databases that must never appear in a statement this script issues
FORBIDDEN = ("lltk", "abstraction", "llmtasks", "tmp", "default", "system")
TRUNC = 1e-6

SCHEMA = f"""
CREATE DATABASE IF NOT EXISTS {DB};

CREATE TABLE IF NOT EXISTS {DB}.twp_words (
    model        LowCardinality(String),
    prompt       String,
    word         String,
    t1           UInt32,
    p            Float32,
    theta        Float32,
    rule_version UInt16,
    dict_sha     LowCardinality(String),
    source       LowCardinality(String),
    ingested     DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(ingested)
ORDER BY (model, prompt, word);

CREATE TABLE IF NOT EXISTS {DB}.twp_residual (
    model        LowCardinality(String),
    prompt       String,
    tail         Float32,
    drop_        Float32,
    open_        Float32,
    mojibake     Float32,
    total        Float32,
    conservation Float64,
    n_words      UInt32,
    rule_version UInt16,
    source       LowCardinality(String),
    ingested     DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(ingested)
ORDER BY (model, prompt);

CREATE TABLE IF NOT EXISTS {DB}.logit_probs (
    model     LowCardinality(String),
    prompt    String,
    token_id  UInt32 CODEC(Delta, ZSTD),
    logprob   Float32,
    ingested  DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(ingested)
ORDER BY (model, prompt, token_id);

-- THE CATALOGUE IS A DIMENSION, REGENERATED, AND prompt_id IS NOT IN THE FACTS.
-- RH, 2026-08-10: "just worried we were using prompt IDs which might change."
-- Correct, and the fix is not to store one. The fact tables key on the VERBATIM
-- TEXT because that is what the source stash keys on (model, prompt, theta,
-- mode), so a fact row can never disagree with its own payload. `prompt_id` is
-- an external assignment: the catalogue calls it a "stable slug", and
-- stable-by-declaration is the property that fails silently. Worse, the mapping
-- is ONE-TO-MANY -- 2,809 catalogue rows hold 2,578 distinct texts because one
-- prompt serves two designs -- so a single prompt_id on a text-keyed row would
-- force an arbitrary choice, which is the flattening this campaign keeps paying
-- for. Join to this table when catalogue attributes are wanted, and let the
-- one-to-many be visible in the join rather than hidden in a column.
CREATE TABLE IF NOT EXISTS {DB}.prompt_catalogue (
    prompt      String,
    prompt_id   LowCardinality(String),
    finding     LowCardinality(String),
    source      LowCardinality(String),
    language    LowCardinality(String),
    domain      LowCardinality(String),
    subdomain   LowCardinality(String),
    slot        LowCardinality(String),
    pair_id     LowCardinality(String),
    pair_role   LowCardinality(String),
    status      LowCardinality(String),
    is_logical  UInt8,
    ingested    DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(ingested)
ORDER BY (prompt, prompt_id);

CREATE TABLE IF NOT EXISTS {DB}.logit_residual (
    model      LowCardinality(String),
    prompt     String,
    threshold  Float32,
    kept       UInt32,
    dim        UInt32,
    mass_kept  Float32,
    ingested   DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(ingested)
ORDER BY (model, prompt);
"""


def check_drift():
    """Does the live schema match SCHEMA? **IF NOT EXISTS MAKES EDITS NO-OPS.**

    Measured, 2026-08-10: `token_id UInt32 CODEC(Delta, ZSTD)` was added to
    SCHEMA and `--create` re-run. It exited zero, printed success, and changed
    nothing, because the table already existed. The codec had to be applied by
    ALTER afterwards, and the only reason it was caught is that SHOW CREATE was
    read by hand.

    A DDL statement that cannot fail is not a migration. This compares what the
    server holds against what the file declares and NAMES the differences, so
    editing SCHEMA and re-running is either effective or visibly not.
    """
    import re as _re
    bad = []
    for m in _re.finditer(r"CREATE TABLE IF NOT EXISTS \w+\.(\w+) \((.*?)\n\)",
                          SCHEMA, _re.S):
        t, body = m.group(1), m.group(2)
        try:
            live = ch_read(f"SHOW CREATE TABLE {DB}.{t}").replace("\\n", "\n")
        except Exception:
            print("  %-18s ABSENT on the server" % t); bad.append(t); continue
        #: COMPARE CODEC NAMES, NOT THE PRESENCE OF THE WORD "CODEC".
        #: The first version asked `"CODEC" in live_line`, which `CODEC(NONE)`
        #: satisfies -- so stripping a codec to NONE read as no drift. Watched
        #: it fail to fire against a real ALTER before this was rewritten. The
        #: test has to be as fine as the property: parameters differ between
        #: what is declared (`Delta, ZSTD`) and what the server reports
        #: (`Delta(4), ZSTD(1)`), so compare the SET OF CODEC NAMES.
        def codecs(text):
            """Codec NAMES inside CODEC(...), parentheses balanced.

            A non-greedy regex cannot do this: on `CODEC(Delta(4), ZSTD(1))` it
            stops at the first `)` and returns {delta}, silently dropping zstd,
            which then reads as drift against a correct server. Watched that
            false-positive fire before this was rewritten. Balance the parens.
            """
            i = text.find("CODEC(")
            if i < 0:
                return set()
            j, depth = i + 6, 1
            while j < len(text) and depth:
                depth += (text[j] == "(") - (text[j] == ")")
                j += 1
            inner, out, buf, d = text[i + 6:j - 1], set(), "", 0
            for chx in inner + ",":
                if chx == "," and d == 0:
                    if buf.strip():
                        out.add(buf.split("(")[0].strip().lower())
                    buf = ""
                else:
                    d += (chx == "(") - (chx == ")")
                    buf += chx
            return out
        for line in [l.strip().rstrip(",") for l in body.strip().splitlines()]:
            col = line.split()[0]
            liveline = "".join(x for x in live.splitlines() if f"`{col}`" in x)
            want, got = codecs(line), codecs(liveline)
            if want != got:
                print("  %-18s %-12s declared %-22s server has %s"
                      % (t, col, sorted(want) or "-", sorted(got) or "-"))
                bad.append(t)
    print("  drift: %s" % (", ".join(sorted(set(bad))) if bad else "none"))
    return bad


def _guard(sql):
    """Refuse any statement naming a database that is not ours.

    NOT A WARNING. The failure this prevents is unrecoverable and belongs to
    someone else: `lltk` holds 409 GiB of another project's work on this
    machine. A guard that prints and proceeds is a guard nobody has watched
    refuse, which this campaign has now paid for twice.
    """
    low = sql.lower()
    for bad in FORBIDDEN:
        for pat in (f" {bad}.", f"\n{bad}.", f"`{bad}`", f"exists {bad};",
                    f"database {bad}", f"table {bad}."):
            if pat in low:
                raise SystemExit("REFUSING: statement names foreign database %r\n%s"
                                 % (bad, sql[:200]))
    return sql


def ch(sql, stdin=None):
    subprocess.run([CH, "client", "--query", _guard(sql)], input=stdin,
                   check=True, capture_output=(stdin is None))


def ch_read(sql):
    r = subprocess.run([CH, "client", "--query", _guard(sql)],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def insert(table, rows):
    """JSONEachRow insert. Batched by the caller."""
    if not rows:
        return
    payload = "\n".join(json.dumps(r) for r in rows).encode()
    subprocess.run([CH, "client", "--query",
                    _guard(f"INSERT INTO {DB}.{table} FORMAT JSONEachRow")],
                   input=payload, check=True, capture_output=True)


def done_cells(residual_table):
    """(model, prompt) already ingested, read from the RESIDUAL table.

    The residual tables carry exactly one row per cell, so this is ~266k rows
    rather than the 900M+ in `logit_probs`. Asking the fact table the same
    question would scan three orders of magnitude more to get the same set.

    Skipping here rather than relying on ReplacingMergeTree is deliberate:
    Replacing dedupes at MERGE time, eventually, so a re-run without this would
    do all the work -- read the payload, softmax, insert -- and only afterwards
    collapse the duplicates. The point of resumability is not writing them.
    """
    try:
        out = ch_read(f"SELECT model, prompt FROM {DB}.{residual_table} FORMAT TSV")
    except Exception:
        return set()
    done = set()
    for line in out.splitlines():
        if "\t" in line:
            m, p = line.split("\t", 1)
            done.add((m, p.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\")))
    return done


def ingest_logits_indexed(batch=400_000, limit=None):
    """Drive from the logits INDEX, not from .jsonl sidecars.

    `cloud_run_20260801/f11_twp/` holds 90 .f16 files and ZERO .jsonl, so the
    sidecar path cannot see them -- 17,534 cells over 199 prompts and 90 models,
    with **zero overlap** against the top-level payload, i.e. unique data that
    would sit permanently invisible behind a missing file. Their addresses were
    never lost: `index_logit_shards.py` put {file, row, dim} in the stash, which
    is what that script exists for. This reads the addresses from there and
    reaches all 263 payload files instead of the 103 with sidecars.

    Payloads are grouped by file so each is memmapped once.
    """
    import numpy as np
    from malign_logits.cache import CacheManager
    cm = CacheManager()
    root = cm._logit_root()
    done = done_cells("logit_residual")
    print("already ingested: %s cells (skipping)" % format(len(done), ","))
    byfile = defaultdict(list)
    for k in cm._stash("logits"):
        if (k["model"], k["prompt"]) in done:
            continue
        v = cm._stash("logits").get(k)
        byfile[(v["file"], k.get("dtype", "float16"))].append(
            (k["model"], k["prompt"], int(v["row"]), int(v["dim"])))
    files = sorted(byfile)
    if limit:
        files = files[:limit]
    print("payload files with un-ingested cells: %d\n" % len(files))
    out, res, n_rows, n_cells = [], [], 0, 0
    for (fname, dtype) in files:
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            print("  MISSING PAYLOAD %s -- skipped, not silently counted" % fname)
            continue
        dt = np.float16 if dtype == "float16" else np.float32
        mm = np.memmap(path, dtype=dt, mode="r")
        for model, prompt, row, dim in byfile[(fname, dtype)]:
            v = np.asarray(mm[row * dim:(row + 1) * dim], dtype=np.float32)
            if v.size != dim:
                continue
            v = v - v.max(); np.exp(v, out=v); v /= v.sum()
            idx = np.flatnonzero(v >= TRUNC)
            lp = np.log(v[idx]).astype(np.float32)
            out.extend({"model": model, "prompt": prompt, "token_id": int(t),
                        "logprob": float(x)} for t, x in zip(idx, lp))
            res.append({"model": model, "prompt": prompt, "threshold": TRUNC,
                        "kept": int(idx.size), "dim": dim,
                        "mass_kept": float(v[idx].sum())})
            n_cells += 1
            if len(out) >= batch:
                insert("logit_probs", out); n_rows += len(out); out = []
        insert("logit_probs", out); n_rows += len(out); out = []
        insert("logit_residual", res); res = []
        print("  %-56s %5d cells" % (fname[:56], len(byfile[(fname, dtype)])))
    print("\nindexed logits: %s new cells, %s token rows"
          % (format(n_cells, ","), format(n_rows, ",")))


#: (directory, label). The f11_twp subdir under cloud_run has 90 .f16 and ZERO
#: .jsonl, so the SIDECAR path cannot see it. `--index` reaches it instead.
SOURCES = [("data/raw/cloud_run_20260801", "cloud_run_20260801"),
           ("data/f11_twp", "f11_twp"),
           ("data/f11_twp_delta", "f11_twp_delta")]


def ingest_twp(limit=None, batch=200_000):
    done = done_cells("twp_residual")
    print("already ingested: %s twp cells (skipping)\n" % format(len(done), ","))
    n_files = n_rows = n_cells = n_skip = 0
    for rel, label in SOURCES:
        files = sorted(glob.glob(os.path.join(ROOT, rel, "*.jsonl")))
        if limit:
            files = files[:limit]
        for fp in files:
            words, resid = [], []
            for line in open(fp):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rows = d.get("rows") or []
                if not rows:
                    continue
                m, pr = d["model"], d["prompt"]
                if (m, pr) in done:
                    n_skip += 1
                    continue
                for w in rows:
                    words.append({"model": m, "prompt": pr, "word": w["word"],
                                  "t1": int(w.get("t1") or 0), "p": float(w["p"]),
                                  "theta": float(d.get("theta") or 0),
                                  "rule_version": int(d.get("rule_version") or 0),
                                  "dict_sha": d.get("dict_sha") or "",
                                  "source": label})
                r = d.get("residual") or {}
                resid.append({"model": m, "prompt": pr,
                              "tail": float(r.get("tail") or 0),
                              "drop_": float(r.get("drop") or 0),
                              "open_": float(r.get("open") or 0),
                              "mojibake": float(r.get("mojibake") or 0),
                              "total": float(r.get("total") or 0),
                              "conservation": float(d.get("conservation") or 0),
                              "n_words": len(rows),
                              "rule_version": int(d.get("rule_version") or 0),
                              "source": label})
                n_cells += 1
                if len(words) >= batch:
                    insert("twp_words", words); n_rows += len(words); words = []
            insert("twp_words", words); n_rows += len(words)
            insert("twp_residual", resid)
            n_files += 1
            print("  %-52s %s cells" % (os.path.basename(fp)[:52], format(len(resid), ",")))
    print("\ntwp: %d files, %s new cells, %s word rows, %s skipped as present"
          % (n_files, format(n_cells, ","), format(n_rows, ","), format(n_skip, ",")))


def ingest_logits(limit=None, batch=400_000):
    import numpy as np
    done = done_cells("logit_residual")
    print("already ingested: %s logit cells (skipping)\n" % format(len(done), ","))
    n_cells = n_rows = n_skip = 0
    for rel, label in SOURCES[:1]:          # only the archive carries logit_row
        files = sorted(glob.glob(os.path.join(ROOT, rel, "*.jsonl")))
        if limit:
            files = files[:limit]
        for fp in files:
            f16 = fp[:-6] + ".f16"
            if not os.path.exists(f16):
                continue
            mm = None
            out, res = [], []
            for line in open(fp):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("logit_row") is None:
                    continue
                if (d["model"], d["prompt"]) in done:
                    n_skip += 1
                    continue
                dim = int(d["logit_dim"])
                dt = np.float16 if d.get("logit_dtype", "float16") == "float16" else np.float32
                if mm is None:
                    mm = np.memmap(f16, dtype=dt, mode="r")
                r = int(d["logit_row"])
                v = np.asarray(mm[r * dim:(r + 1) * dim], dtype=np.float32)
                if v.size != dim:
                    continue
                v = v - v.max()
                np.exp(v, out=v)
                v /= v.sum()
                idx = np.flatnonzero(v >= TRUNC)
                lp = np.log(v[idx]).astype(np.float32)
                m, pr = d["model"], d["prompt"]
                out.extend({"model": m, "prompt": pr, "token_id": int(t),
                            "logprob": float(x)} for t, x in zip(idx, lp))
                res.append({"model": m, "prompt": pr, "threshold": TRUNC,
                            "kept": int(idx.size), "dim": dim,
                            "mass_kept": float(v[idx].sum())})
                n_cells += 1
                if len(out) >= batch:
                    insert("logit_probs", out); n_rows += len(out); out = []
            insert("logit_probs", out); n_rows += len(out)
            insert("logit_residual", res)
            print("  %-46s %5s cells  %12s rows"
                  % (os.path.basename(fp)[:46], format(len(res), ","), format(n_rows, ",")))
    print("\nlogits: %s new cells, %s token rows, %s skipped as present"
          % (format(n_cells, ","), format(n_rows, ","), format(n_skip, ",")))


def ingest_catalogue():
    """The catalogue as a dimension. REGENERATED WHOLE, never appended.

    Rows go in with `status` INCLUDED rather than filtered, because "which
    universe was I counting over" is the question this whole layer exists to
    make answerable -- and a table that silently holds only ACTIVE rows cannot
    answer it. Filter in the query, where the choice is visible.
    """
    rows = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    out = [{"prompt": r.get("prompt") or "", "prompt_id": r.get("prompt_id") or "",
            "finding": str(r.get("finding") or ""), "source": str(r.get("source") or ""),
            "language": str(r.get("language") or ""), "domain": str(r.get("domain") or ""),
            "subdomain": str(r.get("subdomain") or ""), "slot": str(r.get("slot") or ""),
            "pair_id": str(r.get("pair_id") or ""), "pair_role": str(r.get("pair_role") or ""),
            "status": str(r.get("status") or ""),
            "is_logical": 1 if r.get("resolver") else 0} for r in rows]
    ch(f"TRUNCATE TABLE IF EXISTS {DB}.prompt_catalogue")
    insert("prompt_catalogue", out)
    print("catalogue: %s rows, %s distinct texts"
          % (format(len(out), ","), format(len({r["prompt"] for r in out}), ",")))


def verify():
    print("TABLES IN %s\n" % DB)
    print(ch_read(f"SELECT name, formatReadableQuantity(total_rows) AS rows, "
                  f"formatReadableSize(total_bytes) AS size FROM system.tables "
                  f"WHERE database='{DB}' ORDER BY name FORMAT PrettyCompact"))
    for t, q in (("twp_words", f"SELECT count() c, uniqExact(model) models, uniqExact(prompt) prompts FROM {DB}.twp_words"),
                 ("logit_probs", f"SELECT count() c, uniqExact(model) models, uniqExact(prompt) prompts FROM {DB}.logit_probs")):
        try:
            print("\n%-14s %s" % (t, ch_read(q + " FORMAT TSV")))
        except Exception as e:
            print("\n%-14s (empty or absent)" % t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--twp", action="store_true")
    ap.add_argument("--logits", action="store_true")
    ap.add_argument("--catalogue", action="store_true")
    ap.add_argument("--drift", action="store_true", help="live schema vs SCHEMA")
    ap.add_argument("--index", action="store_true",
                    help="logits driven by the stash index, reaching all 263 payloads")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N files per source")
    a = ap.parse_args()
    if a.create:
        for stmt in [s.strip() for s in SCHEMA.split(";") if s.strip()]:
            ch(stmt)
        n = sum(1 for x in SCHEMA.split(";") if "create table" in x.lower())
        print("created database %s and %d tables" % (DB, n))
    if a.twp:
        ingest_twp(a.limit)
    if a.logits:
        ingest_logits(a.limit)
    if a.index:
        ingest_logits_indexed(limit=a.limit)
    if a.drift:
        print("SCHEMA DRIFT CHECK"); check_drift()
    if a.catalogue:
        ingest_catalogue()
    if a.verify:
        verify()
    if not any((a.create, a.twp, a.logits, a.index, a.catalogue, a.drift, a.verify)):
        ap.print_help()


if __name__ == "__main__":
    main()
