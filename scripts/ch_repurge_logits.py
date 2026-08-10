#!/usr/bin/env python
"""Purge and re-ingest EXACTLY the cells named in a key list. Nothing else.

    scripts/ch_repurge_logits.py --dry-run
    scripts/ch_repurge_logits.py --run

WHY A DEDICATED SCRIPT RATHER THAN `ch_ingest.py --index`. The indexed ingest
skips cells already in `logit_residual`, so deleting the exposed cells and
re-running it WOULD repair them -- and would also sweep in every other cell
that happens to be un-ingested at that moment. There were 253 of those when
this was written. They are not wrong, and ingesting them is not harmful, but
**an operation that does more than it was agreed to do is not the operation
that was agreed** ([5294]/[5295]), and the delta afterwards would no longer
be attributable. This does the named 1,391 and nothing else.

THE HAZARD THIS SCRIPT EXISTS TO AVOID IS MUTATION ORDERING. ClickHouse
`ALTER TABLE ... DELETE` is ASYNCHRONOUS. It applies to the parts that exist
when it is submitted, so an insert racing a running mutation is a coin flip on
whether the new rows survive -- and the failure is silent, leaving a cell
deleted and not replaced. **Every mutation is waited on to `is_done = 1`
before a single row is written back.**

THE KEY LIST IS READ, NEVER COMPUTED HERE. `data/ch_logit_repurge.json` was
produced by measuring exposure before anything changed; recomputing it now
would silently re-scope the repair to whatever the store looks like today.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
KEYS = os.path.join(ROOT, "data", "ch_logit_repurge.json")
TRUNC = 1e-6
TABLES = ("logit_probs", "logit_residual")

#: the other project's databases. Same list as ch_ingest's guard, and the same
#: reason: this script issues DELETEs, which is the one verb that cannot be
#: taken back by re-running it.
FORBIDDEN = ("lltk", "abstraction", "llmtasks", "tmp", "default", "system")


def q(sql, quiet=False):
    """Refuse WRITES against anything but our database. Reads of `system` are
    required and allowed.

    **THE FIRST VERSION REFUSED ANY STATEMENT NAMING `system`, AND IT FIRED
    BETWEEN THE DELETE AND THE INSERT** -- after both `ALTER TABLE ... DELETE`
    mutations had been submitted and before a single row was written back,
    which is precisely the half-repaired state the mutation-ordering care in
    this module's docstring exists to prevent. `wait_mutations` reads
    `system.mutations`; it has to.

    That is the same over-broad guard I already had to rescope in
    `ch_ingest.py` earlier today, reintroduced here from memory rather than
    from the fixed original. A guard is only safe if it distinguishes the verb
    from the noun: what is dangerous is WRITING to another database, not
    mentioning one.
    """
    low = sql.lower()
    verbs = ("drop", "truncate", "alter", "insert", "delete", "create", "rename")
    is_write = any(low.lstrip().startswith(v) for v in verbs)
    for bad in FORBIDDEN:
        if ("%s." % bad) in low and is_write:
            raise SystemExit("REFUSING: WRITE naming %r\n%s" % (bad, sql[:200]))
    if is_write and DB not in sql:
        raise SystemExit("REFUSING: write does not name %s\n%s" % (DB, sql[:200]))
    r = subprocess.run([CH, "client", "--query", sql], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("clickhouse: %s" % r.stderr.strip()[:400])
    return r.stdout


def esc(x):
    return x.replace("\\", "\\\\").replace("'", "\\'")


def wait_mutations(timeout=1800):
    """Block until no mutation on our tables is outstanding."""
    t0 = time.time()
    while True:
        out = q("SELECT count() FROM system.mutations WHERE database='%s' "
                "AND is_done = 0 FORMAT TSV" % DB).strip()
        n = int(out or 0)
        if n == 0:
            return
        if time.time() - t0 > timeout:
            raise SystemExit("mutations still running after %ds -- NOT inserting"
                             % timeout)
        print("    ... %d mutation(s) outstanding" % n)
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--keys", default=KEYS)
    a = ap.parse_args()

    import numpy as np
    from malign_logits.cache import CacheManager
    cm = CacheManager()

    keys = json.load(open(a.keys))
    print("key list %s\n  cells %s\n" % (a.keys, format(len(keys), ",")))

    # ── resolve every cell BEFORE deleting anything ───────────────────
    #
    # **RESOLVE FIRST, DELETE SECOND.** If a cell turns out to be unreadable,
    # this must find out while the old rows are still in place: a delete
    # followed by a failed read leaves the store worse than the defect did.
    plan, unreadable = [], []
    for k in keys:
        ent = cm.get_logits_entry(k["model"], k["prompt"], mode="raw",
                                  dtype="float16")
        if not ent:
            unreadable.append((k["model"], k["prompt"], "no index entry"))
            continue
        path = cm.logit_path(ent)
        if not os.path.exists(path):
            unreadable.append((k["model"], k["prompt"], "payload missing"))
            continue
        row, dim = int(ent["row"]), int(ent["dim"])
        mm = np.memmap(path, dtype=np.float16, mode="r")
        if (row + 1) * dim > mm.size:
            unreadable.append((k["model"], k["prompt"], "row past EOF"))
            continue
        plan.append((k["model"], k["prompt"], path, row, dim))

    print("resolved and readable  %s" % format(len(plan), ","))
    print("UNREADABLE (skipped, NOT deleted)  %s" % format(len(unreadable), ","))
    for m, p, why in unreadable[:5]:
        print("    %-42s %s" % (m.split("/")[-1][:42], why))

    dirs = {}
    for _, _, path, _, _ in plan:
        d = "data/f11_twp" if "/data/f11_twp/" in path else "cloud_run"
        dirs[d] = dirs.get(d, 0) + 1
    print("resolved directories   %s" % dirs)

    if a.dry_run:
        print("\nDRY RUN -- nothing deleted, nothing written.")
        return 0

    # ── delete, wait, re-insert ───────────────────────────────────────
    #
    # Deleting only what will be replaced: `plan`, not `keys`. An unreadable
    # cell keeps whatever it has, which may be wrong, and is reported above
    # rather than emptied.
    conds = " OR ".join("(model='%s' AND prompt='%s')" % (esc(m), esc(p))
                        for m, p, _, _, _ in plan)
    for t in TABLES:
        print("\ndeleting from %s ..." % t)
        q("ALTER TABLE %s.%s DELETE WHERE %s" % (DB, t, conds))
    print("waiting for mutations to finish before writing back ...")
    wait_mutations()

    left = int(q("SELECT count() FROM %s.logit_residual WHERE %s FORMAT TSV"
                 % (DB, conds)).strip() or 0)
    if left:
        raise SystemExit("REFUSING to insert: %d residual rows survived the "
                         "delete, so the mutation did not do what it said" % left)
    print("confirmed: 0 rows remain for the purged cells")

    out, res, n_rows, n_cells = [], [], 0, 0
    bypath = {}
    for model, prompt, path, row, dim in plan:
        if path not in bypath:
            bypath[path] = np.memmap(path, dtype=np.float16, mode="r")
        v = np.asarray(bypath[path][row * dim:(row + 1) * dim], dtype=np.float32)
        v = v - v.max(); np.exp(v, out=v); v /= v.sum()
        idx = np.flatnonzero(v >= TRUNC)
        lp = np.log(v[idx]).astype(np.float32)
        out.extend({"model": model, "prompt": prompt, "token_id": int(t),
                    "logprob": float(x)} for t, x in zip(idx, lp))
        res.append({"model": model, "prompt": prompt, "threshold": TRUNC,
                    "kept": int(idx.size), "dim": dim,
                    "mass_kept": float(v[idx].sum())})
        n_cells += 1
        if len(out) >= 400_000:
            insert("logit_probs", out); n_rows += len(out); out = []
    insert("logit_probs", out); n_rows += len(out)
    insert("logit_residual", res)
    print("\nre-ingested %s cells, %s token rows" % (format(n_cells, ","),
                                                     format(n_rows, ",")))
    back = int(q("SELECT uniqExact((model,prompt)) FROM %s.logit_residual "
                 "WHERE %s FORMAT TSV" % (DB, conds)).strip() or 0)
    print("cells present again: %s of %s" % (format(back, ","),
                                             format(len(plan), ",")))
    return 0 if back == len(plan) else 1


def insert(table, rows):
    if not rows:
        return
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    r = subprocess.run(
        [CH, "client", "--query",
         "INSERT INTO %s.%s FORMAT JSONEachRow" % (DB, table)],
        input=payload, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("insert into %s: %s" % (table, r.stderr.strip()[:400]))


if __name__ == "__main__":
    sys.exit(main())
