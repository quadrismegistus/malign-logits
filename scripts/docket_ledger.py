"""DERIVE THE LEDGER FROM THE DOCKET. The rules exist; the index does not.

    .venv/bin/python scripts/docket_ledger.py              # print
    .venv/bin/python scripts/docket_ledger.py --write docs/ledger.md
    .venv/bin/python scripts/docket_ledger.py --seat malign --since 1700

WHY. [1930].7, from the pen: roughly forty rules have been booked across three
seats and EVERY ONE LIVES INSIDE A NUMBERED POST, interleaved with the
adjudication that produced it. `docket show` made them retrievable; retrievable
is not indexed. **EVERY RULE THIS PROJECT HAS BOOKED ABOUT KEEPING RECORDS
HONEST IS STORED IN THE ONE RECORD THAT FOLLOWS NONE OF THEM.**

WHAT THIS IS AND IS NOT. It is a READ over the docket's own sqlite store,
opened READ-ONLY. It changes nothing about the docket, and it does not preempt
the queryable-tags proposal at [1930] — that changes the tool and is RH's; this
derives an index from the artifact and is reversible by deleting one file.

THE REMEDY IT APPLIES IS THE ONE THE PEN NAMED: **derive the prose from the
artifact, or delete the prose.** Seven instances of prose-not-matching-artifact
were caught on 2026-08-01 alone — a finding's frontmatter, a producer's
docstring, a spec's opening, two file headers, a guide's worked example, a
commit message, and a JSON row reading as clean while carrying two lint
failures. A hand-maintained ledger file would have been the eighth.

STALENESS IS STAMPED, NOT ASSUMED. `--write` records the post id the index was
derived at and the count. A reader who sees a docket at [2100] against an index
stamped [1931] knows without checking anything else. **A DERIVED FILE THAT DOES
NOT CARRY ITS DERIVATION POINT IS A HEADER WAITING TO GO STALE** — which is
exactly the defect that cost the M03 redraft a scope number today.

WHAT IT CANNOT DO. The extraction is a regex over prose written by three seats
who agreed on no format. It finds the `Ledger:` / `Ledger, sharpened:` /
`Ledger candidate:` family. **A RULE STATED WITHOUT THAT WORD IS INVISIBLE TO
IT, and the count below is a floor, never a total.** Reported as a floor.
"""

import argparse
import os
import re
import sqlite3
import sys

DB = os.path.expanduser("~/.agentdocket/docket.db")

#: The families three seats actually used. Checked against the corpus rather
#: than assumed -- `Ledger candidate:` and `Ledger, sharpened:` both occur and
#: a bare `\bLedger:` pattern would have silently dropped them.
LEDGER = re.compile(
    r"Ledger(?:\s+[a-z][^:*]{0,60})?[:,]\s*\*\*(?P<rule>.+?)(?:\*\*|$)",
    re.S)


def rows(db, seat=None, since=0):
    con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
    #: The column is `sender`, not `seat` -- read from the schema, not guessed
    #: from the CLI's vocabulary, which calls the same thing a seat throughout.
    q = "SELECT id, sender, ts, body FROM messages WHERE id > ?"
    args = [since]
    if seat:
        q += " AND sender = ?"
        args.append(seat)
    try:
        return con.execute(q + " ORDER BY id", args).fetchall()
    finally:
        con.close()


def extract(body):
    """Every ledger line in one post. A post may book more than one."""
    out = []
    for m in LEDGER.finditer(body):
        rule = " ".join(m.group("rule").split())
        if len(rule) > 12:
            out.append(rule)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--seat", help="only rules booked by this seat")
    ap.add_argument("--since", type=int, default=0, help="only posts after this id")
    ap.add_argument("--write", metavar="PATH", help="write a stamped markdown index")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"no docket store at {args.db}", file=sys.stderr)
        return 2

    found, head, n_posts = [], 0, 0
    for pid, seat, ts, body in rows(args.db, args.seat, args.since):
        n_posts += 1
        head = max(head, pid)
        for rule in extract(body):
            found.append((pid, seat, ts[:10], rule))

    lines = [f"# Ledger — derived from the docket at [{head}]", "",
             f"**{len(found)} rules extracted from {n_posts} posts. THIS IS A FLOOR:**",
             "a rule stated without the word *Ledger* is invisible to the",
             "extractor. Regenerate with `scripts/docket_ledger.py --write`.", ""]
    for pid, seat, day, rule in found:
        lines.append(f"- **[{pid}]** `{seat}` {day} — {rule}")

    text = "\n".join(lines) + "\n"
    if args.write:
        with open(args.write, "w") as f:
            f.write(text)
        print(f"wrote {args.write} — {len(found)} rules, derived at [{head}] "
              f"over {n_posts} posts")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
