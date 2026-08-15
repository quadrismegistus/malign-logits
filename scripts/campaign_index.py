#!/usr/bin/env python
"""The method ledger as a one-line-per-rule index.

    uv run python scripts/campaign_index.py           # every rule
    uv run python scripts/campaign_index.py grep WORD # rules matching WORD
    uv run python scripts/campaign_index.py --since 6200   # booked from [N] on

WHY THIS EXISTS. `CAMPAIGN.md`'s method ledger is **166 rules across 2,598
lines in one flat section**, and 51 of them were added on 2026-08-15 alone.
The document's stated job is to be re-read after a compaction. **Nobody
re-reads 2,600 lines**, so in practice the ledger is consulted by whoever
already remembers which rule they want -- which is exactly the population
that does not need it.

**IT GENERATES RATHER THAN TRANSCRIBES, AND THAT IS THE POINT.** A
hand-written index is stale the first time anyone books a rule without
updating it, and this campaign booked the one-way-link defect twice on
2026-08-15: discharges written in queue entries that never reached the pool
([6179]), and a BLOCKED -> DEAD transition written in STATUS CHANGES that
never reached its entry ([6258]). **An index maintained by hand is that
defect with a schedule.** Same argument as `ch.inventory_md()`, which
replaced a row table that had drifted by up to 2.9x.

WHAT IT DOES NOT DO. It does not rank, group, or summarise. The opener line
of a rule is the author's own compression of it and this prints that,
nothing else. **A rule whose opener does not say what it is has a writing
problem the index cannot fix**, and surfacing that is useful.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "CAMPAIGN.md")
START = "## The method ledger, short form"


def rules():
    """(line_no, opener, body) for every rule in the ledger section."""
    lines = open(DOC, encoding="utf-8").read().splitlines()
    try:
        first = next(i for i, l in enumerate(lines) if l.startswith(START))
    except StopIteration:
        sys.exit("no '%s' heading in CAMPAIGN.md" % START)
    out, cur = [], None
    for i in range(first + 1, len(lines)):
        l = lines[i]
        #: A RULE STARTS AT A BOLD-CAPS BULLET. Continuation lines are
        #: indented; a nested bullet inside a rule is not a new rule.
        if re.match(r"^- \*\*[A-Z]", l):
            if cur:
                out.append(cur)
            cur = [i + 1, l, []]
        elif cur is not None:
            cur[2].append(l)
    if cur:
        out.append(cur)
    return out


def opener(rule):
    """The rule's own headline, unwrapped and stripped of markup."""
    _, head, body = rule
    text = head[2:]
    #: THE OPENER OFTEN WRAPS. Keep pulling continuation lines while the
    #: bold run is still open, so a rule whose name spans three lines
    #: prints as one -- the wrap is a fact about the margin, not the rule.
    for l in body:
        if text.count("**") % 2 == 0:
            break
        text += " " + l.strip()
    m = re.match(r"\*\*(.+?)\*\*", text, re.S)
    name = (m.group(1) if m else text)
    return re.sub(r"\s+", " ", name).strip(" *")


def main():
    args = sys.argv[1:]
    rs = rules()
    if args and args[0] == "grep" and len(args) > 1:
        pat = re.compile(args[1], re.I)
        rs = [r for r in rs if pat.search(r[1] + "\n".join(r[2]))]
    elif args and args[0] == "--since" and len(args) > 1:
        lo = int(args[1])
        keep = []
        for r in rs:
            ids = [int(x) for x in re.findall(r"\[(\d{4,5})\]", "\n".join(r[2]) + r[1])]
            if ids and max(ids) >= lo:
                keep.append(r)
        rs = keep
    for r in rs:
        print("  %5d  %s" % (r[0], opener(r)))
    print("\n  %d rules. CAMPAIGN.md is %d lines; the ledger is one flat section."
          % (len(rs), sum(1 for _ in open(DOC, encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
