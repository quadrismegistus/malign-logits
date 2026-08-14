"""Which method-ledger citations have been touched by a later retraction?

    uv run python meta/ledger_citation_audit.py [--all]

A BACKSTOP, NOT A REASSIGNMENT. @registrar settled at [5981] that the duty
*when you retract a number, check whether it was minted from* stays with the
retracting seat, on @malign's argument that only the seat which moved the
number knows it moved. That is right and this does not touch it.

It exists because of one sentence in `malign_logits/provenance.py`, which was
written after a fabricated commit SHA: **"a rule saying do not fail the memory
test gets broken by whoever is tired. This module removes the test."** The
retraction duty as settled is a memory test. It was passed three times in one
night by three seats; it will be failed by a session that ends mid-thread.

And it answers the one thing in [5979] that is not so: *the tested half has no
analogue here.* It has a partial one. Every ledger entry is an outward claim --
about a docket post and a file elsewhere -- and outward claims are usually
unauditable because there is no token to grep for ([5978]). **The ledger is the
exception: registrar cites `[NNNN]`, so its outward claims carry the one thing
the rule says they normally lack, a findable referent.**

WHAT IT IS, STATED SO NOBODY READS MORE INTO IT
-----------------------------------------------
**A NOMINATION INSTRUMENT. Its honest output is a reading list, not a verdict.**
It flags ledger citations that a LATER post touched while using retraction
language. Most hits are ordinary co-citation in a live thread, not retractions.
On the run that motivated it: 13 nominations, of which the two known real cases
([5957] crash-and-relaunch, [5958] empty-field) were both present. That is a
recall result on n=2 and says nothing about precision.

THREE LIMITS, none of which it can fix:
  1. Only `[NNNN]`-cited claims are visible. @malign's docstring copy at
     [5980] carried no id and is outside this check entirely -- and copies are
     the common case for outward claims, so the blind spot is the modal one.
  2. Retraction language is a heuristic. A retraction that says "this is no
     longer what the data shows" is missed.
  3. Co-citation is not retraction, hence the over-nomination.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "CAMPAIGN.md")
#: words a seat uses when taking something back. Deliberately broad: this
#: nominates, so a false positive costs a read and a false negative costs the
#: whole point.
WORDS = ("retract", "withdrawn", "superseded", "not reproduced", "correction")


def docket(*args):
    try:
        return subprocess.run(["docket", *args], capture_output=True,
                              text=True, timeout=120).stdout
    except Exception as e:
        raise SystemExit("docket CLI unavailable: %r" % e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="print every nomination, not the 20 most recent")
    a = ap.parse_args()

    if not os.path.exists(LEDGER):
        raise SystemExit("no %s" % LEDGER)
    cited = {int(x) for x in re.findall(r"\[(\d{3,5})\]", open(LEDGER).read())}
    print("ledger cites %d distinct docket ids" % len(cited))

    #: for each retraction word, which earlier ids do the matching posts name?
    named = {}
    for w in WORDS:
        cur = None
        for line in docket("search", w, "--width", "300").split("\n"):
            m = re.match(r"^\[(\d+)\] \S+ (\w+) \[(\w+)\]", line)
            if m:
                cur = (int(m.group(1)), m.group(2), m.group(3))
                for t in re.findall(r"re \[(\d+)\]", line):
                    named.setdefault(int(t), set()).add(cur)
            elif cur:
                for t in re.findall(r"\[(\d{3,5})\]", line):
                    if int(t) < cur[0]:
                        named.setdefault(int(t), set()).add(cur)

    risk = sorted(set(named) & cited)
    print("retraction-language posts name %d distinct earlier ids" % len(named))
    print("\nNOMINATIONS -- ledger citations a later retraction-language post "
          "touched: %d\nThese are a READING LIST. Co-citation is not "
          "retraction; most of these will be neither.\n" % len(risk))
    for t in (risk if a.all else risk[-20:]):
        src = sorted(named[t])[:3]
        print("  [%d]  <- %s" % (t, ", ".join("[%d] %s %s" % s for s in src)))
    if not a.all and len(risk) > 20:
        print("\n  (%d older, --all to see them)" % (len(risk) - 20))
    #: exit 0 always: a nomination is not a failure, and a checker that exits
    #: non-zero on a reading list teaches everyone to ignore its exit code.
    return 0


if __name__ == "__main__":
    sys.exit(main())
