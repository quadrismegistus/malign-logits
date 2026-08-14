#!/usr/bin/env python
"""Is a plot-debt entry actually open? Three conditions, not one.

    uv run python scripts/plot_debt_state.py            # every candidate entry
    uv run python scripts/plot_debt_state.py --open     # only the plausibly open
    uv run python scripts/plot_debt_state.py M01 15     # one entry, verbose

WHY THIS EXISTS. On 2026-08-14 the registrar promoted `shortlist 2` to the
queue and it had been shipped that morning by the seat it was promoted to.
**The check that failed was checking INPUTS: both artifacts existed — and they
existed precisely because the figure had already been drawn from them.**
`shortlist 1`, offered as the replacement, was shipped too. dario then mapped
the pool: **all ten shortlist entries shipped or blocked, and six did not say
so** ([6179], [6180]).

The pool has no marking discipline that survives a busy day, while the queue
beside it does — and the discharges were written in the QUEUE entries, so the
link existed, was correct, and ran ONE WAY. A reader starting from the pool
finds open items.

**THE THREE CONDITIONS.** An entry is plausibly open only if all hold:

    1. its named artifact EXISTS          (else it is blocked, not open)
    2. NO producer asserts its numbers    (the one that catches a drawn figure)
    3. NO figure matches it on disk

Condition 2 is the one that would have caught the bad promotion, and it is the
one nobody runs, because condition 1 feels like the check.

**WHAT THIS TOOL IS NOT.** It does not prove an entry is open. Absence of a
number match is not absence of a figure — most entries name too few
distinctive values for the test to have power, which is why a sweep found
producers for only 5 of 59 and that number means almost nothing on its own.
**Its output is a shortlist for the per-entry check, not a verdict.** Read
what the entry specifies, then look for a producer asserting it; that is what
worked on the shortlist and it is per-entry work.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBT = os.path.join(ROOT, "meta", "plot-debt.md")


def entries():
    """(folder, n, text) for every per-folder candidate entry."""
    s = open(DEBT, encoding="utf-8").read()
    pf = s[s.index("## Per-folder candidate lists"):s.index("## Fences: do-not-plot")]
    #: AN ENTRY IS A BLOCK, NOT A LINE. The first version of this function read
    #: one line per entry and missed every continuation -- so `M01 15`'s named
    #: artifact, which sits on its second line, was invisible and the entry
    #: reported `named []`. That is dario's own [6177] finding -- *a first line
    #: is a name for an entry, not a relation to it* -- reproduced inside the
    #: tool written to stop exactly this. Accumulate until the next numbered
    #: line or the next heading.
    folder, out, cur = None, [], None
    def flush():
        if cur:
            out.append((cur[0], cur[1], re.sub(r"\s+", " ", " ".join(cur[2]))))
    for line in pf.splitlines():
        h = re.match(r"^### (\S+)", line)
        if h:
            flush(); cur = None
            folder = h.group(1)
            continue
        m = re.match(r"^(\d+)\. (.+)", line)
        if m and folder:
            flush()
            cur = (folder, m.group(1), [m.group(2)])
        elif cur is not None and line.strip() and line.startswith(("    ", "\t", "   ")):
            cur[2].append(line.strip())
        elif cur is not None and not line.strip():
            pass
    flush()
    return out


def _numbers(txt):
    """Distinctive values an entry declares. Decimals first: they discriminate."""
    dec = re.findall(r"-?\d+\.\d+", txt)
    ints = [x for x in re.findall(r"\b\d{2,5}\b", txt) if x not in dec]
    return (dec + ints)[:3]


def state(folder, n, txt, sources=None, figs=None):
    mod = folder.split("_")[0]
    if sources is None:
        sources = {f: open(f, errors="ignore").read()
                   for f in glob.glob(os.path.join(ROOT, "meta", "*", "scripts", "*.py"))}
    if figs is None:
        figs = glob.glob(os.path.join(ROOT, "meta", "*", "figures", "*.png"))

    #: 1. the artifact it names
    named = re.findall(r"`([^`]+\.(?:csv|json|parquet|tsv))`", txt)
    have = []
    for a in named:
        base = os.path.basename(a).split("{")[0]
        have += [h for h in glob.glob(os.path.join(ROOT, "**", "*%s*" % base),
                                      recursive=True) if ".git" not in h][:1]

    #: 2. a producer asserting its numbers -- the condition that catches a
    #: figure already drawn, and the one the bad promotion skipped
    nums = _numbers(txt)
    drawn_by = ""
    if len(nums) >= 2:
        for f, code in sources.items():
            if mod in f and all(x in code for x in nums[:2]) \
               and re.search(r"\.save\(|savefig", code):
                drawn_by = os.path.basename(f)
                break

    #: 3. an explicit discharge already written in the entry
    marked = bool(re.search(r"SHIPPED|DISCHARGED|BLOCKED", txt))
    return {"artifact": bool(have) if named else None, "named": named,
            "numbers": nums, "drawn_by": drawn_by, "marked": marked}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only_open = "--open" in sys.argv
    rows = entries()
    if len(args) == 2:
        rows = [r for r in rows if r[0].startswith(args[0]) and r[1] == args[1]]
        for f, n, t in rows:
            st = state(f, n, t)
            print("%s candidate %s\n  %s\n" % (f, n, t))
            for k, v in st.items():
                print("  %-10s %s" % (k, v))
        return 0

    sources = {f: open(f, errors="ignore").read()
               for f in glob.glob(os.path.join(ROOT, "meta", "*", "scripts", "*.py"))}
    n_open = 0
    print("  %-22s %-3s %-40s %s" % ("FOLDER", "#", "ENTRY", "STATE"))
    for f, n, t in rows:
        st = state(f, n, t, sources)
        if st["marked"]:
            lab = "marked discharged"
        elif st["drawn_by"]:
            lab = "LIKELY DRAWN by %s" % st["drawn_by"]
        elif st["artifact"] is False:
            lab = "artifact missing"
        else:
            lab = "plausibly open -- VERIFY PER-ENTRY"
            n_open += 1
        if only_open and "plausibly" not in lab:
            continue
        print("  %-22s %-3s %-40s %s" % (f, n, t[:40], lab))
    print("\n  %d entries, %d plausibly open." % (len(rows), n_open))
    print("  PLAUSIBLY OPEN IS NOT OPEN. Read what the entry specifies and look")
    print("  for a producer asserting it before promoting -- inputs-present is")
    print("  what made queue 16 wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
