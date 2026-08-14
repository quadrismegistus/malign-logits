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
    4. the artifact is THE artifact       (right study is not right instrument)

Condition 2 is the one that would have caught the bad promotion, and it is the
one nobody runs, because condition 1 feels like the check.

**CONDITION 4 CAME FROM THE NEXT FAILURE, AND ALL THREE OTHERS PASSED** ([6182],
dario). `queue 17` was promoted on 1-3 and is not drawable: its numbers are on
the SPAN `<guilt>` and its artifact carries the FIELD `guilt_or_shame` plus
span COUNTS with no per-span labels. dario computed the field version and the
MEDIAN matched (+0.752 against +0.8) while every tail compressed -- **a
population change moves the centre and this did not; a broader instrument
catching borderline cases pulls extremes inward.** The artifact is the right
study, the right passages, the wrong instrument. *A producer that writes AN
artifact is not a producer that writes THE value*, one step earlier in the
chain. **This condition is NOT MECHANISED and is flagged for the reader**: the
tool prints the artifact's columns for any entry it calls plausibly open, so a
human can ask whether the quantity is even in there.

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

    #: 4. WHAT IS ACTUALLY IN THE ARTIFACT. Not a verdict -- a prompt for the
    #: human. queue 17 passed 1-3 and failed here: right study, wrong
    #: instrument, and no column in the file could have produced its numbers.
    cols = []
    for h in have:
        if h.endswith(".parquet"):
            try:
                import pandas as pd
                cols = list(pd.read_parquet(h).columns)
            except Exception:
                cols = ["<unreadable>"]
            break
    return {"artifact": bool(have) if named else None, "named": named,
            "numbers": nums, "drawn_by": drawn_by, "marked": marked,
            "tracked": all(_tracked(h) for h in have) if have else None,
            "columns": cols}


def reproduce(entry_numbers, path):
    """Do the entry's numbers fall out of the named file, and as WHICH aggregate?

    **THE STEP THE FIRST TWO PROMOTIONS SKIPPED.** Conditions 1-4 ask whether
    the artifact exists, is the right one, and is unused. This asks the
    question underneath: does the headline actually come out of it.

    And it reports the aggregate, because `queue 18` turned on that. Its
    entry claims twin 0.327 / random 0.060; those are MEANS, and the medians
    are 0.336 / 0.054. **A per-family dot plot reaches for the median by
    default and the gap reads as rounding rather than as a different
    statistic** -- the same trap that cost two seats an hour on `queue 17`,
    where three populations shared a median to one decimal and disagreed in
    every tail.
    """
    try:
        import pandas as pd
        d = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    except Exception as e:
        return {"error": str(e)[:60]}
    want = [float(x) for x in entry_numbers if re.match(r"^-?\d+\.\d+$", x)]
    if not want:
        return {}
    hits = {}
    for col in d.select_dtypes("number").columns:
        for agg in ("mean", "median", "sum", "max", "min"):
            v = getattr(d[col], agg)()
            for w in want:
                if abs(v - w) < 0.0006:
                    hits.setdefault(w, []).append("%s.%s=%.4f" % (col, agg, v))
    return {"matched": hits,
            "unmatched": [w for w in want if w not in hits],
            "rows": len(d)}


def _tracked(path):
    import subprocess
    r = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


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
                print("  %-10s %s" % (k, str(v)[:140]))
            #: EVERY MATCHING FILE, NOT THE FIRST. A glob on
            #: `v_displacement_twin*` returns five files; the first version
            #: took `[0]` and reported the entry's numbers UNMATCHED because
            #: it happened to read the residualised variant. **A false
            #: negative that would have rejected a good candidate**, and the
            #: same first-hit trap as reading an entry's first line. The
            #: winner is whichever file reproduces, and which one it is IS
            #: the answer the drawer needs.
            for h in sorted(glob.glob(os.path.join(ROOT, "**", "*%s*" %
                            os.path.basename(st["named"][0]).split("{")[0]),
                            recursive=True) if st["named"] else []):
                if ".git" in h or not h.endswith((".csv", ".parquet")):
                    continue
                r = reproduce(st["numbers"], h)
                if r.get("matched"):
                    print("  REPRODUCES %s -> %s" % (os.path.basename(h), r["matched"]))
                elif r.get("unmatched"):
                    print("  no match   %s" % os.path.basename(h))
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
