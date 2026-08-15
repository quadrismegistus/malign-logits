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


WORD_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "thirteen": 13, "fourteen": 14, "fifteen": 15}


def population(path):
    """Rows in the named artifact, or None if it cannot be counted cheaply."""
    try:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
        if path.endswith((".csv", ".tsv")):
            with open(path, errors="ignore") as f:
                return sum(1 for _ in f) - 1
        if path.endswith((".jsonl", ".json")) and path.endswith(".jsonl"):
            with open(path, errors="ignore") as f:
                return sum(1 for line in f if line.strip())
    except Exception:
        return None
    return None


def selection(txt, path):
    """CONDITION 5: does the entry quote a count SMALLER than its file holds?

    **dario's finding on the M03 audit ([6197]), and it would have pointed at
    five of seven entries.** Conditions 1-4 ask whether the artifact is the
    right one. This asks what the entry did to it. *65 rows* against a
    702-row file, *six verbs* against eleven, *eight domains* against
    thirteen, *95 shared fields* against 267 -- in every case the number was
    real and the route to it was not the obvious one.

    **A COUNT IN AN ENTRY IS A SELECTION RULE COMPRESSED TO AN INTEGER, AND
    THE RULE LIVES IN THE FINDING, NEVER IN THE ENTRY.** The drawer who reads
    only the entry reconstructs the number by the obvious route and gets a
    different one: dario split `b_word_delta_by_word` on which arm moved more
    and got 37/28 against the booked 43/22, because the split is by the SIGN
    of `median_d`. Same file, same 65 words, wrong figure.

    Advisory, like condition 4. A mismatch means GO READ THE DOC; it does not
    mean the entry is wrong. Section numbers and years trip it too, which is
    the acceptable direction to be wrong in.

    **THE DENOMINATOR IS A ROW COUNT AND THE ENTRY MAY NOT BE ABOUT ROWS.**
    Candidate 7 prints `95 of 763003` and the honest comparison is 95 of 267
    FIELDS -- the entry counts columns, the checker counts rows. The flag is
    right and the number beside it is the wrong instrument, which is
    condition 4 turned on this function. **Read the printed denominator as a
    reason to look, never as the entry's own denominator.**

    Measured against dario's hand audit of M03: fires on all five it found,
    plus candidate 8, which dario also flagged as spanning two populations.
    Both clauses were mutation-checked ACROSS THE WHOLE POOL and both move
    the count (19 real; 24 without the smaller-than filter; 14 without the
    spelled-out words). **On M03's seven alone the smaller-than clause looked
    vacuous** -- a sample too small to discriminate will report a live clause
    as dead, so mutation-check on the full population or not at all.
    """
    pop = population(path)
    if not pop or pop < 2:
        return None
    #: SPELLED-OUT NUMBERS COUNT. The first version matched `\d{2,6}` only and
    #: missed two of the five dario found by hand -- *six verbs of eleven* and
    #: *eight domains of thirteen* are selections written in words. **A digit
    #: predicate is a claim that selections are written in digits**, and half
    #: of these are not.
    #: A DOCKET ID IS NOT A COUNT, AND NEITHER IS A CHECKPOINT STEP. Found by
    #: hand-checking the ONE flag outside the sample this check was tuned on:
    #: `M05 5` flagged [5430, 257, 128], of which only 257 is a selection --
    #: 5430 is the docket post that qualified the entry and 128 is
    #: `step 128`. **A checker validated only on the sample that motivated it
    #: is the subsample error one level up**, which is the rule this same
    #: commit booked. Strip both before counting.
    #: **THE STRIP MOVES NO TALLY: 20 entries flag with it and 20 without.**
    #: It changes the integers PRINTED BESIDE each flag, which is the whole
    #: output a human acts on -- so the count test says it is dead and it is
    #: not. A mutation that leaves the headline unchanged can still be
    #: load-bearing on the part a reader uses; measure the thing consumed,
    #: not the thing counted.
    clean = re.sub(r"\[\d{3,6}\]|\bstep\s*\d+", " ", txt, flags=re.I)
    got = {int(x) for x in re.findall(r"\b\d{2,6}\b", clean)}
    got |= {v for w, v in WORD_NUM.items()
            if re.search(r"\b%s\b" % w, txt, re.I)}
    small = sorted({v for v in got if 1 < v < pop}, reverse=True)
    if not small:
        return None
    return {"file_rows": pop, "entry_counts": small[:4]}


RETRACT = re.compile(
    r"retract|withdraw|superseded|supersedes|refuted|refutes|does not hold|did not hold"
    r"|does NOT replicate|is dead|now dead|confounded", re.I)


def retracted(folder, txt):
    """CONDITION 6: is the SECTION this entry cites retracted further down it?

    **dario, [6203], drawing M04 candidate 8.** Section 3's headline table is
    alive and the ordering three paragraphs below it is dead: FALLER <
    NONMOVER < RISER is withdrawn at `### Retraction:` six lines under the
    table it summarises, because base probabilities 0.062 / 0.089 / 0.201
    make "ordered by alignment status" and "ordered by how probable the word
    was" the same ordering on that cell.

    **A FIGURE IS WHERE A RETRACTED RESULT GOES TO BE REVIVED.** The picture
    outlives the paragraph that withdrew it, and nobody re-reads a retraction
    while looking at a PNG. **Nothing at the table says the table is dead.**

    SECTION-SCOPED ON PURPOSE. Retraction vocabulary appears somewhere in
    most findings documents -- WITHDRAW in 36 files, SUPERSED in 28 -- so a
    document-level flag fires on nearly everything and means nothing. Scoped
    to the cited section it points at the paragraph that matters.

    Advisory. Returns the retracting headings so the reader can judge.
    """
    #: THE LETTER SUFFIX IS THE WHOLE QUESTION. The first version read
    #: `§3d` as `§3` and flagged four M01 entries against a withdrawal that
    #: lives in `§3a` -- sibling subsections of a section whose body says
    #: **"Safety survives"**. Citing §3d is not citing §3a. Capture the
    #: suffix and report a sibling retraction as SIBLING, because **a flag
    #: stated at a precision it has not earned is the false-CUT error**
    #: ([6199]) in a different tool.
    hits = []
    for doc_tok, sec, suf in re.findall(
            r"\b([A-Z][A-Za-z_]*)\s*§\s*(\d+)([a-z]?)", txt):
        for path in glob.glob(os.path.join(ROOT, "meta", folder,
                                           "findings", "*.md")):
            base = os.path.basename(path).lower()
            if not (base.startswith(doc_tok.lower() + "_")
                    or base.startswith(doc_tok.lower() + ".")):
                continue
            body = open(path, errors="ignore").read().splitlines()
            #: THE FRONT MATTER AND THE LETTERED SIBLINGS. dario [6205]:
            #: *"I had read as far as the retraction and stopped, because the
            #: retraction was what I was checking for."* **A grep for
            #: retractions finds the one you grep for.** Worse here: this
            #: doc's `## 3b.` is a TOP-LEVEL sibling of `## 3.`, so a scan
            #: that stops at the next `##` terminates EXACTLY where the
            #: supersession begins -- and the document says so in its front
            #: matter, before section 1: *"sections 2 and 3 are superseded by
            #: 3b and 3c wherever they disagree."* The tool read neither.
            first = next((k for k, l in enumerate(body)
                          if re.match(r"^##\s+(?!#)", l)), len(body))
            for line in body[:first]:
                if RETRACT.search(line) and re.search(r"\bsection", line, re.I):
                    hits.append("%s [FRONT MATTER] %s"
                                % (os.path.basename(path), line.strip()[:64]))
            for k, line in enumerate(body):
                m2 = re.match(r"^##\s+%s([a-z])[.\s]" % re.escape(sec), line)
                nxt = next((j for j in range(k + 1, len(body))
                            if re.match(r"^##\\s+(?!#)", body[j])), len(body))
                if m2 and RETRACT.search("\n".join(body[k:nxt])):
                    hits.append("%s [SIBLING SECTION %s%s] %s"
                                % (os.path.basename(path), sec, m2.group(1),
                                   line.lstrip("# ").strip()[:56]))
            start = None
            for i, line in enumerate(body):
                if re.match(r"^##\s+%s[.\s]" % re.escape(sec), line):
                    start = i
                elif start is not None and re.match(r"^##\s+(?!#)", line):
                    break
            else:
                i = len(body)
            if start is None:
                continue
            for line in body[start + 1:i]:
                if not (line.startswith("#") and RETRACT.search(line)):
                    continue
                head = line.lstrip("# ").strip()
                m = re.match(r"^%s([a-z])\b" % re.escape(sec), head)
                if suf and m and m.group(1) != suf:
                    kind = "SIBLING %s%s" % (sec, m.group(1))
                elif suf and m and m.group(1) == suf:
                    kind = "CITED %s%s" % (sec, suf)
                else:
                    kind = "IN §%s" % sec
                hits.append("%s [%s] %s" % (os.path.basename(path), kind,
                                            head[:64]))
    return hits or None


def state(folder, n, txt, sources=None, figs=None):
    mod = folder.split("_")[0]
    if sources is None:
        sources = {f: open(f, errors="ignore").read()
                   for f in glob.glob(os.path.join(ROOT, "meta", "*", "scripts", "*.py"))}
    if figs is None:
        figs = glob.glob(os.path.join(ROOT, "meta", "*", "figures", "*.png"))

    #: 1. the artifact it names
    named = re.findall(r"`([^`]+\.(?:csv|json|parquet|tsv))`", txt)
    have, all_hits, ambiguous = [], [], []
    for a in named:
        base = os.path.basename(a).split("{")[0]
        hits = [h for h in glob.glob(os.path.join(ROOT, "**", "*%s*" % base),
                                     recursive=True) if ".git" not in h]
        #: EXACT BASENAME FIRST. A `*d_ladder.csv*` glob also returns
        #: `d_ladder_fields.csv`, and on candidate 7 the first hit was a
        #: 763,003-row file against the 267 the entry is actually about --
        #: **the same first-match trap I warned dario about two posts before
        #: writing this line.** Prefer the exact name; keep the rest visible.
        exact = [h for h in hits if os.path.basename(h) == os.path.basename(a)]
        have += (exact or hits)[:1]
        all_hits += hits
        #: PROVE THE CHOICE IMMATERIAL RATHER THAN JUSTIFY IT (dario, [6207],
        #: on `attn_norm_sweep{,_full}.json`). Picking the exact basename is
        #: still a JUDGMENT about which file the entry meant. Where several
        #: match, measure whether they agree: if they do, the choice cannot
        #: have mattered and no justification is owed; if they do not, that
        #: is a finding and it belongs on the panel, not in a shrug.
        #: **Cheaper than arguing, and it converts my first-match trap from a
        #: judgment into an assert.**
        #: **AND IT IS BLIND WHERE IT CANNOT MEASURE.** `population()` returns
        #: None for a plain `.json`, so for most JSON artifacts this proves
        #: nothing and stays silent. **Silence here is UNMEASURED, not
        #: agreement** -- dario's own [6199] rule, and the failure mode would
        #: be to read a quiet check as a clean one.
        if len(hits) > 1:
            pops = {os.path.basename(h): population(h) for h in hits}
            seen = {v for v in pops.values() if v is not None}
            if len(seen) > 1:
                ambiguous.append("%s -> %s" % (os.path.basename(a), pops))

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
    #: 5. a count that is a SELECTION, not a population. See selection().
    sel = selection(txt, have[0]) if have else None

    return {"artifact": bool(have) if named else None, "named": named,
            "numbers": nums, "drawn_by": drawn_by, "marked": marked,
            "tracked": all(_tracked(h) for h in have) if have else None,
            "columns": cols, "selection": sel,
            "ambiguous_file": ambiguous or None}


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
    n_open = n_sel = 0
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
            if st["selection"]:
                s = st["selection"]
                n_sel += 1
                lab += "  [SELECTS %s of %d -- read the doc]" % (
                    "/".join(str(x) for x in s["entry_counts"][:2]), s["file_rows"])
        if only_open and "plausibly" not in lab:
            continue
        print("  %-22s %-3s %-40s %s" % (f, n, t[:40], lab))
    print("\n  %d entries, %d plausibly open, %d quoting a SELECTION."
          % (len(rows), n_open, n_sel))
    print("  A count in an entry is a selection rule compressed to an integer,")
    print("  and the rule is in the FINDING, not the entry (dario, [6197]).")
    print("  PLAUSIBLY OPEN IS NOT OPEN. Read what the entry specifies and look")
    print("  for a producer asserting it before promoting -- inputs-present is")
    print("  what made queue 16 wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
