#!/usr/bin/env python3
"""Which booked claims cite evidence that was rewritten after they were booked?

Not a base rate of false claims -- nothing cheap gives that. This is the
POPULATION AT RISK, for scoping an expensive reproduce-the-numbers check.

Origin: on 2026-07-26 a claim in CLAUDE.md was found to cite numbers that
matched nothing in `data/battery_results.csv`. The claim was correct; the file
had been silently narrowed from 9 families to 1 by `malign battery --family
zephyr`. Evidence destroyed, briefly misread as claim unsupported.

An EXISTENCE check cannot find this: the file existed throughout. The screen
here asks the question that can -- was the cited file modified after the claim
citing it was booked -- and it does catch that instance.

NO THRESHOLD, DELIBERATELY. An earlier version emitted a binary flag, which
over-flagged: F25 cites eight files each with one commit landing the day after
the doc, which is a booking sequence, not evidence rewritten under a claim. The
tempting fix is "ignore commits within N days", which just relocates the
judgement into N. Instead this reports, per citation, the NUMBER of commits
postdating the claim and their DATE SPAN, and lets the reader see the
difference: 1 commit spanning 1 day is visibly not 6 commits spanning months.

KNOWN COVERAGE LIMIT. This project books claims in at least four places:
findings/*.md and CLAUDE.md (in-repo, covered here), and
notes/claims-ledger-draft.md and paper/theory-machines-v3.qmd (in Dropbox,
NOT under version control -- verified 2026-07-26). The manuscript is the
TERMINAL booking site, where a number stops being internal and becomes
published. A clean run here is therefore not coverage: the screen's reach stops
at the repository boundary, and the site whose errors cost most is outside it.
"""
import argparse, glob, os, re, subprocess, sys

DATA_RE = re.compile(r"`?(data/[\w./-]+\.(?:csv|parquet|json|jsonl))`?")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def commits_after(path, date):
    """Dates of commits to `path` strictly after `date`."""
    out = git("log", f"--since={date}", "--format=%ad", "--date=short", "--", path)
    return sorted(d for d in out.split() if d > date)


def booked_date_file(path):
    """First commit of a whole file -- right for findings/, which are added once."""
    d = [x for x in git("log", "--diff-filter=A", "--format=%ad", "--date=short",
                        "--", path).split() if x]
    return d[-1] if d else None


def booked_date_claim(path, needle):
    """First commit introducing `needle` -- right for CLAUDE.md, edited continuously."""
    d = [x for x in git("log", "--format=%ad", "--date=short", "-S", needle,
                        "--", path).split() if x]
    return d[-1] if d else None


def report(rows):
    print(f"{'claim site':38s}{'booked':>11s}  {'cited file':42s}{'after':>6s}{'span':>26s}")
    for site, booked, f, after in sorted(rows, key=lambda r: (-len(r[3]), r[1])):
        if not after:
            span = "-"
        elif len(after) == 1:
            span = f"{after[0]}"
        else:
            span = f"{after[0]} .. {after[-1]}"
        print(f"{site[:37]:38s}{booked:>11s}  {f[:41]:42s}{len(after):>6d}{span:>26s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", nargs="*", default=[],
                    help="extra 'FILE::claim text' pairs to date via git log -S")
    a = ap.parse_args()

    rows, missing = [], []
    for f in sorted(glob.glob("findings/F*.md")):
        booked = booked_date_file(f)
        if not booked:
            continue
        for c in sorted(set(DATA_RE.findall(open(f).read()))):
            if not os.path.exists(c):
                missing.append((os.path.basename(f), booked, c))
                continue
            rows.append((os.path.basename(f), booked, c, commits_after(c, booked)))

    # CLAUDE.md claims must be dated by claim text, not by the file's first commit.
    for spec in a.claims:
        site, _, needle = spec.partition("::")
        booked = booked_date_claim(site, needle)
        if not booked:
            print(f"  (could not date claim in {site}: {needle[:40]!r})", file=sys.stderr)
            continue
        for c in sorted(set(DATA_RE.findall(open(site).read()))):
            if os.path.exists(c):
                rows.append((f"{site}::{needle[:18]}", booked, c, commits_after(c, booked)))

    report(rows)
    touched = [r for r in rows if r[3]]
    print(f"\n{len(touched)} of {len(rows)} citations have post-booking commits; "
          f"{len(rows) - len(touched)} clean")
    if missing:
        print(f"\n{len(missing)} cited files are GONE (unresolvable without git archaeology):")
        for s, b, c in missing:
            print(f"   {s:38s} booked {b}  {c}")
    print("\nCOVERAGE: findings/ and any --claims sites, both in-repo. "
          "notes/claims-ledger-draft.md and paper/*.qmd are in Dropbox and NOT "
          "under git; the manuscript is the terminal booking site and is outside "
          "this mechanism's reach.")


if __name__ == "__main__":
    main()
