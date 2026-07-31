#!/usr/bin/env python3
"""Spec conformance checker — built after FIVE false FAILs in one afternoon.

Every one of those was the CHECKER, not the file. The failure modes, each of
which cost a reading and one of which broke an explicit standing rule:

    1. LINE-ORIENTED     grep missed a phrase spanning a newline, and printed a
                         green OK on a file that contained it ([1400])
    2. REGEX METACHARS   pattern `|D_z|` is grep alternation; matched nothing
                         and reported the file missing its own ordering ([1501])
    3. CASE              pattern `per stratum` vs file `PER STRATUM` ([1553]) —
                         against [1403].2, which rules fragments case-insensitive
    4. HYPHEN            pattern `positive control` vs file `POSITIVE-CONTROL`
    5. SECTION BOUNDARY  a `---` separator appended before a new section was
                         attributed to the PRECEDING section as a change ([1507])

So: flatten whitespace, casefold, fold hyphens to spaces, treat every pattern as
a LITERAL, and compare sections by CONTENT rather than by boundary.

    python3 spec_check.py FILE --require "phrase" ... [--absent "phrase" ...]
    python3 spec_check.py OLD NEW --sections        # which sections really moved
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path


def norm(s: str) -> str:
    """The one normalisation every check shares. Order matters: NFKC before
    casefold, hyphen-folding before whitespace collapse."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("‑", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[-_]+", " ", s)          # hyphen/underscore == space
    s = re.sub(r"\s+", " ", s)            # newlines vanish; phrases survive
    return s.casefold().strip()


def sections(path: Path) -> dict[str, str]:
    """Split on markdown headings, keyed by the heading's normalised text.

    The VALUE excludes trailing separators, so appending a section after §N
    does not read as a change to §N — failure mode 5.
    """
    text = path.read_text()
    parts = re.split(r"^(#{1,3} .*)$", text, flags=re.M)
    out, cur = {}, None
    for part in parts:
        if re.match(r"^#{1,3} ", part):
            cur = norm(part)
        elif cur is not None:
            out[cur] = out.get(cur, "") + part
    return {k: re.sub(r"\s*-{3,}\s*$", "", v).strip() for k, v in out.items()}


def check(path: Path, require: list[str], absent: list[str]) -> int:
    hay = norm(path.read_text())
    fails = 0
    for pat in require:
        if norm(pat) in hay:
            print(f"  ok    present: {pat}")
        else:
            print(f"  FAIL  MISSING: {pat}")
            fails += 1
    for pat in absent:
        if norm(pat) in hay:
            print(f"  FAIL  SHOULD BE ABSENT: {pat}")
            fails += 1
        else:
            print(f"  ok    absent:  {pat}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--require", action="append", default=[])
    ap.add_argument("--absent", action="append", default=[])
    ap.add_argument("--sections", action="store_true",
                    help="two files: report which sections differ by CONTENT")
    a = ap.parse_args()

    if a.sections:
        if len(a.files) != 2:
            sys.exit("--sections needs exactly two files")
        old, new = sections(Path(a.files[0])), sections(Path(a.files[1]))
        moved = [k for k in sorted(set(old) | set(new))
                 if old.get(k, "") != new.get(k, "")]
        for k in moved:
            tag = "NEW" if k not in old else "GONE" if k not in new else "CHANGED"
            print(f"  {tag:>7}  {k}")
        if not moved:
            print("  no section differs by content")
        return 0

    fails = check(Path(a.files[0]), a.require, a.absent)
    print(f"\n  {'PASS' if not fails else f'FAIL ({fails})'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
