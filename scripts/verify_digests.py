#!/usr/bin/env python3
"""Classes 2 and 3 of citation: PRODUCER-FILE digests and SOURCE-DATA digests.

    uv run .venv/bin/python scripts/verify_digests.py

`verify_citations.py` resolves COMMIT SHAs against git history and cannot see
either of these. On 2026-07-31 a producer hash posted to the docket and pinned in
a memo (`b454c98c`) disagreed with the file on disk (`390ea9c0`), and the only
reason it was caught is that the custodian happened to check before committing.
Had custody been a formality the wrong file would have entered the repo under the
right name, and every later reader would have had the wrong instrument for the
right number.

A CITATION VERIFIER CERTIFIES THE CLASS IT WAS BUILT FOR AND NO OTHER. Three
classes, three universes to resolve against:

    class 1  commit shas       git history                    verify_citations.py
    class 2  producer digests  instrument dirs + scripts/     here
    class 3  source digests    declared project-owned dirs    here

TWO RULES THE SAME DAY'S INCIDENTS PAID FOR:

  OWNERSHIP IS JUDGED ON realpath, NEVER ON THE LITERAL STRING.
  `~/github/abslithists/abstraction/data/fields/sources/` reads as a
  home-directory project path and lands on `/Volumes/chambers` — removable media.
  A check on the written path passes it; a check on what it resolves to does not.

  ONE-OF-FIVE AND ONE-OF-ONE ARE DIFFERENT FACTS, AND ONLY THE SECOND IS A PIN.
  Brysbaert's digest resolved in five places across three volumes, all identical
  bytes. "0b4082db resolves" was true and concealed the situation, so the report
  prints WHERE a digest resolved and HOW MANY candidates carried it.
"""
import argparse
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

#: PROJECT-OWNED. A pinned source outside these is a dependency wearing a
#: citation's clothes: the pin survives a move (that is what content-addressing
#: buys) but every path naming it breaks SILENTLY AT READ TIME.
OWNED = [
    REPO / "data",
    REPO / "scripts",
    pathlib.Path(os.path.expanduser(
        "~/Dropbox/Prof/Articles/TheoryMachines/norms_sources")),
]
#: WHERE PRODUCERS LIVE. The seat directories are not project-owned and are not
#: meant to be -- a producer is authored there and COMMITTED here, so a digest
#: resolving only at a seat is a custody debt, reported as such.
PRODUCERS = [
    REPO / "scripts",
    pathlib.Path(os.path.expanduser(
        "~/Dropbox/Prof/Articles/TheoryMachines/agents/lacan/instruments")),
]
DIGEST_RE = re.compile(r"`?\b([0-9a-f]{16,64})\b`?")
MIN_BYTES = 512      # below this a "digest" is more likely an id than a file


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def index(dirs, exts=None):
    """digest -> [paths]. MULTIPLE PATHS PER DIGEST IS THE NORMAL CASE and the
    reason this is a list: copies, backups, and symlinks into other volumes all
    carry identical bytes, and collapsing them to one would hide the spread.

    `recurse_symlinks=True` IS LOAD-BEARING AND THE DEFAULT IS A TRAP. `rglob`
    does not traverse symlinked directories unless told to, so the first version
    of this file reported the Brysbaert digest as `resolves 1` while a SECOND
    copy sat at ~/github/abslithists/abstraction/data/... -- a symlink into
    /Volumes/chambers, and the copy the norms producer actually pins.

    THE FALSE "1" WAS WORSE THAN A MISS, because the whole point of the count is
    to distinguish one-of-one from one-of-five, and a non-following scan reports
    the reassuring number. A SCAN THAT DOES NOT FOLLOW SYMLINKS ANSWERS ABOUT
    THE TREE, NOT ABOUT THE FILESYSTEM."""
    out = {}
    seen_real = set()
    for d in dirs:
        if not d.exists():
            continue
        # os.walk(followlinks=True), NOT rglob(recurse_symlinks=True): the latter
        # needs Python 3.13 and this repo's venv is 3.11, so the "fix" raised
        # TypeError at import on the interpreter that actually runs it. A GUARD
        # THAT NEEDS A NEWER INTERPRETER THAN THE ONE IT RUNS UNDER IS NOT A GUARD.
        for root, _dirs, names in os.walk(d, followlinks=True):
            for nm in names:
                p = pathlib.Path(root) / nm
                if not p.is_file():
                    continue
                if exts and p.suffix not in exts:
                    continue
                try:
                    if p.stat().st_size < MIN_BYTES:
                        continue
                    # one entry per DISTINCT FILE; a symlinked directory
                    # otherwise reports the same inode twice and inflates the
                    # candidate count the whole report turns on
                    rp = os.path.realpath(p)
                    if rp in seen_real:
                        continue
                    seen_real.add(rp)
                    out.setdefault(sha256(p), []).append(p)
                except OSError:
                    continue
    return out


def owned(p):
    """realpath, never the literal string -- see the module docstring."""
    rp = pathlib.Path(os.path.realpath(p))
    return any(str(rp).startswith(str(pathlib.Path(os.path.realpath(o))))
               for o in OWNED)


def main(a):
    prod = index(PRODUCERS, exts={".py", ".json"})
    src = index([pathlib.Path(os.path.realpath(o)) for o in OWNED] + a.extra)
    print(f"producer index {len(prod)} digests   source index {len(src)} digests")

    cites = {}
    for d in ("findings", "docs", "meta"):
        for f in sorted((REPO / d).rglob("*.md")):
            for m in DIGEST_RE.finditer(f.read_text(errors="replace")):
                cites.setdefault(m.group(1), set()).add(f"{d}/{f.name}")
    print(f"digest-shaped tokens cited: {len(cites)}\n")

    bad, rows, unresolved = [], [], []
    for tok in sorted(cites):
        hits = ([(p, "producer") for k, ps in prod.items() if k.startswith(tok) for p in ps]
                + [(p, "source") for k, ps in src.items() if k.startswith(tok) for p in ps])
        if not hits:
            # A COMMIT SHA IS ALSO HEX, so "no file carries this" is not yet a
            # failure -- class 1 may own it. But IT IS A FAILURE IF NOBODY DOES.
            #
            # THE FIRST VERSION OF THIS FILE SKIPPED THESE ENTIRELY and therefore
            # passed a planted wrong producer hash: a fabricated digest resolves
            # to no file, looked like a commit SHA to this checker and like a file
            # digest to verify_citations.py, and fell between them. A CITATION
            # THAT TWO CHECKERS EACH BELIEVE THE OTHER OWNS IS CHECKED BY NEITHER.
            if subprocess.run(["git", "cat-file", "-e", f"{tok}^{{commit}}"],
                              capture_output=True, cwd=REPO).returncode == 0:
                rows.append((tok, "commit", "", 0, sorted(cites[tok])))
            else:
                rows.append((tok, "UNRESOLVED", "", 0, sorted(cites[tok])))
                unresolved.append((tok, sorted(cites[tok])))
            continue
        own = [p for p, _ in hits if owned(p)]
        status = "resolves" if own else "UNOWNED"
        if not own:
            bad.append((tok, sorted(cites[tok]), hits[0][0]))
        rows.append((tok, status, own[0] if own else hits[0][0], len(hits),
                     sorted(cites[tok])))

    print(f"{'digest':20s}{'status':12s}{'n':>3s}  where")
    for tok, status, where, n, files in rows:
        if status == "commit":
            continue
        w = str(where)
        w = w.replace(str(REPO), ".").replace(os.path.expanduser("~"), "~")
        print(f"{tok[:18]:20s}{status:12s}{n:>3d}  {w[:64]}")

    fp = hashlib.sha256(
        ("\n".join(sorted(prod)) + "\x00" + "\n".join(sorted(src))).encode()).hexdigest()
    print(f"\nfingerprint {fp[:32]}  (producer index + source index, sorted, "
          "NUL-separated, sha256)")
    for tok, files in unresolved:
        print(f"{tok[:18]:20s}{'UNRESOLVED':12s}{0:>3d}  neither a file nor a commit "
              f"-- {', '.join(files)}")
    n_file = sum(1 for r in rows if r[1] != "commit")
    print(f"{n_file - len(bad)} of {n_file} file digests resolve inside a "
          f"project-owned directory")
    if unresolved:
        print("\nUNRESOLVED -- no file carries this digest and no commit has this sha.")
        print("A CITATION THAT TWO CHECKERS EACH BELIEVE THE OTHER OWNS IS CHECKED BY")
        print("NEITHER; this is the gap between class 1 and class 2.")
        return 1
    if bad:
        print("\nUNOWNED -- resolves, but not in a directory this project controls:")
        for tok, files, where in bad:
            print(f"   {tok[:18]}  {', '.join(files)}\n      -> {where}")
        print("\nA DIGEST THAT RESOLVES AGAINST A FILE THIS PROJECT DOES NOT OWN IS A"
              "\nPASS TODAY AND A MYSTERY IN A MONTH. Consolidate one canonical copy"
              "\ninto a project-owned directory and re-point every declared path.")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", type=pathlib.Path, default=[],
                    help="additional directories to search for source digests")
    sys.exit(main(ap.parse_args()))
