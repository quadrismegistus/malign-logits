"""freeze_gates.py — THE FOUR FREEZE GATES, RUNNABLE, FOR ANY REGISTRATION.

**WHY IT EXISTS.** Q's freeze ran the four gates by hand and gate 1 passed on
a question it does not ask. It reported *"working tree clean"* — true, and
true the way an empty repository is clean: **zero modifications to tracked
files, and 38 paths never added, among them P's producer and P's result.**

    **A GATE NAMED CUSTODY THAT MEASURES MODIFICATION CANNOT SEE THE FILE
    WITH NO RESTORE PATH.** A file that has never been `git add`ed is
    invisible to a modification check and is exactly the file at risk.
    `chmod 444` and an escrow give a COPY; git gives a RESTORE PATH.

Gate 3 has its own history and it is the opposite failure. Its first form
matched any string like "not frozen" and fired on **two false positives** —
a QUOTATION of the superseded status line (framed in the past tense) and an
aphorism about reading rules. **A gate that matches a string it has not
scoped costs the round it fires in**, so gate 3 here excludes quoted material
and non-status subjects, and says what it excluded.

THE FOUR GATES:

  1 **CUSTODY**   the registration is tracked; NO modifications to tracked
                  files under its tree; **AND NO UNTRACKED PATHS under it.**
  2 **HASH**      live file == escrow CONTENT == escrow FILENAME ==
                  COMMITTED BLOB. Four-way, not two-way.
  3 **STATUS**    the header is past-anchored and FROZEN, with no LIVE
                  present-tense claim that the document is unfrozen.
  4 **LOCK**      live and escrow are mode 444.

**WHAT IT CANNOT DO.** Every gate here is about the RECORD — that the bytes
are pinned, restorable, locked and honestly dated. **None of them reads a
hypothesis.** A registration can pass all four and test the wrong quantity on
the wrong unit; that is what the parse rounds are for.
"""
import hashlib
import io
import os
import re
import subprocess
import sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True).stdout.strip()


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True, cwd=ROOT).stdout


def h16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def gate1_custody(reg_rel, tree_rel):
    print("=== GATE 1  CUSTODY")
    ok = True
    tracked = sh("git", "ls-files", "--error-unmatch", reg_rel).strip()
    print("  registration tracked in git        %s" % ("PASS" if tracked else "**FAIL**"))
    ok &= bool(tracked)

    status = [l for l in sh("git", "status", "--porcelain", tree_rel).split("\n") if l.strip()]
    mods = [l for l in status if not l.startswith("??")]
    untracked = [l for l in status if l.startswith("??")]

    print("  no modifications under %-11s %s" % (tree_rel, "PASS" if not mods else "**FAIL** (%d)" % len(mods)))
    for l in mods[:5]:
        print("      %s" % l)
    ok &= not mods

    #: THE CHECK GATE 1 DID NOT HAVE. An untracked file is not a modification
    #: and is the only file with no restore path.
    print("  **no UNTRACKED paths under it**     %s"
          % ("PASS" if not untracked else "**FAIL** (%d)" % len(untracked)))
    for l in untracked[:8]:
        print("      %s" % l)
    if len(untracked) > 8:
        print("      ... and %d more" % (len(untracked) - 8))
    ok &= not untracked
    return ok


def gate2_hash(reg_rel, escrow_rel, escrow_pat):
    print("=== GATE 2  HASH, four-way")
    live = h16(os.path.join(ROOT, reg_rel))
    esc = h16(os.path.join(ROOT, escrow_rel))
    claim = re.search(escrow_pat, os.path.basename(escrow_rel)).group(1)
    blob = hashlib.sha256(
        subprocess.run(["git", "show", "HEAD:%s" % reg_rel],
                       capture_output=True, cwd=ROOT).stdout).hexdigest()[:16]
    for label, v in (("live file", live), ("escrow CONTENT", esc),
                     ("escrow FILENAME", claim), ("committed blob", blob)):
        print("  %-18s %s   %s" % (label, v, "PASS" if v == live else "**FAIL**"))
    return esc == live == claim == blob


def gate3_status(reg_rel):
    print("=== GATE 3  STATUS, past-anchored")
    lines = io.open(os.path.join(ROOT, reg_rel), encoding="utf-8").read().split("\n")
    hdr = next((l for l in lines[:8] if l.startswith("**STATUS")), "")
    frozen = hdr.startswith("**STATUS, FROZEN")
    print("  header line reads FROZEN            %s" % ("PASS" if frozen else "**FAIL**"))

    #: scoped: quoted material is testimony ABOUT a superseded line, and an
    #: aphorism whose subject is a rule is not a claim about this document.
    PAT = re.compile(r"(draft,?\s+not frozen|not yet frozen|is not frozen|not in force)", re.I)
    QUOTED = re.compile(r'\*"[^"]*"\*')
    live = []
    for i, l in enumerate(lines, 1):
        s = QUOTED.sub("", l)
        for m in PAT.finditer(s):
            before = s[max(0, m.start() - 70):m.start()]
            if re.search(r"\b(a|an|the)\s+\w*\s*rule\b", before, re.I):
                continue
            live.append((i, s[max(0, m.start() - 50):m.end() + 30].strip()))
    for i, frag in live:
        print("      L%d  %s" % (i, frag))
    print("  no LIVE present-tense unfrozen claim %s"
          % ("PASS" if not live else "**FAIL** (%d)" % len(live)))
    return frozen and not live


def gate4_lock(reg_rel, escrow_rel):
    print("=== GATE 4  LOCK")
    ok = True
    for p in (reg_rel, escrow_rel):
        mode = oct(os.stat(os.path.join(ROOT, p)).st_mode)[-3:]
        good = mode == "444"
        print("  %-58s %s  %s" % (os.path.basename(p), mode, "PASS" if good else "**FAIL**"))
        ok &= good
    return ok


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: freeze_gates.py <registration-relpath> <escrow-relpath>")
    reg_rel, escrow_rel = sys.argv[1], sys.argv[2]
    tree_rel = os.path.dirname(os.path.dirname(reg_rel))
    pat = r"[.-]([0-9a-f]{16})\.md$"

    results = [gate1_custody(reg_rel, tree_rel),
               gate2_hash(reg_rel, escrow_rel, pat),
               gate3_status(reg_rel),
               gate4_lock(reg_rel, escrow_rel)]
    print("\n" + "=" * 62)
    if all(results):
        print("ALL FOUR GATES PASS.")
        print("**The record is pinned, restorable, locked and honestly dated.**")
        print("**No gate here read a hypothesis. That is what the parse is for.**")
        return 0
    print("**%d OF 4 GATES FAILED**" % results.count(False))
    return 1


if __name__ == "__main__":
    sys.exit(main())
