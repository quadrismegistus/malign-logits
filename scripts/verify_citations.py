#!/usr/bin/env python3
"""Every commit SHA cited in findings/ and docs/ must resolve in HEAD's history.

WHY THIS EXISTS. On 2026-07-27, F39 was found to cite `5e0e5a3` for the claim
"script committed before running" -- the single sentence whose entire function
is to let a reader verify that a pre-registered script was not tuned after
results were seen. NO SUCH COMMIT EVER EXISTED. It was written from memory
after a chained `git commit && ...` that never printed a SHA, repeated to two
audit seats, and it survived six hours inside a finding graded *verified* and
an audit that passed. Nobody resolved it, including its author, during an hour
spent arguing that published figures must be recomputable from published
artifacts.

A COMMIT REFERENCE IS SUCH A FIGURE. That was the blind spot: the project had
a standing check for numbers and did not apply it to identifiers, which are the
load-bearing evidence for every pre-registration claim it makes.

The same day's repository rewrites (stripping oversized data blobs) orphaned six
further citations that WERE real -- a second, milder failure mode with the same
symptom. Both are caught here.

THREE OUTCOMES, and the middle one is the reason this is not just `git show`:

    in history   the SHA is an ancestor of HEAD -- verifiable by a reader
    ORPHANED     the object exists locally but is unreachable from HEAD. It
                 will vanish at the next `git gc` and was very likely never
                 pushed, so a reader cloning the repository cannot resolve it.
                 `git cat-file -e` PASSES on these, which is why that is the
                 wrong check.
    FABRICATED   no such object, anywhere

Exit status is non-zero if anything is not `in history`, so this can gate a
commit. Run after any history rewrite, and before publishing.
"""
import pathlib
import re
import subprocess
import sys

SHA_RE = re.compile(r"`([0-9a-f]{7,40})`")
# Hex-looking tokens that are not commits. Extend rather than loosening the regex.
NOT_A_SHA = {"deadbeef", "cafebabe"}

# SHAs that provably never existed and are cited ONLY inside the text that
# documents their non-existence. Each needs a reason, and the reason is checked
# by a human, not by this script. An entry here is an admission, not a waiver:
# it says "this identifier is discussed, not relied upon". Never add a SHA here
# to silence a failure -- a citation that a finding actually RESTS on must be
# remapped to a real commit instead.
DOCUMENTED_NONEXISTENT = {
    "5e0e5a3": "fabricated in F39; cited only in its own correction notice, "
               "which states it never existed. Real commit: 2b58732.",
}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def classify(sha):
    if git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return "FABRICATED", ""
    subject = git("log", "-1", "--format=%s", sha).stdout.strip()
    if git("merge-base", "--is-ancestor", sha, "HEAD").returncode == 0:
        return "in history", subject
    return "ORPHANED", subject


def main():
    cites = {}
    for d in ("findings", "docs"):
        for f in sorted(pathlib.Path(d).glob("*.md")):
            for m in SHA_RE.finditer(f.read_text()):
                s = m.group(1)
                if s not in NOT_A_SHA and s not in DOCUMENTED_NONEXISTENT:
                    cites.setdefault(s, set()).add(f"{d}/{f.name}")

    bad = []
    for sha, why in sorted(DOCUMENTED_NONEXISTENT.items()):
        print(f"{sha:10s}{'documented':13s}{why[:60]}")
    print(f"{'sha':10s}{'status':13s}subject / cited in")
    for sha in sorted(cites):
        status, subject = classify(sha)
        if status != "in history":
            bad.append((sha, status, sorted(cites[sha])))
        # Non-commit hex (hashes, IDs) shows up as FABRICATED; report the file so
        # a false positive is diagnosable rather than merely alarming.
        note = subject[:46] if subject else ", ".join(sorted(cites[sha]))
        print(f"{sha:10s}{status:13s}{note}")

    print(f"\n{len(cites) - len(bad)} of {len(cites)} citations resolve in HEAD's history")
    if bad:
        print("\nNOT VERIFIABLE BY A READER:")
        for sha, status, files in bad:
            print(f"   {sha}  {status:11s} {', '.join(files)}")
        print("\nOrphaned SHAs survive locally until `git gc` and were probably never"
              "\npushed. Remap them to the equivalent commit in current history"
              "\n(match on commit subject) rather than deleting the citation.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
