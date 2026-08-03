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
import os
import re
import subprocess
import sys

# The HEAD this inventory's verifications are pinned to. A verification expires
# under history REWRITING, not under history GROWTH: ancestry is monotonic
# under append, so if X was an ancestor of PIN and PIN is still an ancestor of
# HEAD, X is still an ancestor and the verification stands (lacan).
#
# That distinction is what keeps rule 14 from being a treadmill. HEAD moves
# several times an hour; under the naive reading every inventory is stale on
# arrival, and a rule violated by every commit is one everyone learns to skip.
# So the cheap check is ON THE PIN, not on every citation -- one merge-base
# call re-validates the whole inventory, and it fails loudly exactly when a
# rewrite has happened, which is when re-verification is genuinely owed.
#
# THE OTHER HALF OF THAT SCOPE, and it bites immediately (lacan): a pin
# validates only the citations that EXISTED WHEN IT WAS TAKEN. Growth does not
# invalidate old verifications and it does not extend them either. A citation
# added after the pin is outside its coverage and --pin will pass while saying
# nothing about it. So RE-PIN AFTER ADDING CITATIONS. The full run below always
# checks every citation against current HEAD, so the pin is a fast
# revalidation, never a substitute for it.
PIN = "5dd59a51e4e014bc45e3e58c9bcb2b6376fa87c5"

SHA_RE = re.compile(r"`([0-9a-f]{7,40})`")
# Hex-looking tokens that are not commits. Extend rather than loosening the regex.
NOT_A_SHA = {"deadbeef", "cafebabe"}

# SHAs that provably never existed and are cited ONLY inside the text that
# documents their non-existence. Each needs a reason, and the reason is checked
# by a human, not by this script. An entry here is an admission, not a waiver:
# it says "this identifier is discussed, not relied upon". Never add a SHA here
# to silence a failure -- a citation that a finding actually RESTS on must be
# remapped to a real commit instead.
# Region markers for SHAs that are DISCUSSED, not relied upon: the left column
# of a remap table, a quoted historical reference, a correction notice. lacan's
# point is that this distinction is POSITIONAL and no automated check can make
# it -- a superseded SHA in a mapping row is indistinguishable by grep from a
# live citation. So the author declares it and the tool enforces the
# declaration, which is the only division of labour that works: the tool cannot
# infer intent, and an author who must remember to run a check will not.
#
#   <!-- citation-check: historical -->   ... SHAs here are not verified ...
#   <!-- citation-check: end -->
#
# Everything outside such a region is a live citation and must resolve.
HISTORICAL_OPEN = "<!-- citation-check: historical -->"
HISTORICAL_CLOSE = "<!-- citation-check: end -->"

DOCUMENTED_NONEXISTENT = {
    "5e0e5a3": "fabricated in F39; cited only in its own correction notice, "
               "which states it never existed. Real commit: 2b58732.",
}


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


#: **CONTENT HASHES ARE NOT COMMITS AND THIS CHECKER DOES NOT OWN THEM.**
#: Every backticked 7-40 hex was classified as a claimed commit, so
#: `sha256[:16]` frozen-artifact digests -- C v6's `06f0272d7f21b901`,
#: pairs_d's `84011269d00eea6b`, the population's `3ed3e286e633c2fc` -- were
#: printed FABRICATED, which is this campaign's heaviest word applied to the
#: freeze discipline's own receipts. The registrations only entered the glob
#: when they were committed, so an old assumption met new files.
#:
#: The rule this restores is already booked: A CITATION VERIFIER CERTIFIES THE
#: CLASS IT WAS BUILT FOR AND NO OTHER -- and its converse, that a citation two
#: checkers each believe the other owns is checked by NEITHER. So a non-commit
#: is handed to the digest class by name and only called FABRICATED when
#: neither class resolves it.
DIGEST_LEN = 16          #: sha256[:16], the campaign's frozen-artifact notation
_DIGESTS = {}            #: filled lazily; see digest_index()


def digest_index():
    """sha256[:16] -> paths, over the project's own instrument directories."""
    if _DIGESTS:
        return _DIGESTS
    import hashlib
    roots = ("meta", "scripts", "malign_logits")
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "__pycache__", "node_modules")]
            for fn in filenames:
                if not fn.endswith((".py", ".md", ".json", ".txt")):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, "rb") as fh:
                        h = hashlib.sha256(fh.read()).hexdigest()[:DIGEST_LEN]
                except OSError:
                    continue
                _DIGESTS.setdefault(h, []).append(fp)
                #: **AND DIGESTS DECLARED INSIDE ARTIFACTS, which are a third
                #: thing again.** `3ed3e286e633c2fc` is not the sha of any FILE
                #: -- it is the sha of the population's ID SET, recorded in
                #: `population_d_684.json` as `id_set_sha256_16`. A citation can
                #: name a set whose enumeration lives in a committed artifact,
                #: and only the artifact can resolve it.
                if fn.endswith(".json"):
                    try:
                        import json as _json
                        with open(fp) as fh:
                            obj = _json.load(fh)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if (isinstance(v, str) and len(v) >= DIGEST_LEN
                                    and "sha" in k.lower()):
                                _DIGESTS.setdefault(v[:DIGEST_LEN],
                                                    []).append(f"{fp}:{k}")
    return _DIGESTS


def classify(sha):
    if git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        #: not a commit -- ask the class that DOES own content digests before
        #: reaching for the word FABRICATED
        if len(sha) == DIGEST_LEN:
            hit = digest_index().get(sha)
            if hit:
                return "content-hash", hit[0]
            #: **A CITATION OF SUPERSEDED BYTES IS NOT UNRESOLVED.** Six status
            #: headers were rewritten today, so every hash naming their PRIOR
            #: bytes stopped matching the working tree while remaining perfectly
            #: correct about what it cited. Git holds those bytes; the commit
            #: that last carried them is the answer.
            was = _HIST.get(sha)
            if was:
                return "content-hash", was
        return "UNRESOLVED", ""
    #: **RESTORED. My splice deleted these four lines and every COMMIT citation
    #: then fell off the end of the function returning None** -- the run
    #: crashed at the third row with a TypeError, which is the loud failure and
    #: therefore the lucky one.
    subject = git("log", "-1", "--format=%s", sha).stdout.strip()
    if git("merge-base", "--is-ancestor", sha, "HEAD").returncode == 0:
        return "in history", subject
    return "ORPHANED", subject


_HIST = {}


def history_index():
    """sha256[:16] -> locator, over COMMITTED versions of tracked files.

    **BUILT ONCE, NOT PER-CITATION.** The first version called a per-sha search
    that walked every file x every commit with a subprocess per blob; under a
    timeout it emitted three rows and looked like a clean result. **A search
    whose cost is (citations x files x commits) is not a check, it is a hang
    with partial output** -- and partial output from a killed process is
    indistinguishable, in the terminal, from a short answer.
    """
    if _HIST:
        return _HIST
    import hashlib
    paths = [p for p in git("ls-files", "meta").stdout.split()
             if p.endswith((".md", ".json"))]
    for path in paths:
        for c in git("log", "--format=%H", "--", path).stdout.split()[:8]:
            blob = git("rev-parse", f"{c}:{path}").stdout.strip()
            if not blob:
                continue
            body = subprocess.run(["git", "cat-file", "blob", blob],
                                  capture_output=True).stdout
            h = hashlib.sha256(body).hexdigest()[:DIGEST_LEN]
            _HIST.setdefault(h, f"SUPERSEDED bytes of {path} "
                                f"(last at {git('log', '-1', '--format=%h', c).stdout.strip()})")
    return _HIST


def check_pin():
    """Re-validate the whole inventory in one command. See PIN above."""
    if subprocess.run(["git", "cat-file", "-e", f"{PIN}^{{commit}}"],
                      capture_output=True).returncode != 0:
        print(f"PIN {PIN[:7]} NO LONGER EXISTS -- history was rewritten.")
        return False
    ok = subprocess.run(["git", "merge-base", "--is-ancestor", PIN, "HEAD"],
                        capture_output=True).returncode == 0
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    if ok:
        print(f"pin {PIN[:7]} is an ancestor of HEAD {head} -- "
              f"every verification below still stands (append, not rewrite).")
    else:
        print(f"PIN {PIN[:7]} IS NOT AN ANCESTOR of HEAD {head}. History was "
              f"REWRITTEN; every citation needs re-verification.")
    return ok


def main():
    if "--pin" in sys.argv:
        return 0 if check_pin() else 1
    pin_ok = check_pin()
    print()
    cites = {}
    # `meta` IS GLOBBED RECURSIVELY AND THAT IS THE POINT. It was outside this
    # check twice over -- wrong directory, and nested one level down, so
    # `meta/M01_displacement/README.md` was invisible to a top-level `*.md`.
    # The meta layer is where clauses are assembled into the paper's sentences,
    # each one carrying a docket citation as its evidence, which makes it the
    # LAST place an unresolvable identifier should be able to sit.
    for d in ("findings", "docs", "meta"):
        for f in sorted(pathlib.Path(d).rglob("*.md")):
            text = f.read_text()
            # Blank out declared-historical regions so their SHAs are not
            # treated as claims. Replaced with spaces to preserve offsets.
            out, i = [], 0
            while True:
                a = text.find(HISTORICAL_OPEN, i)
                if a < 0:
                    out.append(text[i:]); break
                b = text.find(HISTORICAL_CLOSE, a)
                if b < 0:
                    out.append(text[i:a]); break
                out.append(text[i:a]); out.append(" " * (b + len(HISTORICAL_CLOSE) - a))
                i = b + len(HISTORICAL_CLOSE)
            text = "".join(out)
            for m in SHA_RE.finditer(text):
                s = m.group(1)
                if s not in NOT_A_SHA and s not in DOCUMENTED_NONEXISTENT:
                    cites.setdefault(s, set()).add(f"{d}/{f.name}")

    bad = []
    for sha, why in sorted(DOCUMENTED_NONEXISTENT.items()):
        print(f"{sha:10s}{'documented':13s}{why[:60]}")
    print(f"{'sha':10s}{'status':13s}subject / cited in")
    digest_index(); _HIST.update(history_index())
    for sha in sorted(cites):
        status, subject = classify(sha)
        if status not in ("in history", "content-hash"):
            bad.append((sha, status, sorted(cites[sha])))
        # A non-commit that resolves as a frozen-artifact digest is REPORTED,
        # not accused: it names the file whose bytes it is.
        note = subject[:46] if subject else ", ".join(sorted(cites[sha]))
        print(f"{sha:10s}{status:13s}{note}")

    print(f"\n{len(cites) - len(bad)} of {len(cites)} citations resolve in HEAD's history")
    if not pin_ok:
        return 1
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
