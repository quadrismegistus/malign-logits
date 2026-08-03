"""FREEZE GATE — the checklist as a mechanism instead of a memory. [3502].1.

**WHY THIS EXISTS.** A census on 2026-08-03 found 5 of 47 M01 artifacts write
protected, and three signed registrations untracked. Neither gap was a decision:
locking and tracking were steps someone had to REMEMBER, and the artifacts that
got them are the ones that were freshest in mind that day. **Every gate that day
passed on the unlocked files, because every gate asked about bytes already in
front of it and none asked whether they existed anywhere else.**

    ORDERING RULE, and it is the whole design:
    ASK "IS IT TRACKED" BEFORE "ARE THE BYTES STABLE."
    Byte-stability is a question you can only ask of something you already have.
    Tracking is the question of whether you will still have it.

**WHAT THIS GATE CANNOT DO, ANNOUNCED BY THE GATE ITSELF.** The specification
parse is a READING task — walk every operative phrase asking "what would I type?"
— and no mechanical check performs it. This gate deliberately carries **no
attestation field** for it: a boolean somebody sets is exactly the unfalsifiable
artifact the parse exists to prevent. Instead the gate PRINTS its own blind spot
on every run, pass or fail, so a green result can never be read as "the parse
ran."

    python3 freeze_gate.py --selftest       prove every check FIRES
    python3 freeze_gate.py <path> [...]     gate one or more artifacts
"""
import os
import subprocess
import sys

#: The present-tense form that expires at the freeze. See Amendment A §A1: it is
#: true when written, true at every signature round, and false from the instant
#: of the freeze -- which no sweep is triggered by.
EXPIRING_STATUS = "nothing is in force"
TIME_INVARIANT = "status at drafting"


def custody_state(path):
    """COMMITTED / STAGED / UNTRACKED for the file's CURRENT bytes.

    **STAGED IS NOT TRACKED.** A `git add` writes the blob into the object
    database and it is recoverable from there -- but an UNREFERENCED blob is
    pruned by `git gc` after its grace period (two weeks by default), so a
    staged-only freeze has a restore path WITH AN EXPIRY DATE.

    The first version asked `git ls-files`, which returns success for a
    staged-only file, and would have certified a staged-only registration as
    tracked. **The check tested a weaker property than the rule states** -- the
    same shape as the status row that vanished on a file extension.

    The question is about THESE BYTES, not this path: a file committed at an
    older revision and since edited is not committed.
    """
    #: **REALPATH, NOT ABSPATH.** `git rev-parse --show-toplevel` returns a
    #: resolved path, so on any system where a component of the path is a
    #: symlink -- macOS `/tmp`, and this campaign's own reorganised directories
    #: -- `relpath(abspath, toplevel)` produces `../../..`-style garbage, the
    #: HEAD lookup fails, and a COMMITTED file is reported STAGED. A scan that
    #: follows links and one that does not are wrong in opposite directions;
    #: resolving both sides is right about either.
    ap = os.path.realpath(path)
    d = os.path.dirname(ap) or "."

    def git(*a):
        return subprocess.run(("git",) + a, capture_output=True, text=True, cwd=d)

    if git("ls-files", "--error-unmatch", ap).returncode != 0:
        return "UNTRACKED"
    here = git("hash-object", ap).stdout.strip()
    top = git("rev-parse", "--show-toplevel").stdout.strip()
    if not (here and top):
        return "UNTRACKED"
    rel = os.path.relpath(ap, top)
    in_head = git("rev-parse", "HEAD:%s" % rel)
    if in_head.returncode == 0 and in_head.stdout.strip() == here:
        return "COMMITTED"
    #: **STAGED AND MODIFIED ARE DIFFERENT STATES AND THE MESSAGE DIFFERS.**
    #: The first version returned STAGED for both, asserting "the blob is
    #: unreferenced and gc prunes it" -- which is FALSE for a worktree edit that
    #: was never added, whose bytes are in no object store at all. Both fail
    #: custody; only one of them has a restore path to expire.
    idx = git("ls-files", "-s", "--", ap).stdout.split()
    if len(idx) >= 2 and idx[1] == here:
        return "STAGED"
    return "MODIFIED"


def is_tracked(path):
    """Retained for callers that only need the boolean. CUSTODY.

    **THE PATH HANDED TO GIT MUST BE ABSOLUTE.** The first version passed the
    caller's RELATIVE path while running git from the file's own directory, so
    git resolved it against the wrong root and returned "untracked" for three
    registrations that were tracked. A false negative under the wrong key --
    and the selftest passed anyway, because its only case asserted that an
    untracked file FIRES. See `selftest`: the case that catches this is the one
    asserting a TRACKED file does NOT fire.
    """
    ap = os.path.abspath(path)
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", ap],
                           capture_output=True, text=True,
                           cwd=os.path.dirname(ap) or ".")
        return r.returncode == 0
    except Exception:
        return False


def is_locked(path):
    """No write bit for anyone. Converts an unconscious edit into a decision."""
    return not (os.stat(path).st_mode & 0o222)


def status_line_is_time_invariant(text):
    """A status line must be written so nothing that happens later falsifies it.

    Returns (ok, why). **THE TEST IS THE PROPERTY, NOT A PHRASE.**

    The first version matched two literal strings lifted from one document's
    wording, and on the campaign corpus it was wrong in BOTH directions:

        registration_d_pairs_v6.md   "## STATUS AS OF 2026-08-01" -- already the
                                     dated form, reported as having NO status
        registration_g_magnitude.md  "STATUS: DRAFT FOR FREEZE" -- the expiring
                                     form, reported with the WRONG REASON because
                                     it lacks the literal 'nothing is in force'

    **The property is whether a later event can falsify the line.** Present-tense
    state ("DRAFT", "nothing is in force") expires at the freeze. A line anchored
    to a DATE is a claim about a moment that has already passed and cannot decay,
    however it is worded.
    """
    import re
    #: **A LINE CONTAINING THE WORD "STATUS" IS NOT A STATUS DECLARATION.**
    #: The predicate was moved from a phrase to the property, and the LINE
    #: SELECTOR stayed a label match -- so the property test ran on the wrong
    #: lines. It classified `registration_d_pairs_v5.md` as carrying an undated
    #: status line on the strength of "...its PRIMARY STATUS is preserved", a
    #: sentence about an experimental arm. That false positive reached a signed
    #: amendment and an authorization before anyone read the line it matched.
    #:
    #: A declaration OPENS its line, allowing markdown heading/emphasis marks.
    head = text.split("\n")[:40]
    lines = [l for l in head if re.match(r"^[\s#*_>|-]*status\b", l, re.I)]
    if not lines:
        return False, ("NO STATUS LINE in the first 40 lines -- the document "
                       "makes no claim about its own state, so nothing here "
                       "can be checked and nothing later can be contradicted")

    dated = [l for l in lines if re.search(r"\d{4}-\d{2}-\d{2}", l)]
    if dated:
        return True, "date-anchored: %r" % dated[0].strip()[:60]

    return False, ("status line is UNDATED and present-tense, so the freeze "
                   "falsifies it: %r" % lines[0].strip()[:60])


def gate(path, sha_of=None):
    """Run every MECHANICAL condition. Returns (ok, rows) -- rows always print.

    `sha_of` is injected only by the selftest; production always hashes the
    real file.
    """
    import hashlib
    rows, ok = [], True

    def rec(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        rows.append(("PASS" if passed else "FAIL", name, detail))

    if not os.path.exists(path):
        return False, [("FAIL", "exists", "no file at %r" % path)]

    #: ORDER IS LOAD-BEARING. Custody first.
    state = custody_state(path)
    rec("custody", state == "COMMITTED", {
        "COMMITTED": "these bytes are in HEAD",
        "STAGED": "STAGED ONLY -- the blob is unreferenced and `git gc` prunes it "
                  "after its grace period. A restore path with an expiry date "
                  "is not custody.",
        "MODIFIED": "MODIFIED IN THE WORKTREE, NEVER ADDED -- these bytes are in "
                    "NO object store. There is nothing to expire because there "
                    "is nothing to restore from.",
        "UNTRACKED": "UNTRACKED -- one filesystem location is not frozen, it is "
                     "merely unmodified",
    }[state])

    body = open(path, "rb").read()
    #: The OBSERVED hash, printed as what was found. A gate that echoes its
    #: expectation testifies to its configuration, not its work. [3502].2(b)
    observed = (sha_of if sha_of is not None
                else hashlib.sha256(body).hexdigest()[:16])
    rec("hash", True, "OBSERVED %s (%d bytes)" % (observed, len(body)))

    #: **AN N/A IS NOT A PASS -- IT IS A CHECK THAT DID NOT RUN, AND THAT IS ONLY
    #: TRUE IF IT IS VISIBLE.** The first version simply omitted this row for
    #: non-markdown, so a three-row gate on a `.py` read as "everything
    #: mechanical passed" when one mechanical condition was never evaluated --
    #: and left no absence to notice, since nothing was printed. That is this
    #: file's own spec-parse principle applied to one blind spot and not the
    #: other. Found by the second seat, not by the selftest.
    if path.endswith(".md"):
        good, why = status_line_is_time_invariant(body.decode("utf-8", "replace"))
        rec("status line", good, why)
    else:
        rows.append((" n/a", "status line", "NOT EVALUATED -- not a registration "
                     "document. An n/a is a check that did not run, not a pass."))

    rec("locked", is_locked(path),
        "read-only" if is_locked(path) else
        "WRITABLE -- an edit here would be unconscious rather than decided")

    return ok, rows


def report(path):
    ok, rows = gate(path)
    print("\n%s" % path)
    for state, name, detail in rows:
        print("  [%s] %-12s %s" % (state, name, detail))
    #: PRINTED ON EVERY RUN, PASS OR FAIL. The gate announces its own blind spot
    #: so a green result is never read as more than it is.
    print("  [ -- ] spec parse   NOT CHECKED BY THIS GATE AND NOT ATTESTABLE IN "
          "IT.\n                    A reading task: walk every operative phrase "
          "asking\n                    'what would I type?'. Must be run by a "
          "seat that did\n                    NOT draft the document.")
    return ok


def require_frozen(registration_path):
    """REFUSE to proceed unless the registration passes every mechanical gate.

    **A PRODUCER THAT CONSUMES A REGISTRATION CALLS THIS FIRST.**

    Registration M was signed by both seats, the owner gave the word, and the
    file was UNTRACKED AND WRITABLE while a producer read it. The gate that
    would have caught it existed, was written by the same seat, and had been run
    on the sibling registration four hours earlier. It was not run here because
    producing a column is not a freeze, so no ceremony fired.

    **THE GATE WAS NOT ATTACHED TO ANYTHING. IT RAN WHEN SOMEONE REMEMBERED.**
    This is the same shape as the locking gap that failed all campaign until it
    was wired into the freeze event -- the fix is never "remember", it is an
    edge in the call graph.
    """
    ok, rows = gate(registration_path)
    if not ok:
        bad = "; ".join("%s: %s" % (n, d) for s, n, d in rows if s == "FAIL")
        raise SystemExit(
            "REFUSING TO PRODUCE: %s is not frozen.\n  %s\n"
            "A producer consuming an unfrozen registration is a run whose "
            "premise can change under it." % (registration_path, bad))
    observed = next((d for s, n, d in rows if n == "hash"), "")
    print("  registration gate PASSED: %s\n    %s" % (registration_path, observed))
    return True


def selftest():
    """Every check must be proven to FAIL on a broken input, not just pass."""
    import tempfile
    ok = True

    def case(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print("  [%s] %s" % ("ok" if cond else "FAIL", name))

    #: EVERY CASE BELOW IS A WORDING THAT ACTUALLY OCCURS IN THE CORPUS. The
    #: first version of this selftest used only the wording of the document its
    #: author had open, and passed while the checker was wrong in both
    #: directions on four real files.
    good, why = status_line_is_time_invariant(
        "**STATUS AT DRAFTING (2026-08-03): draft; nothing computed.**")
    case("'STATUS AT DRAFTING (<date>)' passes", good)

    v6, why = status_line_is_time_invariant("## STATUS AS OF 2026-08-01 — added [2073]")
    case("'STATUS AS OF <date>' passes -- DIFFERENT WORDS, SAME PROPERTY", v6)

    bad, why = status_line_is_time_invariant("STATUS: DRAFT. Nothing is in force.")
    case("the EXPIRING form fires", not bad)
    case("...and the reason names the falsifier", "freeze falsifies it" in why)

    g, why = status_line_is_time_invariant(
        "**STATUS: DRAFT FOR FREEZE. Nothing below §0 has been computed.")
    case("'STATUS: DRAFT FOR FREEZE' fires -- no literal phrase in common", not g)
    case("...and it fires as UNDATED, not as missing", "UNDATED" in why)

    #: THE FIXTURE MUST NOT CONTAIN THE TOKEN IT CLAIMS IS ABSENT. The first
    #: version read "# A document with no status at all" -- which names the very
    #: word, so the checker correctly saw an undated status line and the case
    #: failed against working code. Third fixture defect in this file, against
    #: zero defects in the checker since the abspath fix.
    none, why = status_line_is_time_invariant("# A registration\n\nSome prose.")
    case("a MISSING status line fires", not none)
    case("...and is distinguished from an undated one", "NO STATUS LINE" in why)

    case("the word STATUS is matched CASE-INSENSITIVELY",
         status_line_is_time_invariant("status as of 2026-08-01")[0])

    case("a STATUS mention deep in the body is not read as the status line",
         not status_line_is_time_invariant(
             "# Title\n" + "\n" * 60 + "status as of 2026-08-01")[0])

    #: THE REAL FALSE POSITIVE, verbatim from registration_d_pairs_v5.md. It
    #: reached a signed amendment and an authorization. The document has NO
    #: status declaration; the match was a sentence about an experimental arm.
    v5 = ("# Registration D v5\n\nA DELTA on frozen c_delta_v6. **v1-v3 are "
          "SUPERSEDED.** v4's H1-signed arm carries over VERBATIM and is "
          "untouched — it was designed before any unblinding and its primary "
          "status is preserved:\nDrafted by seats that have never seen the "
          "withheld value. **The population does not yet exist.**\n")
    ok_v5, why_v5 = status_line_is_time_invariant(v5)
    case("'its primary status is preserved' is NOT a status declaration",
         not ok_v5 and "NO STATUS LINE" in why_v5)

    case("a declaration is recognised at the START of its line, through markdown",
         status_line_is_time_invariant("## STATUS AS OF 2026-08-01")[0]
         and status_line_is_time_invariant("**STATUS AT DRAFTING (2026-08-03)**")[0]
         and not status_line_is_time_invariant("**STATUS: DRAFT FOR FREEZE.**")[0])

    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.md")
    open(p, "w").write("**STATUS AT DRAFTING (2026-08-03): draft.**\n")

    #: THE CASE THAT CATCHES THE REAL DEFECT. The original selftest asserted
    #: only that an untracked file FIRES -- which a checker returning False for
    #: everything satisfies. A tracked file must NOT fire, and it must not fire
    #: WHEN NAMED BY A RELATIVE PATH FROM A DIFFERENT DIRECTORY, which is how
    #: every real invocation names it.
    #: `--full-name` is load-bearing: bare `git ls-files` prints names relative
    #: to the CWD, and joining those to the repo root builds a path to nothing.
    #: The first version of this case did exactly that and failed against a
    #: correct `is_tracked` -- the test was broken, not the checker.
    tracked_abs = subprocess.run(["git", "ls-files", "--full-name"],
                                 capture_output=True, text=True,
                                 cwd=os.path.dirname(
                                     os.path.abspath(__file__))).stdout.split("\n")[0]
    if tracked_abs:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))
                              ).stdout.strip()
        full = os.path.join(root, tracked_abs)
        rel = os.path.relpath(full, os.getcwd())
        case("a TRACKED file does NOT fire (absolute)", is_tracked(full))
        case("a TRACKED file does NOT fire (relative, from elsewhere)",
             is_tracked(rel))
    else:
        case("a TRACKED file could be located for the test", False)

    #: THE THREE CUSTODY STATES, CONSTRUCTED IN A SCRATCH REPO RATHER THAN
    #: ASSERTED. STAGED is the state the first version could not distinguish
    #: from COMMITTED, so it is the case that must exist.
    import subprocess as sp
    r = os.path.join(d, "repo")
    os.makedirs(r, exist_ok=True)
    G = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    sp.run(G + ["init", "-q", r], capture_output=True)
    f = os.path.join(r, "a.md")
    open(f, "w").write("**STATUS AT DRAFTING (2026-08-03): draft.**\n")
    case("an UNTRACKED file reports UNTRACKED", custody_state(f) == "UNTRACKED")
    sp.run(G + ["-C", r, "add", "a.md"], capture_output=True)
    case("a STAGED-ONLY file reports STAGED, not COMMITTED",
         custody_state(f) == "STAGED")
    sp.run(G + ["-C", r, "commit", "-qm", "x"], capture_output=True)
    case("a COMMITTED file reports COMMITTED", custody_state(f) == "COMMITTED")
    open(f, "a").write("edited\n")
    case("a COMMITTED-then-EDITED file is NOT committed -- the question is about "
         "THESE BYTES", custody_state(f) != "COMMITTED")
    #: THE THREE FAILING STATES ARE DISTINGUISHED, not merged into one message.
    #: A worktree edit that was never added has NO blob anywhere; saying its
    #: blob will be pruned is a false statement about a recoverable artifact.
    case("...and it reports MODIFIED, not STAGED -- never added, so no blob",
         custody_state(f) == "MODIFIED")
    sp.run(G + ["-C", r, "add", "a.md"], capture_output=True)
    case("...and once ADDED it reports STAGED", custody_state(f) == "STAGED")

    _, rows = gate(p)
    names = {n: s for s, n, _ in rows}
    case("an UNTRACKED file fires the custody row", names.get("custody") == "FAIL")
    case("a WRITABLE file fires", names.get("locked") == "FAIL")
    case("custody is asked BEFORE byte-stability",
         [n for _, n, _ in rows].index("custody") <
         [n for _, n, _ in rows].index("hash"))

    os.chmod(p, 0o444)
    _, rows = gate(p)
    case("...and a LOCKED file passes that check",
         {n: s for s, n, _ in rows}.get("locked") == "PASS")
    os.chmod(p, 0o644)

    case("a MISSING file fires", not gate(os.path.join(d, "nope.md"))[0])

    #: THE ROW COUNT IS THE ASSERTION. A silently-omitted check leaves nothing to
    #: notice, so the case must count rows, not inspect the ones that exist.
    py = os.path.join(d, "x.py")
    open(py, "w").write("# not a registration\n")
    _, md_rows = gate(p)
    _, py_rows = gate(py)
    case("a NON-markdown file emits the SAME NUMBER OF ROWS as a markdown one",
         len(py_rows) == len(md_rows))
    case("...and that row declares itself N/A rather than passing",
         any(n == "status line" and s.strip() == "n/a" for s, n, _ in py_rows))
    case("...and an n/a does NOT count toward the gate's verdict",
         "NOT EVALUATED" in [d_ for s, n, d_ in py_rows if n == "status line"][0])

    #: The blind-spot line is not decoration: prove it is emitted even when
    #: everything else FAILS, since that is the case a reader most wants to
    #: over-read.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(p)
    case("the blind-spot line prints even on a FAILING gate",
         "NOT CHECKED BY THIS GATE" in buf.getvalue())

    print("selftest %s" % ("PASS -- every check proven to fire" if ok
                           else "FAIL -- DO NOT GATE WITH THIS"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) < 2:
        sys.exit(selftest())
    results = [report(p) for p in sys.argv[1:]]
    print("\n%d/%d artifacts pass every MECHANICAL condition."
          % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)
