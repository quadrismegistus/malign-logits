"""Make "committed before running" a checkable property of the output.

WHY. On 2026-07-27 a fabricated commit SHA was found in F39, in the sentence
"Script committed at X before running" -- the pre-registration guarantee. The
day's response was a checker (`scripts/verify_citations.py`) and two proposed
rules about citing values. lacan then found the hole under all of them:

    "Script committed at X" DOES NOT ESTABLISH THAT THE SCRIPT THAT RAN IS THE
    SCRIPT AT X.

If the working tree was dirty, they differ, the guarantee is void, and every
SHA in the sentence still resolves perfectly. No citation discipline catches
that, because nothing in the citation is wrong.

The second finding was worse: the value the author was supposed to cite had
never been emitted by anything. `data/preference_corpus_gate_v2.json` carried
two keys and no provenance; no analysis script recorded the HEAD it ran under.
So the discipline being demanded was to hold an unprinted string across a
chained command and transcribe it correctly hours later. That is a memory test,
and a rule saying "do not fail the memory test" gets broken by whoever is
tired. This module removes the test.

WHAT IT RECORDS, and why the third field is the load-bearing one:

    commit        HEAD at run time -- what the citation names
    tree_clean    whether anything was uncommitted (lacan's ask)
    script_blob   git hash-object of the RUNNING FILE, and whether it equals
                  the blob at HEAD for that path
    closure       the same check for every first-party file the run EXECUTES,
                  not just the entry point

`script_blob` is strictly stronger than `tree_clean`. A dirty tree does not
imply the *cited script* differed; a clean tree is not needed if the script's
own bytes match what HEAD holds. This identifies the exact code that ran,
independent of unrelated edits elsewhere in the tree, so a checker can confirm
the pre-registration claim from the artifact alone without trusting the author
and without requiring the whole repository to have been pristine.

WHY CLOSURE AND NOT JUST THE ENTRY POINT (lacan, and it lands on this
project's own code). The blob check answers "did the cited commit's bytes run"
for the file you launched. It says nothing about what that file executes at
runtime. `scripts/tier2_gate_grid.py` -- the script whose commit is named as
THE registration in the power-floor ruling -- does:

    exec(open('scripts/tier2_power_check.py').read().split(...)[0])

and the registered gate reaches the same code through `tier2_construct_grid`.
Either file could be edited between commit and run while the entry point's
blob matches HEAD perfectly, voiding the pre-registration guarantee with every
check proposed today returning green. A check that passes on the script most
needing it is not a check.

Declared closure is EXPLICIT because it has to be: `exec(open(...).read())`
leaves no trace in `sys.modules`, and neither does `exec_module` on a spec that
was never registered -- which is exactly how the gate loads its grid. The
`sys.modules` sweep is a safety net for ordinary imports, not the mechanism.

Usage, three lines at the top of any run whose result will be pre-registered:

    from malign_logits.provenance import provenance
    ...
    prov = provenance(__file__, closure=["scripts/thing_it_execs.py"])
    json.dump({"provenance": prov, **results}, fh)
"""
from __future__ import annotations

import pathlib
import subprocess


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def provenance(script_path: str | None = None,
               closure: "list[str] | None" = None) -> dict:
    """Provenance for a registered run. Never raises; degrades to nulls.

    Failure to determine provenance must not abort an expensive run, but it
    must be VISIBLE in the output rather than absent -- an empty field is a
    finding, a missing key is a silence.
    """
    commit = _git("rev-parse", "HEAD") or None
    status = _git("status", "--porcelain")
    p = dict(
        commit=commit,
        commit_short=commit[:7] if commit else None,
        tree_clean=(status == "") if commit else None,
        # porcelain v1 is "XY<space>PATH"; slicing a fixed offset silently
        # ate a character in testing, so strip the status field explicitly.
        dirty_paths=sorted(l[2:].lstrip() for l in status.splitlines())[:20] or None,
        script=None,
        script_blob=None,
        script_matches_commit=None,
    )
    if not script_path:
        return p

    f = pathlib.Path(script_path).resolve()
    root = _git("rev-parse", "--show-toplevel")
    try:
        rel = str(f.relative_to(root)) if root else f.name
    except ValueError:
        rel = f.name
    p["script"] = rel
    p["script_blob"] = _git("hash-object", str(f)) or None
    # The claim that matters: are the bytes that just executed the bytes the
    # cited commit holds for this path?
    at_head = _git("rev-parse", f"HEAD:{rel}")
    if p["script_blob"] and at_head:
        p["script_matches_commit"] = p["script_blob"] == at_head

    # Everything else this run executes. Declared paths first; then a sweep of
    # already-imported first-party modules, which catches ordinary imports but
    # NOT exec'd source -- hence the declaration.
    paths = list(closure or [])
    import sys as _sys
    for m in list(_sys.modules.values()):
        f2 = getattr(m, "__file__", None)
        if not f2 or not root:
            continue
        rp = str(pathlib.Path(f2).resolve())
        # The virtualenv lives INSIDE the repo root, so a naive "under root"
        # test swept 1,139 site-packages files into the closure. They are not
        # tracked, so every one resolved to matches_commit=None -- and the
        # summary line reported "1152 file(s) match HEAD". A check that counts
        # unverified files as passes is worse than no check, and this one
        # produced its false green on its first real run.
        if not rp.startswith(root + "/") or "/.venv/" in rp or "/site-packages/" in rp:
            continue
        paths.append(rp)
    seen, cl = set(), []
    for q in paths:
        qp = pathlib.Path(q).resolve()
        try:
            qrel = str(qp.relative_to(root)) if root else str(qp)
        except ValueError:
            continue
        # Some third-party modules carry a RELATIVE __file__, which resolves
        # against cwd and lands inside the repo root as a bare filename that
        # does not exist. Drop those rather than reporting them unverified.
        if qrel in seen or qrel == rel or not qp.exists():
            continue
        seen.add(qrel)
        blob = _git("hash-object", str(qp)) if qp.exists() else ""
        head = _git("rev-parse", f"HEAD:{qrel}")
        cl.append(dict(path=qrel, blob=blob or None,
                       tracked=bool(head),
                       matches_commit=(blob == head) if (blob and head) else None))
    p["closure"] = sorted(cl, key=lambda d: d["path"]) or None
    # True only if every closure file was actually CHECKED and matched.
    # Unknown (untracked) counts against it rather than being ignored.
    p["closure_matches_commit"] = (
        all(c["matches_commit"] is True for c in cl) if cl else None)
    p["closure_unverified"] = [c["path"] for c in cl
                               if c["matches_commit"] is not True] or None
    return p


def describe(p: dict) -> str:
    """One line for stdout, so the value is OBSERVED and not remembered."""
    if not p.get("commit"):
        return "provenance: UNAVAILABLE (not a git checkout?)"
    bits = [f"commit {p['commit_short']}"]
    bits.append("tree clean" if p.get("tree_clean") else "TREE DIRTY")
    m = p.get("script_matches_commit")
    if m is True:
        bits.append(f"{p['script']} matches HEAD")
    elif m is False:
        bits.append(f"*** {p['script']} DIFFERS FROM HEAD -- "
                    f"the cited commit is not what ran ***")
    cl = p.get("closure") or []
    bad = [c["path"] for c in cl if c["matches_commit"] is False]
    if bad:
        bits.append(f"*** CLOSURE DIFFERS FROM HEAD: {', '.join(bad)} ***")
    elif cl:
        good = sum(1 for c in cl if c["matches_commit"] is True)
        unk = len(cl) - good
        bits.append(f"closure {good}/{len(cl)} verified against HEAD"
                    + (f", {unk} UNVERIFIED (untracked)" if unk else ""))
    return "provenance: " + ", ".join(bits)
