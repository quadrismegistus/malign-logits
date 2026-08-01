"""MUTATION HARNESS FOR SELF-TESTS. [2291]'s matrix standard, made reusable.

    .venv/bin/python scripts/mutate_selftest.py <target.py> --matrix m04

WHY IT EXISTS. The M04 producer's self-test reported **7 of 7 pass**, and it
reproduced at a second seat. **One of the seven could not fail.** Case 7 read

    guarded = "gate_4_contrasts" not in body or "--dry" in body

and `--dry` appears unconditionally in `main()`, so the OR short-circuits true
forever. A mutant that made `main()` call `gate_4_contrasts` ungated — **exactly
what case 7 exists to catch** — passed. Only running the mutant showed it.

Then the repair pass fixed that and **a second unfalsifiable case surfaced one
layer down**: case 2 called its guard and asserted on the RETURN VALUE, so a
guard with its check deleted entirely still passed. Two rounds, two cases that
could not fail, both found by mutation and neither by reading.

    A REPAIR PASS IS VERIFIED BY ALL MUTANTS, NOT BY THE MUTANT THAT
    MOTIVATED IT.

THE RULE THE MATRIX ENCODES. **A known-answer case that checks a RETURN VALUE
tests the stub it was fed; only a case that checks the RAISE tests the guard.**
Every structural case carries both arms — the passing input returns, the
violating input raises. Cases written `try/raise/except` had their mutants
caught; the case written `assert len(result)==1` did not. **The difference
between the caught and the surviving mutant is exactly the arm count.**

WHAT A SURVIVING MUTANT MEANS. The self-test cannot distinguish the guard from
its absence. That is a defect in the TEST, never in the mutant — the mutant is
just the question asked out loud.

WHAT THIS DOES NOT DO. It cannot invent the mutations. Each matrix is written by
hand against a specific file's guards, because a mutation that does not
correspond to a real failure mode proves nothing when caught. **The matrices
below are the ones a real defect taught.**
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Each entry: name -> (anchor, replacement). The anchor MUST be present or the
#: mutant is reported UNAPPLIED rather than passing — an absent anchor looks
#: exactly like a caught mutant otherwise, which is this file's own failure mode.
MATRICES = {
    "m04": {
        #: RE-ANCHORED at 66ef2e08: main() now calls run_pipeline, so the old
        #: anchor is gone. The mutant is unchanged in spirit -- reach stage 4
        #: without passing gates 1-3 -- and is now expressed as skipping the
        #: gate_2 halt inside run_pipeline.
        "M1 gate_2 halt bypassed": (
            "if not gate_2_divergence(", "if False and not gate_2_divergence("),
        "M2 pos0 leaks into the curve": (
            "CURVE = list(range(1, 10))", "CURVE = list(range(0, 10))"),
        "M3 control passes on flat": (
            "ok = med < DIVERGENCE_MIN_SLOPE and frac_falling > 0.5",
            "ok = med <= DIVERGENCE_MIN_SLOPE or frac_falling >= 0.0"),
        "M4 commensurability guard neutered": (
            "if not (set(a) & set(b)):", "if False:"),
        "M5 empty tripwire neutered": (
            "    if not rows:", "    if False:"),
        "M6 lineage-unit guard neutered": (
            "if len(lins) == len(labels) and len(labels) > 1:", "if False:"),
    },
}



def deletion_mutants(src):
    """DERIVE a mutant per function by EMPTYING its body. [2295]'s procedure.

    **The hand-written matrix needs anchors, and anchors drift.** This needs
    nothing: it finds every top-level `def`, replaces its body with a permissive
    stub, and asks whether the self-test notices.

    **A check whose subject can be emptied without the check noticing is
    measuring something adjacent to what it names.** That is the day's dominant
    defect class, and unlike the diagnosis ("this test is exercising its
    fixture") the deletion test is a PROCEDURE: one edit, no judgement.

    It found M6 the hard way first — a hand-built bare-resolve mutant, arrived
    at by seeing that a four-line stub inside the test was the real subject.
    **The procedure gets there by rote.**

    LIMIT, NAMED: functions with no body to delete are invisible to it. The M04
    self-test's cases 3 and 7 grep the file's own source, so they have no
    function subject and this says nothing about them. Their equivalent is a
    hand-written mutant, which is why both forms stay.
    """
    out = []
    for m in re.finditer(r"^def ([a-z_][a-z0-9_]*)\(([^)]*)\):\n", src, re.M):
        name, sig = m.group(1), m.group(2)
        if name in ("main", "selftest", "run", "deletion_mutants"):
            continue
        start = m.end()
        nxt = re.search(r"^(def |# ─|if __name__)", src[start:], re.M)
        end = start + (nxt.start() if nxt else len(src) - start)
        #: A permissive stub: returns something truthy-ish and raises nothing,
        #: so ONLY a case that checks the guard's raise can notice.
        stub = "    return [] if 'assert' in __name__ else None\n\n"
        out.append((f"DEL {name}", src[:start] + stub + src[end:]))
    return out


def orphan_functions(src):
    """Top-level functions NOT REACHABLE FROM main(). The [2298] shape.

    **This is the check whose absence cost a clearance** — and whose FIRST
    version, written twenty minutes later, missed the same defect for the same
    reason one level down.

    v1 counted references anywhere in the file and reported "none" on the very
    file that was broken, because `gate_2_divergence` is called twice inside the
    SELF-TEST and case 7 greps the literal string `gate_4_contrasts`. **A
    pipeline stage exercised only by its own test looks called.** Same error
    class as the two-namespace false zero and the wildcard that cleared 28
    files: **the count was taken over the wrong region, and it erred toward
    CLEARANCE, which is the direction nobody re-reads.**

    v2 walks the actual call graph from `main()`, transitively, and EXCLUDES
    `selftest` from the walk — because a function reachable only through the
    self-test is precisely what "verifies beautifully and does nothing" means.
    """
    bodies = {}
    for m in re.finditer(r"^(?:def|class) ([A-Za-z_][A-Za-z0-9_]*)\(?", src, re.M):
        name = m.group(1)
        start = m.end()
        nxt = re.search(r"^(def |class |# \u2500|if __name__)", src[start:], re.M)
        bodies[name] = src[start:start + (nxt.start() if nxt else len(src) - start)]
    if "main" not in bodies:
        return []
    seen, stack = set(), ["main"]
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in bodies:
            continue
        seen.add(cur)
        for other in bodies:
            #: the self-test is NOT a route to the pipeline
            if other == "selftest" or other == cur:
                continue
            if re.search(rf"\b{re.escape(other)}\s*\(", bodies[cur]):
                stack.append(other)
    return [(n, 0, 0) for n in sorted(bodies)
            if n not in seen and n not in ("main", "selftest")
            and not n.startswith("_") and n[0].islower()]



#: ── THE HARNESS'S OWN KNOWN-ANSWER CASES ──────────────────────────────────────
#: [2305]: an auditor's instrument meets the same bar as the audited. This
#: reachability check was WRONG TWICE in twenty minutes and cleared nothing until
#: it had these. Each fixture is one of those two failures, plus the two shapes
#: they must not over-report.

_FIX = {
    "A unreachable function is FLAGGED": ("""
def orphan(x):
    return x
def helper(x):
    return x
def main():
    return helper(1)
""", True),
    #: v1's FAILURE. It counted references anywhere and reported "none" on the
    #: broken file, because a stage called twice inside the self-test looks
    #: called. A SELF-TEST IS NOT A ROUTE TO THE PIPELINE.
    "B reached only from selftest is FLAGGED": ("""
def stage(x):
    return x
def selftest():
    return stage(1)
def main():
    return selftest()
""", True),
    #: v2's FAILURE. It walked only top-level defs and flagged a function that a
    #: CLASS METHOD reaches, over-reporting on correct code.
    "C reached via a class method is NOT flagged": ("""
def norm(s):
    return s
class Lineages:
    def __init__(self):
        self.x = norm("a")
def main():
    return Lineages()
""", False),
    "D everything reachable is NOT flagged": ("""
def a(x):
    return x
def b(x):
    return a(x)
def main():
    return b(1)
""", False),
}


def harness_selftest():
    """Four cases, one per failure this check actually had. No target needed."""
    print("HARNESS SELF-TEST — reachability, [2305]'s bar applied to the auditor\n")
    ok = True
    for name, (src, should_flag) in _FIX.items():
        flagged = bool(orphan_functions(src))
        good = flagged == should_flag
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<44}"
              f"flagged={flagged} expected={should_flag}")
    print(f"\n  {'all pass' if ok else '*** THIS CHECK CLEARS NOTHING UNTIL IT PASSES'}")
    return ok

def run(path, args=("--selftest",)):
    r = subprocess.run([sys.executable, path, *args],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?")
    ap.add_argument("--harness-selftest", action="store_true")
    ap.add_argument("--matrix", choices=sorted(MATRICES))
    ap.add_argument("--deletion", action="store_true",
                    help="[2295]: empty each function body and ask whether the "
                         "self-test notices; derives its own mutants, no anchors")
    args = ap.parse_args()

    if args.harness_selftest:
        return 0 if harness_selftest() else 1

    src = open(args.target).read()

    #: BASELINE FIRST. A target whose self-test already fails makes every mutant
    #: look caught, and the matrix would report a perfect score on a broken file.
    rc, _ = run(args.target)
    print(f"BASELINE  {os.path.basename(args.target)}  "
          f"self-test {'PASSES' if rc == 0 else '*** FAILS'}")
    if rc != 0:
        print("  Refusing: mutants are meaningless against a failing baseline.")
        return 2

    orphans = orphan_functions(src)
    print(f"\nREACHABILITY — top-level functions unreachable from main()")
    if orphans:
        for n, _, _ in orphans:
            print(f"  *** {n:<34} NOT REACHABLE from main() (self-test excluded)")
        print("  A component wired to nothing passes every test aimed at its"
              "\n  internals. Ask whether the PROGRAM reaches it, not whether it works.")
    else:
        print("  none — every top-level function is reachable from main()")

    survived_del = []
    if args.deletion:
        muts = deletion_mutants(src)
        print(f"\nDELETION TEST — {len(muts)} function bodies emptied, "
              f"each expected to be CAUGHT ([2295])\n")
        with tempfile.TemporaryDirectory() as td:
            for name, mutant in muts:
                p = os.path.join(td, "mutant.py")
                open(p, "w").write(mutant)
                rc, _ = run(p)
                if rc == 0:
                    survived_del.append(name)
                print(f"  {name:<40}{'CAUGHT' if rc else '*** SURVIVED'}")
        print(f"\n  {len(muts) - len(survived_del)} of {len(muts)} caught")
        if survived_del:
            print("  A function whose BODY can be emptied without the self-test")
            print("  noticing is not the thing its case is measuring.")
        if not args.matrix:
            return 0 if (not survived_del and not orphans) else 1

    matrix = MATRICES[args.matrix]
    print(f"\nMUTATION MATRIX '{args.matrix}' — {len(matrix)} mutants, "
          f"each expected to be CAUGHT\n")
    survived, unapplied = [], []
    with tempfile.TemporaryDirectory() as td:
        for name, (a, b) in matrix.items():
            if a not in src:
                unapplied.append(name)
                print(f"  {name:<40}*** ANCHOR MISSING — NOT APPLIED")
                continue
            p = os.path.join(td, "mutant.py")
            open(p, "w").write(src.replace(a, b, 1))
            rc, _ = run(p)
            caught = rc != 0
            if not caught:
                survived.append(name)
            print(f"  {name:<40}{'CAUGHT' if caught else '*** SURVIVED'}")

    n = len(matrix)
    print(f"\n  {n - len(survived) - len(unapplied)} of {n} caught")
    if unapplied:
        print(f"  {len(unapplied)} NOT APPLIED — anchors drifted; the matrix "
              f"needs updating before its score means anything.")
    if survived:
        print("\n  A SURVIVING MUTANT IS A DEFECT IN THE TEST, NOT THE MUTANT.")
        print("  The self-test cannot distinguish the guard from its absence.")
    return 0 if (not survived and not unapplied and not survived_del
                 and not orphans) else 1


if __name__ == "__main__":
    sys.exit(main())
