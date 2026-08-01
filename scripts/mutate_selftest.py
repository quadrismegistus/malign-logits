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
        "M1 main calls gate_4 ungated": (
            '    print("This file is the artifact',
            '    gate_4_contrasts({}, 0.0, [0.0]*9)\n'
            '    print("This file is the artifact'),
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

def run(path, args=("--selftest",)):
    r = subprocess.run([sys.executable, path, *args],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--matrix", choices=sorted(MATRICES))
    ap.add_argument("--deletion", action="store_true",
                    help="[2295]: empty each function body and ask whether the "
                         "self-test notices; derives its own mutants, no anchors")
    args = ap.parse_args()

    src = open(args.target).read()

    #: BASELINE FIRST. A target whose self-test already fails makes every mutant
    #: look caught, and the matrix would report a perfect score on a broken file.
    rc, _ = run(args.target)
    print(f"BASELINE  {os.path.basename(args.target)}  "
          f"self-test {'PASSES' if rc == 0 else '*** FAILS'}")
    if rc != 0:
        print("  Refusing: mutants are meaningless against a failing baseline.")
        return 2

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
            return 0 if not survived_del else 1

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
    return 0 if (not survived and not unapplied and not survived_del) else 1


if __name__ == "__main__":
    sys.exit(main())
