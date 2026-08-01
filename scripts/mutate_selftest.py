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


def run(path, args=("--selftest",)):
    r = subprocess.run([sys.executable, path, *args],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--matrix", required=True, choices=sorted(MATRICES))
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
    return 0 if (not survived and not unapplied) else 1


if __name__ == "__main__":
    sys.exit(main())
