"""PRE-FREEZE POPULATION GATE — two clauses, because one is not enough.

Ruled at [1889].3, superseding [1887].4. Run before any population freeze.

    .venv/bin/python scripts/population_freeze_gate.py

WHY TWO CLAUSES. [1887].4 required every file in the declared population be
re-hashed against its source. That is necessary and it cannot find what is missing
from the list: `round2b_power.yaml` — 120 pairs, the power redraft — was in no
repo at all, so no hash check over the population would ever have looked at it.
A hash check over a list cannot find what is absent from the list.

    (i)  every file in the declared population RE-HASHED against its source
    (ii) every DIRECTORY the population is drawn from LISTED, and anything in it
         not in the population NAMED — included, or excluded with a reason

Clause (ii) is the audit's own field-check discipline applied to the substrate:
ENUMERATE, DO NOT SAMPLE.

WHAT THIS EXISTS TO PREVENT, stated because a checker whose reason is undocumented
gets deleted by the next person who finds it redundant. On 2026-07-31/08-01 NINE
governing artifacts were found surviving in a single location — a Dropbox folder, a
working tree, a job scratch directory that dies with its session. Two of them
(Registration E v1 and v2) are cited by hash in the docket record and resolve to
nothing, permanently. The worst was `pair_authoring_template.md`: the standard 600
pairs were judged against, in Dropbox only, while they were being judged.

AND THE FAILURE THIS CATCHES THAT NOBODY EXPECTED: committing an artifact is not
tracking it. The M03 scenarios were committed at 01:27 and the cross-SPEAKER
`ought to` addition landed at 05:32; for four hours the repo carried a population
predating a change that had been verified four ways and declared clean. It was
found by accident. A COMMITTED COPY OF A LIVE ARTIFACT IS A CLAIM ABOUT A
TIMESTAMP, NOT ABOUT A FILE.
"""

import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROPBOX = os.path.expanduser("~/Dropbox/Prof/Articles/TheoryMachines")

#: (repo path, live source path). The source is where the artifact is AUTHORED;
#: the repo copy is a snapshot and goes stale silently.
POPULATIONS = {
    "M01 pair drafts": [
        (f"pair_drafts/{n}", f"{DROPBOX}/pair_drafts/{n}") for n in (
            "round2_animal.yaml", "round2_betrayal.yaml", "round2_desecration.yaml",
            "round2_power.yaml", "round2_theft.yaml", "round2b_power.yaml",
            "sonnet_covert.yaml", "sonnet_sexual.yaml", "sonnet_threat.yaml",
            "sonnet_unarmed.yaml", "sonnet_weapons.yaml", "EXCLUSIONS.json")
    ],
    "M03 scenarios": [
        (f"pair_drafts/{n}", f"{DROPBOX}/pair_drafts/{n}")
        for n in ("m03_scenarios_A.yaml", "m03_scenarios_B.yaml")
    ],
    "governing documents": [
        ("meta/M01_displacement/audit/pair_authoring_template.md",
         f"{DROPBOX}/agents/lacan/pair_authoring_template.md"),
        ("meta/M03_proceduralization/prompt_authoring_guide.md",
         f"{DROPBOX}/agents/registrar/m03_prompt_authoring_guide.md"),
    ],
}

#: Directories clause (ii) enumerates, and what in them is deliberately NOT a
#: population member. An unlisted file is a FINDING, not a default.
DIRECTORIES = {
    f"{DROPBOX}/pair_drafts": {
        "quarantine_severity_quota": "the severity-quota drafts, quarantined; "
                                     "committed separately as the record of WHY",
        "prompt_queue_candidates.md": "repo-native, no Dropbox source",
    },
}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    fail = 0

    print("CLAUSE (i) — every declared file re-hashed against its source")
    for pop, files in POPULATIONS.items():
        print(f"\n  {pop}")
        for rel, src in files:
            rp = os.path.join(ROOT, rel)
            if not os.path.exists(rp):
                print(f"    {os.path.basename(rel):<34} *** NOT IN REPO")
                fail += 1
                continue
            if not os.path.exists(src):
                print(f"    {os.path.basename(rel):<34} source absent — cannot verify "
                      f"(NOT a pass)")
                fail += 1
                continue
            a, b = sha(rp), sha(src)
            if a == b:
                print(f"    {os.path.basename(rel):<34} {a}  ok")
            else:
                print(f"    {os.path.basename(rel):<34} repo {a} / source {b}  *** STALE")
                fail += 1

    print("\nCLAUSE (ii) — every source directory enumerated, non-members named")
    declared = {os.path.basename(s) for fs in POPULATIONS.values() for _, s in fs}
    for d, known in DIRECTORIES.items():
        print(f"\n  {d}")
        if not os.path.isdir(d):
            print("    *** DIRECTORY MISSING")
            fail += 1
            continue
        for entry in sorted(os.listdir(d)):
            if entry.startswith("."):
                continue
            if entry in declared:
                continue
            if entry in known:
                print(f"    {entry:<40} excluded: {known[entry]}")
            else:
                print(f"    {entry:<40} *** UNDECLARED — include it or give a reason")
                fail += 1

    print("\nUNTRACKED CHECK — anything in the repo's population dirs not in git")
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all",
                          "pair_drafts", "meta"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    if out:
        for line in out.splitlines():
            print(f"    *** {line}")
            fail += 1
    else:
        print("    clean")

    print("\n" + "=" * 66)
    if fail:
        print(f"GATE CLOSED — {fail} finding(s). DO NOT FREEZE.")
        return 1
    print("GATE OPEN — population verified against source, substrate enumerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
