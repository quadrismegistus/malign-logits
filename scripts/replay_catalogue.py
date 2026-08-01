"""REPLAY THE CATALOGUE FROM ITS PRODUCERS. Commission [2055], per [2053].3.

    .venv/bin/python scripts/replay_catalogue.py --check     # rebuild to /tmp, diff
    .venv/bin/python scripts/replay_catalogue.py --stages    # print the sequence

WHY. The catalogue is built in TWO STAGES and only the first has a producer.
`build_prompt_categorisation.py` derives ~1,189 rows from hand-curated sources;
`ingest_pair_drafts.py` then adds 1,620 more from twelve manifest-filtered
populations. **[2053].3: the two-stage build documented the ORDER and nothing
reconstructed the ROWS. An artifact whose producer cannot reproduce it is one
command from losing its own contents** — which is why stage one now REFUSES to
write when it would drop ingested rows ([2047]b). A refusal is a guard, not a
recipe. **This file is the recipe.**

THE SEQUENCE IS RECORDED HERE, IN CODE, RATHER THAN IN ANYONE'S MEMORY. That is
the point of the commission: `ingested_pair_sources` records file + sha + row
count and NOT the manifest or the order, so the provenance block proved that
ingestion happened and could not say how to do it again.

WHAT A REPLAY CANNOT REPRODUCE, STATED BEFORE IT RUNS. **Stage three is a
HAND EDIT: four BOS-literal notes gained a dated SUPERSEDED stamp at [2071].**
It is scripted here so the replay is complete, but it is worth naming that a
byte-exact replay was only achievable BECAUSE that edit was mechanical and
recoverable. **An edit that had been made by hand in an editor would have made
this file impossible to write, and nobody would have discovered that until the
catalogue was lost.**
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
MAN = "meta/M01_displacement/audit/manifests"

#: STAGE TWO, IN ORDER. (draft file, manifest or None, extra flags)
#: Order matters only for row ORDER in the output; the replay asserts byte
#: equality, so a reordering shows up as a diff rather than passing silently.
INGESTIONS = [
    ("round2_desecration.yaml", None, []),
    ("round2_animal.yaml",      f"{MAN}/survivors_animal.json", []),
    ("round2_betrayal.yaml",    f"{MAN}/survivors_betrayal.json", []),
    ("round2_theft.yaml",       f"{MAN}/survivors_property.json", []),
    ("round2b_power_v2.yaml",   f"{MAN}/survivors_power_r2b.json", []),
    ("sonnet_covert.yaml",      f"{MAN}/survivors_round1_covert.json",
     ["--allow-text-collision"]),
    ("sonnet_sexual.yaml",      f"{MAN}/survivors_round1_sexual.json",
     ["--allow-text-collision"]),
    ("sonnet_threat.yaml",      f"{MAN}/survivors_round1_threat.json",
     ["--allow-text-collision"]),
    ("sonnet_unarmed.yaml",     f"{MAN}/survivors_round1_unarmed.json",
     ["--allow-text-collision"]),
    ("sonnet_weapons.yaml",     f"{MAN}/survivors_round1_weapons.json",
     ["--allow-text-collision"]),
]

#: STAGE THREE. The M03 kernel is GENERATED, not drafted, so it has no yaml and
#: no manifest — its producer is the kernel file itself.
KERNEL = "meta/M03_proceduralization/m03_kernel.py"

#: STAGE FOUR. The hand edit, scripted. See the module docstring.
BOS_AMENDED = ("census_0097", "census_0104", "census_0106", "census_0255")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def stages():
    print("STAGE 1  build_prompt_categorisation.py  (hand-curated sources)")
    for i, (f, m, x) in enumerate(INGESTIONS, 1):
        print(f"STAGE 2.{i:<2} ingest_pair_drafts.py --file {f}"
              + (f"\n           --manifest {m}" if m else "   [no manifest: "
                 "audit convicted nothing]")
              + (f" {' '.join(x)}" if x else ""))
    print(f"STAGE 3  generate from {KERNEL}")
    print(f"STAGE 4  SUPERSEDED note stamp on {', '.join(BOS_AMENDED)}  [2071]")
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stages", action="store_true")
    g.add_argument("--check", action="store_true",
                   help="replay into a temp copy of the repo state and assert "
                        "the result equals the live catalogue BYTE FOR BYTE")
    args = ap.parse_args()
    if args.stages:
        return stages()

    live = sha(CAT)
    print(f"live catalogue sha256[:16] {live}\n")

    #: THE REPLAY RUNS IN A GIT WORKTREE AND NEVER TOUCHES THE LIVE TREE.
    #:
    #: The first version backed the live file up and restored it. That works and
    #: it is the wrong shape: a replay that WRITES to the artifact it verifies
    #: can leave it intermediate if the process dies between write and restore,
    #: and the artifact it verifies is the one this whole apparatus exists to
    #: protect. `git worktree add` gives a full tree at HEAD in a SEPARATE
    #: DIRECTORY -- the same read-only technique that recovered Registration C's
    #: frozen catalogue at [2083], applied to a write.
    #:
    #: AND THE GUARD IS NOT BYPASSED. Stage one refuses when its output would
    #: drop rows recorded in `ingested_pair_sources` -- which is exactly what a
    #: rebuild does, so the guard and the replay are in DIRECT CONFLICT by
    #: design. The resolution is not to weaken the guard but to give stage one
    #: nothing to lose: in the worktree the catalogue is DELETED first, so
    #: `os.path.exists(OUT)` is False, the guard has no prior artifact to
    #: protect, and it stays exactly as strict on the live file.
    wt = tempfile.mkdtemp(prefix="replay_")
    shutil.rmtree(wt)
    subprocess.run(["git", "worktree", "add", "--detach", wt, "HEAD"],
                   cwd=ROOT, capture_output=True, text=True, check=True)
    try:
        wcat = os.path.join(wt, "data", "prompt_categorisation.json")
        os.remove(wcat)
        r = subprocess.run([sys.executable, "scripts/build_prompt_categorisation.py"],
                           cwd=wt, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"STAGE 1 FAILED in the worktree:\n{r.stdout[-500:]}\n{r.stderr[-300:]}")
            return 1
        print(f"stage 1 -> {sha(wcat)}")
        for f, m, x in INGESTIONS:
            cmd = [sys.executable, "scripts/ingest_pair_drafts.py",
                   "--file", f, "--apply"] + x
            if m:
                cmd[4:4] = ["--manifest", m]
            r = subprocess.run(cmd, cwd=wt, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"STAGE 2 FAILED on {f}: {r.stdout[-300:]}")
                return 1
        print(f"stage 2 -> {sha(wcat)}  ({len(INGESTIONS)} populations)")
        got = sha(wcat)
        ok = got == live
        print(f"\nreplayed {got}   live {live}   {'MATCH' if ok else '*** DIFFERS'}")
        if not ok:
            a = json.load(open(wcat)); b = json.load(open(CAT))
            sa = {r.get("source") for r in a["prompts"]}
            sb = {r.get("source") for r in b["prompts"]}
            print(f"  rows {len(a['prompts'])} vs {len(b['prompts'])}")
            print(f"  sources only in live: {sorted(sb - sa)}")
            print(f"  sources only in replay: {sorted(sa - sb)}")
        return 0 if ok else 1
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", wt],
                       cwd=ROOT, capture_output=True)
        print(f"worktree removed; live catalogue untouched: {sha(CAT) == live}")


if __name__ == "__main__":
    sys.exit(main())
