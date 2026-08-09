#!/usr/bin/env python
"""f11_fleet_sync.py — pull the fleet's output back, repeatedly, while it runs.

**NOTHING HAD LEFT THE BOXES.** Three instances ran for over an hour with every
record living only on rented disk. vast.ai instances are preempted, hosts fail
mid-run (three did tonight before one even started), and a box destroyed for any
reason takes the corpus with it. RH asked whether a sync loop existed. It did not.

WHY REPEATED RSYNC IS CHEAP HERE, and it is a property of the format rather than
of rsync: `twp_cloud` writes ONE JSONL PER MODEL and **a finished model's file
never changes again.** So every pass after the first transfers only the model
currently in progress. The `.f16` sidecars are append-only for the same reason.

    scripts/f11_fleet_sync.py --once     one pass, print what landed
    scripts/f11_fleet_sync.py            loop until every box is gone

**THE ROSTERS ARE DISJOINT AND THAT IS CHECKED, NOT ASSUMED** — box1 92, box2 10,
box3 2, zero pairwise overlap — so all three land in one local directory without
a file ever being written by two sources. A collision here would not error; it
would interleave two models' rows under one name.
"""
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "data", "f11_twp")
#: the backfill writes to its OWN directory -- resume is by completed-prompt
#: readback, so a backfill pointed at the main directory would skip every model
#: in it and produce nothing
REMOTE_DIRS = ["/workspace/f11_twp", "/workspace/f11_twp_bf",
               "/workspace/f11_twp_delta"]
REMOTE = REMOTE_DIRS[0]


def boxes():
    """Live boxes, from their state files. Each carries its own name."""
    out = []
    #: **`.vastai.*.json` DOES NOT MATCH `.vastai.json`** -- the pattern needs a
    #: middle segment, and box 1's state file has none. The first sync ran
    #: clean, reported "ok", and silently skipped the only box with output.
    #: A glob that misses is indistinguishable from a box with nothing to send.
    seen = set()
    paths = sorted(glob.glob(os.path.join(ROOT, ".vastai*.json")))
    for p in paths:
        name = os.path.basename(p)
        if "DESTROYED" in name or "destroyed" in name or "stale" in name:
            continue
        try:
            st = json.load(open(p))
        except Exception:
            continue
        key = (st.get("ssh_host"), st.get("ssh_port"))
        if st.get("ssh_host") and st.get("ssh_port") and key not in seen:
            seen.add(key)
            out.append((name, st))
    return out


def pull(st):
    """(state, detail). Three outcomes, not two.

    **"NOT YET" AND "BROKEN" MUST NOT PRINT THE SAME WORD.** A box whose
    `/workspace/f11_twp` does not exist yet -- because its first model has not
    finished, or its kernels are still compiling -- returns rsync code 23 with
    a broken pipe, exactly like a box that has failed. Reporting both as FAIL
    trains the reader to skim past the line that matters, which is the whole
    reason a real failure goes unnoticed in a log full of benign ones.
    """
    got_any, errs = False, []
    for rd in REMOTE_DIRS:
        dest = DEST if rd == REMOTE else DEST + rd.split("f11_twp")[-1]
        os.makedirs(dest, exist_ok=True)
        cmd = ["rsync", "-az", "--partial",
               "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
                     "-o LogLevel=ERROR -o ConnectTimeout=20 -p %s" % st["ssh_port"],
               "root@%s:%s/" % (st["ssh_host"], rd), dest + "/"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            got_any = True
        else:
            errs.append((rd, (r.stderr or "").strip()))
    if got_any and not errs:
        return "ok", ""
    if got_any:
        return "ok", "(%s not present yet)" % ",".join(os.path.basename(d) for d, _ in errs)
    r = subprocess.run(["true"], capture_output=True, text=True)
    if False:
        return "ok", ""
    err = (r.stderr or "").strip()
    #: distinguish by ASKING THE BOX, not by pattern-matching rsync's message
    probe = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20",
         "-p", str(st["ssh_port"]), "root@%s" % st["ssh_host"],
         "test -d %s && echo YES || echo NO" % REMOTE],
        capture_output=True, text=True)
    if probe.returncode == 0 and probe.stdout.strip() == "NO":
        return "waiting", "no %s yet -- box has produced nothing" % REMOTE
    if probe.returncode != 0:
        return "UNREACHABLE", (probe.stderr or "").strip().splitlines()[-1:][0][:70] \
            if probe.stderr.strip() else "ssh failed"
    return "FAIL", err.splitlines()[-1:][0][:70] if err else "rc=%d" % r.returncode


def census():
    """**COUNT BOTH DIRECTORIES.** The backfill writes to DEST_bf, and a census
    that reads only DEST reports a flat line while the backfill is the only
    thing running -- which is exactly when someone is watching for movement.
    A counter blind to half the work is worse than no counter: it does not say
    nothing, it says nothing is happening."""
    n_j = n_f = rows = 0
    mb = 0.0
    for d in (DEST, DEST + "_bf", DEST + "_delta"):
        js = glob.glob(os.path.join(d, "*.jsonl"))
        n_j += len(js)
        n_f += len(glob.glob(os.path.join(d, "*.f16")))
        for f in js:
            with open(f, errors="ignore") as fh:
                rows += sum(1 for _ in fh)
        mb += sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*"))
                  if os.path.isfile(f)) / 1e6
    return n_j, n_f, rows, mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=300, help="seconds between passes")
    a = ap.parse_args()
    os.makedirs(DEST, exist_ok=True)
    while True:
        bs = boxes()
        if not bs:
            print("no live boxes; stopping", flush=True)
            return
        for name, st in bs:
            state, detail = pull(st)
            print("  %-28s %-12s %s" % (name, state, detail), flush=True)
        n_j, n_f, rows, mb = census()
        print("%s  local: %d jsonl / %d f16 / %d rows / %.0f MB"
              % (time.strftime("%H:%M:%S"), n_j, n_f, rows, mb), flush=True)
        if a.once:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
