#!/usr/bin/env python3
"""Turn a finished BLT box into a bge box, in place.

    python scripts/bge_fleet_launch.py --shard 1 [--dry-run]
    python scripts/bge_fleet_launch.py --all-idle

WHY IN PLACE. Each box already holds `blt_passages.jsonl.gz` (224 MB) from the
BLT run, and bge reads the SAME corpus with the SAME sharding. Re-renting and
re-provisioning would re-upload 224 MB per box and pay the provisioning failure
rate documented in the runbook, to arrive back where the box already is. What
actually ships is an 8 KB script.

THE GUARD THAT MATTERS: REFUSES WHILE BLT IS STILL RUNNING ON THAT BOX.
`pip install sentence-transformers` resolves its own `transformers` version, and
BLT is pinned to 5.4.0 (5.15.0 fails BLT outright). A running process keeps the
modules it already imported, so an upgrade mid-run is survivable -- but a BLT
shard that is later RESTARTED, for a crash or a resume, would come back on an
untested transformers. The window where that is invisible is exactly the window
where a box looks fine. So: BLT proc must be 0 on this box before anything is
installed.

AND IT DOES NOT DESTROY ANYTHING. The BLT output stays in /workspace/blt
untouched; bge writes to /workspace/bge. Teardown is a separate decision made
against byte-level verification, not something a launcher should do implicitly.
"""
import argparse, json, os, subprocess, sys

BOXES_IN = "data/blt_fleet/instances.json"
BOXES_OUT = "data/bge_fleet/instances.json"
POLICY = "refuse"          # lacan [5955]
DEPS = "sentence-transformers nltk stanza"


def ssh(b, cmd, timeout=1800):
    return subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
         "-o", "StrictHostKeyChecking=no", "-p", str(b["port"]),
         "root@%s" % b["host"], cmd],
        capture_output=True, text=True, timeout=timeout)


def blt_done(b):
    """(is_done, detail). A box is done when its BLT process is GONE."""
    r = ssh(b, "pgrep -fc 'blt_clou[d]'; cat /workspace/blt/blt_shard*.jsonl "
               "2>/dev/null | wc -l", timeout=120)
    if r.returncode != 0:
        return False, "unreachable (rc=%d) -- NOT grounds to conclude anything" % r.returncode
    parts = [x for x in r.stdout.split() if x.strip()]
    if len(parts) < 2:
        return False, "unparsed probe output %r" % r.stdout[:60]
    proc, rows = int(parts[0]), int(parts[1])
    return proc == 0, "blt proc=%d, %s rows written" % (proc, f"{rows:,}")


def launch(b, of, dry):
    ok, detail = blt_done(b)
    print("  shard %s (%s): %s" % (b["shard"], b["id"], detail))
    if not ok:
        print("     REFUSING: BLT still live here. Installing deps now could "
              "resolve a transformers version BLT has never been tested on, "
              "and it would only surface on a restart.")
        return None
    if dry:
        print("     dry-run: would install deps, ship bge_cloud.py, launch "
              "shard %s/%s --mixed-policy %s" % (b["shard"], of, POLICY))
        return None

    r = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
         "-P", str(b["port"]), "scripts/bge_cloud.py",
         "root@%s:/workspace/bge_cloud.py" % b["host"]],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("     scp failed rc=%d %s" % (r.returncode, r.stderr.strip()[:120]))
        return None

    #: Install, then WARM THE SPLITTERS AND THE MODEL in the same command, so a
    #: missing stanza model or an unresolvable dep fails HERE, visibly, rather
    #: than 200 passages into a run that reports its normal rate. bge_cloud's
    #: own warm() asserts on probe output; this just makes it happen at launch.
    cmd = ("cd /workspace && pip install -q %s 2>&1 | tail -3 && "
           "tmux kill-session -t bge 2>/dev/null; "
           "tmux new-session -d -s bge "
           "'python3 /workspace/bge_cloud.py --input /workspace/blt_passages.jsonl.gz "
           "--out /workspace/bge --shard %s --of %s --mixed-policy %s "
           "> /workspace/bge.log 2>&1'; sleep 2; pgrep -fc 'bge_clou[d]'"
           % (DEPS, b["shard"], of, POLICY))
    r = ssh(b, cmd)
    print("     %s" % (r.stdout.strip()[-200:] or r.stderr.strip()[-200:]))
    rec = dict(b)
    rec["purpose"] = "bge-m3 sentence embeddings, lacan commission [5896]"
    rec["mixed_policy"] = POLICY
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--all-idle", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    boxes = json.load(open(BOXES_IN))
    of = len(boxes)
    targets = boxes if a.all_idle else [b for b in boxes if b["shard"] == a.shard]
    if not targets:
        sys.exit("  no box matches --shard %s" % a.shard)

    existing = []
    if os.path.exists(BOXES_OUT):
        existing = json.load(open(BOXES_OUT))
    known = {r["id"] for r in existing}

    for b in targets:
        rec = launch(b, of, a.dry_run)
        if rec and rec["id"] not in known:
            existing.append(rec); known.add(rec["id"])
    if existing and not a.dry_run:
        os.makedirs(os.path.dirname(BOXES_OUT), exist_ok=True)
        json.dump(existing, open(BOXES_OUT, "w"), indent=1)
        print("  %s now lists %d box(es)" % (BOXES_OUT, len(existing)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
