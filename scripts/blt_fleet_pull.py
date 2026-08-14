#!/usr/bin/env python3
"""Incremental pull of BLT shard output. Run every cycle, not at the end.

The boxes write as they go (append + flush every 200 passages), so a pull at
any moment gets everything scored so far. What makes that worth doing on a
LOOP rather than at the end is last night: a box was lost 4.1 h into an 8 h job
and everything already pulled survived, everything not pulled did not.

rsync --append-verify is correct here and --append is NOT: both files are
strictly append-only on the box, but --append trusts that the common prefix
matches without checking, and a partial write on the far side would then be
silently stitched onto a good local prefix. --append-verify re-reads the
overlap. Costs a checksum of what we already hold; buys knowing it is the same
bytes.
"""
import json, os, subprocess, sys, concurrent.futures as cf

DEST = "data/raw/blt_fleet"


def pull(b):
    d = os.path.join(DEST, str(b['id']))
    os.makedirs(d, exist_ok=True)
    cmd = ['rsync', '-a', '--append-verify', '--partial',
           '-e', 'ssh -p %s -o StrictHostKeyChecking=no -o BatchMode=yes '
                 '-o ConnectTimeout=15' % b['port'],
           'root@%s:/workspace/blt/' % b['host'], d + '/']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        return (b['shard'], b['id'], 'ERR %s' % type(e).__name__, 0, 0)
    rows = 0
    for f in os.listdir(d):
        if f.endswith('.jsonl'):
            with open(os.path.join(d, f), 'rb') as fh:
                rows += sum(1 for _ in fh)
    sz = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    return (b['shard'], b['id'], 'ok' if r.returncode == 0 else 'rsync rc=%d' % r.returncode,
            rows, sz)


def main():
    #: Destroyed boxes stay in the record for provenance but are not
    #: pulled from -- rsync to a dead host is a 15s timeout per cycle.
    boxes = [b for b in json.load(open("data/blt_fleet/instances.json"))
             if not b.get("destroyed")]
    with cf.ThreadPoolExecutor(len(boxes)) as ex:
        res = sorted(ex.map(pull, boxes))
    tot_rows = 0
    for s, i, st, rows, sz in res:
        tot_rows += rows
        print("  shard %d  %-10s %-12s %9s rows  %6.2f GB" % (s, i, st, f"{rows:,}", sz/1e9))
    #: MEASURE THE DISK, not the sum of what each pull reported -- a failed pull
    #: reporting 0 for data that is on disk is how TOTAL LOCAL went backwards
    #: last night ([5885]).
    disk = 0
    for root_, _d, fs in os.walk(DEST):
        for f in fs:
            try: disk += os.path.getsize(os.path.join(root_, f))
            except OSError: pass
    print("  HELD LOCALLY %s rows, %.2f GB (measured on disk)" % (f"{tot_rows:,}", disk/1e9))
    st = os.statvfs('.')
    #: BOTH UNITS. A "keep local free above 20 GB" floor is unit-dependent at
    #: exactly the level where it fires: 21.15 GB decimal and 19.70 GiB binary
    #: are the same disk, and df prints the second while this printed only the
    #: first. Reporting one unit lets the floor read CLEAR and BREACHED
    #: simultaneously depending on who checks it.
    free = st.f_bavail * st.f_frsize
    print("  local free %.1f GB / %.1f GiB (df prints GiB)"
          % (free / 1e9, free / 2 ** 30))
    return 0


if __name__ == '__main__':
    sys.exit(main())
