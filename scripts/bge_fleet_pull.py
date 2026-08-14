#!/usr/bin/env python3
"""Incremental pull of bge shard output. Run every cycle, not at the end.

Sibling of `blt_fleet_pull.py`, deliberately a SEPARATE FILE rather than a flag
on that one: the BLT fleet is live while this is being written, and a shared
tool edited mid-run costs visibility on the fleet it is monitoring.

Same reasoning as the BLT puller, which applies unchanged:

rsync --append-verify is correct here and --append is NOT: both files are
strictly append-only on the box, but --append trusts that the common prefix
matches without checking, and a partial write on the far side would then be
silently stitched onto a good local prefix. --append-verify re-reads the
overlap.

ONE DIFFERENCE THAT MATTERS. bge writes a `.manifest.json` that is REWRITTEN
each run, not appended -- it carries the `_about` fence and the running counts.
`--append-verify` on a rewritten file is wrong: it would treat the new shorter
content as an extension of the old. The manifest is therefore pulled in a
second pass with plain `-a` (size+mtime, whole-file replace), and the
append-verify pass excludes it.
"""
import json, os, subprocess, sys, concurrent.futures as cf

#: --meta-only pulls the .jsonl and .manifest.json but NOT the .f32 sidecar.
#: The vectors are ~98.5% of the volume (42.6 KB/passage against ~300 bytes of
#: jsonl), and the full run projects to 19.2 GB against 20.6 GB of local free
#: space -- so the sidecar cannot land here until a destination is settled.
#: The metadata still travels every cycle, because it is the part that says
#: WHICH passages were embedded under WHICH splitter with how many sentences,
#: and it is not reconstructible from a dead box. The vectors are.
META_ONLY = "--meta-only" in sys.argv

DEST = "data/raw/bge_fleet"
BOXES = "data/bge_fleet/instances.json"
REMOTE = "/workspace/bge/"


def _ssh(b):
    return ('ssh -p %s -o StrictHostKeyChecking=no -o BatchMode=yes '
            '-o ConnectTimeout=15' % b['port'])


def pull(b):
    d = os.path.join(DEST, str(b['id']))
    os.makedirs(d, exist_ok=True)
    rc = 0
    #: PASS 1: append-only artifacts (.jsonl, .f32, .refused.jsonl).
    a = ['rsync', '-a', '--append-verify', '--partial', '--exclude', '*.manifest.json']
    if META_ONLY:
        a += ['--exclude', '*.f32']
    a += ['-e', _ssh(b), 'root@%s:%s' % (b['host'], REMOTE), d + '/']
    #: PASS 2: the manifest, which is REWRITTEN and must not be appended to.
    m = ['rsync', '-a', '--include', '*.manifest.json', '--include', '*/',
         '--exclude', '*', '-e', _ssh(b),
         'root@%s:%s' % (b['host'], REMOTE), d + '/']
    for cmd in (a, m):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            rc = rc or r.returncode
        except Exception as e:
            return (b['shard'], b['id'], 'ERR %s' % type(e).__name__, 0, 0, 0)

    #: Count the EMBEDDED rows only. `.refused.jsonl` is also .jsonl and holds
    #: the mixed stratum under `refuse` -- counting it would inflate progress
    #: with passages that were deliberately not embedded.
    rows = sent = 0
    for f in os.listdir(d):
        if f.endswith('.jsonl') and not f.endswith('.refused.jsonl'):
            with open(os.path.join(d, f)) as fh:
                for line in fh:
                    rows += 1
                    try:
                        sent += json.loads(line)["n_sentences"]
                    except Exception:
                        pass
    sz = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    return (b['shard'], b['id'], 'ok' if rc == 0 else 'rsync rc=%d' % rc, rows, sent, sz)


def main():
    if not os.path.exists(BOXES):
        print("  no %s yet -- bge fleet not launched" % BOXES)
        return 0
    boxes = json.load(open(BOXES))
    with cf.ThreadPoolExecutor(max(len(boxes), 1)) as ex:
        res = sorted(ex.map(pull, boxes))
    tot_rows = tot_sent = 0
    for s, i, st, rows, sent, sz in res:
        tot_rows += rows; tot_sent += sent
        print("  shard %d  %-10s %-12s %9s passages  %10s sentences  %6.2f GB"
              % (s, i, st, f"{rows:,}", f"{sent:,}", sz / 1e9))
    #: MEASURE THE DISK, not the sum of what each pull reported -- a failed pull
    #: reporting 0 for data that is on disk is how TOTAL LOCAL went backwards
    #: ([5885]).
    disk = 0
    for root_, _d, fs in os.walk(DEST):
        for f in fs:
            try: disk += os.path.getsize(os.path.join(root_, f))
            except OSError: pass
    print("  HELD LOCALLY %s passages, %s sentences, %.2f GB (measured on disk)"
          % (f"{tot_rows:,}", f"{tot_sent:,}", disk / 1e9))
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
