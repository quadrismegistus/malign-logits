#!/usr/bin/env python3
"""Merge the verse fleet's per-instance output into one declarable source.

    scripts/verse_fleet_merge.py [--apply]

WHY A MERGE AT ALL. `ch_ingest.ingest_twp` globs `<source>/*.jsonl` -- FLAT, one
level. The fleet wrote to `data/raw/verse_fleet/<instance_id>/`, so a source
pointing at the parent sees nothing and a source per instance is eight
declarations for one study.

WHY IT MUST SELECT ON ROW COUNT. The tree carries debris from the night the
fleet ran: a duplicate assignment briefly had two boxes on the same 15 models,
leaving partial files behind, and a box that died mid-write left a 959-row
`step8000`. Every complete model in this corpus is EXACTLY 1,820 rows. Taking
the first file a walk finds would ingest a 79-row partial over a complete rung
-- the same shape that cost `opening_matched` its construction ([5874]).

WHY HARD LINKS. Same filesystem, so they cost zero bytes and are real directory
entries the ingest's glob resolves. Symlinks are NOT safe here: rglob does not
follow them, which this campaign has been bitten by before.

The .f16 tier is NOT merged -- it is 63 GB and reaches ClickHouse through
`ingest_logits_indexed`, keyed by the pointers inside the jsonl.
"""
import os, sys, json, glob, collections

SRC  = "data/raw/verse_fleet"
DST  = "data/raw/verse_fleet_merged"
ROWS = 1820
SPECS = "data/verse_fleet/specs"


def declared():
    out = set()
    for s in sorted(glob.glob(os.path.join(SPECS, "shard*.json"))):
        for m in json.load(open(s)):
            out.add((m if isinstance(m, str) else m["model"]).replace("/", "__"))
    return out


def main(argv):
    apply = "--apply" in argv
    want = declared()
    cand = collections.defaultdict(list)          # rung -> [(rows, path)]
    for p in glob.glob(os.path.join(SRC, "*", "*.jsonl")):
        if "_SUPERSEDED" in p:
            continue
        with open(p, "rb") as fh:
            n = sum(1 for _ in fh)
        cand[os.path.basename(p)[:-6]].append((n, p))

    chosen, partial, dupes, missing = {}, [], [], []
    for rung in sorted(want):
        opts = cand.get(rung, [])
        full = [(n, p) for n, p in opts if n >= ROWS]
        if not full:
            missing.append((rung, [n for n, _ in opts]))
            continue
        if len(full) > 1:
            dupes.append((rung, [os.path.basename(os.path.dirname(p)) for _, p in full]))
        chosen[rung] = max(full)[1]
    for rung, opts in cand.items():
        for n, p in opts:
            if n < ROWS:
                partial.append((rung, n, os.path.basename(os.path.dirname(p))))
    extra = sorted(set(cand) - want)

    print("  declared rungs        %d" % len(want))
    print("  complete and chosen   %d" % len(chosen))
    print("  MISSING               %d %s" % (len(missing), missing or ""))
    print("  partial files skipped %d" % len(partial))
    for r, n, d in partial:
        print("      %-46s %5d rows  (%s)" % (r[:46], n, d))
    print("  rungs with >1 complete copy: %d" % len(dupes))
    for r, ds in dupes[:5]:
        print("      %-46s %s" % (r[:46], ds))
    print("  files not in the declared roster: %d %s" % (len(extra), extra or ""))

    if not apply:
        print("\n  dry run. re-run with --apply to hard-link into %s" % DST)
        return 0
    if missing:
        print("\n  REFUSING: %d declared rungs have no complete file" % len(missing))
        return 1
    os.makedirs(DST, exist_ok=True)
    made = 0
    for rung, src in chosen.items():
        dst = os.path.join(DST, rung + ".jsonl")
        if os.path.exists(dst):
            os.unlink(dst)
        os.link(src, dst)
        made += 1
    total = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST))
    st = os.statvfs(".")
    print("\n  hard-linked %d rungs into %s" % (made, DST))
    print("  apparent size %.2f GB, ACTUAL disk cost 0 bytes (hard links)" % (total / 1e9))
    print("  local free unchanged at %.2f GB" % (st.f_bavail * st.f_frsize / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
