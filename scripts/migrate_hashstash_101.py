#!/usr/bin/env python3
"""One-shot: migrate all stashes to the hashstash 1.0.1 key format, retire psyche_derived.

Background: hashstash 1.0 changed the lmdb key-to-address encoding, so stashes
written under 0.4.0 enumerate but cannot be read on 1.0+. 1.0.1's migrate()
reads the raw stored entries and re-stores them under the new encoding.
Validated 2026-07-05 on the real generations stash: 213,081/213,081, 0 failed.

psyche_derived is NOT migrated: every one of its 4,718 entries is shadowed by
the typed stashes on the read path (verified 2026-07-05, 0 uncovered), so it is
moved to psyche_derived.old and dropped with the other .old dirs.

Run AFTER upgrading:  uv pip install 'hashstash[rec]>=1.0.1'
(and bump the pin in requirements.txt in the same commit as this run's results).

Per stash under data/raw/cache/:
  1. migrate(dry_run=True)  — count recoverable entries, nothing written
  2. migrate into <name>.new (string dest inherits the lz4+b64 layout)
  3. verify: dest count == dry-run count, plus value spot-checks vs legacy reads
  4. swap:  <name> -> <name>.old,  <name>.new -> <name>
  5. carry non-stash sibling files across (e.g. probe_embeddings/*.npy)

Old data is kept as <name>.old until you delete it; the script prints one
cleanup command at the end. Stash-by-stash keeps peak extra disk to roughly
the largest single stash (~48GB, sent_embeddings) against ~162GB free.

Usage:
    python scripts/migrate_hashstash_101.py --dry-run          # counts only
    python scripts/migrate_hashstash_101.py                    # full run
    python scripts/migrate_hashstash_101.py --only generations # one stash
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RETIRED = {"psyche_derived"}  # fully shadowed by typed stashes; drop, don't migrate
SPOT_CHECKS = 5


def _deep_eq(a, b):
    """Value equality that survives NaN (nan != nan) and numpy arrays nested
    in dicts/lists (== raises or is elementwise there). Both shapes occur in
    real stashes: mega_generations has NaN token fields, reasoning_logits has
    arrays inside dicts."""
    import numpy as np
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and np.array_equal(a, b, equal_nan=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b) or repr(a) == repr(b)  # repr: nan == nan
    except Exception:
        return repr(a) == repr(b)


def stash_dirs(cache_root):
    for name in sorted(os.listdir(cache_root)):
        path = os.path.join(cache_root, name)
        if not os.path.isdir(path) or name.endswith((".old", ".new")):
            continue
        yield name, path


def sibling_files(path):
    """Non-stash files living beside the lmdb dir (e.g. probe_embeddings/*.npy)."""
    return [f for f in os.listdir(path)
            if not f.startswith("lmdb.hashstash") and not f.startswith("pairtree.hashstash")]


def check_disk(path, cache_root):
    need = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs)
    free = shutil.disk_usage(cache_root).free
    if free < need * 1.5:
        raise RuntimeError(f"{path}: need ~{need/1e9:.1f}GB x1.5, only {free/1e9:.1f}GB free")


def migrate_one(name, path, dry_run=False):
    from malign_logits.cache import open_stash

    src = open_stash(path)
    t0 = time.time()
    dry = src.migrate(dest=None, dry_run=True)
    print(f"  dry-run: total={dry['total']} migrated={dry['migrated']} "
          f"failed={dry['failed']} ({time.time()-t0:.0f}s)")
    if dry_run:
        return dry
    if dry["failed"]:
        raise RuntimeError(f"{name}: {dry['failed']} unreadable entries — investigate before migrating")
    if dry["total"] == 0:
        print("  empty stash — skipping")
        return dry

    check_disk(path, os.path.dirname(path))
    new_path = path + ".new"
    if os.path.exists(new_path):
        raise RuntimeError(f"{new_path} already exists — clean up a previous run first")

    t0 = time.time()
    res = src.migrate(dest=new_path, dry_run=False)
    print(f"  migrate: migrated={res['migrated']} failed={res['failed']} "
          f"first_error={res.get('first_error')} ({time.time()-t0:.0f}s)")
    if res["failed"] or res["migrated"] != dry["migrated"]:
        raise RuntimeError(f"{name}: migrate mismatch (dry {dry['migrated']} vs real {res['migrated']}, "
                           f"failed {res['failed']}) — old stash left untouched")

    dest = open_stash(new_path)
    if len(dest) == 0:
        raise RuntimeError(f"{name}: migrated stash reads empty")
    legacy = open_stash(path, legacy_read=True)
    checked = 0
    for k in dest.keys():
        if not _deep_eq(legacy[k], dest[k]):
            raise RuntimeError(f"{name}: value mismatch on {k}")
        checked += 1
        if checked >= SPOT_CHECKS:
            break
    print(f"  verify: len={len(dest)}, {checked} value spot-checks OK")

    for f in sibling_files(path):
        shutil.move(os.path.join(path, f), os.path.join(new_path, f))
        print(f"  carried sibling file: {f}")

    os.rename(path, path + ".old")
    os.rename(new_path, path)
    print(f"  swapped: {name}.old kept for manual cleanup")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="counts only, no writes")
    ap.add_argument("--only", help="migrate a single stash by name")
    args = ap.parse_args()

    import hashstash
    ver = tuple(int(x) for x in hashstash.__version__.split(".")[:3])
    if ver < (1, 0, 1):
        sys.exit(f"hashstash {hashstash.__version__} installed — this script requires >=1.0.1")

    from malign_logits.cache import CACHE_ROOT

    report = {}
    for name, path in stash_dirs(CACHE_ROOT):
        if args.only and name != args.only:
            continue
        if name in RETIRED:
            if args.dry_run:
                print(f"\n[{name}] RETIRED — would move to {name}.old (not migrated)")
            else:
                os.rename(path, path + ".old")
                print(f"\n[{name}] RETIRED — moved to {name}.old (fully shadowed by typed stashes)")
            report[name] = "retired"
            continue
        print(f"\n[{name}]")
        try:
            res = migrate_one(name, path, dry_run=args.dry_run)
            report[name] = f"{res['migrated']} entries"
        except Exception as e:
            report[name] = f"FAILED: {e}"
            print(f"  !! {e}")

    print("\n=== Summary ===")
    for name, status in report.items():
        print(f"  {name:22s} {status}")
    if not args.dry_run:
        print("\nAfter checking the summary, reclaim space with:")
        print(f"  rm -rf {CACHE_ROOT}/*.old")


if __name__ == "__main__":
    main()
