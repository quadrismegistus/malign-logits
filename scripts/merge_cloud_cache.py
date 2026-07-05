#!/usr/bin/env python3
"""Merge cloud-generated cache into local cache without overwriting.

Usage:
    # 1. Download cloud cache to temp dir
    rsync -avz -e "ssh -p PORT" root@HOST:/workspace/malign-logits/data/raw/cache/trees/ /tmp/cloud_cache/

    # 2. Merge into local
    python scripts/merge_cloud_cache.py /tmp/cloud_cache

    # Or one-liner with custom local path:
    python scripts/merge_cloud_cache.py /tmp/cloud_cache --local data/raw/cache/trees
"""

import argparse
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Merge cloud cache into local cache")
    parser.add_argument("cloud_path", help="Path to downloaded cloud cache directory")
    parser.add_argument("--local", default=None,
                        help="Local cache path (default: project's trees stash)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count entries without writing")
    args = parser.parse_args()

    from malign_logits.cache import open_stash

    cloud = open_stash(args.cloud_path)

    if args.local:
        local_path = args.local
    else:
        local_path = str(Path(__file__).parent.parent / "data" / "raw" / "cache" / "trees")

    local = open_stash(local_path)

    cloud_keys = list(cloud.keys())
    local_keys = set(local.keys())

    new = [k for k in cloud_keys if k not in local_keys]
    existing = [k for k in cloud_keys if k in local_keys]

    print(f"Cloud:    {len(cloud_keys)} entries")
    print(f"Local:    {len(local_keys)} entries")
    print(f"New:      {len(new)} (will add)")
    print(f"Overlap:  {len(existing)} (will overwrite with cloud version)")

    if args.dry_run:
        print("Dry run — no changes made.")
        return

    merged = 0
    for k in tqdm(cloud_keys, desc="Merging", unit="entry"):
        try:
            local[k] = cloud[k]
            merged += 1
        except Exception as e:
            tqdm.write(f"  Skip: {str(e)[:60]}")

    print(f"Merged {merged}/{len(cloud_keys)} entries")
    print(f"Local now has {len(list(local.keys()))} entries")


if __name__ == "__main__":
    main()
