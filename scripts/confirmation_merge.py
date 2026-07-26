#!/usr/bin/env python3
"""Move confirmation-census generations from the cloud stash into the local one.

Two halves, because syncing lmdb files between machines is riskier and far
larger than moving the payload: 14,768 generations is ~12 MB of JSONL against a
multi-GB stash directory, and a partial rsync of an lmdb environment can leave a
corrupt store rather than a short one.

    export   (run on the cloud box)  stash -> JSONL, driven by the manifest
    import   (run locally)           JSONL -> stash, with the collision guard

THE COLLISION GUARD BELONGS HERE, NOT ON THE CLOUD. On a fresh remote stash
there is nothing to collide with, so the guard in confirmation_generate.py is
vacuous there by construction -- it can only protect the store that already
holds the census. This is the real check, and it runs before any write.

Usage:
    # on cloud
    python scripts/confirmation_merge.py export --out /workspace/cloud_gens.jsonl
    # locally, after scp
    python scripts/confirmation_merge.py import --in data/cloud_gens.jsonl --dry-run
    python scripts/confirmation_merge.py import --in data/cloud_gens.jsonl --go
"""
import argparse, csv, json, os, sys, collections

from malign_logits.cache import get_cache

MANIFEST = "data/confirmation_generation_manifest.csv"


def manifest_rows(run_ids=None, backend=None):
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if run_ids and r["run_id"] not in run_ids:
                continue
            if backend and r["backend"] != backend:
                continue
            yield r


def cmd_export(a):
    cm = get_cache()
    n, miss = 0, 0
    with open(a.out, "w") as fh:
        for r in manifest_rows(set(a.run_ids.split(",")) if a.run_ids else None,
                               a.backend):
            txt = cm.get_generation(r["model"], r["prompt"],
                                    temp=float(r["temp"]), idx=int(r["idx"]))
            if txt is None:
                miss += 1
                continue
            fh.write(json.dumps({k: r[k] for k in
                                 ("run_id", "backend", "family", "role", "model",
                                  "prompt", "temp", "idx")} | {"text": txt}) + "\n")
            n += 1
    print(f"exported {n} generations -> {a.out}")
    if miss:
        print(f"  WARNING: {miss} manifest rows had no stash entry", file=sys.stderr)


def cmd_import(a):
    cm = get_cache()
    rows = [json.loads(l) for l in open(a.inp)]
    by_arm = collections.Counter((r["family"], r["role"]) for r in rows)

    collide = [r for r in rows
               if cm.get_generation(r["model"], r["prompt"],
                                    temp=float(r["temp"]), idx=int(r["idx"])) is not None]
    print(f"{len(rows)} incoming generations across {len(by_arm)} arms")
    for k in sorted(by_arm):
        print(f"  {k[0]:16s} {k[1]:8s} {by_arm[k]}")
    print(f"\ncollisions with the LOCAL stash: {len(collide)}")
    if collide:
        c = collide[0]
        print(f"  e.g. {c['model']} idx={c['idx']} prompt={c['prompt'][:40]!r}")
        print("  ABORT: a key already exists locally, so provenance would be "
              "ambiguous. Resolve before importing.")
        return 1
    if not a.go:
        print("\nHOLDING. Re-run with --go to write.")
        return 0
    for r in rows:
        cm.set_generation(r["model"], r["prompt"], r["text"],
                          temp=float(r["temp"]), idx=int(r["idx"]))
    print(f"\nimported {len(rows)} generations into the local stash")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.set_defaults(fn=cmd_export)
    e.add_argument("--out", required=True)
    e.add_argument("--run-ids", help="comma-separated; default all")
    e.add_argument("--backend", default="vllm")
    i = sub.add_parser("import"); i.set_defaults(fn=cmd_import)
    i.add_argument("--in", dest="inp", required=True)
    i.add_argument("--go", action="store_true")
    i.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
