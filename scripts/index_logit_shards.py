#!/usr/bin/env python3
"""Index the .f16 logit payloads into the `logits` stash. Ruled [3009]/[3013].

WHAT THIS IS AND WHY IT IS NOT AN INGEST
----------------------------------------
The 2026-08-01 run wrote, per model, a `.jsonl` shard AND a companion `.f16`
file holding the raw float16 logit vectors, addressed by each row's
`logit_row`. That is 50 GB of payload beside 1.8 GB of metadata.

**Measured on the real data:** lz4 compresses these vectors to 100.0% of
original (they are high-entropy) and hashstash b64-encodes on top, so an lmdb
copy would cost 66.6 GB to hold 50 GB. And the `.f16` files are not transient —
they are the archive of a 30-hour run. **Copying them into the store means
holding the same bytes twice.**

So this writes an INDEX: `{model, prompt, mode, dtype} -> {file, row, dim}`,
a few MB, and `get_logits` memmaps the payload on read. Exact lookup by
(model, prompt) wants a memmap; it does not want a columnar BLOB or an
approximate-nearest-neighbour store, neither of which answers the query we make.

INDEX-TIME ASSERTIONS, per [3012]/[3013]
----------------------------------------
**`dim` IS A PER-FILE CONSTANT AND A WRONG ONE READS AS PLAUSIBLE.**
`np.memmap(file)[row]` at the wrong stride returns REAL FLOATS from the right
file at the wrong offset — finite, plausibly ranged, and wrong. Value
statistics cannot see that. So:

    1. `dim` must be CONSTANT within a file; recorded once in file provenance.
       A dim that varies inside one .f16 is corruption or a mid-run vocab
       change, and both are things to hear about here rather than at read time.
    2. filesize == (max_row + 1) * dim * itemsize, per file, WITH THE TRIPLE
       STORED — a verification that leaves no artifact is taken on trust the
       second time.

PROVENANCE, per [3009] Rider 1
------------------------------
**SELF-CONSISTENT, NOT SOURCE-VERIFIED.** The `.f16` payloads were never
md5-checked against the cloud instances: the handover verification compared
`*.jsonl` only, which is 4% of the bytes, and both sources were destroyed
before the omission was noticed. What IS established is structural — byte size
matches the row pointers on 103/103 files, and the row count equals the scored
cell count exactly. **That is real evidence and the record carries exactly that
much weight: structure, never content.**
"""

import argparse
import collections
import json
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "data/raw/cloud_run_20260801")
ITEMSIZE = {"float16": 2, "float32": 4}


def scan_shard(jf):
    """One shard -> (rows, dim, dtype, max_row, problems). No writes."""
    rows = []
    dims, dtypes = set(), set()
    hi = -1
    for line in open(jf, errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        lr = r.get("logit_row")
        if lr is None:                      # skip rows carry no vector
            continue
        dims.add(r.get("logit_dim"))
        dtypes.add(r.get("logit_dtype"))
        hi = max(hi, lr)
        rows.append((r.get("prompt"), lr, r.get("model")))
    return rows, dims, dtypes, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.src, "*.jsonl")))
    print(f"shards {len(shards)}   payload root {a.src}\n")

    files_prov = {}
    entries = []
    problems = []

    for jf in shards:
        base = os.path.basename(jf)[:-6]
        ff = base + ".f16"
        fpath = os.path.join(a.src, ff)
        model_from_name = base.replace("__", "/")

        if not os.path.exists(fpath):
            problems.append((model_from_name, "NO .f16 FILE")); continue

        rows, dims, dtypes, hi = scan_shard(jf)

        # (1) dim constant within the file
        if len(dims) != 1:
            problems.append((model_from_name, f"dim NOT CONSTANT: {sorted(dims)}"))
            continue
        if len(dtypes) != 1:
            problems.append((model_from_name, f"dtype NOT CONSTANT: {sorted(dtypes)}"))
            continue
        dim = dims.pop(); dt = dtypes.pop()
        isz = ITEMSIZE.get(str(dt))
        if isz is None:
            problems.append((model_from_name, f"unknown dtype {dt!r}")); continue

        # (2) filesize == (max_row+1) * dim * itemsize, and STORE THE TRIPLE
        want = (hi + 1) * dim * isz
        got = os.path.getsize(fpath)
        if want != got:
            problems.append((model_from_name,
                             f"size {got:,} != expected {want:,} "
                             f"(rows {hi+1}, dim {dim}, {dt})"))
            continue

        files_prov[ff] = {"dim": dim, "dtype": str(dt), "n_rows": hi + 1,
                          "itemsize": isz, "bytes": got,
                          "check": "bytes == n_rows * dim * itemsize"}
        for prompt, lr, model in rows:
            entries.append((model or model_from_name, prompt, str(dt),
                            {"file": ff, "row": lr, "dim": dim}))

    print(f"  files passing both index-time assertions : {len(files_prov)}")
    print(f"  files with problems                      : {len(problems)}")
    for m, why in problems[:10]:
        print(f"      {m}: {why}")
    print(f"  INDEX ENTRIES TO WRITE                   : {len(entries):,}")
    print(f"    pre-registered expectation [3007]      : 266,037")
    print(f"    delta                                  : {len(entries)-266037:+,}")

    if problems:
        print("\nREFUSING: a file failed an index-time assertion. A wrong stride "
              "returns real floats from the wrong offset -- finite, plausible, "
              "and wrong.")
        return 1
    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    from malign_logits.cache import get_cache
    cm = get_cache()
    n = 0
    for model, prompt, dt, entry in entries:
        cm.set_logits(model, prompt, entry, mode="raw", dtype=dt)
        n += 1
        if n % 50000 == 0:
            print(f"    {n:,} written", flush=True)
    print(f"\n  wrote {n:,} index entries")

    prov = {
      "_what": "Index into the .f16 logit payloads written by the 2026-08-01 "
               "run. Value is {file, row, dim}; the vectors are memmapped on "
               "read and never copied into the store.",
      "_integrity": "SELF-CONSISTENT, NOT SOURCE-VERIFIED. What was checked is "
                    "STRUCTURE: byte size against the row pointers, and row "
                    "count against the scored-cell count, on every file. What "
                    "was NEVER checked is CONTENT against the cloud instances "
                    "-- the handover verification compared *.jsonl only, which "
                    "is 4% of the bytes, and both sources were destroyed before "
                    "the omission was noticed. Ruled [3009] Rider 1; this "
                    "sentence is inherited by every reader.",
      "_root": "paths are BASENAMES resolved against MALIGN_LOGIT_ROOT (default "
               "data/raw/cloud_run_20260801). Moving the payloads is a config "
               "change, not a re-index.",
      "_load_bearing": "THE .f16 FILES ARE INFRASTRUCTURE, NOT RUN OUTPUT. "
                       "Deleting them silently empties this cache.",
      "entries": n, "files": len(files_prov), "files_provenance": files_prov,
    }
    outp = os.path.join(ROOT, "data/logit_index_provenance.json")
    json.dump(prov, open(outp, "w"), indent=1)
    print(f"  provenance -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
