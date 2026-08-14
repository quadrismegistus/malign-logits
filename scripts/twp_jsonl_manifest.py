"""A manifest of every twp-shaped .jsonl cell on disk, with its provenance.

    uv run python scripts/twp_jsonl_manifest.py            # build
    uv run python scripts/twp_jsonl_manifest.py --report   # read it back

    -> data/twp_jsonl_manifest.parquet   (one row per LINE, not per cell)
    -> data/twp_jsonl_manifest.json      (_about + the summary tables)

WHY. RH asked for this after a redundancy audit produced a table that was
wrong in every row. I attributed each cell to the FIRST directory a scan
happened to meet it in, which made `data/twp_cloud` read as "4,738 cells,
88.4% un-ingested" when it actually holds 34,198 cells of which 4,744 are
un-ingested. **The 88.4% was a property of my glob order, not of the
directory.** A path column removes the inference entirely: provenance is
recorded, not derived from the order something was walked in.

THE UNIT IS THE LINE, DELIBERATELY. One (model, prompt) cell can appear in
several files -- that is the redundancy this exists to find -- so collapsing
to the cell here would destroy the thing being looked for. `n_paths` in the
report is the count that answers "is this duplicated"; the parquet keeps
every occurrence with the file it came from.

COLUMNS
    model, prompt            the cell
    path, dir, line          where it is, exactly
    theta, rule_version      the run's declared parameters
    dict_sha, revision       and its identity
    n_words                  len(rows) -- the twp payload size
    tail, drop, open         residual mass, whatever the record carries
    conservation             the run's own accounting figure
    logit_row, hidden_row    pointers into the .f16 / .hidden.f32 tiers,
                             so a reader can tell which cells have a binary
                             sidecar and which are twp-only
    mtime, produced          WHEN the file was written, epoch and ISO.

`produced` IS THE FILE'S MTIME AND THAT IS A CLAIM WORTH FENCING. These are
append-only run outputs, so mtime is the production date -- but it is the
date of the LAST write, not the first, and any copy made without preserving
timestamps would reset it. `rsync -a` preserves them and the .hidden.f32
migration used it, so today's dates survived that move. Treat `produced` as
"not written after this", which is what an mtime can actually support.

`in_ch` IS DELIBERATELY NOT A COLUMN. Whether a cell is in ClickHouse is a
statement about a moment, and a column would freeze one moment into an
artifact that outlives it -- tonight's own rule about verification. The
report joins against CH live and prints the answer with a timestamp instead.
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_PQ = os.path.join(ROOT, "data/twp_jsonl_manifest.parquet")
OUT_JS = os.path.join(ROOT, "data/twp_jsonl_manifest.json")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")


def is_twp(path):
    """A twp-shaped file declares rows + theta + model + prompt on line 1."""
    try:
        with open(path, "rb") as f:
            r = json.loads(f.readline())
    except Exception:
        return False
    return all(k in r for k in ("rows", "theta", "model", "prompt"))


def build():
    import pandas as pd

    files = sorted(p for p in glob.glob(os.path.join(ROOT, "data/**/*.jsonl"),
                                        recursive=True) if is_twp(p))
    print("%d twp-shaped files" % len(files), flush=True)
    rows = []
    t0 = time.time()
    for i, p in enumerate(files):
        rel = os.path.relpath(p, ROOT)
        d = "/".join(rel.split("/")[:3]) if rel.startswith("data/raw") \
            else "/".join(rel.split("/")[:2])
        mt = os.path.getmtime(p)
        with open(p, "rb") as f:
            for ln, raw in enumerate(f, 1):
                try:
                    r = json.loads(raw)
                except Exception:
                    continue
                res = r.get("residual") or {}
                rows.append({
                    "model": r.get("model"), "prompt": r.get("prompt"),
                    "path": rel, "dir": d, "line": ln,
                    "theta": r.get("theta"),
                    "rule_version": r.get("rule_version"),
                    "dict_sha": r.get("dict_sha"),
                    "revision": r.get("revision"),
                    "n_words": len(r.get("rows") or []),
                    "tail": res.get("tail"), "drop": res.get("drop"),
                    "open": res.get("open"),
                    "conservation": r.get("conservation"),
                    "logit_row": r.get("logit_row"),
                    "hidden_row": r.get("hidden_row"),
                    "mtime": mt,
                    "produced": time.strftime("%Y-%m-%d %H:%M",
                                              time.localtime(mt))})
        if (i + 1) % 200 == 0:
            print("  %d/%d files, %s rows, %.0fs"
                  % (i + 1, len(files), format(len(rows), ","),
                     time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PQ, index=False)
    print("\nwrote %s: %s rows" % (os.path.relpath(OUT_PQ, ROOT),
                                   format(len(df), ",")))
    return df


def report(df=None):
    import pandas as pd

    if df is None:
        df = pd.read_parquet(OUT_PQ)
    n_cells = df.groupby(["model", "prompt"]).ngroups
    print("lines %s | distinct cells %s | files %d | dirs %d"
          % (format(len(df), ","), format(n_cells, ","),
             df.path.nunique(), df.dir.nunique()))

    #: REDUNDANCY: the same cell written by more than one file.
    per = df.groupby(["model", "prompt"]).path.nunique()
    dup = per[per > 1]
    print("\nREDUNDANCY: %s cells appear in >1 file (%.1f%% of cells)"
          % (format(len(dup), ","), 100 * len(dup) / max(1, n_cells)))
    if len(dup):
        print("  copies per cell: %s"
              % dict(collections.Counter(dup.values).most_common(5)))
        d2 = df.set_index(["model", "prompt"]).loc[dup.index]
        pair = collections.Counter()
        for _, g in d2.groupby(level=[0, 1]):
            ds = tuple(sorted(set(g.dir)))
            if len(ds) > 1:
                pair[ds] += 1
        print("  most common directory pairs holding the same cell:")
        for ds, n in pair.most_common(6):
            print("    %6d  %s" % (n, " + ".join(ds)))

    #: PER DIRECTORY, with the denominator the earlier audit got wrong.
    print("\n  %-34s %9s %9s %6s %6s  %-11s %-11s"
          % ("dir", "lines", "cells", "files", "GB", "first", "last"))
    for d, g in sorted(df.groupby("dir"), key=lambda kv: -len(kv[1])):
        sz = sum(os.path.getsize(os.path.join(ROOT, p))
                 for p in g.path.unique() if os.path.exists(os.path.join(ROOT, p)))
        print("  %-34s %9s %9s %6d %6.1f  %-11s %-11s"
              % (d, format(len(g), ","),
                 format(g.groupby(["model", "prompt"]).ngroups, ","),
                 g.path.nunique(), sz / 2 ** 30,
                 g.produced.min()[:10], g.produced.max()[:10]))

    #: LIVE join against CH, with its moment stated rather than stored.
    try:
        q = ("SELECT model, prompt FROM malign_logits.twp_words "
             "GROUP BY model, prompt FORMAT JSONEachRow")
        out = subprocess.run([CH, "client", "-q", q], capture_output=True,
                             text=True, timeout=1800).stdout
        ch = {(json.loads(l)["model"], json.loads(l)["prompt"])
              for l in out.split("\n") if l.strip()}
    except Exception as e:
        print("\n  CH unavailable (%s); skipping the ingest join" % type(e).__name__)
        return 0
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cells = set(zip(df.model, df.prompt))
    gap = cells - ch
    print("\nAGAINST CLICKHOUSE, as of %s -- a statement about this moment only"
          % stamp)
    print("  disk cells %s | CH cells %s | on disk NOT in CH %s | in CH NOT on disk %s"
          % (format(len(cells), ","), format(len(ch), ","),
             format(len(gap), ","), format(len(ch - cells), ",")))
    #: attribute the gap by EVERY path holding it, not by first sighting
    hold = collections.Counter()
    g = df[[(m, p) in gap for m, p in zip(df.model, df.prompt)]]
    for d, gg in g.groupby("dir"):
        hold[d] = gg.groupby(["model", "prompt"]).ngroups
    print("\n  un-ingested cells BY EVERY DIRECTORY HOLDING THEM")
    print("  (a cell in two directories is counted under both -- that is the")
    print("   point; first-occurrence attribution is what produced a wrong table)")
    for d, n in hold.most_common():
        tot = df[df.dir == d].groupby(["model", "prompt"]).ngroups
        print("    %-38s %8s of %8s  %5.1f%%"
              % (d, format(n, ","), format(tot, ","), 100 * n / max(1, tot)))
    json.dump({"_about":
               "Manifest of every twp-shaped .jsonl cell on disk, one row per "
               "LINE so redundancy survives. Provenance is RECORDED (path, dir, "
               "line) and never inferred from scan order -- an earlier audit "
               "attributed cells to whichever directory a glob met first and "
               "reported data/twp_cloud as 88.4% un-ingested when the true "
               "figure over its own contents is 13.9%. `in_ch` is NOT stored: "
               "it is a statement about a moment and belongs in a report with a "
               "timestamp, not frozen into an artifact.",
               "built": stamp, "n_lines": int(len(df)), "n_cells": len(cells),
               "n_files": int(df.path.nunique()),
               "ch_cells_at_build": len(ch),
               "disk_not_in_ch": len(gap), "ch_not_on_disk": len(ch - cells),
               "redundant_cells": int(len(dup)),
               "gap_by_dir_all_holders": dict(hold)},
              open(OUT_JS, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT_JS, ROOT))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="read the existing manifest instead of rebuilding")
    a = ap.parse_args()
    return report() if a.report else report(build())


if __name__ == "__main__":
    sys.exit(main())
