#!/usr/bin/env python
"""Recompute rt_band for rows written before the HTML fix. Run AFTER the run.

    python y_repair_rt.py --dry     # what would change, write nothing
    python y_repair_rt.py           # rewrite in place

WHY: `roundtrip()` stripped anything tag-shaped from the coder's reproduction
while comparing against a source that still contained it. Generated passages
carry HTML -- <br>, <p>, <strong>, <i>, <URL>, a <NAME> placeholder -- which the
coder reproduced faithfully and the comparison scored as drift. Measured on the
first 9,975 rows: 51 carry a non-vocabulary tag and 11 of the 40 SEVERE rows
were among them, so 28% of the severe population was the instrument.

REFUSES WHILE THE RUNNER IS ALIVE. The results file is append-only and the
runner holds it open; rewriting underneath would race a writer and could
truncate a chunk. A repair that can corrupt the thing it repairs is worse than
the defect.

REPORTS THE PRE-PATCH NUMBERS. The old band is kept in `rt_band_pre_fix` on any
row that changes, so the correction is auditable rather than a silent
improvement -- and so nobody later compares a repaired file against a quoted
figure from the unrepaired one and finds a discrepancy with no explanation.
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
OUT = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    alive = subprocess.run(["pgrep", "-f", "y_run_manifest.py"],
                           capture_output=True).returncode == 0
    if alive and not a.dry:
        print("REFUSED: y_run_manifest.py is running and holds this file open.")
        print("Rewriting under an append-only writer can truncate a chunk.")
        print("Wait for it to finish, or use --dry.")
        return 1

    from malign_logits.tasks.code_y_superego_v3 import roundtrip

    #: role IS part of the coordinate -- the same key whose first version
    #: omitted it and resolved half the manifest to the wrong arm.
    def key(pair, role, pid, word, seq_i):
        return (pair, role, pid, word, seq_i)

    rows = [json.loads(l) for l in open(OUT)]
    need = {key(r["pair"], r["role"], r["prompt_id"], r["word"], r["seq_i"])
            for r in rows if r.get("parsed")}
    texts = {}
    for f in [x for x in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl")))
              if "FAILED" not in x]:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "sequences" not in r:
                continue
            for i, s in enumerate(r["sequences"]):
                k = key(r.get("pair"), r.get("role"), r.get("prompt_id"), r.get("word"), i)
                if k in need:
                    texts[k] = s.get("text") or ""

    moved = collections.Counter()
    changed = 0
    for r in rows:
        if not r.get("parsed"):
            continue
        src = texts.get(key(r["pair"], r["role"], r["prompt_id"], r["word"], r["seq_i"]))
        if src is None:
            continue
        new = roundtrip(src, r.get("tagged") or "")
        if new["rt_band"] != r.get("rt_band"):
            moved[(r.get("rt_band"), new["rt_band"])] += 1
            changed += 1
            r["rt_band_pre_fix"] = r.get("rt_band")
            r.update(new)
    print("rows %d   bands changed %d" % (len(rows), changed))
    for (old, new), c in moved.most_common():
        print("   %-12s -> %-12s %4d" % (old, new, c))
    after = collections.Counter(r.get("rt_band") for r in rows if r.get("parsed"))
    print("\nband distribution after repair:")
    n = sum(after.values())
    for k, v in after.most_common():
        print("   %-12s %5d  %5.2f%%" % (k, v, 100 * v / n))
    if a.dry:
        print("\n--dry: nothing written")
        return 0
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)
    print("\nrewrote %s (via %s, atomic replace)" % (OUT, os.path.basename(tmp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
