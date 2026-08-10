#!/usr/bin/env python
"""Which directory holds each f11_twp logit row? Decided per ENTRY, by rank test.

    scripts/resolve_logit_dirs.py --write
    scripts/resolve_logit_dirs.py --limit 400        # pilot

THE DEFECT. The logits index stores a bare basename resolved against ONE root,
`data/raw/cloud_run_20260801`. Two runs wrote `f11_twp` payloads into TWO
directories -- that root's subdirectory and `data/f11_twp/` -- holding DIFFERENT
SUBSETS of the same index at OVERLAPPING ROW NUMBERS. So a lookup lands on
whichever file the root names, and where that row belongs to the other run it
returns a full, finite, plausible vector BELONGING TO A DIFFERENT CELL. No
error, no short read, nothing to notice.

    index entries under f11_twp/         17,534
      row PAST EOF -> raises or skips     2,734
      row WITHIN the file -> SERVED      14,800   <- silently, ~half wrong

**BOTH STORES ARE AFFECTED, NOT JUST ClickHouse.** malign [5283] ran the same
test through `cm.get_logits` and found 39 of 80 correct. The index and the root
are shared, so the ordinary getter resolves exactly as the ingest did. That is
why this writes a resolution map rather than only a re-ingest list.

THE TEST IS COLUMN (C)'s, AND THE THREE WRONG VERSIONS ARE WORTH RECORDING.
Rank twp's top-word FIRST TOKEN (`t1`) in the candidate vector: the correct file
gives rank 0, the wrong one a large arbitrary rank. Attempts that failed first:
comparing the logit ARGMAX to twp's top WORD (twp words are multi-token
expansions, so the top word need not be the argmax token), and doing that again
with the twp source pinned -- which returned 2/5 versus 2/5 and read as
"inconclusive" when it was merely mis-specified.

TWP IS READ AT A PINNED SOURCE. A cell can be scored in three runs with
different values, so the reference must come from the same run as the payload
under test or the test compares two unrelated things.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
CH = "/opt/homebrew/bin/clickhouse"
DB = "malign_logits"
ALT = os.path.join(ROOT, "data", "f11_twp")      # the second directory
OUT = os.path.join(ROOT, "data", "logit_dir_resolution.json")


def _esc(x):
    return x.replace("\\", "\\\\").replace("'", "\\'")


def _unesc(x):
    return (x.replace("\\'", "'").replace("\\t", "\t")
             .replace("\\n", "\n").replace("\\\\", "\\"))


def top_t1(model):
    """prompt -> t1 of the highest-probability word, per source, from twp."""
    out = defaultdict(dict)
    r = subprocess.run(
        [CH, "client", "--query",
         "SELECT source, prompt, argMax(t1, p) FROM %s.twp_words WHERE model='%s' "
         "GROUP BY source, prompt FORMAT TSV" % (DB, _esc(model))],
        capture_output=True, text=True).stdout
    for l in r.splitlines():
        f = l.split("\t")
        if len(f) == 3 and f[2].isdigit():
            out[f[0]][_unesc(f[1])] = int(f[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import numpy as np
    from malign_logits.cache import CacheManager
    cm = CacheManager()
    s = cm._stash("logits")
    root = cm._logit_root()

    ents = defaultdict(list)
    for k in s:
        v = s.get(k)
        if v["file"].startswith("f11_twp/"):
            ents[k["model"]].append((k["prompt"], v["file"], int(v["row"]),
                                     int(v["dim"]), k.get("dtype", "float16")))
    models = sorted(ents)
    if a.limit:
        models = models[:max(1, a.limit // 200)]
    print("models with f11_twp entries: %d\n" % len(models))

    res, tally = [], Counter()
    for mi, m in enumerate(models, 1):
        t1s = top_t1(m)
        #: PINNED SOURCE. Prefer the run whose payload we are testing; fall back
        #: only if that run never scored the cell.
        order = ["cloud_run_20260801", "f11_twp", "f11_twp_delta"]
        mm = {}
        for prompt, f, row, dim, dt in ents[m]:
            t1 = None
            for src in order:
                if prompt in t1s.get(src, {}):
                    t1 = t1s[src][prompt]
                    break
            if t1 is None:
                tally["no twp reference"] += 1
                continue
            cands = {"cloud_run": os.path.join(root, f),
                     "data_f11_twp": os.path.join(ALT, os.path.basename(f))}
            ranks = {}
            for lab, path in cands.items():
                if not os.path.exists(path):
                    ranks[lab] = None
                    continue
                if path not in mm:
                    mm[path] = np.memmap(path, dtype=np.float16 if dt == "float16"
                                         else np.float32, mode="r")
                arr = mm[path]
                if (row + 1) * dim > arr.size:
                    ranks[lab] = None            # past EOF in this candidate
                    continue
                vec = np.asarray(arr[row * dim:(row + 1) * dim], dtype=np.float32)
                ranks[lab] = int((vec > vec[t1]).sum())
            #: THE RULE IS SET FROM THE RANK DISTRIBUTION, NOT FROM EXAMPLES.
            #: Measured over 834 entries with both candidates readable:
            #:
            #:   MIN rank (better candidate)  median 0, p90 6, p99 7, <=20 in 100%
            #:   MAX rank (worse)             median 775, p75 22,096, p90 38,701
            #:
            #: The correct file ranks twp's top-word first token in the TOP 7,
            #: essentially always; the wrong one lands at a random point in a
            #: 64k-256k vocabulary. So argmin with a <=20 gate is sufficient and
            #: no margin term is needed. Two earlier rules were tuned to cases I
            #: had happened to look at -- a flat `rank <= 2` (left 200 of 1,186
            #: undecidable) and a ratio-with-floor-200 (missed `cloud=0 alt=130`,
            #: which is decisive by inspection). Both were fitted to samples;
            #: this is fitted to the distribution.
            live = {l: r for l, r in ranks.items() if r is not None}
            pick = None
            if live:
                lo = min(live, key=lambda l: live[l])
                if live[lo] <= 20:
                    pick = lo
                    if len(live) == 1:
                        tally["only one candidate readable"] += 1
                    elif max(live.values()) <= 20:
                        tally["correct in BOTH"] += 1
            tally["resolved -> " + pick if pick else "UNRESOLVED"] += 1
            res.append({"model": m, "prompt": prompt, "file": f, "row": row,
                        "dir": pick, "rank_cloud_run": ranks.get("cloud_run"),
                        "rank_data_f11_twp": ranks.get("data_f11_twp")})
        if mi % 10 == 0:
            print("  %d/%d models, %s entries" % (mi, len(models), format(len(res), ",")))

    print("\nENTRIES: %s\n" % format(len(res), ","))
    for k, n in tally.most_common():
        print("  %-28s %s" % (k, format(n, ",")))
    unres = [r for r in res if not r["dir"]]
    if unres:
        print("\n  UNRESOLVED SAMPLE (correct in neither candidate):")
        for r in unres[:4]:
            print("     %-30s row %-5d cloud=%s alt=%s"
                  % (r["model"].split("/")[-1][:30], r["row"],
                     r["rank_cloud_run"], r["rank_data_f11_twp"]))
    if a.write:
        json.dump({"_about": "per-entry directory resolution for f11_twp logit "
                             "payloads, decided by rank of twp's top-word t1",
                   "_root": root, "_alt": ALT, "n": len(res),
                   "tally": dict(tally), "entries": res}, open(OUT, "w"))
        print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
