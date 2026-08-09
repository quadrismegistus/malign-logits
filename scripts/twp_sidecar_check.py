#!/usr/bin/env python
"""twp_sidecar_check.py — the invariant addendum v3 §4's read-back waiver rests on.

**A WAIVER THAT RESTS ON AN UNENFORCED INVARIANT IS A DOCUMENT SAYING SOMETHING,
NOT A RUNNER DOING IT.** v3 §4 waived the per-checkpoint read-back on the ground
that ingest reads 100% of records -- then named the one thing ingest would have
to check for that to hold, and `twp_ingest.py` does not check it. Measured an
hour after writing the waiver, which is the same question registrar asked about
the producer swap, turned on my own document.

THE INVARIANT. `twp_cloud` writes word rows to `<model>.jsonl` and the raw logit
vector to `<model>.f16`, and the pairing is POSITIONAL: `logit_row` n indexes the
nth vector in the binary. Nothing keys them together. So a lost or duplicated
append shifts every subsequent row and returns real floats for the wrong prompt
-- finite, plausibly ranged, wrong, and passing every check that tests values
rather than alignment.

    rows in the .f16     = filesize / (logit_dim * 2)
    logit-bearing lines  = jsonl lines whose logit_row is not null
    max(logit_row) + 1   must equal both
    logit_row sequence   must be 0..n-1 exactly, no gaps, no repeats

    scripts/twp_sidecar_check.py data/f11_twp
"""
import json, os, sys


def check(d):
    bad = tot = infl = 0
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".jsonl")):
        base = fn[:-6]
        rows, dims = [], set()
        for ln in open(os.path.join(d, fn), errors="ignore"):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("logit_row") is None:
                continue
            rows.append(int(r["logit_row"]))
            if r.get("logit_dim"):
                dims.add(int(r["logit_dim"]))
        tot += 1
        p = os.path.join(d, base + ".f16")
        probs = []
        if not rows:
            print("  %-46s no logit-bearing lines" % base[:46])
            continue
        if len(dims) != 1:
            probs.append("logit_dim is not constant: %s" % sorted(dims))
        else:
            dim = dims.pop()
            if not os.path.exists(p):
                probs.append("sidecar MISSING")
            else:
                n = os.path.getsize(p)
                if n % (dim * 2):
                    probs.append("size %d not a multiple of dim %d x 2" % (n, dim))
                elif n // (dim * 2) != len(rows):
                    #: **IN-FLIGHT IS NOT CORRUPT.** rsync can catch a model
                    #: between its jsonl flush and its sidecar flush, so FEWER
                    #: sidecar rows on a model that has not reached its prompt
                    #: count is a partial sync, complete next pass. MORE rows,
                    #: or a gap in the sequence, is never that. Both refuse to
                    #: pass, but they must not print the same word -- the
                    #: ingest already drew this line and this checker did not,
                    #: so two instruments disagreed about the same invariant.
                    short = n // (dim * 2) < len(rows)
                    probs.append(("IN-FLIGHT: " if short else "") +
                                 "sidecar has %d rows, jsonl has %d"
                                 % (n // (dim * 2), len(rows)))
        if sorted(rows) != list(range(len(rows))):
            probs.append("logit_row is not 0..n-1 (gaps or repeats)")
        if probs:
            inflight = all(x.startswith("IN-FLIGHT") for x in probs)
            if inflight:
                infl += 1
            else:
                bad += 1
            print("  [%s] %-40s %s" % ("in-flight" if inflight else "FAIL  ",
                                       base[:40], "; ".join(probs)))
    print("\n%d model file(s): %d DEFECTIVE, %d in-flight (retry after the run)"
          % (tot, bad, infl))
    return bad


if __name__ == "__main__":
    sys.exit(1 if check(sys.argv[1] if len(sys.argv) > 1 else "data/f11_twp") else 0)
