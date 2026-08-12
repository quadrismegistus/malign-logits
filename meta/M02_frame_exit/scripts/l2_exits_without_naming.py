#!/usr/bin/env python
"""The exits-without-naming bound: how much frame-exit does the
naming-rate route (second_order_naming / naming_form_control) miss?

    uv run python meta/M02_frame_exit/scripts/l2_exits_without_naming.py

RH's L2-scope question resolved into this residual (needs-doc Addendum 6
Bin B item 1 -> discharged-with-residual, 2026-08-12): the naming findings
measure frame exit VIA contradiction-lexeme rates; a passage can exit the
frame without naming (refusal, meta-commentary, register break). The
coded artifact `results/l2_treatment_paired500.jsonl` (m02_l2_treatment_v1,
50-word window, 499 coded passages, paired arms) carries both routes as
independent fields, so the bound is a read, not a run.

Definitions, from the coded fields:
    EXIT      frame_exit == YES  or  refusal == YES
    NAMED     tension_named == YES
    silent exit    EXIT and not NAMED   <- what naming-rates cannot see
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

SRC = "meta/M02_frame_exit/results/l2_treatment_paired500.jsonl"
OUT = "meta/M02_frame_exit/results/l2_exits_without_naming.json"


def main():
    rows = [json.loads(l) for l in open(SRC)]
    stats = {}
    for arm in ("base", "aligned"):
        sub = [r for r in rows if r["arm"] == arm
               and r.get("degenerate") != "YES"]
        n = len(sub)
        ex = [r for r in sub if r["frame_exit"] == "YES"
              or r["refusal"] == "YES"]
        named = [r for r in sub if r["tension_named"] == "YES"]
        silent = [r for r in ex if r["tension_named"] != "YES"]
        stats[arm] = dict(
            n=n, exit=len(ex), named=len(named), silent_exit=len(silent),
            exit_rate=len(ex) / n, named_rate=len(named) / n,
            silent_rate=len(silent) / n,
            named_share_of_exits=(1 - len(silent) / len(ex)) if ex else None)
        print(f"{arm:8} n={n}  EXIT {len(ex)} ({len(ex)/n:.1%})  "
              f"NAMED {len(named)} ({len(named)/n:.1%})  "
              f"SILENT EXIT {len(silent)} ({len(silent)/n:.1%})")

    # per-group consistency of the aligned silent-exit excess over base
    bygrp = defaultdict(lambda: {"base": [0, 0], "aligned": [0, 0]})
    for r in rows:
        if r.get("degenerate") == "YES":
            continue
        g = bygrp[r["group"]][r["arm"]]
        g[1] += 1
        if (r["frame_exit"] == "YES" or r["refusal"] == "YES") \
                and r["tension_named"] != "YES":
            g[0] += 1
    up = dn = eq = 0
    for g, d in bygrp.items():
        rb = d["base"][0] / d["base"][1] if d["base"][1] else 0
        ra = d["aligned"][0] / d["aligned"][1] if d["aligned"][1] else 0
        up += ra > rb
        dn += ra < rb
        eq += ra == rb
    print(f"groups where aligned silent-exit rate exceeds base: "
          f"{up} up / {dn} down / {eq} tied of {len(bygrp)}")

    b, a = stats["base"], stats["aligned"]
    print(f"\nTHE BOUND: naming captures {a['named_share_of_exits']:.0%} of "
          f"aligned exits; {a['silent_rate']:.1%} of aligned passages exit "
          f"silently (base {b['silent_rate']:.1%}).")
    json.dump(dict(source=SRC, definition="EXIT = frame_exit|refusal; "
                   "SILENT = EXIT & !tension_named; degenerate excluded",
                   window="50 words (the coding doc's own)",
                   stats=stats,
                   groups=dict(up=up, down=dn, tied=eq, total=len(bygrp))),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
