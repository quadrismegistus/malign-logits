#!/usr/bin/env python
"""Pole separation on the PYTHIA ladder, and the cross-group null OLMo never had.

    uv run python m05_pole_sep_pythia.py --null      # the null first, it gates
    uv run python m05_pole_sep_pythia.py --run

Two things the OLMo ladder could not do, both re-reads of hidden states already
on disk (154 pythia-6.9b step sidecars, 21 en f11 groups with both poles).

## 1. THE CROSS-GROUP NULL, WHICH GATES EVERYTHING ELSE

`pole_sep` is computed from a group's own two pole prompts and is IDENTICAL
across role -- max |both - control_a| = 0.00 over 1,521 cells -- so the
same-side conjunction controls give no comparison. Nothing in
`m05_pole_sep.csv` establishes that the trajectory is about POLES rather than
about any two distinct prompts.

The null pairs `pole_a` of group X with `pole_a` of group Y. Same arithmetic,
same model, same layer, two prompts that are simply different rather than
opposed. If the null tracks the real column, `pole_sep` is a
prompts-are-different detector and every developmental reading built on it is
about representation learning in general, not about poles.

**This runs first and is reported whatever it says.**

## 2. THE ELEVEN RUNGS OLMO DOES NOT HAVE

OLMo's collapse (stage1-step0 0.7945 -> stage1-step1000 0.2273, 21 of 21
groups, p 9.5e-07) is ONE LINE SEGMENT: its first non-zero rung is step 1000.
Pythia has eleven checkpoints below that -- steps 0, 1, 2, 4, 8, 16, 32, 64,
128, 256, 512 -- and malign's [5430] puts an eight-fold rise in words-per-cell
between steps 8 and 128, inside the same window.

So this asks whether the collapse is an EVENT with a locatable shape or an
artefact of having only two points.

## WHAT [5445]/[5446] ESTABLISHED AND THIS INHERITS

At initialisation OLMo's `pole_sep` is FLAT ACROSS ALL 32 LAYERS (1.1x spread,
unguarded) against 18.9x by step 16,000. A flat depth profile is what a random
projection gives. So a high value at step 0 is not pole separation, and the
layer profile -- not the final-layer level -- is the diagnostic that separates
the two regimes. Both are reported per rung here.

**STEP IS AN HONEST KEY ON PYTHIA.** One continuous run, no stages, unlike
OLMo where `stage1-step1000` / `stage2-step1000` / `stage3-step1000` all carry
step == 1000 and cost this campaign a corrected finding today.

The arithmetic is imported from `m05_pole_sep`, not reimplemented: that module's
`geometry()` was validated at exactly zero against `l3_geometry.parquet` before
it was allowed to run.
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from m05_pole_sep import geometry, index_hidden, read  # noqa: E402

OUT = os.path.join(CAMP, "results", "m05_pole_sep_pythia.csv")
NULL = os.path.join(CAMP, "results", "m05_pole_sep_crossgroup_null.csv")
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")


def groups():
    """{group: (pole_a_text, pole_b_text)} for en f11 groups with both poles."""
    cat = json.load(open(CAT))["prompts"]
    g = collections.defaultdict(dict)
    for p in cat:
        if p.get("domain") == "contradiction" and p.get("group_id"):
            g[p["group_id"]][p.get("group_role")] = p
    out = {}
    for gid, v in g.items():
        if not gid.startswith("f11_") or not {"POLE_A", "POLE_B"} <= set(v):
            continue
        if v.get("BOTH", {}).get("language", "en") != "en":
            continue
        out[gid] = (v["POLE_A"]["prompt"].strip(), v["POLE_B"]["prompt"].strip())
    return out


def pythia_checkpoints(idx):
    ck = [m for m in idx if re.match(r"^EleutherAI/pythia-6\.9b@step\d+$", m)]
    return sorted(ck, key=lambda m: int(m.split("@step")[1]))


def sep_only(ha, hb):
    """pole_sep per layer, from the imported `geometry`.

    `geometry` takes a third vector but the separation term does not depend on
    it. VERIFIED, not assumed: over the 22 committed OLMo base-main groups,
    max |pole_sep(a,b,both) - pole_sep(a,b,a)| is EXACTLY 0, and
    pole_sep(a,b,both) reproduces the committed column to 5e-7 under
    prefer="fleet".
    """
    _, _, sep = geometry(ha, hb, ha)
    return sep


def do_null(idx, models, G, out=NULL):
    """Cross-group: pole_a of X against pole_a of Y, all disjoint pairs."""
    gids = sorted(G)
    rows = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "step", "kind", "group_x", "group_y", "layer", "sep"])
        for m in models:
            step = int(m.split("@step")[1]) if "@step" in m else -1
            H = {}
            for gid, (a, b) in G.items():
                ha, hb = read(idx, m, a), read(idx, m, b)
                if ha is not None and hb is not None:
                    H[gid] = (ha, hb)
            if len(H) < 4:
                print("  %-44s only %d groups, skipped" % (m, len(H)))
                continue
            ks = sorted(H)
            for gid in ks:                                   # the real column
                for L, v in enumerate(sep_only(*H[gid])):
                    w.writerow([m, step, "REAL", gid, gid, L, "%.6f" % v]); rows += 1
            for i in range(len(ks)):                         # the null
                for j in range(i + 1, len(ks)):
                    x, y = ks[i], ks[j]
                    for L, v in enumerate(sep_only(H[x][0], H[y][0])):
                        w.writerow([m, step, "NULL", x, y, L, "%.6f" % v]); rows += 1
            print("  %-44s %d groups" % (m, len(H)), flush=True)
    print("wrote %d rows -> %s" % (rows, os.path.relpath(out, ROOT)))


def do_ladder(idx, G):
    ck = pythia_checkpoints(idx)
    print("pythia step checkpoints with hidden states: %d" % len(ck))
    n = miss = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "step", "group", "layer", "n_layers", "pole_sep"])
        for m in ck:
            step = int(m.split("@step")[1])
            got = 0
            for gid, (a, b) in sorted(G.items()):
                ha, hb = read(idx, m, a), read(idx, m, b)
                if ha is None or hb is None:
                    miss += 1
                    continue
                sep = sep_only(ha, hb)
                for L, v in enumerate(sep):
                    w.writerow([m, step, gid, L, len(sep), "%.6f" % v]); n += 1
                got += 1
            print("  step %-8d %2d groups" % (step, got), flush=True)
    print("wrote %d rows (%d cells with no hidden state) -> %s"
          % (n, miss, os.path.relpath(OUT, ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--olmo-null", action="store_true",
                    help="the cross-group null on OLMo's own ladder too")
    a = ap.parse_args()
    #: FLEET, not wider. The committed `m05_pole_sep.csv` was produced with
    #: prefer="fleet" and recomputing it from the wider store differs by
    #: 6.8e-4 against 5e-7 from the fleet -- two stores, same model and prompt,
    #: different bytes. Pythia is absent from the fleet's 95 OLMo checkpoints
    #: so it resolves to the wider store either way (154 checkpoints under
    #: both), but the OLMo null must match the instrument it is a null for.
    idx = index_hidden(prefer="fleet")
    G = groups()
    print("en f11 groups with both poles: %d" % len(G))
    if a.null or a.olmo_null:
        models = []
        if a.null:
            ck = pythia_checkpoints(idx)
            models += [m for m in ck
                       if int(m.split("@step")[1]) in (0, 8, 128, 1000, 16000, 143000)]
        if a.olmo_null:
            models += [m for m in idx if m.startswith("allenai/Olmo-3-1025-7B")
                       and ("step0" in m or "step1000" in m or "step16000" in m)]
        print("null over %d checkpoints" % len(models))
        do_null(idx, models, G)
    if a.run:
        do_ladder(idx, G)


if __name__ == "__main__":
    main()
