"""Emit the base>aligned pairs whose base is its lineage's representative.

    uv run python scripts/lineage_representative_pairs.py --write

WHY THIS EXISTS AS A FILE AND NOT AS A LINE IN A NOTEBOOK. The roster has been
counted as 37, 42, 21 and 32 in one evening because four calculations used four
units, and on 2026-08-10 as "52 pairs" and "46 lineages" in consecutive
sentences with neither labelled. 52 and 46 are both correct and they are not
the same population:

    52  base>aligned pairs in the forced-arms battery
    46  independent lineages those pairs span
    62  lineages in the whole registry (158 models) -- NEVER a roster number

THERE ARE NOW TWO 46s AND THEY ARE NOT THE SAME POPULATION (2026-08-12).
This file's 46 is a property of a FROZEN ARTIFACT -- the lineages
`data/forced_arms_105_v3.json` happens to span. The registry answers the same
question live and gives different members:

    47   base->superego canonical lineages DECLARED in the registry
    46   the same, restricted to edges that have data      <- gpt-sw3-6.7b-v2
                                                              is gated, 0 rows
    46   the lineages this file's battery spans            <- a different 46

The two 46s agree on 45 and differ by one member each:

    collected only   EleutherAI/pythia-2.8b   (the archangel base; the battery
                                               carries pythia-6.9b instead)
    battery only     BAAI/Aquila2-7B          (its aligned arm is AquilaChat2-7B
                                               at position EGO, and RH's rule of
                                               2026-08-12 is that a model pair is
                                               base->superego. The registry
                                               declares no_superego='none-published'
                                               for this family; whether that is
                                               right is UNVERIFIED -- the HF card
                                               shows only an `sft=True` inference
                                               flag and the Aquila2 technical
                                               report abstract is silent. If
                                               AquilaChat2 turns out to be
                                               preference-tuned the declared count
                                               is 48, not 47.)

AND THE DEPENDENCY ON THE BATTERY FILE IS REMOVABLE. Verified 2026-08-12:
MODEL_FAMILIES' base -> (superego or ego), kept where the base is its lineage's
representative, reproduces this file's 46 EXACTLY -- 46 of 46, nothing
roster-only. So the registry alone determines the roster, and reading
`forced_arms_105_v3.json` is what freezes this file at the battery's membership
rather than the campaign's current one.

A cross-lineage test wants 46. Anything that reports 52 while treating the rows
as independent is counting Falcon3-1B, -3B and -7B as three observations of
three things when the vendor's own card calls two of them compressions of the
third.

THE SELECTION IS THE MAP'S, NOT THIS SCRIPT'S. `lineage_to_representative` in
data/lineage_map_models.json is the stored answer, and this file looks it up.
It does not re-derive it. The previous implementation of "pick a
representative" sorted on a size parsed out of the model id, read
`archangel_sft-dpo_pythia2-8b` as 8.0B (it is a 2.8B pythia) and elected that
ALIGNED arm to stand for the pythia lineage.

ONE THING THIS SCRIPT DOES DECIDE, AND IT IS CHECKED RATHER THAN ASSUMED. The
map's representative is chosen over every registry member of a lineage; the
battery's unit is a PAIR. Those can disagree -- a lineage's representative
model need not be the base of any pair in the battery. At the time of writing
they do not disagree: all five multi-pair lineages have the stored
representative heading an actual pair. `--strict` (the default) REFUSES rather
than silently falling back, because a fallback here would quietly substitute a
different model for the one the map names and every downstream count would
still read as 46.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = os.path.join(ROOT, "data", "forced_arms_105_v3.json")
MAP = os.path.join(ROOT, "data", "lineage_map_models.json")
OUT = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")


def pairs_by_lineage(arms=ARMS):
    from malign_logits.lineage import lineage_of
    cells = json.load(open(arms))["cells"]
    g = defaultdict(set)
    for c in cells:
        base = c["pair"].split(">")[0]
        g[lineage_of(base)].add(c["pair"])
    return {k: sorted(v) for k, v in g.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=ARMS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--write", action="store_true")
    #: **THE RESTRICTED TABLE IS A PRODUCT, NOT AN AD-HOC STEP.** The previous
    #: `forced_arms_46reps.json` was produced by restricting an arms table to
    #: these pairs by hand, and nothing recorded it -- so its `_derived_from`
    #: named the wrong superset (it carried a prompt _v3 lacks) and could not be
    #: rebuilt. Emitting it here makes the derivation reproducible and carries
    #: the input digest forward, per [5468].
    ap.add_argument("--arms-out", metavar="JSON",
                    help="also write the arms table restricted to the kept pairs")
    ap.add_argument("--loose", action="store_true",
                    help="fall back to the lowest-sorting pair when the stored "
                         "representative heads no pair. Prints the substitution; "
                         "the result is NOT the map's answer.")
    a = ap.parse_args()

    rep = json.load(open(MAP))["lineage_to_representative"]
    g = pairs_by_lineage(a.arms)
    n_pairs = sum(len(v) for v in g.values())
    print("battery: %d pairs across %d lineages" % (n_pairs, len(g)))

    chosen, dropped, trouble = [], [], []
    for lin, ps in sorted(g.items()):
        bases = {p.split(">")[0]: p for p in ps}
        r = rep.get(lin)
        if r in bases:
            chosen.append(bases[r])
            dropped += [p for p in ps if p != bases[r]]
        else:
            trouble.append((lin, r, ps))
            if a.loose:
                chosen.append(ps[0])
                dropped += ps[1:]

    if trouble:
        print("\n%d lineage(s) whose stored representative heads no battery pair:"
              % len(trouble))
        for lin, r, ps in trouble:
            print("  %s  rep=%s  pairs=%s" % (lin, r, ps))
        if not a.loose:
            raise SystemExit(
                "REFUSING. Either the map is stale (rebuild: scripts/"
                "build_lineage_map.py --write) or the battery lacks the "
                "representative's pair. --loose substitutes and says so.")

    if a.arms_out:
        import hashlib
        src = json.load(open(a.arms))
        keep = set(chosen)
        cells = [c for c in src["cells"] if c["pair"] in keep]
        sha = lambda pth: hashlib.sha256(open(pth, "rb").read()).hexdigest()[:16]
        out = {k: v for k, v in src.items() if k != "cells"}
        out.update(
            n_cells=len(cells), cells=cells,
            _producer="scripts/lineage_representative_pairs.py --arms-out",
            _invocation=" ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
            _derived_from={
                "arms": os.path.relpath(a.arms, ROOT),
                "arms_sha256_16": sha(a.arms),
                "lineage_map": os.path.relpath(MAP, ROOT),
                "lineage_map_sha256_16": sha(MAP),
                "rule": "one pair per lineage, the map's stored representative; "
                        "scale siblings of a kept pair are dropped",
                "kept": len(chosen), "dropped": len(dropped)},
            _inherited_inputs=src.get("_inputs"))
        pth = a.arms_out if os.path.isabs(a.arms_out) else os.path.join(ROOT, a.arms_out)
        json.dump(out, open(pth, "w"))
        print("\nwrote %s -- %d cells over %d pairs" % (pth, len(cells), len(keep)))

    print("\nkept %d, dropped %d (scale siblings of a kept pair):"
          % (len(chosen), len(dropped)))
    for p in sorted(dropped):
        print("   -", p)

    if a.write:
        with open(a.out, "w") as fh:
            fh.write("\n".join(sorted(chosen)) + "\n")
        print("\nwrote %s (%d pairs)" % (a.out, len(chosen)))
    else:
        print("\n(dry run; --write to emit %s)" % a.out)


if __name__ == "__main__":
    main()
