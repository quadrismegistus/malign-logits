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
