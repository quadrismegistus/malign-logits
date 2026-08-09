"""How much of L3 can be run from the landed fleet, with no GPU and no generation?

    uv run python l3_coverage.py

`l3_pilot_layerwise.py` measured where a BOTH representation sits between its
poles, per layer, on TWO checkpoints and ONE triplet, and found the arms
diverging at layer 7 and reconverging by the top. Everything it needs is
h_A, h_B, h_AB at every layer: three prompts, one forward pass each.

**THE FLEET ALREADY WROTE THEM.** `twp_cloud.py` emitted a `.hidden.f32` sidecar
per model holding the final-position residual at every layer including the
embedding, float32, and each jsonl record carries its own `hidden_row` index and
`hidden_shape`, so the pairing is explicit rather than positional-by-convention.

    row n of <model>.hidden.f32   ==   the record whose hidden_row is n
    width                          ==   prod(hidden_shape) floats

## THE POPULATION IS READ FROM THE SOURCE OF TRUTH. IT IS NOT DERIVED.

`data/f11_quintuplets.json` is the source of record for F11 and this script
READS it. An earlier version of this file built the population from
`prompt_categorisation.json` and then compared the result to the source of
record, which passed, and was still wrong in the way that matters: a population
you derive and then check is a population you chose. Two consequences, both of
which actually bit:

  - the derivation hardcoded `("POLE_A","POLE_B","BOTH")`, so `control_a` and
    `control_b` -- fields sitting on every entry of the source of record -- were
    invisible BY CONSTRUCTION and could only be mentioned, never counted.
  - it reimplemented status resolution (ACTIVE > DISPUTED > RETIRED) that the
    source of record had already performed and carries in a `status` field.

This is [5146]'s failure at one remove: there a pole-shaped helper's `CORE`
tuple was used as a population definition; here a pole-shaped tuple was written
by hand next to the document that defines one. Read the source.

## AND THE SOURCE OF RECORD IS ITSELF INCOMPLETE, WHICH THE GATE CANNOT SEE

A quintuplet is five cells. **`BOTH_MATCHED` is a sixth, it exists for 10
groups, all 10 of them ARE in the source of record, and the source of record has
no field for it.** So a spec gated against `f11_quintuplets.json` -- the repair
proposed at [5146].4 and [5147].3 -- passes while still omitting BOTH_MATCHED.
The gate is right and it is not sufficient; this script therefore reads that role
from `prompt_categorisation.json` and reports it SEPARATELY, labelled as coming
from outside the source of record, rather than quietly folding it in.

NOT A FINDING. This is an inventory, and it decides whether L3 is a pilot or a
study before anyone spends a night on it.
"""
import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

DIRS = [os.path.join(ROOT, "data", "f11_twp"), os.path.join(ROOT, "data", "f11_twp_bf")]
SRC = os.path.join(ROOT, "data", "f11_quintuplets.json")
#: what the L3 geometry needs: t = (h_AB - h_B).(h_A - h_B) / |h_A - h_B|^2
TRIPLE = ("pole_a", "pole_b", "both")
#: what gives it a null: is a NON-contradictory conjunction in the same place?
CONTROL = ("control_a", "control_b")


def load_source():
    """The population, read from the source of record, with its own inputs verified.

    The verification here is of THE SOURCE, not of a derivation beside it: the
    file byte-copies its strings from three inputs and pins their hashes, so a
    drifted input means the population on disk is not the population that was
    agreed. That is the only check this script is entitled to make about it.
    """
    Q = json.load(open(SRC))
    bad = []
    for name, pinned in Q["_sources"].items():
        p = os.path.join(ROOT, "data", name)
        if pinned is None:
            continue
        live = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None
        if live != pinned:
            bad.append("%s is %s, pinned %s" % (name, live, pinned))
    print("SOURCE OF RECORD  %s" % os.path.relpath(SRC, ROOT))
    print("   selftest at build: %s" % str(Q.get("_selftest"))[:60])
    for name, pinned in Q["_sources"].items():
        print("   input %-34s pinned %s%s"
              % (name, pinned, "" if not any(name in b for b in bad) else "   DRIFTED"))
    if bad:
        for b in bad:
            print("   REFUSED: %s" % b)
        raise SystemExit(1)
    return Q


def main():
    Q = load_source()
    quints = Q["quintuplets"]
    counts = Q["_counts"]

    #: STATUS IS CARRIED BY THE SOURCE, NOT RE-DERIVED HERE. RETIRED is out of
    #: the population; MIXED is the declared negative control, run BESIDE the
    #: primary and never pooled with it.
    live = [q for q in quints if q["status"] != "RETIRED"]
    control_groups = {q["group"] for q in quints if q["status"].startswith("MIXED")}
    retired = [q["group"] for q in quints if q["status"] == "RETIRED"]
    print("   groups %d   live %d   retired %s   negative control %s"
          % (len(quints), len(live), retired or "none", sorted(control_groups)))
    print("   status counts from the source: %s" % counts["by_status"])

    #: BOTH_MATCHED IS NOT IN THE SOURCE OF RECORD. Read separately and labelled.
    cat = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    bmatch = {}
    for r in cat:
        if (str(r.get("group_id") or "").startswith("f11")
                and r.get("group_role") == "BOTH_MATCHED" and r.get("prompt")):
            bmatch[r["group_id"]] = r["prompt"]
    print("   BOTH_MATCHED: %d cells, FROM prompt_categorisation.json, absent from"
          % len(bmatch))
    print("      the source of record -- a gate against it cannot see this role.")

    #: shared BOTH cells: one contradiction measurement, two pole-pairs.
    seen = collections.Counter(q["both"] for q in live)
    for s, n in seen.items():
        if n > 1:
            sh = sorted(q["group"] for q in live if q["both"] == s)
            print("   SHARED `BOTH`, never pool: %s" % " + ".join(sh))

    #: model -> prompts that have a residual on disk
    have = collections.defaultdict(set)
    shapes = {}
    for d in DIRS:
        for p in glob.glob(d + "/*.jsonl"):
            for line in open(p):
                r = json.loads(line)
                if r.get("hidden_row") is None:
                    continue
                have[r["model"]].add(r["prompt"])
                shapes.setdefault(r["model"], tuple(r["hidden_shape"]))
    print("\nfleet: %d models with at least one residual row" % len(have))

    pairs = json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))

    def covered(m, q, roles):
        return all(q.get(k) and q[k] in have.get(m, ()) for k in roles)

    cells, ctl_cells, bm_cells = [], [], []
    usable, no_resid = set(), 0
    for pr in pairs:
        b, a = pr["base"], pr["aligned"]
        if b not in have or a not in have:
            no_resid += 1
            continue
        for q in live:
            g = q["group"]
            if covered(b, q, TRIPLE) and covered(a, q, TRIPLE):
                cells.append((pr["family"], pr["stage"], b, a, g))
                usable.add((b, a))
                if covered(b, q, CONTROL) and covered(a, q, CONTROL):
                    ctl_cells.append((pr["family"], b, a, g))
                if g in bmatch and bmatch[g] in have[b] and bmatch[g] in have[a]:
                    bm_cells.append((pr["family"], b, a, g))

    print("\n" + "=" * 84)
    print("WHAT L3 CAN RUN TODAY")
    print("=" * 84)
    print("   base/aligned pairs in the roster              %d" % len(pairs))
    print("   pairs dropped, one arm has no residuals       %d" % no_resid)
    print("   pairs with >=1 complete TRIPLE in both arms   %d" % len(usable))
    print("   (pair, group) TRIPLE cells                    %d" % len(cells))
    print("   ... of which also have BOTH CONTROLS          %d" % len(ctl_cells))
    print("   ... of which also have BOTH_MATCHED           %d" % len(bm_cells))
    print("\n   %d of %d live groups carry authored controls in the source of record."
          % (sum(1 for q in live if all(q.get(k) for k in CONTROL)), len(live)))
    print("   The fleet scored no CONTROL cell ([5146]), so THE GEOMETRY HAS NO")
    print("   CONJUNCTION NULL: it can say where BOTH sits between its poles, and")
    print("   not whether any conjunction sits there. That is the same missing null")
    print("   `l3_pilot_layerwise.py` recorded, unchanged by 90 checkpoints of data.")

    if not cells:
        return 0
    print("\n   BY STAGE")
    for s, c in collections.Counter(x[1] for x in cells).most_common():
        print("      %-8s %4d cells" % (s, c))
    print("\n   BY LANGUAGE")
    lang = {q["group"]: q["language"] for q in live}
    for l, c in collections.Counter(lang[x[4]] for x in cells).most_common():
        print("      %-8s %4d cells" % (l, c))
    print("\n   LAYER DEPTHS (shape_per_row from the records themselves)")
    for sh, c in collections.Counter(shapes[b] for _, _, b, _, _ in cells).most_common(6):
        print("      (%d layers, d_model %d)   %d cells" % (sh[0], sh[1], c))

    out = os.path.join(CAMP, "results", "l3_coverage.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"_about": "base/aligned pairs with a complete POLE_A/POLE_B/BOTH in "
                         "both arms, residuals on disk. Population READ FROM "
                         "data/f11_quintuplets.json, the source of record.",
               "_producer": "meta/M02_frame_exit/scripts/l3_coverage.py",
               "_no_null": "no CONTROL cell was scored by the fleet ([5146]); "
                           "control coverage is %d cells." % len(ctl_cells),
               "_both_matched_note": "BOTH_MATCHED is absent from the source of "
                                     "record and read from prompt_categorisation.json.",
               "n_pairs": len(usable), "n_cells": len(cells),
               "n_control_cells": len(ctl_cells), "n_both_matched_cells": len(bm_cells),
               "cells": [{"family": f, "stage": s, "base": b, "aligned": a, "group": g}
                         for f, s, b, a, g in cells]},
              open(out, "w"), ensure_ascii=False, indent=1)
    print("\nwrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
