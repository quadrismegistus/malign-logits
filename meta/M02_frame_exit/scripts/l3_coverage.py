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

This counts how many base/aligned pairs have a COMPLETE triplet in BOTH arms,
which is the unit the geometry needs. A pair missing one pole in one arm is not
a degraded cell, it is no cell.

**THE MANIFEST IS NOT USED AND SHOULD NOT BE.** `data/f11_twp/hidden_manifest.json`
is written only by a non-dry-run ingest, and the ingest has been dry-run only, so
it currently describes 31 of the 90 sidecars on disk. It is correct about those 31
and silent about the rest, which is the shape that reads as completeness. The
jsonl records are the record; they were written by the producer that wrote the
bytes.

NOT A FINDING. This is an inventory, and it decides whether L3 is a pilot or a
study before anyone spends a night on it.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

DIRS = [os.path.join(ROOT, "data", "f11_twp"), os.path.join(ROOT, "data", "f11_twp_bf")]
CORE = ("POLE_A", "POLE_B", "BOTH")


def main():
    pairs = json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))
    cat = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]

    #: group -> role -> prompt. STATUS IS NOT A DETAIL HERE, and neither a
    #: no-filter build nor a plain ACTIVE filter is right:
    #:
    #:   f11_species_wolf   all three cells RETIRED            -> DROP
    #:   f11_reason         poles DISPUTED, BOTH ACTIVE        -> RUN, as the
    #:   f11_reason_zh      same shape                            declared
    #:                                                            NEGATIVE CONTROL
    #:
    #: An ACTIVE-only filter deletes f11_reason, which is the group whose result
    #: OUTRANKS the primary's: if the effects appear at a manipulation this weak
    #: (10 of 12 top completions shared) they are not about contradiction. A
    #: no-filter build keeps the retired wolf triplet instead.
    #:
    #: f11_reason ALSO CARRIES ITS POLES TWICE, once DISPUTED and once RETIRED
    #: with byte-identical strings, so a last-write-wins dict is ORDER-DEPENDENT
    #: on file order. Rank explicitly.
    RANK = {"ACTIVE": 3, "DISPUTED": 2, "RETIRED": 1}
    best = collections.defaultdict(dict)
    for r in cat:
        g, role = str(r.get("group_id") or ""), r.get("group_role")
        if not (g.startswith("f11") and r.get("prompt") and role):
            continue
        sc = RANK.get(r.get("status"), 0)
        if sc > best[g].get(role, (0, None))[0]:
            best[g][role] = (sc, r["prompt"])
    groups = {g: {k: v[1] for k, v in roles.items()} for g, roles in best.items()}
    #: a group survives if it has all three core cells and is not wholly retired
    live = {g for g, roles in best.items()
            if any(sc > RANK["RETIRED"] for sc, _ in roles.values())}
    full = {g: v for g, v in groups.items()
            if g in live and all(k in v for k in CORE)}
    control = {g for g in full
               if any(sc == RANK["DISPUTED"] for sc, _ in best[g].values())}
    dropped_retired = sorted(set(groups) - live)

    #: model -> set of prompts that have a residual on disk
    have = collections.defaultdict(set)
    shapes, where = {}, {}
    for d in DIRS:
        for p in glob.glob(d + "/*.jsonl"):
            for line in open(p):
                r = json.loads(line)
                if r.get("hidden_row") is None:
                    continue
                m = r["model"]
                have[m].add(r["prompt"])
                shapes.setdefault(m, tuple(r["hidden_shape"]))
                where.setdefault(m, os.path.basename(d))

    print("fleet: %d models with at least one residual row" % len(have))
    print("f11 groups with a complete POLE_A/POLE_B/BOTH definition: %d of %d"
          % (len(full), len(groups)))
    en = [g for g in full if not g.endswith("_zh")]
    print("   english %d   chinese %d" % (len(en), len(full) - len(en)))
    print("   dropped as wholly RETIRED: %s" % (", ".join(dropped_retired) or "none"))
    print("   carried as DECLARED NEGATIVE CONTROL, never pooled with the primary: %s"
          % ", ".join(sorted(control)))
    #: f11_holy and f11_holy_b share one BOTH string byte-for-byte. Pooling them
    #: counts one measurement twice; the flag travels with the inventory.
    both_seen = collections.Counter(v["BOTH"] for v in full.values())
    dup = {s for s, n in both_seen.items() if n > 1}
    for s in sorted(dup):
        shared = sorted(g for g, v in full.items() if v["BOTH"] == s)
        print("   SHARED `BOTH` CELL, must never be pooled: %s" % " + ".join(shared))
    print("")

    #: THE UNIT IS THE (PAIR, GROUP) CELL AND BOTH ARMS MUST BE COMPLETE.
    cells, by_pair, missing = [], collections.Counter(), collections.Counter()
    usable_pairs = []
    for pr in pairs:
        b, a = pr["base"], pr["aligned"]
        if b not in have or a not in have:
            missing["one arm has no residuals at all"] += 1
            continue
        n = 0
        for g, roles in full.items():
            need = [roles[k] for k in CORE]
            if all(x in have[b] for x in need) and all(x in have[a] for x in need):
                cells.append((pr["family"], pr["stage"], b, a, g))
                n += 1
        by_pair[(pr["family"], b, a)] = n
        if n:
            usable_pairs.append(pr)
        else:
            missing["both arms present, no complete triplet"] += 1

    print("=" * 84)
    print("WHAT L3 CAN RUN TODAY")
    print("=" * 84)
    print("   base/aligned pairs in the roster            %d" % len(pairs))
    print("   pairs with >=1 complete triplet in BOTH arms %d" % len(usable_pairs))
    print("   (pair, group) cells                          %d" % len(cells))
    for k, v in missing.most_common():
        print("   dropped: %-42s %d" % (k, v))

    if not cells:
        print("\n   nothing to run.")
        return 0

    print("\n   BY STAGE")
    for s, c in collections.Counter(x[1] for x in cells).most_common():
        print("      %-8s %4d cells" % (s, c))
    print("\n   BY FAMILY (pairs, cells)")
    fam = collections.defaultdict(lambda: [set(), 0])
    for f, s, b, a, g in cells:
        fam[f][0].add((b, a))
        fam[f][1] += 1
    for f in sorted(fam, key=lambda x: -fam[x][1]):
        print("      %-16s %2d pairs  %4d cells" % (f, len(fam[f][0]), fam[f][1]))

    print("\n   BY GROUP, most-covered first")
    gc = collections.Counter(x[4] for x in cells)
    for g, c in gc.most_common():
        print("      %-24s %3d pairs" % (g, c))

    print("\n   LAYER DEPTHS PRESENT (shape_per_row from the records themselves)")
    ds = collections.Counter(shapes[b] for _, _, b, _, _ in cells)
    for sh, c in sorted(ds.items(), key=lambda x: -x[1]):
        print("      (%d layers, d_model %d)   %d cells" % (sh[0], sh[1], c))

    out = os.path.join(CAMP, "results", "l3_coverage.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"_about": "base/aligned pairs with a complete f11 triplet in both "
                         "arms, residuals on disk. Unit is the (pair, group) cell.",
               "_producer": "meta/M02_frame_exit/scripts/l3_coverage.py",
               "n_pairs": len(usable_pairs), "n_cells": len(cells),
               "cells": [{"family": f, "stage": s, "base": b, "aligned": a, "group": g}
                         for f, s, b, a, g in cells]},
              open(out, "w"), ensure_ascii=False, indent=1)
    print("\nwrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
