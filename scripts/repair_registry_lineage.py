"""REGISTRY LINEAGE REPAIR. Ruled [2148] over [2147] over [2143].1.

    .venv/bin/python scripts/repair_registry_lineage.py --dry-run
    .venv/bin/python scripts/repair_registry_lineage.py --apply

THE RULE, in its definitional form ([2148].7, superseding the family-restricted
version that produced the question): **`base_of` names the TRUE PRETRAINING
BASE wherever the record establishes it, crossing family labels freely.** A
family label is a curation convenience; a pretraining run is a fact about the
weights, and only the second licenses an independence claim.

WHAT IS BEING REPAIRED, and none of it is a schema change.

**(1) TWENTY-SIX MISSING EDGES.** `base_of()` walks `relations` upward and
returns the model itself when nothing names it as a child — so 26 aligned
checkpoints (12 dpo, 12 sft, 2 rlvr) read as their own base. **They are not
corrupt values; they are absent edges**, and `sft_of` / `dpo_of` / `rlvr_of`
are already in the vocabulary. Tulu's two cross to `meta-llama/Llama-3.1-8B`
under the definitional rule.

**(2) THREE STAGE FIELDS, AND THE THREE RELATIONS THAT SAY THE SAME THING.**
[2148].6 found `no-math-data`, `no-persona-data` and `no-wildchat-data` carrying
`stage=dpo` while `no-safety-data` carries `stage=sft`. Four models, one naming
pattern, one disagreeing. **They are SFT data-ablations by name and by
construction — the fourth is right and the three are wrong.**

**AND THE RELATION KIND IS WRONG IN THE SAME THREE PLACES:** the three carry
`dpo_of meta-llama/Llama-3.1-8B` while `no-safety-data` carries `sft_of`.
**Repairing only `stage` would leave the falsehood in the relation set, which
is where `base_of` actually reads** — the field and the edge would agree again
only by both being right, and fixing one of them makes them disagree.

WHAT IS NOT REPAIRED HERE. `same_base_as` is 77 of 181 relations and
`base_of()` consumes it as a parent edge, walking a SIBLING link upward and
taking whichever matches first in FILE ORDER. That is a live defect and a
resolver change, not a data change; it is reported at [2152] and left for a
ruling. **This repair does not touch it, so the counts it produces still carry
that dependency and the map's header says so.**
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "data", "model_registry.json")

#: [2148].6 — SFT data-ablations mislabelled `dpo` in both field and edge.
ABLATION_FIX = ("allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data",
                "allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data",
                "allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data")
#: [2148].7 — the definitional rule crossing family labels.
CROSS_FAMILY = {"allenai/Llama-3.1-Tulu-3-8B-SFT": "meta-llama/Llama-3.1-8B",
                "allenai/Llama-3.1-Tulu-3-8B-DPO": "meta-llama/Llama-3.1-8B"}
STAGE_REL = {"sft": "sft_of", "dpo": "dpo_of", "rlvr": "rlvr_of"}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from malign_logits.registry import Registry
    r = Registry()
    d = json.load(open(REG))
    models = {m["model_id"]: m for m in d["models"]}
    by_family = {}
    for m in d["models"]:
        by_family.setdefault(m["family"], []).append(m["model_id"])

    stage_edits, rel_edits, new_edges = [], [], []

    #: (2) stage, and the edge that repeats it
    for mid in ABLATION_FIX:
        if models[mid]["stage"] != "sft":
            stage_edits.append((mid, models[mid]["stage"], "sft"))
    for rel in d["relations"]:
        if rel["child"] in ABLATION_FIX and rel["relation"] == "dpo_of":
            rel_edits.append((rel["child"], "dpo_of", "sft_of", rel["parent"]))

    #: (1) missing edges — computed AFTER the stage fix, because the edge kind
    #: is derived from the stage and a stale stage would emit a stale kind.
    staged = {m: (dict(stage_edits and {} or {}) or {}) for m in ()}
    fixed_stage = {mid: "sft" for mid in ABLATION_FIX}
    for mid in sorted(models):
        st = fixed_stage.get(mid, models[mid]["stage"])
        if st in ("base", None, "") or r.base_of(mid) != mid:
            continue
        if mid in CROSS_FAMILY:
            parent = CROSS_FAMILY[mid]
        else:
            cands = [x for x in by_family[models[mid]["family"]]
                     if models[x]["stage"] == "base"]
            if len(cands) != 1:
                print(f"  *** UNRESOLVED {mid} (family bases: {cands})")
                continue
            parent = cands[0]
        new_edges.append({"parent": parent, "child": mid,
                          "relation": STAGE_REL.get(st, "sft_of")})

    print(f"STAGE EDITS       {len(stage_edits)}")
    for mid, a, b in stage_edits:
        print(f"    {mid:<52} {a} -> {b}")
    print(f"\nRELATION EDITS    {len(rel_edits)}   "
          f"(the same falsehood in the set base_of actually reads)")
    for c, a, b, p in rel_edits:
        print(f"    {c:<52} {a} -> {b}  ({p})")
    print(f"\nNEW EDGES         {len(new_edges)}")
    import collections
    print(f"    by kind: {dict(collections.Counter(e['relation'] for e in new_edges))}")
    for e in new_edges[:6]:
        print(f"    {e['child']:<52} {e['relation']} {e['parent']}")
    print(f"    ... and {max(0, len(new_edges)-6)} more")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0

    for mid, _, new in stage_edits:
        models[mid]["stage"] = new
    for rel in d["relations"]:
        if rel["child"] in ABLATION_FIX and rel["relation"] == "dpo_of":
            rel["relation"] = "sft_of"
    d["relations"].extend(new_edges)
    d.setdefault("_provenance", {})["lineage_repair_2148"] = {
        "rule": ("base_of names the TRUE PRETRAINING BASE wherever the record "
                 "establishes it, crossing family labels freely ([2148].7)"),
        "stage_edits": len(stage_edits), "relation_edits": len(rel_edits),
        "new_edges": len(new_edges),
        "not_repaired": ("same_base_as consumed as a parent edge by base_of(), "
                         "77 of 181 relations, first-match-in-file-order — a "
                         "resolver defect, reported [2152], unruled"),
    }
    with open(REG, "w") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {REG}: {len(d['relations'])} relations "
          f"({len(new_edges)} added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
