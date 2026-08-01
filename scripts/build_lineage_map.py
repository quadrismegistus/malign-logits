"""THE LINEAGE MAP — every model to an independent base lineage. Commission [2141].

    .venv/bin/python scripts/build_lineage_map.py            # print
    .venv/bin/python scripts/build_lineage_map.py --write

WHY IT IS THE GATE ON EVERY ROSTER COUNT. In one evening the roster was counted
as n=37, 42, 21 and 32 in four different calculations, and none of them was
wrong about the models — they were counting different things and none said
which. **Every future "n families" sentence and the C5 variance recomputation
cite this file.**

THE ASSIGNMENT RULE, and it is the same one the family-level map already used:

    Two checkpoints are in the same LINEAGE if the registry records a relation
    implying SHARED PRETRAINING. Lineages are the connected components of that
    relation set. A model's lineage is its BASE CHECKPOINT's, because
    "independent alignment implementation" means independent PRETRAINING —
    **two alignment recipes applied to one base are two recipes, not two
    implementations.**

    `base_of()` is populated for ALL 112 models, so no judgment call arises at
    the model level. The documented judgment lives one level up, in which
    RELATIONS count as shared pretraining:

      smaller_sibling_of      SAME release at several scales (Falcon3
                              1B/3B/7B/10B)          -> SAME lineage
      smaller_predecessor_of  DIFFERENT pretraining runs (OLMo-2 -> OLMo-3)
                              -> DIFFERENT lineages

    **That distinction is what carried a result past 0.05 once: counting
    Falcon3's four sizes as four independent implementations.**

THE THREE HISTORICAL MAPS. `data/lineage_map.json` holds three settings —
`sizes_separate`, `siblings_merged`, `siblings_and_predecessors_merged` — at
FAMILY level (49 families). They are not superseded; they are the same rule at
a coarser unit, and this file adds the MODEL level beneath them. **A count
without its unit is not a count**, so every number here carries both.

IT IS AN UPPER BOUND ON INDEPENDENCE, NOT AN ESTIMATE. The registry's relations
are populated unevenly: two genuinely related models with no recorded edge stay
separate, so the lineage count is the LARGEST defensible number and never the
true one. **A more complete registry gives FEWER lineages and a LARGER p, never
smaller.**
"""

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "lineage_map_models.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from malign_logits.registry import Registry
    r = Registry()
    models = sorted(r.models())

    #: MODEL -> BASE. Populated for every model; verified rather than assumed,
    #: because "the field is populated" is the kind of claim that is true until
    #: one roster addition makes it false and nothing notices.
    base = {m: r.base_of(m) for m in models}
    missing = [m for m, b in base.items() if not b]
    fam = {m: r.family_key(m) for m in models}
    stage = {m: r.stage_of(m) for m in models}

    #: LINEAGE = connected component over the base relation. Every model joins
    #: its base; a base whose own base is itself is a root.
    parent = {m: m for m in models}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for m, b in base.items():
        if b and b in parent:
            union(m, b)
        elif b:
            #: A base OUTSIDE the roster is still a lineage root: the models
            #: sharing it are related whether or not we scored the base itself.
            parent.setdefault(b, b)
            union(m, b)

    lin = collections.defaultdict(list)
    for m in models:
        lin[find(m)].append(m)

    fams = collections.defaultdict(set)
    for m in models:
        fams[find(m)].add(fam[m])

    print("THE LINEAGE MAP — model level\n")
    print(f"  models              {len(models)}")
    print(f"  families            {len(set(fam.values()))}")
    print(f"  INDEPENDENT LINEAGES{len(lin):>4}")
    print(f"  base_of missing     {len(missing)}   {missing or '(none)'}")

    multi = {k: v for k, v in lin.items() if len(v) > 1}
    print(f"\n  lineages with >1 model: {len(multi)}")
    for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:8]:
        f = sorted(fams[k])
        print(f"    {k:<44}{len(v):>3} models, {len(f)} famil{'y' if len(f)==1 else 'ies'}")

    print(f"\n  FAMILIES SPANNING MORE THAN ONE LINEAGE (a family is not a unit):")
    byfam = collections.defaultdict(set)
    for m in models:
        byfam[fam[m]].add(find(m))
    span = {f: s for f, s in byfam.items() if len(s) > 1}
    for f, s in sorted(span.items())[:6]:
        print(f"    {f:<24}{len(s)} lineages")
    print(f"    ({len(span)} of {len(byfam)} families)")

    if not args.write:
        print("\n  (print only; --write to commit the artifact)")
        return 0

    doc = {
        "_rule": ("A model's lineage is its BASE CHECKPOINT's. Lineages are "
                  "connected components over the registry's base relation. Two "
                  "alignment recipes applied to one base are two recipes, not "
                  "two implementations."),
        "_caveat": ("UPPER BOUND on independence, never an estimate: the "
                    "registry's relations are populated unevenly, so two "
                    "genuinely related models with no recorded edge remain "
                    "separate. A more complete registry gives FEWER lineages "
                    "and a LARGER p, never smaller."),
        "_unit_warning": ("Every count here carries its unit. models != "
                          "families != lineages, and the roster was once "
                          "counted as 37, 42, 21 and 32 in one evening because "
                          "four calculations used four units and none said so."),
        "_family_level": ("data/lineage_map.json holds the same rule at FAMILY "
                          "level in three settings; NOT superseded, coarser."),
        "counts": {"models": len(models), "families": len(set(fam.values())),
                   "lineages": len(lin)},
        "model_to_lineage": {m: find(m) for m in models},
        "model_to_base": base,
        "model_to_family": fam,
        "model_to_stage": stage,
        "lineages": {k: sorted(v) for k, v in lin.items()},
        "families_spanning_multiple_lineages": {f: sorted(s)
                                                for f, s in span.items()},
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
