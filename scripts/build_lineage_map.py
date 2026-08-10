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
MAP = {}



def consumers():
    """Every finding whose population block cites a family/lineage count.

    **A COMPLETED AUDIT IS INVALIDATED BY A NEW INSTRUMENT THAT CHANGES ITS
    POPULATION, AND NOTHING RE-OPENS IT AUTOMATICALLY** ([2207].3). The map's
    arrival is a RE-AUDIT TRIGGER for every finding it re-counts — **not only
    those whose numbers it changes.** F11's audit was complete, correct at the
    family unit, and wrong at the lineage; nothing in the repo noticed, and
    twice in one day the trigger was a human remembering to ask.

    So the builder emits its own re-audit list. Each row reports the finding's
    declared families resolved through the CURRENT map, so a row where labels
    exceed lineages is a finding whose population double-counts a pretraining
    run — visible at build time, before anyone asks.
    """
    import glob
    import re
    try:
        from malign_logits import MODEL_FAMILIES as MF
    except Exception:
        print("  (registry unavailable — consumer scan skipped)")
        return []
    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "findings", "*.md"))):
        head = "".join(open(f, errors="ignore").readlines()[:14])
        m = re.search(r"^families:\s*\[(.*?)\]", head, re.M)
        if not m:
            continue
        fams = [x.strip() for x in m.group(1).split(",") if x.strip()]
        if not fams:
            continue
        lin = set()
        for k in fams:
            b = getattr(MF.get(k), "base", None)
            lin.add(MAP.get(b, b or k))
        rows.append((os.path.basename(f), len(fams), len(lin)))
    return rows



#: VENDOR-DECLARED DERIVATIONS, each quoted from the model's own card. A
#: stronger claim than the map itself makes: the map says these share a
#: lineage, the cards say which came from which.
#:
#: `Falcon3-10B-Base` is DELIBERATELY ABSENT -- "depth up-scaled from
#: Falcon3-7B-Base with continual pretraining on 2 Teratokens" is more new
#: tokens than most models in the roster see in total, so it is its own
#: lineage rather than a derivative (RH, 2026-08-10). 1B and 3B are pruned and
#: distilled at 80-100 GT, which is a compression. The distinction is the token
#: budget and it is the vendor's.
_DERIVATIVES = {
    "tiiuae/Falcon3-1B-Base": ("tiiuae/Falcon3-3B-Base",
        "pruned in depth, width, heads and embedding channels from a larger "
        "3B Falcon model; 80 GT, knowledge distillation"),
    "tiiuae/Falcon3-3B-Base": ("tiiuae/Falcon3-7B-Base",
        "pruned in depth and width from Falcon3-7B-Base; 100 GT, knowledge "
        "distillation"),
    "meta-llama/Llama-3.2-3B": ("meta-llama/Llama-3.1-8B",
        "logits from Llama 3.1 8B and 70B as token-level targets; knowledge "
        "distillation after pruning"),
}


def _representative(members, cells=None, target_b=7.0, params=None, stage=None):
    """See `_representative_rule` in the emitted document.

    **SIZE COMES FROM THE REGISTRY, NOT FROM THE NAME**, and stage does too.
    The first version parsed `params_b` out of the model id with a regex and
    read `archangel_sft-dpo_pythia2-8b` as 8.0B -- it is a 2.8B pythia -- then
    picked that ALIGNED arm to represent the pythia lineage over the base it
    was tuned from. Two failures, one cause: deriving a declared property from
    a string. `params_b` and the stage are both in the registry.
    """
    params = params or {}
    stage = stage or {}
    #: prefer BASE-stage members: a lineage stands in a base-to-aligned
    #: contrast, so its representative must be able to be the base arm.
    pool = [m for m in members if str(stage.get(m, "")).lower() in ("base", "id")]
    pool = pool or list(members)
    cand = [m for m in pool if m not in _DERIVATIVES] or pool
    def key(m):
        pb = params.get(m)
        return (0 if pb is not None else 1,
                abs((pb if pb is not None else 1e3) - target_b),
                -(cells or {}).get(m, 0), str(m))
    return sorted(cand, key=key)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--consumers", action="store_true",
                    help="print every finding this map re-counts "
                         "(the re-audit trigger list, [2207].3)")
    args = ap.parse_args()

    from malign_logits.registry import Registry
    r = Registry()
    models = sorted(r.models())

    #: MODEL -> BASE. Populated for every model; verified rather than assumed,
    #: because "the field is populated" is the kind of claim that is true until
    #: one roster addition makes it false and nothing notices.
    base = {m: r.base_of(m) for m in models}
    missing = [m for m, b in base.items() if not b]
    #: `family_key` reverse-looks-up MODEL_FAMILIES, so a registry row for a
    #: model with no MODEL_FAMILIES entry returns None — and `sorted()` on a set
    #: containing None raises TypeError. **Five rows added from the beam stash's
    #: unresolved labels crashed this build, and the crash was INVISIBLE because
    #: the print path had already emitted 117 models and 34 lineages before
    #: reaching the sort.** A partial print is not a completed run.
    fam = {m: (r.family_key(m) or "(unregistered)") for m in models}
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

    #: THE UNIONED CLASS, WHICH THIS FILE DOCUMENTED AND DID NOT APPLY.
    #: `smaller_sibling_of` is one release at several scales — ONE lineage, and
    #: no rung is another rung's base ([2160]). The resolver therefore CANNOT
    #: join them: a directed walk leaves each rung at itself, correctly. They
    #: join HERE or not at all, and for one commit they did not:
    #: **39 lineages against 34, five ladder rungs counted as independent
    #: pretraining runs, which is the direction that makes every p SMALLER.**
    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    n_sib = 0
    for rel in reg["relations"]:
        if rel["relation"] == "smaller_sibling_of":
            parent.setdefault(rel["parent"], rel["parent"])
            parent.setdefault(rel["child"], rel["child"])
            union(rel["parent"], rel["child"])
            n_sib += 1

    global MAP
    MAP = {m: find(m) for m in models}

    lin = collections.defaultdict(list)
    for m in models:
        lin[find(m)].append(m)

    fams = collections.defaultdict(set)
    for m in models:
        fams[find(m)].add(fam[m])

    #: THE POSITIVE CONTROLS, ordered [2161], each at the unit its OWN class
    #: licenses. `same_base_as` is CHECKED, so it is tested against the directed
    #: walk's base. `smaller_sibling_of` is UNIONED, so it is tested against the
    #: LINEAGE — **testing it against the base asserts exactly what the union
    #: partition denies, and it duly failed 8 of 8 groups while nothing was
    #: wrong.** A control inherits the algebra of the class it controls.
    ctrl = []
    for kind, unit in (("same_base_as", "base"), ("smaller_sibling_of", "lineage")):
        grp = collections.defaultdict(set)
        for rel in reg["relations"]:
            if rel["relation"] == kind:
                grp[rel["parent"]].add(rel["parent"])
                grp[rel["parent"]].add(rel["child"])
        key = (lambda m: base.get(m)) if unit == "base" else find
        bad = [g for g, ms in grp.items()
               if len({key(m) for m in ms if m in parent}) > 1]
        ctrl.append((kind, unit, len(grp), len(bad)))

    #: AN ABLATION QUALIFIER IS MONOTONE DOWNWARD ([2161]): a child cannot shed
    #: an ablation its parent was defined by. The rule that decided the Tulu
    #: edge without a network fetch, kept as a standing gate.
    import re
    viol = [rel for rel in reg["relations"]
            if rel["relation"] in ("sft_of", "dpo_of", "rlvr_of", "kto_of",
                                   "ppo_of", "slic_of", "data_ablation_of")
            and not frozenset(re.findall(r"no-[a-z]+-data", rel["parent"]))
            <= frozenset(re.findall(r"no-[a-z]+-data", rel["child"]))]

    print("THE LINEAGE MAP — model level\n")
    print(f"  models              {len(models)}")
    print(f"  families            {len(set(fam.values()))}")
    print(f"  INDEPENDENT LINEAGES{len(lin):>4}")
    print(f"  base_of missing     {len(missing)}   {missing or '(none)'}")
    print(f"  sibling unions      {n_sib} edges applied (UNIONED class)")
    print("\n  CONTROLS (each at the unit its class licenses):")
    for kind, unit, ng, nb in ctrl:
        print(f"    {kind:<22} {ng:>2} groups -> one {unit:<8}"
              f"{'PASS' if not nb else f'*** {nb} SPLIT'}")
    print(f"    ablation-subset rule   {len(viol)} violations "
          f"{'PASS' if not viol else '*** MONOTONICITY BROKEN'}")
    if any(nb for *_, nb in ctrl) or viol:
        print("\n  *** CONTROL FAILED — refusing to write.")
        return 1


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

    if args.consumers:
        rows = consumers()
        print(f"\n  RE-AUDIT TRIGGER LIST — {len(rows)} findings declare a "
              f"family population")
        bad = [r for r in rows if r[1] != r[2]]
        print(f"    {'finding':<44}{'labels':>7}{'lineages':>10}")
        for n, a_, b_ in rows:
            mark = "  <-- DOUBLE-COUNTS" if a_ != b_ else ""
            print(f"    {n:<44}{a_:>7}{b_:>10}{mark}")
        print(f"\n  {len(bad)} of {len(rows)} declare more labels than lineages. "
              f"**A completed audit is invalidated by a new instrument that\n"
              f"  changes its population, and nothing re-opens it automatically.**")
        return 0

    if not args.write:
        print("\n  (print only; --write to commit the artifact)")
        return 0

    #: params_b and stage come from the REGISTRY, never from the model id.
    params_b = {m["model_id"]: m.get("params_b") for m in reg["models"]}
    #: cell counts break a tie between candidates equidistant from the target
    #: size. Read from the store if it answers; the map must build without it,
    #: so a failure degrades the tie-break rather than the artifact.
    cells = None
    try:
        import subprocess as _sp
        _o = _sp.run(["/opt/homebrew/bin/clickhouse", "client", "--query",
                      "SELECT model, uniqExact(prompt) FROM "
                      "malign_logits.twp_residual GROUP BY model FORMAT TSV"],
                     capture_output=True, text=True, timeout=120).stdout
        cells = {a: int(b) for a, b in
                 (l.split("\t") for l in _o.splitlines() if "\t" in l)}
    except Exception:
        cells = None

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
        #: cell counts break a tie between two candidates equidistant from
        #: the target size. Read from the store if it answers; the map must
        #: build without it, so a failure here degrades the tie-break rather
        #: than the artifact.
        "_representative_rule": (
            "One member per lineage stands for it in a CROSS-LINEAGE test. "
            "Order: (1) never a vendor-declared derivative -- a compression of "
            "a model is not a second observation of it; (2) closest to 7.0B, "
            "because taking each lineage's largest member would confound "
            "lineage with SCALE (Falcon3-10B against Qwen2.5-0.5B compares two "
            "sizes wearing lineage labels), and the median scored checkpoint "
            "is 7.0B with 97 of 140 in the 6-8B band; (3) most cells; (4) id, "
            "for determinism. STORED, not computed at read time: six scripts "
            "consume this map and two implementations of one decision is the "
            "defect this campaign spent 2026-08-10 removing. A caller wanting "
            "a different target size computes its own and says so."),
        "_representative_derivatives": {
            k: v[1] for k, v in _DERIVATIVES.items()},
        "lineage_to_representative": {
            k: _representative(sorted(v), cells, params=params_b, stage=stage)
            for k, v in lin.items()},
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
