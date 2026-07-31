"""The published lineage map: how many INDEPENDENT alignment implementations?

Docket [1796]/[1798]. In one evening the roster was counted as n=37, n=42, n=21
and n=32 in four different calculations, and every headline quantity depends on
it — H2's p, the family ICC, sigma, the MDE table, S. The four numbers were not
four legitimately scoped populations; they were four answers to one unasked
question.

Lacan's framing is the right one: the deliverable is not "establish n" but
**establish the RULE and publish the MAP**, so that when a claim says "26 of 37"
a reader can see WHICH 37 and WHY.

    .venv/bin/python scripts/lineage_map.py        -> data/lineage_map.json

THE RULE. Two checkpoints are in the same lineage if the registry records a
relation between them that implies shared pretraining. Lineages are the connected
components of that relation set. A FAMILY's lineage is its BASE checkpoint's,
because "independent alignment implementation" means independent PRETRAINING —
two alignment recipes applied to one base are two recipes, not two implementations.

    UNIONING          same_base_as, sft_of, dpo_of, kto_of, ppo_of, slic_of,
                      rlvr_of, reasoning_of, data_ablation_of
    THE ONE JUDGEMENT  smaller_sibling_of / smaller_predecessor_of — a size ladder
                      is one recipe at several scales. Counting the four falcon3
                      sizes as four implementations overstates independence;
                      counting them as one discards real variation. TOGGLED AND
                      REPORTED BOTH WAYS rather than decided here.

WHAT THIS MAP IS NOT. It is an UPPER BOUND on independence, not an estimate. The
registry's relations are populated unevenly, so two genuinely related models with
no recorded edge stay separate. Every count derived from it should say so.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits import MODEL_FAMILIES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "data", "model_registry.json")
OUT = os.path.join(ROOT, "data", "lineage_map.json")

#: Relations implying SHARED PRETRAINING. Declared here, not inferred.
UNIONING = {"same_base_as", "sft_of", "dpo_of", "kto_of", "ppo_of", "slic_of",
            "rlvr_of", "reasoning_of", "data_ablation_of"}
#: THE SIZE RELATIONS ARE NOT ONE THING, and the registry's own naming says so.
#: `smaller_sibling_of` joins one release at several scales (Falcon3 1B/3B/7B/10B,
#: Qwen2.5 0.5B/7B, Olmo-3 7B/32B) -- one recipe, one pretraining design, and
#: merging them is right. `smaller_predecessor_of` joins OLMo-2-0425-1B to
#: Olmo-3-1025-7B, which are DIFFERENT GENERATIONS on different pretraining data;
#: merging those is not a size ladder but a lineage claim the data do not support.
SIBLING = {"smaller_sibling_of"}
PREDECESSOR = {"smaller_predecessor_of"}
SIZE_RELATIONS = SIBLING | PREDECESSOR


class Union:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: the lexicographically smaller id becomes the root, so
            # the map does not depend on edge order in the registry file
            lo, hi = sorted((ra, rb))
            self.p[hi] = lo


def build(extra):
    reg = json.load(open(REGISTRY))
    u = Union()
    for m in reg["models"]:
        u.find(m["model_id"] if isinstance(m, dict) else m)
    kept = UNIONING | extra
    n_edges = 0
    for e in reg["relations"]:
        if e["relation"] in kept:
            u.union(e["parent"], e["child"])
            n_edges += 1
    return u, n_edges


def family_lineage(u):
    """A family's lineage is its BASE checkpoint's component."""
    out, unresolved = {}, []
    for key, fam in MODEL_FAMILIES.items():
        anchor = getattr(fam, "base", None) or fam.ego or fam.superego
        if not anchor:
            unresolved.append(key)
            continue
        out[key] = u.find(anchor)
    return out, unresolved


def main():
    doc = {"_rule": __doc__.split("THE RULE.")[1].split("WHAT THIS MAP IS NOT")[0].strip(),
           "_caveat": "UPPER BOUND on independence, not an estimate: the registry's "
                      "relations are populated unevenly, so two genuinely related "
                      "models with no recorded edge remain separate.",
           "_unioning": sorted(UNIONING), "_sibling": sorted(SIBLING), "_predecessor": sorted(PREDECESSOR)}

    for extra, tag in ((set(), "sizes_separate"),
                       (SIBLING, "siblings_merged"),          # <- the principled one
                       (SIZE_RELATIONS, "siblings_and_predecessors_merged")):
        u, n_edges = build(extra)
        fam2lin, unresolved = family_lineage(u)
        lineages = {}
        for k, lin in sorted(fam2lin.items()):
            lineages.setdefault(lin, []).append(k)
        doc[tag] = {"n_edges_used": n_edges,
                    "n_families": len(fam2lin),
                    "n_lineages": len(lineages),
                    "unresolved_families": unresolved,
                    "lineages": {k: v for k, v in sorted(lineages.items())}}
        print(f"{tag}:  {len(fam2lin)} families -> {len(lineages)} lineages "
              f"({n_edges} relations used)")
        multi = {k: v for k, v in lineages.items() if len(v) > 1}
        for lin, fams in sorted(multi.items(), key=lambda x: -len(x[1])):
            print(f"    {str(lin).split('/')[-1]:<32} {len(fams):>2}  {', '.join(sorted(fams))}")
        if unresolved:
            print(f"    UNRESOLVED (no checkpoint to anchor): {unresolved}")
        print()

    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {OUT}")
    print("\nEVERY COUNT OVER THE ROSTER SHOULD NOW CITE THIS FILE AND SAY WHICH")
    print("TOGGLE IT USED. A count that does not is a count whose denominator")
    print("nobody can check -- which is the whole reason this file exists.")


if __name__ == "__main__":
    main()
