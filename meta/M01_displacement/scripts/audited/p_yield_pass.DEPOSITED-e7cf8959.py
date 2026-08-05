#!/usr/bin/env python3
"""REGISTRATION P's FREEZE PRECONDITION — A-YIELD PER PARTITION. COUNTS ONLY.

**THIS PASS COMPUTES NO HYPOTHESIS QUANTITY AND IS BUILT SO THAT IT CANNOT.**
It answers one question: on N's population, how many cells in each partition
survive to a scoreable `A`? That decides whether P's contrasts are worth
registering, and it feeds the MDE that makes a null interpretable.

**WHY THIS IS NOT A PEEK, AND THE PRECEDENT.** Yield is COUNTS, not values:
how many words clear the norm join in each role. It never touches a z-score.
Registration O did exactly this before it froze (`o_fluent_pass.py`, ordered as
a freeze precondition), and the numbers went into §O1.3 as the ground for
stating whether H2/H3 stood at full strength.

    THE STRUCTURAL GUARANTEE: this file never stores a z-value and never
    forms a weighted mean. `lookup` is called for PRESENCE only and its
    value is discarded at the call site. There is no code path here that
    could emit an A statistic, by construction and not by restraint.

**AND IT DOES NOT COMPUTE `tail_excess` BY PARTITION EITHER.** The zero-faller
count is the analysed-cell denominator and is a population fact; the
distributional statistic itself is P's to test after P freezes.

    python meta/M01_displacement/scripts/p_yield_pass.py [--limit-edges N]
"""

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PAIRS = os.path.join(CAMPAIGN, "results", "population_d_684.json")
OUT = os.path.join(ROOT, "data", "p_yield_pass.json")

SENTINEL = re.compile(r"^<<<.*>>>$")
CJK = re.compile(r"[一-鿿]")

#: The domain grouping. Declared here rather than derived so the partition is
#: a RULE a reader can check, not a count someone produced.
TRANSGRESSIVE = {"violence", "sexual", "profanity", "substance", "death",
                 "taboo", "animal", "betrayal", "power", "property"}
INSTITUTIONAL = {"institutional", "labor", "housing", "medical", "utilities",
                 "civic", "insurance", "education", "banking", "benefits",
                 "immigration", "police", "consumer", "transport", "finance"}


def english_stimuli():
    """N §3/§3.0: distinct stimuli, sentinels out, zh excluded. Published 2,199."""
    from malign_logits.prompts import Prompts
    out = set()
    for p in Prompts().all():
        t = p if isinstance(p, str) else (getattr(p, "text", None) or str(p))
        if SENTINEL.match(t) or CJK.search(t):
            continue
        out.add(t)
    return out


def partition_map():
    """text -> partition label. The 684 pairs by role, the rest by domain."""
    from malign_logits.prompts import Prompts
    byid = {str(p.id): p for p in Prompts().all()}
    stems = json.load(open(PAIRS))["ids"]
    lab = {}
    for s in stems:
        for suf, name in (("_M", "pair_marked"), ("_U", "pair_unmarked")):
            p = byid.get(s + suf)
            if p:
                lab[p.text] = name
    txt2row = {}
    for p in Prompts().all():
        txt2row.setdefault(p.text, p.row or {})
    for t, row in txt2row.items():
        if t in lab:
            continue
        d = row.get("domain") or "(none)"
        lab[t] = ("nonpair_transgressive" if d in TRANSGRESSIVE else
                  "nonpair_institutional" if d in INSTITUTIONAL else
                  "nonpair_neutral" if d == "neutral" else
                  "nonpair_literary" if d == "literary" else
                  "nonpair_contradiction" if d == "contradiction" else
                  "nonpair_other")
    return lab


def main():
    from malign_logits.movement import movement, word_probs, CANONICAL, RESIDUAL_KEY
    import m01_concentration as CC
    import m01_norms as N
    import m01_registration_b as B

    stim = english_stimuli()
    print("population  %d English stimuli  (N publishes 2,199)" % len(stim), flush=True)
    if len(stim) != 2199:
        raise SystemExit("REFUSING: N's population is 2,199; this derives %d" % len(stim))

    lab = partition_map()
    _p, mods, _h, _d = CC.frozen_population()
    edges, _drop = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    edges = [(fam, mid(s.pre), mid(s.post)) for fam, _pos, s in edges]
    if "--limit-edges" in sys.argv:
        edges = edges[:int(sys.argv[sys.argv.index("--limit-edges") + 1])]
    bases = sorted({b for _f, b, _p in edges})
    print("edges %d over %d base clusters" % (len(edges), len(bases)), flush=True)

    norms, _f, _r = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")] for d in ("arousal", "valence")}

    #: counts ONLY. There is no container here that could hold a z-value.
    C = collections.defaultdict(collections.Counter)
    per_cluster = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    roles = collections.defaultdict(list)

    for ei, (fam, pre, post) in enumerate(edges, 1):
        for t in stim:
            part = lab.get(t, "nonpair_other")
            A, Bp = word_probs(pre, t), word_probs(post, t)
            if A is None or Bp is None:
                C[part]["absent"] += 1
                continue
            C[part]["cells"] += 1
            per_cluster[part][pre]["cells"] += 1
            m = movement({**A.probs, RESIDUAL_KEY: A.residual},
                         {**Bp.probs, RESIDUAL_KEY: Bp.residual}, CANONICAL)
            if not m.fallers:
                C[part]["zero_faller"] += 1
                continue
            C[part]["analysed"] += 1
            per_cluster[part][pre]["analysed"] += 1
            kf = kr = 0
            for w in m.fallers:
                k = N.norm_key(w, "en", fold=False)
                if N.is_function_word(k, "en"):
                    continue
                #: PRESENCE ONLY. The value is discarded on this line and is
                #: never bound to a name that leaves the loop.
                if all(N.lookup(tabs[d], k.casefold(), "en")[0] is not None for d in tabs):
                    kf += 1
            for w in m.risers:
                k = N.norm_key(w, "en", fold=False)
                if N.is_function_word(k, "en"):
                    continue
                if all(N.lookup(tabs[d], k.casefold(), "en")[0] is not None for d in tabs):
                    kr += 1
            roles[part].append((kf, kr))
            if kf >= B.QUALIFYING_MIN and kr >= B.QUALIFYING_MIN:
                C[part]["A_cells"] += 1
                per_cluster[part][pre]["A_cells"] += 1
        done = sum(v["cells"] for v in C.values())
        print("  [%2d/%d] %-22s cells so far %6d" % (ei, len(edges), fam, done), flush=True)

    print("\n=== A-YIELD BY PARTITION (counts only; no A computed)", flush=True)
    print("  %-24s %7s %7s %7s %8s %8s" % (
        "partition", "cells", "zero-f", "analysd", "A-cells", "yield%"), flush=True)
    for part in sorted(C, key=lambda p: -C[p]["cells"]):
        c = C[part]
        an, a = c["analysed"], c["A_cells"]
        print("  %-24s %7d %7d %7d %8d %8s" % (
            part, c["cells"], c["zero_faller"], an, a,
            ("%.1f" % (100.0 * a / an)) if an else "n/a"), flush=True)

    print("\n=== PER-CLUSTER A-CELLS, the unit P would test on", flush=True)
    for part in ("pair_marked", "pair_unmarked", "nonpair_transgressive",
                 "nonpair_neutral"):
        d = per_cluster.get(part, {})
        vals = sorted(v["A_cells"] for v in d.values())
        if not vals:
            continue
        print("  %-24s clusters %2d   A-cells/cluster min %d  median %d  max %d"
              % (part, len(vals), vals[0], vals[len(vals) // 2], vals[-1]), flush=True)

    payload = {
        "_what": "Registration P freeze precondition: A-yield per partition. "
                 "COUNTS ONLY — no A statistic and no tail_excess is computed here.",
        "_population": {"stimuli": len(stim), "edges": len(edges),
                        "clusters": len(bases)},
        "_partition_rule": {"transgressive": sorted(TRANSGRESSIVE),
                            "institutional": sorted(INSTITUTIONAL)},
        "partitions": {k: dict(v) for k, v in C.items()},
        "per_cluster_A_cells": {p: {b: dict(c) for b, c in d.items()}
                                for p, d in per_cluster.items()},
        "role_counts": {k: v for k, v in roles.items()},
    }
    json.dump(payload, open(OUT, "w"), indent=1, sort_keys=True)
    print("\nwrote %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
