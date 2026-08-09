#!/usr/bin/env python
"""f11_l1_vocab.py — N3's candidate vocabulary over the quintuplet population.

    scripts/f11_l1_vocab.py --show
    scripts/f11_l1_vocab.py --write     -> data/f11_l1_candidate_vocab.json

**STEP 1 OF THREE, AND THE ONLY ONE THAT IS A COMPUTATION.** N3
(`n3_frame_exit_registration.md` §2) defines the L1 measure as:

    1. DISCOVER  every surface with next-token probability >= 0.001, filtered to
                 alphabetic, length >= 2, nonzero English unigram frequency
    2. CODE      each surface, BLIND, into POLE1 / POLE2 / IN-FRAME / OFF-FRAME
    3. MEASURE   masses -> excess_M = M(AB) - mean(M(A), M(B)), classified at
                 t = 0.05 with the sensitivity curve over {.02,.03,.05,.08,.10}

Step 2 needs coders. **So the L1 arm is not a pure computation either** -- the
registration's primary is `coded frame_exit` at L2 and its L1 sibling is coded
surfaces. This script produces the thing to be coded and PRICES that work; it
computes no mass and classifies nothing.

**READ FROM twp, NEVER FROM LOGITS** (RH [5136]): twp's theta and N3's threshold
are both 0.001, and twp at that floor is complete for every word above it. The
logit store would give P(token); N3's surfaces are WORDS.
"""
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

THETA = 0.001
OUT = os.path.join(ROOT, "data", "f11_l1_candidate_vocab.json")


def eligible(w):
    """N3 §2.2 filter: alphabetic, length >= 2, nonzero English unigram freq."""
    if not w or len(w) < 2 or not w.isalpha():
        return False
    try:
        from wordfreq import word_frequency
        return word_frequency(w.lower(), "en") > 0
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    from f11_quintuplet_spec import population, PROMPT_ROLES
    from malign_logits.cache import get_cache
    from malign_logits.registry import Registry
    cm = get_cache()
    byrole, allp, q = population()

    #: group -> {role: prompt}, so a surface can be attributed to its cell
    qs = q["quintuplets"]
    items = qs.items() if isinstance(qs, dict) else [(e.get("group"), e) for e in qs]
    #: **STATUS IS FILTERED AT ANALYSIS TIME.** The source of record CARRIES
    #: status and does not filter ([5084].2), so a consumer that reads it whole
    #: gets 44 groups: 41 ACTIVE, 2 MIXED (f11_reason / _zh) and 1 RETIRED
    #: (f11_species_wolf). The first version of this script pooled all 44 and
    #: built a vocabulary partly out of a retired group -- surfaces that would
    #: have been coded and then thrown away, and worse, coded INTO the same
    #: pool as the primary.
    #:
    #: reason/_zh are the WEAK-MANIPULATION NEGATIVE CONTROL: run beside, never
    #: pooled ([5152]). If the effects appear there too they are not about
    #: contradiction, and that only means anything if the two vocabularies are
    #: kept apart from the start.
    groups, control_groups, dropped = {}, {}, {}
    for gid, v in items:
        if not isinstance(v, dict):
            continue
        name = v.get("group", gid)
        st = (v.get("status") or "").upper()
        cells = {r: v.get(r) for r in PROMPT_ROLES
                 if isinstance(v.get(r), str) and v.get(r)}
        if "RETIRED" in st:
            dropped[name] = v.get("status")
        elif name.startswith("f11_reason"):
            control_groups[name] = cells
        else:
            groups[name] = cells
    print("POPULATION, status-filtered at analysis time")
    print("   primary groups          %d" % len(groups))
    print("   negative control beside %d  (%s)"
          % (len(control_groups), ", ".join(sorted(control_groups))))
    print("   dropped as RETIRED      %d  (%s)"
          % (len(dropped), ", ".join("%s=%s" % kv for kv in dropped.items())))
    print()

    ck = sorted({m for p in Registry().base_aligned_pairs()
                 for m in (p["base"], p["aligned"])})
    surfaces = collections.Counter()
    per_cell, missing = {}, collections.Counter()
    for mid in ck:
        for gid, roles in groups.items():
            for role, prompt in roles.items():
                v = cm.get_true_word_probs(mid, prompt, theta=THETA)
                if not v or not v.get("rows"):
                    missing[(mid in ck, role)] += 1
                    continue
                keep = {r["word"] for r in v["rows"]
                        if r.get("p", 0) >= THETA and eligible(r.get("word", ""))}
                per_cell[(mid, gid, role)] = len(keep)
                for w in keep:
                    surfaces[w] += 1

    print("N3 CANDIDATE VOCABULARY over the PRIMARY population")
    print("  checkpoints            %d" % len(ck))
    print("  groups                 %d" % len(groups))
    print("  cells with twp data    %d" % len(per_cell))
    print("  DISTINCT SURFACES      %d   <- the coding volume" % len(surfaces))
    print("  surfaces per cell      mean %.1f  max %d"
          % ((sum(per_cell.values()) / len(per_cell)) if per_cell else 0,
             max(per_cell.values()) if per_cell else 0))
    print("\n  most frequent surfaces: %s"
          % ", ".join(w for w, _ in surfaces.most_common(12)))

    if a.write:
        json.dump({
            "_about": "N3 candidate vocabulary (registration section 2.2) over the "
                      "quintuplet population. TO BE CODED, blind, into "
                      "POLE1/POLE2/IN-FRAME/OFF-FRAME. This file computes no mass "
                      "and classifies nothing.",
            "_producer": "scripts/f11_l1_vocab.py",
            "_source": "true_word_probs at theta=%.3f (RH [5136]: the twp column, "
                       "never logits)" % THETA,
            "theta": THETA, "checkpoints": len(ck), "groups": len(groups),
            "cells": len(per_cell), "n_surfaces": len(surfaces),
            "surfaces": sorted(surfaces),
            "surface_cell_counts": dict(surfaces.most_common()),
        }, open(OUT, "w"), ensure_ascii=False, indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
