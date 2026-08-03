#!/usr/bin/env python3
"""(1c) THE POOL-LEVEL DELTA. Do the missed risers reach `word_pool_d2`?

Ordered at [3796].2 as the number that scopes the whole re-pricing.

`cell_roles` under the pre-fix instrument emitted a SUBSET of the corrected
riser set ([3795]): the residual-as-faller inflated the null, which raised every
survivor's threshold, which rejected risers the corrected null admits. **The old
code never invented a riser; it only missed them.**

`build_word_pool.py` then applies its own filters, so a missed riser reaches the
pool only if it survives all of them:

    is_present; language == en; decompose(None) non-empty; cell_roles CANONICAL;
    function words dropped; any missing V/A/D drops the word;
    >= QUALIFYING_MIN words in EACH role

**TWO EFFECTS, AND THE SECOND IS THE ONE THAT MATTERS.** Added risers can
(1) add WORDS to a cell already in the pool, and (2) push a cell over the
`>= QUALIFYING_MIN` riser floor so that a CELL ENTERS THE POOL that was not in
it. (2) changes the pool's cell membership, which is what every count, sample
and average over the pool rests on.

This builds both pools — old riser membership and new — through the real
`build_word_pool` filter chain, and diffs them.

    python scripts/pool_delta_1c.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

import within_pair as W                                    # noqa: E402
import m01_norms as N                                      # noqa: E402
import m01_registration_b as B                             # noqa: E402
import m01_concentration as CC                             # noqa: E402
from malign_logits.movement import CANONICAL, RESIDUAL_KEY  # noqa: E402

OUT = os.path.join(ROOT, "data", "pool_delta_1c.json")


def contaminated_risers(P, Q, fall):
    """The riser set the PRE-FIX instrument produced: null from the ratio
    computed with the residual bucket IN the faller set."""
    keys = set(P) | set(Q)
    old_fall = set(fall) | {RESIDUAL_KEY}
    R = 1.0 - sum(Q.get(k, 0.0) for k in old_fall)
    S = sum(P.get(k, 0.0) for k in keys if k not in old_fall)
    if S <= 0:
        return None
    ratio = R / S
    return {k for k in keys if k not in old_fall and k != RESIDUAL_KEY
            and max(P.get(k, 0.0), Q.get(k, 0.0)) > CANONICAL.min_prob
            and (Q.get(k, 0.0) - P.get(k, 0.0)) > CANONICAL.delta
            and Q.get(k, 0.0) > P.get(k, 0.0) * ratio}


def keep_words(roles, tabs):
    """build_word_pool's per-word filter chain, verbatim in effect."""
    keep = []
    for w, wt, role in roles:
        k = N.norm_key(w, "en", fold=False)
        if N.is_function_word(k, "en"):
            continue
        z = {d: N.lookup(tabs[d], k.casefold(), "en")[0]
             for d in ("arousal", "valence", "dominance")}
        if any(v is None for v in z.values()):
            continue
        keep.append((w, role))
    return keep


def main():
    pairs, _ = W.m01_pairs()
    _p, models, _h, _drift = CC.frozen_population()
    edges, _ = CC.operation_edges(models)
    norms, _f, _r = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")]
            for d in ("arousal", "valence", "dominance")}
    texts = {t for v in pairs.values() for t in v.values()}
    print(f"pair member texts {len(texts)}   edges {len(edges)}   "
          f"QUALIFYING_MIN {B.QUALIFYING_MIN}")

    new_cells, old_cells = {}, {}
    added_words = collections.Counter()
    n_seen = n_moved = 0
    for fam, pos, step in sorted(edges):
        for t in texts:
            c = step.cell(t)
            if not c.is_present or c.language != "en":
                continue
            try:
                if not c.decompose(None):
                    continue
                roles = N.cell_roles(c, "CANONICAL")
                m = c.movement(CANONICAL)
            except Exception:
                continue
            if m is None:
                continue
            n_seen += 1
            key = f"{t}\x1f{fam}\x1f{pos}"

            P = {**c.pre.probs, RESIDUAL_KEY: c.pre.residual}
            Q = {**c.post.probs, RESIDUAL_KEY: c.post.residual}
            old_r = contaminated_risers(P, Q, m.fallers)
            if old_r is None:
                old_r = set(m.risers)
            if old_r != set(m.risers):
                n_moved += 1
            old_roles = [(w, wt, r) for (w, wt, r) in roles
                         if r == "faller" or w in old_r]

            for label, rr, store in (("new", roles, new_cells),
                                     ("old", old_roles, old_cells)):
                keep = keep_words(rr, tabs)
                nf = sum(1 for _, role in keep if role == "faller")
                if nf < B.QUALIFYING_MIN or len(keep) - nf < B.QUALIFYING_MIN:
                    continue
                store[key] = keep
            if key in new_cells:
                nw = {w for w, r in new_cells[key] if r == "riser"}
                ow = {w for w, r in old_cells.get(key, []) if r == "riser"}
                for w in nw - ow:
                    added_words[w] += 1

    only_new = sorted(set(new_cells) - set(old_cells))
    only_old = sorted(set(old_cells) - set(new_cells))
    both = set(new_cells) & set(old_cells)
    word_moved = [k for k in both if new_cells[k] != old_cells[k]]

    print(f"\ncells examined                       {n_seen:>7,}")
    print(f"  cells whose RISER SET moved        {n_moved:>7,}")
    print(f"\nPOOL CELLS, old instrument           {len(old_cells):>7,}")
    print(f"POOL CELLS, fixed instrument         {len(new_cells):>7,}")
    print(f"  CELLS ADDED to the pool            {len(only_new):>7,}")
    print(f"  CELLS LOST from the pool           {len(only_old):>7,}"
          f"   (must be 0: the old set is a SUBSET)")
    print(f"  cells in both, WORD LIST CHANGED   {len(word_moved):>7,}")
    print(f"\ndistinct riser words added           {len(added_words):>7,}")
    print(f"total riser word-instances added     {sum(added_words.values()):>7,}")
    if added_words:
        print("  most frequently added:", added_words.most_common(8))


    # ---- MEMBER-LEVEL YIELD, both instruments. [3809]: the methods sentence
    # is a RATIO (92.8% designed vs 23.9% literary) and both arms move UP, so
    # the direction does not settle whether the CONTRAST is preserved.
    role_of = {}
    for pid, mem in pairs.items():
        for role, t in mem.items():
            role_of[t] = role
    def yields(store):
        by = collections.defaultdict(int)
        for k in store:
            by[k.split("\x1f")[0]] += 1
        return by
    y_new, y_old = yields(new_cells), yields(old_cells)
    print("\nMEMBER-LEVEL YIELD (>=1 qualifying cell), designed pairs:")
    print(f"  {'arm':<10}{'members':>9}{'OLD >=1':>10}{'%':>8}{'NEW >=1':>10}{'%':>8}")
    memyield = {}
    for arm in sorted({role_of.get(t) for t in texts if role_of.get(t)}):
        mem = sorted(t for t in texts if role_of.get(t) == arm)
        o = sum(1 for t in mem if y_old.get(t, 0) > 0)
        n_ = sum(1 for t in mem if y_new.get(t, 0) > 0)
        memyield[arm] = {"members": len(mem), "old": o, "new": n_}
        print(f"  {arm:<10}{len(mem):>9}{o:>10}{100*o/len(mem):>7.1f}%"
              f"{n_:>10}{100*n_/len(mem):>7.1f}%")
    import statistics as _st
    for lab, y in (("OLD", y_old), ("NEW", y_new)):
        v = sorted(y.get(t, 0) for t in texts)
        print(f"  cells/member {lab}: median {_st.median(v)}  "
              f"q1 {v[len(v)//4]}  q3 {v[3*len(v)//4]}")

    json.dump({"member_yield": memyield, "cells_examined": n_seen, "riser_set_moved": n_moved,
               "pool_cells_old": len(old_cells), "pool_cells_new": len(new_cells),
               "cells_added": only_new, "cells_lost": only_old,
               "cells_word_changed": len(word_moved),
               "distinct_words_added": len(added_words),
               "word_instances_added": sum(added_words.values()),
               "added_words": dict(added_words)}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
