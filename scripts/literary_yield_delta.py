#!/usr/bin/env python3
"""THE LITERARY ARM OF THE METHODS RATIO, RE-DERIVED BOTH WAYS. [3812].

The methods sentence is **92.8% designed vs 23.9% literary = 3.9x**. The
designed arm is PINNED — 635/684 members under both instruments, unmoved
([3811]) — so the ratio moves if and only if this arm moves, and it can only
rise. **3.9x is a ceiling.**

**UNITS ARE HELD AS THEY WERE CLAIMED, NOT REPAIRED** ([3812]): the literary
figure is **1,021 of 4,268 CELLS**; the designed figure is 635 of 684 MEMBERS.
The ratio has always been across those two units. Putting both on cells would
be a cleaner comparison of something nobody claimed.

Population: `Prompts.where(domain="literary")` x the 44 `operation_edges`, the
same population `run_l_found_prose.py` builds. Filters: the `build_word_pool.py`
chain — decompose non-empty, language en, `cell_roles` CANONICAL, function words
dropped, any missing V/A/D drops the word, `>= QUALIFYING_MIN` in EACH role.

BANDS, DECLARED AT [3812] BEFORE THIS NUMBER EXISTED:

    literary post-fix   ratio      reading
    <= 27%              >= 3.4x    the methods sentence stands as written
    27-38%              2.4-3.4x   stands, with the ratio RESTATED
    > 38%               < 2.4x     the sentence is REWRITTEN

    python scripts/literary_yield_delta.py
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

import m01_norms as N                                       # noqa: E402
import m01_registration_b as B                              # noqa: E402
import m01_concentration as CC                              # noqa: E402
from malign_logits.movement import CANONICAL, RESIDUAL_KEY   # noqa: E402
from malign_logits.prompts import Prompts                    # noqa: E402
from pool_delta_1c import contaminated_risers, keep_words    # noqa: E402

OUT = os.path.join(ROOT, "data", "literary_yield_delta.json")
DESIGNED_NUM, DESIGNED_DEN = 635, 684          # members, pinned at [3811]


def main():
    prompts = list(Prompts.where(domain="literary"))
    _p, models, _h, _d = CC.frozen_population()
    edges, _dropped = CC.operation_edges(models)
    norms, _f, _r = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")]
            for d in ("arousal", "valence", "dominance")}
    print(f"literary prompts {len(prompts)}   edges {len(edges)}   "
          f"QUALIFYING_MIN {B.QUALIFYING_MIN}")

    n_cells = old_q = new_q = 0
    added_words = collections.Counter()
    diag = collections.Counter()
    for fam, pos, step in sorted(edges):
        for p in prompts:
            c = step.cell(p.text)
            if not c.is_present:
                diag["cell absent"] += 1
                continue
            if c.language != "en":
                diag["not en"] += 1
                continue
            n_cells += 1
            try:
                if not c.decompose(None):
                    diag["decompose empty"] += 1
                    continue
                roles = N.cell_roles(c, "CANONICAL")
                m = c.movement(CANONICAL)
            except Exception as e:
                diag[f"error {type(e).__name__}"] += 1
                continue
            if m is None:
                diag["movement None"] += 1
                continue

            P = {**c.pre.probs, RESIDUAL_KEY: c.pre.residual}
            Q = {**c.post.probs, RESIDUAL_KEY: c.post.residual}
            old_r = contaminated_risers(P, Q, m.fallers)
            if old_r is None:
                old_r = set(m.risers)
            old_roles = [(w, wt, r) for (w, wt, r) in roles
                         if r == "faller" or w in old_r]

            qual = {}
            for label, rr in (("new", roles), ("old", old_roles)):
                keep = keep_words(rr, tabs)
                nf = sum(1 for _, role in keep if role == "faller")
                qual[label] = (nf >= B.QUALIFYING_MIN
                               and len(keep) - nf >= B.QUALIFYING_MIN)
                if label == "new":
                    nw = {w for w, role in keep if role == "riser"}
            if qual["old"]:
                old_q += 1
            if qual["new"]:
                new_q += 1
            if qual["new"] and not qual["old"]:
                ow = {w for w, role in keep_words(old_roles, tabs)
                      if role == "riser"}
                for w in nw - ow:
                    added_words[w] += 1

    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    old_pct, new_pct = pct(old_q, n_cells), pct(new_q, n_cells)
    d_pct = pct(DESIGNED_NUM, DESIGNED_DEN)
    print(f"\ncells in population                {n_cells:>7,}"
          f"   (booked figure: 4,268)")
    print(f"  diagnostics: {dict(diag)}")
    print(f"\nQUALIFYING CELLS, pre-fix          {old_q:>7,}   {old_pct:>6.1f}%"
          f"   (booked: 1,021 = 23.9%)")
    print(f"QUALIFYING CELLS, post-fix         {new_q:>7,}   {new_pct:>6.1f}%")
    print(f"  cells ADDED                      {new_q - old_q:>7,}")
    print(f"\nRATIO  designed {d_pct:.1f}% (members) : literary (cells)")
    print(f"  pre-fix   {d_pct / old_pct if old_pct else float('nan'):>5.2f}x"
          f"   (booked 3.9x)")
    print(f"  POST-FIX  {d_pct / new_pct if new_pct else float('nan'):>5.2f}x")
    band = ("<=27%: STANDS AS WRITTEN" if new_pct <= 27 else
            "27-38%: STANDS, RATIO RESTATED" if new_pct <= 38 else
            ">38%: SENTENCE REWRITTEN")
    print(f"\n  BAND ([3812], declared before this number): {band}")
    if added_words:
        print(f"  words that carried newly-qualifying cells: "
              f"{added_words.most_common(6)}")

    json.dump({"cells": n_cells, "qualifying_old": old_q,
               "qualifying_new": new_q, "pct_old": old_pct, "pct_new": new_pct,
               "designed_pct": d_pct,
               "ratio_old": d_pct / old_pct if old_pct else None,
               "ratio_new": d_pct / new_pct if new_pct else None,
               "band": band, "diagnostics": dict(diag),
               "added_words": dict(added_words)}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
