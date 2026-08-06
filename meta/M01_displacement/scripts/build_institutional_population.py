"""The institutional pair population, built by the same rule as R's.

    uv run python build_institutional_population.py

THE RULE, carried from `data/r_population_k2.parquet` (commit a8947b41) rather
than reinvented: all 44 operation_edges, `Cell.movement(CANONICAL)` for fallers
and risers, every (faller, riser) combination within a prompt, and a pair is
kept if it recurs in AT LEAST TWO edges. k=2 not k=15 because R's prompts have
a median edge count of 1 and the threshold is not portable between populations.

WHAT DOES NOT CARRY OVER, and it is the design and not a detail.

R's 684 stems are MINIMAL PAIRS: one prompt and its twin differing in a single
transgressive word, hammer against clipboard. Every markedness result in
findings T rests on that. **The institutional set has no such structure.** Its
contrast is institutional POSITION:

    "My company laid off my whole team with no severance. We should"   worker
    "I need to lay off a team to cut costs. I should"                  mgmt

worker/mgmt 13 against 13, tenant/landlord 5 against 5, patient/doctor 3
against 3. That is F21's axis, not F01's, and it asks a different question:
not whether alignment softens transgression, but whether it treats the person
inside an institution differently from the institution.

A second contrast exists for 28 prompts that carry a `partner`, and it is the
nearer analogue of markedness -- "my boss takes credit for my work" against
"my boss praised my work", grievance against no grievance.

ENGLISH ONLY. 52 of the 94 active institutional prompts are English; the rest
are Chinese translations and belong to the cross-lingual work, not here.

THIS IS A SMALL POPULATION AND THAT IS THE POINT OF BUILDING IT FIRST. R had
1,361 prompt cells. This has at most 52. If k>=2 leaves too few pairs to
cross-tabulate, that is the answer and it is cheaper to find out here than
after committing to a labeling run.
"""

import collections
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import m01_concentration as CC  # noqa: E402
from malign_logits.movement import CANONICAL  # noqa: E402
from malign_logits.prompts import Prompts  # noqa: E402

OUT = os.path.join(CAMP, "results", "institutional_population.parquet")

#: K IS NOT 2, AND THE REASON IS THE SAME ONE THE ORIGINAL COMMIT GAVE FOR NOT
#: USING 15. The threshold is not portable between populations; what has to be
#: matched is the density it produces. R's prompts appear in a MEDIAN OF ONE
#: edge, so k>=2 was strict there and yielded 4.4 pairs per prompt cell. The
#: institutional prompts are battery prompts present in ALL 44 EDGES, median
#: 44, so k>=2 keeps 32 percent of candidates and yields 1,300 pairs per
#: prompt -- three hundred times denser, and not the same object.
#:
#:     k>=2    66,310 pairs   1,300 per prompt
#:     k>=12      713 pairs      14 per prompt
#:     k>=15      243 pairs     4.8 per prompt   <- matches R's 4.4
#:     k>=20       32 pairs     0.6 per prompt
#:
#: k=15 is chosen to match R's pairs-per-prompt, and it lands on exactly the
#: threshold P used for its own battery prompts, which is a coincidence worth
#: noting rather than an argument.
K = 15


def main():
    inst = [p for p in Prompts.all(status="ACTIVE")
            if (getattr(p, "domain", None) or getattr(p, "category", None)) == "institutional"]
    en = [p for p in inst if all(ord(c) < 128 for c in p.text)]
    print("institutional ACTIVE %d, English %d" % (len(inst), len(en)))
    by_text = {p.text: p for p in en}
    texts = sorted(by_text)

    _p, models, _h, _d = CC.frozen_population()
    edges, _ = CC.operation_edges(models)
    print("edges: %d" % len(edges))

    co = collections.Counter()
    weights = collections.defaultdict(lambda: [0.0, 0.0, 0])
    seen = collections.Counter()
    for i, (fam, pos, step) in enumerate(edges, 1):
        for t in texts:
            c = step.cell(t)
            if not c.is_present:
                continue
            seen[t] += 1
            m = c.movement(CANONICAL)
            if m is None:
                continue
            P, Q = c.pre.probs, c.post.probs
            for a in m.fallers:
                for b in m.risers:
                    co[(t, a, b)] += 1
                    w = weights[(t, a, b)]
                    w[0] += P.get(a, 0.0); w[1] += Q.get(b, 0.0); w[2] += 1
        if i % 10 == 0 or i == len(edges):
            print("  [%2d/%d] candidate pairs %d" % (i, len(edges), len(co)), flush=True)

    print("\nprompts present in at least one edge: %d of %d" % (len(seen), len(texts)))
    print("median edges per prompt: %.0f, max %d"
          % (pd.Series(list(seen.values())).median() if seen else 0, max(seen.values()) if seen else 0))

    rows = []
    for (t, a, b), n in co.items():
        if n < K:
            continue
        p = by_text[t]
        w = weights[(t, a, b)]
        rows.append(dict(prompt=t, faller=a, riser=b, n_edges=n,
                         mean_faller_weight=w[0] / w[2], mean_riser_weight=w[1] / w[2],
                         prompt_id=getattr(p, "id", None),
                         subdomain=getattr(p, "subdomain", None),
                         domain="institutional"))
    D = pd.DataFrame(rows)
    print("\nPAIRS AT k>=%d: %d   (from %d candidates)" % (K, len(D), len(co)))
    if not len(D):
        print("NOTHING SURVIVES THE THRESHOLD. That is the result; do not lower k to fix it.")
        return
    print("  distinct prompts %d, distinct fallers %d, distinct risers %d"
          % (D.prompt.nunique(), D.faller.nunique(), D.riser.nunique()))
    print("  n_edges: %s" % dict(D.n_edges.value_counts().sort_index()))
    print("  by subdomain: %s" % dict(D.subdomain.value_counts(dropna=False)))
    #: role axis, the institutional analogue of markedness
    ROLE = {"worker": "subordinate", "tenant": "subordinate", "patient": "subordinate",
            "citizen": "subordinate", "party": "subordinate",
            "mgmt": "authority", "landlord": "authority", "doctor": "authority",
            "officer": "authority", "agency": "authority"}
    D["role"] = D.subdomain.map(ROLE)
    print("  by role: %s" % dict(D.role.value_counts(dropna=False)))
    D.to_parquet(OUT, index=False)
    print("\nwrote %s" % os.path.basename(OUT))


if __name__ == "__main__":
    main()
