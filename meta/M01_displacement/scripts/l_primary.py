#!/usr/bin/env python3
"""Registration L's DECLARED PRIMARY. Frozen text `72e4b4a94d7c467e`, §L5.

**THIS FILE IS THE TEST AND ONLY THE TEST.** The descriptive layer — the ladder,
the movement columns, `H_retained` — is lacan's and stays lacan's. **A second
implementation of the description by the seat running the inference destroys the
independence the split was buying** ([3643]); this file consumes the committed
per-cell artifact and computes nothing that artifact already holds.

§L5, verbatim in its operative parts:

    PRIMARY   McNemar exact PER FAMILY on that family's 97 discordant cells,
              per rung -- one 2x2 per family, 44 of them, because there is
              exactly one cell per (prompt, family).
              COMBINED ACROSS THE 34 BASE CLUSTERS BY STOUFFER ON THE SIGNED z,
              EQUAL WEIGHTS PER CLUSTER, NOT PER FAMILY.
              A cluster holding k families contributes ONE z, formed as the
              unweighted mean of its k families' z.
              THE 44-FAMILY TABLE AND THE 34-CLUSTER COMBINATION BOTH PRINT,
              and any n stated names which it is.

**§L5 ORDERS NO MDE and this file computes none.** L's nulls are ordinary:
"not detected at this n", licensing no claim in either direction (§L9).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

REGISTRATION_SHA16 = "72e4b4a94d7c467e"
RUNGS = ("argmax", "top20", "retained")     #: L4.4 (role) is NOT a McNemar rung
CELLS = os.path.join(CAMPAIGN, "results", "result_l_found_prose.json")
#: THE PIN, UPDATED UNDER [3828] — RH's word ordering the D-family re-run on the
#: repaired `movement.py`. Protocol ratified at [3830]: the upstream's new hash
#: is POSTED first ([3837]), the pin moves in a commit that changes nothing else,
#: and THE SUPERSEDED VALUE STAYS HERE. Deleting it would destroy the only record
#: that a swap happened, and "re-run under a ruling" would become indistinguishable
#: from "someone edited a constant until the gate passed."
CELLS_SHA16_SUPERSEDED = "f883672020269b95"   #: pre-fix, escrowed at [3837]
CELLS_SHA16 = "18d1b6c9ad2a37af"   #: post-fix re-run, [3837]; verified at this seat


# ══════════════════════════════════════════════════════════════════════════
# the statistic
# ══════════════════════════════════════════════════════════════════════════
def mcnemar_exact(b, c):
    """Two-sided exact McNemar. Returns (p, signed z).

    `b` = base hit & aligned MISS  (a LOSS under alignment)
    `c` = base miss & aligned HIT  (a GAIN)

    **THE SIGN CONVENTION IS DECLARED HERE AND IT MATTERS FOR STOUFFER:**
    z is POSITIVE when b > c, i.e. when alignment LOSES the human's word.
    §L9's rows are written about falls, so a positive z means "the rung fell".

    Exact because §L5 says exact: the conditional null is Binomial(b+c, 1/2)
    and the normal approximation is not trustworthy at these cell counts --
    a family with b+c = 3 is common here and chi-square would invent
    precision the design does not have.
    """
    n = b + c
    if n == 0:
        return None, None                    #: no discordant cells: no test
    #: two-sided exact: P(|X - n/2| >= |b - n/2|)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    #: signed z from the p-value, direction from b vs c. A z of 0 when b == c
    #: is correct: no evidence in either direction.
    if b == c:
        return p, 0.0
    z = abs(_ppf(p / 2))
    return p, (z if b > c else -z)


def _ppf(q):
    """Inverse standard-normal CDF by bisection. No scipy dependency."""
    if q <= 0:
        return -40.0
    if q >= 1:
        return 40.0
    lo, hi = -40.0, 40.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def stouffer(zs):
    """Equal-weight Stouffer over the CLUSTER z's. §L5.

    **The units are the 34 BASE CLUSTERS, never the 44 families** -- seven
    families share `meta-llama/Llama-3.1-8B`, so their base arms are the same
    distribution and their z's are not independent draws. Weighting per family
    would count that base seven times.
    """
    zs = [z for z in zs if z is not None]
    if not zs:
        return None, None, 0
    Z = sum(zs) / math.sqrt(len(zs))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(Z) / math.sqrt(2))))
    return Z, p, len(zs)


# ══════════════════════════════════════════════════════════════════════════
# the run
# ══════════════════════════════════════════════════════════════════════════
def base_of_family():
    """family -> base checkpoint, from the campaign's own edge builder."""
    sys.path.insert(0, os.path.join(os.path.dirname(CAMPAIGN), "scripts"))
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(CAMPAIGN)), "scripts"))
    import m01_concentration as CC
    _p, models, _h, _d = CC.frozen_population()
    edges, _ = CC.operation_edges(models)
    out = {}
    for fam, _pos, step in edges:
        pre = step.pre
        out[fam] = getattr(pre, "id", None) or getattr(pre, "model_id", None) or str(pre)
    return out


def run(cells_path=CELLS):
    if not os.path.exists(cells_path):
        raise SystemExit(
            f"REFUSING: {cells_path} does not exist.\n"
            "The primary consumes the per-cell artifact from L's descriptive "
            "run. Pooled rates cannot produce it -- a pooled rate has already "
            "summed away the pairing McNemar depends on.")
    import hashlib
    blob = open(cells_path, "rb").read()
    got = hashlib.sha256(blob).hexdigest()[:16]
    if got != CELLS_SHA16:
        raise SystemExit(f"REFUSING: {cells_path} hashes {got}, expected "
                         f"{CELLS_SHA16}. The test does not run on unidentified cells.")
    print(f"cells gate PASSED; {os.path.basename(cells_path)} OBSERVED {got}")
    doc = json.loads(blob)
    rows = doc["rows"]
    if len(rows) != doc.get("n_rows", len(rows)):
        raise SystemExit("REFUSING: n_rows disagrees with the row count.")
    fam_base = base_of_family()

    out = {"_registration": REGISTRATION_SHA16, "_cells": cells_path, "rungs": {}}
    for rung in RUNGS:
        per_fam = {}
        for r in rows:
            key = r["family"]
            bh, ah = bool(r[f"base_{rung}"]), bool(r[f"aligned_{rung}"])
            d = per_fam.setdefault(key, {"b": 0, "c": 0, "n": 0})
            d["n"] += 1
            if bh and not ah:
                d["b"] += 1
            elif ah and not bh:
                d["c"] += 1
        fam_z = {}
        for fam, d in per_fam.items():
            p, z = mcnemar_exact(d["b"], d["c"])
            fam_z[fam] = {"b": d["b"], "c": d["c"], "n": d["n"], "p": p, "z": z}
        #: cluster: one z per BASE, the unweighted mean of its families'
        by_base = {}
        for fam, rec in fam_z.items():
            if rec["z"] is None:
                continue
            by_base.setdefault(fam_base.get(fam, f"?{fam}"), []).append(rec["z"])
        clust = {b: sum(v) / len(v) for b, v in by_base.items()}
        Z, p, k = stouffer(list(clust.values()))
        out["rungs"][rung] = {
            "families": fam_z, "n_families": len(fam_z),
            "clusters": clust, "n_clusters": k,
            "stouffer_Z": Z, "stouffer_p": p,
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
def selftest(verbose=True):
    ok, fail = 0, []

    def check(label, cond):
        nonlocal ok
        if cond:
            ok += 1
            verbose and print(f"  ok   {label}")
        else:
            fail.append(label); print(f"  FAIL {label}")

    #: McNemar against hand-computable cases
    p, z = mcnemar_exact(0, 0)
    check("no discordant cells -> no test", p is None and z is None)
    p, z = mcnemar_exact(5, 5)
    check("b == c -> p 1.0 and z EXACTLY 0", abs(p - 1.0) < 1e-12 and z == 0.0)
    #: b=10,c=0,n=10: two-sided exact p = 2 * (1/2)^10 = 0.001953125
    p, z = mcnemar_exact(10, 0)
    check("b=10 c=0 -> exact p = 2/2^10", abs(p - 2 / 1024) < 1e-15)
    check("...and z is POSITIVE (alignment LOST the word)", z > 0)
    p2, z2 = mcnemar_exact(0, 10)
    check("the mirror case has the same p", abs(p - p2) < 1e-15)
    check("...and the OPPOSITE sign", abs(z + z2) < 1e-9)
    #: b=1,c=0,n=1 -> p = 2*(1/2) = 1.0
    check("b=1 c=0 -> p 1.0, the smallest table says nothing",
          abs(mcnemar_exact(1, 0)[0] - 1.0) < 1e-12)

    #: Stouffer
    Z, p, k = stouffer([1.0] * 4)
    check("Stouffer of four 1.0 z's is 2.0", abs(Z - 2.0) < 1e-12 and k == 4)
    Z, _, _ = stouffer([2.0, -2.0])
    check("equal and opposite z's cancel to 0", abs(Z) < 1e-12)
    check("Stouffer of nothing is None", stouffer([])[0] is None)
    #: **the clustering is the point: 7 families on one base must not count 7x**
    seven = [1.0] * 7
    check("a cluster's z is the MEAN of its families, not their sum",
          abs(sum(seven) / len(seven) - 1.0) < 1e-12)

    #: the refusal
    try:
        run("/nonexistent/cells.json"); check("missing cells REFUSES", False)
    except SystemExit:
        check("missing cells REFUSES", True)

    #: §L4.4 is not a McNemar rung -- the frozen text says so
    check("role is NOT among the tested rungs", "role" not in RUNGS)
    n = ok + len(fail)
    print(f"\n{ok}/{n} checks passed" + ("" if not fail else f"; FAILED: {fail}"))
    return not fail


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    res = run()
    print(f"L PRIMARY -- registration {REGISTRATION_SHA16}\n")
    for rung, r in res["rungs"].items():
        print(f"  {rung:<10} families {r['n_families']:>3}  clusters {r['n_clusters']:>3}"
              f"   Stouffer Z {r['stouffer_Z']:+.4f}   p {r['stouffer_p']:.5f}")
    print("\nNO MDE (§L5 orders none). A null reads 'not detected at this n' "
          "and licenses no claim in either direction (§L9).")
