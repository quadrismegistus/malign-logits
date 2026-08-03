"""REGISTRATION M's REQUIRED COLUMN. Rows only; M computes no statistic here.

Frozen registration: registration_m_perturbation_null.md @ 3506032d552438e4.
Every choice below is FROZEN THERE, not made here:
  §M2   W = the BASE-RETAINED set (named because "retained" admits three
        readings and they are different nulls)
  §M2   R = sum over W of max(0, p_base - p_aligned);  lambda = 1 - R/sum(p_base)
  §M2   REFUSAL: lambda <= 0 excludes the cell and the count PRINTS
  §M2a  p_aligned = 0 for a word evicted below theta -- the CONSERVATIVE endpoint
        of the known interval [0, theta), taken because it MINIMISES the excess
        the contraction hypothesis needs to be large

The producing seat proposed the hypothesis under test. It produces rows and
NOTHING ELSE: no `e`, no rho, no z, no table. [3699].
"""
import collections, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import m01_concentration as CC
from malign_logits.prompts import Prompts

REGISTRATION = "3506032d552438e4"
THETA = 0.001


def norm(w):
    return (w or "").casefold().rstrip(".,;:!?\"')")


def main():
    d4 = {}
    for f in ("data/d4_fiction_sites.json", "data/d4_fiction_sites_2500.json"):
        for r in json.load(open(os.path.join(ROOT, f))):
            d4.setdefault(r["prompt"], r)
    prompts = Prompts.where(domain="literary")
    _p, models, _h, _d = CC.frozen_population()
    edges, _dr = CC.operation_edges(models)

    rows, diag = [], collections.Counter()
    for fam, pos, step in sorted(edges):
        for p in prompts:
            c = step.cell(p.text)
            if not c.is_present:
                diag["cell absent"] += 1
                continue
            pb, pa = c.pre.probs, c.post.probs
            #: §M2 -- W is the BASE-RETAINED set.
            W = list(pb.items())
            base_mass = sum(v for _w, v in W)
            #: §M2a -- an evicted word's p_aligned is 0, the conservative endpoint.
            R = sum(max(0.0, v - pa.get(w, 0.0)) for w, v in W)
            lam = 1.0 - (R / base_mass) if base_mass > 0 else 0.0
            if lam <= 0:
                diag["REFUSED: lambda <= 0"] += 1
                continue
            gold = (d4.get(p.text) or {}).get("next_actual")
            gn = norm(gold)
            gb = next((w for w in pb if w == gold or norm(w) == gn), None)
            if gb is None:
                diag["gold not base-retained"] += 1
                continue
            ga = next((w for w in pa if w == gold or norm(w) == gn), None)
            p_base_gold = pb[gb]
            p_aligned_gold = (pa[ga] if ga else 0.0)
            rows.append({
                "family": fam, "base": step.pre.id, "aligned": step.post.id,
                "prompt_id": p.id, "gold": gold,
                "p_base_gold": p_base_gold,
                "p_aligned_gold": p_aligned_gold,
                "gold_evicted": int(ga is None),
                "R": R, "base_mass": base_mass, "lambda": lam,
                "n_W": len(W),
                "margin": math.log10(p_base_gold / THETA),
                "d_null": -math.log10(lam),
            })

    src = open(os.path.abspath(__file__), "rb").read()
    out = {
        "_what": "Registration M's required column. ROWS ONLY -- no e, no rho, "
                 "no z, no table, no test.",
        "_registration": "registration_m_perturbation_null.md @ %s" % REGISTRATION,
        "_producer_sha256_16": hashlib.sha256(src).hexdigest()[:16],
        "_theta": THETA,
        "_frozen_choices": {
            "W": "§M2: the BASE-RETAINED set",
            "p_aligned_evicted": "§M2a: 0, the conservative endpoint of [0, theta)",
            "refusal": "§M2: lambda <= 0 excludes the cell; count in diagnostics",
        },
        "_produced_by": "the seat that proposed the hypothesis; adjudication is "
                        "the other seat's per [3699]",
        "n_rows": len(rows), "diagnostics": dict(diag), "rows": rows,
    }
    path = os.path.join(os.path.dirname(HERE), "results", "result_m_column.json")
    open(path, "w").write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("  rows %d" % len(rows))
    print("  diagnostics %s" % dict(diag))
    print("  %s" % path)
    print("  sha256[:16] %s" % hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])


if __name__ == "__main__":
    main()
