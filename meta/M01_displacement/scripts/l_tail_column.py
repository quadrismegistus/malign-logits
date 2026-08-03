"""THE TAIL-CONTRACTION FALSIFIER'S INPUT COLUMN. Rows only; NO TEST HERE.

[3648] proposed: alignment's loss of the human's word runs through TAIL
CONTRACTION, not substitution. Its falsifier needs one thing the L artifact
does not carry -- WHERE THE GOLD WORD SAT IN THE BASE DISTRIBUTION.

    IF contraction:  eviction rate rises MONOTONICALLY as p_base(gold) falls,
                     and risers concentrate near the peak.
    IF NOT:          eviction is independent of p_base(gold), the contraction
                     story is dead, and something site-selective is removing
                     the human's word regardless of where it sat.

**THIS FILE EMITS ROWS AND COMPUTES NO TEST.** The hypothesis is this seat's;
the adjudication is not. A SEPARATE artifact so the primary's hash-gated input
(`f883672020269b95`) is untouched.
"""
import collections, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import m01_concentration as CC, m01_norms as N
from malign_logits.movement import CANONICAL
from malign_logits.prompts import Prompts


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

    rows = []
    for fam, pos, step in sorted(edges):
        for p in prompts:
            c = step.cell(p.text)
            if not c.is_present:
                continue
            gold = (d4.get(p.text) or {}).get("next_actual")
            gn = norm(gold)
            pb, pa = c.pre.probs, c.post.probs
            gb = next((w for w in pb if w == gold or norm(w) == gn), None)
            ga = next((w for w in pa if w == gold or norm(w) == gn), None)
            order = sorted(pb.items(), key=lambda kv: -kv[1])
            rank = next((i + 1 for i, (w, _) in enumerate(order) if w == gb), None)
            try:
                rr = {w: r for w, _wt, r in N.cell_roles(c, "CANONICAL")}
            except Exception:
                rr = {}
            role = next((r for w, r in rr.items() if w == gold or norm(w) == gn), None)
            rows.append({
                "family": fam, "base": step.pre.id, "prompt_id": p.id,
                #: WHERE THE GOLD WORD SAT IN BASE -- None when it was never
                #: retained there.  NOT zero: absent and p=0 are different.
                "p_base_gold": (pb[gb] if gb else None),
                "rank_base_gold": rank,
                "n_retained_base": len(pb),
                #: percentile of retained mass BELOW it; None when absent
                "pctile_base_gold": (None if rank is None
                                     else 1.0 - (rank - 1) / max(len(order), 1)),
                "retained_base": int(gb is not None),
                "retained_aligned": int(ga is not None),
                "evicted": int(gb is not None and ga is None),
                "admitted": int(gb is None and ga is not None),
                "gold_role": (role if role else
                              ("PRESENT-UNMOVED" if gb else "ABSENT")),
            })

    src = open(os.path.abspath(__file__), "rb").read()
    out = {
        "_what": "p_base(gold) and its rank, per (family, prompt). ROWS ONLY -- "
                 "NO TEST, NO p, NO VERDICT. Input to [3648]'s falsifier.",
        "_hypothesis_under_test": "[3648]: tail contraction, not substitution. "
                                  "PROPOSED BY THIS SEAT; NOT ADJUDICATED BY IT.",
        "_falsifier": "eviction rate vs p_base(gold): MONOTONIC -> consistent "
                      "with contraction; FLAT -> contraction is dead.",
        "_absent_is_not_zero": "p_base_gold is None where the gold word was not "
                               "retained under base. Never coerced to 0.0.",
        "_producer_sha256_16": hashlib.sha256(src).hexdigest()[:16],
        "_registration": "registration_l_found_prose.md @ 72e4b4a94d7c467e",
        "n_rows": len(rows), "rows": rows,
    }
    p = os.path.join(os.path.dirname(HERE), "results", "result_l_tail_column.json")
    open(p, "w").write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("  rows %d" % len(rows))
    print("  %s" % p)
    print("  sha256[:16] %s" % hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])


if __name__ == "__main__":
    main()
