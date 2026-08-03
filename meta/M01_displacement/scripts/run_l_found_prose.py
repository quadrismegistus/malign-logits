"""REGISTRATION L PRODUCER — per-cell rows for the declared primary.

Frozen registration: registration_l_found_prose.md @ 72e4b4a94d7c467e.
Cached data only; no inference. Emits ONE ROW PER (family, prompt) so the
declared test -- McNemar per family, Stouffer over the 34 base clusters -- can
be run by a seat that did not write the reading rule.

THE RUN EMITS AN ARTIFACT. A run whose output exists only in a terminal is a
run nobody else can test, which is how this producer came to be written.
"""
import collections, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import m01_concentration as CC, m01_norms as N
from malign_logits.movement import CANONICAL
from malign_logits.prompts import Prompts

REGISTRATION_SHA16 = "72e4b4a94d7c467e"


def norm(w):
    return (w or "").casefold().rstrip(".,;:!?\"')")


def top_word(wp):
    pr = wp.probs
    if not pr:
        return None, 0
    m = max(pr.values())
    tied = [w for w, v in pr.items() if v == m]
    return (tied[0] if len(tied) == 1 else None), len(tied)


def h_retained(wp):
    """§L7: residual as ONE additional bin, NOT renormalised. 0*log0 = 0."""
    ps = list(wp.probs.values())
    res = max(0.0, 1.0 - sum(ps))
    bins = ps + ([res] if res > 0 else [])
    return -sum(p * math.log(p) for p in bins if p > 0), res


def build():
    d4 = {}
    for f in ("data/d4_fiction_sites.json", "data/d4_fiction_sites_2500.json"):
        for r in json.load(open(os.path.join(ROOT, f))):
            d4.setdefault(r["prompt"], r)
    prompts = Prompts.where(domain="literary")
    _p, models, _h, _d = CC.frozen_population()
    edges, _dropped = CC.operation_edges(models)

    rows, diag = [], collections.Counter()
    for fam, pos, step in sorted(edges):
        for p in prompts:
            c = step.cell(p.text)
            if not c.is_present:
                diag["cell absent"] += 1
                continue
            try:
                dec = c.decompose(CANONICAL)          # §L3: EXPLICIT, never None
            except Exception as e:
                diag["decompose %s" % type(e).__name__] += 1
                continue
            gold = (d4.get(p.text) or {}).get("next_actual")
            gn = norm(gold)
            row = {"family": fam, "base": step.pre.id, "aligned": step.post.id,
                   "prompt_id": p.id, "gold": gold}
            for f in ("n_fallers", "n_risers", "departed", "arrived",
                      "concentration", "js_total"):
                row[f] = dec.get(f)                   # None stays None, never 0
            for arm, wp in (("base", c.pre), ("aligned", c.post)):
                pr = wp.probs
                tw, nt = top_word(wp)
                top20 = [w for w, _ in sorted(pr.items(), key=lambda kv: -kv[1])[:20]]
                row["%s_argmax" % arm] = (None if tw is None
                                          else int(tw == gold or norm(tw) == gn))
                row["%s_tied" % arm] = int(nt > 1)
                row["%s_top20" % arm] = int(any(w == gold or norm(w) == gn for w in top20))
                row["%s_retained" % arm] = int(any(w == gold or norm(w) == gn for w in pr))
                row["%s_n_retained" % arm] = len(pr)
                h, res = h_retained(wp)
                row["%s_H_retained" % arm] = h
                row["%s_residual" % arm] = res
            try:
                rr = {w: r for w, _wt, r in N.cell_roles(c, "CANONICAL")}
            except Exception:
                rr = {}
            hit = next((r for w, r in rr.items() if w == gold or norm(w) == gn), None)
            row["gold_role"] = (hit if hit else
                                ("PRESENT-UNMOVED"
                                 if any(w == gold or norm(w) == gn for w in c.pre.probs)
                                 else "ABSENT"))
            rows.append(row)
    return rows, diag


if __name__ == "__main__":
    rows, diag = build()
    src = open(os.path.abspath(__file__), "rb").read()
    out = {
        "_what": "Registration L per-cell rows: movement, the four-rung ladder, "
                 "H_retained. DESCRIPTIVE ONLY -- no test, no p, no verdict.",
        "_registration": "registration_l_found_prose.md @ %s" % REGISTRATION_SHA16,
        "_producer_sha256_16": hashlib.sha256(src).hexdigest()[:16],
        "_unit": "one row per (family, prompt). The DECLARED PRIMARY clusters at "
                 "the 34 BASE checkpoints, equal weight per cluster -- see §L5. "
                 "THESE ROWS ARE NOT THE TEST.",
        "_undefined": "concentration is None where undefined. NEVER coerced to 0.",
        "n_rows": len(rows), "diagnostics": dict(diag), "rows": rows,
    }
    p = os.path.join(os.path.dirname(HERE), "results", "result_l_found_prose.json")
    blob = json.dumps(out, indent=1, sort_keys=True)
    open(p, "w").write(blob + "\n")
    print("  rows %d" % len(rows))
    print("  diagnostics %s" % dict(diag))
    print("  %s" % p)
    print("  sha256[:16] %s" % hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
