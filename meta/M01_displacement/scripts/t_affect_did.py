#!/usr/bin/env python
"""T addendum: does alignment converge the transgressive/neutral AFFECT gap,
at the EDGE unit? Difference-in-differences across base->aligned edges.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M01_displacement/scripts/t_affect_did.py

M05-C found, on ONE lineage's checkpoint trajectory, that alignment narrows
the transgressive/neutral gap in affect/drive fields (RID:aggression,
valence, arousal...). This runs the same difference-in-differences at T's
own unit -- one vote per base->aligned alignment edge across the roster --
so the SIGN of the convergence gets the generalisation the rungs cannot give.

Per edge e and affect field f:
    base_gap    = mean over the minimal pairs of (mass_marked - mass_unmarked)
                  in the BASE arm
    aligned_gap = same in the ALIGNED arm
    DiD_e(f)    = aligned_gap - base_gap
Across edges, Wilcoxon signed-rank on DiD_e(f) (one vote per edge) + sign
count; BH-FDR over the declared affect set. A negative DiD where the base
gap is positive = the transgressive advantage shrinks under alignment
(convergence).

DECLARED FIELD SET (pre-specified from M05-C, not fished): the norm affect/
concreteness bins, RID drives, and WN contact/perception/cognition. Field
mass = twp probability landing in the field (fields.count/norms single-word,
weighted by p). Population: the full catalogue MARKED/UNMARKED twin set (en, ACTIVE). Edge unit: Registry().base_aligned_pairs(), ambiguous
excluded.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PAIRS = "data/beam_sample_105_plus_anger.csv"
OUT = "meta/M01_displacement/results/t_affect_did.csv"

DECLARED = {
    "NORM: valence=negative", "NORM: valence=positive", "NORM: arousal=aroused",
    "NORM: arousal=calm", "NORM: dominance=dominant", "NORM: dominance=submissive",
    "NORM: concreteness=concrete", "NORM: concreteness=abstract",
    "RID: aggression", "RID: sensation:vision", "RID: abstraction",
    "RID: regressive_cognition:concreteness",
    "WN: contact", "WN: perception", "WN: cognition",
    "USAS: Calm/Violent/Angry",
}


def main():
    from scipy.stats import wilcoxon

    from malign_logits import fields
    from malign_logits.movement import word_probs
    from malign_logits.registry import Registry

    edges = [(p["base"], p["aligned"]) for p in Registry().base_aligned_pairs()
             if not p.get("ambiguous")]
    import json
    pc = json.load(open("data/prompt_categorisation.json"))["prompts"]
    by_pair = defaultdict(dict)
    for p in pc:
        if p.get("language", "en") != "en" or p.get("status") != "ACTIVE":
            continue
        role = p.get("pair_role") or p.get("group_role")
        pid = p.get("pair_id") or p.get("group_id")
        if role in ("MARKED", "UNMARKED") and pid:
            by_pair[pid][role] = p["prompt"]
    pairs = [(v["MARKED"], v["UNMARKED"])
             for v in by_pair.values() if {"MARKED", "UNMARKED"} <= set(v)]
    print(f"edges {len(edges)} | pairs {len(pairs)} | fields {len(DECLARED)}")

    cache = {}

    def affect(w):
        k = w.strip()
        if k not in cache:
            fs = set()
            try:
                for c in fields.count(k, "usas", True, True)["counts"]:
                    if c == "E3":  # Calm/Violent/Angry base code
                        fs.add("USAS: Calm/Violent/Angry")
                for c in fields.count(k, "rid", True, True)["counts"]:
                    fs.add("RID: " + c.rstrip(":"))
                for c in fields.count(k, "wordnet", True, True)["counts"]:
                    fs.add("WN: " + c)
                for dim, r in fields.norms(k).items():
                    for b in r["counts"]:
                        fs.add(f"NORM: {dim}={b}")
            except Exception:
                pass
            cache[k] = fs & DECLARED
        return cache[k]

    pm_cache = {}

    def prompt_masses(model, prompt):
        key = (model, prompt)
        if key not in pm_cache:
            wp = word_probs(model, prompt)
            fm = defaultdict(float)
            if wp is not None and wp.n_rows:
                for w, pp in wp.probs.items():
                    for f in affect(w):
                        fm[f] += pp
            pm_cache[key] = (fm if (wp is not None and wp.n_rows) else None)
        return pm_cache[key]

    def gaps_for(model):
        """field -> mean over pairs of (marked_mass - unmarked_mass)."""
        acc = defaultdict(list)
        for mk, un in pairs:
            a, b = prompt_masses(model, mk), prompt_masses(model, un)
            if a is None or b is None:
                continue
            for f in DECLARED:
                acc[f].append(a.get(f, 0.0) - b.get(f, 0.0))
        return {f: np.mean(v) for f, v in acc.items() if v}

    did = defaultdict(list)
    base_gaps = defaultdict(list)
    for i, (base, al) in enumerate(edges, 1):
        bg, ag = gaps_for(base), gaps_for(al)
        for f in DECLARED:
            if f in bg and f in ag:
                did[f].append(ag[f] - bg[f])
                base_gaps[f].append(bg[f])
        pm_cache.clear()  # free per-edge; bases/aligned rarely repeat
        if i % 10 == 0:
            print(f"  {i}/{len(edges)} edges", flush=True)

    rows = []
    for f in DECLARED:
        d = np.array(did[f])
        if len(d) < 8:
            continue
        stat, p = wilcoxon(d) if np.any(d != 0) else (np.nan, 1.0)
        rows.append((f, float(np.mean(base_gaps[f])), float(d.mean()),
                     int(np.sum(d < 0)), len(d), float(p)))
    rows.sort(key=lambda r: r[5])
    m = len(rows)
    out = []
    for i, (f, bg, dd, nneg, n, p) in enumerate(rows, 1):
        out.append((f, bg, dd, nneg, n, p, min(p * m / i, 1.0)))
    for i in range(len(out) - 2, -1, -1):
        out[i] = out[i][:6] + (min(out[i][6], out[i + 1][6]),)

    with open(OUT, "w") as fh:
        wr = csv.writer(fh)
        wr.writerow(["field", "mean_base_gap", "mean_DiD", "n_neg", "n_edges",
                     "p_wilcoxon", "q_bh"])
        wr.writerows(out)

    print(f"\nEDGE-UNIT AFFECT DiD (one vote per edge, n={out and out[0][4]} "
          f"edges; BH-FDR over {m}):")
    print(f"{'field':42} {'baseGap':>8} {'DiD':>8} {'neg/n':>7} {'q':>8}")
    for f, bg, dd, nneg, n, p, q in out:
        star = "  *" if q < 0.05 else ""
        print(f"{f:42} {bg:+8.4f} {dd:+8.4f} {nneg:>3}/{n:<3} {q:8.3g}{star}")
    sig = [r for r in out if r[6] < 0.05]
    print(f"\n{len(sig)} significant at q<0.05. wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
