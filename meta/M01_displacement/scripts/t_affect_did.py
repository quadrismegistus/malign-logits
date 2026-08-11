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
    base_gap    = mean over 105 minimal pairs of (mass_marked - mass_unmarked)
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
weighted by p). Population: the 105 beam-sample minimal pairs (a subset of
T's twins; noted). Edge unit: Registry().base_aligned_pairs(), ambiguous
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
    by_stem = defaultdict(dict)
    for r in csv.DictReader(open(PAIRS)):
        by_stem[r["stem"]][r["member"]] = r
    pairs = [(v["MARKED"]["prompt"], v["UNMARKED"]["prompt"])
             for v in by_stem.values() if {"MARKED", "UNMARKED"} <= set(v)]
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

    def gap(model, f):
        vals = []
        for mk, un in pairs:
            a, b = word_probs(model, mk), word_probs(model, un)
            if a is None or b is None:
                continue
            am = sum(p for w, p in a.probs.items() if f in affect(w))
            bm = sum(p for w, p in b.probs.items() if f in affect(w))
            vals.append(am - bm)
        return np.mean(vals) if vals else np.nan

    did = defaultdict(list)          # field -> [DiD per edge]
    base_gaps = defaultdict(list)
    for i, (base, al) in enumerate(edges, 1):
        for f in DECLARED:
            bg, ag = gap(base, f), gap(al, f)
            if not (np.isnan(bg) or np.isnan(ag)):
                did[f].append(ag - bg)
                base_gaps[f].append(bg)
        if i % 10 == 0:
            print(f"  {i}/{len(edges)} edges")

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
