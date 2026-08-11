#!/usr/bin/env python
"""Does alignment WIDEN the marked/neutral field gap? Difference-in-differences
sign-flip permutation, within the OLMo lineage (the SHAPE/timing test).

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_widening_null.py

RH (2026-08-11). The divergence permutation ([earlier]) showed the marked-
neutral affective gap is real and label-tied -- but since the prompts differ
by design, much of it is base-level. This tests WIDENING: per pair, the
difference-in-differences
    DiD = [mass(marked,ALIGN) - mass(unmarked,ALIGN)]
        - [mass(marked,BASE)  - mass(unmarked,BASE)]
averaged within region. NULL: sign-flip the per-pair DiD (= swap the
marked/unmarked label), B times; BH-FDR over tested fields. A positive-and-
significant DiD in a field where MARKED already leads means alignment pushed
the transgressive member further; the sign says which way alignment moved
the gap.

SCOPE, STATED LOUDLY: this is ONE LINEAGE (OLMo). 95 rungs are not 95
independent observations -- they are one trajectory, so this establishes the
SHAPE/timing of widening for OLMo, NOT that aligned models in general widen
the gap. The generalisable existence test is a 46-lineage base->aligned DiD
on the same 105 pairs (M01 store), which this script does not attempt.
BASE region = base_step + base_endpoint; ALIGN = sft/dpo/rlvr.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PAIRS = "data/beam_sample_105_plus_anger.csv"
POP = "data/m05_checkpoint_population.json"
LEX = "meta/M01_displacement/lexicons"
OUT = "data/m05_widening_null.json"
B = 20000
RNG = np.random.default_rng(20260811)
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2, None: 3}
BASE_ROLES = {"base_step", "base_endpoint"}
ALIGN_ROLES = {"sft_step", "sft_endpoint", "dpo_endpoint", "rlvr_step"}
_TAG = {r[0]: r[1] for r in csv.reader(open(f"{LEX}/usas_tagset.tsv"),
                                       delimiter="\t") if len(r) >= 2}


def usas_label(code):
    for cand in (code, re.sub(r'[+\-@%]+$', '', code),
                 re.sub(r'[a-z]$', '', re.sub(r'[+\-@%]+$', '', code))):
        if cand in _TAG:
            return _TAG[cand]
    b = re.match(r'[A-Z]\d+(\.\d+)*', code)
    return _TAG.get(b.group(0), code) if b else code


def main():
    from malign_logits import fields
    from malign_logits.movement import word_probs

    pop = json.load(open(POP))["checkpoints"]
    pop = sorted(pop, key=lambda c: (ROLE_ORDER[c["role"]],
                                     STAGE_ORDER.get(c.get("stage")),
                                     c.get("step", 0)))
    base_mids = [(c["model_id"] if c["revision"] == "main"
                  else f"{c['model_id']}@{c['revision']}")
                 for c in pop if c["role"] in BASE_ROLES]
    align_mids = [(c["model_id"] if c["revision"] == "main"
                   else f"{c['model_id']}@{c['revision']}")
                  for c in pop if c["role"] in ALIGN_ROLES]

    by_stem = defaultdict(dict)
    for r in csv.DictReader(open(PAIRS)):
        by_stem[r["stem"]][r["member"]] = r
    pairs = [(s, v["MARKED"]["prompt"], v["UNMARKED"]["prompt"])
             for s, v in by_stem.items() if {"MARKED", "UNMARKED"} <= set(v)]
    print(f"pairs {len(pairs)} | base ckpts {len(base_mids)} | "
          f"align ckpts {len(align_mids)}")

    cache = {}

    def ff(w):
        k = w.strip()
        if k not in cache:
            fs = set()
            try:
                for c in fields.count(k, "usas", True, True)["counts"]:
                    fs.add("USAS: " + usas_label(c))
                for c in fields.count(k, "rid", True, True)["counts"]:
                    fs.add("RID: " + c.rstrip(":"))
                for c in fields.count(k, "wordnet", True, True)["counts"]:
                    fs.add("WN: " + c)
                for dim, r in fields.norms(k).items():
                    for b in r["counts"]:
                        fs.add(f"NORM: {dim}={b}")
            except Exception:
                pass
            cache[k] = fs
        return cache[k]

    def fmass(mid, prompt):
        wp = word_probs(mid, prompt)
        fm = defaultdict(float)
        if wp is None or wp.n_rows == 0:
            return fm
        for w, p in wp.probs.items():
            for f in ff(w):
                fm[f] += p
        return fm

    # per (stem, field, region): mean over region ckpts of (marked - unmarked)
    def region_gap(mids):
        acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # stem->field->[sum,n]
        for mid in mids:
            for s, mk, un in pairs:
                a, b = fmass(mid, mk), fmass(mid, un)
                for f in set(a) | set(b):
                    acc[s][f][0] += a.get(f, 0) - b.get(f, 0)
                    acc[s][f][1] += 1
        return acc

    base_acc = region_gap(base_mids)
    align_acc = region_gap(align_mids)

    fieldset = set()
    for s in align_acc:
        fieldset |= set(align_acc[s])
    stems = [s for s, _, _ in pairs]

    # present floor from base_endpoint mass
    base_main_mass = defaultdict(float)
    for s, mk, un in pairs:
        a, b = fmass("allenai/Olmo-3-1025-7B", mk), \
            fmass("allenai/Olmo-3-1025-7B", un)
        for f in set(a) | set(b):
            base_main_mass[f] = max(base_main_mass[f], a.get(f, 0), b.get(f, 0))

    def gap_of(acc, s, f):
        sm, n = acc[s][f]
        return sm / n if n else 0.0

    results = []
    for f in fieldset:
        if base_main_mass[f] < 0.003:
            continue
        did = np.array([gap_of(align_acc, s, f) - gap_of(base_acc, s, f)
                        for s in stems])
        obs = did.mean()
        signs = RNG.choice([-1.0, 1.0], size=(B, len(did)))
        null = (signs * did).mean(axis=1)
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1)
        base_g = np.mean([gap_of(base_acc, s, f) for s in stems])
        align_g = np.mean([gap_of(align_acc, s, f) for s in stems])
        results.append((f, obs, p, base_g, align_g))

    results.sort(key=lambda r: r[2])
    m = len(results)
    out = []
    for i, (f, obs, p, bg, ag) in enumerate(results, 1):
        out.append(dict(field=f, did=float(obs), p=float(p),
                        q_bh=float(min(p * m / i, 1.0)),
                        base_gap=float(bg), align_gap=float(ag),
                        widened=bool(abs(ag) > abs(bg))))
    for i in range(len(out) - 2, -1, -1):
        out[i]["q_bh"] = min(out[i]["q_bh"], out[i + 1]["q_bh"])
    json.dump({"_about": "difference-in-differences sign-flip null: does "
               "alignment widen the marked-neutral field gap. ONE LINEAGE "
               "(OLMo) -- SHAPE/timing, not generalisation.",
               "B": B, "n_pairs": len(pairs), "n_fields": m,
               "results": out}, open(OUT, "w"), indent=1)

    sig = [r for r in out if r["q_bh"] < 0.05]
    print(f"\n{len(sig)} fields with significant DiD at q<0.05 "
          f"(BH-FDR over {m}): alignment MOVED the gap")
    print(f"{'field':44} {'baseGap':>8} {'alignGap':>8} {'DiD':>8} {'q':>8}")
    for r in sorted(sig, key=lambda r: -abs(r["did"]))[:22]:
        wid = "WIDER" if r["widened"] else "narrower"
        print(f"{r['field']:44} {r['base_gap']:+8.4f} {r['align_gap']:+8.4f} "
              f"{r['did']:+8.4f} {r['q_bh']:8.2g}  {wid}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
