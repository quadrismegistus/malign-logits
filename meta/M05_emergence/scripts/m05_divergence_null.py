#!/usr/bin/env python
"""Permutation null for the marked/neutral field divergence (within-pair sign-flip).

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_divergence_null.py

RH (2026-08-11): make the divergence quotable. The 105 minimal pairs each
have a MARKED (transgressive) and UNMARKED (neutral) twin. For each field,
the statistic is the mean over pairs of (mass_marked - mass_unmarked),
averaged over the ALIGNMENT region (SFT/DPO/RLVR). The NULL swaps the two
labels within each pair independently (sign-flip of the per-pair gap) -- the
correct null for a minimal-pair design: it asks whether the gap is tied to
the transgressive/neutral assignment or is pair-level noise. Two-sided p from
B sign-flips; Benjamini-Hochberg FDR across all tested fields.

Focused extraction: alignment-region checkpoints only, carrying stem+member,
so the pairing the permutation needs exists (the pooled fine parquet dropped
stem). Fields tested = those with base-endpoint mass >= 0.003 (present floor).
"""
import csv
import json
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
POP = "data/m05_checkpoint_population.json"
LEX = "meta/M01_displacement/lexicons"
OUT = "data/m05_divergence_null.json"
B = 20000
RNG = np.random.default_rng(20260811)
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2, None: 3}
ALIGN_ROLES = {"sft_step", "sft_endpoint", "dpo_endpoint", "rlvr_step"}

import re  # noqa: E402
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
    align = [(c["model_id"] if c["revision"] == "main"
              else f"{c['model_id']}@{c['revision']}")
             for c in pop if c["role"] in ALIGN_ROLES]
    base_main = "allenai/Olmo-3-1025-7B"

    by_stem = defaultdict(dict)
    for r in csv.DictReader(open(PAIRS)):
        by_stem[r["stem"]][r["member"]] = r
    pairs = [(s, v["MARKED"]["prompt"], v["UNMARKED"]["prompt"])
             for s, v in by_stem.items() if {"MARKED", "UNMARKED"} <= set(v)]
    print(f"pairs {len(pairs)} | alignment checkpoints {len(align)}")

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

    def prompt_field_mass(mid, prompt):
        wp = word_probs(mid, prompt)
        fm = defaultdict(float)
        if wp is None or wp.n_rows == 0:
            return fm
        for w, p in wp.probs.items():
            for f in ff(w):
                fm[f] += p
        return fm

    # per (stem, field): mean over alignment checkpoints of (marked - unmarked)
    # and base-endpoint mass for the present floor
    gap = defaultdict(lambda: defaultdict(list))   # field -> stem -> [gap per ckpt]
    base_mass = defaultdict(float)
    for s, mk, un in pairs:
        fm_b_mk = prompt_field_mass(base_main, mk)
        fm_b_un = prompt_field_mass(base_main, un)
        for f in set(fm_b_mk) | set(fm_b_un):
            base_mass[f] = max(base_mass[f],
                               fm_b_mk.get(f, 0), fm_b_un.get(f, 0))
    for mid in align:
        for s, mk, un in pairs:
            a = prompt_field_mass(mid, mk)
            b = prompt_field_mass(mid, un)
            for f in set(a) | set(b):
                gap[f][s].append(a.get(f, 0) - b.get(f, 0))

    tested = [f for f in gap if base_mass[f] >= 0.003]
    print(f"fields tested (present >=0.003): {len(tested)}")

    results = []
    stems = [s for s, _, _ in pairs]
    for f in tested:
        # per-stem mean gap over alignment ckpts
        g = np.array([np.mean(gap[f].get(s, [0.0])) for s in stems])
        obs = g.mean()
        signs = RNG.choice([-1.0, 1.0], size=(B, len(g)))
        null = (signs * g).mean(axis=1)
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1)
        results.append((f, obs, p, base_mass[f]))

    # BH-FDR
    results.sort(key=lambda r: r[2])
    m = len(results)
    out = []
    for i, (f, obs, p, bm) in enumerate(results, 1):
        q = min(p * m / i, 1.0)
        out.append(dict(field=f, signed_gap=float(obs), p=float(p),
                        q_bh=float(q), base_mass=float(bm),
                        dir=("MARKED" if obs >= 0 else "UNMARKED")))
    # enforce monotone q
    for i in range(len(out) - 2, -1, -1):
        out[i]["q_bh"] = min(out[i]["q_bh"], out[i + 1]["q_bh"])

    json.dump({"_about": "within-pair sign-flip permutation null for marked-"
               "neutral field divergence, alignment region, BH-FDR",
               "B": B, "n_pairs": len(pairs), "n_fields": m,
               "results": out}, open(OUT, "w"), indent=1)

    sig = [r for r in out if r["q_bh"] < 0.05]
    print(f"\n{len(sig)} fields significant at q<0.05 (BH-FDR over {m}):")
    print(f"{'field':46} {'gap':>9} {'p':>9} {'q':>8}  dir")
    for r in sorted(sig, key=lambda r: -abs(r["signed_gap"]))[:25]:
        print(f"{r['field']:46} {r['signed_gap']:+9.4f} {r['p']:9.2g} "
              f"{r['q_bh']:8.2g}  {r['dir']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
