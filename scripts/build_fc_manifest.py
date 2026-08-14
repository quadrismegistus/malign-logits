#!/usr/bin/env python
"""build_fc_manifest.py — the work manifests for the forced-continuation pass.

    scripts/build_fc_manifest.py            write data/fc_manifest_{mps,vast}.json
    scripts/build_fc_manifest.py --show     print the split, write nothing

ONE PAIR PER BASE MODEL, aligned half in the SUPEREGO position.

Grouping is by `model_to_base`, NOT by lineage. `lineage` pools Falcon3-1B/3B/
7B/10B and Qwen2.5-0.5B with Qwen2.5-7B, so grouping by it produces CROSS-SIZE
pairs (a 7B base against a 0.5B aligned model) -- the trap m05_sites.py already
documents. Grouping by base gives 36 pairs where lineage gave 31.

**`stage=dpo` IS DERIVED FROM `position=superego`, NOT FROM TRAINING METHOD.**
phi-4-reasoning and Qwen3-8B carry it too. The picks occupy the superego slot,
which is what was asked for; nobody may later read "36 DPO checkpoints" as a
claim about how they were trained.

SORTED BY PAIR SIZE ASCENDING. The smallest pairs run first so a defect in the
driver surfaces on a 1 GB pair in two minutes rather than on a 40 GB pair in an
hour. Cost-ordering would do the opposite.

THE SPLIT. Everything that is kernel-hungry (hybrid), too large to hold two-up
on 96 GB, or pathologically slow on MPS (RWKV at 134 s/prompt, ~30x a
transformer) goes to the rented CUDA box, plus enough transformer pairs to
balance wall clock. Balancing for SPEED, not cost: the remote side is faster,
so it takes MORE than half the work. Past ~20 pairs the gain stops -- the
remote side's own time rises to meet the local side's.
"""
import argparse
import collections
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

REG = os.path.join(ROOT, "data", "model_registry.json")
LIN = os.path.join(ROOT, "data", "lineage_map_models.json")
SAMPLE = os.path.join(ROOT, "data", "beam_sample_105.csv")
POP = os.path.join(ROOT, "data", "r_population_k2.parquet")
OUT_MPS = os.path.join(ROOT, "data", "fc_manifest_mps.json")
OUT_VAST = os.path.join(ROOT, "data", "fc_manifest_vast.json")

#: measured 2026-08-05, scripts/beam_eta.py --table
RATE = {"transformer": 4.50, "moe": 3.44, "ssm": 21.25,
        "linear_attn_rnn": 134.22, "hybrid": None}
PAIR_GB_CAP = 60          #: two models must sit together on 96 GB with beam state
N_TO_REMOTE = 20          #: balance point; see the module docstring



_MEMBERSHIP_RECIPE = ("sha256 of ','.join(sorted(stem+'|'+member)) over "
                      "data/beam_sample_105.csv, first 16 hex")


def _membership_hash():
    """Computed, never typed. See the note at its use site."""
    import csv as _csv, hashlib as _h
    rows = list(_csv.DictReader(open(os.path.join(ROOT, "data",
                                                  "beam_sample_105.csv"))))
    key = ",".join(sorted(r["stem"] + "|" + r["member"] for r in rows))
    return _h.sha256(key.encode()).hexdigest()[:16]


def build():
    reg = {m["model_id"]: m for m in json.load(open(REG))["models"]}
    m2b = json.load(open(LIN))["model_to_base"]
    grid = [m for m, r in reg.items()
            if r.get("in_grid_spec") and r.get("status") == "ACTIVE"]
    pairs = [(m2b[m], m) for m in grid
             if m in m2b and m2b[m] in reg and m2b[m] != m]

    sample = pd.read_csv(SAMPLE)
    prompts = sorted(set(sample["prompt"]))
    pop = pd.read_parquet(POP)
    sub = pop[pop.stem.isin(set(sample.stem))]

    #: sites per edge, and the actual (prompt, faller, riser) triples
    sites = collections.defaultdict(list)
    for r in sub.itertuples():
        for e in (r.edges if hasattr(r.edges, "__len__") else []):
            b, a = str(e).split(">")
            sites[(b, a)].append({"prompt": r.prompt, "faller": r.faller,
                                  "riser": r.riser, "stem": r.stem,
                                  "member": r.member})

    by_base = collections.defaultdict(list)
    for b, a in pairs:
        by_base[b].append(a)

    def rank(a, b):
        """**SITES FIRST, THEN POSITION.** An aligned model with no recorded
        edge in the population yields ZERO forced continuations, so preferring
        DPO on name alone picked seven pairs that would generate beams and
        measure no damage at all -- the population's `edges` name particular
        aligned checkpoints and for several bases the only sited one is the
        RLVR/Instruct checkpoint, not the DPO one.

        Within the sited options: superego first (the DPO position), then
        reinforced_superego (RLVR) where no superego edge exists, then ego.
        Data ablations never -- they vary the training data, which is a
        different question from what this pass measures.
        """
        n_sites = len(sites.get((b, a), []))
        pos = (reg[a].get("position") or "").lower()
        pos_rank = {"superego": 0, "reinforced_superego": 1}.get(pos, 3)
        ablation = 9 if "no-" in a.lower() else 0
        return (ablation, 0 if n_sites else 1, pos_rank, -n_sites, a)

    def gb(m):
        return (reg[m].get("params_b") or 7) * 2

    rows = []
    for b, v in by_base.items():
        a = sorted(v, key=lambda x: rank(x, b))[0]
        st = sites.get((b, a), [])
        archs = {reg[b]["architecture"], reg[a]["architecture"]}
        pair_gb = gb(b) + gb(a)
        forced = ("hybrid" in archs) or (pair_gb > PAIR_GB_CAP)
        hours = 0.0
        for m in (b, a):
            r = RATE.get(reg[m]["architecture"])
            if r:
                hours += (len(prompts) * r + 2 * len(st) * r) * 1.15
        rows.append({
            "base": b, "aligned": a,
            "stage": reg[a].get("stage"), "position": reg[a].get("position"),
            "arch_base": reg[b]["architecture"], "arch_aligned": reg[a]["architecture"],
            "params_b_base": reg[b].get("params_b"), "params_b_aligned": reg[a].get("params_b"),
            "pair_gb_fp16": pair_gb,
            "n_sites": len(st), "sites": st,
            "est_mps_hours": round(hours / 3600, 3),
            "must_remote": forced,
            "must_remote_reason": ("hybrid: needs mamba-ssm + causal-conv1d"
                                   if "hybrid" in archs else
                                   "%d GB pair exceeds the %d GB two-up cap" % (pair_gb, PAIR_GB_CAP)
                                   if forced else None),
        })

    #: REMOTE = forced first, then the most expensive, until the balance point.
    forced_rows = [r for r in rows if r["must_remote"]]
    rest = sorted([r for r in rows if not r["must_remote"]],
                  key=lambda r: -r["est_mps_hours"])
    remote = forced_rows + rest[:max(0, N_TO_REMOTE - len(forced_rows))]
    remote_ids = {(r["base"], r["aligned"]) for r in remote}
    local = [r for r in rows if (r["base"], r["aligned"]) not in remote_ids]

    #: SMALLEST FIRST, both sides -- debug on tiny models.
    key = lambda r: (r["pair_gb_fp16"], r["base"])
    return sorted(local, key=key), sorted(remote, key=key), prompts, rows


def summarise(local, remote, prompts):
    print("prompts %d | pairs %d (local %d, remote %d)"
          % (len(prompts), len(local) + len(remote), len(local), len(remote)))
    for name, rs in (("LOCAL (MPS)", local), ("REMOTE (vast.ai)", remote)):
        h = sum(r["est_mps_hours"] for r in rs)
        dl = sum(r["pair_gb_fp16"] for r in rs)
        print("\n%s — %d pairs, %.1f MPS-equivalent hours, %d GB of weights"
              % (name, len(rs), h, dl))
        print("  %-4s %-46s %6s %6s %7s  %s"
              % ("GB", "pair", "sites", "hours", "arch", "note"))
        for r in rs:
            print("  %-4.0f %-46s %6d %6.2f %7s  %s"
                  % (r["pair_gb_fp16"],
                     "%s > %s" % (r["base"].split("/")[-1][:20],
                                  r["aligned"].split("/")[-1][:22]),
                     r["n_sites"], r["est_mps_hours"],
                     r["arch_aligned"][:7], r["must_remote_reason"] or ""))


def preflight_refuse(pairs):
    """**A MANIFEST THAT CANNOT RUN SHOULD NOT BE WRITTEN.** Four pairs died on
    a rented box for reasons knowable from the registry before it was rented,
    and the checker that knows them lived only inside a report nobody called.
    Imported rather than re-implemented so the two cannot drift.
    """
    import json as _json
    sys.path.insert(0, HERE)
    from probe_model_requirements import blockers_for
    reg = {m["model_id"]: m
           for m in _json.load(open(os.path.join(ROOT, "data",
                                                 "model_registry.json")))["models"]}
    bad = []
    for b, a in pairs:
        why = blockers_for(b, a, reg)
        #: NOT PROBED is a warning, not a refusal -- refusing on it would block
        #: every newly registered model until someone ran a probe, and a gate
        #: that blocks ordinary work is a gate that gets removed.
        hard = [w for w in why if w.startswith("VOCAB")]
        if hard:
            bad.append((b, a, hard))
        elif why:
            print("  note %-28s %-28s %s"
                  % (b.split("/")[-1][:28], a.split("/")[-1][:28], ",".join(why)))
    if bad:
        print("\n** REFUSING: %d pair(s) cannot be cross-scored as specified"
              % len(bad))
        for b, a, why in bad:
            print("   %-30s %-30s %s"
                  % (b.split("/")[-1], a.split("/")[-1], "; ".join(why)))
        print("   The smaller vocabulary cannot score the other's beams. Either")
        print("   drop the pair or run with the out-of-vocab beam guard, which")
        print("   both drivers now carry (it DROPS such beams and prints a count).")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print, write nothing")
    a = ap.parse_args()
    local, remote, prompts, allrows = build()
    summarise(local, remote, prompts)
    if a.show:
        print("\n--show: nothing written")
        return
    common = {"producer": "scripts/build_fc_manifest.py",
              "sample": "data/beam_sample_105.csv",
              "sample_membership_sha256_16": _membership_hash(),
              "sample_membership_recipe": _MEMBERSHIP_RECIPE,
              #: **WAS A HARDCODED LITERAL (a6e9f245aca657f7) AND NO RECIPE
              #: REPRODUCED IT.** Six candidates were tried against the current
              #: sample file (bytes a274034428c36a44) and all failed, so the
              #: field could not have noticed the sample changing -- a canary
              #: that cannot fire, same class as a control whose threshold is
              #: unreachable at the chosen n. Now COMPUTED, with the recipe
              #: carried beside the value so a reader can check it rather than
              #: trust it. The old literal is superseded and unreproducible.
              "n_prompts": len(prompts), "prompts": prompts,
              "n_beams": 100, "max_tokens": 10, "mode": "raw",
              "arms": ["undisturbed", "force_faller", "force_riser"],
              "note": ("one pair per BASE MODEL (not per lineage: lineage pools "
                       "across sizes); aligned half in the superego position; "
                       "stage=dpo is derived from position, not training method; "
                       "pairs sorted SMALLEST FIRST so defects surface cheaply")}
    for path, rs, where in ((OUT_MPS, local, "mps"), (OUT_VAST, remote, "vast")):
        json.dump(dict(common, target=where, n_pairs=len(rs),
                       est_mps_hours=round(sum(r["est_mps_hours"] for r in rs), 2),
                       weights_gb_fp16=sum(r["pair_gb_fp16"] for r in rs),
                       pairs=rs), open(path, "w"), indent=1)
        print("\nwrote %s  (%d pairs)" % (os.path.relpath(path, ROOT), len(rs)))


if __name__ == "__main__":
    main()
