#!/usr/bin/env python
"""build_fleet.py — turn a list of model IDs into a provisioning plan.

    scripts/build_fleet.py --models a.txt --boxes 6
    scripts/build_fleet.py --models a.txt --boxes 6 --write --tag gen1
    scripts/build_fleet.py --profile-only ssm --models a.txt

Consumes `data/model_requirements.json`. Emits, per box: the profile to launch,
the package pins that box needs, and a spec file naming only the models that box
can actually run.

## WHY GROUPING COMES BEFORE BALANCING

**A box can only run models it can run, and a box that downloads a model it
cannot load has paid for the download anyway.** On 2026-08-10 a `dense` box
pulled 15 GB of Zamba2 and failed on a kernel it did not have; four more
checkpoints burned downloads on transformers 5.14.1 before `tf457` existed as a
name. So the partition is by REQUIREMENT first and by cell count second, never
the reverse.

## IT REFUSES RATHER THAN GUESSES

A model with no row in `model_requirements.json` is NOT assigned to `default`.
It stops the plan and is named. That is the exact shape of the day's worst
silent defect: Baichuan2 fell out of every shard when a spec was regenerated,
had zero cells anywhere, and no completion count showed it -- **a model absent
from the plan is absent from the denominator too**, so the fleet reported 100%
of a roster that had quietly shrunk.

Blocked models (gated, dead repo) are excluded WITH THEIR REASON in the plan,
not dropped. A hole with a reason beside it is a decision; a hole without one is
an accident nobody can date.

## EQUAL-CELLS ASSUMPTION

`--cells-per-model` (default 2579, the ACTIVE battery) makes balancing a count.
That is right for twp, where every model scores the same prompt list. For
generation it is a starting point, not a fact: per-model cost varies with
parameters and layer count by ~25x across this roster (measured 0.38 to 9.99
s/prompt), so a balanced plan is balanced in CELLS and not in TIME. Pass
`--weight-by-params` to bias the split by model size instead.
"""
import argparse, json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

REQ = os.path.join(ROOT, "data", "model_requirements.json")

#: **AN ENVIRONMENT IS NOT A MACHINE SHAPE.** `tf457` and `torch26` describe
#: PACKAGE state, not hardware, and `cloud_profiles.json` holds only shapes --
#: so a launch command naming them would have failed at the vast search. The
#: hardware a tf457 box needs is whatever its models need; the pin is applied
#: after setup. torch26 collapses into a normal box because every profile
#: already pins torch>=2.6, which is the whole reason that environment exists
#: as a LABEL rather than a provisioning difference.
LAUNCH_PROFILE = {"default": "dense", "torch26": "dense",
                  "ssm": "ssm", "twogpu": "twogpu", "tf457": "dense"}


def load_requirements():
    if not os.path.exists(REQ):
        raise SystemExit("data/model_requirements.json missing -- run "
                         "scripts/build_model_requirements.py --write first")
    d = json.load(open(REQ))
    return {r["model"]: r for r in d["requirements"]}, d.get("_environments", {})


def plan(models, n_boxes, req, cells_per_model, weight_by_params):
    unknown = [m for m in models if m not in req]
    if unknown:
        raise SystemExit(
            "REFUSING: %d model(s) have no requirements row, and assigning them to "
            "`default` is how a checkpoint silently leaves a fleet:\n   %s\n"
            "Add them to the registry and re-run build_model_requirements.py."
            % (len(unknown), "\n   ".join(unknown)))

    blocked = [(m, req[m]["blocked_reason"]) for m in models if req[m]["blocked"]]
    runnable = [m for m in models if not req[m]["blocked"]]

    groups = defaultdict(list)
    for m in runnable:
        groups[req[m]["profile"]].append(m)

    #: WEIGHT the split so a box is balanced in TIME, not just in model count.
    def weight(m):
        if not weight_by_params:
            return cells_per_model
        p = req[m].get("params_b") or 7.0
        return cells_per_model * max(0.25, p / 7.0)

    total = sum(weight(m) for m in runnable)
    out, box_id = [], 0
    for prof in sorted(groups, key=lambda p: -sum(weight(m) for m in groups[p])):
        ms = sorted(groups[prof], key=lambda m: -weight(m))
        share = sum(weight(m) for m in ms) / total if total else 0
        #: at least one box per profile: a profile with work and no box is a
        #: silent drop, which is the failure this whole script guards
        k = max(1, min(len(ms), round(share * n_boxes)))
        bins = [[] for _ in range(k)]
        load = [0.0] * k
        for m in ms:                     # longest-processing-time first
            i = load.index(min(load))
            bins[i].append(m); load[i] += weight(m)
        pins = sorted({req[m]["transformers"] for m in ms} |
                      {"torch" + req[m]["torch"] for m in ms})
        kern = sorted({k2 for m in ms for k2 in req[m]["kernels"]})
        for b in bins:
            if not b:
                continue
            out.append({"box": box_id, "profile": prof,
                        "launch_profile": LAUNCH_PROFILE.get(prof, "dense"),
                        "models": b,
                        "cells": int(sum(cells_per_model for _ in b)),
                        "gpus": max(req[m]["gpus"] for m in b),
                        "min_vram_gb": max(req[m]["min_vram_gb"] for m in b),
                        "transformers": sorted({req[m]["transformers"] for m in b}),
                        "kernels": kern,
                        "compute_dtype": sorted({req[m]["compute_dtype"] for m in b
                                                 if req[m]["compute_dtype"]}),
                        "revisions": {m: req[m]["revision"] for m in b
                                      if req[m]["revision"]},
                        "tokenizer_overrides": {m: req[m]["tokenizer_loader"] for m in b
                                                if req[m]["tokenizer_loader"]}})
            box_id += 1
    return out, blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="file with one model id per line; "
                                     "default = every non-blocked checkpoint")
    ap.add_argument("--boxes", type=int, default=6)
    ap.add_argument("--cells-per-model", type=int, default=2579)
    ap.add_argument("--weight-by-params", action="store_true",
                    help="balance by model size, not model count -- per-prompt cost "
                         "spans ~25x across this roster")
    ap.add_argument("--profile-only")
    ap.add_argument("--tag", default="fleet")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    req, envs = load_requirements()
    if a.models:
        models = [l.strip() for l in open(a.models) if l.strip()
                  and not l.startswith("#")]
    else:
        models = sorted(req)
    if a.profile_only:
        models = [m for m in models if req.get(m, {}).get("profile") == a.profile_only]

    boxes, blocked = plan(models, a.boxes, req, a.cells_per_model, a.weight_by_params)

    print("FLEET PLAN  %d models -> %d boxes" % (len(models), len(boxes)))
    if blocked:
        print("\n  EXCLUDED, blocked (named, not dropped):")
        for m, why in blocked:
            print("    %-46s %s" % (m[:46], (why or "")[:70]))
    print()
    for b in boxes:
        print("  box %d  profile=%-8s %d models  %s cells  %dx GPU >=%dGB"
              % (b["box"], b["profile"], len(b["models"]), f"{b['cells']:,}",
                 b["gpus"], b["min_vram_gb"]))
        pins = [p for p in b["transformers"] if p != ">=4.57"]
        if pins:   print("         PIN transformers%s" % ", ".join(pins))
        if b["kernels"]: print("         kernels: %s" % ", ".join(b["kernels"]))
        if b["compute_dtype"]: print("         dtype: %s" % ", ".join(b["compute_dtype"]))
        if b["revisions"]: print("         revisions: %s" % b["revisions"])
        if b["tokenizer_overrides"]:
            print("         tokenizer: %s" % b["tokenizer_overrides"])
        for m in b["models"]: print("           %s" % m)

    print("\n  LAUNCH")
    for b in boxes:
        lp = b["launch_profile"]
        #: a tf457 box carrying Zamba2 needs the KERNELS as well as the pin, so
        #: its shape is `ssm` even though its environment is tf457
        if b["kernels"] and lp == "dense":
            lp = "ssm"
        print("    MALIGN_VAST_STATE=.vastai.%s%d.json malign cloud --yes launch "
              "--profile %s   # env=%s" % (a.tag, b["box"], lp, b["profile"]))
        pins = [p for p in b["transformers"] if p != ">=4.57"]
        if pins:
            print("      # then on the box: pip install 'transformers%s'" % pins[0])

    if a.write:
        p = os.path.join(ROOT, "data", "fleet_plan_%s.json" % a.tag)
        json.dump({"_about": "Provisioning plan derived from model_requirements.json. "
                             "Grouped by REQUIREMENT first, balanced second.",
                   "_producer": "scripts/build_fleet.py",
                   "tag": a.tag, "n_boxes": len(boxes),
                   "cells_per_model": a.cells_per_model,
                   "blocked": [{"model": m, "reason": w} for m, w in blocked],
                   "boxes": boxes}, open(p, "w"), indent=1)
        print("\n  wrote %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
