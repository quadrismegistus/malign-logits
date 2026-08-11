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

from malign_logits.lineage import base_model_of   # the ONLY sanctioned parse

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
    #: **A `repo@revision` CHECKPOINT INHERITS ITS REPO'S REQUIREMENTS.** Every
    #: field in a requirements row -- transformers floor, torch floor, kernels,
    #: VRAM, compute dtype, package pins -- is a property of the ARCHITECTURE and
    #: the weights' size. A training step does not change any of them: M05's 43
    #: SFT rungs are one 7B model at 43 revisions.
    #:
    #: Exact match still wins, so a checkpoint with its own declared row keeps it.
    #: **The refusal below is deliberately kept**: an id that resolves to nothing
    #: at EITHER grain still stops the fleet, because assigning it `default` is
    #: how a checkpoint silently leaves a run.
    def R(m):
        return req.get(m) or req.get(base_model_of(m))

    #: ── PAIR MODE ────────────────────────────────────────────────────────
    #: **A UNIT IS WHAT CANNOT BE SPLIT ACROSS BOXES.** For twp that is a
    #: checkpoint. For a passage run it is a PAIR: `vllm_y_run.py` loads base,
    #: generates, frees it, loads aligned, generates, then CROSS-SCORES both
    #: sets in one process. Split a pair and cross-scoring is gone, and
    #: `scored_by_base`/`scored_by_aligned` is the field every downstream
    #: analysis reads.
    #:
    #: So the grouping and balancing below are unchanged and operate on UNITS;
    #: only the resolution of a unit's requirements differs. One code path, not
    #: a second planner — the pin-partition rule, the repo-grain inheritance and
    #: the refusal are the parts that must not be duplicated.
    def merge_pair(u):
        """Requirements for a pair: the STRICTER of its two arms, except VRAM.

        **VRAM IS max(), NOT sum(), AND THAT IS THE WHOLE SAVING.** The arms are
        resident SEQUENTIALLY, never together. The Y run recorded what sizing on
        the pair costs: it "forced two A100s at $0.69 and $1.03/hr for shards
        whose biggest model is 28 GB and 20 GB." Sizing on the pair is correct
        for a concurrent load and wrong for this runner.

        Package pins UNION, because a pin is box-wide state and the box must
        satisfy both arms. `blocked` propagates: an unrunnable arm makes the
        pair unrunnable, since a pair missing an arm has no contrast to compute.
        """
        a, b = R(u[0]), R(u[1])
        if a is None or b is None:
            return None
        pk = dict(a.get("packages") or {})
        for k, v in (b.get("packages") or {}).items():
            #: a genuine disagreement is not mergeable and must not be papered
            #: over by last-wins; the pair is refused with both values named
            if k in pk and pk[k] != v:
                return {"blocked": True,
                        "blocked_reason": "arms disagree on %s: %s vs %s"
                                          % (k, pk[k], v), "packages": pk,
                        "profile": a["profile"], "min_vram_gb": 0, "gpus": 1,
                        "transformers": a["transformers"], "torch": a["torch"],
                        "kernels": [], "compute_dtype": None, "revision": None,
                        "tokenizer_loader": None, "params_b": 0,
                        "packages_reason": {}}
            pk[k] = v
        blocked = bool(a["blocked"] or b["blocked"])
        why = a["blocked_reason"] if a["blocked"] else b["blocked_reason"]
        return {
            "profile": a["profile"] if a["profile"] == b["profile"] else "default",
            "packages": pk,
            "packages_reason": {**(a.get("packages_reason") or {}),
                                **(b.get("packages_reason") or {})},
            "min_vram_gb": max(a["min_vram_gb"], b["min_vram_gb"]),   # SEQUENTIAL
            "gpus": max(a["gpus"], b["gpus"]),
            "transformers": max(a["transformers"], b["transformers"]),
            "torch": max(a["torch"], b["torch"]),
            "kernels": sorted(set(a["kernels"]) | set(b["kernels"])),
            "compute_dtype": a["compute_dtype"] or b["compute_dtype"],
            "revision": None, "tokenizer_loader": None,
            "params_b": max(a.get("params_b") or 0, b.get("params_b") or 0),
            "blocked": blocked, "blocked_reason": why,
        }

    _PAIRREQ = {}

    def U(u):
        """Requirements for a UNIT, whichever kind it is."""
        if isinstance(u, str):
            return R(u)
        if u not in _PAIRREQ:
            _PAIRREQ[u] = merge_pair(u)
        return _PAIRREQ[u]

    def label(u):
        return u if isinstance(u, str) else "%s>%s" % u

    unknown = [label(m) for m in models if U(m) is None]
    if unknown:
        raise SystemExit(
            "REFUSING: %d model(s) have no requirements row at either the "
            "checkpoint or the repo grain, and assigning them to `default` is how "
            "a checkpoint silently leaves a fleet:\n   %s\n"
            "Add them to the registry and re-run build_model_requirements.py."
            % (len(unknown), "\n   ".join(unknown)))

    blocked = [(label(m), U(m)["blocked_reason"]) for m in models if U(m)["blocked"]]
    runnable = [m for m in models if not U(m)["blocked"]]

    #: **PACKAGE PINS PARTITION A FLEET EXACTLY AS transformers DOES.** A pin is
    #: box-wide state: two checkpoints that disagree on `sentencepiece` cannot
    #: share a box any more than two that disagree on `transformers` can. Added
    #: 2026-08-10, when internlm2 turned out to need sentencepiece==0.2.1 while
    #: the rest of the roster runs 0.2.2 -- grouping on `profile` alone would
    #: have put them together and the pin applied last would silently win.
    #:
    #: Keyed on the FULL pin set, not on the profile label, because `tf457` now
    #: describes two different environments: with the sentencepiece pin and
    #: without. A label that no longer determines the environment is the thing
    #: that makes a fleet unreproducible.
    def pinkey(m):
        pk = U(m).get("packages") or {}
        return tuple(sorted(pk.items()))

    groups = defaultdict(list)
    for m in runnable:
        groups[(U(m)["profile"], pinkey(m))].append(m)

    #: WEIGHT the split so a box is balanced in TIME, not just in model count.
    def weight(m):
        if not weight_by_params:
            return cells_per_model
        p = U(m).get("params_b") or 7.0
        return cells_per_model * max(0.25, p / 7.0)

    total = sum(weight(m) for m in runnable)
    out, box_id = [], 0
    for gkey in sorted(groups, key=lambda p: -sum(weight(m) for m in groups[p])):
        prof, pins_t = gkey
        ms = sorted(groups[gkey], key=lambda m: -weight(m))
        share = sum(weight(m) for m in ms) / total if total else 0
        #: at least one box per profile: a profile with work and no box is a
        #: silent drop, which is the failure this whole script guards
        k = max(1, min(len(ms), round(share * n_boxes)))
        bins = [[] for _ in range(k)]
        load = [0.0] * k
        for m in ms:                     # longest-processing-time first
            i = load.index(min(load))
            bins[i].append(m); load[i] += weight(m)
        pins = sorted({U(m)["transformers"] for m in ms} |
                      {"torch" + U(m)["torch"] for m in ms})
        kern = sorted({k2 for m in ms for k2 in U(m)["kernels"]})
        for b in bins:
            if not b:
                continue
            out.append({"box": box_id, "profile": prof,
                        "launch_profile": LAUNCH_PROFILE.get(prof, "dense"),
                        "packages": dict(pins_t),
                        "packages_reason": {k: (U(b[0]).get("packages_reason") or {}).get(k)
                                            for k, _ in pins_t},
                        "models": [label(m) for m in b],
                        "pairs": [{"base": m[0], "aligned": m[1]}
                                  for m in b if not isinstance(m, str)] or None,
                        "cells": int(sum(cells_per_model for _ in b)),
                        "gpus": max(U(m)["gpus"] for m in b),
                        "min_vram_gb": max(U(m)["min_vram_gb"] for m in b),
                        "transformers": sorted({U(m)["transformers"] for m in b}),
                        "kernels": kern,
                        "compute_dtype": sorted({U(m)["compute_dtype"] for m in b
                                                 if U(m)["compute_dtype"]}),
                        "revisions": {label(m): U(m)["revision"] for m in b
                                      if U(m)["revision"]},
                        "tokenizer_overrides": {label(m): U(m)["tokenizer_loader"] for m in b
                                                if U(m)["tokenizer_loader"]}})
            box_id += 1
    return out, blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="file with one model id per line; "
                                     "default = every non-blocked checkpoint")
    #: **PAIR MODE.** A json carrying pairs -- either a `pairs` list of
    #: {base, aligned}, or a forced-arm table whose cells carry `pair` as
    #: "base>aligned". The pair becomes the indivisible unit; see merge_pair().
    ap.add_argument("--pairs", metavar="JSON",
                    help="plan PAIRS instead of models: the unit that cannot be "
                         "split, because cross-scoring happens inside one process")
    ap.add_argument("--boxes", type=int, default=6)
    ap.add_argument("--cells-per-model", type=int, default=2579)
    ap.add_argument("--weight-by-params", action="store_true",
                    help="balance by model size, not model count -- per-prompt cost "
                         "spans ~25x across this roster")
    ap.add_argument("--profile-only")
    ap.add_argument("--tag", default="fleet")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.models and a.pairs:
        raise SystemExit("REFUSING: --models and --pairs are two different units. "
                         "Pick one; a plan mixing them balances nothing.")

    req, envs = load_requirements()

    def load_pairs(path):
        """[(base, aligned)] from a pairs list OR a forced-arm table.

        Two shapes because two producers write them: `y_shard_*.json` carries
        `pairs: [{base, aligned}]`, and `forced_arms_*.json` carries `cells`
        whose `pair` is "base>aligned". Accepting both is not laxity -- it is
        the difference between planning from the artifact a run will actually
        consume and asking an operator to transcribe it into a third format.
        """
        d = json.load(open(path))
        out = []
        for r in (d.get("pairs") or []):
            if isinstance(r, dict) and r.get("base") and r.get("aligned"):
                out.append((r["base"], r["aligned"]))
        if not out:
            for c in (d.get("cells") or []):
                pr = c.get("pair")
                if pr and ">" in pr:
                    out.append(tuple(pr.split(">", 1)))
        seen, uniq = set(), []
        for t in out:
            if t not in seen:
                seen.add(t); uniq.append(t)
        if not uniq:
            raise SystemExit("REFUSING: no pairs found in %s -- expected a "
                             "`pairs` list of {base,aligned} or `cells` with "
                             "`pair` as 'base>aligned'." % path)
        return uniq
    if a.pairs:
        models = load_pairs(a.pairs)
        print("PAIR MODE: %d pairs, %d checkpoints. The pair is indivisible -- "
              "cross-scoring happens inside one process." % (len(models),
              len({m for t in models for m in t})), file=sys.stderr)
    elif a.models:
        models = [l.strip() for l in open(a.models) if l.strip()
                  and not l.startswith("#")]
    else:
        models = sorted(req)
    if a.profile_only:
        models = [m for m in models
                  if (req.get(m, {}) if isinstance(m, str) else {}).get("profile")
                  == a.profile_only]

    boxes, blocked = plan(models, a.boxes, req, a.cells_per_model, a.weight_by_params)

    UNIT = "pairs" if a.pairs else "models"
    print("FLEET PLAN  %d %s -> %d boxes" % (len(models), UNIT, len(boxes)))
    if blocked:
        print("\n  EXCLUDED, blocked (named, not dropped):")
        for m, why in blocked:
            print("    %-46s %s" % (m[:46], (why or "")[:70]))
    print()
    for b in boxes:
        print(("  box %d  profile=%-8s %d " + UNIT + "  %s cells  %dx GPU >=%dGB")
              % (b["box"], b["profile"], len(b["models"]), f"{b['cells']:,}",
                 b["gpus"], b["min_vram_gb"]))
        pins = [p for p in b["transformers"] if p != ">=4.57"]
        if pins:   print("         PIN transformers%s" % ", ".join(pins))
        if b["packages"]:
            print("         PIN %s" % ", ".join("%s%s" % kv for kv in sorted(b["packages"].items())))
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
        extra = ["%s%s" % kv for kv in sorted(b["packages"].items())]
        #: **THE PACKAGE PINS GO IN THE SAME pip CALL AS transformers.** Installing
        #: them separately lets pip resolve the second call against the first and
        #: quietly move one of them; internlm2's whole failure mode is one package
        #: being a version away from working.
        if pins or extra:
            allp = (["transformers%s" % pins[0]] if pins else []) + extra
            print("      # then on the box: pip install %s"
                  % " ".join("'%s'" % p for p in allp))

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
