#!/usr/bin/env python
"""f11_quintuplet_spec.py — the population FROM THE REGISTRATION'S SOURCE OF RECORD.

    scripts/f11_quintuplet_spec.py --show
    scripts/f11_quintuplet_spec.py --write

**THIS EXISTS BECAUSE THE FIRST FLEET RAN THE WRONG POPULATION.** It was built
from `f11_canonical_texts.py`, whose `CORE = ("POLE_A","POLE_B","BOTH")` — an
artifact written for the L1 SPAN PRECONDITION, a question about poles. The
registration's §"Source of record" names `data/f11_quintuplets.json`, which
carries FIVE prompt roles plus `both_matched`. 104 checkpoints ran 115 of 199
prompts and NONE of the 72 control texts, so the registration's declared
secondaries — CONTROL_A vs CONTROL_B, and mean(CONTROLS) vs mean(POLES),
[5063].1's missing cell — were not computable from the result.

**THE GATE BELOW IS THE POINT OF THE FILE.** A spec is a claim about which
population is being measured, and nothing checked that claim against the
document that defines it. `--write` refuses unless every prompt role in the
source of record is represented. A population error is invisible in the output:
every cell is correct, the counts are plausible, the run is clean, and the
question the registration asked is simply not answered.
"""
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

SRC = os.path.join(ROOT, "data", "f11_quintuplets.json")
OUT = os.path.join(ROOT, "data", "f11_twp_spec.quintuplet.json")
#: every role in the source of record that is a PROMPT. Metadata fields
#: (group, language, status, subdomain, controls_status, controls_note) are not.
PROMPT_ROLES = ("pole_a", "pole_b", "both", "control_a", "control_b",
                "both_matched")
SCAN = ("Falcon-H1", "Falcon3-Mamba", "falcon-mamba", "Zamba2")


def population(status=None):
    q = json.load(open(SRC))
    qs = q["quintuplets"]
    items = qs.items() if isinstance(qs, dict) else [(e.get("group"), e) for e in qs]
    byrole, allp = collections.defaultdict(set), []
    seen = set()
    for gid, v in items:
        if not isinstance(v, dict):
            continue
        if status and v.get("status") not in status:
            continue
        for r in PROMPT_ROLES:
            t = v.get(r)
            if isinstance(t, str) and t.strip():
                byrole[r].add(t)
                if t not in seen:
                    seen.add(t); allp.append(t)
    return byrole, allp, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--status", default=None,
                    help="comma-separated; default CARRIES status rather than "
                         "filtering, matching the source of record's own note")
    a = ap.parse_args()
    st = tuple(x.strip() for x in a.status.split(",")) if a.status else None
    byrole, allp, q = population(st)

    print("SOURCE OF RECORD: %s" % os.path.relpath(SRC, ROOT))
    print("  builder: %s" % q.get("_about", "")[:70])
    print("\nPROMPT ROLES (metadata fields excluded)")
    for r in PROMPT_ROLES:
        print("   %-14s %3d" % (r, len(byrole.get(r, ()))))
    print("   %-14s %3d distinct prompts" % ("TOTAL", len(allp)))

    # ---- THE GATE ------------------------------------------------------
    empty = [r for r in PROMPT_ROLES if not byrole.get(r)]
    if empty:
        sys.exit("REFUSING: roles present in the source of record but absent "
                 "from this spec: %s. That is the defect this file exists to "
                 "prevent." % ", ".join(empty))
    prev = os.path.join(ROOT, "data", "f11_twp_spec.json")
    if os.path.exists(prev):
        old = set(json.load(open(prev))["spec"][0]["prompts"])
        print("\nvs the FIRST fleet's population (%d prompts):" % len(old))
        print("   carried over %d | NEWLY INCLUDED %d" % (len(set(allp) & old),
                                                          len(set(allp) - old)))
        ctl = set().union(byrole["control_a"], byrole["control_b"],
                          byrole["both_matched"])
        print("   of the newly included, %d are control/matched texts"
              % len(ctl - old))

    if a.write:
        from malign_logits.registry import Registry
        from f11_l1_logits import native_dtype
        ck = sorted({m for p in Registry().base_aligned_pairs()
                     for m in (p["base"], p["aligned"])})
        spec = []
        for m in ck:
            e = {"model": m, "prompts": allp}
            #: **compute_dtype TRAVELS WITH THE SPEC**, not with a sibling file
            #: someone remembered to patch. Falcon-H1 at fp16 measures finite
            #: 1/12; the first backfill inherited the float16 default because
            #: its spec was generated from one that declared nothing.
            if any(k.lower() in m.lower() for k in SCAN):
                e["compute_dtype"] = native_dtype(m, default="bfloat16")
            spec.append(e)
        json.dump({"_meta": {
            "about": "F11 population FROM THE REGISTRATION'S SOURCE OF RECORD, "
                     "all prompt roles. Replaces f11_twp_spec.json, which "
                     "carried only POLE_A/POLE_B/BOTH.",
            "producer": "scripts/f11_quintuplet_spec.py",
            "source_of_record": "data/f11_quintuplets.json",
            "roles": {r: len(byrole[r]) for r in PROMPT_ROLES},
            "prompts": len(allp), "models": len(spec),
            "cells": len(allp) * len(spec),
            "compute_dtype_declared": sum(1 for e in spec if "compute_dtype" in e),
        }, "spec": spec}, open(OUT, "w"), ensure_ascii=False, indent=1)
        print("\nwrote %s  (%d models x %d prompts = %d cells)"
              % (os.path.relpath(OUT, ROOT), len(spec), len(allp),
                 len(spec) * len(allp)))


if __name__ == "__main__":
    main()
