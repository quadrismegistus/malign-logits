#!/usr/bin/env python
"""build_twp_fill_spec.py — the twp census fill spec: every ACTIVE prompt still
owed, for every checkpoint we know about.

    scripts/build_twp_fill_spec.py                report only
    scripts/build_twp_fill_spec.py --write        emit data/twp_fill_spec.json

Feeds `scripts/twp_cloud.py --models data/twp_fill_spec.json` on a rented box.
Nothing here loads a model or touches the Hub.

## A RESUMABLE FILL, NOT A MANIFEST

lacan [5255].2: the roster moved 146 -> 150 -> 152 -> 154 in one session, and it
moved in ways that CHANGE MEANING rather than count -- a stage corrected from
`dpo` to `ppo`, an aligned arm moved from superego to ego, a base given a pinned
revision. **A spec frozen this morning would have scored the wrong Aquila base
and mislabelled two stages, and a stale manifest reads as completion.**

So this is regenerated immediately before every launch, and it emits only cells
the stash does not already hold. Re-running it after a partial fleet emits the
remainder; re-running it after a complete one emits nothing, which is the only
honest form of "done".

## THE STATUS FILTER IS NOT OPTIONAL

Goes through `Prompts.where()`, never the raw catalogue. 2,809 rows are 2,590
ACTIVE + 215 RETIRED + 4 DISPUTED, and 163 texts exist only under retired ids.
A fleet sized off the raw file would score 163 retired prompts on every
checkpoint -- and `<<<LOGICAL:BOS>>>`, which is not a text but a per-family
resolution, superseded four literal BOS strings that are RETIRED for exactly
that reason.

## THREE EXCLUSION CLASSES, EACH NAMED IN THE OUTPUT

    BLOCKED     cannot run today and not because of us -- gated pending a
                grant, or a dead repo id
    PINNED-BROKEN  a declared revision the tokenizer will not load. NOT the
                same as unpinned: an unpinned run is at least internally
                consistent, where a half-pinned one pairs 100k weights with a
                144k tokenizer
    NOTHING-OWED   already complete

**An exclusion carries its reason into the spec file.** A roster with a hole is
what the unit rule exists to prevent, and a hole with a reason beside it is a
decision rather than an absence.
"""
import argparse, hashlib, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

#: Cannot run today, and the reason is not ours to fix by running anything.
BLOCKED = {
    "AI-Sweden-Models/gpt-sw3-6.7b":
        "gated=manual; RH applied, no grant yet (403 on config.json)",
    "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct":
        "gated=manual; RH applied, no grant yet",
    "mosaicml/mpt-7b":
        "repo id DEAD -- 404 at the API, not a permissions error. Should come "
        "out of base_aligned_pairs.json rather than sit in a blocked list.",
    "mosaicml/mpt-7b-instruct": "repo id DEAD -- 404 at the API",
}

#: A declared revision whose TOKENIZER will not load. Recorded apart from
#: BLOCKED because the fix is ours and the failure is ours.
PINNED_BROKEN = {
    "BAAI/Aquila2-7B":
        "revisions={'base': '9c76e143...'} is honoured now (twp_cloud."
        "declared_revision + twp.load_tokenizer(revision=)), but AutoTokenizer "
        "at that revision raises OSError 'Unable to load vocabulary from file'. "
        "Running it UNPINNED would score vocab 143,973 against a chat arm at "
        "100,008 -- dimensionally undefined and silent. Queue the chat arm; "
        "hold the base until the 2023 tokenizer loads.",
}


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="data/twp_fill_spec.json")
    ap.add_argument("--pairs-first", action="store_true", default=True,
                    help="order checkpoints that complete a base->aligned pair first")
    a = ap.parse_args()

    from malign_logits.prompts import Prompts
    from malign_logits.cache import get_cache
    from malign_logits.registry import Registry

    cm = get_cache()
    prompts, seen = [], set()
    for pr in Prompts.where(status="ACTIVE"):
        t = pr.row.get("prompt")
        if t and t not in seen:
            seen.add(t); prompts.append(t)

    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    ms = reg.get("models") or reg
    ids = (set(ms) if isinstance(ms, dict)
           else {m.get("model_id") or m.get("id") for m in ms if isinstance(m, dict)})
    pairs = Registry().base_aligned_pairs()
    in_pair = set()
    for p in pairs:
        in_pair.add(p["base"]); in_pair.add(p["aligned"])
        ids.add(p["base"]); ids.add(p["aligned"])
    ids = sorted(i for i in ids if i)

    spec, excluded, tally = [], [], Counter()
    for mid in ids:
        if mid in BLOCKED:
            excluded.append({"model": mid, "class": "BLOCKED",
                             "reason": BLOCKED[mid]}); tally["BLOCKED"] += 1; continue
        if mid in PINNED_BROKEN:
            excluded.append({"model": mid, "class": "PINNED-BROKEN",
                             "reason": PINNED_BROKEN[mid]}); tally["PINNED-BROKEN"] += 1; continue
        owed = [t for t in prompts if not cm.has_true_word_probs(mid, t)]
        if not owed:
            tally["NOTHING-OWED"] += 1; continue
        spec.append({"model": mid, "prompts": owed,
                     "_owed": len(owed), "_completes_pair": mid in in_pair})
        tally["QUEUED"] += 1

    #: PAIRS BEFORE SINGLETONS ([5257]). A checkpoint completing a base->aligned
    #: pair buys a unit of inference; a registry singleton buys a checkpoint.
    #: Pairs have been the short currency in every ordering argument this week,
    #: and a fleet that dies half way should have spent its time on them.
    spec.sort(key=lambda e: (not e["_completes_pair"], -e["_owed"], e["model"]))

    cells = sum(e["_owed"] for e in spec)
    print("TWP CENSUS FILL")
    print("  ACTIVE prompts (deduped on text)   %d   sha %s"
          % (len(prompts), sha16("\n".join(prompts))))
    print("  checkpoints known                  %d" % len(ids))
    for k in ("QUEUED", "NOTHING-OWED", "BLOCKED", "PINNED-BROKEN"):
        print("    %-14s %3d" % (k, tally[k]))
    print("  CELLS TO RUN                       %s" % f"{cells:,}")
    print()
    full = [e for e in spec if e["_owed"] == len(prompts)]
    part = [e for e in spec if e["_owed"] < len(prompts)]
    print("  never-scored %d ckpts (%s cells) | partial %d ckpts (%s cells)"
          % (len(full), f"{sum(e['_owed'] for e in full):,}",
             len(part), f"{sum(e['_owed'] for e in part):,}"))
    print("\n  %-52s %7s %s" % ("checkpoint", "owed", "completes a pair"))
    for e in spec:
        print("    %-50s %7d %s" % (e["model"][:50], e["_owed"],
                                    "yes" if e["_completes_pair"] else "-"))
    if excluded:
        print("\n  EXCLUDED, with reasons:")
        for e in excluded:
            print("    %-46s %-14s %s" % (e["model"][:46], e["class"], e["reason"][:70]))

    if a.write:
        out = {"_meta": {"_about": "twp census fill: ACTIVE prompts still owed, "
                                   "per checkpoint. REGENERATE before every launch.",
                         "_producer": "scripts/build_twp_fill_spec.py",
                         "prompts": len(prompts),
                         "prompt_list_sha256_16": sha16("\n".join(prompts)),
                         "models": len(spec), "cells_to_run": cells,
                         "checkpoints_known": len(ids),
                         "excluded": excluded},
               "spec": [{"model": e["model"], "prompts": e["prompts"]} for e in spec]}
        p = os.path.join(ROOT, a.out)
        json.dump(out, open(p, "w"))
        print("\n  wrote %s  (%.1f MB)" % (a.out, os.path.getsize(p) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
