#!/usr/bin/env python
"""f11_l1_cloud_roster.py — split the L1 roster into LOCAL and CLOUD.

**THE RULE IS RH'S AND IT IS SIMPLE: anything not already on disk goes to
cloud.** Local disk is at 93% with 67 GiB free; the undownloaded checkpoints
are ~300 GB. A cloud box downloads them onto its own disk, so the split costs
nothing locally and needs no eviction of weights we already paid to fetch.

The four LOCAL_SKIP checkpoints join them: 2 on memory (70B, ~140 GB bf16
against 96 GB) and 2 on disk (32B pair, 128 GB). Different limits, same
destination.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import f11_l1_logits as L
from f11_canonical_texts import load
from malign_logits.registry import Registry
from malign_logits.cache import get_cache

OUT = os.path.join(ROOT, "data", "f11_l1_cloud_roster.json")


def main():
    cm = get_cache()
    kept, _ = load(("ACTIVE",))
    bad = {g for g, v in kept.items()
           if not L.span_fail(v["POLE_A"], v["POLE_B"], group=g)[0]}
    prompts = sorted({t for g, v in kept.items() if g not in bad
                      for t in v.values()})
    ck = sorted({m for p in Registry().base_aligned_pairs()
                 for m in (p["base"], p["aligned"])})

    done, local, cloud = [], [], []
    for m in ck:
        if sum(1 for p in prompts if cm.has_logits(m, p, mode="raw",
                                                   dtype="float32")):
            done.append(m)
        elif m in L.LOCAL_SKIP:
            cloud.append({"model": m, "why": L.LOCAL_SKIP[m]})
        elif L.weights_gb(m) < 0.5:
            cloud.append({"model": m, "why": "not on local disk"})
        else:
            local.append(m)

    print("L1 SPLIT — %d checkpoints, %d prompts" % (len(ck), len(prompts)))
    print("  done already      %d" % len(done))
    print("  LOCAL (on disk)   %d   %.0f GB already fetched, no new download"
          % (len(local), sum(L.weights_gb(m) for m in local)))
    print("  CLOUD             %d" % len(cloud))
    for c in sorted(cloud, key=lambda x: x["why"]):
        print("     %-52s %s" % (c["model"], c["why"]))
    print("\n  cloud forward passes: %d x %d = %d"
          % (len(prompts), len(cloud), len(prompts) * len(cloud)))
    json.dump({
        "_about": "L1 local/cloud split. RH's rule: anything not on local disk "
                  "goes to cloud. Local disk 93% full, 67 GiB free; the cloud "
                  "set is ~300 GB of weights that would not fit.",
        "_producer": "scripts/f11_l1_cloud_roster.py",
        "n_prompts": len(prompts), "prompts": prompts,
        "done": done, "local": local, "cloud": cloud,
    }, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("\nwrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
